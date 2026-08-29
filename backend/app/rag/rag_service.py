import asyncio
import uuid
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langsmith import traceable

from app.rag.vector_store import VectorStoreService
from app.rag.reorder_service import reorder_service
from app.rag.retrieval_contract import (
    RetrievalCandidate,
    RetrievalTrace,
    StageObservation,
    candidate_from_document,
    single_index_version,
)
from app.utils.factory import chat_model
from app.utils.prompt_loader import load_prompt
from app.core.logger_handler import logger
from app.services.note_service import note_service

# 双路无交集拒答的阶段错误码（RAG-008）。写成常量是为了让测试与台账引用同一个串。
NO_ROUTE_AGREEMENT_CODE = "NO_ROUTE_AGREEMENT"
# 生成层拒答标记。模型自认参考资料不足时输出它，由代码解析而非靠自然语言判断。
NO_ANSWER_MARKER = "[[NO_ANSWER]]"
# 基础设施失败的兜底文案。对用户是友好提示，但它**不是拒答**：
# `no_answer` 仍为 False，异常也不外抛。评测必须能把它与真正的作答区分开，
# 否则 LLM 挂掉会被记成「该拒却答了」，与真幻觉无法分辨。提成常量供评测引用，
# 避免评测侧写死字符串副本后与生产漂移。
GENERATION_TIMEOUT_MESSAGE = "抱歉，生成摘要超时，请稍后再试。"
GENERATION_ERROR_MESSAGE = "抱歉，处理您的请求时出现了错误。"
INFRASTRUCTURE_FAILURE_MESSAGES = (
    GENERATION_TIMEOUT_MESSAGE,
    GENERATION_ERROR_MESSAGE,
)


class RagService:
    def __init__(
        self,
        user_id: str = None,
        thinking_callback=None,
        *,
        vector_store=None,
        note_service_override=None,
    ):
        """
        :param vector_store: 显式注入的向量库服务；为 None 时使用生产单例。
                             离线评测注入 VectorStoreService.for_explicit_target(...)，
                             避免读到生产索引。
        :param note_service_override: 显式注入的笔记服务；为 None 时使用生产单例。
                             离线评测必须注入空实现：笔记候选没有 provenance 元数据，
                             会让 execution_from_trace 拒绝整条 Query。
        """
        self.vector_store = (
            vector_store if vector_store is not None else VectorStoreService()
        )
        self.note_service = (
            note_service_override if note_service_override is not None
            else note_service
        )
        self.retriever = None
        self.user_id = user_id
        self.prompt_text = load_prompt(prompt_type="rag_summary_prompt")
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.chat_model = chat_model
        self.chain = self._init_chain()
        self.hyde_prompt_template = PromptTemplate.from_template("基于以下问题，生成一个详细的假设性回答，我会根据你的这个假设性回答在向量数据库里检索文档：\n\n问题：{query}\n\n假设性回答：")
        self.thinking_callback = thinking_callback

    async def _notify_weights(self, query: str = None) -> None:
        """把融合权重发到 thinking 流。

        必须在 HyDE 之前调用，保持既有 thinking 事件次序不变。
        """
        if not self.thinking_callback:
            return
        weights = await self.vector_store.get_dynamic_weights(query)
        await self.thinking_callback({
            "type": "thinking",
            "stage": "retrieval",
            "content": f"初始化检索器（向量权重: {weights[0]:.1f}, BM25权重: {weights[1]:.1f}）",
            "details": {
                "vector_weight": weights[0],
                "bm25_weight": weights[1]
            }
        })

    async def initialize_retriever(self, query: str = None):
        """
        初始化检索器并缓存到 self.retriever。

        `retrieve_document` 不再调用它 —— 生产路径改走
        `vector_store.retrieve_with_routes`，因为拒答门禁需要看到两路各自的结果
        （RAG-008），而 `EnsembleRetriever.invoke` 只返回融合后的列表。
        保留此方法有两个用途：`__main__` 的手工验证，以及离线消融预置检索器后
        让 `retrieve_document` 短路到单路路径。

        :param query: 查询语句，用于动态调整权重
        """
        if self.retriever is None:
            await self._notify_weights(query)
            self.retriever = await self.vector_store.get_retriever(query, self.user_id)


    def _init_chain(self):
        """初始化链"""
        chain = (
                self.prompt_template
                | self.chat_model
                | StrOutputParser()
        )
        return chain

    @traceable
    async def generate_hypothetical_document(self, query: str) -> str:
        """
        使用HyDE技术生成假设性文档
        :param query: 用户查询
        :return: 假设性文档内容
        """
        try:
            hyde_chain = (
                self.hyde_prompt_template
                | self.chat_model
                | StrOutputParser()
            )
            hypothetical_doc = await hyde_chain.ainvoke({"query": query})
            logger.info(f"【HyDE】生成的假设性文档:\n{hypothetical_doc}")
            return hypothetical_doc
        except Exception as e:
            logger.error(f"【HyDE】生成假设性文档失败: {e}")
            return query

    @traceable
    async def retrieve_document(
        self,
        query: str,
        *,
        raise_errors: bool = False,
        routes_sink: list | None = None,
    ) -> list:
        """使用HyDE技术 从向量数据库里检索文档

        :param routes_sink: 可选的出参。传入列表时，本次检索的 `RouteRetrieval`
                  会被 append 进去，供上层做双路交集判定（RAG-008）。用出参而不是
                  实例属性，是为了让调用方无法读到上一次查询的残留；单路兜底与
                  消融预置检索器时不会 append，交集因此保持「未定义」。
        """
        if not self.user_id:
            if raise_errors:
                raise ValueError("严格检索要求非空 user_id")
            logger.warning("【HyDE】user_id为空，不进行任何检索")
            return []

        try:
            # 权重必须在 HyDE 之前播报，保持 thinking 事件次序不变。
            await self._notify_weights(query)

            # 使用HyDE技术生成假设性文档
            logger.info(f"【HyDE】开始处理查询: {query}")
            
            if self.thinking_callback:
                await self.thinking_callback({
                    "type": "thinking",
                    "stage": "hyde",
                    "content": f"正在基于查询「{query}」生成假设性文档..."
                })
            
            hypothetical_doc = await self.generate_hypothetical_document(query)
            
            if self.thinking_callback:
                await self.thinking_callback({
                    "type": "thinking",
                    "stage": "hyde",
                    "content": "假设性文档生成完成",
                    "details": {
                        "hypothetical_doc_preview": hypothetical_doc[:200] + "..." if len(hypothetical_doc) > 200 else hypothetical_doc
                    }
                })
            
            # 使用假设性文档进行检索
            logger.info("【HyDE】使用假设性文档进行检索")
            
            if self.thinking_callback:
                await self.thinking_callback({
                    "type": "thinking",
                    "stage": "retrieval",
                    "content": "正在向量数据库中检索相关文档..."
                })
            
            if self.retriever is not None:
                # 消融预置了检索器：保持单路语义，不产出路由信息。
                documents = await self.retriever.ainvoke(hypothetical_doc)
            else:
                routes = await self.vector_store.retrieve_with_routes(
                    hypothetical_doc,
                    self.user_id,
                    weight_query=query,
                )
                documents = list(routes.fused)
                if routes_sink is not None and routes.both_routes_present:
                    routes_sink.append(routes)

            # 同时检索笔记库
            note_docs = []
            try:
                note_docs = await asyncio.to_thread(
                    self.note_service.notes_store.similarity_search,
                    hypothetical_doc, k=3,
                    filter={"user_id": self.user_id}
                )
            except Exception as e:
                if raise_errors:
                    raise
                logger.error(f"【RAG】检索笔记失败: {e}")

            # 标记来源并合并（笔记在前，知识库在后）
            for doc in documents:
                doc.metadata["source_type"] = "knowledge_base"
            for doc in note_docs:
                doc.metadata["source_type"] = "note"
            all_documents = note_docs + documents

            logger.info(f"【HyDE】检索到 {len(documents)} 个知识库文档, {len(note_docs)} 个笔记文档")

            if self.thinking_callback:
                doc_previews = []
                for i, doc in enumerate(all_documents, 1):
                    preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                    if doc.metadata.get("source_type") == "note":
                        source = f"笔记《{doc.metadata.get('title', '无标题')}》"
                    else:
                        source = doc.metadata.get("original_filename", doc.metadata.get("source", "unknown"))
                    doc_previews.append({
                        "index": i,
                        "preview": preview,
                        "source": source,
                    })
                await self.thinking_callback({
                    "type": "thinking",
                    "stage": "retrieval",
                    "content": f"检索到 {len(note_docs)} 篇相关笔记, {len(documents)} 篇知识库文档",
                    "details": {
                        "documents": doc_previews
                    }
                })

            return all_documents
        except Exception as e:
            if raise_errors:
                raise
            logger.error(f"【HyDE】检索文档失败: {e}")
            return []

    @traceable
    async def reorder_documents(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        """
        对文档进行重排序
        :param query: 查询语句
        :param candidates: 保留稳定身份的候选列表
        :return: 追加重排阶段信息后的候选列表
        """
        if self.thinking_callback:
            await self.thinking_callback({
                "type": "thinking",
                "stage": "reorder",
                "content": f"正在对 {len(candidates)} 个文档进行重排序..."
            })

        result = await reorder_service.reorder_documents(
            query,
            candidates,
            thinking_callback=self.thinking_callback,
        )
        if result["success"]:
            reordered_candidates = [
                item["candidate"] for item in result["documents"]
            ]
            logger.info(f"【RAG】文档重排序成功，返回 {len(reordered_candidates)} 个文档")
            
            if self.thinking_callback:
                score_details = []
                for i, doc in enumerate(result["documents"], 1):
                    score_details.append({
                        "rank": i,
                        "score": round(doc.get("similarity", 0), 4),
                        "preview": doc.get("document", "")[:100] + "..." if len(doc.get("document", "")) > 100 else doc.get("document", "")
                    })
                await self.thinking_callback({
                    "type": "thinking",
                    "stage": "reorder",
                    "content": f"重排序完成，返回 {len(reordered_candidates)} 个文档",
                    "details": {
                        "scores": score_details
                    }
                })
            
            return reordered_candidates
        else:
            logger.warning(f"【RAG】重排序失败: {result['error']}")
            return [
                candidate.observe(
                    StageObservation(
                        stage="rerank",
                        route="cross_encoder",
                        rank=rank,
                        outcome="degraded",
                        error_code="RERANK_FAILED",
                    )
                )
                for rank, candidate in enumerate(candidates, 1)
            ]

    async def get_retrieval_trace(
        self,
        query: str,
        *,
        query_id: str,
    ) -> RetrievalTrace:
        """以冻结 Query 身份执行检索，异常不得伪装成正常拒答。"""
        if not self.user_id:
            raise ValueError("严格检索要求非空 user_id")
        return await self._build_retrieval_trace(
            query,
            query_id=query_id,
            raise_errors=True,
        )

    async def _build_retrieval_trace(
        self,
        query: str,
        *,
        query_id: str,
        raise_errors: bool,
    ) -> RetrievalTrace:
        trace = RetrievalTrace(
            query_id=query_id,
            user_id=self.user_id,
            candidates=(),
            no_answer=True,
        )
        routes_sink: list = []
        documents = (
            await self.retrieve_document(
                query, raise_errors=True, routes_sink=routes_sink
            )
            if raise_errors
            else await self.retrieve_document(query, routes_sink=routes_sink)
        )
        candidates = [
            candidate_from_document(
                document,
                user_id=self.user_id,
                rank=rank,
            )
            for rank, document in enumerate(documents, 1)
        ]
        reordered_candidates = await self.reorder_documents(query, candidates)
        if not reordered_candidates:
            # 零候选拒答：候选为空时无从得知 index_version，它保持 None。
            # 裸 RagService 不回显 request 的自报版本（M3_EVALUATION_REPORT 契约），
            # 补齐身份是具体评测工厂的职责，见 scripts/m3_run_eval.py 的
            # _RetrievalOnlyRagService.get_retrieval_trace。
            return trace

        # 双路交集为 0 → 拒答（RAG-008）。门禁是尺度无关的：不比较任何绝对分数，
        # 只看向量路与词法路是否指向同一批证据，换语料、换 IDF、换 avgdl 都不需要
        # 重新调参。这是刻意放弃的另一个方案的反面 —— BM25 绝对阈值在
        # development 语料上能拒对 4/10，但阈值随语料规模漂移，不能当生产常量。
        # 单路兜底时 overlap_count 是 None 而不是 0，不触发拒答。
        overlap = routes_sink[0].overlap_count if routes_sink else None
        has_note_evidence = any(
            candidate.source_type == "note" for candidate in reordered_candidates
        )
        if overlap == 0 and not has_note_evidence:
            # 候选保留：拒答理由要可审计，「拒答了什么」比「拒答了」信息量大。
            # 但不做 select_for_context —— 没有任何候选会进生成上下文。
            refused_candidates = [
                candidate.observe(
                    StageObservation(
                        stage="no_answer_gate",
                        route="route_agreement",
                        rank=rank,
                        outcome="degraded",
                        error_code=NO_ROUTE_AGREEMENT_CODE,
                    )
                )
                for rank, candidate in enumerate(reordered_candidates, 1)
            ]
            logger.info(
                f"【RAG】双路无交集，拒答: query_id={query_id} "
                f"候选数={len(refused_candidates)}"
            )
            return RetrievalTrace(
                query_id=query_id,
                user_id=self.user_id,
                candidates=tuple(refused_candidates),
                index_version=single_index_version(refused_candidates),
                no_answer=True,
            )

        selected_candidates = [
            candidate.select_for_context() if rank <= 3 else candidate
            for rank, candidate in enumerate(reordered_candidates, 1)
        ]
        return RetrievalTrace(
            query_id=query_id,
            user_id=self.user_id,
            candidates=tuple(selected_candidates),
            index_version=single_index_version(selected_candidates),
            no_answer=False,
        )

    @staticmethod
    def _is_refusal(summary: str) -> bool:
        """判定一段摘要是否为拒答（RAG-008）。

        只认标记，不做自然语言判断：「资料里没有提到」这类措辞既可能是拒答，也
        可能是答案的一部分，靠关键词匹配会把正常回答误判成拒答。

        标记必须在开头 —— 允许其后跟一行说明，但不接受出现在正文中间的标记，
        那更可能是模型在复述要求而不是在拒答。
        """
        if not isinstance(summary, str):
            return False
        return summary.strip().startswith(NO_ANSWER_MARKER)

    @staticmethod
    def _strip_marker(summary: str) -> str:
        """取出标记之后的说明文字，用于日志；不进用户可见回答。"""
        return summary.strip()[len(NO_ANSWER_MARKER):].strip()

    def _refusal_result(
        self,
        trace: RetrievalTrace,
        *,
        documents: list,
        detail: str = "",
    ) -> dict:
        """生成层拒答的统一返回。

        标记本身绝不能进 summary —— 解析失败时用户会看到 [[NO_ANSWER]]，
        这是解析型标记最常见的泄漏方式。
        """
        if detail:
            logger.info(f"【RAG】生成层拒答: {detail}")
        else:
            logger.info("【RAG】生成层拒答")
        return {
            "documents": documents,
            "summary": "抱歉，我在你的资料里没有找到能回答这个问题的内容。",
            "no_answer": True,
            "retrieval_trace": trace.to_dict(),
        }

    @traceable
    async def get_documents_and_summary(
        self,
        query: str,
        *,
        query_id: str | None = None,
    ) -> dict:
        """
        获取文档列表和摘要
        :param query: 查询语句
        :return: 包含文档列表和摘要的字典
        """
        if not self.user_id:
            logger.warning("【RAG】user_id为空，不返回任何文档")
            return {
                "documents": [],
                "summary": "抱歉，我没有找到相关的信息。",
                "no_answer": True,
            }

        query_id = query_id or str(uuid.uuid4())
        trace = RetrievalTrace(
            query_id=query_id,
            user_id=self.user_id,
            candidates=(),
            no_answer=True,
        )

        try:
            trace = await self._build_retrieval_trace(
                query,
                query_id=query_id,
                raise_errors=False,
            )

            # 如果没有检索到文档
            if not trace.candidates:
                return {
                    "documents": [],
                    "summary": "抱歉，我没有找到相关的信息。",
                    "no_answer": True,
                    "retrieval_trace": trace.to_dict(),
                }

            # 检索层已判拒答（双路无交集）：不进生成，省掉三次 LLM 调用。
            if trace.no_answer:
                return self._refusal_result(
                    trace,
                    documents=[],
                    detail=f"检索层门禁 {NO_ROUTE_AGREEMENT_CODE}",
                )

            reordered_documents = [
                candidate.reranker_text for candidate in trace.candidates
            ]

            # 使用分批总结策略
            try:
                # 对每个文档单独总结（使用线程池并发处理）
                individual_summaries = []
                max_documents = 3  # 使用前3个最相关的文档
                
                if self.thinking_callback:
                    await self.thinking_callback({
                        "type": "thinking",
                        "stage": "summarize",
                        "content": f"正在对前 {min(max_documents, len(reordered_documents))} 个最相关文档进行总结..."
                    })
                
                # 定义单个文档总结函数
                async def summarize_document(i, doc):
                    logger.info(f"【RAG】正在总结第{i}个文档")
                    if self.thinking_callback:
                        await self.thinking_callback({
                            "type": "thinking",
                            "stage": "summarize",
                            "content": f"正在总结第 {i} 个文档..."
                        })
                    # 为单个文档构建上下文
                    single_context = f"【参考资料{i}】:{doc}\n"
                    # 生成单个文档的摘要
                    import time
                    start_time = time.time()
                    single_summary = await asyncio.wait_for(
                        self.chain.ainvoke({"input": query, "context": single_context}),
                        timeout=30.0  # 单个文档总结超时时间
                    )
                    end_time = time.time()
                    logger.info(f"【RAG】第{i}个文档总结耗时: {end_time - start_time:.2f}秒")
                    return single_summary
                
                # 使用线程池并发处理文档总结
                tasks = []
                for i, doc in enumerate(reordered_documents[:max_documents], 1):
                    tasks.append(summarize_document(i, doc))
                
                # 并发执行所有总结任务，最多5个线程
                import time
                start_time = time.time()
                individual_summaries = await asyncio.gather(*tasks)
                end_time = time.time()
                logger.info(f"【RAG】所有文档总结完成，总耗时: {end_time - start_time:.2f}秒")

                # 如果只有一个文档，直接返回其摘要
                if len(individual_summaries) == 1:
                    if self._is_refusal(individual_summaries[0]):
                        return self._refusal_result(
                            trace,
                            documents=reordered_documents,
                            detail=self._strip_marker(individual_summaries[0]),
                        )
                    logger.info("【RAG】生成摘要成功")
                    return {
                        "documents": reordered_documents,
                        "summary": individual_summaries[0],
                        "no_answer": False,
                        "retrieval_trace": trace.to_dict(),
                    }

                # 逐文档拒答是正常现象：文档 1 没有答案不代表文档 2 也没有。因此
                # 先剔除拒答的分支摘要，只把有内容的送进合并阶段 —— 否则合并阶段
                # 会看到「无法回答」的字样并把它写进最终答案，稀释真实答案。
                usable_summaries = [
                    summary
                    for summary in individual_summaries
                    if not self._is_refusal(summary)
                ]
                if not usable_summaries:
                    # 全部分支都拒答：不必再调一次 LLM 去合并一堆拒答。
                    return self._refusal_result(
                        trace,
                        documents=reordered_documents,
                        detail=f"{len(individual_summaries)} 个分支摘要全部拒答",
                    )

                # 合并多个文档的摘要，生成最终总结
                combined_context = "以下是多个文档的摘要，请综合这些信息生成最终的回答：\n\n"
                for i, summary in enumerate(usable_summaries, 1):
                    combined_context += f"【文档{i}摘要】:{summary}\n\n"

                logger.info("【RAG】合并摘要完成，开始生成最终总结")
                
                if self.thinking_callback:
                    await self.thinking_callback({
                        "type": "thinking",
                        "stage": "summarize",
                        "content": "正在综合多个文档生成最终回答..."
                    })
                
                # 生成最终总结
                final_summary = await asyncio.wait_for(
                    self.chain.ainvoke({"input": query, "context": combined_context}),
                    timeout=30.0  # 最终总结超时时间
                )
                
                if self._is_refusal(final_summary):
                    return self._refusal_result(
                        trace,
                        documents=reordered_documents,
                        detail=self._strip_marker(final_summary),
                    )

                logger.info("【RAG】生成摘要成功")
                return {
                    "documents": reordered_documents,
                    "summary": final_summary,
                    "no_answer": False,
                    "retrieval_trace": trace.to_dict(),
                }
            except asyncio.TimeoutError:
                # 超时是失败，不是拒答：评测里 run_error 与 predicted_no_answer
                # 是两类，把错误记成拒答会虚高拒答率。
                logger.error("【RAG】生成摘要超时")
                return {
                    "documents": reordered_documents,
                    "summary": GENERATION_TIMEOUT_MESSAGE,
                    "no_answer": False,
                    "retrieval_trace": trace.to_dict(),
                }
        except Exception as e:
            logger.error(f"【RAG】生成摘要失败: {e}", exc_info=True)
            return {
                "documents": [],
                "summary": GENERATION_ERROR_MESSAGE,
                "no_answer": False,
                "retrieval_trace": trace.to_dict(),
            }

    @traceable
    async def rag_summary(self, query: str) -> str:
        """RAG 摘要"""
        result = await self.get_documents_and_summary(query)
        return result.get("summary", GENERATION_ERROR_MESSAGE)

if __name__ == '__main__':
    import asyncio
    
    async def main():
        service = RagService()
        await service.initialize_retriever()
        result = await service.rag_summary("小户型适合什么扫地机器人")
        print(result)
    
    asyncio.run(main())
