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
# 证据禁止型拒答标记（`RAG-026`）。与上面那个**不是同一件事**，合并阶段的处置相反：
#
#   `[[NO_ANSWER]]`        = 「本块没有这个信息」 -> 剔除该分支，其他分支照常合并。
#                            这是对的：文档 1 没有答案不代表文档 2 也没有。
#   `[[EVIDENCE_FORBIDS]]` = 「本块的证据明确表示不能据此作此推断」 -> 否决整条查询。
#                            它不是「缺信息」，而是「有信息，且该信息否定了作答的前提」。
#
# 压成一类的后果是实测过的：dev-035「之前那次中继变慢属于哪个错误码」，rank 1 块里
# 同时有「E2101 是缓冲溢出」和「旧记录…不能据此判断错误码」，rank 5 块只有错误码定义
# 而不含那句禁止。深度 5 起 rank 5 进窗口后块内自洽、自信作答，rank 1 的异议被当成
# 「这块没答案」剔掉，合并阶段看不到任何反对意见 —— 于是输出一个**引用完全正确、
# 每个原子事实都能核对、只有推理那一步错了**的假答案。这类假答案对引用校验免疫。
EVIDENCE_FORBIDS_MARKER = "[[EVIDENCE_FORBIDS]]"
# 两类拒答标记的统一入口。`_is_refusal` 对两者都为真，故既有行为逐位不变；
# 需要区分处置的地方单独调 `_is_evidence_forbidden`。
REFUSAL_MARKERS = (NO_ANSWER_MARKER, EVIDENCE_FORBIDS_MARKER)
# `refusal_kind` 的取值（`RAG-026`）。作答路径为 None，两条拒答路径取下面之一。
# 提成常量供评测与测试引用同一个串，避免两侧各写一份字面量后静默漂移。
REFUSAL_KIND_INFORMATION_MISSING = "information_missing"
REFUSAL_KIND_EVIDENCE_FORBIDS = "evidence_forbids"
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
# 进入生成上下文的文档数上限（RAG-018）。原为 `get_documents_and_summary` 里的行内
# 魔数 3，提为常量+构造参数只是为了让评测能扫 {3,5,8} 曲线；**默认值仍是 3，生产
# 行为逐位不变**，故不产生可比性断点。
#
# 它同时决定两处，必须一起走：一是送进摘要的文档数，二是 trace 里哪些候选被标
# `selected_for_context`。只改前者会让 trace 谎报实际进上下文的条数，而 R@3 这类
# 生产决定性指标正是按这个标记算的。
DEFAULT_MAX_CONTEXT_DOCUMENTS = 3

# 单次 LLM 调用的超时上限（`RAG-024`）。原为 map 与 reduce 两处各自的行内魔数
# `30.0`，提为常量+构造参数，**默认值仍是 30.0，生产行为逐位不变**。
#
# 取值依据（不是拍的）：`scripts/rag024_latency_probe.py` 在 `qwen3.7-plus` 上对
# 曾稳定撞墙的 4 条查询做无墙实测，12 次 map 调用的中位 13.5s、p90 22.6s、最大
# 26.1s，0 次越过 30s。所以在该模型上 30s 够用，改它没有实测依据；而 `qwen3.8-max`
# 上 45%~65% 的 answerable 行撞墙，那是模型延迟分布右移所致，不是这个值本身错。
# 提为参数只为让评测能在不改生产默认值的前提下扫取值，避免又一次可比性断点。
DEFAULT_LLM_CALL_TIMEOUT_S = 30.0

def _empty_generation_health() -> dict:
    """分支健康度的空形状（`RAG-024`）。

    所有返回路径都带 `generation_health` 这个键，形状恒定 —— 评测侧可以无条件读，
    不必写 `.get(..., {}) or {}` 那种防御代码。未进入生成阶段的路径（检索为空、
    检索层门禁拒答）保持全 0：0 个分支跑过，0 个失败。

    用函数而不是模块级 dict 常量：`branch_errors` 是 list，共享一个实例会让某个
    调用方的 append 泄漏到其他所有返回值里。
    """
    return {
        "branches_total": 0,
        "branches_succeeded": 0,
        "branches_failed": 0,
        "branch_errors": [],
        "degraded": False,
    }


class RagService:
    def __init__(
        self,
        user_id: str = None,
        thinking_callback=None,
        *,
        vector_store=None,
        note_service_override=None,
        max_documents: int = DEFAULT_MAX_CONTEXT_DOCUMENTS,
        llm_call_timeout_s: float = DEFAULT_LLM_CALL_TIMEOUT_S,
    ):
        """
        :param vector_store: 显式注入的向量库服务；为 None 时使用生产单例。
                             离线评测注入 VectorStoreService.for_explicit_target(...)，
                             避免读到生产索引。
        :param note_service_override: 显式注入的笔记服务；为 None 时使用生产单例。
                             离线评测必须注入空实现：笔记候选没有 provenance 元数据，
                             会让 execution_from_trace 拒绝整条 Query。
        :param max_documents: 进入生成上下文的文档数上限，默认 3（`RAG-018`）。
                             生产不传，走默认值；只有扫曲线的评测才显式指定。
                             同时决定 trace 里 `selected_for_context` 的标记深度。
        :param llm_call_timeout_s: 单次 LLM 调用超时秒数，默认 30.0（`RAG-024`）。
                             map 与 reduce 两处共用同一个值。生产不传，走默认值。
        """
        if not isinstance(max_documents, int) or isinstance(max_documents, bool):
            raise TypeError(f"max_documents 必须是 int，收到 {type(max_documents).__name__}")
        if max_documents < 1:
            raise ValueError(f"max_documents 必须 >= 1，收到 {max_documents}")
        self.max_documents = max_documents
        # bool 是 int 的子类，会被 isinstance(x, (int, float)) 放过去，故单独挡掉：
        # `llm_call_timeout_s=True` 等于 1 秒，那是静默的荒谬行为而不是报错。
        if isinstance(llm_call_timeout_s, bool) or not isinstance(
            llm_call_timeout_s, (int, float)
        ):
            raise TypeError(
                "llm_call_timeout_s 必须是数值，收到 "
                f"{type(llm_call_timeout_s).__name__}"
            )
        if llm_call_timeout_s <= 0:
            raise ValueError(
                f"llm_call_timeout_s 必须 > 0，收到 {llm_call_timeout_s}"
            )
        self.llm_call_timeout_s = float(llm_call_timeout_s)
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

        # 深度与下面送进摘要的 `self.max_documents` 必须一致，否则 trace 会谎报
        # 实际进上下文的条数（`RAG-018`）。
        selected_candidates = [
            candidate.select_for_context() if rank <= self.max_documents else candidate
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
        """判定一段摘要是否为拒答（`RAG-008`），两类拒答都算（`RAG-026`）。

        只认标记，不做自然语言判断：「资料里没有提到」这类措辞既可能是拒答，也
        可能是答案的一部分，靠关键词匹配会把正常回答误判成拒答。

        标记必须在开头 —— 允许其后跟一行说明，但不接受出现在正文中间的标记，
        那更可能是模型在复述要求而不是在拒答。

        本函数**故意对两类标记都返回真**：所有「是否拒答」的判断（单文档路径、
        最终摘要、全部拒答的早返回）语义不变，`RAG-026` 只在需要区分**处置方式**
        的那一处额外调 `_is_evidence_forbidden`。这样新标记不改动任何既有路径。
        """
        if not isinstance(summary, str):
            return False
        return summary.strip().startswith(REFUSAL_MARKERS)

    @staticmethod
    def _is_evidence_forbidden(summary: str) -> bool:
        """判定一段摘要是否为**证据禁止型**拒答（`RAG-026`）。

        与 `_is_refusal` 的区别只在处置：这一类必须否决整条查询的合并，而不是
        被当作「这块没答案」剔除掉。判据同样只认开头的标记。
        """
        if not isinstance(summary, str):
            return False
        return summary.strip().startswith(EVIDENCE_FORBIDS_MARKER)

    @classmethod
    def _refusal_kind_of(cls, summary: str) -> str:
        """把一段拒答摘要映射到 `refusal_kind`（`RAG-026`）。

        只在已知 `_is_refusal` 为真时调用。非拒答文本会落到信息缺失型，这个
        默认值本身不表达任何判断 —— 调用点保证不会走到那里。
        """
        if cls._is_evidence_forbidden(summary):
            return REFUSAL_KIND_EVIDENCE_FORBIDS
        return REFUSAL_KIND_INFORMATION_MISSING

    @staticmethod
    def _strip_marker(summary: str) -> str:
        """取出标记之后的说明文字，用于日志；不进用户可见回答。

        必须覆盖两类标记（`RAG-026`）。只剥 `[[NO_ANSWER]]` 的话，一条
        `[[EVIDENCE_FORBIDS]]` 摘要会被原样返回、标记留在文本里 —— 而调用方
        把这个返回值当纯说明文字用，标记就此泄漏。这是解析型标记最常见的漏法。
        """
        if not isinstance(summary, str):
            return ""
        text = summary.strip()
        for marker in REFUSAL_MARKERS:
            if text.startswith(marker):
                return text[len(marker):].strip()
        return text

    def _refusal_result(
        self,
        trace: RetrievalTrace,
        *,
        documents: list,
        detail: str = "",
        health: dict | None = None,
        refusal_kind: str = REFUSAL_KIND_INFORMATION_MISSING,
    ) -> dict:
        """生成层拒答的统一返回。

        标记本身绝不能进 summary —— 解析失败时用户会看到 [[NO_ANSWER]]，
        这是解析型标记最常见的泄漏方式。

        `health` 是本次生成的分支健康度（`RAG-024`）。拒答路径尤其需要带上它：
        一次拒答是在 3 个分支都活着时给出的，还是在 2 个分支已经失败、只剩 1 个
        分支说"没有"时给出的，两者可信度完全不同，而返回值原本无法区分。

        `refusal_kind` 是拒答的语义类别（`RAG-026`）。它必须进返回值而不只是留在
        日志里：`RAG-024` 的教训就是「区分做在代码里但不进产物，等于没做」——
        评测读不到的区分无法用来判断一次拒答是「系统正确识别了证据不足」还是
        「碰巧所有分支都没话说」。默认取信息缺失型，因为绝大多数拒答是那一类。
        """
        if detail:
            logger.info(f"【RAG】生成层拒答（{refusal_kind}）: {detail}")
        else:
            logger.info(f"【RAG】生成层拒答（{refusal_kind}）")
        return {
            "documents": documents,
            "summary": "抱歉，我在你的资料里没有找到能回答这个问题的内容。",
            "no_answer": True,
            "retrieval_trace": trace.to_dict(),
            "generation_health": (
                health if health is not None else _empty_generation_health()
            ),
            "refusal_kind": refusal_kind,
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
                "generation_health": _empty_generation_health(),
                # 没进生成阶段，故没有生成层拒答语义可言（`RAG-026`）。
                # 这条 `no_answer=True` 来自缺少 user_id，不是模型判断证据不足。
                "refusal_kind": None,
            }

        query_id = query_id or str(uuid.uuid4())
        trace = RetrievalTrace(
            query_id=query_id,
            user_id=self.user_id,
            candidates=(),
            no_answer=True,
        )
        # 在 try 之前建好：最外层 except 也要能带上它，否则一次早期崩溃返回的
        # 结果里会缺这个键，形状恒定的承诺就破了。
        health = _empty_generation_health()

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
                    "generation_health": health,
                    # 检索为空，生成阶段未发生（`RAG-026`）。
                    "refusal_kind": None,
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
                max_documents = self.max_documents  # 默认 3，见 DEFAULT_MAX_CONTEXT_DOCUMENTS

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
                        timeout=self.llm_call_timeout_s,  # RAG-024，默认仍 30.0
                    )
                    end_time = time.time()
                    logger.info(f"【RAG】第{i}个文档总结耗时: {end_time - start_time:.2f}秒")
                    return single_summary
                
                # 使用线程池并发处理文档总结
                tasks = []
                for i, doc in enumerate(reordered_documents[:max_documents], 1):
                    tasks.append(summarize_document(i, doc))
                
                # 并发执行所有总结任务
                import time
                start_time = time.time()
                # RAG-024：`return_exceptions=True` 是必须的。没有它，任一分支超时
                # 就会从 gather 抛出，整条查询连同其余**已经成功**的分支一起报废 ——
                # 实测在 qwen3.8-max 上毁掉 45%~65% 的可回答查询。分支数等于
                # max_documents，所以窗口越宽这个单点故障越容易发生。
                branch_results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()

                # 失败分支必须与「拒答分支」分开收集。两者都导致该分支没有可用摘要，
                # 但语义相反：拒答是模型读完了说没有，失败是根本没读到。混在一起
                # 会让一次基础设施故障变成一次"拒答"，把拒答率虚高 —— 那正是本方法
                # 末尾 `except asyncio.TimeoutError` 分支在防的事，也是 `RAG-026`
                # 那类「失败与拒答同形」混淆的同一形态。
                individual_summaries = []
                branch_errors = []
                branch_exceptions = []
                for branch_number, outcome in enumerate(branch_results, 1):
                    if isinstance(outcome, BaseException):
                        branch_exceptions.append(outcome)
                        branch_errors.append(
                            {
                                "branch": branch_number,
                                "error_type": type(outcome).__name__,
                                "error": str(outcome),
                            }
                        )
                        logger.error(
                            f"【RAG】第{branch_number}个文档总结失败: "
                            f"{type(outcome).__name__}: {outcome}"
                        )
                    else:
                        individual_summaries.append(outcome)
                logger.info(
                    f"【RAG】所有文档总结完成，总耗时: {end_time - start_time:.2f}秒"
                    f"（成功 {len(individual_summaries)}，失败 {len(branch_errors)}）"
                )

                health = {
                    "branches_total": len(tasks),
                    "branches_succeeded": len(individual_summaries),
                    "branches_failed": len(branch_errors),
                    "branch_errors": branch_errors,
                    "degraded": bool(branch_errors),
                }

                # 全部分支都失败：这是基础设施故障，不是拒答，必须走失败兜底。
                # 若这里返回拒答，评测会把一次全线故障记成"系统正确地拒绝回答"。
                #
                # 重抛时保留原异常类型，别一律转成超时：`return_exceptions=True` 之前
                # 连接失败会冒泡到最外层记成 GENERATION_ERROR_MESSAGE，转成超时会把
                # 「连不上」写成「太慢」。两者都是 INFRASTRUCTURE_FAILURE_MESSAGES，
                # 不影响判分，但错因分类会失真 —— 这正是本条要修的同一类混淆。
                if individual_summaries == []:
                    non_timeouts = [
                        exc
                        for exc in branch_exceptions
                        if not isinstance(exc, asyncio.TimeoutError)
                    ]
                    if non_timeouts:
                        raise non_timeouts[0]
                    raise asyncio.TimeoutError(
                        f"全部 {len(branch_errors)} 个分支摘要均超时"
                    )

                # 判据是 len(tasks) 而不是 len(individual_summaries)：这个分支要表达
                # 的是「本来只有一个文档」，而失败分支现在已被剔除，用后者会让
                # 「3 篇里 2 篇失败」也走进单文档路径 —— 那会跳过合并阶段且不留痕迹。
                # 两者在无失败时恒等，所以这不改变正常路径的行为。
                # 此处 individual_summaries 必非空：全失败已在上面重抛。
                if len(tasks) == 1:
                    if self._is_refusal(individual_summaries[0]):
                        return self._refusal_result(
                            trace,
                            documents=reordered_documents,
                            detail=self._strip_marker(individual_summaries[0]),
                            health=health,
                            refusal_kind=self._refusal_kind_of(
                                individual_summaries[0]
                            ),
                        )
                    logger.info("【RAG】生成摘要成功")
                    return {
                        "documents": reordered_documents,
                        "summary": individual_summaries[0],
                        "no_answer": False,
                        "retrieval_trace": trace.to_dict(),
                        "generation_health": health,
                        "refusal_kind": None,
                    }

                # 证据禁止型拒答否决整条查询（`RAG-026`）。必须在剔除之前判，因为
                # 剔除会把这条异议连同「本块没答案」一起丢掉 —— 那正是 dev-035 误答
                # 的成因：一个块内自洽的分支独占合并输入，禁止性证据被当成缺信息。
                #
                # 是**否决**而不是投票。dev-035 在深度 5 是 4 拒 1 答、深度 8 是 7 拒
                # 1 答，「过半拒答就拒答」能盖住这一条，但盖不住「禁止性证据恰好也在
                # 一个自信块里」的情形 —— 那会把语义问题换成阈值问题，换来的达标是
                # 假的。一条明确的「不能据此推断」不该被任何数量的「我这块能答」压过。
                forbidding_summaries = [
                    summary
                    for summary in individual_summaries
                    if self._is_evidence_forbidden(summary)
                ]
                if forbidding_summaries:
                    return self._refusal_result(
                        trace,
                        documents=reordered_documents,
                        detail=(
                            f"{len(forbidding_summaries)}/{len(individual_summaries)} "
                            f"个分支的证据明确禁止此推断: "
                            f"{self._strip_marker(forbidding_summaries[0])}"
                        ),
                        health=health,
                        refusal_kind=REFUSAL_KIND_EVIDENCE_FORBIDS,
                    )

                # 逐文档拒答是正常现象：文档 1 没有答案不代表文档 2 也没有。因此
                # 先剔除拒答的分支摘要，只把有内容的送进合并阶段 —— 否则合并阶段
                # 会看到「无法回答」的字样并把它写进最终答案，稀释真实答案。
                #
                # 走到这里 `individual_summaries` 里已无证据禁止型，故此处剔掉的
                # 全是信息缺失型 —— 剔除对它们是正确处置。
                usable_summaries = [
                    summary
                    for summary in individual_summaries
                    if not self._is_refusal(summary)
                ]
                if not usable_summaries:
                    # 活着的分支全部拒答：不必再调一次 LLM 去合并一堆拒答。
                    # 带上 health —— 若同时有分支失败，这条拒答是「部分证据缺失下的
                    # 拒答」，不该和证据齐全时的拒答记成同一件事。
                    return self._refusal_result(
                        trace,
                        documents=reordered_documents,
                        detail=(
                            f"{len(individual_summaries)} 个分支摘要全部拒答"
                            + (
                                f"，另有 {len(branch_errors)} 个分支失败"
                                if branch_errors
                                else ""
                            )
                        ),
                        health=health,
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
                
                # 生成最终总结。这道墙和 map 那道是各自独立的 30s，`RAG-024` 的
                # 实测里它确实发过火（dev-035 深度 8 第 2 轮：map 侧 8 个分支约
                # 7.8s 走完，最终却记到 37795 ms，撞的是这里）。所以只改 map 一处
                # 不够，两处必须同源。
                final_summary = await asyncio.wait_for(
                    self.chain.ainvoke({"input": query, "context": combined_context}),
                    timeout=self.llm_call_timeout_s,  # RAG-024，默认仍 30.0
                )

                if self._is_refusal(final_summary):
                    return self._refusal_result(
                        trace,
                        documents=reordered_documents,
                        detail=self._strip_marker(final_summary),
                        health=health,
                        refusal_kind=self._refusal_kind_of(final_summary),
                    )

                logger.info("【RAG】生成摘要成功")
                return {
                    "documents": reordered_documents,
                    "summary": final_summary,
                    "no_answer": False,
                    "retrieval_trace": trace.to_dict(),
                    "generation_health": health,
                    "refusal_kind": None,
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
                    "generation_health": health,
                    # 失败不是拒答，故无拒答语义（`RAG-026`）。这与 `no_answer=False`
                    # 是同一个判断的两面：把失败记成拒答会虚高拒答率。
                    "refusal_kind": None,
                }
        except Exception as e:
            logger.error(f"【RAG】生成摘要失败: {e}", exc_info=True)
            return {
                "documents": [],
                "summary": GENERATION_ERROR_MESSAGE,
                "no_answer": False,
                "retrieval_trace": trace.to_dict(),
                "generation_health": health,
                "refusal_kind": None,
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
