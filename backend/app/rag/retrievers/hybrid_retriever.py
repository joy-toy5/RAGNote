import asyncio
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from app.utils.config import chroma_config
from .tokenization import cjk_bigram_tokenize


class PositiveScoreBM25Retriever(BM25Retriever):
    """只返回 BM25 打分为正的候选，并以稳定次序排列（RAG-005）。

    父类 `get_top_n` 用 `np.argsort`（quicksort，不稳定）取 top-k，对 0 分文档
    一视同仁 —— 一个与查询毫无词法交集的文档，会因为 numpy 的分区次序进入候选
    并在融合里获得权重。词法检索路返回「零词法交集」的文档没有意义。

    并列次序同时改为按语料位置稳定排序：父类的并列次序是 numpy 分区的产物，
    同版本内可复现、跨版本无保证（见 RAG-015）。
    """

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:  # noqa: ANN001
        tokens = self.preprocess_func(query)
        if not tokens:
            return []
        scores = self.vectorizer.get_scores(tokens)
        ranked = sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda index: (-scores[index], index),
        )
        return [self.docs[index] for index in ranked[: self.k]]


class HybridRetriever:
    """混合检索器（BM25 + 向量检索）"""

    def __init__(
        self,
        vectors_store: Chroma,
        k: int | None = None,
        fusion_weights: tuple[float, float] | list[float] | None = None,
    ):
        """
        :param vectors_store: Chroma 向量库
        :param k: 显式召回数量；为 None 时沿用 chroma.yaml 的 k。
                  离线评测需要显式传入，否则 top_k 声明与实际召回不一致。
        :param fusion_weights: 显式 [向量权重, BM25 权重]；为 None 时沿用
                  chroma.yaml。消融需要显式传入，否则 retrieval_config 里声明的
                  权重与实际融合不一致，attestation 就是假的。
        """
        if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
            raise ValueError("召回数量 k 必须是正整数")
        self.vectors_store = vectors_store
        self.k = k if k is not None else chroma_config['k']
        self.fusion_weights = self._validate_weights(
            fusion_weights
            if fusion_weights is not None
            else (
                chroma_config['bm25']['vector_weight'],
                chroma_config['bm25']['bm25_weight'],
            )
        )

    @staticmethod
    def _validate_weights(weights) -> tuple[float, float]:
        """权重必须是两个正数，且向量权重不低于 BM25。

        RAG-005 的教训：`[vec=0.3, bm25=0.7]` 把 0.7 给了失效的那一路。修好分词
        后仍保持向量不低于 BM25，因为 RRF 是 weight/(rank+60)：向量权重 0.6 时
        rank 20 得 0.6/80=0.00750，BM25 权重 0.4 时 rank 1 得 0.4/61=0.00656，
        向量 top-k 因此必然全部排在任何 BM25 独有候选之前，`Recall@k` 不会低于
        纯向量。该性质要求权重比 > 80/61 ≈ 1.311。
        """
        if len(tuple(weights)) != 2:
            raise ValueError("融合权重必须是 [向量权重, BM25权重] 两项")
        vector_weight, bm25_weight = (float(value) for value in weights)
        if vector_weight <= 0 or bm25_weight < 0:
            raise ValueError("向量权重必须为正，BM25 权重不能为负")
        if bm25_weight > vector_weight:
            raise ValueError("BM25 权重不得高于向量权重（RAG-005）")
        return vector_weight, bm25_weight

    async def get_bm25_retriever(self, user_id: str):
        """
        获取BM25检索器
        :param user_id: 用户ID，必须提供，否则返回None
        :return: BM25Retriever实例
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("构造检索器必须提供有效的用户 ID")

        all_docs_result = await asyncio.to_thread(
            self.vectors_store.get,
            include=['documents', 'metadatas'],
            where={'user_id': user_id}
        )
        documents = []
        for i, doc_content in enumerate(all_docs_result['documents']):
            metadata = all_docs_result['metadatas'][i] if i < len(all_docs_result['metadatas']) else {}
            documents.append(Document(page_content=doc_content, metadata=metadata))

        if documents:
            # preprocess_func 同时用于语料侧与查询侧；两边必须是同一个函数，
            # 否则词形不一致，匹配恒为空。默认的 text.split() 会让中文整句
            # 退化成单个 token（RAG-005）。
            bm25_retriever = PositiveScoreBM25Retriever.from_documents(
                documents=documents,
                k=self.k,
                preprocess_func=cjk_bigram_tokenize,
            )
            return bm25_retriever
        else:
            return None

    async def _get_all_documents(self) -> list[Document]:
        """
        获取向量库中的所有文档
        :return: 文档列表
        """
        all_docs = await asyncio.to_thread(
            self.vectors_store.get,
            include=['documents', 'metadatas']
        )
        documents = []
        for i, doc in enumerate(all_docs['documents']):
            metadata = all_docs['metadatas'][i] if i < len(all_docs['metadatas']) else {}
            documents.append(Document(page_content=doc, metadata=metadata))
        return documents

    async def get_retriever(self, query: str | None, user_id: str) -> BaseRetriever:
        """
        获取混合检索器（BM25 + 向量检索）
        :param query: 查询语句，用于动态调整权重
        :param user_id: 用户ID，用于过滤用户的文档，不能为空
        :return: EnsembleRetriever实例或单独的向量检索器
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("构造检索器必须提供有效的用户 ID")

        filter_dict = {'user_id': user_id}
        vector_retriever = self.vectors_store.as_retriever(
            search_type='similarity',
            search_kwargs={'k': self.k, 'filter': filter_dict},
        )
        bm25_retriever = await self.get_bm25_retriever(user_id)
        weights = await self.get_dynamic_weights(query)

        # BM25 权重为 0 表示词法路无词可匹配，此时不构造融合器：让它参与并集
        # 只会把 0 权重候选拼进候选表，等于用噪声扩大候选集。
        if bm25_retriever and weights[1] > 0:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=weights
            )
            return ensemble_retriever
        else:
            return vector_retriever

    async def get_dynamic_weights(self, query: str = None):
        """
        返回融合权重 [向量检索权重, BM25检索权重]。

        权重由评测指标选定（`m3_dev_v2` 40 条可回答查询的消融网格），不再依赖
        查询长度。被删掉的旧启发式是「短查询偏关键词」：它对无空格中文恒有
        `len(query.split()) == 1`，在 `query_length < 20` 时给出
        `[vec=0.3, bm25=0.7]` —— 来自英文假设，在中文上恰好反向，把 0.7 权重
        给了当时失效的那一路（RAG-005）。

        唯一保留的查询相关分支是分词兜底：查询切不出任何 token 时（纯标点、
        emoji、空白）词法路无词可匹配，权重整体给向量。这依据分词器而非长度。

        :param query: 查询语句
        :return: 权重列表 [向量检索权重, BM25检索权重]
        """
        vector_weight, bm25_weight = self.fusion_weights
        if not query or not cjk_bigram_tokenize(query):
            return [1.0, 0.0]
        return [vector_weight, bm25_weight]
