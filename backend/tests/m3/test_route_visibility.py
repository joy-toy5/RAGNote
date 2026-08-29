"""锁住两路可见检索的等价性与交集语义（RAG-008）。

拒答门禁要判断「向量路和词法路是否指向同一批证据」，这需要两路各自的结果，
而 `EnsembleRetriever.invoke` 只给融合后的列表。`retrieve_with_routes` 手动跑两路
再调用库内的 `weighted_reciprocal_rank`，因此本文件守的第一件事是：这个改写没有
动融合次序 —— 否则所有已冻结的检索指标都不再可比。

第二件事是交集的定义：
1. 交集必须用融合用的同一个身份键（生产没传 id_key，故为 page_content）；
2. 单路兜底时交集是「未定义」而非 0，否则向量兜底路径上每条查询都会被拒答。

全部离线：向量路用假检索器，词法路是真 BM25（纯 CPU），无 embedding 调用。
"""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.rag.retrievers.hybrid_retriever import (
    HybridRetriever,
    RouteRetrieval,
    _fusion_key,
)
from app.rag.retrievers.tokenization import cjk_bigram_tokenize


class _FixedRetriever(BaseRetriever):
    """返回预置文档的向量路替身，忽略查询。"""

    documents: list[Document]

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return list(self.documents)


def _doc(text: str, **metadata) -> Document:
    return Document(page_content=text, metadata={"user_id": "u1", **metadata})


CORPUS = [
    _doc("缓存目录空间不足时先清理临时文件再重启服务", chunk_id="c1"),
    _doc("HTTP 429 应该按指数退避重试并遵守 Retry-After", chunk_id="c2"),
    _doc("向量库的索引重建需要停写并记录 manifest", chunk_id="c3"),
    _doc("日志轮转策略按天切分并保留十四天", chunk_id="c4"),
]


def _fake_store(corpus: list[Document], vector_hits: list[Document]):
    """假 Chroma：只需支撑 `.get()`（BM25 语料）与 `.as_retriever()`（向量路）。"""
    return SimpleNamespace(
        get=lambda **kwargs: {
            "documents": [d.page_content for d in corpus],
            "metadatas": [d.metadata for d in corpus],
        },
        as_retriever=lambda **kwargs: _FixedRetriever(documents=vector_hits),
    )


# --- 等价性：改写不得动融合次序 ---------------------------------------------


def test_routes_fusion_matches_plain_invoke() -> None:
    """`retrieve_with_routes().fused` 必须与 `get_retriever().ainvoke()` 逐位相同。

    这是整个改写的前提。不相同就意味着已冻结的 Recall/nDCG 全部失效。
    """
    store = _fake_store(CORPUS, [CORPUS[2], CORPUS[0], CORPUS[3]])
    retriever = HybridRetriever(store, k=4, fusion_weights=(0.6, 0.4))
    query = "缓存目录空间不足"

    async def _run():
        baseline = await (await retriever.get_retriever(query, "u1")).ainvoke(query)
        routes = await retriever.retrieve_with_routes(query, "u1")
        return baseline, routes

    baseline, routes = asyncio.run(_run())
    assert [d.page_content for d in routes.fused] == [
        d.page_content for d in baseline
    ]


def test_fusion_equivalence_holds_under_randomised_route_shapes() -> None:
    """随机化两路交叠形态，逐位比对融合结果。

    单个例子过了不足以说明等价，交集为空/部分/全等三种形态的次序都要对。
    """
    rng = random.Random(20260828)
    for _ in range(60):
        vector_hits = rng.sample(CORPUS, rng.randint(1, len(CORPUS)))
        store = _fake_store(rng.sample(CORPUS, len(CORPUS)), vector_hits)
        retriever = HybridRetriever(store, k=4, fusion_weights=(0.6, 0.4))
        query = rng.choice(["缓存目录", "HTTP 429 重试", "索引重建", "日志轮转"])

        async def _run(r=retriever, q=query):
            baseline = await (await r.get_retriever(q, "u1")).ainvoke(q)
            routes = await r.retrieve_with_routes(q, "u1")
            return baseline, routes

        baseline, routes = asyncio.run(_run())
        assert [d.page_content for d in routes.fused] == [
            d.page_content for d in baseline
        ], f"融合次序不一致：query={query!r} vector_hits={len(vector_hits)}"


# --- 交集语义 ---------------------------------------------------------------


def test_overlap_is_none_when_only_one_route_ran() -> None:
    """单路兜底时交集未定义。

    这是本轮最容易写错的一条：把未定义当 0，向量兜底路径上的每条查询都会
    被拒答门禁截掉。
    """
    routes = RouteRetrieval(
        fused=(CORPUS[0],),
        vector_documents=(CORPUS[0],),
        both_routes_present=False,
    )
    assert routes.overlap_count is None
    assert routes.overlap_count != 0


def test_overlap_counts_documents_hit_by_both_routes() -> None:
    routes = RouteRetrieval(
        fused=(CORPUS[0], CORPUS[1], CORPUS[2]),
        vector_documents=(CORPUS[0], CORPUS[1]),
        bm25_documents=(CORPUS[1], CORPUS[2]),
        both_routes_present=True,
    )
    assert routes.overlap_count == 1


def test_overlap_is_zero_when_both_routes_ran_and_disagree() -> None:
    """0 与 None 是两种事实：两路都跑了但毫无交集，才是拒答门禁的触发条件。"""
    routes = RouteRetrieval(
        fused=(CORPUS[0], CORPUS[2]),
        vector_documents=(CORPUS[0],),
        bm25_documents=(CORPUS[2],),
        both_routes_present=True,
    )
    assert routes.overlap_count == 0


def test_overlap_uses_the_same_identity_key_as_fusion() -> None:
    """交集必须按 page_content 而非 chunk_id。

    生产构造 EnsembleRetriever 时没传 id_key，库内 RRF 按 page_content 去重累加。
    交集若改用 chunk_id，同内容不同 chunk_id 会被算成「无交集」，而融合实际上
    已经把它们折叠并累加了 —— 两个定义必须一致。
    """
    same_text_other_id = _doc(CORPUS[0].page_content, chunk_id="other")
    routes = RouteRetrieval(
        fused=(CORPUS[0],),
        vector_documents=(CORPUS[0],),
        bm25_documents=(same_text_other_id,),
        both_routes_present=True,
    )
    assert _fusion_key(same_text_other_id) == _fusion_key(CORPUS[0])
    assert same_text_other_id.metadata["chunk_id"] != CORPUS[0].metadata["chunk_id"]
    assert routes.overlap_count == 1


def test_overlap_dedup_matches_fusion_collapse() -> None:
    """去重定义与 RRF 折叠一致：同内容在融合里也只占一位。

    语料必须异质。两条完全相同的文档会让每个词的 n(q)=N、IDF 转负，
    `PositiveScoreBM25Retriever` 把整路清空，测的就不是折叠了。
    """
    duplicate = _doc(CORPUS[0].page_content, chunk_id="dup")
    store = _fake_store([*CORPUS, duplicate], [CORPUS[0], duplicate])
    retriever = HybridRetriever(store, k=4, fusion_weights=(0.6, 0.4))
    query = "缓存目录空间不足"

    routes = asyncio.run(retriever.retrieve_with_routes(query, "u1"))
    fused_keys = [_fusion_key(d) for d in routes.fused]
    assert len(fused_keys) == len(set(fused_keys))
    assert len([k for k in fused_keys if k == CORPUS[0].page_content]) == 1
    assert routes.overlap_count == 1


def test_overlap_deduplicates_within_a_route() -> None:
    """同一路重复命中同一内容只算一次，与 RRF 的折叠一致。"""
    routes = RouteRetrieval(
        fused=(CORPUS[0],),
        vector_documents=(CORPUS[0], _doc(CORPUS[0].page_content, chunk_id="dup")),
        bm25_documents=(CORPUS[0],),
        both_routes_present=True,
    )
    assert routes.overlap_count == 1


# --- 端到端：真 HybridRetriever 上的两条路径 --------------------------------


def test_empty_corpus_falls_back_to_single_route() -> None:
    """语料为空 → 没有 BM25 检索器 → 单路，交集未定义而非 0。"""
    store = _fake_store([], [CORPUS[0]])
    retriever = HybridRetriever(store, k=4, fusion_weights=(0.6, 0.4))

    routes = asyncio.run(retriever.retrieve_with_routes("缓存目录", "u1"))
    assert routes.both_routes_present is False
    assert routes.overlap_count is None
    assert routes.bm25_documents == ()
    assert [d.page_content for d in routes.fused] == [CORPUS[0].page_content]


def test_untokenizable_query_falls_back_to_single_route() -> None:
    """查询切不出 token → 权重 [1.0, 0.0] → 与 get_retriever 同一条兜底分支。"""
    store = _fake_store(CORPUS, [CORPUS[0]])
    retriever = HybridRetriever(store, k=4, fusion_weights=(0.6, 0.4))

    assert cjk_bigram_tokenize("???") == []
    routes = asyncio.run(retriever.retrieve_with_routes("???", "u1"))
    assert routes.both_routes_present is False
    assert routes.overlap_count is None
    assert routes.weights == (1.0, 0.0)


def test_both_routes_present_when_bm25_matches() -> None:
    store = _fake_store(CORPUS, [CORPUS[0], CORPUS[1]])
    retriever = HybridRetriever(store, k=4, fusion_weights=(0.6, 0.4))

    routes = asyncio.run(retriever.retrieve_with_routes("缓存目录空间不足", "u1"))
    assert routes.both_routes_present is True
    assert routes.bm25_documents != ()
    assert routes.overlap_count is not None
    assert routes.weights == (0.6, 0.4)


def test_weight_query_decides_weights_independently_of_search_query() -> None:
    """生产用原始查询定权重、用 HyDE 文档检索；两者不是同一个串。

    传入切不出 token 的原始查询时，即便检索文本是正常中文，也必须走单路 ——
    否则权重判定的输入被 HyDE 文档悄悄替换掉了。
    """
    store = _fake_store(CORPUS, [CORPUS[0]])
    retriever = HybridRetriever(store, k=4, fusion_weights=(0.6, 0.4))

    routes = asyncio.run(
        retriever.retrieve_with_routes("缓存目录空间不足", "u1", weight_query="???")
    )
    assert routes.weights == (1.0, 0.0)
    assert routes.both_routes_present is False


def test_bm25_running_but_matching_nothing_counts_as_zero_overlap() -> None:
    """词法路跑了却零命中 → 交集 0（可触发拒答），而不是「未定义」。

    这与单路兜底是两种事实：这里 BM25 检索器存在、权重给了词法路、查询也切出
    了 token，只是语料里没有任何正分匹配。词法路一个证据都没找到，正是拒答
    门禁要用的信号，所以这条行为必须显式锁住而不是隐式发生。
    """
    store = _fake_store(CORPUS, [CORPUS[0]])
    retriever = HybridRetriever(store, k=4, fusion_weights=(0.6, 0.4))

    routes = asyncio.run(retriever.retrieve_with_routes("量子色动力学禁闭", "u1"))
    assert routes.both_routes_present is True
    assert routes.bm25_documents == ()
    assert routes.overlap_count == 0


def test_routes_require_a_valid_user_id() -> None:
    store = _fake_store(CORPUS, [CORPUS[0]])
    retriever = HybridRetriever(store, k=4, fusion_weights=(0.6, 0.4))

    with pytest.raises(ValueError, match="有效的用户 ID"):
        asyncio.run(retriever.retrieve_with_routes("缓存目录", "  "))
