"""锁住 BM25 中文分词与融合权重的契约（RAG-005）。

这些测试全部离线、无 embedding 调用：分词是纯函数，BM25 是纯 CPU。
它们守的是修复前的两个失效面：
1. 中文整句退化成单个 token，导致 BM25 在中文上匹配不到任何词；
2. 动态权重按 `len(query.split())` 判断，对无空格中文恒为 1，把高权重
   给了失效的那一路。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.rag.retrievers.hybrid_retriever import (
    HybridRetriever,
    PositiveScoreBM25Retriever,
)
from app.rag.retrievers.tokenization import TOKENIZER_ID, cjk_bigram_tokenize

CHINESE_QUERY = "缓存目录空间不足时应该怎么处理"


def test_chinese_query_produces_more_than_one_token() -> None:
    """RAG-005 验收标准点名的那一条：中文查询必须切出多于 1 个 token。

    修复前 `text.split()` 对这句返回 ['缓存目录空间不足时应该怎么处理']，
    而该 token 是一整句话，语料中永不出现，故 BM25 恒不匹配。
    """
    tokens = cjk_bigram_tokenize(CHINESE_QUERY)
    assert len(tokens) > 1
    assert tokens != [CHINESE_QUERY]
    assert len(tokens) == len(CHINESE_QUERY) - 1  # 15 字 -> 14 个 bigram
    assert tokens[0] == "缓存"


def test_tokenizer_preserves_character_order() -> None:
    """bigram 的存在理由：保留字序。单字切分会让这两句无法区分。"""
    assert cjk_bigram_tokenize("缓存") == ["缓存"]
    assert cjk_bigram_tokenize("存缓") == ["存缓"]
    assert cjk_bigram_tokenize("缓存") != cjk_bigram_tokenize("存缓")


def test_latin_and_identifiers_do_not_regress() -> None:
    """英文与标识符必须仍按整词切，不能被 CJK 规则波及。"""
    assert cjk_bigram_tokenize("cache directory full") == [
        "cache",
        "directory",
        "full",
    ]
    assert "cache_quota_mb" in cjk_bigram_tokenize("cache_quota_mb 的默认值")
    assert "e1001" in cjk_bigram_tokenize("E1001 是什么问题")


@pytest.mark.parametrize("text", ["", "   ", "???", "。。。", "😀"])
def test_degenerate_input_yields_no_tokens(text: str) -> None:
    """切不出 token 时必须返回空列表，让调用方跳过词法路而不是喂噪声。"""
    assert cjk_bigram_tokenize(text) == []


def test_single_character_is_kept() -> None:
    """单字成行不能被 bigram 规则丢掉。"""
    assert cjk_bigram_tokenize("缓") == ["缓"]
    assert cjk_bigram_tokenize("a") == ["a"]


def test_tokenizer_identity_is_declared() -> None:
    """分词器身份进 retrieval_config -> config_sha256；改行为必须改 id。"""
    assert TOKENIZER_ID == "cjk_bigram.v1"


def _corpus() -> list:
    from langchain_core.documents import Document

    return [
        Document(page_content="服务错误码清单", metadata={"chunk_id": "c1"}),
        Document(page_content="缓存目录空间不足时清理缓存并重启同步", metadata={"chunk_id": "c2"}),
        Document(page_content="cache_quota_mb 默认值为 512", metadata={"chunk_id": "c3"}),
    ]


def test_bm25_matches_chinese_query_after_tokenization() -> None:
    """端到端：修复前这个查询在中文语料上召回为空。"""
    retriever = PositiveScoreBM25Retriever.from_documents(
        documents=_corpus(),
        k=3,
        preprocess_func=cjk_bigram_tokenize,
    )
    hits = retriever.invoke("缓存目录空间不足怎么处理")
    assert hits, "中文查询必须能召回到含相同词的 chunk"
    assert hits[0].metadata["chunk_id"] == "c2"


def test_zero_score_candidates_are_dropped() -> None:
    """零词法交集的文档不得进入候选：那是 numpy 分区次序，不是检索信号。"""
    retriever = PositiveScoreBM25Retriever.from_documents(
        documents=_corpus(),
        k=3,
        preprocess_func=cjk_bigram_tokenize,
    )
    hits = retriever.invoke("缓存目录")
    ids = [hit.metadata["chunk_id"] for hit in hits]
    assert "c2" in ids
    assert "c3" not in ids, "与查询无任何共同 token 的 chunk 不应出现"
    assert len(hits) < 3


def test_query_without_tokens_retrieves_nothing() -> None:
    retriever = PositiveScoreBM25Retriever.from_documents(
        documents=_corpus(),
        k=3,
        preprocess_func=cjk_bigram_tokenize,
    )
    assert retriever.invoke("???") == []


def test_fusion_weights_never_favour_bm25() -> None:
    """RAG-005 的硬约束：BM25 权重不得高于向量权重。"""
    with pytest.raises(ValueError, match="BM25 权重不得高于向量权重"):
        HybridRetriever(SimpleNamespace(), fusion_weights=(0.3, 0.7))


def test_weights_are_not_derived_from_query_length() -> None:
    """删掉的旧启发式：同一权重必须与查询长度无关。

    修复前 20 字以下给 [0.3, 0.7]、50 字以上给 [0.7, 0.3]；那是英文假设。
    """
    retriever = HybridRetriever(SimpleNamespace(), fusion_weights=(0.6, 0.4))
    short = asyncio.run(retriever.get_dynamic_weights("缓存不足"))
    medium = asyncio.run(retriever.get_dynamic_weights(CHINESE_QUERY))
    long = asyncio.run(retriever.get_dynamic_weights("缓存目录" * 20))
    spaced = asyncio.run(retriever.get_dynamic_weights("cache directory full"))
    assert short == medium == long == spaced == [0.6, 0.4]


def test_untokenizable_query_routes_entirely_to_vector() -> None:
    """唯一保留的查询相关分支，依据是分词结果而非长度。"""
    retriever = HybridRetriever(SimpleNamespace(), fusion_weights=(0.6, 0.4))
    assert asyncio.run(retriever.get_dynamic_weights("???")) == [1.0, 0.0]
    assert asyncio.run(retriever.get_dynamic_weights(None)) == [1.0, 0.0]


def test_vector_dominance_property_holds_for_configured_weights() -> None:
    """向量 top-k 必须整体排在任何 BM25 独有候选之前。

    RRF 是 weight/(rank+60)。该性质等价于
    vector_weight/(k+60) > bm25_weight/(1+60)，它是「Recall@k 不低于纯向量」
    的构造保证，而不是实测巧合。
    """
    from app.utils.config import chroma_config

    vector_weight = chroma_config["bm25"]["vector_weight"]
    bm25_weight = chroma_config["bm25"]["bm25_weight"]
    top_k = 20
    assert vector_weight / (top_k + 60) > bm25_weight / (1 + 60)

