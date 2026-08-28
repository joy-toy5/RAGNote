"""锁住 development 基线所依赖的事实，避免文档数字与资产静默漂移。

这些测试不重跑 run（那需要真实 embedding 调用），只校验冻结资产本身：
数据集身份、qrels 形状、索引可还原性、以及报告里被文档引用的指标。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BACKEND_ROOT / "evals" / "datasets" / "m3_dev_v2"
QRELS_PATH = BACKEND_ROOT / "evals" / "qrels" / "m3_dev_v2_qrels.json"
REPORTS_DIR = BACKEND_ROOT / "evals" / "reports"

INDEX_VERSION = "6958f78b05cb992622c585a8706e878e04f05499ac756d9e7d0000af81051f58"
DATASET_SHA256 = "809b671ba0fea8c434bba9da6c0eb07a19a373350d67a239f66e1ecd5dc46fa7"
QRELS_SHA256 = "11305020ffaa7ca244696934536b44cee9ec5a02545daab9dc356678a4550160"

# RAG-005 的验收门槛（台账 6.5）：修复后 hybrid 不得低于 vector_only 基线。
RAG_005_GATE = {"recall_at_3": 0.863, "recall_at_20": 1.0, "mrr_at_10": 0.731}

# 报告标签 -> (run 文件名, retrieval_config 里声明的 mode)。
# 不能从文件名反推 mode：修复后的 hybrid 变体文件名带分词器后缀，但生产只有
# 一条 hybrid 路径，声明的 mode 仍是 "hybrid"。
RUN_MODES = {
    "hybrid": ("m3_dev_v2_hybrid.json", "hybrid"),
    "vector_only": ("m3_dev_v2_vector_only.json", "vector_only"),
    "hybrid_cjk": ("m3_dev_v2_hybrid_cjk.json", "hybrid"),
    "hybrid_cjk_w50": ("m3_dev_v2_hybrid_cjk_w50.json", "hybrid"),
}

# 文档 5.2.2 引用的数字；改动检索行为必须同时更新文档与这里。
# hybrid 与 vector_only 是 RAG-005 修复**前**的冻结基线，其数字不得再变：
# 修复后的生产路径是 hybrid_cjk。旧 hybrid 只能由冻结产物复现，当前源码已不
# 包含空白分词与长度启发式那条路径。
EXPECTED_AGGREGATE = {
    "hybrid": {
        "recall_at_3": 0.5,
        "recall_at_5": 0.625,
        "recall_at_10": 0.725,
        "recall_at_20": 0.85,
        "mrr_at_10": 0.539384920635,
        "precision_at_3": 0.225,
        "false_answer_rate": 1.0,
        "cross_user_hit_count": 0,
    },
    "vector_only": {
        "recall_at_3": 0.8625,
        "recall_at_5": 0.95,
        "recall_at_10": 0.975,
        "recall_at_20": 1.0,
        "mrr_at_10": 0.73125,
        "precision_at_3": 0.36666666666666664,
        "false_answer_rate": 1.0,
        "cross_user_hit_count": 0,
    },
    # RAG-005 修复后的生产路径：cjk_bigram.v1 分词 + 固定权重 0.6/0.4 + 剔除
    # 0 分候选。七个指标全部不低于 vector_only。
    "hybrid_cjk": {
        "recall_at_3": 0.9,
        "recall_at_5": 0.95,
        "recall_at_10": 0.9875,
        "recall_at_20": 1.0,
        "mrr_at_10": 0.8125,
        "ndcg_at_10": 0.842532249182,
        "precision_at_3": 0.391666666667,
        "false_answer_rate": 1.0,
        "cross_user_hit_count": 0,
    },
    # 权重消融：0.5/0.5 的 MRR/nDCG 更高，但 R@5 少 1 条，且失去「向量 top-k
    # 必然排在任何 BM25 独有候选之前」的构造保证（需权重比 > 80/61）。
    "hybrid_cjk_w50": {
        "recall_at_3": 0.9,
        "recall_at_5": 0.9375,
        "recall_at_10": 0.9875,
        "recall_at_20": 1.0,
        "mrr_at_10": 0.8375,
        "ndcg_at_10": 0.856618866504,
        "precision_at_3": 0.391666666667,
        "false_answer_rate": 1.0,
        "cross_user_hit_count": 0,
    },
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


requires_dataset = pytest.mark.skipif(
    not DATASET_DIR.exists(),
    reason="m3_dev_v2 冻结数据集不在工作树中",
)
@requires_dataset
def test_dataset_identity_is_frozen() -> None:
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["index_version"] == INDEX_VERSION
    assert manifest["dataset_version"] == "v2"
    embedding = manifest["index_config"]["embedding"]
    assert embedding == {
        "dimension": 1024,
        "model": "qwen3.7-text-embedding",
        "provider": "ALIYUN",
    }


@requires_dataset
def test_qrels_shape_matches_documented_baseline() -> None:
    raw = QRELS_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == QRELS_SHA256
    qrels = json.loads(raw.decode("utf-8"))
    assert qrels["index_version"] == INDEX_VERSION
    entries = qrels["entries"]
    assert len(entries) == 58
    assert len({entry["query_id"] for entry in entries}) == 40
    # 已知覆盖缺口：每条 qrel 仍只解析到 1 个 chunk。这条断言是缺口的守卫，
    # 一旦出现替代 chunk，它会失败并提醒同步文档。
    assert {len(entry["chunk_ids"]) for entry in entries} == {1}


@requires_dataset
def test_every_chunk_reconstructs_from_normalized_text() -> None:
    """chunks.jsonl 不存正文；正文必须能从规范化文本按字符区间逐字节还原。"""
    corpus = {
        item["corpus_item_id"]: item
        for item in load_jsonl(DATASET_DIR / "corpus.jsonl")
    }
    texts = {}
    for item_id, item in corpus.items():
        text = (DATASET_DIR / item["normalized_text_path"]).read_text(encoding="utf-8")
        assert (
            hashlib.sha256(text.encode("utf-8")).hexdigest()
            == item["normalized_text_sha256"]
        )
        texts[item_id] = text

    chunks = load_jsonl(DATASET_DIR / "chunks.jsonl")
    assert len(chunks) == 108
    assert len({chunk["chunk_id"] for chunk in chunks}) == 108
    for chunk in chunks:
        assert "content" not in chunk, "chunks.jsonl 不应存正文"
        assert chunk["page_number"] is None, "page map 未冻结"
        assert chunk["document_revision"] == 1
        content = texts[chunk["corpus_item_id"]][chunk["char_start"] : chunk["char_end"]]
        assert (
            hashlib.sha256(content.encode("utf-8")).hexdigest()
            == chunk["content_sha256"]
        )


@pytest.mark.parametrize("mode", sorted(EXPECTED_AGGREGATE))
def test_baseline_report_matches_documented_numbers(mode: str) -> None:
    report_path = REPORTS_DIR / f"m3_dev_v2_{mode}" / "report.json"
    if not report_path.exists():
        pytest.skip(f"{mode} 基线报告不在工作树中")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["metadata"]["index_version"] == INDEX_VERSION
    assert report["metadata"]["dataset_sha256"] == DATASET_SHA256
    assert report["hard_gate"]["passed"] is True
    aggregate = report["aggregate"]
    for metric, expected in EXPECTED_AGGREGATE[mode].items():
        assert aggregate[metric] == pytest.approx(expected, abs=5e-4), metric


def _aggregate(mode: str) -> dict:
    path = REPORTS_DIR / f"m3_dev_v2_{mode}" / "report.json"
    if not path.exists():
        pytest.skip(f"{mode} 报告不在工作树中")
    return json.loads(path.read_text(encoding="utf-8"))["aggregate"]


def test_hybrid_cjk_is_not_worse_than_vector_only() -> None:
    """RAG-005 修复后的门禁：混合检索不得低于纯向量的任一指标。

    这条替换了 `test_vector_only_beats_hybrid_until_bm25_is_fixed`。修复前融合
    结果差于任一单路（纯向量把目标 chunk 排 rank 1/2/3，融合压到 rank 21/21/23），
    那条测试锁住的是缺陷；这条锁住的是修复。
    """
    fixed = _aggregate("hybrid_cjk")
    vector_only = _aggregate("vector_only")
    for metric in (
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "mrr_at_10",
        "ndcg_at_10",
        "precision_at_3",
    ):
        assert fixed[metric] >= vector_only[metric] - 5e-4, metric


def test_hybrid_cjk_meets_documented_acceptance_gate() -> None:
    """台账 6.5 RAG-005 写明的三条具体门槛，不得放宽。"""
    fixed = _aggregate("hybrid_cjk")
    assert fixed["recall_at_3"] >= RAG_005_GATE["recall_at_3"]
    assert fixed["recall_at_20"] == pytest.approx(RAG_005_GATE["recall_at_20"])
    assert fixed["mrr_at_10"] >= RAG_005_GATE["mrr_at_10"]


def test_prefix_baseline_records_that_fusion_was_worse_than_either_route() -> None:
    """保留修复前的事实：这是 RAG-005 的证据，不能因为已修复就丢掉。

    冻结的 hybrid 报告是当时生产路径的产物；当前源码已不含空白分词与长度
    启发式，因此它只能由冻结产物复现，不能由当前代码重跑得到。
    """
    broken = _aggregate("hybrid")
    vector_only = _aggregate("vector_only")
    for metric in ("recall_at_3", "recall_at_20", "mrr_at_10", "ndcg_at_10"):
        assert vector_only[metric] > broken[metric], metric
    # Recall@20 未达 1.0 曾是排序问题而非漏召回：修复后同一索引下达到 1.0。
    assert broken["recall_at_20"] < 1.0
    assert _aggregate("hybrid_cjk")["recall_at_20"] == pytest.approx(1.0)


def test_weight_ablation_is_recorded_and_does_not_change_production_default() -> None:
    """0.5/0.5 消融必须留在产物里：权重是被测过才定的，不是拍的。"""
    from app.utils.config import chroma_config

    assert chroma_config["bm25"]["vector_weight"] == 0.6
    assert chroma_config["bm25"]["bm25_weight"] == 0.4
    chosen = _aggregate("hybrid_cjk")
    alternative = _aggregate("hybrid_cjk_w50")
    # 取舍是实测的：0.5/0.5 排序质量更高，但 Recall@5 更低。
    assert alternative["mrr_at_10"] > chosen["mrr_at_10"]
    assert alternative["recall_at_5"] < chosen["recall_at_5"]


def test_baseline_runs_record_disabled_stages() -> None:
    """基线关闭了 HyDE 与 Reranker；关闭必须留在产物里，不能只写在文档上。"""
    runs_dir = BACKEND_ROOT / "evals" / "runs"
    for label, (filename, declared_mode) in RUN_MODES.items():
        path = runs_dir / filename
        if not path.exists():
            pytest.skip(f"{label} run 不在工作树中")
        run = json.loads(path.read_text(encoding="utf-8"))
        config = run["retrieval_config"]
        assert config["top_k"] >= 20
        assert config["hyde"]["enabled"] is False
        assert config["reranker"]["enabled"] is False
        assert config["note_store"]["enabled"] is False
        assert config["retriever"]["mode"] == declared_mode
        codes = {
            stage["error_code"]
            for query in run["queries"]
            for candidate in query["candidates"]
            for stage in candidate["stages"]
            if stage["stage"] == "rerank"
        }
        assert codes == {"RERANK_DISABLED_UNTRAINED_HEAD"}


def test_fixed_runs_declare_bm25_tokenizer_and_explicit_weights() -> None:
    """分词器与权重必须进 retrieval_config，否则 config_sha256 没有绑定它们。

    「一切影响检索结果的开关必须写进 retrieval_config」是 evals 的资产边界；
    换分词器就是换检索行为，隐瞒它等于 attestation 造假。
    """
    from app.rag.retrievers.tokenization import TOKENIZER_ID

    runs_dir = BACKEND_ROOT / "evals" / "runs"
    expected_weights = {
        "m3_dev_v2_hybrid_cjk.json": {"vector": 0.6, "bm25": 0.4},
        "m3_dev_v2_hybrid_cjk_w50.json": {"vector": 0.5, "bm25": 0.5},
    }
    seen = set()
    for filename, weights in expected_weights.items():
        path = runs_dir / filename
        if not path.exists():
            pytest.skip(f"{filename} 不在工作树中")
        run = json.loads(path.read_text(encoding="utf-8"))
        retriever = run["retrieval_config"]["retriever"]
        assert retriever["bm25_tokenizer"] == TOKENIZER_ID
        assert retriever["bm25_zero_score_candidates"] == "dropped"
        assert retriever["weights"] == weights
        assert isinstance(retriever["weights"], dict), "权重必须是显式数值，不能是启发式名称"
        seen.add(run["retrieval_config_sha256"])
    # 只有权重不同的两条 run 必须得到不同的 config_sha256。
    assert len(seen) == len(expected_weights)


def test_prefix_baseline_still_declares_the_old_heuristic() -> None:
    """修复前的 run 必须仍记录旧启发式：它是 RAG-005 的证据。"""
    path = BACKEND_ROOT / "evals" / "runs" / "m3_dev_v2_hybrid.json"
    if not path.exists():
        pytest.skip("修复前 hybrid run 不在工作树中")
    retriever = json.loads(path.read_text(encoding="utf-8"))["retrieval_config"][
        "retriever"
    ]
    assert retriever["weights"] == "dynamic_by_query_length"
    assert "bm25_tokenizer" not in retriever
