"""锁住 RAG-003 的 per_route_k 曲线结论与生产深度的一致性。

默认只读冻结探针产物（`evals/probes/rag003_k_curve.json`），不重跑检索。
带 `-m integration` 时会重跑探针并与冻结产物逐位比对（约 15 秒，零 embedding
调用）。

探针为什么可信：它在 `per_route_k=20` 上复现三个冻结报告的七项聚合指标，
逐位相等才继续输出曲线。校准结果记在产物的 `calibration` 里。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
PROBE_PATH = BACKEND_ROOT / "evals" / "probes" / "rag003_k_curve.json"
PROBE_SCRIPT = BACKEND_ROOT / "scripts" / "rag003_k_curve_probe.py"
INDEX_VERSION = "6958f78b05cb992622c585a8706e878e04f05499ac756d9e7d0000af81051f58"

# 生产 chroma.yaml 的 per_route_k。曲线证明这个深度是被测过的。
PRODUCTION_PER_ROUTE_K = 5
# 生产只把前 3 条送进上下文（rag_service 的 max_documents）。因此 R@3 是生产
# 决定性指标，R@20 不是。
PRODUCTION_CONTEXT_DEPTH = 3

# 生产深度（per_route_k=5，hybrid 0.6/0.4）的实测值。文档 5.2.7 引用这些数字。
PRODUCTION_DEPTH_METRICS = {
    "recall_at_3": 0.9125,
    "recall_at_5": 0.95,
    "recall_at_10": 0.975,
    "mrr_at_10": 0.808333333333,
    "ndcg_at_10": 0.835912970986,
    "precision_at_3": 0.4,
    "candidates_mean": 7.38,
    "candidates_max": 10,
}

requires_probe = pytest.mark.skipif(
    not PROBE_PATH.exists(),
    reason="RAG-003 探针产物不在工作树中",
)


def _probe() -> dict:
    return json.loads(PROBE_PATH.read_text(encoding="utf-8"))


def _row(probe: dict, mode: str, per_route_k: int) -> dict:
    matches = [
        row
        for row in probe["curve"]
        if row["mode"] == mode and row["per_route_k"] == per_route_k
    ]
    assert len(matches) == 1, f"{mode}@{per_route_k} 在曲线里不唯一"
    return matches[0]


@requires_probe
def test_probe_identity_and_calibration_are_recorded() -> None:
    """探针身份进产物；校准未通过的曲线不得被引用。"""
    probe = _probe()
    assert probe["index_version"] == INDEX_VERSION
    assert probe["rrf_c"] == 60
    assert probe["calibration"]["passed"] is True
    # 三个冻结报告全部逐位复现，否则探针与生产路径已不是同一件事。
    targets = {item["report"] for item in probe["calibration"]["targets"]}
    assert targets == {
        "m3_dev_v2_vector_only",
        "m3_dev_v2_hybrid_cjk",
        "m3_dev_v2_hybrid_cjk_w50",
    }
    assert all(
        not item["mismatches"] for item in probe["calibration"]["targets"]
    )
    assert (
        probe["probe_sha256"]
        == hashlib.sha256(PROBE_SCRIPT.read_bytes()).hexdigest()
    ), "探针脚本已改动但产物未重跑"


def test_production_retrieval_depth_is_the_measured_one() -> None:
    """生产 k 必须落在曲线测过的深度上，否则"生产路径未被测量"这个洞又回来了。"""
    from app.utils.config import chroma_config

    assert chroma_config["k"] == PRODUCTION_PER_ROUTE_K
    if not PROBE_PATH.exists():
        pytest.skip("RAG-003 探针产物不在工作树中")
    measured = {row["per_route_k"] for row in _probe()["curve"]}
    assert PRODUCTION_PER_ROUTE_K in measured


def test_production_context_depth_makes_recall_at_3_decisive() -> None:
    """生产只消费前 3 条；R@3 因此是决定性指标。

    这条是源码契约测试：`max_documents` 一旦改动，k 的取舍依据就变了，门禁必须
    失败以强制重测，而不是让文档继续引用一个过期的结论。
    """
    source = (BACKEND_ROOT / "app" / "rag" / "rag_service.py").read_text(
        encoding="utf-8"
    )
    assert f"max_documents = {PRODUCTION_CONTEXT_DEPTH}" in source
    assert f"rank <= {PRODUCTION_CONTEXT_DEPTH}" in source
    if PROBE_PATH.exists():
        assert _probe()["production_context_depth"] == PRODUCTION_CONTEXT_DEPTH


@requires_probe
def test_production_depth_metrics_match_documented_numbers() -> None:
    """文档 5.2.7 引用的生产深度数字。"""
    row = _row(_probe(), "hybrid", PRODUCTION_PER_ROUTE_K)
    for metric, expected in PRODUCTION_DEPTH_METRICS.items():
        assert row[metric] == pytest.approx(expected, abs=5e-4), metric


@requires_probe
def test_recall_at_20_is_unreachable_at_production_depth() -> None:
    """`Recall@20 = 1.000` 属于 per_route_k >= 14，不属于生产路径。

    per_route_k=5 时融合候选池最多 10 条（两路各 5 条、并集去重），因此 @20 与
    @10 必然相等 —— 把 1.000 当成生产数字写进简历是不成立的。
    """
    probe = _probe()
    production = _row(probe, "hybrid", PRODUCTION_PER_ROUTE_K)
    assert production["candidates_max"] <= 2 * PRODUCTION_PER_ROUTE_K
    assert production["recall_at_20"] == production["recall_at_10"]
    assert production["recall_at_20"] < 1.0
    # 曲线里达到 1.000 的最浅深度必须明显深于生产深度。
    reaching = sorted(
        row["per_route_k"]
        for row in probe["curve"]
        if row["mode"] == "hybrid" and row["recall_at_20"] >= 1.0
    )
    assert reaching, "曲线里没有任何深度达到 Recall@20 = 1.000"
    assert min(reaching) > PRODUCTION_PER_ROUTE_K


@requires_probe
def test_set_dominance_holds_at_every_measured_depth() -> None:
    """构造性保证：融合表前 k 项的集合恒等于向量 top-k 的集合。

    权重比 0.6/0.4 = 1.5 > (k+60)/61 在 k <= 31 时成立，因此向量 top-k 的 RRF
    得分必然高于任何 BM25 独有候选。这条是 RAG-005 权重选择的真正依据。
    """
    dominance = _probe()["dominance"]
    assert dominance["set_dominance_holds"] is True
    assert all(
        not violations
        for violations in dominance["set_dominance_violations"].values()
    )


@requires_probe
def test_cutoff_dominance_is_not_claimed_beyond_its_range() -> None:
    """支配性只覆盖 `cutoff >= per_route_k`，不覆盖更浅的 cutoff。

    集合相等不含次序相等：同时命中两路的文档得分累加，会在向量 top-k **内部**
    上移，把相关块挤出更浅的 cutoff。台账早期把这条写成了「Recall@k 不低于纯
    向量」的构造保证，范围过宽 —— `k=10/cutoff=5` 就是反例。这条测试同时锁住
    「有保证的区间没有违例」和「无保证的区间确实存在违例」，防止文档回退到
    过宽的说法。
    """
    rows = _probe()["dominance"]["cutoff_rows"]
    guaranteed = [row for row in rows if row["guaranteed"]]
    unguaranteed = [row for row in rows if not row["guaranteed"]]
    assert guaranteed and unguaranteed
    for row in guaranteed:
        assert not row["worse_query_ids"], (
            f"k={row['per_route_k']} cutoff={row['cutoff']} 违反构造性保证"
        )
    assert any(row["worse_query_ids"] for row in unguaranteed), (
        "无保证区间没有任何违例：要么数据变了，要么该性质其实更强，两者都必须"
        "先查清再改文档"
    )


@requires_probe
def test_fusion_beats_vector_only_at_production_depth() -> None:
    """生产深度上融合仍优于纯向量 —— 这是实测结论，不是构造保证。"""
    probe = _probe()
    hybrid = _row(probe, "hybrid", PRODUCTION_PER_ROUTE_K)
    vector = _row(probe, "vector_only", PRODUCTION_PER_ROUTE_K)
    assert hybrid["recall_at_3"] > vector["recall_at_3"]
    assert hybrid["mrr_at_10"] > vector["mrr_at_10"]
    assert hybrid["ndcg_at_10"] > vector["ndcg_at_10"]
    assert hybrid["recall_at_5"] >= vector["recall_at_5"] - 5e-4


@pytest.mark.integration
def test_probe_reproduces_frozen_curve() -> None:
    """重跑探针并与冻结产物逐位比对（零 embedding 调用，约 15 秒）。"""
    import subprocess
    import sys

    if not PROBE_PATH.exists():
        pytest.skip("RAG-003 探针产物不在工作树中")
    with __import__("tempfile").TemporaryDirectory() as directory:
        output = Path(directory) / "curve.json"
        completed = subprocess.run(
            [sys.executable, str(PROBE_SCRIPT), "--json", str(output)],
            cwd=BACKEND_ROOT,
            env={**__import__("os").environ, "PYTHONPATH": str(BACKEND_ROOT)},
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        fresh = json.loads(output.read_text(encoding="utf-8"))
    assert fresh == _probe(), "重跑结果与冻结探针产物不一致"
