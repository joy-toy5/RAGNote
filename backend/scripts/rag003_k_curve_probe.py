"""RAG-003：per_route_k 的召回/成本帕累托曲线（零 embedding 调用）。

生产 `chroma.yaml` 用 `k: 5`，而全部 M3 基线都在 `top_k=20` 测的 —— 生产检索
深度从未被测量过。本探针把这条脚注消掉，且不花任何 embedding 调用。

零成本的依据（两条路分开看）：

1. 向量路：Chroma `similarity_search(k=n, filter=...)` 返回按距离升序的前 n 个
   邻居，因此 `k=5` 的结果就是已冻结 `k=20` 结果的**前缀**。同一索引、同一
   embedding、同一 filter 下这是确定的，所以向量排名可以直接从冻结 run
   `m3_dev_v2_vector_only.json` 里截取，不必重新调用 embedding API。
2. BM25 路：纯 CPU，用生产类 `PositiveScoreBM25Retriever` 现算，不碰 embedding。

融合逻辑复刻 `EnsembleRetriever.weighted_reciprocal_rank`：按 `page_content`
聚合 `weight/(rank+c)`（c=60），再对「向量表接 BM25 表」的链做稳定排序。

本探针**不是** run 流水线的一部分，因此故意不命名为 `m3_*`：`m3_run_eval.py`
的 `source_fingerprint` glob 是 `app/**/*.py`、`scripts/m3_*.py`、
`app/config/*.yaml`，把只读测量工具放进去会让每个未来 run 的指纹都因为「加了
个分析脚本」而变化，毁掉与冻结 run 的可比性。探针自身身份见输出里的
`probe_sha256`。

先自校准再出曲线：在 `per_route_k=20` 上复现三个冻结报告的聚合指标，全部逐位
相等才继续。校准失败说明探针与生产路径已经不是同一件事，此时曲线不可信。

用法::

    PYTHONPATH=. .venv/bin/python scripts/rag003_k_curve_probe.py
    PYTHONPATH=. .venv/bin/python scripts/rag003_k_curve_probe.py --json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Any

# 必须在导入任何 app 模块之前关掉，避免离线探针产生网络依赖。
os.environ["LANGCHAIN_TRACING_V2"] = "false"

BACKEND = Path(__file__).resolve().parent.parent
DATASET_MANIFEST = BACKEND / "evals/datasets/m3_dev_v2/manifest.json"
INDEX_DIR = BACKEND / "evals/indexes/m3_dev_v2"
COLLECTION_NAME = "m3-dev-v2-eval"
VECTOR_RUN = BACKEND / "evals/runs/m3_dev_v2_vector_only.json"
RRF_C = 60
SWEEP_K = (3, 5, 8, 10, 15, 20)
# 校准目标：报告目录 -> (模式, 权重)
CALIBRATION = {
    "m3_dev_v2_vector_only": ("vector_only", None),
    "m3_dev_v2_hybrid_cjk": ("hybrid", (0.6, 0.4)),
    "m3_dev_v2_hybrid_cjk_w50": ("hybrid", (0.5, 0.5)),
}
METRIC_KEYS = (
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "mrr_at_10",
    "ndcg_at_10",
    "precision_at_3",
)


def _rounded(value: float) -> float:
    """与 reporting._rounded 同语义：12 位，避免浮点尾差造成假差异。"""
    return round(float(value), 12)


def load_corpus(store: Any) -> tuple[dict[str, Any], list[Any]]:
    """按生产的取法读语料：`vectors_store.get(where={'user_id': ...})`。

    返回 (chunk_id -> Document, 全量 Document 列表)。BM25 侧的并列次序声明是
    「stable_by_corpus_position」，这个位置就是 Chroma `get()` 的返回次序，因此
    必须用同一个入口取，不能自己另排一遍。
    """
    from langchain_core.documents import Document

    raw = store.get(include=["documents", "metadatas"])
    documents = []
    by_chunk_id: dict[str, Any] = {}
    for index, content in enumerate(raw["documents"]):
        metadata = raw["metadatas"][index] if index < len(raw["metadatas"]) else {}
        document = Document(page_content=content, metadata=metadata)
        documents.append(document)
        chunk_id = metadata.get("chunk_id")
        if chunk_id is None:
            raise SystemExit("语料缺少 chunk_id，无法与 qrels 对齐")
        if chunk_id in by_chunk_id:
            raise SystemExit(f"chunk_id 在索引里重复：{chunk_id}")
        by_chunk_id[chunk_id] = document
    return by_chunk_id, documents


def load_vector_ranks(path: Path) -> dict[str, list[str]]:
    """从冻结 vector_only run 取每条 Query 的向量路排名（chunk_id 列表）。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    ranks: dict[str, list[str]] = {}
    for query in payload["queries"]:
        if query["error_code"] is not None:
            raise SystemExit(f"冻结 run 里有错误 Query，不能作为向量基准：{query['query_id']}")
        ordered = sorted(query["candidates"], key=lambda item: item["rank"])
        if [item["rank"] for item in ordered] != list(range(1, len(ordered) + 1)):
            raise SystemExit(f"{query['query_id']} 的 rank 不连续，无法当作前缀截取")
        ranks[query["query_id"]] = [item["chunk_id"] for item in ordered]
    return ranks


def fuse(
    vector_documents: list[Any],
    bm25_documents: list[Any],
    weights: tuple[float, float],
) -> list[Any]:
    """复刻 EnsembleRetriever.weighted_reciprocal_rank（id_key=None 走 page_content）。"""
    scores: dict[str, float] = defaultdict(float)
    for documents, weight in zip((vector_documents, bm25_documents), weights):
        for rank, document in enumerate(documents, start=1):
            scores[document.page_content] += weight / (rank + RRF_C)
    seen: set[str] = set()
    unique: list[Any] = []
    for document in chain(vector_documents, bm25_documents):
        if document.page_content not in seen:
            seen.add(document.page_content)
            unique.append(document)
    # sorted 稳定：并列时保留 chain 次序（向量表在前），与库一致。
    return sorted(unique, reverse=True, key=lambda item: scores[item.page_content])


async def rank_one_query(
    *,
    query_text: str,
    query_id: str,
    user_id: str,
    per_route_k: int,
    mode: str,
    weights: tuple[float, float] | None,
    vector_ranks: dict[str, list[str]],
    by_chunk_id: dict[str, Any],
    bm25_cache: dict[tuple[str, int], Any],
) -> list[str]:
    """返回该 Query 在给定配置下的候选 chunk_id 排名。"""
    from app.rag.retrievers.tokenization import cjk_bigram_tokenize

    frozen = vector_ranks[query_id]
    if per_route_k > len(frozen):
        raise SystemExit(
            f"per_route_k={per_route_k} 超过冻结向量深度 {len(frozen)}，"
            "无法在零 embedding 成本下测量"
        )
    vector_documents = [by_chunk_id[chunk_id] for chunk_id in frozen[:per_route_k]]
    if mode == "vector_only":
        return [document.metadata["chunk_id"] for document in vector_documents]

    retriever = bm25_cache[(user_id, per_route_k)]
    # 权重兜底与生产同源：切不出 token 就整体给向量，不构造融合器。
    if not cjk_bigram_tokenize(query_text) or weights[1] <= 0:
        return [document.metadata["chunk_id"] for document in vector_documents]
    bm25_documents = await retriever.ainvoke(query_text)
    fused = fuse(vector_documents, bm25_documents, weights)
    return [document.metadata["chunk_id"] for document in fused]


async def measure(
    *,
    dataset: Any,
    qrels_by_query: dict[str, tuple[Any, ...]],
    per_route_k: int,
    mode: str,
    weights: tuple[float, float] | None,
    vector_ranks: dict[str, list[str]],
    by_chunk_id: dict[str, Any],
    bm25_cache: dict[tuple[str, int], Any],
) -> dict[str, Any]:
    """跑一格配置，返回宏平均指标 + 候选规模统计。"""
    from app.evaluation.metrics import EvidenceJudgment, ranking_metrics

    per_query: list[Any] = []
    candidate_counts: list[int] = []
    for query in dataset.queries:
        ranked = await rank_one_query(
            query_text=query.query,
            query_id=query.query_id,
            user_id=query.user_id,
            per_route_k=per_route_k,
            mode=mode,
            weights=weights,
            vector_ranks=vector_ranks,
            by_chunk_id=by_chunk_id,
            bm25_cache=bm25_cache,
        )
        candidate_counts.append(len(ranked))
        if query.answerability != "answerable":
            continue
        judgments = tuple(
            EvidenceJudgment(
                evidence_id=entry.anchor_id,
                relevance=entry.relevance,
                chunk_ids=frozenset(entry.chunk_ids),
            )
            for entry in qrels_by_query[query.query_id]
        )
        per_query.append(ranking_metrics(judgments, ranked))

    result = {
        f"recall_at_{cutoff}": _rounded(
            math.fsum(item.recall_at[cutoff] for item in per_query) / len(per_query)
        )
        for cutoff in (3, 5, 10, 20)
    }
    for key, attribute in (
        ("mrr_at_10", "mrr_at_10"),
        ("ndcg_at_10", "ndcg_at_10"),
        ("precision_at_3", "precision_at_3"),
    ):
        values = [getattr(item, attribute) for item in per_query]
        result[key] = _rounded(math.fsum(values) / len(values))
    result["candidates_min"] = min(candidate_counts)
    result["candidates_max"] = max(candidate_counts)
    result["candidates_mean"] = _rounded(
        math.fsum(candidate_counts) / len(candidate_counts)
    )
    return result


async def build_bm25_cache(
    store: Any,
    user_ids: set[str],
    depths: tuple[int, ...],
) -> dict[tuple[str, int], Any]:
    """用生产 HybridRetriever 逐 (user, k) 造 BM25 检索器，保证与生产同构。"""
    from app.rag.retrievers.hybrid_retriever import HybridRetriever

    cache: dict[tuple[str, int], Any] = {}
    for depth in depths:
        retriever = HybridRetriever(store, k=depth)
        for user_id in sorted(user_ids):
            built = await retriever.get_bm25_retriever(user_id)
            if built is None:
                raise SystemExit(f"user_id={user_id} 在索引里没有语料")
            cache[(user_id, depth)] = built
    return cache


def frozen_aggregate(report_dir: Path) -> dict[str, float]:
    payload = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    return {key: payload["aggregate"][key] for key in METRIC_KEYS}


async def run(arguments: argparse.Namespace) -> int:
    from langchain_chroma import Chroma

    from app.evaluation.dataset import load_dataset
    from app.evaluation.qrels import compile_qrels

    if not (INDEX_DIR / "chroma.sqlite3").exists():
        raise SystemExit(f"{INDEX_DIR} 不像是已构建的 Chroma 索引")

    dataset = load_dataset(str(DATASET_MANIFEST))
    qrels = compile_qrels(dataset)
    qrels_by_query: dict[str, list[Any]] = defaultdict(list)
    for entry in qrels.entries:
        qrels_by_query[entry.query_id].append(entry)
    grouped = {key: tuple(value) for key, value in qrels_by_query.items()}

    vector_ranks = load_vector_ranks(VECTOR_RUN)
    if sorted(vector_ranks) != sorted(query.query_id for query in dataset.queries):
        raise SystemExit("冻结 run 的 Query 集合与 dataset 不一致")

    # embedding_function=None：get() 不做任何嵌入，探针因此零 embedding 调用。
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=None,
        persist_directory=str(INDEX_DIR),
    )
    try:
        by_chunk_id, _ = load_corpus(store)
        user_ids = {query.user_id for query in dataset.queries}
        depths = tuple(sorted(set(SWEEP_K) | {20}))
        bm25_cache = await build_bm25_cache(store, user_ids, depths)

        probe_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        print(f"probe_sha256   : {probe_sha256}")
        print(f"index_version  : {dataset.index_version}")
        print(f"queries        : {len(dataset.queries)}（可回答 "
              f"{sum(q.answerability == 'answerable' for q in dataset.queries)}）")

        calibration = await calibrate(
            dataset=dataset,
            grouped=grouped,
            vector_ranks=vector_ranks,
            by_chunk_id=by_chunk_id,
            bm25_cache=bm25_cache,
        )
        if not calibration["passed"]:
            print("\n校准失败：探针与冻结报告不一致，曲线不可信。")
            return 1

        curve = await sweep(
            dataset=dataset,
            grouped=grouped,
            vector_ranks=vector_ranks,
            by_chunk_id=by_chunk_id,
            bm25_cache=bm25_cache,
        )
        dominance = await check_dominance(
            dataset=dataset,
            grouped=grouped,
            vector_ranks=vector_ranks,
            by_chunk_id=by_chunk_id,
            bm25_cache=bm25_cache,
        )
    finally:
        store._client._system.stop()

    if arguments.json:
        Path(arguments.json).write_text(
            json.dumps(
                {
                    "probe_sha256": probe_sha256,
                    "index_version": dataset.index_version,
                    "dataset_sha256": qrels.dataset_sha256,
                    "rrf_c": RRF_C,
                    "calibration": calibration,
                    "curve": curve,
                    "dominance": dominance,
                    "production_context_depth": 3,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n写入 {arguments.json}")
    return 0


async def calibrate(**context: Any) -> dict[str, Any]:
    """在 per_route_k=20 上复现三个冻结报告；逐位相等才认为探针可信。"""
    print("\n=== 校准（per_route_k=20 对齐冻结报告）===")
    rows = []
    passed = True
    for name, (mode, weights) in CALIBRATION.items():
        report_dir = BACKEND / "evals/reports" / name
        if not (report_dir / "report.json").exists():
            raise SystemExit(f"缺少冻结报告：{report_dir}")
        expected = frozen_aggregate(report_dir)
        actual = await measure(
            dataset=context["dataset"],
            qrels_by_query=context["grouped"],
            per_route_k=20,
            mode=mode,
            weights=weights,
            vector_ranks=context["vector_ranks"],
            by_chunk_id=context["by_chunk_id"],
            bm25_cache=context["bm25_cache"],
        )
        mismatches = {
            key: {"frozen": expected[key], "probe": actual[key]}
            for key in METRIC_KEYS
            if expected[key] != actual[key]
        }
        passed = passed and not mismatches
        print(f"{'OK  ' if not mismatches else 'FAIL'} {name}")
        for key, pair in mismatches.items():
            print(f"       {key}: frozen={pair['frozen']} probe={pair['probe']}")
        rows.append({"report": name, "mismatches": mismatches})
    return {"passed": passed, "targets": rows}


async def check_dominance(**context: Any) -> dict[str, Any]:
    """区分两种支配性，因为它们的适用范围不同。

    集合支配：权重比 > (k+60)/61 时，向量 top-k 的 RRF 得分恒高于任何 BM25 独有
    候选，因此 `set(fused[:k]) == set(vector_top_k)`。这是构造性的。

    逐 cutoff 支配：**不成立**。集合相等不含次序相等 —— 同时出现在两路的文档会
    累加得分，从而在向量 top-k **内部**上移，把相关块挤出更浅的 cutoff。因此
    `cutoff < per_route_k` 时融合可能低于纯向量，只有 `cutoff >= per_route_k`
    才有保证。这里把违例逐条数出来，避免把实测通过写成构造保证。
    """
    from app.evaluation.metrics import EvidenceJudgment, ranking_metrics

    print("\n=== 支配性 ===")
    dataset = context["dataset"]
    vector_ranks = context["vector_ranks"]
    set_violations: dict[int, list[str]] = {}
    cutoff_rows = []
    for per_route_k in SWEEP_K:
        violations = []
        discordance: dict[int, dict[str, list[str]]] = {
            cutoff: {"worse": [], "better": []} for cutoff in (3, 5, 10, 20)
        }
        for query in dataset.queries:
            fused = await rank_one_query(
                query_text=query.query,
                query_id=query.query_id,
                user_id=query.user_id,
                per_route_k=per_route_k,
                mode="hybrid",
                weights=(0.6, 0.4),
                vector_ranks=vector_ranks,
                by_chunk_id=context["by_chunk_id"],
                bm25_cache=context["bm25_cache"],
            )
            vector = vector_ranks[query.query_id][:per_route_k]
            if set(fused[:per_route_k]) != set(vector):
                violations.append(query.query_id)
            if query.answerability != "answerable":
                continue
            judgments = tuple(
                EvidenceJudgment(
                    evidence_id=entry.anchor_id,
                    relevance=entry.relevance,
                    chunk_ids=frozenset(entry.chunk_ids),
                )
                for entry in context["grouped"][query.query_id]
            )
            hybrid_metrics = ranking_metrics(judgments, fused)
            vector_metrics = ranking_metrics(judgments, vector)
            for cutoff in (3, 5, 10, 20):
                delta = hybrid_metrics.recall_at[cutoff] - vector_metrics.recall_at[cutoff]
                if delta < 0:
                    discordance[cutoff]["worse"].append(query.query_id)
                elif delta > 0:
                    discordance[cutoff]["better"].append(query.query_id)
        set_violations[per_route_k] = violations
        for cutoff, sides in discordance.items():
            cutoff_rows.append(
                {
                    "per_route_k": per_route_k,
                    "cutoff": cutoff,
                    "guaranteed": cutoff >= per_route_k,
                    "worse_query_ids": sides["worse"],
                    "better_query_ids": sides["better"],
                }
            )
    total_set_violations = sum(len(value) for value in set_violations.values())
    print(f"集合支配违例（全部深度合计）: {total_set_violations}")
    print("逐 cutoff：融合低于纯向量的 Query 数")
    for row in cutoff_rows:
        if not row["worse_query_ids"]:
            continue
        mark = "（本应有保证！）" if row["guaranteed"] else "（无保证，符合预期）"
        print(
            f"  k={row['per_route_k']:<3} cutoff={row['cutoff']:<3} "
            f"更差 {len(row['worse_query_ids'])} 条 {row['worse_query_ids']} {mark}"
        )
    broken = [
        row for row in cutoff_rows if row["guaranteed"] and row["worse_query_ids"]
    ]
    return {
        "set_dominance_violations": {
            str(key): value for key, value in set_violations.items()
        },
        "set_dominance_holds": total_set_violations == 0,
        "cutoff_rows": cutoff_rows,
        "guaranteed_range_broken": broken,
    }


async def sweep(**context: Any) -> list[dict[str, Any]]:
    """对 hybrid(0.6/0.4) 与 vector_only 扫 per_route_k。"""
    print("\n=== per_route_k 曲线 ===")
    header = (
        f"{'mode':<12}{'k':>3}  {'R@3':>7}{'R@5':>7}{'R@10':>7}{'R@20':>7}"
        f"{'MRR@10':>9}{'nDCG@10':>9}{'P@3':>8}  {'cand(min/mean/max)':>20}"
    )
    print(header)
    print("-" * len(header))
    rows = []
    for mode, weights in (("hybrid", (0.6, 0.4)), ("vector_only", None)):
        for per_route_k in SWEEP_K:
            metrics = await measure(
                dataset=context["dataset"],
                qrels_by_query=context["grouped"],
                per_route_k=per_route_k,
                mode=mode,
                weights=weights,
                vector_ranks=context["vector_ranks"],
                by_chunk_id=context["by_chunk_id"],
                bm25_cache=context["bm25_cache"],
            )
            rows.append({"mode": mode, "per_route_k": per_route_k, **metrics})
            print(
                f"{mode:<12}{per_route_k:>3}  "
                f"{metrics['recall_at_3']:>7.4f}{metrics['recall_at_5']:>7.4f}"
                f"{metrics['recall_at_10']:>7.4f}{metrics['recall_at_20']:>7.4f}"
                f"{metrics['mrr_at_10']:>9.4f}{metrics['ndcg_at_10']:>9.4f}"
                f"{metrics['precision_at_3']:>8.4f}  "
                f"{metrics['candidates_min']:>6}/{metrics['candidates_mean']:>6.2f}/"
                f"{metrics['candidates_max']:<6}"
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG-003 per_route_k 曲线（只读）")
    parser.add_argument("--json", default=None, help="把校准与曲线写成 JSON")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
