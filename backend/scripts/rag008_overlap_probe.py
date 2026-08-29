"""RAG-008：双路交集分布（零 embedding 调用）。

拒答门禁的判据是「向量路与词法路是否指向同一批证据」。上线这条门禁前必须先
知道交集在 development 集上的实际分布 —— 否则门禁要么永不触发（白加），要么
误伤可回答查询（更糟）。本探针只测量，不改任何行为。

零成本的依据与 RAG-003 探针相同：
1. 向量路：Chroma `similarity_search(k=n, filter=...)` 的结果是冻结 run 的前缀，
   直接从 `m3_dev_v2_vector_only.json` 截取，不重复调用 embedding API；
2. 词法路：纯 CPU，用生产类现算。

交集必须按 `page_content` 统计：生产构造 `EnsembleRetriever` 时没传 `id_key`，
库内 RRF 的去重与累加都按 `page_content`。换成 chunk_id 会让「两路命中同一
文档」与「RRF 实际折叠了哪些」不是同一件事。

刻意不叫 `m3_*`：`m3_run_eval.py` 的 `source_fingerprint` glob 是
`app/**/*.py`、`scripts/m3_*.py`、`app/config/*.yaml`，把只读测量工具放进去会让
每个未来 run 的指纹都因为「加了个分析脚本」而变化。探针身份见输出 probe_sha256。

用法::

    PYTHONPATH=. .venv/bin/python scripts/rag008_overlap_probe.py
    PYTHONPATH=. .venv/bin/python scripts/rag008_overlap_probe.py --json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

# 必须在导入任何 app 模块之前关掉，避免离线探针产生网络依赖。
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

BACKEND = Path(__file__).resolve().parent.parent
DATASET_MANIFEST = BACKEND / "evals/datasets/m3_dev_v2/manifest.json"
INDEX_DIR = BACKEND / "evals/indexes/m3_dev_v2"
COLLECTION_NAME = "m3-dev-v2-eval"
VECTOR_RUN = BACKEND / "evals/runs/m3_dev_v2_vector_only.json"
HYBRID_RUN = BACKEND / "evals/runs/m3_dev_v2_hybrid_cjk.json"
PER_ROUTE_K = 20
WEIGHTS = (0.6, 0.4)


def load_corpus(store: Any) -> dict[str, Any]:
    """按生产入口读语料，保留 Chroma `get()` 的返回次序。

    BM25 的并列次序声明是 stable_by_corpus_position，这个「位置」就是 get() 的
    次序，所以必须用同一个入口取，不能自己另排一遍。
    """
    from langchain_core.documents import Document

    raw = store.get(include=["documents", "metadatas"])
    by_chunk_id: dict[str, Any] = {}
    for index, content in enumerate(raw["documents"]):
        metadata = raw["metadatas"][index] if index < len(raw["metadatas"]) else {}
        chunk_id = metadata.get("chunk_id")
        if chunk_id is None:
            raise SystemExit("语料缺少 chunk_id，无法与冻结 run 对齐")
        if chunk_id in by_chunk_id:
            raise SystemExit(f"chunk_id 在索引里重复：{chunk_id}")
        by_chunk_id[chunk_id] = Document(page_content=content, metadata=metadata)
    return by_chunk_id


def load_vector_ranks(path: Path) -> dict[str, list[str]]:
    """从冻结 vector_only run 取每条查询的向量路排名（chunk_id 列表）。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    ranks: dict[str, list[str]] = {}
    for query in payload["queries"]:
        if query["error_code"] is not None:
            raise SystemExit(f"冻结 run 里有错误查询，不能当向量基准：{query['query_id']}")
        ordered = sorted(query["candidates"], key=lambda item: item["rank"])
        if [item["rank"] for item in ordered] != list(range(1, len(ordered) + 1)):
            raise SystemExit(f"{query['query_id']} 的 rank 不连续，无法当前缀截取")
        ranks[query["query_id"]] = [item["chunk_id"] for item in ordered]
    return ranks


async def measure_query(
    *,
    query: Any,
    vector_ranks: dict[str, list[str]],
    by_chunk_id: dict[str, Any],
    bm25_cache: dict[str, Any],
    weight_probe: Any,
) -> dict[str, Any]:
    """测一条查询的两路交集，判据与生产 `retrieve_with_routes` 同源。"""
    from app.rag.retrievers.hybrid_retriever import _fusion_key

    frozen = vector_ranks[query.query_id]
    if PER_ROUTE_K > len(frozen):
        raise SystemExit(
            f"PER_ROUTE_K={PER_ROUTE_K} 超过冻结向量深度 {len(frozen)}，"
            "无法在零 embedding 成本下测量"
        )
    vector_documents = [by_chunk_id[chunk_id] for chunk_id in frozen[:PER_ROUTE_K]]
    bm25_retriever = bm25_cache[query.user_id]
    # 权重判据与生产同源：离线评测不开 HyDE，weight_query 就是原始查询。
    weights = await weight_probe.get_dynamic_weights(query.query)

    if not (bm25_retriever and weights[1] > 0):
        return {
            "query_id": query.query_id,
            "answerability": query.answerability,
            "both_routes_present": False,
            "weights": [_rounded(weights[0]), _rounded(weights[1])],
            "vector_hits": len(vector_documents),
            "bm25_hits": None,
            "overlap_count": None,
            "would_refuse": False,
        }

    bm25_documents = await bm25_retriever.ainvoke(query.query)
    vector_keys = {_fusion_key(document) for document in vector_documents}
    bm25_keys = {_fusion_key(document) for document in bm25_documents}
    overlap = len(vector_keys & bm25_keys)
    return {
        "query_id": query.query_id,
        "answerability": query.answerability,
        "both_routes_present": True,
        "weights": [_rounded(weights[0]), _rounded(weights[1])],
        "vector_hits": len(vector_documents),
        "bm25_hits": len(bm25_documents),
        "overlap_count": overlap,
        # 生产门禁：两路都在且交集 0 且无笔记证据 → 拒答。离线路笔记关闭。
        "would_refuse": overlap == 0,
    }


def _rounded(value: float) -> float:
    return round(float(value), 12)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按可回答性分组统计，因为门禁的收益与代价分别落在两组里。"""
    groups: dict[str, list[dict[str, Any]]] = {"answerable": [], "unanswerable": []}
    for row in rows:
        key = "answerable" if row["answerability"] == "answerable" else "unanswerable"
        groups[key].append(row)

    summary: dict[str, Any] = {}
    for key, group in groups.items():
        overlaps = [row["overlap_count"] for row in group if row["overlap_count"] is not None]
        summary[key] = {
            "queries": len(group),
            "single_route_fallback": sum(1 for row in group if not row["both_routes_present"]),
            "overlap_min": min(overlaps) if overlaps else None,
            "overlap_max": max(overlaps) if overlaps else None,
            "overlap_mean": _rounded(sum(overlaps) / len(overlaps)) if overlaps else None,
            "zero_overlap": [row["query_id"] for row in group if row["overlap_count"] == 0],
            "would_refuse": sum(1 for row in group if row["would_refuse"]),
        }
    return summary


async def run(arguments: argparse.Namespace) -> int:
    from langchain_chroma import Chroma

    from app.evaluation.dataset import load_dataset
    from app.rag.retrievers.hybrid_retriever import HybridRetriever

    if not (INDEX_DIR / "chroma.sqlite3").exists():
        raise SystemExit(f"{INDEX_DIR} 不像是已构建的 Chroma 索引")

    dataset = load_dataset(str(DATASET_MANIFEST))
    vector_ranks = load_vector_ranks(VECTOR_RUN)
    if sorted(vector_ranks) != sorted(query.query_id for query in dataset.queries):
        raise SystemExit("冻结 run 的查询集合与 dataset 不一致")

    # embedding_function=None：get() 不做嵌入，探针因此零 embedding 调用。
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=None,
        persist_directory=str(INDEX_DIR),
    )
    try:
        by_chunk_id = load_corpus(store)
        retriever = HybridRetriever(store, k=PER_ROUTE_K, fusion_weights=WEIGHTS)
        bm25_cache: dict[str, Any] = {}
        for user_id in sorted({query.user_id for query in dataset.queries}):
            built = await retriever.get_bm25_retriever(user_id)
            if built is None:
                raise SystemExit(f"user_id={user_id} 在索引里没有语料")
            bm25_cache[user_id] = built

        rows = [
            await measure_query(
                query=query,
                vector_ranks=vector_ranks,
                by_chunk_id=by_chunk_id,
                bm25_cache=bm25_cache,
                weight_probe=retriever,
            )
            for query in dataset.queries
        ]
    finally:
        store._client._system.stop()

    probe_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = summarise(rows)
    print(f"probe_sha256   : {probe_sha256}")
    print(f"index_version  : {dataset.index_version}")
    print(f"per_route_k    : {PER_ROUTE_K}  weights={WEIGHTS}")
    print(f"queries        : {len(rows)}")
    print("\n=== 交集分布（按可回答性）===")
    for key in ("answerable", "unanswerable"):
        stats = summary[key]
        print(
            f"{key:<14} n={stats['queries']:<3} "
            f"单路兜底={stats['single_route_fallback']:<3} "
            f"交集 min/mean/max={stats['overlap_min']}/{stats['overlap_mean']}/"
            f"{stats['overlap_max']}  会拒答={stats['would_refuse']}"
        )

    print("\n=== 逐条（交集升序）===")
    for row in sorted(rows, key=lambda item: (item["overlap_count"] is not None, item["overlap_count"] or 0)):
        flag = "拒答" if row["would_refuse"] else "    "
        overlap = "None" if row["overlap_count"] is None else str(row["overlap_count"])
        print(
            f"{row['query_id']:<9} {row['answerability']:<22} "
            f"w={row['weights'][1]:<5} bm25={row['bm25_hits']} 交集={overlap:<3} {flag}"
        )

    if arguments.json:
        Path(arguments.json).write_text(
            json.dumps(
                {
                    "probe_sha256": probe_sha256,
                    "index_version": dataset.index_version,
                    "per_route_k": PER_ROUTE_K,
                    "weights": list(WEIGHTS),
                    "summary": summary,
                    "per_query": rows,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG-008 双路交集分布探针（只读）")
    parser.add_argument("--json", help="把逐条结果写到这个路径")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
