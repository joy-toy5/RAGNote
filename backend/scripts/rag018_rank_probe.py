"""RAG-018 配套：查一条 query 的候选排名，定位「哪一块在 depth 3→5 之间进了上下文」。

为什么需要它：`rag018_curve_md*.json` 只存了 `candidate_count`，没存候选的 id 和
排名。于是「dev-035 在深度 5 出现假答案」这个观测无法归因到具体某一块 —— 只知道
窗口宽了，不知道宽进来的是什么。没有归因，这个曲线点就只是个数字。

零 API 成本走的是生产自带的 `RagService.get_retrieval_trace`（`rag_service.py:335`），
它只跑到检索为止，不进摘要，因此**结构上**不可能发生生成调用 —— 不依赖任何"应该
不会调到"的假设。`selected_for_context` 标记在 `:426` 完成，早于任何摘要调用，
所以这里读到的排名与真实评测逐位一致。

它还带 `raise_errors=True`：检索异常会抛出而不是伪装成正常拒答，正合归因需要 ——
一次检索故障不该被读成"这一档就是拒答了"。

检索侧与 `rag008_answer_eval.py` 逐位同构（HyDE→identity、跳过 reranker），
否则排名不可比。

刻意不叫 `m3_*`：只读工具不该进 `FINGERPRINT_PATTERNS`。

用法::

    PYTHONPATH=. .venv/bin/python scripts/rag018_rank_probe.py dev-035 --depth 8
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query_id")
    parser.add_argument("--index-dir", default="evals/indexes/m3_dev_v2")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument(
        "--grep",
        default="E2101|不能据此判断|中继缓冲区溢出",
        help="在候选正文里高亮的正则，用来看关键块落在第几名",
    )
    arguments = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(override=False)

    from app.evaluation.dataset import load_dataset
    from app.rag.rag_service import RagService
    from app.rag.vector_store import VectorStoreService
    from app.utils.factory import get_embed_model
    from scripts.rag008_answer_eval import (
        COLLECTION_NAME,
        DATASET_MANIFEST,
        TOP_K,
        _EmptyNoteService,
    )

    embed_model = get_embed_model()
    dataset = load_dataset(DATASET_MANIFEST)
    matched = [q for q in dataset.queries if q.query_id == arguments.query_id]
    if not matched:
        raise SystemExit(f"数据集里没有 {arguments.query_id} —— 不猜。")
    query = matched[0]

    index_dir = Path(arguments.index_dir).resolve()
    if not (index_dir / "chroma.sqlite3").exists():
        raise SystemExit(f"{index_dir} 不像是已构建的 Chroma 索引")

    store = VectorStoreService.for_explicit_target(
        persist_directory=str(index_dir),
        collection_name=COLLECTION_NAME,
        embedding_function=embed_model,
        top_k=TOP_K,
    )
    service = RagService(
        user_id=query.user_id,
        vector_store=store,
        note_service_override=_EmptyNoteService(),
        max_documents=arguments.depth,
    )

    async def _identity_hyde(text: str) -> str:
        return text

    async def _skip_rerank(_text: str, candidates: list) -> list:
        return candidates

    service.generate_hypothetical_document = _identity_hyde
    service.reorder_documents = _skip_rerank

    print(f"query_id        : {query.query_id}")
    print(f"user_id         : {query.user_id}")
    print(f"query           : {query.query}")
    print(f"answerability   : {query.answerability}")
    print(f"no_answer_reason: {getattr(query, 'no_answer_reason', None)}")
    print(f"max_documents   : {arguments.depth}\n")

    trace_object = await service.get_retrieval_trace(
        query.query, query_id=query.query_id
    )
    trace: dict[str, Any] = trace_object.to_dict()
    candidates = trace.get("candidates") or []
    pattern = re.compile(arguments.grep)
    shown = max(arguments.depth, 10)

    print(f"候选共 {len(candidates)} 条；selected_for_context 由 max_documents={arguments.depth} 决定")
    print(f"检索层门禁 no_answer={trace.get('no_answer')}（走 get_retrieval_trace，未进摘要）\n")
    print(f"{'rank':5s} {'in_ctx':7s} {'chunk_id':14s} {'来源':26s} hit  正文首 50 字")
    for rank, candidate in enumerate(candidates[:shown], 1):
        text = str(candidate.get("content") or "")
        flat = " ".join(text.split())
        hit = "HIT" if pattern.search(text) else "   "
        in_ctx = "YES" if candidate.get("selected_for_context") else "no"
        chunk_id = str(candidate.get("chunk_id") or "?")[:12]
        name = str(candidate.get("display_name") or "?")[:24]
        print(f"{rank:<5d} {in_ctx:7s} {chunk_id:14s} {name:26s} {hit}  {flat[:50]}")

    hits = [
        rank
        for rank, candidate in enumerate(candidates, 1)
        if pattern.search(str(candidate.get("content") or ""))
    ]
    print(f"\n命中 /{arguments.grep}/ 的候选排名：{hits or '无'}")
    print("→ <=3 在深度 3 就已进上下文；4~5 只在深度 5 起进；6~8 只在深度 8 起进。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
