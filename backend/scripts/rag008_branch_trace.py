"""诊断单条查询的分支级拒答：打印进入生成窗口的文档原文与每个分支的摘要。

只读诊断，不写任何 run 产物。刻意不叫 m3_*，避免进 source_fingerprint。
用法:
    PYTHONPATH=. LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
      .venv/bin/python scripts/rag008_branch_trace.py --index-dir evals/indexes/m3_dev_v2 \
      --query-id dev-003 --model qwen-max
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

DATASET = Path("evals/datasets/m3_dev_v2")


def _load_query(query_id: str) -> dict:
    for line in (DATASET / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["query_id"] == query_id:
            return row
    raise SystemExit(f"未找到 query_id={query_id}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(override=False)

    from app.rag.rag_service import RagService
    from app.rag.vector_store import VectorStoreService
    from app.utils.factory import get_embed_model
    from scripts.rag008_answer_eval import COLLECTION_NAME, TOP_K, _EmptyNoteService

    embed_model = get_embed_model()
    query = _load_query(args.query_id)

    chat_model = None
    if args.model:
        # 与生产、与 rag008_answer_eval 同一构造函数（`RAG-019`）。
        from app.utils.factory import build_aliyun_chat_model

        chat_model = build_aliyun_chat_model(model_name=args.model, streaming=True)

    store = VectorStoreService.for_explicit_target(
        persist_directory=str(Path(args.index_dir).resolve()),
        collection_name=COLLECTION_NAME,
        embedding_function=embed_model,
        top_k=TOP_K,
    )
    service = RagService(
        user_id=query["user_id"],
        vector_store=store,
        note_service_override=_EmptyNoteService(),
    )
    if chat_model is not None:
        service.chat_model = chat_model
        service.chain = service._init_chain()

    async def _identity_hyde(text: str) -> str:
        return text

    async def _skip_rerank(text: str, candidates: list) -> list:
        return candidates

    service.generate_hypothetical_document = _identity_hyde
    service.reorder_documents = _skip_rerank

    # chain 是 pydantic 模型，字段不可增补；用代理对象转发并记录 (context, 输出)。
    calls: list[tuple[str, str]] = []

    class _TracingChain:
        def __init__(self, inner) -> None:
            self._inner = inner

        async def ainvoke(self, payload, *a, **kw):
            out = await self._inner.ainvoke(payload, *a, **kw)
            calls.append((payload.get("context", ""), out))
            return out

        def __getattr__(self, name):
            return getattr(self._inner, name)

    service.chain = _TracingChain(service.chain)

    result = await service.get_documents_and_summary(
        query["query"], query_id=query["query_id"]
    )

    print(f"问题: {query['query']}")
    print(f"answerability: {query.get('answerability')}")
    print(f"最终 no_answer={result.get('no_answer')}")
    print(f"最终 summary: {result.get('summary')!r}")
    print(f"进入生成的文档数: {len(result.get('documents') or [])}")
    print(f"chain 调用次数: {len(calls)}")
    # `RAG-024`：分支健康度。degraded=True 表示有分支失败，此时的答案（或拒答）
    # 是在部分证据缺失的情况下产出的，不能与证据齐全的同类结果混记。
    print(f"generation_health: {result.get('generation_health')}")
    for i, (context, out) in enumerate(calls, 1):
        print(f"\n===== 第 {i} 次 chain 调用 =====")
        print(f"--- context ({len(context)} 字符) ---")
        print(context[:1200])
        print("--- 输出 ---")
        print(repr(out)[:400])


if __name__ == "__main__":
    asyncio.run(main())
