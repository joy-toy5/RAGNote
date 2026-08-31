"""量 RAG-013 修 `logits_to_keep` 之后的真实耗时与峰值内存。

背景：A/B 消融第一条 query 就被 OOM kill（exit 137）。原因不是 CPU 算力不够，
而是 `_score_pairs` 先算全序列 logits 再切末位，模型物化了
[batch, seq, 151669] 的 float32 张量（batch=30、seq=600 即 10.17 GiB）。

本脚本用真实语料按真实候选数打分，报告墙钟耗时与进程峰值 RSS，供判断两臂
A/B（2 × 50 条 query）在本机是否跑得完。不写任何评测产物。
"""

from __future__ import annotations

import asyncio
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def peak_rss_gib() -> float:
    """ru_maxrss 在 Linux 上是 KiB。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def load_real_chunks(count: int) -> list[str]:
    """按 chunks.jsonl 的字符偏移切 normalized 文本，即入索引的原文。

    chunks.jsonl 只存偏移不存正文，正文在 corpus.jsonl 指向的
    `normalized_text_path`。按偏移切出来的串跟索引里那份逐字一致，长度分布才有
    代表性 —— 这次 OOM 恰恰是长度导致的。
    """
    import json

    root = Path("evals/datasets/m3_dev_v2")
    text_by_item: dict[str, str] = {}
    with (root / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text_by_item[record["corpus_item_id"]] = (
                root / record["normalized_text_path"]
            ).read_text(encoding="utf-8")

    texts: list[str] = []
    with (root / "chunks.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            whole = text_by_item.get(record["corpus_item_id"])
            if whole is None:
                continue
            piece = whole[record["char_start"] : record["char_end"]]
            if piece.strip():
                texts.append(piece)
            if len(texts) >= count:
                break
    if len(texts) < count:
        raise SystemExit(f"语料不足：要 {count} 条，只有 {len(texts)} 条")
    return texts


def main() -> int:
    from app.rag.reorder_service import ReorderService

    query = "quarantine_policy 默认是什么"
    documents = load_real_chunks(30)
    lengths = [len(d) for d in documents]
    print(f"候选 {len(documents)} 条，字符长度 min={min(lengths)} "
          f"max={max(lengths)} sum={sum(lengths)}")
    print(f"加载前峰值 RSS {peak_rss_gib():.2f} GiB")

    service = ReorderService()

    started = time.monotonic()
    asyncio.run(service.model)  # 触发加载与两项自检
    load_seconds = time.monotonic() - started
    print(f"加载 + 自检 {load_seconds:.2f}s，峰值 RSS {peak_rss_gib():.2f} GiB")

    started = time.monotonic()
    result = asyncio.run(service.reorder_documents(query, documents))
    score_seconds = time.monotonic() - started

    if not result["success"]:
        print(f"重排失败：{result['error']}")
        return 1

    print(f"打分 {len(documents)} 条候选 {score_seconds:.2f}s，"
          f"峰值 RSS {peak_rss_gib():.2f} GiB")
    print(f"单条 query 合计（不含加载）{score_seconds:.2f}s")
    print(f"两臂 A/B 预估：50 × {score_seconds:.2f}s ≈ "
          f"{50 * score_seconds / 60:.1f} 分钟/臂（仅重排部分）")

    top = result["documents"][:3]
    print("top3 分数：" + ", ".join(f"{item['similarity']:.6f}" for item in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
