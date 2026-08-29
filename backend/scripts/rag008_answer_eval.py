"""RAG-008：生成层拒答的真调 API 评测（false_answer_rate 的唯一可测口径）。

为什么需要它：`m3_run_eval.py` 的执行器是 `answer_path="retrieval_only"`，只跑到
检索为止。那条路上 `predicted_no_answer` 只能来自检索层门禁，而门禁在
development 语料上结构性打不响（每用户 35~37 块、两路各取 top-20，鸽笼原理下
交集恒 > 0，见 `scripts/rag008_overlap_probe.py`）。所以离线 run 的
`false_answer_rate = 1.000` 测的不是系统，是「门禁单独的漏判率」。生成层拒答
（`app/prompt/rag_summarize.txt` 第 5 条 + `RagService._is_refusal`）只能靠真
LLM 观测。

隔离设计：**只把生成层变成真的**。HyDE 仍替换为 identity、reranker 仍跳过，
检索侧与冻结 hybrid run 逐位同构，因此指标变化只能归因于生成层。

可复现性状态：
1. （RAG-017 已修）`source_fingerprint` 的 glob 现在包含 `app/prompt/*.txt`，
   改 `rag_summarize.txt` 会改指纹。本脚本仍单独记 `prompt_sha256`：它刻意不叫
   `m3_*`，不进那套 glob，自带记录才能把结果绑到当时的 prompt 上。
2. `ChatTongyi` 只有 `top_p=0.7`，没有 temperature=0，**生成不确定**。20 条样本
   上 false_answer_rate 的分辨率是 0.1，重跑会抖。用 `--repeat 2` 报多次结果，
   不要拿单次当定论。

刻意不叫 `m3_*`：同 `source_fingerprint` 理由，只读评测工具不该改变每个未来 run
的指纹。

用法::

    PYTHONPATH=. .venv/bin/python scripts/rag008_answer_eval.py \\
        --index-dir evals/indexes/m3_dev_v2 \\
        --output evals/runs/rag008_answer_dev20.json --repeat 2
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

# 必须在导入任何 app 模块之前关掉：评测只该产生对 LLM 的调用，不该顺带把
# 中间结果发到第三方追踪服务。放在调用方的环境变量里不可靠，写在这里。
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

BACKEND = Path(__file__).resolve().parent.parent
DATASET_MANIFEST = BACKEND / "evals/datasets/m3_dev_v2/manifest.json"
COLLECTION_NAME = "m3-dev-v2-eval"
PROMPT_FILE = BACKEND / "app/prompt/rag_summarize.txt"
ANSWERABLE_SAMPLE = 10
TOP_K = 20


class _EmptyNotesStore:
    """与 m3_run_eval 同源：笔记候选无 provenance，离线评测必须返回空。

    这里还有第二个理由：笔记证据会绕过检索层门禁（`has_note_evidence`），
    留着会让「生成层单独的表现」混进笔记路的影响。
    """

    @staticmethod
    def similarity_search(*args: Any, **kwargs: Any) -> list:
        return []


class _EmptyNoteService:
    def __init__(self) -> None:
        self.notes_store = _EmptyNotesStore()


def select_queries(dataset: Any) -> list[Any]:
    """10 条不可回答全取 + 10 条可回答按 user 分层定额抽。

    不用随机抽样：按 (user_id, query_id) 排序后轮转取，同一个数据集永远得到同一
    批查询，重跑之间的差异因此只来自 LLM，不来自样本。
    """
    unanswerable = sorted(
        (q for q in dataset.queries if q.answerability == "unanswerable"),
        key=lambda q: q.query_id,
    )
    by_user: dict[str, list[Any]] = {}
    for query in sorted(
        (q for q in dataset.queries if q.answerability == "answerable"),
        key=lambda q: (q.user_id, q.query_id),
    ):
        by_user.setdefault(query.user_id, []).append(query)

    answerable: list[Any] = []
    index = 0
    while len(answerable) < ANSWERABLE_SAMPLE:
        added = False
        for user_id in sorted(by_user):
            bucket = by_user[user_id]
            if index < len(bucket) and len(answerable) < ANSWERABLE_SAMPLE:
                answerable.append(bucket[index])
                added = True
        if not added:
            break
        index += 1
    return sorted(unanswerable + answerable, key=lambda q: q.query_id)


async def answer_one(
    *,
    query: Any,
    index_dir: str,
    vector_store_factory: Any,
    chat_model: Any = None,
) -> dict[str, Any]:
    """跑完整答案路：真 LLM 生成，HyDE/reranker 仍替换掉。"""
    from app.rag.rag_service import INFRASTRUCTURE_FAILURE_MESSAGES, RagService

    store = vector_store_factory()
    service = RagService(
        user_id=query.user_id,
        vector_store=store,
        note_service_override=_EmptyNoteService(),
    )
    if chat_model is not None:
        # 覆盖模型必须连带重建 chain：chain 在 __init__ 里已经绑定了原 chat_model，
        # 只换 self.chat_model 不会影响已经组好的那条链。
        service.chat_model = chat_model
        service.chain = service._init_chain()

    async def _identity_hyde(text: str) -> str:
        return text

    async def _skip_rerank(text: str, candidates: list) -> list:
        return candidates

    # 与离线 run 同样的两处替换：HyDE 与 reranker 不可复现，替掉后检索侧同构。
    service.generate_hypothetical_document = _identity_hyde
    service.reorder_documents = _skip_rerank

    started = time.perf_counter()
    try:
        result = await service.get_documents_and_summary(
            query.query, query_id=query.query_id
        )
        error = None
    except Exception as exc:  # 单条失败不该中断整批，但要如实记下来
        result = {"no_answer": False, "summary": None, "documents": []}
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - started) * 1000

    # 生产把 LLM 失败兜底成友好文案且不外抛异常，`no_answer` 仍是 False。
    # 不在这里认出来，一次 LLM 故障就会被记成「该拒却答了」，与真幻觉不可分辨。
    if error is None and result.get("summary") in INFRASTRUCTURE_FAILURE_MESSAGES:
        error = f"INFRASTRUCTURE_FAILURE:{result['summary']}"

    trace = result.get("retrieval_trace") or {}
    return {
        "query_id": query.query_id,
        "user_id": query.user_id,
        "query": query.query,
        "answerability": query.answerability,
        "no_answer_reason": getattr(query, "no_answer_reason", None),
        "predicted_no_answer": bool(result.get("no_answer")),
        "summary": result.get("summary"),
        "document_count": len(result.get("documents") or []),
        "candidate_count": len(trace.get("candidates") or ()),
        "retrieval_gate_fired": bool(trace.get("no_answer")),
        "error": error,
        "wall_time_ms": round(elapsed_ms, 3),
    }


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """与 `app.evaluation.metrics.no_answer_metrics` 同口径：错误不算正确拒答。"""
    tp = fp = fn = tn = errors = 0
    for row in rows:
        run_error = row["error"] is not None
        errors += int(run_error)
        actual = row["answerability"] == "unanswerable"
        predicted = row["predicted_no_answer"]
        if actual and predicted and not run_error:
            tp += 1
        elif actual:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1

    def _rate(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else round(numerator / denominator, 12)

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "run_error_count": errors,
        "no_answer_precision": _rate(tp, tp + fp),
        "no_answer_recall": _rate(tp, tp + fn),
        # 主验收指标：该拒却答了的比例。
        "false_answer_rate": _rate(fn, tp + fn),
        # 代价指标：能回答却被拒的比例。收益不能只看上一行。
        "refusal_rate_answerable": _rate(fp, fp + tn),
        "gate_fired_count": sum(1 for row in rows if row["retrieval_gate_fired"]),
    }


async def run(arguments: argparse.Namespace) -> int:
    from app.evaluation.dataset import load_dataset
    from app.rag.vector_store import VectorStoreService
    from app.utils.factory import embed_model

    index_dir = Path(arguments.index_dir).resolve()
    if not (index_dir / "chroma.sqlite3").exists():
        raise SystemExit(f"{index_dir} 不像是已构建的 Chroma 索引")
    output = Path(arguments.output)
    if output.exists():
        raise SystemExit(f"{output} 已存在，换个路径以免覆盖既有结果")

    dataset = load_dataset(str(DATASET_MANIFEST))
    selected = select_queries(dataset)
    if len(selected) != 20:
        raise SystemExit(f"预期抽到 20 条，实际 {len(selected)} 条")

    def _factory() -> Any:
        return VectorStoreService.for_explicit_target(
            persist_directory=str(index_dir),
            collection_name=COLLECTION_NAME,
            embedding_function=embed_model,
            top_k=TOP_K,
        )

    prompt_sha256 = hashlib.sha256(PROMPT_FILE.read_bytes()).hexdigest()
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    env_model = os.getenv("CHAT_MODEL_NAME") or os.getenv("ALIYUN_MODEL_NAME")
    model_name = arguments.model or env_model
    chat_model = None
    if arguments.model:
        # 不改 .env：只在本次评测里换模型，并把实际用的名字记进产物。
        from langchain_community.chat_models.tongyi import ChatTongyi

        chat_model = ChatTongyi(
            model=arguments.model,
            api_key=os.getenv("ALIYUN_ACCESS_KEY_SECRET"),
            streaming=True,
            top_p=0.7,
        )
    print(f"script_sha256 : {script_sha256}")
    print(f"prompt_sha256 : {prompt_sha256}  ({PROMPT_FILE.name}，不在 source_fingerprint 内)")
    print(f"model         : {os.getenv('LLM_TYPE', 'ALIYUN')} / {model_name}")
    print(f"index_version : {dataset.index_version}")
    print(f"queries       : {len(selected)}（不可回答 "
          f"{sum(q.answerability == 'unanswerable' for q in selected)}）")

    passes = []
    for attempt in range(1, arguments.repeat + 1):
        print(f"\n=== 第 {attempt} 轮 ===")
        rows = []
        for query in selected:
            row = await answer_one(
                query=query,
                index_dir=str(index_dir),
                vector_store_factory=_factory,
                chat_model=chat_model,
            )
            rows.append(row)
            mark = "拒答" if row["predicted_no_answer"] else "作答"
            correct = (row["predicted_no_answer"]) == (
                row["answerability"] == "unanswerable"
            )
            print(
                f"  {row['query_id']:<9} {row['answerability']:<12} {mark} "
                f"{'OK ' if correct else '错 '} {row['error'] or ''}"
            )
        metrics = score(rows)
        passes.append({"attempt": attempt, "metrics": metrics, "queries": rows})
        print(
            f"  → false_answer_rate={metrics['false_answer_rate']} "
            f"refusal_rate_answerable={metrics['refusal_rate_answerable']} "
            f"errors={metrics['run_error_count']}"
        )

    rates = [item["metrics"]["false_answer_rate"] for item in passes]
    print("\n=== 汇总 ===")
    print(f"false_answer_rate 各轮：{rates}")
    print("生成不确定（ChatTongyi 无 temperature=0，仅 top_p=0.7），"
          "以上为多轮观测值，不是单一定论。")

    output.write_text(
        json.dumps(
            {
                "contract": "rag-note.answer-path-eval.v1",
                "script_sha256": script_sha256,
                "prompt_sha256": prompt_sha256,
                "prompt_file": str(PROMPT_FILE.relative_to(BACKEND)),
                # RAG-017 之后 `app/prompt/*.txt` 已进入 m3_run_eval 的
                # FINGERPRINT_PATTERNS，改 prompt 会改指纹。这里仍单独记
                # prompt_sha256：本脚本刻意不叫 m3_*，不参与那套指纹计算。
                "prompt_in_source_fingerprint": True,
                "answer_path": "generated",
                "llm_type": os.getenv("LLM_TYPE", "ALIYUN"),
                "model": model_name,
                "model_from_env": env_model,
                "model_overridden": bool(arguments.model),
                "top_p": 0.7,
                "deterministic": False,
                "index_version": dataset.index_version,
                "top_k": TOP_K,
                "hyde": "identity_substituted",
                "reranker": "skipped",
                "notes": "empty_note_service",
                "passes": passes,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"写入 {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RAG-008 生成层拒答评测（真调 LLM，有 API 成本）"
    )
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", type=int, default=2, help="重复轮数，默认 2")
    parser.add_argument(
        "--model",
        help="覆盖 CHAT_MODEL_NAME，只作用于本次评测，不写回 .env",
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
