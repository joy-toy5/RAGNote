"""探针：测出 map 阶段单分支的真实延迟分布，为 `RAG-024` 的超时取值提供依据。

为什么需要它（`RAG-024`）：`rag_service.py:562-565`（map）与 `:629-632`（reduce）
各有一道硬编码 `timeout=30.0`。三档曲线在 `qwen3.8-max` 上实测 answerable 侧
45%~65% 的行撞墙报废，而**撞墙的行只知道「超过 30s」，不知道究竟要多久** —— 被
`wait_for` 截断的样本对真实分布是右删失（right-censored）的，拿它反推取值等于
猜。要定 30s 该改成多少，必须先在**没有墙**的条件下量一次。

`.env` 的 `CHAT_MODEL_NAME` 已换成 `qwen3.7-plus`（`qwen3.8-max` 额度耗尽），
所以旧分布整体作废：延迟是模型属性，换模型必须重测。本探针同时验证新模型可达 ——
`RAG-019` 的教训是「配置里写着的模型不等于生产到得了的模型」，不实调不算验证。

设计要点，缺一不可：

1. **必须复现真实 map 调用**，不能拿短问题测。真实调用是「生产 prompt +
   单个真实 chunk 全文 + 真实查询」，输入长度直接决定延迟。因此这里走生产
   `RagService` 自己的 `self.chain`，context 用 `:558` 同样的
   `f"【参考资料{i}】:{doc}\n"` 拼法，query 用数据集原文。
2. **不设 timeout**，让调用跑到自然结束。这正是本探针存在的理由；设了墙就又得到
   一份删失数据。
3. **选样本要选真撞过墙的**：从三份冻结产物里取超时次数最多的几条 answerable
   查询。在从未超时的查询上测延迟，测出来的分布不覆盖要决策的那段右尾。
4. 检索侧走零成本的 `get_retrieval_trace`（`rag_service.py:335`），与真实评测
   逐位同构（`selected_for_context` 在 `:426` 标记，早于任何摘要调用）。

成本：真实生成调用，`--queries N × --depth D` 次 map 调用（默认 4×3=12 次）。
不跑 reduce —— reduce 的输入是各分支摘要，无法在不先跑完 map 的情况下单独测，
且 map 侧右尾才是 33/34 条撞墙行的所在。reduce 那道墙另有 1 条实测样本
（dev-035，md8 第 2 轮，37795 ms）。

只读：不写产物、不改 `.env`、不改生产代码，只打印。刻意不叫 `m3_*` —— 只读诊断
不该进 `FINGERPRINT_PATTERNS`（`m3_run_eval.py:280`）而扰动后续每次 run。

用法：
    PYTHONPATH=. .venv/bin/python scripts/rag024_latency_probe.py \
        [--queries 4] [--depth 3] [--repeat 1]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 与 `rag008_answer_eval.py` 同源，避免两处各写一份常量而悄悄漂移。
from scripts.rag008_answer_eval import (  # noqa: E402
    COLLECTION_NAME,
    DATASET_MANIFEST,
    TOP_K,
    _EmptyNoteService,
)

# PROBE_QUERIES 不写死内容，只写 id：内容以数据集为准，写死会与数据集漂移。
# 这四条是三份冻结产物里 answerable 侧超时次数最多的（6/6、6/6、5/6、5/6），
# 其中 dev-027 正是 `RAG-018` 验收标准点名、却从未成功过一次的那条。
DEFAULT_PROBE_QUERY_IDS = ("dev-027", "dev-037", "dev-002", "dev-003")

PRODUCTION_TIMEOUT_S = 30.0  # rag_service.py:564 / :631 的现值，用于对照


def _build_service(query: object, index_dir: Path) -> object:
    """与 `rag008_answer_eval.answer_one` 同构地建 service。

    同构是硬要求：HyDE 与 reranker 的两处替换若漏掉，检索侧就与真实评测不同，
    量到的延迟对不上要解释的那批行。
    """
    from app.rag.rag_service import RagService
    from app.rag.vector_store import VectorStoreService
    from app.utils.factory import embed_model

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
    )

    async def _identity_hyde(text: str) -> str:
        return text

    async def _skip_rerank(text: str, candidates: list) -> list:
        return candidates

    service.generate_hypothetical_document = _identity_hyde
    service.reorder_documents = _skip_rerank
    return service


async def probe_one(
    *, service: object, query: object, depth: int, repeat: int
) -> list[dict[str, object]]:
    """对一条查询的前 `depth` 个分支各跑 `repeat` 次无墙 map 调用。"""
    # 检索走零成本路径：只到检索为止，结构上不可能发生生成调用。
    trace = await service.get_retrieval_trace(query.query, query_id=query.query_id)
    candidates = trace.candidates[:depth]

    rows: list[dict[str, object]] = []
    for branch_index, candidate in enumerate(candidates, 1):
        # 与 rag_service.py:558 逐字同样的拼法，包括那个换行。
        single_context = f"【参考资料{branch_index}】:{candidate.reranker_text}\n"
        for attempt in range(1, repeat + 1):
            started = time.perf_counter()
            failure: str | None = None
            summary = ""
            try:
                # 关键：不套 asyncio.wait_for —— 要的就是没有墙的那个真值。
                summary = await service.chain.ainvoke(
                    {"input": query.query, "context": single_context}
                )
            except Exception as exc:  # noqa: BLE001 - 探针要如实记下任何失败
                failure = f"{type(exc).__name__}: {exc}"
            elapsed_ms = (time.perf_counter() - started) * 1000
            rows.append(
                {
                    "query_id": query.query_id,
                    "branch": branch_index,
                    "attempt": attempt,
                    "elapsed_ms": elapsed_ms,
                    "over_wall": elapsed_ms > PRODUCTION_TIMEOUT_S * 1000,
                    "chars_in": len(single_context),
                    "chars_out": len(summary or ""),
                    "failure": failure,
                    "is_refusal": (
                        service._is_refusal(summary) if failure is None else None
                    ),
                }
            )
            flag = "  ← 越过 30s 墙" if rows[-1]["over_wall"] else ""
            state = failure or (
                "拒答" if rows[-1]["is_refusal"] else f"作答 {rows[-1]['chars_out']} 字"
            )
            print(
                f"  {query.query_id} 分支{branch_index} 第{attempt}次: "
                f"{elapsed_ms:8.0f} ms  入{rows[-1]['chars_in']:5d} 字  {state}{flag}",
                flush=True,
            )
    return rows


def report(rows: list[dict[str, object]], *, depth: int) -> None:
    """打印分布与取值建议。取值只给依据，不替人决定。"""
    ok = [r for r in rows if r["failure"] is None]
    failed = [r for r in rows if r["failure"] is not None]
    if not ok:
        raise SystemExit("全部调用失败 —— 不给取值建议，先解决可达性。")

    times = sorted(float(r["elapsed_ms"]) for r in ok)
    over = [r for r in ok if r["over_wall"]]

    def pct(p: float) -> float:
        # 小样本上 quantiles 不稳，直接按序位取，并把样本量一起报出来。
        index = min(len(times) - 1, max(0, round(p / 100 * len(times)) - 1))
        return times[index]

    print("\n=== map 单分支延迟分布（无墙实测）===")
    print(f"样本 n={len(times)}（失败 {len(failed)} 次）")
    print(f"  最小 {times[0]:8.0f} ms")
    print(f"  中位 {statistics.median(times):8.0f} ms")
    print(f"  p90  {pct(90):8.0f} ms")
    print(f"  最大 {times[-1]:8.0f} ms")
    print(
        f"  越过现行 30s 墙：{len(over)}/{len(times)} 次"
        + ("" if over else "  ← 本次样本中无一越墙")
    )

    print("\n=== 逐条最大值（决策看的是右尾，不是均值）===")
    by_query: dict[str, list[float]] = {}
    for row in ok:
        by_query.setdefault(str(row["query_id"]), []).append(float(row["elapsed_ms"]))
    for query_id in sorted(by_query):
        values = sorted(by_query[query_id])
        print(
            f"  {query_id}: n={len(values)} 中位 {statistics.median(values):7.0f} ms "
            f"最大 {values[-1]:7.0f} ms"
        )

    print("\n=== 取值依据（不是结论）===")
    print(
        f"  整条查询的墙至少要容纳最慢的那个分支：本次最大 {times[-1]:.0f} ms。"
    )
    print(
        "  注意 map 是并发的（`gather`），所以整条 map 阶段耗时 ≈ 最慢分支，"
        f"而非 {depth} 个分支之和；reduce 那道墙要另算一次调用。"
    )
    if failed:
        print(f"  有 {len(failed)} 次调用失败，取值前须先看清失败原因：")
        for row in failed[:5]:
            print(f"    {row['query_id']} 分支{row['branch']}: {row['failure']}")
    print(
        "  样本量小（本次 n="
        f"{len(times)}），只够定量级、不够定分位点。若要把取值写进生产，"
        "建议把它做成配置项而不是又一个魔数。"
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default="evals/indexes/m3_dev_v2")
    parser.add_argument(
        "--queries",
        type=int,
        default=4,
        help="从 DEFAULT_PROBE_QUERY_IDS 取前 N 条（默认 4）",
    )
    parser.add_argument("--depth", type=int, default=3, help="每条取前 N 个候选分支")
    parser.add_argument("--repeat", type=int, default=1, help="每个分支重复次数")
    parser.add_argument(
        "--query-ids",
        nargs="*",
        default=None,
        help="显式指定 query_id，覆盖 --queries",
    )
    arguments = parser.parse_args()

    from app.evaluation.dataset import load_dataset

    index_dir = Path(arguments.index_dir).resolve()
    if not (index_dir / "chroma.sqlite3").exists():
        raise SystemExit(f"{index_dir} 不像是已构建的 Chroma 索引 —— 不猜、不回退。")

    dataset = load_dataset(str(DATASET_MANIFEST))
    wanted = list(arguments.query_ids or DEFAULT_PROBE_QUERY_IDS[: arguments.queries])
    by_id = {q.query_id: q for q in dataset.queries}
    missing = [qid for qid in wanted if qid not in by_id]
    if missing:
        raise SystemExit(f"数据集里没有这些 query_id：{missing} —— 不猜。")
    queries = [by_id[qid] for qid in wanted]

    # 必须先 import 工厂再读环境变量：`.env` 是工厂 import 时才载入的，早读会得到
    # None 并打出一个假的模型名（初版就是这个缺陷）。模型名直接问客户端本人，
    # 不问环境变量 —— 「配置里写着的」不等于「客户端真用的」，那正是 `RAG-019`。
    from app.utils.factory import chat_model as _production_chat_model

    actual_model = getattr(_production_chat_model, "model_name", None) or getattr(
        _production_chat_model, "model", "<读不到>"
    )
    total = len(queries) * arguments.depth * arguments.repeat
    # base_url 要读 `root_client`：`client` 是 Completions 资源对象，它身上没有
    # base_url，读它会得到一个对象的 repr 而不是端点（初版就是这个缺陷）。
    actual_base_url = getattr(
        getattr(_production_chat_model, "root_client", None), "base_url", "<读不到>"
    )
    print(f"模型（问客户端本人，非读 .env）: {actual_model}")
    print(f"端点（同上）            : {actual_base_url}")
    print(f"index_version           : {dataset.index_version}")
    print(f"现行生产超时            : {PRODUCTION_TIMEOUT_S:.0f}s（map 与 reduce 各一道）")
    print(f"计划真实生成调用        : {total} 次（无 timeout，跑到自然结束）")
    print(f"样本                    : {', '.join(wanted)}\n")

    rows: list[dict[str, object]] = []
    for query in queries:
        service = _build_service(query, index_dir)
        rows.extend(
            await probe_one(
                service=service,
                query=query,
                depth=arguments.depth,
                repeat=arguments.repeat,
            )
        )

    report(rows, depth=arguments.depth)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
