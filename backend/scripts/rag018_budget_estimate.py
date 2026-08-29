"""RAG-018 曲线预算测算：零 API 成本，只读冻结产物与数据集。

为什么单独写一个脚本而不是心算：`RAG-022` 已经把「chain 调用次数 = max_documents + 1」
证伪成上界（`rag_service.py:564` 单条摘要、`:587` 全拒答两处都跳过 reduce），而真实
调用次数取决于每条 query 实际召回了几篇 —— 召回不足 `max_documents` 时 map 阶段就
少调几次。这个数只能从冻结的检索 run 里读，不能估。

刻意不以 `m3_` 开头：`FINGERPRINT_PATTERNS` 只收 `scripts/m3_*.py`，只读诊断工具不
应进 `source_fingerprint` 扰动后续每次 run。

用法：
    .venv/bin/python scripts/rag018_budget_estimate.py
    .venv/bin/python scripts/rag018_budget_estimate.py --settings 3 5 8 --repeat 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

DATASET_MANIFEST = BACKEND / "evals/datasets/m3_dev_v2/manifest.json"
RETRIEVAL_RUN = BACKEND / "evals/runs/m3_dev_v2_hybrid_cjk_rag008.json"

# RAG-022 实测：qwen3.8-max 一次 completion_tokens 332 中 reasoning_tokens 249。
# 思考 token 计费但既不进 content 也不进 reasoning_content，故按可见答案估会低估。
REASONING_BILLING_FACTOR = 332 / 83  # ≈ 4.0，可见 83 token 计费 332


def load_selected_query_ids() -> tuple[list[str], list[str]]:
    """复用生产抽样器本身，避免预算和实跑用两套样本。"""
    from app.evaluation.dataset import load_dataset

    from scripts.rag008_answer_eval import select_queries

    dataset = load_dataset(str(DATASET_MANIFEST))
    selected, _quota = select_queries(dataset)
    answerable = [q.query_id for q in selected if q.answerability == "answerable"]
    unanswerable = [q.query_id for q in selected if q.answerability == "unanswerable"]
    return answerable, unanswerable


def load_candidate_depths() -> dict[str, int]:
    """从冻结检索 run 读每条 query 的候选数，即 len(reordered_documents)。"""
    payload = json.loads(RETRIEVAL_RUN.read_text(encoding="utf-8"))
    depths: dict[str, int] = {}
    rows = payload.get("queries")
    if not rows:
        raise SystemExit(
            f"读不到 queries 数组：{RETRIEVAL_RUN.name} 顶层键为 {list(payload)}。"
            " 不猜结构、不回退默认值 —— 预算数字必须有产物支撑。"
        )
    for row in rows:
        query_id = row.get("query_id")
        if query_id is not None:
            depths[query_id] = len(row.get("candidates") or ())
    return depths


def calls_for(depth: int, max_documents: int, *, all_refused: bool) -> int:
    """map 次数 + 条件 reduce。与 rag_service.py:516-600 的实际分支一致。

    `all_refused` 为真时对应 `:587`「全部分支拒答」早返回，不调 reduce。
    """
    map_calls = min(max_documents, depth)
    if map_calls <= 1:
        return map_calls  # `:564` 单条摘要直接返回，无 reduce
    return map_calls + (0 if all_refused else 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG-018 曲线预算测算（零 API 成本）")
    parser.add_argument("--settings", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()

    answerable, unanswerable = load_selected_query_ids()
    depths = load_candidate_depths()
    all_ids = answerable + unanswerable

    missing = [qid for qid in all_ids if qid not in depths]
    if missing:
        raise SystemExit(
            f"冻结检索 run 里缺 {len(missing)} 条的候选数：{missing}\n"
            f"  产物：{RETRIEVAL_RUN.name}\n"
            "  不用默认值兜底 —— 那会让预算数字失去产物支撑。"
        )

    print(f"样本：可回答 {len(answerable)} 条 + 不可回答 {len(unanswerable)} 条 "
          f"= {len(all_ids)} 条，重复 {args.repeat} 轮")
    print(f"检索深度来源：{RETRIEVAL_RUN.name}（冻结产物，零成本）")
    print()

    depth_list = [depths[qid] for qid in all_ids]
    print(f"候选数分布：min={min(depth_list)} max={max(depth_list)} "
          f"中位={sorted(depth_list)[len(depth_list) // 2]}")
    shallow = [d for d in depth_list if d < max(args.settings)]
    print(f"候选数 < {max(args.settings)} 的条目：{len(shallow)} 条 "
          f"（这些条在大 max_documents 下不会真的多调）")
    print()

    print("口径说明：不可回答那 10 条在冻结产物里 10/10 全拒答且检索门禁 0/10 未触发，")
    print("  即跑完整 map 再走 `:587` 早返回、不调 reduce。下界按此计；上界假设")
    print("  加大窗口后它们不再全拒（那本身就是 RAG-018 要测的风险）。")
    print()
    print(f"{'max_documents':>14} | {'下界/轮':>10} | {'上界/轮':>10} | {'区间':>9}")
    print("-" * 52)
    low_total = high_total = 0
    for setting in args.settings:
        low = sum(
            calls_for(depths[qid], setting, all_refused=qid in unanswerable)
            for qid in all_ids
        )
        high = sum(
            calls_for(depths[qid], setting, all_refused=False) for qid in all_ids
        )
        low_total += low * args.repeat
        high_total += high * args.repeat
        print(f"{setting:>14} | {low:>10} | {high:>10} | {high - low:>9}")

    print("-" * 52)
    print(f"三档合计（含 {args.repeat} 轮）：{low_total} ~ {high_total} 次调用")
    print()
    print(f"计费提醒（RAG-022）：生产模型 qwen3.8-max 计费量约为可见答案的 "
          f"{REASONING_BILLING_FACTOR:.1f} 倍，")
    print("  因为思考 token 计费但不返回。按可见答案长度估算会显著低估。")
    print()
    print("注意：以上是**生成层**调用数。曲线还需要改 max_documents，")
    print("  而它当前是 rag_service.py:520 的行内魔数，评测脚本无 --max-documents 开关。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
