"""RAG-018 配套：把基础设施错误从**两侧**分母里剔除后重算指标。

为什么需要它：`scripts/rag008_answer_eval.py` 的 `score()` 对错误行的处理是不对称的。
一条 answerable 行超时后，生产把它兜底成友好文案且 `no_answer` 仍是 False，于是
`actual=False, predicted=False` 落到最后那个 `else` → 计入 `true_negative`，
**超时被当成"答对了"**。同一条错误若发生在 unanswerable 侧，则走 `elif actual`
→ `false_negative`，方向是保守的。所以：

  - `false_answer_rate`（RAG-018 主验收指标）分母只含 unanswerable，错误在那侧
    被记为 fn，**只会高估不会低估**，可以直接采信。
  - `refusal_rate_answerable`（代价指标）分母 `fp + tn` 被错误行污染，且污染方向
    是把 rate 往下压。depth 越大 → 分支越多 → 超时越多 → 这个指标越"好看"。
    它正好沿着 RAG-018 的自变量方向偏，不修就不能拿来比三档。

本脚本只读产物、不发请求、不改任何生产代码，也**不修**评测脚本本身 —— 三档曲线
必须共用同一个 `script_sha256`，中途改评测脚本会让三个点互相不可比。修正因此以
独立诊断的形式给出。

刻意不叫 `m3_*`：只读工具不该进 `FINGERPRINT_PATTERNS`（见
`scripts/m3_run_eval.py:280`），否则每个未来 run 的 `source_fingerprint` 都会被
一个诊断脚本改掉。

用法::

    PYTHONPATH=. .venv/bin/python scripts/rag018_rescore_excluding_errors.py \\
        evals/runs/rag018_curve_md3.json evals/runs/rag018_curve_md5.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 12)


def rescore(queries: list[dict[str, Any]]) -> dict[str, Any]:
    """与 `rag008_answer_eval.score()` 同口径，但错误行从两侧分母里整体剔除。

    剔除而不是补判：一条超时行的"正确答案应该是什么"是不可观测的，猜它等于伪造
    观测。剔除会掉样本量，那个代价必须显式报出来（`excluded`/`n_scored`），不能
    藏在一个看起来正常的 rate 里。
    """
    tp = fp = fn = tn = 0
    excluded = 0
    for row in queries:
        if row["error"] is not None:
            excluded += 1
            continue
        actual = row["answerability"] == "unanswerable"
        predicted = row["predicted_no_answer"]
        if actual and predicted:
            tp += 1
        elif actual:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1

    answerable_err = sum(
        1
        for row in queries
        if row["error"] is not None and row["answerability"] == "answerable"
    )
    unanswerable_err = sum(
        1
        for row in queries
        if row["error"] is not None and row["answerability"] == "unanswerable"
    )
    return {
        "n_total": len(queries),
        "n_scored": len(queries) - excluded,
        "excluded": excluded,
        "excluded_answerable": answerable_err,
        "excluded_unanswerable": unanswerable_err,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "false_answer_rate_excl": _rate(fn, tp + fn),
        "refusal_rate_answerable_excl": _rate(fp, fp + tn),
    }


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit(
            "用法：rag018_rescore_excluding_errors.py <产物.json> [更多产物.json ...]"
        )

    for raw_path in argv:
        path = Path(raw_path)
        if not path.is_file():
            raise SystemExit(f"产物不存在：{path} —— 不猜、不回退，指标必须有产物支撑。")
        payload = json.loads(path.read_text(encoding="utf-8"))

        passes = payload.get("passes")
        if not isinstance(passes, list) or not passes:
            raise SystemExit(f"读不到 passes 数组：{path} —— 不猜结构。")

        print(f"\n=== {path.name} ===")
        print(
            f"model={payload.get('llm_type')}/{payload.get('model')}  "
            f"max_documents={payload.get('max_documents')}"
            f"（overridden={payload.get('max_documents_overridden')}）"
        )
        print(f"index_version ={payload.get('index_version')}")
        print(f"script_sha256 ={payload.get('script_sha256')}")

        for single_pass in passes:
            queries = single_pass.get("queries")
            if not isinstance(queries, list) or not queries:
                raise SystemExit(f"pass 里读不到 queries 数组：{path} —— 不猜结构。")
            recorded = single_pass.get("metrics") or {}
            fixed = rescore(queries)
            print(f"\n  --- 第 {single_pass.get('attempt')} 轮 ---")
            print(
                f"  错误 {fixed['excluded']}/{fixed['n_total']} 条"
                f"（answerable {fixed['excluded_answerable']} / "
                f"unanswerable {fixed['excluded_unanswerable']}），"
                f"计分样本 {fixed['n_scored']} 条"
            )
            print(
                "  false_answer_rate        原记 "
                f"{recorded.get('false_answer_rate')}  剔错后 "
                f"{fixed['false_answer_rate_excl']}"
            )
            print(
                "  refusal_rate_answerable  原记 "
                f"{recorded.get('refusal_rate_answerable')}  剔错后 "
                f"{fixed['refusal_rate_answerable_excl']}"
            )
            print(
                f"  混淆矩阵（剔错后）tp={fixed['true_positive']} "
                f"fp={fixed['false_positive']} fn={fixed['false_negative']} "
                f"tn={fixed['true_negative']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
