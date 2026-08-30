"""`RAG-013` 复核探针：Qwen3-Reranker-0.6B 经 `CrossEncoder` 加载后打分是否可复现。

只读本地 checkpoint，不调任何 API，不写任何产物。刻意不以 `m3_` 开头，
避免进入 `m3_run_eval.py` 的 `FINGERPRINT_PATTERNS` 扰动后续 run（同 `RAG-018` 两个诊断脚本的处理）。

做两件事：
1. 同一份文件连续加载两次，各自打印 `score.weight` 的求和，看是否相同。
2. 两次加载对**同一组** pair 打分，看排序是否一致。

判据：若 checkpoint 真带训练好的分类头，两次求和必须逐位相同、两次打分必须逐位相同。
"""

from __future__ import annotations

import os
import sys
import warnings

PAIRS = [
    ("查询：如何重置密码", "在设置页面点击“重置密码”，输入原密码后提交。"),  # 相关
    ("查询：如何重置密码", "本产品的年度授权费用为每席位 1200 元。"),  # 不相关
    ("查询：如何重置密码", "忘记密码时可通过绑定邮箱接收验证码完成重置。"),  # 相关
    ("查询：如何重置密码", "服务器机房位于华东二区，配备双路供电。"),  # 不相关
]

RELEVANT_IDX = (0, 2)
IRRELEVANT_IDX = (1, 3)


def load_and_score(model_path: str, tag: str) -> tuple[float | None, list[float]]:
    from sentence_transformers import CrossEncoder

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = CrossEncoder(model_path, max_length=512, device="cpu", local_files_only=True)
        for w in caught:
            text = str(w.message)
            if "newly initialized" in text or "not initialized" in text or "MISSING" in text:
                print(f"  [{tag}] 加载告警: {text.strip()[:200]}")

    inner = getattr(model, "model", None)
    head_sum: float | None = None
    score_w = getattr(inner, "score", None) if inner is not None else None
    if score_w is not None and hasattr(score_w, "weight"):
        head_sum = float(score_w.weight.detach().sum())

    # 逐条打分，不走 batch：`Qwen3ForSequenceClassification` 要靠 `pad_token_id`
    # 定位每条序列的末位非填充 token 来取分类特征，而本 checkpoint 的 config 里
    # 没有 `pad_token_id`，batch > 1 会抛
    # `ValueError: Cannot handle batch sizes > 1 if no padding token is defined.`
    # 这本身就是「该 checkpoint 不是按序列分类模型发布的」又一处证据。
    scores = [float(model.predict([pair])[0]) for pair in PAIRS]
    return head_sum, scores


def main() -> int:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.rag.reorder_service import find_model_path

    configured = os.getenv("RERANKER_MODEL_PATH", r"D:\Hugging_Face\models\Qwen3-Reranker-0.6B")
    path = find_model_path(configured)
    print(f"checkpoint: {path}\n")

    print("第 1 次加载:")
    sum_a, scores_a = load_and_score(path, "load-1")
    print(f"  score.weight 求和: {sum_a}")
    print(f"  打分: {[round(s, 6) for s in scores_a]}\n")

    print("第 2 次加载（同一份文件，同一进程内重新实例化）:")
    sum_b, scores_b = load_and_score(path, "load-2")
    print(f"  score.weight 求和: {sum_b}")
    print(f"  打分: {[round(s, 6) for s in scores_b]}\n")

    print("=" * 70)
    print(f"两次 score.weight 求和相同: {sum_a == sum_b}   ({sum_a} vs {sum_b})")
    print(f"两次打分逐位相同        : {scores_a == scores_b}")
    print(f"两次排序相同            : "
          f"{sorted(range(len(scores_a)), key=lambda i: -scores_a[i]) == sorted(range(len(scores_b)), key=lambda i: -scores_b[i])}")

    for tag, scores in (("load-1", scores_a), ("load-2", scores_b)):
        rel = min(scores[i] for i in RELEVANT_IDX)
        irr = max(scores[i] for i in IRRELEVANT_IDX)
        verdict = "方向正确" if rel > irr else "方向错误"
        print(f"[{tag}] 相关 pair 最低分 {rel:.6f} vs 不相关 pair 最高分 {irr:.6f} -> {verdict}")

    print("=" * 70)
    print("判据：若 checkpoint 带训练好的分类头，上面三个「相同」必须全为 True。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
