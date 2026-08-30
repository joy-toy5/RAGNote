"""`RAG-013` 修复方案验证：按模型卡的 CausalLM yes/no logits 口径打分。

对照 `rag013_reranker_probe.py`（`CrossEncoder` 路径，打分是随机噪声）。
本脚本只读本地 checkpoint，不调 API，不写产物，不改生产代码。

验证三件事，正是台账里 `RAG-013` 的验收标准：
1. 同一 checkpoint 两次独立加载，对同一组 pair 打分**逐位一致**；
2. 构造的相关 / 不相关 pair 分数**方向正确**；
3. batch > 1 能正常工作（左填充 + 显式 pad_token_id）。
"""

from __future__ import annotations

import gc
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 官方 chat template。前后缀必须逐字用原文：分数取的是 assistant 第一个待生成
# 位置上的 logits，模板变一个字，那个位置的分布就变了。
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    'the Query and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

TASK = "Given a web search query, retrieve relevant passages that answer the query"

# 与 rag013_reranker_probe.py 完全同一组 pair，便于对照。
QUERY = "如何重置密码"
DOCS = [
    ("相关  ", "在设置页面点击“重置密码”，输入原密码后提交。"),
    ("不相关", "本产品的年度授权费用为每席位 1200 元。"),
    ("相关  ", "忘记密码时可通过绑定邮箱接收验证码完成重置。"),
    ("不相关", "服务器机房位于华东二区，配备双路供电。"),
]
RELEVANT_IDX = (0, 2)
IRRELEVANT_IDX = (1, 3)


def format_instruction(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def load(model_path: str):
    # padding_side='left' 是硬要求：分数取 logits[:, -1, :]，即序列最后一个位置。
    # 右填充会让最后一个位置变成 pad token，取到的分布与内容无关。
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).eval()
    return tokenizer, model


@torch.no_grad()
def score(tokenizer, model, pairs: list[str], max_length: int = 2048) -> list[float]:
    token_true = tokenizer.convert_tokens_to_ids("yes")
    token_false = tokenizer.convert_tokens_to_ids("no")
    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)

    inputs = tokenizer(
        pairs,
        return_tensors=None,
        add_special_tokens=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
    )
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")

    logits = model(**inputs).logits[:, -1, :]
    stacked = torch.stack([logits[:, token_false], logits[:, token_true]], dim=1)
    return torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp().tolist()


def main() -> int:
    configured = os.getenv(
        "RERANKER_MODEL_PATH", r"D:\Hugging_Face\models\Qwen3-Reranker-0.6B"
    )
    from app.rag.reorder_service import find_model_path

    path = find_model_path(configured)
    print(f"checkpoint: {path}\n")

    pairs = [format_instruction(TASK, QUERY, doc) for _, doc in DOCS]

    runs = []
    for n in (1, 2):
        tokenizer, model = load(path)
        scores = score(tokenizer, model, pairs)
        runs.append(scores)
        print(f"第 {n} 次独立加载（batch={len(pairs)}，一次前向）:")
        for (label, doc), s in zip(DOCS, scores):
            print(f"  {label}  {s:.6f}  {doc[:28]}")
        # 立刻释放再进下一轮。本机可用内存约 5GB，0.6B 参数 float32 约 2.4GB，
        # 两个模型同时驻留会被 OOM killer 杀掉（实测 exit 137）。
        # 顺带这也更贴合验收标准的原意：两次**独立**加载，不是同一实例打两次。
        del model, tokenizer
        gc.collect()
        print()

    a, b = runs
    print("=" * 72)
    print(f"两次打分逐位相同: {a == b}")
    order_a = sorted(range(len(a)), key=lambda i: -a[i])
    order_b = sorted(range(len(b)), key=lambda i: -b[i])
    print(f"两次排序相同    : {order_a == order_b}   {order_a} vs {order_b}")
    for tag, s in (("load-1", a), ("load-2", b)):
        rel = min(s[i] for i in RELEVANT_IDX)
        irr = max(s[i] for i in IRRELEVANT_IDX)
        print(
            f"[{tag}] 相关最低 {rel:.6f} vs 不相关最高 {irr:.6f} -> "
            f"{'方向正确' if rel > irr else '方向错误'}  (间隔 {rel - irr:+.6f})"
        )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise SystemExit(main())
