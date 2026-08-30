"""量一下批不变性在左/右填充下各偏多少，用来给自检挑一个有依据的容差。

背景：单对方向断言抓不住右填充（实测右填充下间隔仍有 0.9700）。右填充真正破坏的
性质是**批不变性**——同一文档单独打分与和一条长得多的文档同批打分，分数必须一致。
左填充下末位仍是真实 token，分数应几乎不变；右填充下短序列的末位变成 pad，
分数应显著漂移。

无产物、无 API 调用。
"""

from __future__ import annotations

import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import reorder_service as mod  # noqa: E402

QUERY = "如何重置密码"
SHORT = "在设置页面点击“重置密码”，输入原密码后提交。"
# 刻意长得多，制造大量填充。
LONG = (
    "服务器机房位于华东二区，配备双路供电与柴油发电机组。"
    "机房温度维持在 22 摄氏度，湿度 45% 到 55% 之间。"
    "所有机柜配备独立配电单元，并接入集中式动环监控系统。"
    "运维团队按季度演练断电切换流程，演练记录归档保存三年。"
) * 3


def measure(padding_side: str) -> None:
    original = mod.AutoTokenizer.from_pretrained

    def patched(model_path, **kwargs):
        kwargs["padding_side"] = padding_side
        return original(model_path, **kwargs)

    mod.AutoTokenizer.from_pretrained = staticmethod(patched)
    service = mod.ReorderService()
    try:
        # 绕过自检直接建实例，这里要测的就是自检本身该用什么判据。
        import asyncio

        try:
            asyncio.run(service.model)
        except RuntimeError as exc:
            print(f"  自检拒绝（{padding_side}）：{str(exc).splitlines()[0]}")
            return

        alone = service._score_pairs([mod._format_rerank_input(QUERY, SHORT)])[0]
        batched = service._score_pairs(
            [
                mod._format_rerank_input(QUERY, SHORT),
                mod._format_rerank_input(QUERY, LONG),
            ]
        )[0]
        print(
            f"  padding_side={padding_side:>5}  单独 {alone:.9f}  "
            f"同批 {batched:.9f}  偏差 {abs(alone - batched):.9f}"
        )
    finally:
        mod.AutoTokenizer.from_pretrained = original
        service._model = None
        service._tokenizer = None
        gc.collect()


def main() -> int:
    if not os.getenv("RERANKER_MODEL_PATH"):
        print("需要 RERANKER_MODEL_PATH")
        return 1
    print("批不变性偏差（越小越好）：")
    measure("left")
    measure("right")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
