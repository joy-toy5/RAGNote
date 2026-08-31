"""量微批切分对分数的实际影响，定性 `test_micro_batching_does_not_change_scores`。

那条 external 测试失败了，但断言消息一直没拿到：同一个 pytest session 里连续
加载四次模型把内存推到 3.74 GiB（可用 5 GiB），进程被 OOM kill，pytest 的输出
没来得及落盘。

本脚本单进程、只加载一次模型，直接把两侧分数打出来，用于区分三种可能：

1. 逐位不等但偏差是浮点量级、排序不变 —— 分批走了不同 kernel。等价性断言应按
   实测量级给容差，而不是凭空写 1e-6。
2. 偏差大或排序变了 —— 分批真的改了分数，微批方案要撤回重做。
3. 逐位相等 —— 测试失败是 session 内存累积的进程级问题，该改测试的隔离方式，
   不该动生产代码。

不写任何评测产物。
"""

from __future__ import annotations

import asyncio
import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUERY = "如何重置密码"
DOCS = [
    "在设置页面点击“重置密码”，输入原密码后提交。",
    "本产品的年度授权费用为每席位 1200 元。",
    "忘记密码时可通过绑定邮箱接收验证码完成重置。",
    "服务器机房位于华东二区，配备双路供电。",
]


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def main() -> int:
    from app.rag import reorder_service as module

    service = module.ReorderService()
    asyncio.run(service.model)  # 加载 + 两项自检
    print(f"加载完成，峰值 RSS {peak_rss_gib():.2f} GiB")
    print(f"_RERANK_MICRO_BATCH = {module._RERANK_MICRO_BATCH}")

    rendered = [module._format_rerank_input(QUERY, d) for d in DOCS]

    one_shot = service._score_batch(rendered)

    original = module._RERANK_MICRO_BATCH
    module._RERANK_MICRO_BATCH = 1
    try:
        split = service._score_pairs(rendered)
    finally:
        module._RERANK_MICRO_BATCH = original

    print("\n一次装完（batch=4，单次 forward）:")
    for doc, score in zip(DOCS, one_shot):
        print(f"  {score:.12f}  {doc[:24]}")
    print("逐条打分（micro_batch=1，4 次 forward）:")
    for doc, score in zip(DOCS, split):
        print(f"  {score:.12f}  {doc[:24]}")

    identical = one_shot == split
    deviations = [abs(a - b) for a, b in zip(one_shot, split)]
    max_dev = max(deviations)
    order_one = sorted(range(len(DOCS)), key=lambda i: -one_shot[i])
    order_split = sorted(range(len(DOCS)), key=lambda i: -split[i])

    print(f"\n逐位相等          : {identical}")
    print(f"最大偏差          : {max_dev:.12e}")
    print(f"一次装完排序      : {order_one}")
    print(f"逐条打分排序      : {order_split}")
    print(f"排序一致          : {order_one == order_split}")
    print(f"当前自检容差      : {module._SELFTEST_MAX_BATCH_DRIFT:g}")
    print(f"偏差是否超自检容差: {max_dev > module._SELFTEST_MAX_BATCH_DRIFT}")
    print(f"峰值 RSS          : {peak_rss_gib():.2f} GiB")

    if identical:
        print("\n结论：分批未改变分数，测试失败属进程级内存问题（可能 3）。")
    elif order_one == order_split and max_dev < 1e-3:
        print("\n结论：浮点量级偏差、排序不变（可能 1）。等价性断言按此量级给容差。")
    else:
        print("\n结论：分批改变了分数或排序（可能 2）。微批方案需撤回重做。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
