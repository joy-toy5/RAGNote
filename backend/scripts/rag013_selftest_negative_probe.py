"""验证 `RAG-013` 的加载自检**真的会拒绝**，而不是只在正常情况下放行。

一道只在正常时通过的自检没有价值——缺陷的本质是失败不可见，所以必须证明
自检在坏掉时会炸。这里注入两种真实存在过的坏法：
  1. 右填充（`padding_side="right"`）——分数会取到 pad 位置，与内容无关；
  2. 模板被改（丢掉官方 suffix）——测量口径变了。
两者都不抛异常、都返回形状正确的浮点数，正是「静默返回噪声」的形态。

无产物、无 API 调用，纯本地推理。
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import reorder_service as mod  # noqa: E402


def run_case(name: str, mutate, expect_reject: bool) -> bool:
    """返回结果是否符合预期。"""
    service = mod.ReorderService()
    mutate(service)
    rejected = False
    detail = ""
    try:
        asyncio.run(service.model)
    except RuntimeError as exc:
        rejected = True
        detail = str(exc).splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        print(f"[其他异常] {name}: {type(exc).__name__}: {exc}")
        return False
    finally:
        service._model = None
        service._tokenizer = None
        gc.collect()

    ok = rejected == expect_reject
    verdict = "拒绝" if rejected else "放行"
    mark = "符合预期" if ok else "不符合预期"
    print(f"[{verdict}／{mark}] {name}")
    if detail:
        print(f"        {detail}")
    return ok


def main() -> int:
    path = os.getenv("RERANKER_MODEL_PATH", "")
    if not path:
        print("需要 RERANKER_MODEL_PATH")
        return 1

    results = []

    print("== 正常加载（应放行）==")
    results.append(run_case("官方口径", lambda s: None, expect_reject=False))

    print("\n== 注入右填充（应拒绝）==")

    original_from_pretrained = mod.AutoTokenizer.from_pretrained

    def right_padding(service) -> None:
        def patched(model_path, **kwargs):
            kwargs["padding_side"] = "right"
            return original_from_pretrained(model_path, **kwargs)

        mod.AutoTokenizer.from_pretrained = staticmethod(patched)

    try:
        results.append(run_case("右填充", right_padding, expect_reject=True))
    finally:
        mod.AutoTokenizer.from_pretrained = original_from_pretrained

    # 模板漂移**行为上抓不住**：实测丢掉官方 suffix 后方向间隔仍有 0.7988、
    # 批不变性偏差仍为 0。所以这里的预期就是「放行」，它由
    # `tests/m3/test_reranker_scoring.py` 的模板哈希断言在源码层兜住。
    # 记录这个已知盲区，而不是假装自检覆盖了它。
    print("\n== 注入模板篡改：丢掉官方 suffix（行为抓不住，预期放行）==")
    original_suffix = mod._RERANK_SUFFIX
    try:
        mod._RERANK_SUFFIX = ""
        results.append(run_case("空 suffix", lambda s: None, expect_reject=False))
    finally:
        mod._RERANK_SUFFIX = original_suffix

    # 验收标准第 4 条：权重未完整载入必须让启动失败。这里直接注入一个
    # missing_key，检验那道闸门真的会拦——而不是只在「恰好没有 missing」时沉默。
    print("\n== 注入 missing_keys（应拒绝）==")
    gate_message = _probe_missing_keys_gate()
    if gate_message:
        print(f"[拒绝／符合预期] missing_keys 闸门\n        {gate_message}")
    else:
        print("[放行／不符合预期] missing_keys 闸门没拦住")
    results.append(bool(gate_message))

    print(f"\n结论：{sum(results)}/{len(results)} 项符合预期")
    return 0 if all(results) else 1


def _probe_missing_keys_gate() -> str:
    """直接调那道闸门，返回被拒绝时的首行信息；未被拒绝返回空串。

    不必真加载一个残缺 checkpoint：闸门读的就是 transformers 交回的
    `loading_info`，构造该输入即可覆盖判定逻辑。
    """
    try:
        mod.ReorderService._assert_all_weights_loaded(
            {"missing_keys": {"score.weight"}, "mismatched_keys": set()}
        )
    except RuntimeError as exc:
        return str(exc).splitlines()[0]
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
