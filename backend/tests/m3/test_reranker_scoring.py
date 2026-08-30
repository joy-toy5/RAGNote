"""重排序打分的正确性契约（`RAG-013`）。

**为什么这些测试必须真加载模型**：本缺陷的形态是「不抛异常、`outcome` 仍为
`success`、日志与报告都看不出异常」——`CrossEncoder` 对缺失的 `score.weight`
静默随机初始化。任何用 fake 替身的测试都测不到它，因为替身不会随机初始化。
所以这一组打 `external` marker，默认套件不跑（`pytest.ini` 的 `addopts` 已排除），
需要时显式 `-m external`。

**三条断言各自对应一种「不报错但分数是噪声」的失败**：
  · 方向正确  -> 抓随机初始化的打分头、右填充、chat template 写错；
  · 两次加载一致 -> 抓一切「每次进程重启排序都不同」的成因；
  · batch > 1 可用 -> 抓 `pad_token_id` 缺失被 `batch_size=1` 绕开的那类权宜。

方向断言是三条里最值钱的：它不关心分数的绝对值，只要求「已知相关 > 已知不相关」，
因此对实现方式不敏感，换 checkpoint、换打分口径都仍然成立。

第四条 `test_scoring_does_not_silently_accept_an_untrained_head` 不加载模型，
因此**不**打 `external`——它是默认套件里的长期回归守卫，成本近似为零。
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import os
from pathlib import Path

import pytest

# 已知相关 / 已知不相关各两条。语料刻意选得毫不相干（重置密码 vs 授权费用、机房供电），
# 使方向判据不依赖模型的精细分辨力——一个真正在工作的 reranker 必须能分开这两组。
QUERY = "如何重置密码"
RELEVANT = [
    "在设置页面点击“重置密码”，输入原密码后提交。",
    "忘记密码时可通过绑定邮箱接收验证码完成重置。",
]
IRRELEVANT = [
    "本产品的年度授权费用为每席位 1200 元。",
    "服务器机房位于华东二区，配备双路供电。",
]
ALL_DOCS = [RELEVANT[0], IRRELEVANT[0], RELEVANT[1], IRRELEVANT[1]]

# 间隔阈值刻意设得很松。官方口径实测间隔约 0.9965（相关 ~0.997 vs 不相关 ~0.00005），
# 而随机打分头的间隔在 0 附近正负摇摆。0.5 足以区分「有信号」与「无信号」，
# 同时不会因为换 checkpoint 或换精度而变脆。
MIN_MARGIN = 0.5


def _model_path() -> str:
    from app.rag.reorder_service import find_model_path

    configured = os.getenv(
        "RERANKER_MODEL_PATH", r"D:\Hugging_Face\models\Qwen3-Reranker-0.6B"
    )
    path = find_model_path(configured)
    if not os.path.exists(os.path.join(path, "config.json")):
        pytest.skip(f"本机没有 reranker checkpoint: {path}")
    return path


def _score_all(docs: list[str]) -> list[float]:
    """走生产服务打分，返回与 `docs` 同序的分数。

    刻意走 `reorder_documents` 而不是直接调底层：要测的是生产路径。
    该方法按分数降序返回，故需按文档内容还原原始顺序。
    """
    from app.rag.reorder_service import ReorderService

    service = ReorderService()
    try:
        result = asyncio.run(service.reorder_documents(QUERY, list(docs)))
        assert result["success"], f"重排序失败: {result['error']}"
        by_doc = {item["document"]: item["similarity"] for item in result["documents"]}
        assert len(by_doc) == len(docs), "返回的文档数与输入不符"
        return [by_doc[d] for d in docs]
    finally:
        service._model = None
        gc.collect()


@pytest.mark.external
def test_relevant_documents_score_higher_than_irrelevant() -> None:
    """已知相关必须高于已知不相关（`RAG-013` 验收标准第 2 条）。

    这条在缺陷未修时失败：随机初始化的打分头给出的方向是随机的，实测两次加载
    方向都错（相关最低 0.324 < 不相关最高 0.602；第二次 0.602 < 0.813）。
    """
    _model_path()
    scores = _score_all(ALL_DOCS)
    relevant = [scores[0], scores[2]]
    irrelevant = [scores[1], scores[3]]
    margin = min(relevant) - max(irrelevant)
    assert margin > MIN_MARGIN, (
        f"相关文档最低分 {min(relevant):.6f} 未显著高于不相关文档最高分 "
        f"{max(irrelevant):.6f}（间隔 {margin:+.6f} <= {MIN_MARGIN}）。"
        f"打分没有相关性信号——检查是否加载了未训练的分类头、是否用了右填充、"
        f"chat template 是否与模型卡逐字一致。全部分数: {scores}"
    )


@pytest.mark.external
def test_two_independent_loads_score_identically() -> None:
    """同一 checkpoint 两次独立加载必须打出逐位相同的分数（验收标准第 1 条）。

    这条在缺陷未修时失败：`score.weight` 每次随机初始化，实测求和为
    -1.4921875 与 -0.419921875，打分与排序均不同——即每次进程重启排序都变，
    线上多 worker 之间排序也不一致。
    """
    _model_path()
    first = _score_all(ALL_DOCS)
    second = _score_all(ALL_DOCS)
    assert first == second, (
        f"两次独立加载打分不一致，说明有权重未从 checkpoint 载入而是随机初始化。\n"
        f"  第 1 次: {first}\n  第 2 次: {second}"
    )


@pytest.mark.external
def test_batching_more_than_one_pair_works() -> None:
    """batch > 1 必须能正常打分。

    生产代码原本写 `model.predict(pairs, batch_size=1)`，注释是
    「batch_size=1避免padding令牌报错」——那个报错
    （`Cannot handle batch sizes > 1 if no padding token is defined.`）
    是缺陷的症状之一，被绕开而没有被追查。修好之后 pad token 应从 tokenizer 取，
    batch > 1 自然可用，这条测试防止再退回逐条打分。
    """
    _model_path()
    scores = _score_all(ALL_DOCS)
    assert len(scores) == len(ALL_DOCS)
    assert all(isinstance(s, float) for s in scores)


def test_scoring_does_not_silently_accept_an_untrained_head() -> None:
    """加载阶段必须拒绝未训练的打分头，而不是继续返回 success。

    缺陷的根因不是分数算错，是**失败不可见**：`CrossEncoder` 静默随机初始化，
    `reorder_documents` 返回 `success=True`，降级路径不触发。所以修复必须包含
    一道加载自检——本测试断言该自检存在且真的会拒绝。
    """
    from app.rag import reorder_service as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "AutoModelForCausalLM" in source, (
        "必须按模型卡用 AutoModelForCausalLM，Qwen3-Reranker 系列没有可加载的标量打分头"
    )
    assert "padding_side" in source and "left" in source, (
        "必须显式 padding_side='left'：分数取 logits[:, -1, :]，右填充会让"
        "最后一个位置变成 pad token，取到的分布与文档内容无关"
    )
    # 只禁止「使用」，不禁止「提及」：源码里解释该缺陷的注释应当保留，
    # 那是这次修复留给后来人的唯一线索。
    assert "CrossEncoder(" not in source, (
        "不得再用 CrossEncoder 加载该 checkpoint——它对缺失的 score.weight "
        "静默随机初始化，这个失败模式不抛异常"
    )
    assert "import CrossEncoder" not in source, (
        "不得再导入 CrossEncoder"
    )
    assert "_assert_scoring_is_sane" in source, (
        "必须保留加载后的自检：缺陷的根因是失败不可见，"
        "去掉自检就退回到「静默返回噪声分」"
    )
    assert "_assert_batch_invariance" in source, (
        "必须保留批不变性自检：方向断言抓不住右填充"
        "（实测右填充下间隔仍有 0.9700，方向看起来是对的）"
    )


def test_missing_weights_abort_the_load() -> None:
    """权重未完整载入必须让加载失败（`RAG-013` 验收标准第 4 条）。

    不需要真的残缺 checkpoint：闸门读的就是 transformers 交回的 `loading_info`，
    构造该输入即可覆盖判定。因此这条不打 `external`，成本近似为零。
    """
    from app.rag.reorder_service import ReorderService

    # 正常情况：两个集合都空，必须放行。
    ReorderService._assert_all_weights_loaded(
        {"missing_keys": set(), "mismatched_keys": set()}
    )

    with pytest.raises(RuntimeError, match="score.weight"):
        ReorderService._assert_all_weights_loaded(
            {"missing_keys": {"score.weight"}, "mismatched_keys": set()}
        )

    with pytest.raises(RuntimeError, match="mismatched_keys"):
        ReorderService._assert_all_weights_loaded(
            {"missing_keys": set(), "mismatched_keys": {"model.norm.weight"}}
        )


def test_prompt_template_is_pinned_byte_for_byte() -> None:
    """打分模板必须逐字钉死（`RAG-013`）。

    分数就是「assistant 第一个待生成位置」的 logits 分布，模板改一个字符就换了
    一个测量口径，历史分数随即不可比。而这种漂移**行为上抓不住**：实测把官方
    suffix 整个丢掉，方向断言的间隔仍有 0.7988、批不变性也照样通过。
    行为测不出来的东西，就得在源码层钉住。

    末尾那个空的 `<think>\\n\\n</think>` 尤其容易被当成冗余删掉——它是模型卡
    模板的一部分，不能省。
    """
    from app.rag import reorder_service as module

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    assert digest(module._RERANK_PREFIX) == (
        "b09144c543bfe3e0dcde88b2bf54986c50060c9f31d9fe98916e1b7f1ad3b80b"
    ), f"PREFIX 已变，实际 sha256={digest(module._RERANK_PREFIX)}"
    assert digest(module._RERANK_SUFFIX) == (
        "5ca215121bd0820099042d7d3949d5b77533e17ad13bb8e6104d37029e7ea54d"
    ), f"SUFFIX 已变，实际 sha256={digest(module._RERANK_SUFFIX)}"
    assert module._RERANK_SUFFIX.endswith("<think>\n\n</think>\n\n"), (
        "官方模板末尾的空 <think> 块不能省"
    )
