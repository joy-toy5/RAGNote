"""RAG-008：`answer_path` 作用域标注与 v1 执行摘要的冻结边界。

背景：离线 run 只跑到检索，`predicted_no_answer` 只能来自检索层门禁，所以
`false_answer_rate` 是「门禁单独的漏判率」而不是系统答错率。这组测试锁两件事：

1. 作用域标签确实跟着数走，报告读者不会把两者混为一谈；
2. `rag-note.eval-execution.v1` 的摘要字段集是冻结的 —— 往里加字段会让已冻结
   的 run 在 `__post_init__` 重算校验时全部加载失败，这个后果必须由测试挡住，
   而不是等下一次加字段时再踩一遍。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.contracts import ANSWER_PATHS, ExecutorDescriptor
from app.evaluation.reporting import render_report_markdown

INDEX_VERSION = "6" * 64
CONFIG_SHA = "7" * 64
FROZEN_RUNS = (
    "m3_dev_v2_vector_only",
    "m3_dev_v2_hybrid_cjk",
    "m3_dev_v2_hybrid_cjk_rag008",
)


def _descriptor(**overrides: str) -> ExecutorDescriptor:
    values: dict[str, str] = {
        "executor_id": "rag-note.test.v1",
        "index_version": INDEX_VERSION,
        "retrieval_config_sha256": CONFIG_SHA,
    }
    values.update(overrides)
    return ExecutorDescriptor(**values)


def test_answer_path_defaults_to_retrieval_only() -> None:
    """默认值必须与旧 run 的实际执行方式一致：它们没跑生成层。"""
    assert _descriptor().answer_path == "retrieval_only"


def test_answer_path_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="answer_path"):
        _descriptor(answer_path="maybe")


def test_execution_identity_v1_field_set_is_frozen() -> None:
    """v1 摘要字段集不许增补 —— 加一个就会让三个冻结 run 全部加载失败。

    这条断言的是键集合本身，不是某个哈希值。哈希值会随索引版本变，键集合不该变。
    """
    assert set(_descriptor().execution_identity_v1()) == {
        "executor_id",
        "index_version",
        "retrieval_config_sha256",
    }


def test_execution_identity_v1_ignores_answer_path() -> None:
    """同一次执行换个作用域标签，v1 摘要不动，因此冻结 run 仍然可加载。"""
    retrieval_only = _descriptor(answer_path="retrieval_only")
    generated = _descriptor(answer_path="generated")
    assert (
        retrieval_only.execution_identity_v1() == generated.execution_identity_v1()
    )


def test_answer_path_is_visible_in_serialised_descriptor() -> None:
    """v1 摘要不带它，但 asdict 要带 —— 否则报告元数据里看不见作用域。"""
    from dataclasses import asdict

    assert asdict(_descriptor(answer_path="generated"))["answer_path"] == "generated"


@pytest.mark.parametrize("name", FROZEN_RUNS)
def test_frozen_runs_still_load_after_answer_path_addition(name: str) -> None:
    """回归门禁：加字段前这三个 run 能加载，加字段后必须仍然能加载。

    `EvaluationRun.__post_init__` 会重算 `execution_sha256` 并比对，所以这条
    测试同时验证了摘要口径没被改动。
    """
    from app.evaluation.runner import load_run

    path = Path(__file__).resolve().parents[2] / "evals/runs" / f"{name}.json"
    if not path.exists():
        pytest.skip(f"缺少冻结 run：{path}")
    run = load_run(path)
    assert run.executor.answer_path == "retrieval_only"
    # 冻结 run 的 JSON 里本就没有这个键，加载靠默认值补齐。
    assert "answer_path" not in json.loads(path.read_text(encoding="utf-8"))["executor"]


def test_all_declared_answer_paths_are_accepted() -> None:
    for value in ANSWER_PATHS:
        assert _descriptor(answer_path=value).answer_path == value


def _refusal_rates(outcomes: list[tuple[bool, bool, bool]]) -> dict[str, object]:
    from app.evaluation.reporting import _refusal_rates as compute

    return compute(outcomes)


def test_refusal_rates_split_benefit_from_cost() -> None:
    """(actual_no_answer, predicted_no_answer, run_error) → 分组拒答率。

    2/4 不可回答被拒（收益），1/6 可回答被误拒（代价）。合成一个数看不出这个分野。
    """
    outcomes = [
        (True, True, False),
        (True, True, False),
        (True, False, False),
        (True, False, False),
        (False, True, False),
        *[(False, False, False)] * 5,
    ]
    rates = _refusal_rates(outcomes)
    assert rates["refusal_count_unanswerable"] == 2
    assert rates["refusal_rate_unanswerable"] == 0.5
    assert rates["query_count_unanswerable"] == 4
    assert rates["refusal_count_answerable"] == 1
    assert rates["refusal_rate_answerable"] == pytest.approx(1 / 6)
    assert rates["query_count_answerable"] == 6


def test_refusal_rate_is_none_when_group_is_empty() -> None:
    """空分组的率是未定义，不是 0 —— 与 overlap_count 同一个理由。"""
    rates = _refusal_rates([(True, True, False)])
    assert rates["refusal_rate_answerable"] is None
    assert rates["query_count_answerable"] == 0
    assert rates["refusal_rate_unanswerable"] == 1.0


def test_markdown_warns_when_generation_did_not_run() -> None:
    """retrieval_only 的报告必须自带警告，否则 false_answer_rate 会被错读。"""
    report = _stub_report(answer_path="retrieval_only")
    rendered = render_report_markdown(report)
    assert "no_answer_scope: \"retrieval_only\"" in rendered
    assert "未执行生成层" in rendered
    assert "`refusal_rate_unanswerable`" in rendered


def test_markdown_omits_warning_when_generation_ran() -> None:
    rendered = render_report_markdown(_stub_report(answer_path="generated"))
    assert "no_answer_scope: \"generated\"" in rendered
    assert "未执行生成层" not in rendered


def _stub_report(*, answer_path: str) -> object:
    """只搭渲染需要的最小结构，避免为了两行断言去跑一整轮评测。"""
    from types import SimpleNamespace

    aggregate = {
        key: 0.0
        for key in (
            "recall_at_3",
            "recall_at_5",
            "recall_at_10",
            "recall_at_20",
            "mrr_at_10",
            "ndcg_at_10",
            "precision_at_3",
            "no_answer_precision",
            "no_answer_recall",
            "no_answer_f1",
            "false_answer_rate",
            "refusal_rate_answerable",
            "refusal_rate_unanswerable",
        )
    }
    aggregate["cross_user_hit_count"] = 0
    aggregate["no_answer_scope"] = answer_path
    return SimpleNamespace(
        aggregate=aggregate,
        metadata={
            "dataset_id": "d",
            "dataset_version": "v2",
            "index_version": INDEX_VERSION,
            "source_revision": "abc1234",
            "source_fingerprint": "0" * 64,
        },
        quality_fingerprint="1" * 64,
        hard_gate={"status": "PASS"},
        failures=(),
    )
