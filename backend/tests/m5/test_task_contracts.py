"""任务输入在数据库等待前校验，JSON 不隐式转换键、容器或数值类型。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.tasking import contracts, repository
from app.tasking.contracts import TaskSubmission


def _submission(**overrides):
    values = dict(
        user_id="user-a",
        kind="note.index",
        idempotency_key="note:revision:1",
        input_fingerprint="a" * 64,
        input_ref="snapshot://notes/1/1",
    )
    values.update(overrides)
    return TaskSubmission(**values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("user_id", ""),
        ("user_id", "user a"),
        ("user_id", "a" * 65),
        ("user_id", "用户"),
        ("user_id", "u\x7f"),
        ("user_id", None),
        ("idempotency_key", ""),
        ("idempotency_key", "a" * 129),
        ("idempotency_key", "key with space"),
        ("idempotency_key", "key\n"),
        ("idempotency_key", 1),
        ("kind", "arbitrary.execute"),
        ("input_fingerprint", ""),
        ("input_fingerprint", "A" * 64),
        ("input_fingerprint", "a" * 63),
        ("input_fingerprint", "g" * 64),
        ("input_fingerprint", None),
        ("input_blob_id", "blob-id"),
        ("input_blob_id", "A" * 64),
        ("resource_id", "not-a-uuid"),
        ("resource_id", "00000000-0000-0000-0000-000000000000"),
        ("resource_id", "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"),
        ("resource_id", "11111111111141118111111111111111"),
        ("retry_of_task_id", "not-a-uuid"),
        ("target_generation", 0),
        ("target_generation", -1),
        ("target_generation", 4_294_967_296),
        ("target_generation", True),
        ("target_generation", 1.5),
        ("input_schema_version", 0),
        ("input_schema_version", True),
        ("input_schema_version", 4_294_967_296),
        ("max_attempts", 0),
        ("max_attempts", -1),
        ("max_attempts", True),
        ("max_attempts", "3"),
        ("max_attempts", 4_294_967_296),
        ("input_ref", ""),
        ("input_ref", " \t"),
        ("input_ref", "中" * 1025),
        ("input_ref", "\ud800"),
        ("input_ref", None),
        ("input_metadata", []),
        ("input_metadata", {1: "silent-key-conversion"}),
        ("input_metadata", {"nested": {1: "invalid"}}),
        ("input_metadata", {"tuple": (1, 2)}),
        ("input_metadata", {"nan": float("nan")}),
        ("input_metadata", {"infinity": float("inf")}),
        ("input_metadata", {"negative_infinity": -float("inf")}),
        ("input_metadata", {"set": {1, 2}}),
        ("input_metadata", {"bytes": b"a"}),
        ("input_metadata", {"surrogate": "\ud800"}),
    ],
)
def test_invalid_submission_does_not_touch_session(field, value):
    session = AsyncMock(spec=AsyncSession)
    with pytest.raises((TypeError, ValueError)):
        asyncio.run(
            repository.create_or_get_task(session, _submission(**{field: value}))
        )
    assert session.mock_calls == []


def test_circular_json_is_rejected_before_io():
    metadata = {}
    metadata["self"] = metadata
    session = AsyncMock(spec=AsyncSession)
    with pytest.raises(ValueError, match="[Cc]ircular|循环"):
        asyncio.run(
            repository.create_or_get_task(session, _submission(input_metadata=metadata))
        )
    assert session.mock_calls == []


@pytest.mark.parametrize(
    "metadata",
    [
        {"n": 2**64},
        {"n": -(2**63) - 1},
        {"nested": [{"n": 2**64 + 1}]},
        {"nested": {"values": [-(2**63) - 2]}},
    ],
)
def test_metadata_rejects_lossy_mysql_integers_before_io(metadata):
    session = AsyncMock(spec=AsyncSession)
    session.flush.side_effect = AssertionError("应在数据库访问前拒绝越界 JSON 整数")
    with pytest.raises(ValueError, match="metadata.*整数"):
        asyncio.run(
            repository.create_or_get_task(session, _submission(input_metadata=metadata))
        )
    assert session.mock_calls == []


@pytest.mark.parametrize("value", [-(2**63), 2**63 - 1, 2**63, 2**64 - 1, True, False])
def test_metadata_preserves_mysql_integer_boundaries_and_booleans(value):
    snapshot = contracts.snapshot_submission(
        _submission(input_metadata={"nested": [{"n": value}]})
    )
    actual = snapshot.input_metadata["nested"][0]["n"]
    assert actual == value
    assert type(actual) is type(value)


def test_snapshot_accepts_storage_boundaries_without_changing_business_text():
    submission = _submission(
        user_id="u" * 64,
        idempotency_key="!" + "k" * 126 + "~",
        input_ref="中" * 1024,
        resource_id="11111111-1111-4111-8111-111111111111",
        input_blob_id="b" * 64,
        target_generation=4_294_967_295,
        input_schema_version=4_294_967_295,
        max_attempts=4_294_967_295,
        input_metadata={"nested": [None, True, 1, 1.0, "e\u0301", {"标签": "值"}]},
    )
    snapshot = contracts.snapshot_submission(submission)
    assert snapshot == submission and snapshot is not submission
    assert snapshot.input_metadata is not submission.input_metadata
    assert snapshot.input_metadata["nested"] is not submission.input_metadata["nested"]
    assert snapshot.input_metadata["nested"][4] == "e\u0301"
    assert (
        contracts.snapshot_submission(replace(submission, input_ref="")).input_ref == ""
    )


@pytest.mark.parametrize("metadata", [None, {}, {"b": [True, 1], "a": "值"}])
def test_canonical_metadata_is_order_stable_but_preserves_json_types(metadata):
    reversed_metadata = dict(reversed(list(metadata.items()))) if metadata else metadata
    assert contracts.canonical_task_metadata(
        metadata
    ) == contracts.canonical_task_metadata(reversed_metadata)
    assert contracts.canonical_task_metadata(
        {"v": True}
    ) != contracts.canonical_task_metadata({"v": 1})
    assert contracts.canonical_task_metadata(
        {"v": 1}
    ) != contracts.canonical_task_metadata({"v": 1.0})
