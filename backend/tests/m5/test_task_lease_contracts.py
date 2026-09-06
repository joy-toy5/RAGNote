"""租约纯合同与首个 await 前的输入拒绝；不加载生产配置或数据库。"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, fields
import os
import subprocess
import sys
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.tasking import leases
from app.tasking.errors import TaskLeaseLost, TaskStateConflict
from app.tasking.lease_contracts import RetryPolicy, TaskLease

TASK_ID = "11111111-1111-4111-8111-111111111111"
OWNER = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
KINDS = ("knowledge.index", "note.index", "note.delete")
MUTATIONS = (
    "heartbeat_task",
    "succeed_task",
    "fail_task",
    "acknowledge_cancellation",
)


def _lease(**changes):
    return TaskLease(
        **dict(task_id=TASK_ID, owner=OWNER, token=1, attempt_no=1) | changes
    )


def _call(operation, session, **changes):
    arguments = {
        "claim_task": {"owner": OWNER, "kinds": KINDS},
        "heartbeat_task": {"lease": _lease()},
        "succeed_task": {"lease": _lease(), "result_ref": "snapshot://result/1"},
        "fail_task": {
            "lease": _lease(),
            "error_code": "INDEX_FAILED",
            "error_summary": "合成失败摘要",
        },
        "acknowledge_cancellation": {"lease": _lease()},
        "list_expired_leases": {"kinds": KINDS},
    }
    return getattr(leases, operation)(session, **(arguments[operation] | changes))


def _assert_rejected_before_await(operation, **changes):
    session = AsyncMock(spec=AsyncSession)
    coroutine = _call(operation, session, **changes)
    try:
        # 只推进到第一个挂起点；不能先等待数据库，再以输入错误掩盖 IO。
        with pytest.raises((TypeError, ValueError)):
            coroutine.send(None)
    finally:
        coroutine.close()
    assert session.mock_calls == []


def test_lease_is_a_frozen_slotted_value_and_lost_is_a_state_conflict():
    lease = _lease()
    assert tuple(field.name for field in fields(lease)) == (
        "task_id",
        "owner",
        "token",
        "attempt_no",
    )
    assert lease == _lease() and hash(lease) == hash(_lease())
    assert not hasattr(lease, "__dict__")
    with pytest.raises(FrozenInstanceError):
        lease.token = 2
    assert issubclass(TaskLeaseLost, TaskStateConflict)


@pytest.mark.parametrize(
    "value",
    [
        "not-a-uuid",
        "00000000-0000-0000-0000-000000000000",
        OWNER.upper(),
        OWNER.replace("-", ""),
        "{" + OWNER + "}",
        "aaaaaaaa-aaaa-4aaa-7aaa-aaaaaaaaaaaa",
        UUID(OWNER),
        None,
    ],
)
def test_lease_ids_are_canonical_nonzero_rfc4122_strings(value):
    for name in ("task_id", "owner"):
        with pytest.raises((TypeError, ValueError)):
            _lease(**{name: value})


@pytest.mark.parametrize(
    "field,maximum", [("token", 2**64 - 1), ("attempt_no", 2**32 - 1)]
)
def test_lease_counters_have_exact_unsigned_bounds_without_boolean_coercion(
    field, maximum
):
    for value in (1, maximum):
        assert getattr(_lease(**{field: value}), field) == value
    for value in (0, -1, maximum + 1, True, False, 1.0, "1"):
        with pytest.raises((TypeError, ValueError)):
            _lease(**{field: value})


def test_retry_policy_defaults_and_non_power_of_two_cap():
    default = RetryPolicy()
    assert (default.base_seconds, default.max_seconds) == (5, 300)
    assert [default.delay_seconds(n) for n in range(1, 9)] == [
        5,
        10,
        20,
        40,
        80,
        160,
        300,
        300,
    ]
    assert [RetryPolicy(7, 20).delay_seconds(n) for n in range(1, 5)] == [7, 14, 20, 20]
    assert RetryPolicy(86400, 86400).delay_seconds(1) == 86400
    assert RetryPolicy(1, 1).delay_seconds(2) == 1


@pytest.mark.parametrize("field", ["base_seconds", "max_seconds"])
def test_retry_policy_rejects_nonpositive_noninteger_and_overlarge_seconds(field):
    for value in (0, -1, 86401, True, 1.0, "5"):
        with pytest.raises((TypeError, ValueError)):
            RetryPolicy(**{field: value})
    with pytest.raises(ValueError):
        RetryPolicy(base_seconds=10, max_seconds=9)


def test_retry_attempt_number_rejects_out_of_contract_values():
    for value in (0, -1, 2**32, True, 1.0, "1"):
        with pytest.raises((TypeError, ValueError)):
            RetryPolicy().delay_seconds(value)


def test_maximum_attempt_backoff_is_bounded_in_time_and_memory(tmp_path, backend_root):
    # 错误实现可能先计算 2**4294967294；在子进程限额内暴露回归，不能拖垮 pytest。
    code = """
import resource
from app.tasking.lease_contracts import RetryPolicy
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
assert RetryPolicy().delay_seconds(2**32 - 1) == 300
assert RetryPolicy(1, 86400).delay_seconds(2**32 - 1) == 86400
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(backend_root),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "kinds",
    [(), ["note.index"], "note.index", ("note.index", "note.index"), ("other",), (1,)],
)
def test_claim_and_expiry_scan_reject_invalid_kinds_before_await(kinds):
    for operation in ("claim_task", "list_expired_leases"):
        _assert_rejected_before_await(operation, kinds=kinds)


@pytest.mark.parametrize("value", [0, 3601, True, 90.0])
def test_claim_and_heartbeat_validate_lease_duration_before_await(value):
    for operation in ("claim_task", "heartbeat_task"):
        _assert_rejected_before_await(operation, lease_seconds=value)


@pytest.mark.parametrize("operation", MUTATIONS)
def test_mutations_require_a_lease_before_await(operation):
    _assert_rejected_before_await(operation, lease=None)


@pytest.mark.parametrize(
    "operation,changes",
    [
        ("claim_task", {"owner": OWNER.upper()}),
        ("claim_task", {"owner": UUID(OWNER)}),
        ("heartbeat_task", {"phase": ""}),
        ("heartbeat_task", {"phase": "a" * 65}),
        ("heartbeat_task", {"phase": "has space"}),
        ("heartbeat_task", {"phase": "index\x7f"}),
        ("heartbeat_task", {"phase": "索引"}),
        ("heartbeat_task", {"progress": -1}),
        ("heartbeat_task", {"progress": 101}),
        ("heartbeat_task", {"progress": True}),
        ("heartbeat_task", {"progress": 1.0}),
        ("succeed_task", {"result_ref": " \t"}),
        ("succeed_task", {"result_ref": "x" * 1025}),
        ("succeed_task", {"result_ref": "\ud800"}),
        ("succeed_task", {"result_ref": None}),
        ("succeed_task", {"result_version": 0}),
        ("succeed_task", {"result_version": 2**32}),
        ("succeed_task", {"result_version": True}),
        ("succeed_task", {"result_summary": ""}),
        ("succeed_task", {"result_summary": "中" * 1025}),
        ("succeed_task", {"result_summary": "\ud800"}),
        ("fail_task", {"error_code": ""}),
        ("fail_task", {"error_code": "E" * 65}),
        ("fail_task", {"error_code": "BAD CODE"}),
        ("fail_task", {"error_code": "ERR\n"}),
        ("fail_task", {"error_summary": " \n"}),
        ("fail_task", {"error_summary": "x" * 1025}),
        ("fail_task", {"error_summary": "\ud800"}),
        ("fail_task", {"error_summary": None}),
        ("fail_task", {"retryable": 1}),
        ("fail_task", {"retry_policy": {"base_seconds": 5}}),
        ("list_expired_leases", {"limit": 0}),
        ("list_expired_leases", {"limit": 101}),
        ("list_expired_leases", {"limit": True}),
        ("list_expired_leases", {"limit": 1.0}),
    ],
)
def test_invalid_operation_parameters_are_rejected_before_first_await(
    operation, changes
):
    _assert_rejected_before_await(operation, **changes)


@pytest.mark.parametrize(
    "operation,changes",
    [
        ("claim_task", {"lease_seconds": 1, "kinds": KINDS}),
        ("heartbeat_task", {"lease_seconds": 3600, "phase": "~" * 64, "progress": 100}),
        ("heartbeat_task", {"lease_seconds": 1, "phase": "!", "progress": 0}),
        ("succeed_task", {"result_ref": "中" * 1024, "result_version": 2**32 - 1}),
        ("succeed_task", {"result_summary": "中" * 1024}),
        (
            "fail_task",
            {"error_code": "!" + "E" * 62 + "~", "error_summary": "中" * 1024},
        ),
        ("list_expired_leases", {"limit": 1}),
        ("list_expired_leases", {"limit": 100}),
    ],
)
def test_valid_storage_boundaries_reach_io_without_trimming(
    monkeypatch, operation, changes
):
    class ReachedDatabase(RuntimeError):
        pass

    session = AsyncMock(spec=AsyncSession)
    session.scalar.side_effect = ReachedDatabase
    clock = AsyncMock(side_effect=ReachedDatabase)
    monkeypatch.setattr(leases, "_database_now", clock)
    with pytest.raises(ReachedDatabase):
        asyncio.run(_call(operation, session, **changes))
    assert clock.await_count + session.scalar.await_count == 1
    session.commit.assert_not_called()
    session.rollback.assert_not_called()
