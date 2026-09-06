"""P2-A 离线租约状态机：真实 SQLite 事务，不冒充 MySQL 并发行锁验收。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import DateTime, delete, event, literal, select, text, update
from sqlalchemy.dialects import mysql, sqlite
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.indexing.models import ContentBlob
from app.tasking import leases, repository
from app.tasking.errors import TaskLeaseLost, TaskStateConflict
from app.tasking.lease_contracts import RetryPolicy, TaskLease
from app.tasking.models import BackgroundTask, TaskAttempt

NOW = datetime(2026, 9, 6, 1, 0)
OWNER_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
OWNER_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
KINDS = ("knowledge.index", "note.index", "note.delete")
MUTATIONS = ("heartbeat", "succeed", "fail", "cancel")


def _task_id(index):
    return f"{index:08x}-1111-4111-8111-111111111111"


def _task(index=1, **changes):
    return BackgroundTask(
        **(
            dict(
                task_id=_task_id(index),
                user_id="lease-test-user",
                kind="note.index",
                idempotency_key=f"lease:{index}",
                input_fingerprint="a" * 64,
                input_ref=f"snapshot://lease/{index}",
                input_metadata={"options": ["合成输入", 1, True]},
                input_schema_version=1,
                status="pending",
                phase="queued",
                progress=0,
                attempt_count=0,
                max_attempts=3,
                lease_token=0,
                created_at=NOW,
                updated_at=NOW,
            )
            | changes
        )
    )


@asynccontextmanager
async def _database(tmp_path, backend_root):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'leases.sqlite3'}")

    @event.listens_for(engine.sync_engine, "connect")
    def configure_sqlite(connection, _):
        connection.isolation_level = None
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    @event.listens_for(engine.sync_engine, "begin")
    def begin_sqlite(connection):
        # legacy SAVEPOINT 不能自行提交；所有变更必须受外层事务控制。
        connection.exec_driver_sql("BEGIN")

    def migrate(connection):
        config = Config()
        config.set_main_option("script_location", str(backend_root / "alembic"))
        config.attributes["connection"] = connection
        command.upgrade(config, "head")

    try:
        async with engine.begin() as connection:
            assert await connection.scalar(text("PRAGMA foreign_keys")) == 1
            await connection.run_sync(migrate)
        yield async_sessionmaker(engine, expire_on_commit=False), engine
    finally:
        await engine.dispose()


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=NOW)

    async def database_now(_session):
        return state.now

    monkeypatch.setattr(leases, "_database_now", database_now)
    monkeypatch.setattr(
        leases,
        "_clock_expression",
        lambda _session: literal(state.now, type_=DateTime()),
    )
    return state


@contextmanager
def _caller_transaction(session):
    with pytest.MonkeyPatch.context() as patch:
        for name in ("commit", "rollback"):
            patch.setattr(
                session, name, AsyncMock(side_effect=AssertionError("事务由调用方结束"))
            )
        yield
        session.commit.assert_not_called()
        session.rollback.assert_not_called()


async def _claim(session, *, owner=OWNER_A, kinds=KINDS, **changes):
    with _caller_transaction(session):
        return await leases.claim_task(session, owner=owner, kinds=kinds, **changes)


async def _running(session, **changes):
    task = _task(**changes)
    session.add(task)
    await session.commit()
    lease = await _claim(session)
    await session.commit()
    return task, lease


async def _mutate(session, operation, lease, **changes):
    functions = {
        "heartbeat": (leases.heartbeat_task, {}),
        "succeed": (leases.succeed_task, {"result_ref": "snapshot://result/1"}),
        "fail": (
            leases.fail_task,
            {"error_code": "INDEX_FAILED", "error_summary": "合成失败"},
        ),
        "cancel": (leases.acknowledge_cancellation, {}),
    }
    function, defaults = functions[operation]
    with _caller_transaction(session):
        return await function(session, lease, **(defaults | changes))


def _row(row):
    return {
        column.key: deepcopy(getattr(row, column.key))
        for column in row.__table__.columns
    }


async def _snapshot(session):
    tasks = list(
        await session.scalars(select(BackgroundTask).order_by(BackgroundTask.task_id))
    )
    attempts = list(
        await session.scalars(
            select(TaskAttempt).order_by(TaskAttempt.task_id, TaskAttempt.attempt_no)
        )
    )
    return [_row(task) for task in tasks], [_row(attempt) for attempt in attempts]


def _capture_selects(session):
    statements = []

    @event.listens_for(session.sync_session, "do_orm_execute")
    def capture(execution):
        if execution.is_select:
            statements.append((execution.statement, dict(execution.execution_options)))

    return statements


def _mysql_sql(statement):
    return " ".join(str(statement.compile(dialect=mysql.dialect())).split())


def _assert_exclusive_refresh(statements, table):
    matches = [
        (query, options)
        for query, options in statements
        if f"FROM {table}" in _mysql_sql(query)
    ]
    assert matches, f"未锁定 {table}"
    # 首次读取必须独占刷新；UPDATE 后的显式当前读由单独用例验证。
    for query, options in matches[:1]:
        assert _mysql_sql(query).endswith("FOR UPDATE")
        assert options.get("populate_existing") is True
    return matches[:1]


def test_claim_filters_eligibility_and_orders_by_creation_then_id(
    tmp_path, backend_root, clock
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            blocked = [
                _task(10, next_run_at=NOW + timedelta(seconds=1)),
                _task(11, status="retry_wait", next_run_at=NOW + timedelta(seconds=1)),
                _task(12, cancel_requested_at=NOW),
                _task(13, status="retry_wait", cancel_requested_at=NOW),
                _task(14, attempt_count=3, max_attempts=3),
                _task(15, status="processing", lease_until=NOW - timedelta(seconds=1)),
                _task(16, status="processing", lease_until=NOW + timedelta(seconds=30)),
                *[
                    _task(20 + index, status=status)
                    for index, status in enumerate(
                        ("succeeded", "failed", "cancelled", "superseded")
                    )
                ],
                _task(30, kind="note.delete"),
            ]
            eligible = [
                _task(3, created_at=NOW - timedelta(seconds=1)),
                _task(2, status="retry_wait", next_run_at=NOW),
                _task(1, next_run_at=NOW - timedelta(seconds=1)),
            ]
            async with sessions() as session:
                session.add_all(blocked + eligible)
                await session.commit()
                before = {task.task_id: _row(task) for task in blocked}
                receipts = [
                    await _claim(session, kinds=("note.index",)) for _ in eligible
                ]
                assert [lease.task_id for lease in receipts] == [
                    _task_id(3),
                    _task_id(1),
                    _task_id(2),
                ]
                assert await _claim(session, kinds=("note.index",)) is None
                assert {task.task_id: _row(task) for task in blocked} == before
                attempts = list(await session.scalars(select(TaskAttempt)))
                assert len(attempts) == 3 and all(
                    item.status == "running" for item in attempts
                )

    asyncio.run(scenario())


def test_claim_supports_each_kind_without_consuming_unsupported_work(
    tmp_path, backend_root, clock
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                session.add_all(
                    [_task(index, kind=kind) for index, kind in enumerate(KINDS, 1)]
                )
                await session.commit()
                for index, kind in enumerate(KINDS, 1):
                    lease = await _claim(session, kinds=(kind,))
                    assert lease.task_id == _task_id(index)
                    assert await _claim(session, kinds=(kind,)) is None

    asyncio.run(scenario())


def test_claim_and_attempt_are_atomic_with_outer_commit_or_rollback(
    tmp_path, backend_root, clock
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                task = _task(lease_token=7)
                session.add(task)
                await session.commit()
                original = _row(task)
                session.add(
                    ContentBlob(
                        blob_id="b" * 64, byte_size=1, storage_uri="blob://synthetic"
                    )
                )
                lease = await _claim(session)
                assert lease == TaskLease(_task_id(1), OWNER_A, 8, 1)
                attempt = await session.get(TaskAttempt, (_task_id(1), 1))
                assert (attempt.lease_owner, attempt.lease_token, attempt.status) == (
                    OWNER_A,
                    8,
                    "running",
                )
                assert task.started_at == task.heartbeat_at == attempt.started_at == NOW
                assert task.lease_until == NOW + timedelta(seconds=90)
                assert task.next_run_at is task.completed_at is None
                assert session.in_transaction()
                await session.rollback()
            async with sessions() as check:
                assert _row(await check.get(BackgroundTask, _task_id(1))) == original
                assert await check.get(TaskAttempt, (_task_id(1), 1)) is None
                assert await check.get(ContentBlob, "b" * 64) is None
                assert await _claim(check) == lease
                await check.commit()
            async with sessions() as check:
                task = await check.get(BackgroundTask, _task_id(1))
                assert (task.status, task.lease_token, task.attempt_count) == (
                    "processing",
                    8,
                    1,
                )
                assert await check.get(TaskAttempt, (_task_id(1), 1)) is not None

    asyncio.run(scenario())


def test_claim_uses_database_time_again_after_exclusive_refresh(
    tmp_path, backend_root, monkeypatch
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                session.add(_task())
                await session.commit()
                statements = _capture_selects(session)
                readings = []

                async def database_now(actual_session):
                    assert actual_session is session
                    readings.append(len(statements))
                    return NOW if len(readings) == 1 else NOW + timedelta(seconds=7)

                monkeypatch.setattr(leases, "_database_now", database_now)
                monkeypatch.setattr(
                    leases,
                    "_clock_expression",
                    lambda _session: literal(
                        NOW + timedelta(seconds=7), type_=DateTime()
                    ),
                )
                lease = await _claim(session)
                assert readings == [0, 1]
                query, _ = _assert_exclusive_refresh(statements, "background_tasks")[0]
                assert (
                    "ORDER BY background_tasks.created_at, background_tasks.task_id"
                    in _mysql_sql(query)
                )
                task = await session.get(BackgroundTask, lease.task_id)
                attempt = await session.get(
                    TaskAttempt, (lease.task_id, lease.attempt_no)
                )
                after_lock = NOW + timedelta(seconds=7)
                assert (
                    task.started_at
                    == task.heartbeat_at
                    == task.updated_at
                    == after_lock
                )
                assert attempt.started_at == after_lock
                assert task.lease_until == after_lock + timedelta(seconds=90)

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", MUTATIONS)
def test_every_mutation_rejects_forged_or_stale_receipt_without_writes(
    tmp_path, backend_root, clock, operation
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                task, lease = await _running(session)
                task.cancel_requested_at = NOW
                await session.commit()
                before = await _snapshot(session)
                for changes in (
                    {"task_id": _task_id(99)},
                    {"owner": OWNER_B},
                    {"token": 2},
                    {"attempt_no": 2},
                ):
                    with pytest.raises(TaskLeaseLost):
                        await _mutate(session, operation, replace(lease, **changes))
                    assert not session.dirty and not session.new
                    assert await _snapshot(session) == before
                assert session.is_active

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", MUTATIONS)
def test_mutations_lock_task_then_attempt_before_expiry_recheck(
    tmp_path, backend_root, clock, monkeypatch, operation
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                task, lease = await _running(session)
                task.cancel_requested_at = NOW
                await session.commit()
                before = await _snapshot(session)
                statements = _capture_selects(session)
                readings = []

                async def after_wait(actual_session):
                    assert actual_session is session
                    task_queries = _assert_exclusive_refresh(
                        statements, "background_tasks"
                    )
                    attempt_queries = _assert_exclusive_refresh(
                        statements, "task_attempts"
                    )
                    assert statements.index(task_queries[0]) < statements.index(
                        attempt_queries[0]
                    )
                    for query, _ in (task_queries[0], attempt_queries[0]):
                        where_sql = _mysql_sql(query).split(" WHERE ", 1)[1]
                        for column in (
                            "task_id",
                            "lease_owner",
                            "lease_token",
                            "status",
                        ):
                            assert column in where_sql
                    readings.append(True)
                    # 入场时仍有效，行锁等待后恰好到期；等号也必须 fencing。
                    return NOW + timedelta(seconds=90)

                monkeypatch.setattr(leases, "_database_now", after_wait)
                with pytest.raises(TaskLeaseLost):
                    await _mutate(session, operation, lease)
                assert readings == [True]
                assert not session.dirty and not session.new
                assert await _snapshot(session) == before

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changes",
    [
        {"lease_owner": OWNER_B},
        {"lease_token": 2},
        {"attempt_count": 2},
        {"lease_until": NOW},
        {"lease_until": None},
        *[
            {"status": status}
            for status in (
                "pending",
                "retry_wait",
                "succeeded",
                "failed",
                "superseded",
                "cancelled",
            )
        ],
    ],
)
def test_committed_external_changes_fence_cached_receipts(
    tmp_path, backend_root, clock, changes
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as stale:
                task, lease = await _running(stale)
                task.cancel_requested_at = NOW
                attempt = await stale.get(
                    TaskAttempt, (lease.task_id, lease.attempt_no)
                )
                # 取消请求的 ORM UPDATE 会使 server onupdate 字段过期；先异步载入，
                # 再结束事务，后续同步快照不能触发隐式数据库 IO。
                await stale.refresh(task)
                await stale.commit()
                cached = _row(task), _row(attempt)
                # 合成外部已提交状态；不把此直接 UPDATE 当作安全重领实现。
                async with sessions.begin() as writer:
                    await writer.execute(
                        update(BackgroundTask)
                        .where(BackgroundTask.task_id == lease.task_id)
                        .values(**changes)
                    )
                assert (_row(task), _row(attempt)) == cached
                async with sessions() as check:
                    persisted = await _snapshot(check)
                for operation in MUTATIONS:
                    with pytest.raises(TaskLeaseLost):
                        await _mutate(stale, operation, lease)
                    assert not stale.dirty and not stale.new
                await stale.commit()
            async with sessions() as check:
                assert await _snapshot(check) == persisted

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "corruption", ["missing", "owner", "token", "number", "status"]
)
def test_running_attempt_inconsistency_is_not_lease_loss_and_never_mutates_history(
    tmp_path, backend_root, clock, corruption
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as stale:
                task, lease = await _running(stale)
                task.cancel_requested_at = NOW
                cached_attempt = await stale.get(TaskAttempt, (lease.task_id, 1))
                await stale.commit()
                assert cached_attempt.status == "running"
                async with sessions.begin() as writer:
                    if corruption == "missing":
                        await writer.execute(
                            delete(TaskAttempt).where(
                                TaskAttempt.task_id == lease.task_id
                            )
                        )
                    else:
                        changes = {
                            "owner": {"lease_owner": OWNER_B},
                            "token": {"lease_token": 2},
                            "number": {"attempt_no": 2},
                            "status": {"status": "succeeded"},
                        }[corruption]
                        await writer.execute(
                            update(TaskAttempt)
                            .where(TaskAttempt.task_id == lease.task_id)
                            .values(**changes)
                        )
                async with sessions() as check:
                    persisted = await _snapshot(check)
                for operation in MUTATIONS:
                    with pytest.raises(TaskStateConflict) as raised:
                        await _mutate(stale, operation, lease)
                    assert not isinstance(raised.value, TaskLeaseLost)
                    assert not stale.dirty and not stale.new
                await stale.commit()
            async with sessions() as check:
                assert await _snapshot(check) == persisted

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", MUTATIONS)
def test_all_mutations_leave_commit_and_rollback_to_the_caller(
    tmp_path, backend_root, clock, operation
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                task, lease = await _running(session)
                if operation == "cancel":
                    # 本 fixture 没有启动执行器；确认取消不冒充线程停止证明。
                    task.cancel_requested_at = NOW
                await session.commit()
                original = await _snapshot(session)
                session.add(
                    ContentBlob(
                        blob_id="b" * 64, byte_size=1, storage_uri="blob://synthetic"
                    )
                )
                clock.now = NOW + timedelta(seconds=1)
                arguments = {"retryable": True} if operation == "fail" else {}
                await _mutate(session, operation, lease, **arguments)
                assert session.in_transaction()
                assert await _snapshot(session) != original
                await session.rollback()
            async with sessions() as check:
                assert await _snapshot(check) == original
                assert await check.get(ContentBlob, "b" * 64) is None

    asyncio.run(scenario())


def test_heartbeat_updates_progress_without_shortening_or_resetting_the_lease(
    tmp_path, backend_root, clock
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                task, lease = await _running(session)
                clock.now = NOW + timedelta(seconds=10)
                assert (
                    await _mutate(
                        session,
                        "heartbeat",
                        lease,
                        lease_seconds=1,
                        phase="indexing",
                        progress=0,
                    )
                    is False
                )
                assert task.lease_until == NOW + timedelta(seconds=90)
                assert task.heartbeat_at == task.updated_at == clock.now
                attempt = await session.get(TaskAttempt, (lease.task_id, 1))
                assert task.phase == attempt.phase == "indexing"
                await session.commit()
                clock.now += timedelta(seconds=10)
                assert await _mutate(session, "heartbeat", lease) is False
                assert task.lease_until == clock.now + timedelta(seconds=90)
                assert task.heartbeat_at == task.updated_at == clock.now
                assert task.phase == attempt.phase == "indexing" and task.progress == 0
                assert task.status == "processing" and attempt.status == "running"
                assert task.started_at == attempt.started_at == NOW
                assert (
                    task.attempt_count == lease.attempt_no
                    and task.lease_token == lease.token
                )
                assert task.completed_at is attempt.finished_at is None

    asyncio.run(scenario())


def test_success_preserves_fencing_heartbeat_and_result_text(
    tmp_path, backend_root, clock
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                task, lease = await _running(session, lease_token=9)
                clock.now += timedelta(seconds=1)
                await _mutate(
                    session, "heartbeat", lease, phase="publishing", progress=75
                )
                await session.commit()
                heartbeat = task.heartbeat_at
                clock.now += timedelta(seconds=1)
                result = await _mutate(
                    session,
                    "succeed",
                    lease,
                    result_ref="snapshot://result/合成",
                    result_version=2**32 - 1,
                    result_summary=" 合成结果摘要 ",
                )
                assert isinstance(result, BackgroundTask)
                assert result.status == "succeeded" and result.progress == 100
                assert result.result_ref == "snapshot://result/合成"
                assert result.result_version == 2**32 - 1
                assert (
                    result.lease_owner
                    is result.lease_until
                    is result.next_run_at
                    is None
                )
                assert result.lease_token == lease.token == 10
                assert (
                    result.completed_at == clock.now
                    and result.heartbeat_at == heartbeat
                )
                attempt = await session.get(TaskAttempt, (lease.task_id, 1))
                assert attempt.status == "succeeded" and attempt.phase == "publishing"
                assert attempt.finished_at == result.completed_at
                assert attempt.result_summary == " 合成结果摘要 "
                assert (
                    attempt.lease_owner,
                    attempt.lease_token,
                    attempt.attempt_no,
                ) == (lease.owner, lease.token, lease.attempt_no)
                await session.commit()
            async with sessions() as check:
                assert (
                    await check.get(BackgroundTask, lease.task_id)
                ).status == "succeeded"
                assert (
                    await check.get(TaskAttempt, (lease.task_id, 1))
                ).result_summary == " 合成结果摘要 "

    asyncio.run(scenario())


def test_retries_back_off_keep_first_start_and_never_rewrite_attempt_history(
    tmp_path, backend_root, clock
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions.begin() as seed:
                seed.add(_task(lease_token=7))
            history = []
            policy = RetryPolicy(base_seconds=5, max_seconds=6)
            for number, delay in enumerate((5, 6, None), 1):
                async with sessions() as session:
                    lease = await _claim(
                        session, owner=OWNER_A if number == 1 else OWNER_B
                    )
                    assert (lease.attempt_no, lease.token) == (number, number + 7)
                    task = await session.get(BackgroundTask, lease.task_id)
                    assert task.started_at == NOW
                    assert task.input_metadata == {"options": ["合成输入", 1, True]}
                    assert (
                        task.input_ref == "snapshot://lease/1"
                        and task.input_fingerprint == "a" * 64
                    )
                    assert task.error_code is task.error_summary is None
                    assert task.completed_at is task.next_run_at is None
                    await session.commit()
                    clock.now += timedelta(seconds=1)
                    await _mutate(
                        session,
                        "heartbeat",
                        lease,
                        phase=f"indexing.{number}",
                        progress=40,
                    )
                    await session.commit()
                    heartbeat = task.heartbeat_at
                    clock.now += timedelta(seconds=1)
                    result = await _mutate(
                        session,
                        "fail",
                        lease,
                        error_code="TEMPORARY_FAILURE",
                        error_summary=f"合成失败 {number}",
                        retryable=True,
                        retry_policy=policy,
                    )
                    assert result.lease_owner is result.lease_until is None
                    assert (
                        result.lease_token == lease.token
                        and result.heartbeat_at == heartbeat
                    )
                    assert result.started_at == NOW
                    attempts = list(
                        await session.scalars(
                            select(TaskAttempt).order_by(TaskAttempt.attempt_no)
                        )
                    )
                    assert [_row(item) for item in attempts[:-1]] == history
                    latest = attempts[-1]
                    assert (
                        latest.status == "failed"
                        and latest.phase == f"indexing.{number}"
                    )
                    assert latest.error_code == result.error_code == "TEMPORARY_FAILURE"
                    assert (
                        latest.error_summary
                        == result.error_summary
                        == f"合成失败 {number}"
                    )
                    assert latest.finished_at == clock.now
                    assert (
                        latest.lease_owner == lease.owner
                        and latest.lease_token == lease.token
                    )
                    history.append(_row(latest))
                    if delay is None:
                        assert (
                            result.status == "failed"
                            and result.completed_at == clock.now
                        )
                        assert result.next_run_at is None
                    else:
                        assert (
                            result.status == "retry_wait"
                            and result.completed_at is None
                        )
                        assert result.next_run_at == clock.now + timedelta(
                            seconds=delay
                        )
                    await session.commit()
                    if delay is not None:
                        clock.now = result.next_run_at - timedelta(microseconds=1)
                        assert await _claim(session) is None
                        await session.commit()
                        clock.now += timedelta(microseconds=1)
            async with sessions() as check:
                assert await _claim(check) is None
                attempts = list(
                    await check.scalars(
                        select(TaskAttempt).order_by(TaskAttempt.attempt_no)
                    )
                )
                assert [_row(item) for item in attempts] == history

    asyncio.run(scenario())


@pytest.mark.parametrize("retryable,cancel_requested", [(False, False), (True, True)])
def test_nonretryable_failure_or_cancel_request_disables_remaining_retry_budget(
    tmp_path, backend_root, clock, retryable, cancel_requested
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                task, lease = await _running(session)
                if cancel_requested:
                    task.cancel_requested_at = NOW
                await session.commit()
                clock.now += timedelta(seconds=1)
                result = await _mutate(session, "fail", lease, retryable=retryable)
                assert result.attempt_count < result.max_attempts
                assert result.status == "failed" and result.completed_at == clock.now
                assert (
                    result.lease_owner
                    is result.lease_until
                    is result.next_run_at
                    is None
                )
                assert result.lease_token == lease.token and result.heartbeat_at == NOW
                attempt = await session.get(TaskAttempt, (lease.task_id, 1))
                assert (
                    attempt.status == "failed"
                    and attempt.error_code == result.error_code
                )
                assert attempt.error_summary == result.error_summary == "合成失败"
                assert attempt.phase == "claimed"

    asyncio.run(scenario())


def test_cancellation_requires_request_and_does_not_rollback_callers_work(
    tmp_path, backend_root, clock
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                task, lease = await _running(session)
                original = await _snapshot(session)
                session.add(
                    ContentBlob(
                        blob_id="b" * 64, byte_size=1, storage_uri="blob://synthetic"
                    )
                )
                await session.flush()
                with pytest.raises(TaskStateConflict) as raised:
                    await _mutate(session, "cancel", lease)
                assert not isinstance(raised.value, TaskLeaseLost)
                assert await _snapshot(session) == original
                assert task.cancel_requested_at is None and session.is_active
                await session.commit()
            async with sessions() as check:
                assert await check.get(ContentBlob, "b" * 64) is not None
                assert await _snapshot(check) == original

    asyncio.run(scenario())


@pytest.mark.parametrize("winner", ["succeed", "cancel"])
def test_committed_cancellation_request_is_not_a_terminal_priority(
    tmp_path, backend_root, clock, monkeypatch, winner
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as worker, sessions() as canceller:
                task, lease = await _running(worker)
                cached = await canceller.get(BackgroundTask, lease.task_id)
                await canceller.commit()
                clock.now += timedelta(seconds=1)
                monkeypatch.setattr(repository, "_now", lambda: clock.now)
                async with sessions.begin() as writer:
                    requested = await repository.request_cancel(
                        writer, lease.task_id, task.user_id
                    )
                    assert (
                        requested.status == "processing"
                        and requested.lease_owner == lease.owner
                    )
                    requested_at = requested.cancel_requested_at
                assert task.cancel_requested_at is None
                assert (
                    await _mutate(
                        worker, "heartbeat", lease, phase="stopping", progress=99
                    )
                    is True
                )
                assert task.status == "processing" and task.completed_at is None
                assert task.cancel_requested_at == requested_at
                assert (
                    task.lease_owner == lease.owner and task.lease_token == lease.token
                )
                await worker.commit()
                heartbeat = task.heartbeat_at
                clock.now += timedelta(seconds=1)
                # 未启动业务执行器；这里只验证停止前置条件满足后的数据库状态序列。
                result = await _mutate(worker, winner, lease)
                expected = "succeeded" if winner == "succeed" else "cancelled"
                assert result.status == expected and result.completed_at == clock.now
                assert result.cancel_requested_at == requested_at
                assert (
                    result.lease_owner
                    is result.lease_until
                    is result.next_run_at
                    is None
                )
                assert (
                    result.lease_token == lease.token
                    and result.heartbeat_at == heartbeat
                )
                attempt = await worker.get(TaskAttempt, (lease.task_id, 1))
                assert attempt.status == expected and attempt.phase == "stopping"
                assert attempt.finished_at == clock.now
                if winner == "succeed":
                    assert result.progress == 100 and result.result_version == 1
                await worker.commit()
                assert cached.status == "processing"
                with pytest.raises(TaskStateConflict):
                    await repository.request_cancel(
                        canceller, lease.task_id, task.user_id
                    )
                assert cached.status == expected
                other = "cancel" if winner == "succeed" else "succeed"
                with pytest.raises(TaskLeaseLost):
                    await _mutate(worker, other, lease)
                with pytest.raises(TaskLeaseLost):
                    await _mutate(worker, winner, lease)
                assert (
                    await worker.get(BackgroundTask, lease.task_id)
                ).status == expected

    asyncio.run(scenario())


def test_expired_discovery_is_read_only_and_never_reclaims_processing(
    tmp_path, backend_root, clock
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, engine):
            async with sessions() as session:
                candidates = [
                    _task(
                        1,
                        kind="knowledge.index",
                        lease_until=NOW - timedelta(seconds=1),
                    ),
                    _task(2, lease_until=NOW),
                    _task(3, kind="note.delete", lease_until=NOW),
                    _task(4, lease_until=NOW + timedelta(seconds=1)),
                ]
                for task in candidates:
                    (
                        task.status,
                        task.lease_owner,
                        task.lease_token,
                        task.attempt_count,
                    ) = ("processing", OWNER_A, 7, 1)
                    task.started_at = task.heartbeat_at = NOW - timedelta(seconds=90)
                    session.add(task)
                await session.flush()
                for task in candidates:
                    session.add(
                        TaskAttempt(
                            task_id=task.task_id,
                            attempt_no=1,
                            lease_owner=OWNER_A,
                            lease_token=7,
                            status="running",
                            phase="indexing",
                            started_at=task.started_at,
                        )
                    )
                session.add_all(
                    [
                        _task(5, lease_until=NOW),
                        _task(6, status="retry_wait", lease_until=NOW),
                        _task(7, status="failed", lease_until=NOW),
                        _task(8, status="processing", lease_until=None),
                    ]
                )
                await session.commit()
                original = await _snapshot(session)
                statements = []

                @event.listens_for(engine.sync_engine, "before_cursor_execute")
                def record(_connection, _cursor, statement, *_):
                    statements.append(statement.lstrip().upper())

                selected_kinds = ("knowledge.index", "note.index")
                expected = [
                    TaskLease(_task_id(1), OWNER_A, 7, 1),
                    TaskLease(_task_id(2), OWNER_A, 7, 1),
                ]
                with _caller_transaction(session):
                    found = await leases.list_expired_leases(
                        session, kinds=selected_kinds
                    )
                    limited = await leases.list_expired_leases(
                        session, kinds=selected_kinds, limit=1
                    )
                    assert set(found) == set(expected)
                    assert len(limited) == 1 and limited[0] in expected
                assert all(
                    not statement.startswith(("INSERT", "UPDATE", "DELETE"))
                    for statement in statements
                )
                assert not session.dirty and not session.new
                assert await _snapshot(session) == original
                # 只选该种类，排除本测试特意加入的 pending 过滤样本。
                assert await _claim(session, kinds=("knowledge.index",)) is None
                for operation in MUTATIONS:
                    with pytest.raises(TaskLeaseLost):
                        await _mutate(session, operation, expected[1])
                await session.commit()
            async with sessions() as check:
                assert await _snapshot(check) == original

    asyncio.run(scenario())


def test_elapsed_time_fences_the_same_receipt_in_a_later_transaction(
    tmp_path, backend_root, clock
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as first:
                task, lease = await _running(first)
                original = await _snapshot(first)
            clock.now = task.lease_until + timedelta(microseconds=1)
            async with sessions() as later:
                for operation in MUTATIONS:
                    with pytest.raises(TaskLeaseLost):
                        await _mutate(later, operation, lease)
                assert await _claim(later) is None
                assert await leases.list_expired_leases(later, kinds=KINDS) == [lease]
                assert await _snapshot(later) == original

    asyncio.run(scenario())


def test_failed_attempt_insert_propagates_and_caller_rollback_undoes_claim(
    tmp_path, backend_root, clock
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                session.add(_task(lease_token=7))
                await session.commit()
                original = await _snapshot(session)

                @event.listens_for(session.sync_session, "before_flush")
                def violate_attempt_check(sync_session, *_):
                    for candidate in sync_session.new:
                        if isinstance(candidate, TaskAttempt):
                            candidate.status = "invalid-attempt-state"

                session.add(
                    ContentBlob(
                        blob_id="b" * 64, byte_size=1, storage_uri="blob://synthetic"
                    )
                )
                with pytest.raises(IntegrityError, match="CHECK constraint failed"):
                    await _claim(session)
                await session.rollback()
            async with sessions() as check:
                assert await _snapshot(check) == original
                assert await check.get(ContentBlob, "b" * 64) is None

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ("claim", *MUTATIONS, "expired"))
def test_database_errors_propagate_without_internal_transaction_recovery(
    monkeypatch, operation
):
    error = OperationalError("SELECT", {}, RuntimeError("合成数据库锁等待错误"))
    session = AsyncMock(spec=AsyncSession)
    session.scalar.side_effect = session.scalars.side_effect = error
    monkeypatch.setattr(leases, "_database_now", AsyncMock(return_value=NOW))
    lease = TaskLease(_task_id(1), OWNER_A, 1, 1)

    async def invoke():
        if operation == "claim":
            return await _claim(session)
        if operation == "expired":
            return await leases.list_expired_leases(session, kinds=KINDS)
        return await _mutate(session, operation, lease)

    with pytest.raises(OperationalError) as raised:
        asyncio.run(invoke())
    assert raised.value is error
    session.commit.assert_not_called()
    session.rollback.assert_not_called()


@pytest.mark.parametrize(
    "dialect,expression",
    [(mysql.dialect(), "UTC_TIMESTAMP()"), (sqlite.dialect(), "CURRENT_TIMESTAMP")],
    ids=["mysql-utc", "sqlite-utc"],
)
def test_database_clock_selects_dialect_utc_not_application_time(dialect, expression):
    session = AsyncMock(spec=AsyncSession)
    session.get_bind.return_value = SimpleNamespace(dialect=dialect)
    session.scalar.return_value = NOW
    assert asyncio.run(leases._database_now(session)) == NOW
    statement = session.scalar.call_args.args[0]
    sql = str(statement.compile(dialect=dialect)).upper()
    assert expression in sql
    session.scalar.assert_awaited_once()
    session.commit.assert_not_called()
    session.rollback.assert_not_called()


def test_database_clock_normalizes_aware_values_and_propagates_driver_error():
    session = AsyncMock(spec=AsyncSession)
    session.get_bind.return_value = SimpleNamespace(dialect=mysql.dialect())
    session.scalar.return_value = NOW.replace(tzinfo=timezone(timedelta(hours=8)))
    assert asyncio.run(leases._database_now(session)) == NOW - timedelta(hours=8)
    error = OperationalError("SELECT UTC_TIMESTAMP()", {}, RuntimeError("合成连接故障"))
    session.scalar.side_effect = error
    with pytest.raises(OperationalError) as raised:
        asyncio.run(leases._database_now(session))
    assert raised.value is error
    session.commit.assert_not_called()
    session.rollback.assert_not_called()


def test_real_sqlite_clock_returns_a_database_datetime(tmp_path, backend_root):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                before = datetime.fromisoformat(
                    await session.scalar(text("SELECT CURRENT_TIMESTAMP"))
                )
                actual = await leases._database_now(session)
                after = datetime.fromisoformat(
                    await session.scalar(text("SELECT CURRENT_TIMESTAMP"))
                )
                assert actual.tzinfo is None and before <= actual <= after

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", MUTATIONS)
@pytest.mark.parametrize(
    "late_by",
    [timedelta(0), timedelta(microseconds=1)],
    ids=["at-deadline", "after-deadline"],
)
def test_update_guard_rejects_expiry_after_the_last_time_select(
    tmp_path, backend_root, clock, monkeypatch, operation, late_by
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, engine):
            async with sessions() as session:
                task, lease = await _running(session)
                task.cancel_requested_at = NOW
                await session.commit()
                original = await _snapshot(session)
                session.add(
                    ContentBlob(
                        blob_id="b" * 64, byte_size=1, storage_uri="blob://synthetic"
                    )
                )
                await session.flush()
                deadline = task.lease_until
                clock.now = deadline - timedelta(seconds=1)
                readings, guarded_updates = [], []

                async def read_then_pause(_session):
                    observed = clock.now
                    readings.append(observed)
                    # SELECT 已取得仍有效的快照；恢复执行时，UPDATE 的数据库时钟已到期。
                    clock.now = deadline + late_by
                    return observed

                @event.listens_for(engine.sync_engine, "after_cursor_execute")
                def count_update(_connection, cursor, statement, *_):
                    if statement.lstrip().upper().startswith("UPDATE BACKGROUND_TASKS"):
                        guarded_updates.append(cursor.rowcount)

                monkeypatch.setattr(leases, "_database_now", read_then_pause)
                with pytest.raises(TaskLeaseLost):
                    await _mutate(session, operation, lease)
                assert readings == [deadline - timedelta(seconds=1)]
                assert guarded_updates == [0]
                assert not session.dirty and not session.new
                assert session.is_active and session.in_transaction()
                assert await _snapshot(session) == original
                # guard miss 不是事务失败，更不能由仓储 rollback 掉调用方已完成的工作。
                await session.commit()
            async with sessions() as check:
                assert await _snapshot(check) == original
                assert await check.get(ContentBlob, "b" * 64) is not None

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", MUTATIONS)
def test_mutation_timestamps_come_from_the_guarded_update_not_the_earlier_select(
    tmp_path, backend_root, clock, monkeypatch, operation
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                task, lease = await _running(session)
                if operation == "cancel":
                    task.cancel_requested_at = NOW
                await session.commit()
                clock.now = NOW + timedelta(seconds=1)

                async def read_then_pause(_session):
                    observed = clock.now
                    clock.now += timedelta(seconds=1)
                    return observed

                monkeypatch.setattr(leases, "_database_now", read_then_pause)
                arguments = {"retryable": True} if operation == "fail" else {}
                await _mutate(session, operation, lease, **arguments)
                updated = NOW + timedelta(seconds=2)
                assert task.updated_at == updated
                attempt = await session.get(TaskAttempt, (lease.task_id, 1))
                if operation == "heartbeat":
                    assert task.heartbeat_at == updated
                    assert task.lease_until == updated + timedelta(seconds=90)
                    assert task.completed_at is attempt.finished_at is None
                elif operation == "fail":
                    assert task.next_run_at == updated + timedelta(seconds=5)
                    assert task.completed_at is None and attempt.finished_at == updated
                else:
                    assert task.completed_at == attempt.finished_at == updated

    asyncio.run(scenario())


def test_same_second_unchanged_heartbeat_is_still_a_matching_lease(
    tmp_path, backend_root, clock
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                _, lease = await _running(session)
                original = await _snapshot(session)
                assert await _mutate(session, "heartbeat", lease) is False
                await session.commit()
                assert await _mutate(session, "heartbeat", lease) is False
                assert await _snapshot(session) == original

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ("claim", *MUTATIONS))
def test_successful_update_refreshes_with_an_explicit_current_lock(
    tmp_path, backend_root, clock, monkeypatch, operation
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                if operation == "claim":
                    task = _task()
                    session.add(task)
                    await session.commit()
                else:
                    task, lease = await _running(session)
                    if operation == "cancel":
                        task.cancel_requested_at = NOW
                    await session.commit()
                refresh = AsyncMock(wraps=session.refresh)
                monkeypatch.setattr(session, "refresh", refresh)
                statements = _capture_selects(session)
                if operation == "claim":
                    assert await _claim(session) is not None
                else:
                    await _mutate(session, operation, lease)
                # MySQL RR 下普通 refresh 可能回读旧快照；同秒 no-op 心跳也须当前读。
                refresh.assert_awaited_once_with(task, with_for_update=True)
                _assert_exclusive_refresh(statements, "background_tasks")
                task_queries = [
                    query
                    for query, _ in statements
                    if "FROM background_tasks" in _mysql_sql(query)
                ]
                assert len(task_queries) >= 2
                assert _mysql_sql(task_queries[-1]).endswith("FOR UPDATE")

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ("claim", *MUTATIONS))
def test_mysql_state_update_contains_fencing_and_utc_clock(monkeypatch, operation):
    class CapturedUpdate(RuntimeError):
        pass

    task = _task()
    lease = TaskLease(task.task_id, OWNER_A, 1, 1)
    session = AsyncMock(spec=AsyncSession)
    session.get_bind.return_value = SimpleNamespace(dialect=mysql.dialect())
    if operation == "claim":
        session.scalar.return_value = task
    else:
        task.status, task.lease_owner, task.lease_token, task.attempt_count = (
            "processing",
            OWNER_A,
            1,
            1,
        )
        task.lease_until = NOW + timedelta(seconds=90)
        task.cancel_requested_at = NOW if operation == "cancel" else None
        attempt = TaskAttempt(
            task_id=task.task_id,
            attempt_no=1,
            lease_owner=OWNER_A,
            lease_token=1,
            status="running",
        )
        session.scalar.side_effect = [task, attempt]
    session.execute.side_effect = CapturedUpdate
    monkeypatch.setattr(leases, "_database_now", AsyncMock(return_value=NOW))

    async def invoke():
        if operation == "claim":
            return await _claim(session)
        arguments = {"retryable": True} if operation == "fail" else {}
        return await _mutate(session, operation, lease, **arguments)

    with pytest.raises(CapturedUpdate):
        asyncio.run(invoke())
    query = session.execute.call_args.args[0]
    sql = _mysql_sql(query)
    assert sql.startswith("UPDATE background_tasks SET ")
    assert "UTC_TIMESTAMP()" in sql.upper()
    assert "CURRENT_TIMESTAMP" not in sql.upper()
    where = sql.split(" WHERE ", 1)[1]
    for column in ("task_id", "lease_token", "attempt_count", "status"):
        assert f"background_tasks.{column}" in where
    if operation != "claim":
        assert "background_tasks.lease_owner" in where
        assert "background_tasks.lease_until >" in where
        assert "UTC_TIMESTAMP()" in where.upper()
    session.commit.assert_not_called()
    session.rollback.assert_not_called()


def test_claim_cannot_wrap_an_exhausted_uint64_token(monkeypatch):
    # SQLite 无法存储无符号 64 位上界；这里只验证进入 SQL 写入前的溢出拒绝。
    task = _task(lease_token=2**64 - 1)
    original = _row(task)
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = task
    monkeypatch.setattr(leases, "_database_now", AsyncMock(return_value=NOW))
    with pytest.raises(TaskStateConflict):
        asyncio.run(_claim(session))
    assert _row(task) == original
    session.execute.assert_not_called()
    session.add.assert_not_called()
    session.flush.assert_not_called()


def test_expired_scan_default_limit_and_refresh_are_read_only(monkeypatch):
    session = AsyncMock(spec=AsyncSession)
    session.scalars.return_value = []
    monkeypatch.setattr(leases, "_database_now", AsyncMock(return_value=NOW))
    assert asyncio.run(leases.list_expired_leases(session, kinds=KINDS)) == []
    query = session.scalars.call_args.args[0]
    sql = _mysql_sql(query)
    compiled = query.compile(dialect=mysql.dialect())
    assert "FOR UPDATE" not in sql and "LOCK IN SHARE MODE" not in sql
    assert query.get_execution_options().get("populate_existing") is True
    assert (
        compiled.params[next(key for key in compiled.params if key.startswith("param"))]
        == 20
    )
    session.flush.assert_not_called()
    session.execute.assert_not_called()
    session.commit.assert_not_called()
    session.rollback.assert_not_called()
