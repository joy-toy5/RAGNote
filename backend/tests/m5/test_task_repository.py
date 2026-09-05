"""真实异步事务边界；仅使用 tmp_path SQLite，不冒充 MySQL 并发验收。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import event, select, update
from sqlalchemy.dialects import mysql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.indexing.models import ContentBlob
from app.tasking import repository
from app.tasking.contracts import TaskSubmission
from app.tasking.errors import TaskIdempotencyConflict, TaskNotFound, TaskStateConflict
from app.tasking.models import BackgroundTask

USER_ID = "task-user-a"
RESOURCE_ID = "11111111-1111-4111-8111-111111111111"
NOW = datetime(2026, 9, 6, 1, 0, tzinfo=timezone.utc)


def _submission(**overrides):
    values = dict(
        user_id=USER_ID,
        kind="note.index",
        idempotency_key="note:1:v1",
        input_fingerprint="a" * 64,
        input_ref="snapshot://notes/1/1",
        resource_id=RESOURCE_ID,
        target_generation=1,
        input_metadata={"options": {"tags": ["原文"]}},
    )
    values.update(overrides)
    return TaskSubmission(**values)


@asynccontextmanager
async def _database(tmp_path, backend_root):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tasks.sqlite3'}")

    @event.listens_for(engine.sync_engine, "connect")
    def configure_sqlite(connection, _):
        # 禁止 legacy SAVEPOINT 自行提交，确保外层 rollback 真的撤销候选任务。
        connection.isolation_level = None
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    @event.listens_for(engine.sync_engine, "begin")
    def begin_sqlite(connection):
        connection.exec_driver_sql("BEGIN")

    def migrate(connection):
        config = Config()
        config.set_main_option("script_location", str(backend_root / "alembic"))
        config.attributes["connection"] = connection
        command.upgrade(config, "head")

    try:
        async with engine.begin() as connection:
            await connection.run_sync(migrate)
        yield async_sessionmaker(engine, expire_on_commit=False), engine
    finally:
        await engine.dispose()


def test_outer_rollback_removes_task_and_other_pending_work(tmp_path, backend_root):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                session.add(
                    ContentBlob(blob_id="b" * 64, byte_size=1, storage_uri="blob://b")
                )
                task, created = await repository.create_or_get_task(
                    session, _submission()
                )
                task_id = task.task_id
                assert created and session.in_transaction()
                await session.rollback()
            async with sessions() as check:
                assert await check.get(BackgroundTask, task_id) is None
                assert await check.get(ContentBlob, "b" * 64) is None

    asyncio.run(scenario())


def test_idempotency_is_user_scoped_and_rejects_changed_input(tmp_path, backend_root):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                original, created = await repository.create_or_get_task(
                    session, _submission()
                )
                same, recreated = await repository.create_or_get_task(
                    session, _submission()
                )
                other, other_created = await repository.create_or_get_task(
                    session,
                    _submission(user_id="task-user-b"),
                )
                assert created and other_created and not recreated
                assert same.task_id == original.task_id != other.task_id
                with pytest.raises(TaskIdempotencyConflict):
                    await repository.create_or_get_task(
                        session, _submission(input_fingerprint="b" * 64)
                    )
                await session.commit()
            async with sessions() as check:
                found, created = await repository.create_or_get_task(
                    check, _submission()
                )
                assert not created and found.task_id == original.task_id

    asyncio.run(scenario())


def test_metadata_snapshot_precedes_database_await(tmp_path, backend_root):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, engine):
            submission = _submission()

            @event.listens_for(engine.sync_engine, "before_cursor_execute")
            def mutate_at_database_boundary(_, __, statement, *___):
                if (
                    statement.lstrip().startswith("SELECT")
                    and "background_tasks" in statement
                ):
                    submission.input_metadata["options"]["tags"].append("外部改写")

            async with sessions() as session:
                task, _ = await repository.create_or_get_task(session, submission)
                assert task.input_metadata == {"options": {"tags": ["原文"]}}
                submission.input_metadata["options"]["tags"].append("返回后改写")
                assert task.input_metadata == {"options": {"tags": ["原文"]}}

    asyncio.run(scenario())


@pytest.mark.parametrize("first,second", [(True, 1), (False, 0), (1, 1.0)])
def test_json_idempotency_keeps_scalar_types(tmp_path, backend_root, first, second):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                await repository.create_or_get_task(
                    session, _submission(input_metadata={"v": first})
                )
                with pytest.raises(TaskIdempotencyConflict):
                    await repository.create_or_get_task(
                        session, _submission(input_metadata={"v": second})
                    )

    asyncio.run(scenario())


def test_repeated_manual_retry_returns_same_new_task(tmp_path, backend_root):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                source, _ = await repository.create_or_get_task(session, _submission())
                source.status = "failed"
                source.completed_at = NOW
                source.lease_token = 9
                await session.commit()
                first = await repository.retry_task(
                    session, source.task_id, USER_ID, idempotency_key="retry:1"
                )
                await session.commit()
                again = await repository.retry_task(
                    session, source.task_id, USER_ID, idempotency_key="retry:1"
                )
                assert first.task_id == again.task_id != source.task_id
                assert first.retry_of_task_id == source.task_id
                assert first.status == "pending" and first.lease_token == 0
                assert source.status == "failed" and source.lease_token == 9
                assert source.completed_at == NOW.replace(tzinfo=None)
                with pytest.raises(TaskIdempotencyConflict):
                    await repository.retry_task(
                        session,
                        source.task_id,
                        USER_ID,
                        idempotency_key=source.idempotency_key,
                    )

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["cancel", "retry"])
def test_mutations_refresh_stale_identity_map(tmp_path, backend_root, operation):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as stale:
                cached, _ = await repository.create_or_get_task(stale, _submission())
                cached.status = "pending" if operation == "cancel" else "failed"
                await stale.commit()
                async with sessions.begin() as writer:
                    await writer.execute(
                        update(BackgroundTask).values(status="succeeded")
                    )
                with pytest.raises(TaskStateConflict):
                    if operation == "cancel":
                        await repository.request_cancel(stale, cached.task_id, USER_ID)
                    else:
                        await repository.retry_task(
                            stale, cached.task_id, USER_ID, idempotency_key="retry:1"
                        )
                assert cached.status == "succeeded"

    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["pending", "retry_wait", "processing"])
def test_cancel_preserves_fencing_and_processing_execution(
    tmp_path, backend_root, status
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                task, _ = await repository.create_or_get_task(session, _submission())
                task.status = status
                task.phase = "indexing"
                task.lease_owner = "worker-a" if status == "processing" else None
                task.lease_until = NOW if status == "processing" else None
                task.lease_token = 7
                await session.commit()
                result = await repository.request_cancel(session, task.task_id, USER_ID)
                assert result.lease_token == 7
                assert result.cancel_requested_at is not None
                if status == "processing":
                    assert result.status == "processing" and result.phase == "indexing"
                    assert (
                        result.completed_at is None and result.lease_owner == "worker-a"
                    )
                    requested_at = result.cancel_requested_at
                    repeated = await repository.request_cancel(
                        session, task.task_id, USER_ID
                    )
                    assert repeated.cancel_requested_at.replace(
                        tzinfo=None
                    ) == requested_at.replace(tzinfo=None)
                else:
                    assert result.status == result.phase == "cancelled"
                    assert result.completed_at == result.cancel_requested_at
                    assert result.lease_owner is result.lease_until is None

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["get", "cancel", "retry"])
def test_task_operations_hide_foreign_users(tmp_path, backend_root, operation):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                task, _ = await repository.create_or_get_task(session, _submission())
                task.status = "failed"
                await session.commit()
                with pytest.raises(TaskNotFound):
                    if operation == "get":
                        await repository.get_task(session, task.task_id, "task-user-b")
                    elif operation == "cancel":
                        await repository.request_cancel(
                            session, task.task_id, "task-user-b"
                        )
                    else:
                        await repository.retry_task(
                            session,
                            task.task_id,
                            "task-user-b",
                            idempotency_key="retry:1",
                        )
                assert await repository.list_tasks(session, "task-user-b") == []

    asyncio.run(scenario())


def _miss_first_lookup(monkeypatch):
    original = repository._get_by_idempotency
    lookups = []

    async def lookup(*args, **kwargs):
        lookups.append(kwargs)
        if len(lookups) == 1:
            return None
        return await original(*args, **kwargs)

    monkeypatch.setattr(repository, "_get_by_idempotency", lookup)
    return lookups


def test_unique_race_keeps_outer_transaction_and_uses_current_read(
    tmp_path, backend_root, monkeypatch
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as seed:
                winner, _ = await repository.create_or_get_task(seed, _submission())
                await seed.commit()
            _miss_first_lookup(monkeypatch)
            async with sessions() as session:
                statements = []

                @event.listens_for(session.sync_session, "do_orm_execute")
                def capture(execution):
                    if execution.is_select:
                        statements.append(execution)

                session.add(
                    ContentBlob(blob_id="b" * 64, byte_size=1, storage_uri="blob://b")
                )
                task, created = await repository.create_or_get_task(
                    session, _submission()
                )
                assert not created and task.task_id == winner.task_id
                assert session.is_active
                await session.commit()
                statement = statements[-1]
                assert str(
                    statement.statement.compile(dialect=mysql.dialect())
                ).endswith("LOCK IN SHARE MODE")
                assert statement.execution_options.get("populate_existing") is True
            async with sessions() as check:
                assert await check.get(ContentBlob, "b" * 64) is not None

    asyncio.run(scenario())


def test_non_target_integrity_error_is_not_mistaken_for_idempotency(
    tmp_path, backend_root, monkeypatch
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as seed:
                await repository.create_or_get_task(seed, _submission())
                await seed.commit()
            _miss_first_lookup(monkeypatch)
            async with sessions() as session:

                @event.listens_for(session.sync_session, "before_flush")
                def violate_check(sync_session, *_):
                    for candidate in sync_session.new:
                        if isinstance(candidate, BackgroundTask):
                            candidate.status = "not-a-state"

                session.add(
                    ContentBlob(blob_id="b" * 64, byte_size=1, storage_uri="blob://b")
                )
                with pytest.raises(IntegrityError, match="CHECK constraint failed"):
                    await repository.create_or_get_task(session, _submission())
                assert session.is_active
                await session.commit()
            async with sessions() as check:
                assert await check.get(ContentBlob, "b" * 64) is not None

    asyncio.run(scenario())


def test_foreign_key_failure_leaves_outer_transaction_usable(tmp_path, backend_root):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                session.add(
                    ContentBlob(blob_id="b" * 64, byte_size=1, storage_uri="blob://b")
                )
                with pytest.raises(IntegrityError, match="FOREIGN KEY"):
                    await repository.create_or_get_task(
                        session, _submission(input_blob_id="c" * 64)
                    )
                assert session.is_active
                await session.commit()
            async with sessions() as check:
                assert await check.get(ContentBlob, "b" * 64) is not None
                assert list(await check.scalars(select(BackgroundTask))) == []

    asyncio.run(scenario())


def test_list_has_deterministic_tie_breaker_and_user_scoped_key_lookup(
    tmp_path, backend_root
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                task_ids = []
                for index in (2, 1, 3):
                    task, _ = await repository.create_or_get_task(
                        session, _submission(idempotency_key=f"key:{index}")
                    )
                    task.task_id = f"{index:08d}-1111-4111-8111-111111111111"
                    task.created_at = task.updated_at = NOW
                    task_ids.append(task.task_id)
                await repository.create_or_get_task(
                    session, _submission(user_id="task-user-b", idempotency_key="key:3")
                )
                await session.commit()
                pages = [
                    await repository.list_tasks(session, USER_ID, 1, offset=i)
                    for i in range(3)
                ]
                assert [page[0].task_id for page in pages] == sorted(
                    task_ids, reverse=True
                )
                found = await repository.list_tasks(
                    session, USER_ID, idempotency_key="key:3"
                )
                assert len(found) == 1 and found[0].user_id == USER_ID
                assert len(await repository.list_tasks(session, USER_ID, 999)) == 3
                assert (
                    len(await repository.list_tasks(session, USER_ID, 0, offset=-1))
                    == 1
                )

    asyncio.run(scenario())


def test_outer_flush_failure_is_not_caught_as_candidate_conflict(
    tmp_path, backend_root, monkeypatch
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            lookups = _miss_first_lookup(monkeypatch)
            async with sessions() as session:
                session.add(
                    ContentBlob(blob_id="b" * 64, byte_size=-1, storage_uri="blob://b")
                )
                with (
                    session.no_autoflush,
                    pytest.raises(IntegrityError, match="byte_size"),
                ):
                    await repository.create_or_get_task(session, _submission())
                assert lookups == []
                await session.rollback()

    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["succeeded", "failed", "superseded", "cancelled"])
def test_cancel_never_overwrites_terminal_state(tmp_path, backend_root, status):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                task, _ = await repository.create_or_get_task(session, _submission())
                task.status = status
                task.completed_at = NOW
                await session.commit()
                with pytest.raises(TaskStateConflict):
                    await repository.request_cancel(session, task.task_id, USER_ID)
                assert task.status == status and task.cancel_requested_at is None
                assert task.completed_at == NOW.replace(tzinfo=None)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status", ["pending", "processing", "retry_wait", "succeeded", "superseded"]
)
def test_retry_rejects_non_retryable_states(tmp_path, backend_root, status):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                task, _ = await repository.create_or_get_task(session, _submission())
                task.status = status
                await session.commit()
                with pytest.raises(TaskStateConflict):
                    await repository.retry_task(
                        session, task.task_id, USER_ID, idempotency_key="retry:1"
                    )
                assert len(await repository.list_tasks(session, USER_ID)) == 1

    asyncio.run(scenario())


def test_manual_retry_rollback_preserves_original_without_new_task(
    tmp_path, backend_root
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as session:
                source, _ = await repository.create_or_get_task(session, _submission())
                source.status = "cancelled"
                source_id = source.task_id
                await session.commit()
                retried = await repository.retry_task(
                    session, source_id, USER_ID, idempotency_key="retry:1"
                )
                new_id = retried.task_id
                await session.rollback()
            async with sessions() as check:
                assert await check.get(BackgroundTask, new_id) is None
                original = await check.get(BackgroundTask, source_id)
                assert original.status == "cancelled"
                assert original.retry_of_task_id is None

    asyncio.run(scenario())


def test_conflicting_unique_race_keeps_callers_work(
    tmp_path, backend_root, monkeypatch
):
    async def scenario():
        async with _database(tmp_path, backend_root) as (sessions, _):
            async with sessions() as seed:
                await repository.create_or_get_task(seed, _submission())
                await seed.commit()
            _miss_first_lookup(monkeypatch)
            async with sessions() as session:
                session.add(
                    ContentBlob(blob_id="b" * 64, byte_size=1, storage_uri="blob://b")
                )
                with pytest.raises(TaskIdempotencyConflict):
                    await repository.create_or_get_task(
                        session, _submission(input_ref="snapshot://different")
                    )
                assert session.is_active
                await session.commit()
            async with sessions() as check:
                assert await check.get(ContentBlob, "b" * 64) is not None

    asyncio.run(scenario())
