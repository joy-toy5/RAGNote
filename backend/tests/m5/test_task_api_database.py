"""临时 SQLite 上的真实 AsyncSession/repository/HTTP 合同，不装配生产应用。"""

from __future__ import annotations

import asyncio
import importlib.util
import sqlite3
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Header, HTTPException
from sqlalchemy import MetaData, event, inspect, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.db.schema_gate import Base
from app.tasking import repository
from app.tasking.contracts import TaskSubmission
from app.tasking.models import BackgroundTask


USER_ID = "user-a"
MISSING_ID = "99999999-9999-4999-8999-999999999999"
RESOURCE_ID = "33333333-3333-4333-8333-333333333333"
NOW = datetime(2026, 9, 5, 8, tzinfo=timezone.utc)
PUBLIC_FIELDS = {
    "task_id", "kind", "status", "resource_id", "target_generation", "progress",
    "created_at", "updated_at", "started_at", "completed_at", "next_run_at",
    "cancel_requested_at", "error_code", "error_summary", "retry_of_task_id",
}


def _sqlite_metadata():
    # 沿用项目 SQLite 测试的排序适配；只改副本，保留全局 ORM 的 MySQL collation。
    metadata = MetaData()
    for table in Base.metadata.sorted_tables:
        copied = table.to_metadata(metadata)
        for column in copied.columns:
            if getattr(column.type, "collation", None):
                column.type = column.type.copy()
                column.type.collation = "BINARY"
    return metadata


def _configure_sqlite(connection, _record):
    connection.isolation_level = None
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _begin_sqlite(connection):
    # 禁用 sqlite3 的 legacy 事务模式：SAVEPOINT 释放不得替代外层 commit。
    connection.exec_driver_sql("BEGIN")


@pytest.fixture
def database_api(monkeypatch, backend_root, tmp_path):
    state = SimpleNamespace(events=[], expired_after_flush=[], errors=[], before_commit=lambda _: None)

    def before_commit(session):
        if not session.in_nested_transaction():
            state.events.append("commit")
            state.before_commit(session)

    def after_commit(session):
        if not session.in_nested_transaction():
            state.events.append("committed")

    def after_flush(session, _context):
        state.expired_after_flush.extend(
            set(inspect(task).expired_attributes)
            for task in session.identity_map.values() if isinstance(task, BackgroundTask)
        )

    async def get_db():
        async with state.sessions() as session:
            event.listen(session.sync_session, "before_commit", before_commit)
            event.listen(session.sync_session, "after_commit", after_commit)
            event.listen(session.sync_session, "after_flush_postexec", after_flush)
            try:
                # 不在 yield 后自动提交，HTTP 成功必须依赖路由的显式 commit。
                yield session
            except Exception as exc:
                state.errors.append(exc)
                await session.rollback()
                state.events.append("rollback")
                raise

    async def identity(x_test_user: str | None = Header(None)):
        if x_test_user is None:
            raise HTTPException(status_code=401, detail="未登录")
        return x_test_user

    for name, attributes in {
        "app.db.db_config": {"get_db": get_db},
        "app.utils.auth_utils": {"get_current_user_id": identity},
    }.items():
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
    spec = importlib.util.spec_from_file_location(
        "_m5_task_api_database", backend_root / "app/router/task_router.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.task_router)
    state.module = module
    build_response = module.success_response

    def observed_response(**kwargs):
        response = build_response(**kwargs)
        state.events.append("response")
        return response

    monkeypatch.setattr(module, "success_response", observed_response)

    async def observed_app(scope, receive, send):
        async def observed_send(message):
            if message["type"] == "http.response.start":
                state.events.append(f"http:{message['status']}")
            await send(message)

        await app(scope, receive, observed_send)

    @asynccontextmanager
    async def open_api():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'task_api_test.sqlite3'}",
            poolclass=NullPool, connect_args={"timeout": 2},
        )
        event.listen(engine.sync_engine, "connect", _configure_sqlite)
        event.listen(engine.sync_engine, "begin", _begin_sqlite)
        try:
            async with engine.begin() as connection:
                await connection.run_sync(_sqlite_metadata().create_all)
            state.sessions = async_sessionmaker(engine, expire_on_commit=True)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=observed_app, raise_app_exceptions=False),
                base_url="http://test", headers={"X-Test-User": USER_ID},
            ) as client:
                state.client = client
                yield state
        finally:
            await engine.dispose()

    return open_api


def _submission(key, user_id=USER_ID):
    return TaskSubmission(
        user_id=user_id, kind="note.index", idempotency_key=key,
        input_fingerprint="a" * 64, input_ref="private://snapshots/source",
        input_metadata={"snapshot": {"value": "private-metadata"}},
        resource_id=RESOURCE_ID, target_generation=2,
    )


async def _seed(sessions, *, key="source-key", user_id=USER_ID, status="failed", updated_at=NOW):
    async with sessions() as session:
        task, created = await repository.create_or_get_task(session, _submission(key, user_id))
        assert created
        task_id = task.task_id
        task.status = status
        task.phase = "private-phase"
        task.created_at = NOW
        task.updated_at = updated_at
        task.completed_at = NOW if status in {"failed", "cancelled"} else None
        task.error_code = "INDEX_TIMEOUT" if status == "failed" else None
        task.error_summary = "索引执行超时，请稍后重试" if status == "failed" else None
        await session.commit()
        return task_id


async def _rows(sessions):
    async with sessions() as session:
        result = await session.execute(select(BackgroundTask.__table__).order_by(BackgroundTask.task_id))
        return [dict(row) for row in result.mappings()]


async def _write(api, task_id, operation, key="retry-client-key", **kwargs):
    payload = {"json": {"idempotency_key": key}} if operation == "retry" else {}
    return await api.client.post(f"/tasks/{task_id}/{operation}", **payload, **kwargs)


def _fail_response(*_args, **_kwargs):
    raise ValueError("private response encoding failure")


def test_sqlite_savepoint_is_not_an_outer_commit(database_api):
    async def scenario():
        async with database_api() as api:
            async with api.sessions() as session:
                assert await session.scalar(text("PRAGMA foreign_keys")) == 1
                task, created = await repository.create_or_get_task(session, _submission("uncommitted"))
                assert created and task.task_id
                # 独立物理连接不可见尚未 commit 的 INSERT；外层 rollback 后也必须为空。
                assert await _rows(api.sessions) == []
                await session.rollback()
            assert await _rows(api.sessions) == []

    asyncio.run(scenario())


@pytest.mark.parametrize("operation,source_status,expected_status", [
    ("cancel", "pending", "cancelled"), ("cancel", "retry_wait", "cancelled"),
    ("cancel", "processing", "processing"),
    ("retry", "failed", "pending"), ("retry", "cancelled", "pending"),
])
def test_writes_reload_expired_defaults_and_commit_before_http(
    database_api, monkeypatch, operation, source_status, expected_status,
):
    # 模拟不支持 INSERT RETURNING 的方言，真实 flush 后时间字段必须先异步加载。
    monkeypatch.setattr(BackgroundTask.__mapper__, "eager_defaults", False)

    async def scenario():
        async with database_api() as api:
            source_id = await _seed(api.sessions, status=source_status)
            before = await _rows(api.sessions)
            response = await _write(api, source_id, operation)
            assert response.status_code == 200, response.text
            assert api.events == ["response", "commit", "committed", "http:200"]
            expired_field = "created_at" if operation == "retry" else "updated_at"
            assert any(expired_field in fields for fields in api.expired_after_flush)
            data = response.json()["data"]
            assert set(data) == PUBLIC_FIELDS
            assert data["status"] == expected_status and data["updated_at"]
            assert data["resource_id"] == RESOURCE_ID and data["target_generation"] == 2
            assert "private" not in response.text and "retry-client-key" not in response.text
            rows = {row["task_id"]: row for row in await _rows(api.sessions)}
            assert rows[data["task_id"]]["status"] == expected_status
            if operation == "retry":
                assert data["task_id"] != source_id and data["retry_of_task_id"] == source_id
                assert rows[source_id] == before[0]
                assert rows[data["task_id"]]["idempotency_key"] == "retry-client-key"
            else:
                assert data["task_id"] == source_id and data["cancel_requested_at"]
                assert bool(data["completed_at"]) == (expected_status == "cancelled")

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["cancel", "retry"])
@pytest.mark.parametrize("failure_stage", ["dto", "response", "commit"])
def test_failed_response_or_commit_rolls_back_real_mutations(
    database_api, monkeypatch, operation, failure_stage,
):
    async def scenario():
        async with database_api() as api:
            source_id = await _seed(api.sessions, status="pending" if operation == "cancel" else "failed")
            before = await _rows(api.sessions)
            if failure_stage == "dto":
                monkeypatch.setattr(api.module, "TaskResponse", SimpleNamespace(model_validate=_fail_response))
            elif failure_stage == "response":
                monkeypatch.setattr(api.module, "success_response", _fail_response)
            else:
                def defer_foreign_key_violation(session):
                    # INSERT/UPDATE 都成功，真正的 DBAPI COMMIT 才因延迟外键约束失败。
                    connection = session.connection()
                    connection.exec_driver_sql("PRAGMA defer_foreign_keys=ON")
                    connection.execute(update(BackgroundTask).where(
                        BackgroundTask.task_id == source_id,
                    ).values(input_blob_id="f" * 64))

                api.before_commit = defer_foreign_key_violation
            response = await _write(api, source_id, operation)
            assert response.status_code == (503 if failure_stage == "commit" else 500)
            assert "private" not in response.text and "FOREIGN KEY" not in response.text
            assert "committed" not in api.events and "http:200" not in api.events
            assert "rollback" in api.events
            assert ("commit" in api.events) == (failure_stage == "commit")
            if failure_stage == "commit":
                assert response.json() == {"detail": "任务存储暂不可用，请稍后重试"}
                assert isinstance(api.errors[-1].__cause__, IntegrityError)
                assert isinstance(api.errors[-1].__cause__.orig, sqlite3.IntegrityError)
            assert await _rows(api.sessions) == before

    asyncio.run(scenario())


@pytest.mark.parametrize("method,suffix", [("GET", ""), ("POST", "/cancel"), ("POST", "/retry")])
def test_foreign_and_missing_rows_are_indistinguishable(database_api, method, suffix):
    async def scenario():
        async with database_api() as api:
            foreign_id = await _seed(api.sessions, user_id="user-b")
            before = await _rows(api.sessions)
            payload = {"json": {"idempotency_key": "retry-key"}} if suffix == "/retry" else {}
            for task_id in (foreign_id, MISSING_ID):
                response = await api.client.request(
                    method, f"/tasks/{task_id}{suffix}", params={"user_id": "user-b"}, **payload,
                )
                assert response.status_code == 404
                assert response.json() == {"detail": "任务不存在"}
            assert "commit" not in api.events
            assert await _rows(api.sessions) == before

    asyncio.run(scenario())


def test_list_and_detail_keep_scope_pagination_and_public_whitelist(database_api):
    async def scenario():
        async with database_api() as api:
            own_id = await _seed(api.sessions, key="SharedKey!")
            await _seed(api.sessions, key="other-key", updated_at=NOW + timedelta(seconds=1))
            await _seed(api.sessions, key="SharedKey!", user_id="user-b")
            for params in (
                {"idempotency_key": "SharedKey!", "user_id": "user-b"}, {"limit": 1, "offset": 1},
            ):
                response = await api.client.get("/tasks", params=params)
                assert response.status_code == 200
                page = response.json()["data"]
                assert set(page) == {"tasks", "limit", "offset"}
                assert [row["task_id"] for row in page["tasks"]] == [own_id]
                assert set(page["tasks"][0]) == PUBLIC_FIELDS
            missing = await api.client.get("/tasks", params={"idempotency_key": "sharedkey!"})
            assert missing.json()["data"]["tasks"] == []
            detail = await api.client.get(f"/tasks/{own_id}", params={"user_id": "user-b"})
            assert detail.status_code == 200 and set(detail.json()["data"]) == PUBLIC_FIELDS
            assert detail.json()["data"]["error_summary"] == "索引执行超时，请稍后重试"
            assert "private" not in detail.text and "SharedKey!" not in detail.text
            assert "commit" not in api.events

    asyncio.run(scenario())


@pytest.mark.parametrize("source_status", ["failed", "cancelled"])
def test_retry_replay_recovers_the_same_task(database_api, source_status):
    async def scenario():
        async with database_api() as api:
            source_id = await _seed(api.sessions, status=source_status)
            first = await _write(api, source_id, "retry")
            replay = await _write(api, source_id, "retry")
            assert first.status_code == replay.status_code == 200, replay.text
            task_id = first.json()["data"]["task_id"]
            assert replay.json()["data"]["task_id"] == task_id
            recovered = await api.client.get("/tasks", params={"idempotency_key": "retry-client-key"})
            assert [row["task_id"] for row in recovered.json()["data"]["tasks"]] == [task_id]
            assert len(await _rows(api.sessions)) == 2
            assert api.events.count("committed") == 2

    asyncio.run(scenario())


def test_retry_keys_are_user_scoped_case_sensitive_and_bound_to_source(database_api):
    async def scenario():
        async with database_api() as api:
            source_id = await _seed(api.sessions)
            other_id = await _seed(api.sessions, key="other-source")
            foreign_id = await _seed(api.sessions, user_id="user-b")
            first = await _write(api, source_id, "retry", "ClientKey!")
            different_case = await _write(api, source_id, "retry", "clientkey!")
            foreign = await _write(api, foreign_id, "retry", "ClientKey!", headers={"X-Test-User": "user-b"})
            assert first.status_code == different_case.status_code == foreign.status_code == 200
            assert len({item.json()["data"]["task_id"] for item in (first, different_case, foreign)}) == 3
            before = await _rows(api.sessions)
            conflict = await _write(api, other_id, "retry", "ClientKey!")
            assert conflict.status_code == 409
            assert conflict.json() == {"detail": "任务状态或幂等键与请求冲突"}
            assert await _rows(api.sessions) == before

    asyncio.run(scenario())


@pytest.mark.parametrize("operation,status", [("cancel", "succeeded"), ("retry", "processing")])
def test_state_conflicts_do_not_commit(database_api, operation, status):
    async def scenario():
        async with database_api() as api:
            task_id = await _seed(api.sessions, status=status)
            before = await _rows(api.sessions)
            response = await _write(api, task_id, operation)
            assert response.status_code == 409
            assert response.json() == {"detail": "任务状态或幂等键与请求冲突"}
            assert "commit" not in api.events and "rollback" in api.events
            assert await _rows(api.sessions) == before

    asyncio.run(scenario())
