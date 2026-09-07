"""Demo上传只验证持久记录和连接独立性，不使用真实模型/Chroma。"""
from __future__ import annotations

import asyncio
import importlib
import io
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.requests import ClientDisconnect

from app.core.task_registry import TaskRegistry
from app.rag.upload_runtime import UploadRuntime
from app.schemas.task import TaskResponse
from app.tasking import repository
from app.tasking.errors import TaskNotFound
from test_task_repository import _database
from test_upload_response_lifecycle import _load_router

USER = "demo-user-a"


def _event(kind, **values):
    return 'event: progress\ndata: ' + json.dumps(dict(event_type=kind, **values)) + '\n\n'


class UploadService:
    def __init__(self, *, failure=False):
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.failure = failure
        self.prepared = None
        self.files = None

    async def _validate_and_read_files(self, files):
        self.files = list(files)
        content = await files[0].read()
        return [dict(content=content, filename="demo.txt", file_index=1, media_type="text/plain")], [], 1

    async def handle_add_vector_multiple_stream(self, files, user_id, *, prepared, lease):
        assert files == [] and user_id == USER
        self.prepared = prepared
        self.started.set()
        yield _event("start", total_files=1)
        await self.release.wait()
        assert self.prepared[0][0]["content"] == b"demo input"
        if self.failure:
            raise RuntimeError("synthetic-private-error")
        yield _event("finish", total_files=1, success_count=1, failed_count=0, progress=100)


@asynccontextmanager
async def _harness(tmp_path, backend_root, monkeypatch, *, failure=False, session_class=None):
    module = importlib.import_module("app.tasking.upload")
    registry = TaskRegistry()
    runtime = UploadRuntime()
    service = UploadService(failure=failure)
    file = UploadFile(io.BytesIO(b"demo input"), filename="demo.txt")
    async with _database(tmp_path, backend_root) as (sessions, engine):
        if session_class is not None:
            sessions = async_sessionmaker(engine, class_=session_class, expire_on_commit=False)
        monkeypatch.setattr(module, "background_tasks", registry)
        monkeypatch.setattr(module, "upload_runtime", runtime)
        monkeypatch.setattr(module, "_session_factory", lambda: sessions)
        try:
            yield module, service, file, sessions, registry, runtime
        finally:
            service.release.set()
            await registry.cancel_and_wait(timeout=1)
            await runtime.shutdown(timeout=1)
            await file.close()


def test_disconnect_does_not_cancel_upload_and_result_remains_queryable(tmp_path, backend_root, monkeypatch):
    async def scenario():
        async with _harness(tmp_path, backend_root, monkeypatch) as (module, service, file, sessions, registry, runtime):
            task_id, events = await module.submit_upload(service, [file], USER)
            first = json.loads((await anext(events)).split("data: ", 1)[1])
            assert first["task_id"] == task_id
            assert first["event_type"] == "start"
            async with sessions() as db:
                assert (await repository.get_task(db, task_id, USER)).status == "processing"
            await events.aclose()
            await file.close()
            assert registry.tasks and runtime.active_count == 1
            service.release.set()
            assert await registry.drain(timeout=1) == frozenset()
            async with sessions() as db:
                task = await repository.get_task(db, task_id, USER)
                assert task.status == "succeeded" and task.progress == 100
                assert task.started_at and task.completed_at
                assert task.result_ref == "/knowledge/list"
                assert TaskResponse.model_validate(task).result_url == "/knowledge/list"
                with pytest.raises(TaskNotFound):
                    await repository.get_task(db, task_id, "demo-user-b")
            assert runtime.active_count == 0
    asyncio.run(scenario())


def test_upload_failure_is_persisted_without_private_exception(tmp_path, backend_root, monkeypatch):
    async def scenario():
        async with _harness(tmp_path, backend_root, monkeypatch, failure=True) as (module, service, file, sessions, registry, _):
            task_id, events = await module.submit_upload(service, [file], USER)
            service.release.set()
            output = [event async for event in events]
            assert await registry.drain(timeout=1) == frozenset()
            async with sessions() as db:
                task = await repository.get_task(db, task_id, USER)
                assert task.status == "failed" and task.completed_at
                assert task.error_code == "UPLOAD_FAILED"
                assert "synthetic-private-error" not in task.error_summary
                assert "synthetic-private-error" not in ''.join(output)
    asyncio.run(scenario())


def test_request_cancel_during_commit_does_not_abandon_the_task(tmp_path, backend_root, monkeypatch):
    async def scenario():
        entered, commit_allowed = asyncio.Event(), asyncio.Event()
        class DelayedSession(AsyncSession):
            async def commit(self):
                if not entered.is_set():
                    entered.set()
                    await commit_allowed.wait()
                await super().commit()
        async with _harness(tmp_path, backend_root, monkeypatch, session_class=DelayedSession) as (module, service, file, sessions, registry, _):
            request = asyncio.create_task(module.submit_upload(service, [file], USER))
            await asyncio.wait_for(entered.wait(), timeout=1)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            commit_allowed.set()
            service.release.set()
            assert await registry.drain(timeout=1) == frozenset()
            async with sessions() as db:
                tasks = await repository.list_tasks(db, USER)
                assert len(tasks) == 1 and tasks[0].status == "succeeded"
    asyncio.run(scenario())


def test_shutdown_timeout_records_failure_and_rejects_new_uploads(tmp_path, backend_root, monkeypatch):
    async def scenario():
        async with _harness(tmp_path, backend_root, monkeypatch) as (module, service, file, sessions, registry, runtime):
            task_id, events = await module.submit_upload(service, [file], USER)
            await anext(events)
            registry.stop_accepting()
            with pytest.raises(HTTPException) as error:
                await module.submit_upload(service, [file], USER)
            assert error.value.status_code == 503
            assert await registry.drain(timeout=0)
            assert await registry.cancel_and_wait(timeout=1) == frozenset()
            async with sessions() as db:
                task = await repository.get_task(db, task_id, USER)
                assert task.status == "failed" and task.error_code == "SHUTDOWN_TIMEOUT"
            await events.aclose()
            assert runtime.active_count == 0
    asyncio.run(scenario())


def test_failed_admission_commit_never_runs_upload(tmp_path, backend_root, monkeypatch):
    async def scenario():
        class BrokenSession(AsyncSession):
            async def commit(self):
                raise RuntimeError("synthetic commit failure")
        async with _harness(tmp_path, backend_root, monkeypatch, session_class=BrokenSession) as (module, service, file, sessions, registry, runtime):
            with pytest.raises(HTTPException) as error:
                await module.submit_upload(service, [file], USER)
            assert error.value.status_code == 503
            assert await registry.drain(timeout=1) == frozenset()
            assert not service.started.is_set() and runtime.active_count == 0
            async with sessions() as db:
                assert await repository.list_tasks(db, USER) == []
    asyncio.run(scenario())


@pytest.mark.parametrize("with_middleware", [False, True])
@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_http_send_failure_keeps_recorded_upload_running(
    tmp_path, backend_root, monkeypatch, with_middleware, spec_version,
):
    async def scenario():
        async with _harness(tmp_path, backend_root, monkeypatch) as (_, service, _, sessions, registry, runtime):
            service_module = SimpleNamespace(KnowledgeService=type(service), get_knowledge_service=lambda: service)
            router = _load_router(monkeypatch, backend_root, service_module, recorded_uploads=True)
            app = FastAPI()
            app.include_router(router.knowledge_router)
            app.dependency_overrides[router.get_current_user_id] = lambda: USER
            if with_middleware:
                @app.middleware("http")
                async def timing_header(request, call_next):
                    response = await call_next(request)
                    response.headers["X-Test-Time"] = "0"
                    return response

            request = httpx.Request(
                "POST", "http://test/knowledge/add/multiple/stream",
                files=[("files", ("demo.txt", b"demo input", "text/plain"))],
            )
            body = request.read()
            received = False
            response_headers = {}
            first_event = None

            async def receive():
                nonlocal received
                if not received:
                    received = True
                    return {"type": "http.request", "body": body, "more_body": False}
                await asyncio.Event().wait()

            async def send(message):
                nonlocal first_event
                if message["type"] == "http.response.start":
                    assert message["status"] == 200
                    response_headers.update(message["headers"])
                if message["type"] == "http.response.body" and message.get("body"):
                    first_event = json.loads(message["body"].decode().split("data: ", 1)[1])
                    raise OSError("受控SSE发送断开")

            scope = {
                "type": "http", "asgi": {"version": "3.0", "spec_version": spec_version},
                "http_version": "1.1", "method": "POST", "scheme": "http",
                "path": "/knowledge/add/multiple/stream", "raw_path": b"/knowledge/add/multiple/stream",
                "query_string": b"", "root_path": "",
                "headers": [(name.lower(), value) for name, value in request.headers.raw],
                "client": ("127.0.0.1", 1), "server": ("test", 80),
            }
            async with asyncio.timeout(2):
                with pytest.raises((OSError, ClientDisconnect)):
                    await app(scope, receive, send)
                task_id = response_headers[b"x-task-id"].decode()
                assert first_event["event_type"] == "start" and first_event["task_id"] == task_id
                # HTTP已关闭UploadFile；执行只依赖接单时取得的bytes，不能靠测试善后补做。
                assert service.files[0].file.closed
                assert registry.tasks and runtime.active_count == 1
                async with sessions() as db:
                    assert (await repository.get_task(db, task_id, USER)).status == "processing"
                service.release.set()
                assert await registry.drain(timeout=1) == frozenset()
                async with sessions() as db:
                    task = await repository.get_task(db, task_id, USER)
                    assert task.status == "succeeded"
                    assert TaskResponse.model_validate(task).result_url == "/knowledge/list"
                assert runtime.active_count == 0
    asyncio.run(scenario())


def test_session_factory_failure_releases_capacity_without_hanging(tmp_path, backend_root, monkeypatch):
    async def scenario():
        async with _harness(tmp_path, backend_root, monkeypatch) as (module, service, file, sessions, registry, runtime):
            def broken_factory():
                raise RuntimeError("synthetic session factory failure")

            monkeypatch.setattr(module, "_session_factory", broken_factory)
            with pytest.raises(HTTPException) as error:
                await asyncio.wait_for(module.submit_upload(service, [file], USER), timeout=1)
            assert error.value.status_code == 503
            assert await registry.drain(timeout=1) == frozenset()
            assert not service.started.is_set() and runtime.active_count == 0
            async with sessions() as db:
                assert await repository.list_tasks(db, USER) == []
    asyncio.run(scenario())


def test_shutdown_during_file_read_rejects_admission_and_releases_capacity(tmp_path, backend_root, monkeypatch):
    async def scenario():
        async with _harness(tmp_path, backend_root, monkeypatch) as (module, service, file, sessions, registry, runtime):
            validate = service._validate_and_read_files

            async def read_during_shutdown(files):
                prepared = await validate(files)
                registry.stop_accepting()
                return prepared

            monkeypatch.setattr(service, "_validate_and_read_files", read_during_shutdown)
            with pytest.raises(HTTPException) as error:
                await module.submit_upload(service, [file], USER)
            assert error.value.status_code == 503
            assert not registry.tasks and not service.started.is_set()
            assert runtime.active_count == 0
            async with sessions() as db:
                assert await repository.list_tasks(db, USER) == []
    asyncio.run(scenario())
