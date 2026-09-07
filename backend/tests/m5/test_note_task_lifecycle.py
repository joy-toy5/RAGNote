from __future__ import annotations

import __future__
import ast
import asyncio
import gc
import inspect
import json
import logging
import sys
import types
import uuid
import weakref
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse

pytestmark = pytest.mark.p0
USER_ID = "offline-user"
CONTENT = "仅用于离线生命周期验证的合成笔记"
LOGGER = logging.getLogger("m5-note-task-lifecycle")


class Field:
    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, value: object) -> tuple[str, object]:
        return self.name, value


class Note(SimpleNamespace):
    id = Field("id")
    user_id = Field("user_id")

    def __init__(self, **values: object) -> None:
        super().__init__(
            tags=None, category=None, created_at=None, updated_at=None, **values
        )


class Statement:
    def __init__(self, model: object) -> None:
        self.model = model
        self.predicates: tuple = ()
        self.updates: dict = {}

    def where(self, *predicates: object) -> Statement:
        self.predicates = predicates
        return self

    def values(self, **values: object) -> Statement:
        self.updates = values
        return self


class RequestSession:
    def __init__(self) -> None:
        self.note = Note(
            id="offline-note", user_id=USER_ID, title="离线笔记", content=CONTENT
        )
        self.commit = AsyncMock()
        self.refresh = AsyncMock()

    def add(self, note: Note) -> None:
        self.note = note

    async def execute(self, statement: Statement) -> SimpleNamespace:
        note = self.note
        if note is not None:
            assert statement.predicates == (("id", note.id), ("user_id", USER_ID))
        return SimpleNamespace(scalar_one_or_none=lambda: note)


class BackgroundSession:
    def __init__(self, owner: SessionFactory) -> None:
        self.owner = owner
        self.events: list[str] = []
        self.statement: Statement | None = None
        self.review: SimpleNamespace | None = None
        self.exit_type: type | None = None
        self.commits = 0

    async def __aenter__(self) -> BackgroundSession:
        self.events.append("enter")
        return self

    async def __aexit__(self, exc_type: type | None, *_: object) -> None:
        self.exit_type = exc_type
        self.events.append("exit")

    async def _operation(self, phase: str) -> None:
        self.events.append(phase)
        if self.owner.block_at == phase:
            self.owner.started.set()
            await self.owner.release.wait()
        if self.owner.fail_at == phase:
            raise RuntimeError("合成后台事务失败")

    async def execute(self, statement: Statement) -> None:
        self.statement = statement
        await self._operation("execute")

    def add(self, review: SimpleNamespace) -> None:
        self.events.append("add")
        self.review = review

    async def commit(self) -> None:
        await self._operation("commit")
        self.commits += 1


class SessionFactory:
    def __init__(self) -> None:
        self.sessions: list[BackgroundSession] = []
        self.block_at: str | None = None
        self.fail_at: str | None = None
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def __call__(self) -> BackgroundSession:
        session = BackgroundSession(self)
        self.sessions.append(session)
        return session


class RecordingRegistry:
    """仅记录调用；调度、引用持有、异常回收与取消均由真实登记器完成。"""

    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.calls: list[SimpleNamespace] = []

    @property
    def tasks(self) -> frozenset[asyncio.Task]:
        return self.delegate.tasks

    def create(self, coro: object, *, name: str) -> asyncio.Task:
        self.calls.append(SimpleNamespace(coro=coro, name=name))
        return self.delegate.create(coro, name=name)

    async def cancel_and_wait(self, *, timeout: float) -> None:
        pending = await self.delegate.cancel_and_wait(timeout=timeout)
        assert not pending, "测试清理时仍有未退出的后台任务"


def _module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _compile_nodes(path: Path, nodes: list[ast.stmt], namespace: dict) -> None:
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    code = compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag)
    exec(code, namespace)


def _load_service(path: Path) -> object:
    """保留真实服务控制流与 registry 导入，禁止执行生产模块初始化。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.ClassDef) and node.name == "NoteService")
        or (
            isinstance(node, ast.ImportFrom) and node.module == "app.core.task_registry"
        )
    ]
    namespace = {
        "asyncio": asyncio,
        "uuid": uuid,
        "json": json,
        "datetime": datetime,
        "timedelta": timedelta,
        "logger": LOGGER,
        "Note": Note,
        "NoteResponse": SimpleNamespace,
        "ReviewRecord": SimpleNamespace,
        "Document": SimpleNamespace,
        "HumanMessage": SimpleNamespace,
        "select": Statement,
        "update": Statement,
        "load_prompt": lambda _: "内容：{content}",
    }
    _compile_nodes(path, nodes, namespace)
    service = object.__new__(namespace["NoteService"])
    service._notes_store = SimpleNamespace(add_documents=Mock())
    return service


def _load_router(path: Path, service: object) -> SimpleNamespace:
    """只执行两个真实入口，移除装饰器和会触发鉴权/数据库导入的默认值。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name in {"create_note", "regenerate_tags"}
    ]
    for node in nodes:
        node.decorator_list = []
        node.args.defaults = [ast.Constant(value=None) for _ in node.args.defaults]
    namespace = {
        "note_service": service,
        "success_response": lambda **values: values,
        "HTTPException": HTTPException,
    }
    _compile_nodes(path, nodes, namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, backend_root: Path) -> SimpleNamespace:
    # 每个测试独立执行登记器源码，绝不关闭其他测试或生产模块的全局实例。
    registry_path = backend_root / "app/core/task_registry.py"
    registry_module = _module("app.core.task_registry")
    exec(
        compile(registry_path.read_text(encoding="utf-8"), str(registry_path), "exec"),
        registry_module.__dict__,
    )
    registry = RecordingRegistry(registry_module.background_tasks)
    registry_module.background_tasks = registry
    sessions = SessionFactory()
    model = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value=SimpleNamespace(
                content='{"tags": ["离线"], "category": "study"}'
            )
        )
    )
    for name in ("app", "app.core", "app.db", "app.utils"):
        package = _module(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    modules = {
        "app.core.task_registry": registry_module,
        "app.utils.factory": _module("app.utils.factory", get_chat_model=lambda: model),
        "app.db.db_config": _module("app.db.db_config", AsyncSessionLocal=sessions),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    service = _load_service(backend_root / "app/services/note_service.py")
    router = _load_router(backend_root / "app/router/note_router.py", service)
    return SimpleNamespace(
        registry=registry,
        sessions=sessions,
        model=model,
        service=service,
        router=router,
        request=RequestSession(),
        payload=SimpleNamespace(title="离线笔记", content=CONTENT),
    )


async def _invoke(harness: SimpleNamespace, entrypoint: str) -> dict:
    if entrypoint == "create":
        return await harness.router.create_note(
            payload=harness.payload, user_id=USER_ID, db=harness.request
        )
    return await harness.router.regenerate_tags(
        note_id=harness.request.note.id, user_id=USER_ID, db=harness.request
    )


@pytest.mark.parametrize("entrypoint", ["create", "manual"])
def test_both_entrypoints_register_through_one_service_method(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, entrypoint: str
) -> None:
    async def scenario() -> None:
        schedule = Mock(wraps=harness.service.schedule_auto_tag)
        monkeypatch.setattr(harness.service, "schedule_auto_tag", schedule)
        try:
            response = await _invoke(harness, entrypoint)
            note_id = harness.request.note.id
            schedule.assert_called_once_with(note_id, USER_ID, CONTENT)
            assert len(harness.registry.calls) == 1
            call = harness.registry.calls[0]
            assert note_id in call.name
            assert call.name
            assert len(harness.registry.tasks) == 1
            assert response["message"] == (
                "笔记创建成功" if entrypoint == "create" else "标签生成任务已提交"
            )
            assert harness.request.commit.await_count == (entrypoint == "create")
            assert harness.service.notes_store.add_documents.call_count == (
                entrypoint == "create"
            )
        finally:
            await harness.registry.cancel_and_wait(timeout=1)

    asyncio.run(scenario())


def test_committed_note_returns_before_background_model_failure(
    harness: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    async def scenario() -> None:
        started, release = asyncio.Event(), asyncio.Event()

        async def fail_after_response(*_: object) -> None:
            started.set()
            await release.wait()
            raise RuntimeError("合成模型失败")

        harness.model.ainvoke.side_effect = fail_after_response
        try:
            response = await asyncio.wait_for(_invoke(harness, "create"), timeout=1)
            assert response["data"].id == harness.request.note.id
            harness.request.commit.assert_awaited_once()
            harness.request.refresh.assert_awaited_once()
            assert len(harness.registry.tasks) == 1
            await asyncio.wait_for(started.wait(), timeout=1)
            assert harness.sessions.sessions == []
            tasks = tuple(harness.registry.tasks)
            release.set()
            await asyncio.gather(*tasks)
            await asyncio.sleep(0)
            assert not harness.registry.tasks
            assert "合成模型失败" in caplog.text
            harness.model.ainvoke.assert_awaited_once()
            assert response["message"] == "笔记创建成功"
            harness.request.commit.assert_awaited_once()
        finally:
            await harness.registry.cancel_and_wait(timeout=1)

    asyncio.run(scenario())


def test_tags_and_review_share_one_independent_transaction(
    harness: SimpleNamespace,
) -> None:
    async def scenario() -> None:
        try:
            task = harness.service.schedule_auto_tag("offline-note", USER_ID, CONTENT)
            assert isinstance(task, asyncio.Task)
            await task
            await asyncio.sleep(0)
            (session,) = harness.sessions.sessions
            assert session is not harness.request
            assert session.events == ["enter", "execute", "add", "commit", "exit"]
            assert session.commits == 1
            assert session.exit_type is None
            assert session.statement.predicates == (
                ("id", "offline-note"),
                ("user_id", USER_ID),
            )
            assert session.statement.updates == {"tags": ["离线"], "category": "study"}
            assert session.review.note_id == "offline-note"
            assert session.review.user_id == USER_ID
            assert session.review.interval_days == 1
            assert session.review.review_count == 0
            assert not harness.registry.tasks
            harness.request.commit.assert_not_awaited()
        finally:
            await harness.registry.cancel_and_wait(timeout=1)

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["execute", "commit"])
def test_background_failure_exits_session_without_retry(
    harness: SimpleNamespace, caplog: pytest.LogCaptureFixture, phase: str
) -> None:
    async def scenario() -> None:
        harness.sessions.fail_at = phase
        try:
            await harness.service.schedule_auto_tag("offline-note", USER_ID, CONTENT)
            await asyncio.sleep(0)
            (session,) = harness.sessions.sessions
            assert session.events[-1] == "exit"
            assert session.exit_type is RuntimeError
            assert session.commits == 0
            assert session.events.count(phase) == 1
            assert "自动标签后台任务失败" in caplog.text
            assert "合成后台事务失败" in caplog.text
            assert len(harness.registry.calls) == 1
            assert not harness.registry.tasks
            harness.model.ainvoke.assert_awaited_once()
            harness.request.commit.assert_not_awaited()
        finally:
            await harness.registry.cancel_and_wait(timeout=1)

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["execute", "commit"])
def test_cancellation_exits_independent_session(
    harness: SimpleNamespace, caplog: pytest.LogCaptureFixture, phase: str
) -> None:
    async def scenario() -> None:
        harness.sessions.block_at = phase
        try:
            task = harness.service.schedule_auto_tag("offline-note", USER_ID, CONTENT)
            await asyncio.wait_for(harness.sessions.started.wait(), timeout=1)
            await harness.registry.cancel_and_wait(timeout=1)
            (session,) = harness.sessions.sessions
            assert task.cancelled()
            assert session.exit_type is asyncio.CancelledError
            assert session.events[-1] == "exit"
            assert session.commits == 0
            assert not harness.registry.tasks
            assert "后台任务执行失败" not in caplog.text
            harness.request.commit.assert_not_awaited()
            harness.model.ainvoke.assert_awaited_once()
        finally:
            await harness.registry.cancel_and_wait(timeout=1)

    asyncio.run(scenario())


def test_unexpected_worker_exception_is_collected_by_registry(
    harness: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        worker = AsyncMock(side_effect=RuntimeError("合成未捕获异常"))
        monkeypatch.setattr(harness.service, "_auto_tag_and_review", worker)
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        unhandled: list[dict] = []
        loop.set_exception_handler(lambda _, context: unhandled.append(context))
        try:
            task = harness.service.schedule_auto_tag("offline-note", USER_ID, CONTENT)
            # wait 不取走任务异常：异常必须由真实登记器的完成回调回收。
            await asyncio.wait({task}, timeout=1)
            await asyncio.sleep(0)
            assert task.done()
            assert not harness.registry.tasks
            task_ref = weakref.ref(task)
            del task
            gc.collect()
            assert task_ref() is None
            assert unhandled == []
            assert "后台任务执行失败" in caplog.text
            worker.assert_awaited_once_with("offline-note", USER_ID, CONTENT)
            assert len(harness.registry.calls) == 1
        finally:
            await harness.registry.cancel_and_wait(timeout=1)
            loop.set_exception_handler(previous_handler)

    asyncio.run(scenario())


def test_fire_and_forget_task_is_held_until_independent_session_exits(
    harness: SimpleNamespace,
) -> None:
    async def scenario() -> None:
        harness.sessions.block_at = "commit"
        try:
            await _invoke(harness, "manual")
            await asyncio.wait_for(harness.sessions.started.wait(), timeout=1)
            (task,) = harness.registry.tasks
            task_ref = weakref.ref(task)
            del task
            gc.collect()
            assert task_ref() is not None
            harness.sessions.release.set()
            await asyncio.wait({task_ref()}, timeout=1)
            await asyncio.sleep(0)
            gc.collect()
            assert not harness.registry.tasks
            assert task_ref() is None
            (session,) = harness.sessions.sessions
            assert session.events[-1] == "exit"
            assert session.commits == 1
        finally:
            await harness.registry.cancel_and_wait(timeout=1)

    asyncio.run(scenario())


@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
def test_repeated_scheduling_is_two_attempts_without_retry_or_deduplication(
    harness: SimpleNamespace, fail: bool
) -> None:
    async def scenario() -> None:
        if fail:
            harness.model.ainvoke.side_effect = RuntimeError("合成模型失败")
        try:
            tasks = [
                harness.service.schedule_auto_tag("offline-note", USER_ID, CONTENT)
                for _ in range(2)
            ]
            assert tasks[0] is not tasks[1]
            assert len(harness.registry.tasks) == 2
            await asyncio.gather(*tasks)
            await asyncio.sleep(0)
            assert len(harness.registry.calls) == 2
            assert harness.model.ainvoke.await_count == 2
            assert not harness.registry.tasks
            assert len(harness.sessions.sessions) == (0 if fail else 2)
            assert sum(session.commits for session in harness.sessions.sessions) == (
                0 if fail else 2
            )
        finally:
            await harness.registry.cancel_and_wait(timeout=1)

    asyncio.run(scenario())


@pytest.mark.parametrize("entrypoint", ["create", "manual"])
def test_shutdown_rejection_is_truthful_and_closes_unscheduled_coroutine(
    harness: SimpleNamespace, caplog: pytest.LogCaptureFixture, entrypoint: str
) -> None:
    async def scenario() -> None:
        await harness.registry.cancel_and_wait(timeout=1)
        caplog.set_level(logging.WARNING, logger=LOGGER.name)
        if entrypoint == "create":
            response = await _invoke(harness, entrypoint)
            assert response["message"] == "笔记创建成功"
            assert response["data"].id == harness.request.note.id
            harness.request.commit.assert_awaited_once()
            assert harness.request.note.id in caplog.text
            assert "未调度" in caplog.text
        else:
            with pytest.raises(HTTPException) as exc_info:
                await _invoke(harness, entrypoint)
            assert exc_info.value.status_code == 503
            assert "未提交" in exc_info.value.detail
            harness.request.commit.assert_not_awaited()
        assert len(harness.registry.calls) == 1
        coro = harness.registry.calls[0].coro
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
        assert not harness.registry.tasks
        assert harness.sessions.sessions == []
        harness.model.ainvoke.assert_not_awaited()

    asyncio.run(scenario())


def test_manual_rejection_preserves_503_in_http_response(
    harness: SimpleNamespace, backend_root: Path
) -> None:
    # 全量导入此模块会初始化 .env 配置；只提取真实 HTTP 异常处理函数。
    path = backend_root / "app/core/failed_response.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "http_exception_handler"
    ]
    namespace = {"JSONResponse": JSONResponse, "logger": LOGGER}
    _compile_nodes(path, nodes, namespace)

    async def scenario() -> None:
        await harness.registry.cancel_and_wait(timeout=1)
        with pytest.raises(HTTPException) as exc_info:
            await _invoke(harness, "manual")
        response = await namespace["http_exception_handler"](
            SimpleNamespace(url="/note/offline-note/auto-tag", method="POST"),
            exc_info.value,
        )
        assert response.status_code == 503
        body = json.loads(response.body)
        assert body["code"] == 503
        assert "未提交" in body["message"]
        assert "已提交" not in body["message"]

    asyncio.run(scenario())


def test_missing_note_does_not_schedule_during_shutdown(
    harness: SimpleNamespace,
) -> None:
    async def scenario() -> None:
        harness.request.note = None
        await harness.registry.cancel_and_wait(timeout=1)
        response = await harness.router.regenerate_tags(
            note_id="absent-note", user_id=USER_ID, db=harness.request
        )
        assert response == {"message": "笔记不存在"}
        assert harness.registry.calls == []
        harness.model.ainvoke.assert_not_awaited()

    asyncio.run(scenario())


def test_body_commit_failure_does_not_schedule(harness: SimpleNamespace) -> None:
    async def scenario() -> None:
        harness.request.commit.side_effect = RuntimeError("合成正文提交失败")
        try:
            with pytest.raises(RuntimeError, match="合成正文提交失败"):
                await _invoke(harness, "create")
            assert harness.registry.calls == []
            harness.request.refresh.assert_not_awaited()
            harness.service.notes_store.add_documents.assert_not_called()
            harness.model.ainvoke.assert_not_awaited()
        finally:
            await harness.registry.cancel_and_wait(timeout=1)

    asyncio.run(scenario())
