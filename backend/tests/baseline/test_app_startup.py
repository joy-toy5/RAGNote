from __future__ import annotations

import asyncio
import builtins
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

DANGEROUS_IMPORTS = {
    "chromadb",
    "langchain_chroma",
    "modelscope",
    "sentence_transformers",
}


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _router(name: str) -> APIRouter:
    router = APIRouter(prefix=f"/_m0/{name}")

    @router.get("/probe")
    async def probe() -> dict[str, str]:
        return {"router": name}

    return router


def _load_isolated_app(
    monkeypatch: pytest.MonkeyPatch,
    main_path: Path,
) -> tuple[types.ModuleType, list[str]]:
    events: list[str] = []

    async def check_database_schema() -> None:
        events.append("check_database_schema")

    async def init_sessions() -> None:
        events.append("init_sessions")

    async def connect_redis() -> None:
        events.append("connect_redis")

    async def close_redis() -> None:
        events.append("close_redis")

    class Engine:
        async def dispose(self):
            events.append("dispose_database")

    class TaskRegistry:
        def start(self) -> None:
            events.append("start_tasks")

        def stop_accepting(self):
            events.append("stop_tasks")

        async def drain(self, *, timeout: float):
            assert timeout > 0
            events.append("drain_tasks")
            return ()

        async def cancel_and_wait(self, *, timeout: float):
            assert timeout > 0
            events.append("cancel_tasks")
            return ()

    class UploadRuntime:
        def start(self) -> None:
            events.append("start_uploads")

        def stop_accepting(self):
            events.append("stop_upload_admission")

        async def shutdown(self, *, timeout: float):
            assert timeout > 0
            events.append("stop_uploads")
            return 0

    def check_reranker() -> None:
        events.append("check_reranker")

    def initialize_rag_resources() -> None:
        events.append("initialize_rag_resources")

    def register_handlers(_: object) -> None:
        events.append("register_handlers")

    def validate_rate_limit_config() -> None:
        events.append("validate_rate_limit")

    def validate_auth_config() -> None:
        events.append("validate_auth")

    class PassthroughRateLimitMiddleware:
        def __init__(self, app: object, **_: object) -> None:
            self.app = app

        async def __call__(self, scope: object, receive: object, send: object) -> None:
            await self.app(scope, receive, send)

    def load_dotenv() -> None:
        events.append("load_dotenv")

    logger = logging.Logger("m0-isolated-startup")
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False

    modules = {
        "app": _package("app"),
        "app.db": _package("app.db"),
        "app.router": _package("app.router"),
        "app.services": _package("app.services"),
        "app.core": _package("app.core"),
        "app.rag": _package("app.rag"),
        "app.utils": _package("app.utils"),
        "app.core.task_registry": _module("app.core.task_registry", background_tasks=TaskRegistry()),
        "app.rag.upload_runtime": _module("app.rag.upload_runtime", upload_runtime=UploadRuntime()),
        "app.db.db_config": _module(
            "app.db.db_config",
            check_database_schema=check_database_schema,
            async_engine=Engine(),
        ),
        "app.db.redis_config": _module(
            "app.db.redis_config",
            connect_redis=connect_redis,
            close_redis=close_redis,
        ),
        "app.services.database_session_manager": _module(
            "app.services.database_session_manager",
            init_database_session_manager=init_sessions,
        ),
        "app.core.failed_response_register": _module(
            "app.core.failed_response_register",
            register_exception_handlers=register_handlers,
        ),
        "app.core.rate_limit": _module(
            "app.core.rate_limit",
            RateLimitMiddleware=PassthroughRateLimitMiddleware,
            validate_rate_limit_config=validate_rate_limit_config,
        ),
        "app.core.logger_handler": _module("app.core.logger_handler", logger=logger),
        "app.utils.auth_utils": _module(
            "app.utils.auth_utils",
            get_cors_origins=lambda: ["http://localhost:5173"],
            validate_auth_config=validate_auth_config,
        ),
        "app.rag.bootstrap": _module(
            "app.rag.bootstrap", initialize_rag_resources=initialize_rag_resources,
        ),
        "app.rag.reorder_service": _module(
            "app.rag.reorder_service",
            check_and_download_reranker_model=check_reranker,
        ),
        "dotenv": _module("dotenv", load_dotenv=load_dotenv),
    }
    router_names = (
        ("chat", "chat_router"),
        ("knowledge_router", "knowledge_router"),
        ("health", "health_router"),
        ("user", "user_router"),
        ("note_router", "note_router"),
        ("review_router", "review_router"),
        ("task_router", "task_router"),
    )
    for module_name, attribute_name in router_names:
        modules[f"app.router.{module_name}"] = _module(
            f"app.router.{module_name}",
            **{attribute_name: _router(module_name)},
        )

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_m0_isolated_main"
    spec = importlib.util.spec_from_file_location(module_name, main_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, events


@pytest.mark.baseline
def test_fastapi_app_assembles_without_real_dependencies(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted_dangerous_imports: list[str] = []
    original_import = builtins.__import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        if any(
            name == target or name.startswith(f"{target}.")
            for target in DANGEROUS_IMPORTS
        ):
            attempted_dangerous_imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracking_import)
    module, events = _load_isolated_app(monkeypatch, backend_root / "main.py")

    assert events == ["load_dotenv", "register_handlers"]

    expected_probes = {
        "/_m0/chat/probe",
        "/_m0/knowledge_router/probe",
        "/_m0/health/probe",
        "/_m0/user/probe",
        "/_m0/note_router/probe",
        "/_m0/review_router/probe",
        "/_m0/task_router/probe",
    }
    assert expected_probes.issubset(
        {route.path for route in module.app.routes if hasattr(route, "path")}
    )

    with TestClient(module.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello World"}
        assert "X-Process-Time" in response.headers
        assert events == [
            "load_dotenv",
            "register_handlers",
            "validate_rate_limit",
            "validate_auth",
            "check_database_schema",
            "init_sessions",
            "connect_redis",
            "check_reranker",
            "initialize_rag_resources",
            "start_tasks",
            "start_uploads",
        ]

    assert events == [
        "load_dotenv",
        "register_handlers",
        "validate_rate_limit",
        "validate_auth",
        "check_database_schema",
        "init_sessions",
        "connect_redis",
        "check_reranker",
        "initialize_rag_resources",
        "start_tasks",
        "start_uploads",
        "stop_tasks",
        "stop_upload_admission",
        "drain_tasks",
        "stop_uploads",
        "close_redis",
        "dispose_database",
    ]
    assert attempted_dangerous_imports == []


@pytest.mark.baseline
@pytest.mark.parametrize("phase", ["check_database_schema", "initialize_rag_resources"])
def test_failed_startup_does_not_open_task_or_upload_admission(
    backend_root: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    module, events = _load_isolated_app(monkeypatch, backend_root / "main.py")

    def fail() -> None:
        events.append("failed")
        raise RuntimeError("合成启动失败")

    async def fail_async() -> None:
        fail()

    monkeypatch.setattr(module, phase, fail_async if phase == "check_database_schema" else fail)
    with pytest.raises(RuntimeError, match="合成启动失败"):
        asyncio.run(module.startup_event())
    assert "start_tasks" not in events
    assert "start_uploads" not in events
    if phase == "check_database_schema":
        assert "initialize_rag_resources" not in events


def test_shutdown_shares_drain_then_cancels_and_closes_resources(backend_root, monkeypatch):
    module, events = _load_isolated_app(monkeypatch, backend_root / "main.py")
    events.clear()

    async def pending_tasks(*, timeout):
        assert timeout == module.SHUTDOWN_GRACE_SECONDS
        events.append("drain_tasks")
        return ("synthetic-pending-task",)

    monkeypatch.setattr(module.background_tasks, "drain", pending_tasks)

    async def scenario():
        draining = module.begin_shutdown()
        assert module.begin_shutdown() is draining
        await module.shutdown_event()
        assert draining.done()

    asyncio.run(scenario())
    assert events.count("drain_tasks") == 1
    assert events[-5:] == [
        "drain_tasks", "cancel_tasks", "stop_uploads", "close_redis", "dispose_database",
    ]
