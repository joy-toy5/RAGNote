from __future__ import annotations

import ast
import asyncio
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest


def _module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_agent_tools(
    monkeypatch: pytest.MonkeyPatch,
    source_path: Path,
) -> tuple[types.ModuleType, list[str]]:
    rag_user_ids: list[str] = []

    class RagService:
        def __init__(self, user_id: str, **_: object) -> None:
            rag_user_ids.append(user_id)

        async def get_documents_and_summary(self, _: str) -> dict[str, object]:
            await asyncio.sleep(0)
            return {"documents": [], "summary": "ok"}

    class Session:
        async def __aenter__(self) -> "Session":
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

    logger = logging.getLogger("m1-agent-tools")
    modules = {
        "app.core.logger_handler": _module("app.core.logger_handler", logger=logger),
        "app.rag.rag_service": _module("app.rag.rag_service", RagService=RagService),
        "app.services.note_service": _module(
            "app.services.note_service", note_service=object()
        ),
        "app.services.review_service": _module(
            "app.services.review_service", review_service=object()
        ),
        "app.db.db_config": _module(
            "app.db.db_config", AsyncSessionLocal=lambda: Session()
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "app.agent._m1_agent_tools"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, rag_user_ids


@pytest.mark.p0
def test_rag_tool_schema_cannot_accept_identity_or_jwt(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = _load_agent_tools(
        monkeypatch, backend_root / "app/agent/agent_tools.py"
    )
    assert set(module.rag_summary_tools.args_schema.model_fields) == {"query"}
    assert module.get_user_info_tools.args_schema.model_fields == {}


@pytest.mark.p0
def test_tool_context_is_required_and_reset(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, user_ids = _load_agent_tools(
        monkeypatch, backend_root / "app/agent/agent_tools.py"
    )

    with pytest.raises(RuntimeError, match="可信用户上下文"):
        asyncio.run(module.rag_summary_tools.ainvoke({"query": "query"}))

    with module.bind_tool_context(module.ToolContext(user_id="user-a")):
        result = asyncio.run(module.rag_summary_tools.ainvoke({"query": "query"}))
        assert "摘要: ok" in result

    with pytest.raises(RuntimeError, match="可信用户上下文"):
        module.require_tool_context()
    assert user_ids == ["user-a"]


@pytest.mark.p0
def test_tool_context_isolated_across_concurrent_tasks(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, user_ids = _load_agent_tools(
        monkeypatch, backend_root / "app/agent/agent_tools.py"
    )

    async def invoke(user_id: str) -> None:
        with module.bind_tool_context(module.ToolContext(user_id=user_id)):
            await module.rag_summary_tools.ainvoke({"query": "same query"})

    async def run_concurrent() -> None:
        await asyncio.gather(invoke("user-a"), invoke("user-b"))

    asyncio.run(run_concurrent())
    assert sorted(user_ids) == ["user-a", "user-b"]


@pytest.mark.p0
def test_agent_entrypoints_require_and_bind_identity(backend_root: Path) -> None:
    source = (backend_root / "app/agent/agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name in {"get_agent_response", "get_agent_stream_response"}
    }
    for function in functions.values():
        positional = [argument.arg for argument in function.args.args]
        assert "user_id" in positional
        defaults_start = len(positional) - len(function.args.defaults)
        user_index = positional.index("user_id")
        assert user_index < defaults_start

    assert source.count("bind_tool_context(") >= 2
    assert "set_current_user_id(" not in source
    assert 'kwargs.setdefault("max_iterations", 8)' in source
    assert 'kwargs.setdefault("max_execution_time", 60)' in source
