"""笔记存储只在显式启动阶段绑定，导入和服务构造不打开客户端。"""
from __future__ import annotations

import importlib.util
import logging
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


@pytest.fixture
def isolated_notes(backend_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[dict] = []
    control = SimpleNamespace(failure=None, entered=None, release=None)

    class Chroma:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if control.entered is not None:
                control.entered.set()
                assert control.release.wait(2), "合成初始化没有被及时放行"
            if control.failure is not None:
                raise control.failure
            self._client = kwargs.get("client", object())

    dependency = type("Dependency", (), {})
    modules = {
        "langchain_chroma": _module("langchain_chroma", Chroma=Chroma),
        "app.models.note": _module("app.models.note", Note=dependency),
        "app.models.review_record": _module("app.models.review_record", ReviewRecord=dependency),
        "app.schemas.models": _module(
            "app.schemas.models", NoteCreate=dependency, NoteUpdate=dependency, NoteResponse=dependency,
        ),
        "app.utils.factory": _module("app.utils.factory", embed_model=object()),
        "app.utils.config": _module("app.utils.config", chroma_config={"persist_directory": str(tmp_path)}),
        "app.utils.path_tool": _module("app.utils.path_tool", get_abstract_path=lambda _: str(tmp_path)),
        "app.core.logger_handler": _module("app.core.logger_handler", logger=logging.getLogger(__name__)),
        "app.core.task_registry": _module("app.core.task_registry", background_tasks=object()),
        "app.utils.prompt_loader": _module("app.utils.prompt_loader", load_prompt=lambda **_: "unused"),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    name = "_m5_note_initialization"
    spec = importlib.util.spec_from_file_location(name, backend_root / "app/services/note_service.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module, calls, control


def test_note_import_and_construction_do_not_open_storage(isolated_notes):
    module, calls, _ = isolated_notes
    service = module.NoteService()
    assert calls == []
    for instance in (module.note_service, service):
        with pytest.raises(RuntimeError, match="初始化"):
            _ = instance.notes_store


def test_note_initialization_borrows_the_explicit_client(isolated_notes):
    module, calls, _ = isolated_notes
    client, embedding = object(), object()
    module.note_service.initialize_storage(client=client, embedding_function=embedding)
    assert calls == [{
        "collection_name": module.NOTES_COLLECTION_NAME,
        "client": client,
        "embedding_function": embedding,
    }]
    assert module.note_service.notes_store._client is client


def test_repeated_initialization_is_idempotent(isolated_notes):
    module, calls, _ = isolated_notes
    client, embedding = object(), object()
    service = module.note_service
    service.initialize_storage(client=client, embedding_function=embedding)
    original = service.notes_store
    service.initialize_storage(client=client, embedding_function=embedding)
    assert service.notes_store is original
    assert len(calls) == 1


@pytest.mark.parametrize("changed", ["client", "embedding_function"])
def test_existing_storage_binding_cannot_be_replaced(isolated_notes, changed):
    module, calls, _ = isolated_notes
    arguments = {"client": object(), "embedding_function": object()}
    service = module.note_service
    service.initialize_storage(**arguments)
    original = service.notes_store
    arguments[changed] = object()
    with pytest.raises(RuntimeError, match="绑定"):
        service.initialize_storage(**arguments)
    assert service.notes_store is original
    assert len(calls) == 1


@pytest.mark.parametrize("missing", ["client", "embedding_function"])
def test_missing_binding_is_rejected_before_storage_construction(isolated_notes, missing):
    module, calls, _ = isolated_notes
    arguments = {"client": object(), "embedding_function": object()}
    arguments[missing] = None
    with pytest.raises(ValueError):
        module.note_service.initialize_storage(**arguments)
    assert calls == []


def test_failed_initialization_does_not_close_borrowed_client(isolated_notes):
    module, calls, control = isolated_notes
    closed = []
    client = SimpleNamespace(close=lambda: closed.append(True))
    embedding = object()
    error = RuntimeError("合成集合初始化失败")
    control.failure = error
    with pytest.raises(RuntimeError) as failure:
        module.note_service.initialize_storage(client=client, embedding_function=embedding)
    assert failure.value is error
    with pytest.raises(RuntimeError, match="初始化"):
        _ = module.note_service.notes_store
    assert closed == []
    control.failure = None
    module.note_service.initialize_storage(client=client, embedding_function=embedding)
    assert module.note_service.notes_store._client is client
    assert len(calls) == 2 and closed == []


def test_parallel_initialization_publishes_one_complete_store(isolated_notes):
    module, calls, control = isolated_notes
    client, embedding = object(), object()
    control.entered = threading.Event()
    control.release = threading.Event()
    service = module.note_service
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(
            service.initialize_storage, client=client, embedding_function=embedding,
        ) for _ in range(4)]
        try:
            assert control.entered.wait(2)
            with pytest.raises(RuntimeError, match="初始化"):
                _ = service.notes_store
            assert len(calls) == 1
        finally:
            control.release.set()
        for future in futures:
            future.result(timeout=2)
    assert len(calls) == 1
    assert service.notes_store._client is client
