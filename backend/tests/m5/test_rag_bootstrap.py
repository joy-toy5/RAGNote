"""启动组合和共享客户端的离线契约；所有存储与模型依赖均为替身。"""
from __future__ import annotations

import ast
import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _load(monkeypatch: pytest.MonkeyPatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bootstrap(backend_root: Path, monkeypatch: pytest.MonkeyPatch):
    events = []
    client, embedding = object(), object()
    control = SimpleNamespace(fail_at=None)

    def step(name, result=None):
        events.append(name)
        if control.fail_at == name:
            raise RuntimeError(f"合成{name}失败")
        return result

    def bind_notes(**kwargs):
        assert kwargs == {"client": client, "embedding_function": embedding}
        step("notes")

    modules = {
        "app.utils.factory": _module(
            "app.utils.factory",
            get_chat_model=lambda: step("chat", object()),
            get_embed_model=lambda: step("embed", embedding),
            get_vision_model=lambda: step("vision", object()),
        ),
        "app.rag.vector_store": _module(
            "app.rag.vector_store", VectorStoreService=lambda: step(
                "vectors", SimpleNamespace(vectors_store=SimpleNamespace(_client=client)),
            ),
        ),
        "app.services.note_service": _module(
            "app.services.note_service", note_service=SimpleNamespace(initialize_storage=bind_notes),
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    module = _load(monkeypatch, "_m5_rag_bootstrap", backend_root / "app/rag/bootstrap.py")
    return module, events, control


def test_bootstrap_import_is_inert(bootstrap):
    _, events, _ = bootstrap
    assert events == []


def test_bootstrap_initializes_models_before_shared_storage(bootstrap):
    module, events, _ = bootstrap
    module.initialize_rag_resources()
    assert events == ["chat", "embed", "vision", "vectors", "notes"]


@pytest.mark.parametrize("failed", ["chat", "embed", "vision", "vectors", "notes"])
def test_bootstrap_failure_stops_later_initializers(bootstrap, failed):
    module, events, control = bootstrap
    control.fail_at = failed
    with pytest.raises(RuntimeError, match=f"合成{failed}失败"):
        module.initialize_rag_resources()
    expected = ["chat", "embed", "vision", "vectors", "notes"]
    assert events == expected[:expected.index(failed) + 1]


@pytest.fixture
def vectors(backend_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls, cleared, closed = [], [], []
    embedding = object()
    control = SimpleNamespace(fail_at=None)

    class Chroma:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if control.fail_at == "client":
                raise RuntimeError("合成客户端失败")
            self._client = SimpleNamespace(close=lambda: closed.append(self))

    class Dependency:
        def __init__(self, *args, **kwargs):
            if control.fail_at == "dependency":
                raise RuntimeError("合成依赖初始化失败")

    modules = {
        "langchain_chroma": _module("langchain_chroma", Chroma=Chroma),
        "app.utils.factory": _module(
            "app.utils.factory", embed_model=embedding, get_embed_model=lambda: embedding,
        ),
        "app.utils.config": _module("app.utils.config", chroma_config={
            "persist_directory": str(tmp_path), "collection_name": "disposable",
        }),
        "app.utils.path_tool": _module("app.utils.path_tool", get_abstract_path=lambda _: str(tmp_path)),
        "app.core.logger_handler": _module("app.core.logger_handler", logger=logging.getLogger(__name__)),
        "app.utils.image_extractor": _module(
            "app.utils.image_extractor", delete_image_directory=lambda *_: None,
            delete_user_all_images=lambda *_: None,
        ),
        "app.rag.retrievers.hybrid_retriever": _module(
            "app.rag.retrievers.hybrid_retriever", HybridRetriever=Dependency,
        ),
        "app.rag.md5_manager": _module("app.rag.md5_manager", MD5Store=Dependency),
        "app.rag.document_handler": _module("app.rag.document_handler", DocumentProcessor=Dependency),
        "chromadb.api.shared_system_client": _module(
            "chromadb.api.shared_system_client",
            SharedSystemClient=SimpleNamespace(clear_system_cache=lambda: cleared.append(True)),
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    module = _load(monkeypatch, "app.rag._m5_storage_bootstrap", backend_root / "app/rag/vector_store.py")
    return module, calls, cleared, closed, embedding, control


def test_default_initialization_never_clears_another_clients_cache(vectors):
    module, calls, cleared, _, embedding, _ = vectors
    first = module.VectorStoreService()
    second = module.VectorStoreService()
    assert first is second and len(calls) == 1
    assert calls[0]["embedding_function"] is embedding
    assert cleared == []


def test_explicit_target_remains_independent_of_the_default_store(vectors, tmp_path):
    module, calls, cleared, closed, embedding, _ = vectors
    explicit = module.VectorStoreService.for_explicit_target(
        persist_directory=str(tmp_path / "explicit"), collection_name="isolated",
        embedding_function=embedding, top_k=2,
    )
    assert module.VectorStoreService._instance is None
    default = module.VectorStoreService()
    assert default is not explicit and len(calls) == 2
    assert cleared == []
    explicit.close()
    assert closed == [explicit.vectors_store]
    with pytest.raises(RuntimeError, match="生产单例"):
        default.close()


@pytest.mark.parametrize("failed", ["client", "dependency"])
def test_failed_default_initialization_requires_a_fresh_process(vectors, failed):
    module, calls, cleared, closed, _, control = vectors
    control.fail_at = failed
    with pytest.raises(RuntimeError, match="合成"):
        module.VectorStoreService()
    control.fail_at = None
    with pytest.raises(RuntimeError, match="重启"):
        module.VectorStoreService()
    assert len(calls) == 1
    assert module.VectorStoreService._initialized is False
    assert cleared == [] and closed == []


@pytest.mark.parametrize("filename", ["vector_store.py", "rag_service.py"])
def test_standalone_examples_prepare_environment_explicitly(backend_root, filename):
    tree = ast.parse((backend_root / "app/rag" / filename).read_text())
    guard = next(node for node in tree.body if isinstance(node, ast.If))
    assert ast.unparse(guard.test) == "__name__ == '__main__'"
    preparation = [
        node.value for node in guard.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "load_dotenv"
    ]
    assert len(preparation) == 1
    assert ast.unparse(preparation[0]) == "load_dotenv(override=False)"
    if filename == "rag_service.py":
        main = next(node for node in guard.body if isinstance(node, ast.AsyncFunctionDef))
        assert ast.unparse(main.body[0]) == "initialize_rag_resources()"
