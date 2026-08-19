from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

def _parse(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _node_source(path: Path, node_type: type[ast.AST], name: str) -> str:
    source, tree = _parse(path)
    for node in ast.walk(tree):
        if isinstance(node, node_type) and getattr(node, "name", None) == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"未找到 {name}: {path}")


def _load_vector_store_with_fakes(
    monkeypatch: pytest.MonkeyPatch,
    source_path: Path,
    persist_dir: Path,
    failure: Exception,
) -> tuple[types.ModuleType, type]:
    class FailingChroma:
        attempts = 0

        def __init__(self, **_: object) -> None:
            type(self).attempts += 1
            raise failure

    class DummyDependency:
        def __init__(self, *_: object, **__: object) -> None:
            pass

    class DummyLogger:
        def __getattr__(self, _: str):
            return lambda *args, **kwargs: None

    def package(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []
        return module

    modules = {
        "app": package("app"),
        "app.rag": package("app.rag"),
        "app.utils": package("app.utils"),
        "app.core": package("app.core"),
        "app.rag.retrievers": types.ModuleType("app.rag.retrievers"),
        "app.rag.retrievers.hybrid_retriever": types.ModuleType(
            "app.rag.retrievers.hybrid_retriever"
        ),
        "app.rag.md5_manager": types.ModuleType("app.rag.md5_manager"),
        "app.rag.document_handler": types.ModuleType("app.rag.document_handler"),
        "app.utils.config": types.ModuleType("app.utils.config"),
        "app.utils.factory": types.ModuleType("app.utils.factory"),
        "app.utils.path_tool": types.ModuleType("app.utils.path_tool"),
        "app.utils.image_extractor": types.ModuleType("app.utils.image_extractor"),
        "app.core.logger_handler": types.ModuleType("app.core.logger_handler"),
        "langchain_chroma": types.ModuleType("langchain_chroma"),
        "langchain_core": package("langchain_core"),
        "langchain_core.documents": types.ModuleType("langchain_core.documents"),
    }
    modules["langchain_chroma"].Chroma = FailingChroma
    modules["langchain_core.documents"].Document = object
    modules["app.utils.config"].chroma_config = {
        "collection_name": "m0-disposable",
        "persist_directory": str(persist_dir),
    }
    modules["app.utils.factory"].embed_model = object()
    modules["app.utils.path_tool"].get_abstract_path = lambda _: str(persist_dir)
    modules["app.core.logger_handler"].logger = DummyLogger()
    modules["app.rag.retrievers"].EmptyRetriever = DummyDependency
    modules["app.rag.retrievers.hybrid_retriever"].HybridRetriever = DummyDependency
    modules["app.rag.md5_manager"].MD5Store = DummyDependency
    modules["app.rag.document_handler"].DocumentProcessor = DummyDependency
    modules["app.utils.image_extractor"].delete_image_directory = lambda *args: None
    modules["app.utils.image_extractor"].delete_user_all_images = lambda *args: None

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "app.rag._m0_vector_store"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, FailingChroma


def _tree_fingerprint(root: Path) -> tuple[tuple[str, int, str], ...]:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            entries.append(
                (
                    path.relative_to(root).as_posix(),
                    path.stat().st_size,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    return tuple(entries)


@pytest.mark.baseline
@pytest.mark.p0
@pytest.mark.parametrize(
    "failure",
    [
        PermissionError("injected permission failure"),
        OSError(16, "injected lock conflict"),
        RuntimeError("injected database corruption"),
    ],
    ids=["permission", "lock", "corruption"],
)
def test_data_001_initialization_failure_preserves_existing_data(
    backend_root: Path,
    isolated_chroma_dir: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    persist_dir, sentinel = isolated_chroma_dir
    before = _tree_fingerprint(persist_dir)
    module, chroma_class = _load_vector_store_with_fakes(
        monkeypatch,
        backend_root / "app/rag/vector_store.py",
        persist_dir,
        failure,
    )

    with pytest.raises(type(failure), match="injected"):
        module.VectorStoreService()

    assert sentinel.exists(), "初始化失败删除了已有 Chroma 数据目录"
    assert _tree_fingerprint(persist_dir) == before
    assert chroma_class.attempts == 1
    assert module.VectorStoreService._initialized is False


@pytest.mark.baseline
@pytest.mark.p0
def test_sec_001_session_listing_requires_authentication(backend_root: Path) -> None:
    function = _node_source(
        backend_root / "app/router/chat.py", ast.AsyncFunctionDef, "get_all_sessions"
    )
    assert "Depends(get_current_user_id)" in function


@pytest.mark.baseline
@pytest.mark.p0
def test_sec_002_reranker_requires_authentication(backend_root: Path) -> None:
    function = _node_source(
        backend_root / "app/router/chat.py", ast.AsyncFunctionDef, "reorder_documents"
    )
    assert "Depends(get_current_user_id)" in function


@pytest.mark.baseline
@pytest.mark.p0
def test_sec_002_reranker_schema_has_resource_limits(backend_root: Path) -> None:
    model = _node_source(
        backend_root / "app/schemas/models.py", ast.ClassDef, "ReorderRequest"
    )
    assert "max_length" in model
    assert "max_items" in model or "max_length" in model.split("documents", 1)[-1]


@pytest.mark.baseline
@pytest.mark.p0
def test_sec_003_related_note_vector_query_is_user_scoped(backend_root: Path) -> None:
    _, tree = _parse(backend_root / "app/services/note_service.py")
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_related_notes":
            target = node
            break
    assert target is not None

    note_calls = [
        call
        for call in ast.walk(target)
        if isinstance(call, ast.Call)
        and ast.unparse(call.func) == "asyncio.to_thread"
        and call.args
        and ast.unparse(call.args[0]).endswith(
            "self._notes_store.similarity_search_with_score"
        )
    ]
    assert len(note_calls) == 1
    filter_values = [
        keyword.value for keyword in note_calls[0].keywords if keyword.arg == "filter"
    ]
    assert filter_values and "user_id" in ast.unparse(filter_values[0])


@pytest.mark.baseline
@pytest.mark.p0
def test_agent_rag_tool_cannot_accept_model_supplied_user_id(
    backend_root: Path,
) -> None:
    _, tree = _parse(backend_root / "app/agent/agent_tools.py")
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "rag_summary_tools"
    )
    assert [argument.arg for argument in function.args.args] == ["query"]


@pytest.mark.baseline
@pytest.mark.p0
def test_jwt_refresh_rejects_expired_tokens(repository_root: Path) -> None:
    function = _node_source(
        repository_root / "DjangoUserService/apps/user/authentications.py",
        ast.FunctionDef,
        "refresh_token",
    )
    assert "verify_exp': False" not in function


@pytest.mark.baseline
@pytest.mark.p0
def test_jwt_refresh_checks_revocation(repository_root: Path) -> None:
    function = _node_source(
        repository_root / "DjangoUserService/apps/user/authentications.py",
        ast.FunctionDef,
        "refresh_token",
    )
    assert "cache.add(" in function
    source = (
        repository_root / "DjangoUserService/apps/user/authentications.py"
    ).read_text(encoding="utf-8")
    assert 'return f"jwt:revoked:{jti}"' in source


@pytest.mark.baseline
@pytest.mark.p0
def test_jwt_refresh_checks_account_status(repository_root: Path) -> None:
    refresh = _node_source(
        repository_root / "DjangoUserService/apps/user/authentications.py",
        ast.FunctionDef,
        "refresh_token",
    )
    active_user = _node_source(
        repository_root / "DjangoUserService/apps/user/authentications.py",
        ast.FunctionDef,
        "_get_active_user",
    )
    assert "_get_active_user(" in refresh
    assert ".status" in active_user


@pytest.mark.baseline
@pytest.mark.p0
def test_sec_004_django_default_settings_are_not_production_settings(
    repository_root: Path,
) -> None:
    source = (
        repository_root / "DjangoUserService/DjangoUserService/settings.py"
    ).read_text(encoding="utf-8")
    assert "DEBUG = False" in source
    assert "CORS_ALLOW_ALL_ORIGINS = False" in source
    assert "django.middleware.csrf.CsrfViewMiddleware" in source.replace("#", "")


@pytest.mark.baseline
@pytest.mark.p0
def test_rate_001_global_rate_limit_is_installed(backend_root: Path) -> None:
    _, tree = _parse(backend_root / "main.py")
    middleware_calls = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and ast.unparse(call.func) == "app.add_middleware"
        and call.args
        and ast.unparse(call.args[0]) == "RateLimitMiddleware"
    ]
    assert middleware_calls


@pytest.mark.baseline
def test_migration_directories_are_not_globally_ignored(repository_root: Path) -> None:
    patterns = (repository_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "migrations/" not in {line.strip() for line in patterns}
