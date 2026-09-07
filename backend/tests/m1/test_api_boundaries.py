from __future__ import annotations

import ast
import asyncio
import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.schemas.models import ReorderRequest


def _module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name
    )
    segment = ast.get_source_segment(source, function)
    assert segment is not None
    return segment


def _load_chat_service(
    monkeypatch: pytest.MonkeyPatch,
    source_path: Path,
) -> tuple[types.ModuleType, SimpleNamespace]:
    class SessionManager:
        def __init__(self) -> None:
            self.user_ids: list[str] = []

        async def get_all_session_ids(self, user_id: str) -> list[str]:
            self.user_ids.append(user_id)
            return ["session-a"]

    manager = SessionManager()
    proxy = SimpleNamespace(session_manager=manager)
    logger = logging.getLogger("m1-chat-service")
    modules = {
        "app.core.logger_handler": _module("app.core.logger_handler", logger=logger),
        "app.rag.rag_service": _module("app.rag.rag_service", RagService=object),
        "app.rag.reorder_service": _module(
            "app.rag.reorder_service", reorder_service=object()
        ),
        "app.agent.agent": _module(
            "app.agent.agent", get_agent_response=lambda *args, **kwargs: None
        ),
        "app.services": _module("app.services", session_manager=proxy),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "app.router._m1_chat_service"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, manager


def _load_note_service(
    monkeypatch: pytest.MonkeyPatch,
    source_path: Path,
) -> tuple[types.ModuleType, list[dict[str, str]]]:
    note_filters: list[dict[str, str]] = []

    class NotesStore:
        def __init__(self, **_: object) -> None:
            pass

        def similarity_search_with_score(
            self,
            _: str,
            *,
            k: int,
            filter: dict[str, str],
        ) -> list[tuple[SimpleNamespace, float]]:
            del k
            note_filters.append(filter)
            return []

    class KnowledgeCollection:
        def similarity_search_with_score(
            self,
            _: str,
            *,
            k: int,
            filter: dict[str, str],
        ) -> list[tuple[SimpleNamespace, float]]:
            del k
            assert filter == {"user_id": "user-a"}
            return []

    class VectorStoreService:
        def __init__(self) -> None:
            self.vectors_store = KnowledgeCollection()

    logger = logging.getLogger("m1-note-service")
    modules = {
        "langchain_chroma": _module("langchain_chroma", Chroma=NotesStore),
        "app.models.note": _module("app.models.note", Note=type("Note", (), {})),
        "app.models.review_record": _module(
            "app.models.review_record", ReviewRecord=type("ReviewRecord", (), {})
        ),
        "app.utils.factory": _module("app.utils.factory", get_embed_model=lambda: object()),
        "app.utils.config": _module(
            "app.utils.config", chroma_config={"persist_directory": "unused"}
        ),
        "app.utils.path_tool": _module(
            "app.utils.path_tool", get_abstract_path=lambda _: "/tmp/m1-unused"
        ),
        "app.core.logger_handler": _module("app.core.logger_handler", logger=logger),
        "app.utils.prompt_loader": _module(
            "app.utils.prompt_loader", load_prompt=lambda _: "unused"
        ),
        "app.rag.vector_store": _module(
            "app.rag.vector_store", VectorStoreService=VectorStoreService
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "app.services._m1_note_service"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, note_filters


def _load_hybrid_retriever(
    monkeypatch: pytest.MonkeyPatch,
    source_path: Path,
) -> types.ModuleType:
    class DummyRetriever:
        pass

    modules = {
        "langchain_chroma": _module("langchain_chroma", Chroma=object),
        "langchain_core.documents": _module(
            "langchain_core.documents", Document=object
        ),
        "langchain_core.retrievers": _module(
            "langchain_core.retrievers", BaseRetriever=DummyRetriever
        ),
        "langchain_community.retrievers": _module(
            "langchain_community.retrievers", BM25Retriever=object
        ),
        "langchain_classic.retrievers": _module(
            "langchain_classic.retrievers", EnsembleRetriever=object
        ),
        # bm25 键随 RAG-005 加入：HybridRetriever 构造时读融合权重。
        "app.utils.config": _module(
            "app.utils.config",
            chroma_config={
                "k": 3,
                "bm25": {
                    "tokenizer": "cjk_bigram.v1",
                    "vector_weight": 0.6,
                    "bm25_weight": 0.4,
                },
            },
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "app.rag.retrievers._m1_hybrid_retriever"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.p0
def test_session_listing_requires_identity_at_every_layer(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _function_source(backend_root / "app/router/chat.py", "get_all_sessions")
    assert "Depends(get_current_user_id)" in route

    module, manager = _load_chat_service(
        monkeypatch, backend_root / "app/router/chat_service.py"
    )
    result = asyncio.run(module.ChatService().handle_get_all_sessions("user-a"))
    assert result == ["session-a"]
    assert manager.user_ids == ["user-a"]


@pytest.mark.p0
@pytest.mark.parametrize(
    "payload",
    [
        {"query": "", "documents": ["document"]},
        {"query": "q" * 4097, "documents": ["document"]},
        {"query": "query", "documents": []},
        {"query": "query", "documents": ["d"] * 33},
        {"query": "query", "documents": ["d" * 12001]},
        {"query": "query", "documents": ["d" * 4000] * 26},
    ],
)
def test_reranker_rejects_oversized_work_before_model(payload: dict) -> None:
    with pytest.raises(ValidationError):
        ReorderRequest.model_validate(payload)


@pytest.mark.p0
def test_reranker_route_requires_identity(backend_root: Path) -> None:
    route = _function_source(backend_root / "app/router/chat.py", "reorder_documents")
    assert "Depends(get_current_user_id)" in route


@pytest.mark.p0
def test_related_note_lookup_is_scoped_in_chroma(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, note_filters = _load_note_service(
        monkeypatch, backend_root / "app/services/note_service.py"
    )
    service = module.NoteService()
    service.initialize_storage(client=object(), embedding_function=object())

    async def get_owned_note(*_: object, **__: object) -> SimpleNamespace:
        return SimpleNamespace(content="same content")

    service.get_note = get_owned_note
    result = asyncio.run(
        service.get_related_notes(None, "note-a", "user-a", top_k=3)
    )

    assert result == []
    assert note_filters == [{"user_id": "user-a", "doc_type": "note"}]


@pytest.mark.p0
def test_retriever_cannot_be_constructed_without_identity(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_hybrid_retriever(
        monkeypatch,
        backend_root / "app/rag/retrievers/hybrid_retriever.py",
    )
    retriever = module.HybridRetriever(SimpleNamespace())
    with pytest.raises(ValueError, match="用户 ID"):
        asyncio.run(retriever.get_retriever("query", ""))
    with pytest.raises(ValueError, match="用户 ID"):
        asyncio.run(retriever.get_bm25_retriever(""))


@pytest.mark.p0
def test_image_paths_are_user_scoped_and_reads_do_not_create_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.utils import image_extractor

    monkeypatch.setattr(image_extractor, "get_data_path", lambda: str(tmp_path))
    md5 = "a" * 32

    missing = image_extractor.get_image_path("user-a", md5, "p0_i0.png")
    assert not missing.exists()
    assert not (tmp_path / "extracted_images").exists()

    user_b_dir = Path(
        image_extractor.get_image_storage_dir("user-b", md5, create=True)
    )
    (user_b_dir / "p0_i0.png").write_bytes(b"user-b")

    assert not image_extractor.get_image_path(
        "user-a", md5, "p0_i0.png"
    ).exists()
    assert image_extractor.get_image_path(
        "user-b", md5, "p0_i0.png"
    ).read_bytes() == b"user-b"


@pytest.mark.p0
@pytest.mark.parametrize(
    ("user_id", "md5", "filename"),
    [
        ("../user-b", "a" * 32, "p0_i0.png"),
        ("user-a", "../" + "a" * 29, "p0_i0.png"),
        ("user-a", "a" * 32, "../secret.png"),
        ("user-a", "a" * 32, "arbitrary.txt"),
    ],
)
def test_image_paths_reject_untrusted_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    user_id: str,
    md5: str,
    filename: str,
) -> None:
    from app.utils import image_extractor

    monkeypatch.setattr(image_extractor, "get_data_path", lambda: str(tmp_path))
    with pytest.raises(ValueError):
        image_extractor.get_image_path(user_id, md5, filename)
    assert not (tmp_path / "extracted_images").exists()


@pytest.mark.p0
def test_batch_images_require_owned_md5(backend_root: Path) -> None:
    function = _function_source(
        backend_root / "app/router/knowledge_service.py", "handle_get_batch_images"
    )
    assert "get_md5_info(user_id, md5)" in function
