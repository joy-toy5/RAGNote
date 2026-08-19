from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

USER_A = "user-a"
USER_B = "user-b"
NOTE_B = "note-b"
SESSION_A = "session-a"
SESSION_B = "session-b"
MD5_A = "a" * 32
MD5_B = "b" * 32


def _module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


class Predicate:
    def __init__(self, field: str, operator: str, value: object) -> None:
        self.field = field
        self.operator = operator
        self.value = value

    def matches(self, record: object) -> bool:
        current = getattr(record, self.field)
        if self.operator == "eq":
            return current == self.value
        if self.operator == "in":
            return current in self.value
        raise AssertionError(f"未支持的测试操作符: {self.operator}")


class Field:
    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, value: object) -> Predicate:
        return Predicate(self.name, "eq", value)

    def in_(self, values: list[str]) -> Predicate:
        return Predicate(self.name, "in", values)


class Statement:
    def __init__(self, operation: str) -> None:
        self.operation = operation
        self.predicates: list[Predicate] = []

    def where(self, *predicates: Predicate) -> "Statement":
        self.predicates.extend(predicates)
        return self


class ScalarResult:
    def __init__(self, records: list[object]) -> None:
        self.records = records

    def scalar_one_or_none(self) -> object | None:
        if len(self.records) > 1:
            raise AssertionError("测试查询意外返回多条记录")
        return self.records[0] if self.records else None

    def scalars(self) -> "ScalarResult":
        return self

    def all(self) -> list[object]:
        return self.records


class FakeNote:
    id = Field("id")
    user_id = Field("user_id")


class NoteDatabase:
    def __init__(self) -> None:
        self.records = {
            NOTE_B: SimpleNamespace(
                id=NOTE_B,
                user_id=USER_B,
                title="B 的笔记",
                content="只属于用户 B 的内容",
                tags=None,
                category=None,
                created_at=None,
                updated_at=None,
            )
        }
        self.statements: list[Statement] = []
        self.commits = 0

    async def execute(self, statement: Statement) -> ScalarResult:
        self.statements.append(statement)
        matches = [
            record
            for record in self.records.values()
            if all(predicate.matches(record) for predicate in statement.predicates)
        ]
        if statement.operation == "delete":
            for record in matches:
                self.records.pop(record.id)
        return ScalarResult(matches)

    async def commit(self) -> None:
        self.commits += 1


class NotesStore:
    def __init__(self, **_: object) -> None:
        self.search_filters: list[dict[str, str]] = []
        self.delete_filters: list[dict[str, str]] = []
        self.documents = [
            SimpleNamespace(
                metadata={
                    "user_id": USER_B,
                    "doc_type": "note",
                    "note_id": NOTE_B,
                }
            )
        ]

    def similarity_search(
        self,
        _: str,
        *,
        k: int,
        filter: dict[str, str],
    ) -> list[object]:
        self.search_filters.append(filter)
        return [
            document
            for document in self.documents[:k]
            if all(document.metadata.get(key) == value for key, value in filter.items())
        ]

    def delete(self, *, where: dict[str, str]) -> None:
        self.delete_filters.append(where)


def _load_note_service(
    monkeypatch: pytest.MonkeyPatch,
    source_path: Path,
) -> types.ModuleType:
    modules = {
        "sqlalchemy": _module(
            "sqlalchemy",
            select=lambda _: Statement("select"),
            delete=lambda _: Statement("delete"),
            func=SimpleNamespace(),
            update=lambda _: Statement("update"),
        ),
        "sqlalchemy.ext": _package("sqlalchemy.ext"),
        "sqlalchemy.ext.asyncio": _module(
            "sqlalchemy.ext.asyncio", AsyncSession=object
        ),
        "langchain_chroma": _module("langchain_chroma", Chroma=NotesStore),
        "langchain_core": _package("langchain_core"),
        "langchain_core.documents": _module(
            "langchain_core.documents", Document=object
        ),
        "langchain_core.messages": _module(
            "langchain_core.messages", HumanMessage=object
        ),
        "app.models.note": _module("app.models.note", Note=FakeNote),
        "app.models.review_record": _module(
            "app.models.review_record", ReviewRecord=object
        ),
        "app.utils.factory": _module("app.utils.factory", embed_model=object()),
        "app.utils.config": _module(
            "app.utils.config", chroma_config={"persist_directory": "unused"}
        ),
        "app.utils.path_tool": _module(
            "app.utils.path_tool", get_abstract_path=lambda _: "/tmp/m1-unused"
        ),
        "app.core.logger_handler": _module(
            "app.core.logger_handler", logger=logging.getLogger("m1-tenant-notes")
        ),
        "app.utils.prompt_loader": _module(
            "app.utils.prompt_loader", load_prompt=lambda _: "unused"
        ),
        "app.schemas.models": _module(
            "app.schemas.models",
            NoteCreate=object,
            NoteUpdate=object,
            NoteResponse=lambda **values: SimpleNamespace(**values),
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "app.services._m1_tenant_note_service"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.p0
def test_user_cannot_read_search_or_delete_another_users_note(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_note_service(
        monkeypatch, backend_root / "app/services/note_service.py"
    )
    service = module.NoteService()
    database = NoteDatabase()

    assert asyncio.run(service.get_note(database, NOTE_B, USER_A)) is None
    assert asyncio.run(service.search_notes(database, USER_A, "只属于用户 B")) == []
    assert asyncio.run(service.delete_note(database, NOTE_B, USER_A)) is False

    assert NOTE_B in database.records
    assert database.commits == 0
    assert service.notes_store.search_filters == [
        {"user_id": USER_A, "doc_type": "note"}
    ]
    assert service.notes_store.delete_filters == []
    for statement in database.statements:
        assert any(
            predicate.field == "user_id" and predicate.value == USER_A
            for predicate in statement.predicates
        )


class FakeVectorCollection:
    def __init__(self) -> None:
        self.get_filters: list[dict[str, str]] = []
        self.delete_filters: list[dict[str, object]] = []
        self.documents = [
            {
                "id": "chunk-a",
                "content": "A 的知识库",
                "metadata": {
                    "user_id": USER_A,
                    "source": "a.pdf",
                    "original_filename": "a.pdf",
                    "md5": MD5_A,
                },
            },
            {
                "id": "chunk-b",
                "content": "B 的知识库",
                "metadata": {
                    "user_id": USER_B,
                    "source": "b.pdf",
                    "original_filename": "b.pdf",
                    "md5": MD5_B,
                },
            },
        ]

    def get(self, *, include: list[str], where: dict[str, str]) -> dict[str, list]:
        del include
        self.get_filters.append(where)
        matches = [
            document
            for document in self.documents
            if all(
                document["metadata"].get(key) == value for key, value in where.items()
            )
        ]
        return {
            "ids": [document["id"] for document in matches],
            "documents": [document["content"] for document in matches],
            "metadatas": [document["metadata"] for document in matches],
        }

    def delete(self, *, where: dict[str, object]) -> None:
        self.delete_filters.append(where)


class FakeMd5Store:
    def __init__(self) -> None:
        self.records = {(USER_A, "a.pdf"): MD5_A, (USER_B, "b.pdf"): MD5_B}

    async def get_md5_by_filename(self, user_id: str, filename: str) -> str | None:
        return self.records.get((user_id, filename))

    async def get_md5_info(self, user_id: str, md5: str) -> dict | None:
        return next(
            (
                {"md5": value, "filename": filename}
                for (owner_id, filename), value in self.records.items()
                if owner_id == user_id and value == md5
            ),
            None,
        )

    async def delete_by_filename(self, user_id: str, filename: str) -> str | None:
        return self.records.pop((user_id, filename), None)

    async def delete_single_md5(self, user_id: str, md5: str) -> bool:
        key = next(
            (
                key
                for key, value in self.records.items()
                if key[0] == user_id and value == md5
            ),
            None,
        )
        if key is None:
            return False
        self.records.pop(key)
        return True


def _load_vector_store(
    monkeypatch: pytest.MonkeyPatch,
    source_path: Path,
) -> tuple[types.ModuleType, list[tuple[str, str]]]:
    image_deletes: list[tuple[str, str]] = []

    class DummyDependency:
        def __init__(self, *_: object, **__: object) -> None:
            pass

    modules = {
        "app.rag.retrievers": _module(
            "app.rag.retrievers", EmptyRetriever=DummyDependency
        ),
        "app.rag.retrievers.hybrid_retriever": _module(
            "app.rag.retrievers.hybrid_retriever", HybridRetriever=DummyDependency
        ),
        "app.rag.md5_manager": _module("app.rag.md5_manager", MD5Store=DummyDependency),
        "app.rag.document_handler": _module(
            "app.rag.document_handler", DocumentProcessor=DummyDependency
        ),
        "app.utils.config": _module(
            "app.utils.config",
            chroma_config={
                "collection_name": "unused",
                "persist_directory": "unused",
            },
        ),
        "app.utils.factory": _module("app.utils.factory", embed_model=object()),
        "app.utils.path_tool": _module(
            "app.utils.path_tool", get_abstract_path=lambda _: "/tmp/m1-unused"
        ),
        "app.core.logger_handler": _module(
            "app.core.logger_handler", logger=logging.getLogger("m1-tenant-vector")
        ),
        "app.utils.image_extractor": _module(
            "app.utils.image_extractor",
            delete_image_directory=lambda user_id, md5: image_deletes.append(
                (user_id, md5)
            ),
            delete_user_all_images=lambda _: None,
        ),
        "langchain_chroma": _module("langchain_chroma", Chroma=DummyDependency),
        "langchain_core.documents": _module(
            "langchain_core.documents", Document=object
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "app.rag._m1_tenant_vector_store"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, image_deletes


@pytest.mark.p0
def test_user_cannot_read_or_delete_another_users_knowledge(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, image_deletes = _load_vector_store(
        monkeypatch, backend_root / "app/rag/vector_store.py"
    )
    service = object.__new__(module.VectorStoreService)
    service.vectors_store = FakeVectorCollection()
    service.md5_store = FakeMd5Store()

    detail = asyncio.run(service.get_document_detail(USER_A, "b.pdf"))
    filename_deleted = asyncio.run(service.delete_by_filename(USER_A, "b.pdf"))
    md5_deleted = asyncio.run(service.delete_single_md5(USER_A, MD5_B))

    assert detail is None
    assert filename_deleted is False
    assert md5_deleted is False
    assert service.vectors_store.get_filters == [{"user_id": USER_A}]
    assert service.vectors_store.delete_filters == []
    assert service.md5_store.records[(USER_B, "b.pdf")] == MD5_B
    assert image_deletes == []


class OrderedVectorCollection:
    def __init__(self, events: list[str], failure_stage: str | None = None) -> None:
        self.events = events
        self.failure_stage = failure_stage
        self.delete_filters: list[dict[str, object]] = []

    def delete(self, *, where: dict[str, object]) -> None:
        self.events.append("vector")
        self.delete_filters.append(where)
        if self.failure_stage == "vector":
            raise RuntimeError("vector delete failed")


class OrderedMd5Store:
    def __init__(self, events: list[str], failure_stage: str | None = None) -> None:
        self.events = events
        self.failure_stage = failure_stage
        self.records = {(USER_A, "a.pdf"): MD5_A, (USER_B, "b.pdf"): MD5_B}

    async def get_md5_by_filename(self, user_id: str, filename: str) -> str | None:
        return self.records.get((user_id, filename))

    async def get_md5_info(self, user_id: str, md5: str) -> dict | None:
        return next(
            (
                {"md5": value, "filename": filename}
                for (owner_id, filename), value in self.records.items()
                if owner_id == user_id and value == md5
            ),
            None,
        )

    async def delete_user_md5(self, user_id: str) -> None:
        self._record_md5_delete()
        self.records = {
            key: value for key, value in self.records.items() if key[0] != user_id
        }

    async def delete_by_filename(self, user_id: str, filename: str) -> str | None:
        self._record_md5_delete()
        return self.records.pop((user_id, filename), None)

    async def delete_single_md5(self, user_id: str, md5: str) -> bool:
        self._record_md5_delete()
        key = next(
            (
                key
                for key, value in self.records.items()
                if key[0] == user_id and value == md5
            ),
            None,
        )
        if key is None:
            return False
        self.records.pop(key)
        return True

    def _record_md5_delete(self) -> None:
        self.events.append("md5")
        if self.failure_stage == "md5":
            raise RuntimeError("md5 delete failed")


class MissingFinalMd5Store(OrderedMd5Store):
    async def delete_by_filename(self, user_id: str, filename: str) -> str | None:
        del user_id, filename
        self._record_md5_delete()
        return None

    async def delete_single_md5(self, user_id: str, md5: str) -> bool:
        del user_id, md5
        self._record_md5_delete()
        return False


def _build_ordered_vector_service(
    module: types.ModuleType,
    events: list[str],
    failure_stage: str | None = None,
):
    service = object.__new__(module.VectorStoreService)
    service.vectors_store = OrderedVectorCollection(events, failure_stage)
    service.md5_store = OrderedMd5Store(events, failure_stage)

    def delete_image(_: str, __: str) -> None:
        events.append("image")
        if failure_stage == "image":
            raise RuntimeError("image delete failed")

    def delete_all_images(_: str) -> None:
        events.append("image")
        if failure_stage == "image":
            raise RuntimeError("image delete failed")

    module.delete_image_directory = delete_image
    module.delete_user_all_images = delete_all_images
    return service


@pytest.mark.p0
@pytest.mark.parametrize(
    ("operation", "expected_md5_calls"),
    [
        ("user", 1),
        ("filename", 1),
        ("single", 1),
    ],
)
def test_md5_only_delete_preserves_vectors_and_images(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    expected_md5_calls: int,
) -> None:
    module, _ = _load_vector_store(
        monkeypatch, backend_root / "app/rag/vector_store.py"
    )
    events: list[str] = []
    service = _build_ordered_vector_service(module, events)

    if operation == "user":
        asyncio.run(service.delete_user_md5(USER_A, delete_documents=False))
    elif operation == "filename":
        assert asyncio.run(
            service.delete_by_filename(USER_A, "a.pdf", delete_documents=False)
        )
    else:
        assert asyncio.run(
            service.delete_single_md5(USER_A, MD5_A, delete_documents=False)
        )

    assert events == ["md5"] * expected_md5_calls
    assert service.vectors_store.delete_filters == []
    assert (USER_B, "b.pdf") in service.md5_store.records


@pytest.mark.p0
@pytest.mark.parametrize("failure_stage", ["vector", "image", "md5"])
def test_delete_by_filename_propagates_failure_and_keeps_retry_identity(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    module, _ = _load_vector_store(
        monkeypatch, backend_root / "app/rag/vector_store.py"
    )
    events: list[str] = []
    service = _build_ordered_vector_service(module, events, failure_stage)

    with pytest.raises(RuntimeError, match=f"{failure_stage} delete failed"):
        asyncio.run(service.delete_by_filename(USER_A, "a.pdf"))

    expected_events = {
        "vector": ["vector"],
        "image": ["vector", "image"],
        "md5": ["vector", "image", "md5"],
    }
    assert events == expected_events[failure_stage]
    assert service.md5_store.records[(USER_A, "a.pdf")] == MD5_A


@pytest.mark.p0
def test_delete_by_filename_retries_idempotent_derived_cleanup(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = _load_vector_store(
        monkeypatch, backend_root / "app/rag/vector_store.py"
    )
    events: list[str] = []
    service = _build_ordered_vector_service(module, events, "image")

    with pytest.raises(RuntimeError, match="image delete failed"):
        asyncio.run(service.delete_by_filename(USER_A, "a.pdf"))
    service.vectors_store.failure_stage = None
    service.md5_store.failure_stage = None
    module.delete_image_directory = lambda _user_id, _md5: events.append("image")

    assert asyncio.run(service.delete_by_filename(USER_A, "a.pdf")) is True
    assert events == ["vector", "image", "vector", "image", "md5"]
    assert (USER_A, "a.pdf") not in service.md5_store.records


@pytest.mark.p0
def test_delete_by_filename_preserves_other_filename_with_same_md5(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = _load_vector_store(
        monkeypatch, backend_root / "app/rag/vector_store.py"
    )
    events: list[str] = []
    service = _build_ordered_vector_service(module, events)
    service.md5_store.records[(USER_A, "a-copy.pdf")] = MD5_A

    assert asyncio.run(service.delete_by_filename(USER_A, "a-copy.pdf")) is True
    assert (USER_A, "a.pdf") in service.md5_store.records
    assert (USER_A, "a-copy.pdf") not in service.md5_store.records


@pytest.mark.p0
@pytest.mark.parametrize("operation", ["filename", "single"])
def test_document_delete_does_not_report_success_when_identity_commit_is_missing(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    module, _ = _load_vector_store(
        monkeypatch, backend_root / "app/rag/vector_store.py"
    )
    events: list[str] = []
    service = _build_ordered_vector_service(module, events)
    service.md5_store = MissingFinalMd5Store(events)

    with pytest.raises(RuntimeError, match="MD5记录在删除前发生变化"):
        if operation == "filename":
            asyncio.run(service.delete_by_filename(USER_A, "a.pdf"))
        else:
            asyncio.run(service.delete_single_md5(USER_A, MD5_A))

    assert events == ["vector", "image", "md5"]


@pytest.mark.p0
def test_delete_user_md5_propagates_failure_without_deleting_identity(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = _load_vector_store(
        monkeypatch, backend_root / "app/rag/vector_store.py"
    )
    events: list[str] = []
    service = _build_ordered_vector_service(module, events, "image")

    with pytest.raises(RuntimeError, match="image delete failed"):
        asyncio.run(service.delete_user_md5(USER_A))

    assert events == ["vector", "image"]
    assert service.md5_store.records[(USER_A, "a.pdf")] == MD5_A


class ScopedPredicate:
    def __init__(self, owner: str, field: str, operator: str, value: object) -> None:
        self.owner = owner
        self.field = field
        self.operator = operator
        self.value = value

    def matches(self, review: object, note: object) -> bool:
        record = review if self.owner == "review" else note
        current = getattr(record, self.field)
        if self.operator == "eq":
            return current == self.value
        if self.operator == "le":
            return current <= self.value
        raise AssertionError(f"未支持的回顾查询操作符: {self.operator}")


class ScopedField:
    def __init__(self, owner: str, name: str) -> None:
        self.owner = owner
        self.name = name

    def __eq__(self, value: object) -> ScopedPredicate:
        return ScopedPredicate(self.owner, self.name, "eq", value)

    def __le__(self, value: object) -> ScopedPredicate:
        return ScopedPredicate(self.owner, self.name, "le", value)

    def asc(self) -> "ScopedField":
        return self


class FakeReviewRecordModel:
    id = ScopedField("review", "id")
    note_id = ScopedField("review", "note_id")
    user_id = ScopedField("review", "user_id")
    next_review_at = ScopedField("review", "next_review_at")


class FakeReviewNoteModel:
    id = ScopedField("note", "id")
    user_id = ScopedField("note", "user_id")
    title = ScopedField("note", "title")
    content = ScopedField("note", "content")
    tags = ScopedField("note", "tags")
    category = ScopedField("note", "category")


class ReviewStatement:
    def __init__(self) -> None:
        self.predicates: list[ScopedPredicate] = []

    def join(self, *_: object, **__: object) -> "ReviewStatement":
        return self

    def where(self, *predicates: ScopedPredicate) -> "ReviewStatement":
        self.predicates.extend(predicates)
        return self

    def order_by(self, _: object) -> "ReviewStatement":
        return self


class ReviewRows:
    def __init__(self, rows: list[tuple[object, str, str, object, object]]) -> None:
        self.rows = rows

    def all(self) -> list[tuple[object, str, str, object, object]]:
        return self.rows


class ReviewDatabase:
    def __init__(self) -> None:
        self.review = SimpleNamespace(
            id="review-a",
            note_id=NOTE_B,
            user_id=USER_A,
            next_review_at=datetime(2000, 1, 1),
            review_count=0,
            last_reviewed_at=None,
            interval_days=1,
        )
        self.note = SimpleNamespace(
            id=NOTE_B,
            user_id=USER_B,
            title="B 的私有标题",
            content="B 的私有回顾内容",
            tags=["private"],
            category="private",
        )
        self.statements: list[ReviewStatement] = []

    async def execute(self, statement: ReviewStatement) -> ReviewRows:
        self.statements.append(statement)
        if all(
            predicate.matches(self.review, self.note)
            for predicate in statement.predicates
        ):
            return ReviewRows(
                [
                    (
                        self.review,
                        self.note.title,
                        self.note.content,
                        self.note.tags,
                        self.note.category,
                    )
                ]
            )
        return ReviewRows([])


def _load_review_service(
    monkeypatch: pytest.MonkeyPatch,
    source_path: Path,
) -> types.ModuleType:
    modules = {
        "sqlalchemy": _module(
            "sqlalchemy",
            select=lambda *_: ReviewStatement(),
            update=lambda *_: ReviewStatement(),
        ),
        "sqlalchemy.ext": _package("sqlalchemy.ext"),
        "sqlalchemy.ext.asyncio": _module(
            "sqlalchemy.ext.asyncio", AsyncSession=object
        ),
        "app.models.note": _module("app.models.note", Note=FakeReviewNoteModel),
        "app.models.review_record": _module(
            "app.models.review_record", ReviewRecord=FakeReviewRecordModel
        ),
        "app.core.logger_handler": _module(
            "app.core.logger_handler", logger=logging.getLogger("m1-review-tenant")
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "app.services._m1_tenant_review_service"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.p0
def test_today_reviews_rejects_mismatched_note_owner(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_review_service(
        monkeypatch, backend_root / "app/services/review_service.py"
    )
    database = ReviewDatabase()

    reviews = asyncio.run(module.ReviewService().get_today_reviews(database, USER_A))

    assert reviews == []
    assert any(
        predicate.owner == "note"
        and predicate.field == "user_id"
        and predicate.value == USER_A
        for predicate in database.statements[0].predicates
    )


class FakeChatSession:
    id = Field("id")
    user_id = Field("user_id")


class FakeChatMessage:
    session_id = Field("session_id")
    created_at = Field("created_at")


class SessionQuery:
    def __init__(self, records: list[object]) -> None:
        self.records = records
        self.predicates: list[Predicate] = []

    def filter(self, *predicates: Predicate) -> "SessionQuery":
        self.predicates.extend(predicates)
        return self

    def order_by(self, _: object) -> "SessionQuery":
        return self

    def _matches(self) -> list[object]:
        return [
            record
            for record in self.records
            if all(predicate.matches(record) for predicate in self.predicates)
        ]

    def first(self) -> object | None:
        matches = self._matches()
        return matches[0] if matches else None

    def all(self) -> list[object]:
        return self._matches()


class SyncSession:
    def __init__(self, sessions: list[object]) -> None:
        self.sessions = sessions

    def query(self, model: type) -> SessionQuery:
        return SessionQuery(self.sessions if model is FakeChatSession else [])


class AsyncDatabaseSession:
    def __init__(self) -> None:
        self.sessions = [
            SimpleNamespace(
                id=SESSION_A,
                user_id=USER_A,
                title="A",
                created_at=None,
                updated_at=None,
            ),
            SimpleNamespace(
                id=SESSION_B,
                user_id=USER_B,
                title="B",
                created_at=None,
                updated_at=None,
            ),
        ]
        self.deleted: list[object] = []

    async def __aenter__(self) -> "AsyncDatabaseSession":
        return self

    async def __aexit__(self, *args: object) -> None:
        del args

    async def run_sync(self, operation):
        return operation(SyncSession(self.sessions))

    async def delete(self, record: object) -> None:
        self.deleted.append(record)

    async def commit(self) -> None:
        pass


def _load_session_manager(
    monkeypatch: pytest.MonkeyPatch,
    source_path: Path,
) -> tuple[types.ModuleType, AsyncDatabaseSession]:
    database = AsyncDatabaseSession()
    modules = {
        "app.db.db_config": _module(
            "app.db.db_config", AsyncSessionLocal=lambda: database
        ),
        "app.models.chat_history": _module(
            "app.models.chat_history",
            ChatSession=FakeChatSession,
            ChatMessage=FakeChatMessage,
        ),
        "app.core.logger_handler": _module(
            "app.core.logger_handler", logger=logging.getLogger("m1-tenant-session")
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "app.services._m1_tenant_session_manager"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, database


@pytest.mark.p0
def test_user_cannot_read_or_delete_another_users_session(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, database = _load_session_manager(
        monkeypatch, backend_root / "app/services/database_session_manager.py"
    )
    manager = module.DatabaseSessionManager()

    with pytest.raises(HTTPException) as error:
        asyncio.run(manager.get_session(SESSION_B, USER_A))
    assert error.value.status_code == 403

    asyncio.run(manager.clear_session(SESSION_B, USER_A))
    assert asyncio.run(manager.get_all_session_ids(USER_A)) == [SESSION_A]
    assert database.deleted == []
    assert {session.id for session in database.sessions} == {SESSION_A, SESSION_B}


def _load_knowledge_router(
    monkeypatch: pytest.MonkeyPatch,
    source_path: Path,
    get_image_path,
) -> types.ModuleType:
    class KnowledgeService:
        pass

    async def get_current_user_id() -> str:
        return USER_A

    def rate_limit(*_: object, **__: object):
        async def dependency() -> None:
            return None

        return dependency

    modules = {
        "app.router.knowledge_service": _module(
            "app.router.knowledge_service",
            KnowledgeService=KnowledgeService,
            get_knowledge_service=lambda: KnowledgeService(),
        ),
        "app.utils.auth_utils": _module(
            "app.utils.auth_utils", get_current_user_id=get_current_user_id
        ),
        "app.utils.image_extractor": _module(
            "app.utils.image_extractor", get_image_path=get_image_path
        ),
        "app.core.success_response": _module(
            "app.core.success_response", success_response=lambda **values: values
        ),
        "app.core.rate_limit": _module("app.core.rate_limit", rate_limit=rate_limit),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "app.router._m1_tenant_knowledge_router"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.p0
def test_image_access_is_user_scoped_and_rejects_path_traversal(
    backend_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.utils import image_extractor

    monkeypatch.setattr(image_extractor, "get_data_path", lambda: str(tmp_path))
    user_b_image = tmp_path / "extracted_images" / USER_B / MD5_B / "p0_i0.png"
    user_b_image.parent.mkdir(parents=True)
    user_b_image.write_bytes(b"user-b-only")

    module = _load_knowledge_router(
        monkeypatch,
        backend_root / "app/router/knowledge_router.py",
        image_extractor.get_image_path,
    )

    with pytest.raises(HTTPException) as cross_user_error:
        asyncio.run(module.serve_knowledge_image(MD5_B, "p0_i0.png", USER_A))
    assert cross_user_error.value.status_code == 404

    for md5, filename in [
        ("..", "p0_i0.png"),
        (MD5_B, "../../secret.png"),
        (MD5_B, "/etc/passwd"),
    ]:
        with pytest.raises(HTTPException) as traversal_error:
            asyncio.run(module.serve_knowledge_image(md5, filename, USER_A))
        assert traversal_error.value.status_code == 404

    assert user_b_image.read_bytes() == b"user-b-only"


@pytest.mark.p0
def test_upload_stream_does_not_override_global_cors_policy(backend_root: Path) -> None:
    source = (backend_root / "app/router/knowledge_router.py").read_text(
        encoding="utf-8"
    )
    assert '"Access-Control-Allow-Origin": "*"' not in source
