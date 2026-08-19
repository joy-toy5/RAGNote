from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.indexing.blob_store import LocalBlobStore
from app.indexing.models import (
    ContentBlob,
    DocumentRevision,
    IndexChunk,
    IndexedDocument,
)
from app.rag.indexing_service import UploadIndexingService


class FakeTransaction:
    def __init__(self, session: Session, events: list[str]) -> None:
        self.session = session
        self.events = events

    async def __aenter__(self) -> None:
        self.events.append("sql_begin")

    async def __aexit__(self, exc_type: object, *_: object) -> None:
        if exc_type is None:
            self.session.commit()
            self.events.append("sql_commit")
        else:
            self.session.rollback()


class FakeAsyncSession:
    def __init__(self, session: Session, events: list[str]) -> None:
        self.session = session
        self.events = events

    async def __aenter__(self) -> FakeAsyncSession:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self.session, self.events)

    async def run_sync(self, function: Any) -> Any:
        return function(self.session)


class FakeChroma:
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail
        self.writes: list[tuple[list[Any], list[str]]] = []

    def add_documents(self, documents: list[Any], *, ids: list[str]) -> None:
        self.events.append("chroma_write")
        if self.fail:
            raise RuntimeError("injected Chroma failure")
        self.writes.append((documents, ids))


class FakeVectorStore:
    def __init__(self, events: list[str], *, fail_chroma: bool = False) -> None:
        self.events = events
        self.vectors_store = FakeChroma(events, fail=fail_chroma)
        self.md5_values: set[str] = set()
        self.md5_by_filename: dict[str, str] = {}
        self.loads = 0

    async def check_md5_hex(self, value: str, user_id: str) -> bool:
        del user_id
        return value in self.md5_values

    async def get_md5_by_filename(self, user_id: str, filename: str) -> str | None:
        del user_id
        return self.md5_by_filename.get(filename)

    async def save_md5_hex(
        self,
        value: str,
        filename: str,
        original_filename: str,
        user_id: str,
    ) -> None:
        del original_filename, user_id
        self.events.append("md5_write")
        self.md5_values.add(value)
        self.md5_by_filename[filename] = value

    async def get_file_document(
        self,
        path: str,
        md5: str,
        user_id: str,
        *,
        source_filename: str,
    ) -> list[SimpleNamespace]:
        del md5, user_id
        self.events.append("parse_blob")
        self.loads += 1
        assert Path(path).read_bytes() == b"alpha beta"
        assert source_filename
        return [SimpleNamespace(page_content="alpha beta", metadata={"page": 0})]

    def split_documents_sync(
        self,
        documents: list[SimpleNamespace],
    ) -> list[SimpleNamespace]:
        metadata = documents[0].metadata
        return [
            SimpleNamespace(
                page_content="alpha",
                metadata={**metadata, "start_index": 0},
            ),
            SimpleNamespace(
                page_content="beta",
                metadata={**metadata, "start_index": 6},
            ),
        ]


@pytest.fixture
def database_session(backend_root: Path) -> Session:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        config = Config(str(backend_root / "alembic.ini"))
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    with Session(engine) as session:
        yield session
    engine.dispose()


def _session_factory(session: Session, events: list[str]) -> Any:
    return lambda: FakeAsyncSession(session, events)


def test_upload_writes_durable_facts_before_explicit_chroma_ids_and_md5(
    database_session: Session,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    vector_store = FakeVectorStore(events)
    indexer = UploadIndexingService(
        vector_store,
        blob_store=LocalBlobStore(tmp_path / "blobs"),
        session_factory=_session_factory(database_session, events),
        index_config={"splitter": "fake-v1"},
    )

    outcome = asyncio.run(
        indexer.index_upload(
            b"alpha beta",
            filename="guide.txt",
            user_id="user-a",
            media_type="text/plain",
        )
    )

    assert not outcome.duplicate
    assert outcome.facts is not None
    assert events == [
        "parse_blob",
        "sql_begin",
        "sql_commit",
        "chroma_write",
        "md5_write",
    ]
    documents, ids = vector_store.vectors_store.writes[0]
    assert ids == [chunk.chunk_id for chunk in outcome.facts.chunks]
    assert all(document.metadata["provenance_status"] == "verified" for document in documents)
    assert all(document.metadata["chunk_id"] in ids for document in documents)

    duplicate = asyncio.run(
        indexer.index_upload(
            b"alpha beta",
            filename="guide.txt",
            user_id="user-a",
            media_type="text/plain",
        )
    )
    assert duplicate.duplicate
    assert vector_store.loads == 1
    assert len(vector_store.vectors_store.writes) == 1


def test_same_blob_with_different_filename_creates_distinct_document(
    database_session: Session,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    vector_store = FakeVectorStore(events)
    indexer = UploadIndexingService(
        vector_store,
        blob_store=LocalBlobStore(tmp_path / "blobs"),
        session_factory=_session_factory(database_session, events),
        index_config={"splitter": "fake-v1"},
    )

    first = asyncio.run(
        indexer.index_upload(
            b"alpha beta",
            filename="first.txt",
            user_id="user-a",
            media_type="text/plain",
        )
    )
    second = asyncio.run(
        indexer.index_upload(
            b"alpha beta",
            filename="second.txt",
            user_id="user-a",
            media_type="text/plain",
        )
    )

    assert first.facts is not None
    assert second.facts is not None
    assert first.facts.blob_id == second.facts.blob_id
    assert first.facts.document_id != second.facts.document_id
    assert session_count(database_session, ContentBlob) == 1
    assert session_count(database_session, IndexedDocument) == 2
    assert len(vector_store.vectors_store.writes) == 2


def test_chroma_failure_keeps_committed_facts_but_does_not_write_md5(
    database_session: Session,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    vector_store = FakeVectorStore(events, fail_chroma=True)
    indexer = UploadIndexingService(
        vector_store,
        blob_store=LocalBlobStore(tmp_path / "blobs"),
        session_factory=_session_factory(database_session, events),
        index_config={"splitter": "fake-v1"},
    )

    with pytest.raises(RuntimeError, match="Chroma failure"):
        asyncio.run(
            indexer.index_upload(
                b"alpha beta",
                filename="guide.txt",
                user_id="user-a",
                media_type="text/plain",
            )
        )

    assert events[-3:] == ["sql_begin", "sql_commit", "chroma_write"]
    assert session_count(database_session, ContentBlob) == 1
    assert session_count(database_session, DocumentRevision) == 1
    assert session_count(database_session, IndexChunk) == 2
    assert vector_store.md5_values == set()


def session_count(session: Session, model: type[Any]) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0
