from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.indexing.blob_store import LocalBlobStore, StoredBlob
from app.indexing.contracts import CharacterSpan
from app.indexing.models import (
    ContentBlob,
    DocumentRevision,
    IndexChunk,
    IndexedDocument,
)
from app.indexing.repository import ChunkDraft, IndexingRequest, persist_index_facts

INDEX_CONFIG = {
    "embedding": {"model": "deterministic-test"},
    "splitter": {"chunk_overlap": 1, "chunk_size": 5},
}


def _request(
    blob_store: LocalBlobStore,
    *,
    content: bytes = b"alpha beta",
    display_name: str = "guide.txt",
) -> IndexingRequest:
    normalized_text = content.decode("ascii")
    return IndexingRequest(
        user_id="user-a",
        display_name=display_name,
        source_type="knowledge_base",
        media_type="text/plain",
        stored_blob=blob_store.put(content),
        normalized_text=normalized_text,
        normalized_text_blob=blob_store.put(normalized_text.encode("utf-8")),
        chunks=(
            ChunkDraft("alpha", CharacterSpan(0, 5), page_number=1),
            ChunkDraft("beta", CharacterSpan(6, 10), page_number=1),
        ),
        index_config=INDEX_CONFIG,
    )


@pytest.fixture
def session(backend_root: Path) -> Session:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        config = Config(str(backend_root / "alembic.ini"))
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    with Session(engine) as database_session:
        yield database_session
    engine.dispose()


def test_fact_write_is_idempotent_for_same_revision_and_index(
    session: Session,
    tmp_path: Path,
) -> None:
    request = _request(LocalBlobStore(tmp_path / "blobs"))

    first = persist_index_facts(session, request)
    session.commit()
    second = persist_index_facts(session, request)
    session.commit()

    assert first == second
    assert session.scalar(select(func.count()).select_from(ContentBlob)) == 1
    assert session.scalar(select(func.count()).select_from(IndexedDocument)) == 1
    assert session.scalar(select(func.count()).select_from(DocumentRevision)) == 1
    assert session.scalar(select(func.count()).select_from(IndexChunk)) == 2


def test_same_blob_with_different_name_keeps_distinct_document_identity(
    session: Session,
    tmp_path: Path,
) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")

    first = persist_index_facts(
        session, _request(blob_store, display_name="first.txt")
    )
    second = persist_index_facts(
        session, _request(blob_store, display_name="second.txt")
    )
    session.commit()

    assert first.blob_id == second.blob_id
    assert first.document_id != second.document_id
    assert session.scalar(select(func.count()).select_from(ContentBlob)) == 1
    assert session.scalar(select(func.count()).select_from(IndexedDocument)) == 2


def test_changed_content_increments_revision_and_changes_chunk_ids(
    session: Session,
    tmp_path: Path,
) -> None:
    blob_store = LocalBlobStore(tmp_path / "blobs")
    first = persist_index_facts(session, _request(blob_store))
    session.commit()
    second_request = IndexingRequest(
        user_id="user-a",
        display_name="guide.txt",
        source_type="knowledge_base",
        media_type="text/plain",
        stored_blob=blob_store.put(b"alpha gamma"),
        normalized_text="alpha gamma",
        normalized_text_blob=blob_store.put(b"alpha gamma"),
        chunks=(
            ChunkDraft("alpha", CharacterSpan(0, 5), page_number=1),
            ChunkDraft("gamma", CharacterSpan(6, 11), page_number=1),
        ),
        index_config=INDEX_CONFIG,
    )

    second = persist_index_facts(session, second_request)
    session.commit()

    assert first.document_id == second.document_id
    assert (first.document_revision, second.document_revision) == (1, 2)
    assert {chunk.chunk_id for chunk in first.chunks}.isdisjoint(
        chunk.chunk_id for chunk in second.chunks
    )


def test_fact_write_rejects_chunk_not_anchored_in_normalized_text(
    session: Session,
    tmp_path: Path,
) -> None:
    request = _request(LocalBlobStore(tmp_path / "blobs"))
    invalid = IndexingRequest(
        user_id=request.user_id,
        display_name=request.display_name,
        source_type=request.source_type,
        media_type=request.media_type,
        stored_blob=request.stored_blob,
        normalized_text=request.normalized_text,
        normalized_text_blob=request.normalized_text_blob,
        chunks=(ChunkDraft("wrong", CharacterSpan(0, 5)),),
        index_config=request.index_config,
    )

    with pytest.raises(ValueError, match="正文与声明"):
        persist_index_facts(session, invalid)

    assert session.scalar(select(func.count()).select_from(ContentBlob)) == 0


def test_fact_write_rejects_normalized_text_artifact_mismatch(
    session: Session,
    tmp_path: Path,
) -> None:
    request = _request(LocalBlobStore(tmp_path / "blobs"))
    invalid = replace(
        request,
        normalized_text_blob=StoredBlob(
            blob_id="f" * 64,
            byte_size=len(request.normalized_text.encode("utf-8")),
            storage_uri=f"cas+file://sha256/ff/ff/{'f' * 64}",
            path=tmp_path / "not-used",
        ),
    )

    with pytest.raises(ValueError, match="工件摘要"):
        persist_index_facts(session, invalid)
