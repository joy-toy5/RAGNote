from __future__ import annotations

from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable

from app.indexing.models import (
    ContentBlob,
    DocumentRevision,
    IndexChunk,
    IndexedDocument,
    IndexVersion,
)
from app.models.chat_history import Base


def _constraint_names(model: type[Base]) -> set[str]:  # type: ignore[valid-type]
    return {constraint.name for constraint in model.__table__.constraints if constraint.name}


def test_index_models_share_the_existing_metadata() -> None:
    expected = {
        "content_blobs",
        "documents",
        "document_revisions",
        "index_versions",
        "index_chunks",
    }
    assert expected <= set(Base.metadata.tables)
    assert all(model.metadata is Base.metadata for model in (
        ContentBlob,
        IndexedDocument,
        DocumentRevision,
        IndexVersion,
        IndexChunk,
    ))


def test_same_blob_can_back_multiple_document_rows() -> None:
    blob_id = "a" * 64
    first = IndexedDocument(
        document_id="123e4567-e89b-42d3-a456-426614174000",
        user_id="user-1",
        source_type="knowledge_base",
        display_name="first.pdf",
    )
    second = IndexedDocument(
        document_id="123e4567-e89b-42d3-a456-426614174001",
        user_id="user-1",
        source_type="knowledge_base",
        display_name="second.pdf",
    )
    revisions = [
        DocumentRevision(document_id=first.document_id, revision=1, blob_id=blob_id),
        DocumentRevision(document_id=second.document_id, revision=1, blob_id=blob_id),
    ]

    assert first.document_id != second.document_id
    assert {revision.blob_id for revision in revisions} == {blob_id}


def test_schema_declares_named_identity_foreign_key_and_unique_constraints() -> None:
    assert "uq_documents_owner_source_name" in _constraint_names(IndexedDocument)
    assert "uq_document_revisions_identity_legacy" in _constraint_names(DocumentRevision)
    assert {
        "fk_index_chunks_document_revision",
        "uq_index_chunks_position",
        "ck_index_chunks_character_span",
        "ck_index_chunks_nonlegacy_span",
    } <= _constraint_names(IndexChunk)
    assert {foreign_key.target_fullname for foreign_key in IndexChunk.__table__.foreign_keys} == {
        "document_revisions.document_id",
        "document_revisions.revision",
        "document_revisions.legacy_index_only",
        "index_versions.index_version",
    }


def test_mysql_ddl_uses_binary_ascii_identity_and_innodb() -> None:
    dialect = mysql.dialect()
    document_ddl = str(CreateTable(IndexedDocument.__table__).compile(dialect=dialect))
    revision_ddl = str(CreateTable(DocumentRevision.__table__).compile(dialect=dialect))
    chunk_ddl = str(CreateTable(IndexChunk.__table__).compile(dialect=dialect))

    assert "COLLATE ascii_bin" in document_ddl
    assert "user_id VARCHAR(64) COLLATE ascii_bin" in document_ddl
    assert "ENGINE=InnoDB" in document_ddl
    assert "revision INTEGER UNSIGNED" in revision_ddl
    assert "FOREIGN KEY(blob_id) REFERENCES content_blobs" in revision_ddl
    assert "document_revision INTEGER UNSIGNED" in chunk_ddl
    assert "FOREIGN KEY(index_version) REFERENCES index_versions" in chunk_ddl


def test_m2_models_do_not_smuggle_worker_state_into_the_contract() -> None:
    forbidden = {"status", "retry_count", "lease_owner", "heartbeat_at", "outbox_id"}
    for model in (ContentBlob, IndexedDocument, DocumentRevision, IndexVersion, IndexChunk):
        assert forbidden.isdisjoint(model.__table__.columns.keys())
