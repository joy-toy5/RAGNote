"""索引事实的同步 SQLAlchemy 事务适配器。"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.indexing.blob_store import StoredBlob
from app.indexing.contracts import (
    CharacterSpan,
    ChunkProvenance,
    canonical_index_config,
    compute_chunk_id,
    compute_index_version,
    compute_text_sha256,
    next_document_revision,
)
from app.indexing.models import (
    ContentBlob,
    DocumentRevision,
    IndexChunk,
    IndexedDocument,
    IndexVersion,
)


class IndexContractConflict(RuntimeError):
    """既有事实与同一稳定身份下的本次输入不一致。"""


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """写入事实表之前、已锚定规范化全文的 chunk。"""

    content: str
    character_span: CharacterSpan
    page_number: int | None = None


@dataclass(frozen=True, slots=True)
class IndexingRequest:
    """一次已完成原件持久化与文本解析的索引事实请求。"""

    user_id: str
    display_name: str
    source_type: Literal["knowledge_base", "note"]
    media_type: str | None
    stored_blob: StoredBlob
    normalized_text: str
    normalized_text_blob: StoredBlob
    chunks: tuple[ChunkDraft, ...]
    index_config: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PersistedChunk:
    chunk_id: str
    chunk_ordinal: int
    content_sha256: str
    character_span: CharacterSpan
    page_number: int | None


@dataclass(frozen=True, slots=True)
class PersistedIndexFacts:
    blob_id: str
    document_id: str
    document_revision: int
    index_version: str
    normalized_text_sha256: str
    normalized_text_uri: str
    source_uri: str
    chunks: tuple[PersistedChunk, ...]


def persist_index_facts(
    session: Session,
    request: IndexingRequest,
    *,
    document_id_factory: Callable[[], uuid.UUID] = uuid.uuid4,
) -> PersistedIndexFacts:
    """在调用方事务中幂等写入 blob、document、revision 和 chunk 事实。"""
    _validate_request(request)
    normalized_text_sha256 = request.normalized_text_blob.blob_id
    index_version = compute_index_version(request.index_config)

    _ensure_blob(session, request)
    document = _get_or_create_document(
        session,
        request,
        document_id_factory=document_id_factory,
    )
    revision = _get_or_create_revision(
        session,
        document.document_id,
        request.stored_blob.blob_id,
        normalized_text_sha256,
        request.normalized_text_blob.storage_uri,
    )
    _ensure_index_version(session, index_version, request.index_config)
    chunks = _build_persisted_chunks(
        request,
        document_id=document.document_id,
        document_revision=revision.revision,
        index_version=index_version,
        normalized_text_sha256=normalized_text_sha256,
    )
    _ensure_chunks(
        session,
        chunks,
        document_id=document.document_id,
        document_revision=revision.revision,
        index_version=index_version,
    )
    session.flush()

    return PersistedIndexFacts(
        blob_id=request.stored_blob.blob_id,
        document_id=document.document_id,
        document_revision=revision.revision,
        index_version=index_version,
        normalized_text_sha256=normalized_text_sha256,
        normalized_text_uri=request.normalized_text_blob.storage_uri,
        source_uri=request.stored_blob.storage_uri,
        chunks=chunks,
    )


def _validate_request(request: IndexingRequest) -> None:
    if not request.user_id.strip():
        raise ValueError("user_id 不能为空")
    if not request.display_name.strip():
        raise ValueError("display_name 不能为空")
    if request.source_type not in {"knowledge_base", "note"}:
        raise ValueError("source_type 不受支持")
    if not request.normalized_text:
        raise ValueError("规范化全文不能为空")
    normalized_bytes = request.normalized_text.encode("utf-8")
    if request.normalized_text_blob.blob_id != compute_text_sha256(
        request.normalized_text
    ):
        raise ValueError("规范化文本工件摘要与全文不一致")
    if request.normalized_text_blob.byte_size != len(normalized_bytes):
        raise ValueError("规范化文本工件字节数与全文不一致")
    if not request.chunks:
        raise ValueError("至少需要一个 chunk")

    for chunk in request.chunks:
        span = chunk.character_span
        if span.end > len(request.normalized_text):
            raise ValueError("chunk 字符区间超出规范化全文")
        if request.normalized_text[span.start : span.end] != chunk.content:
            raise ValueError("chunk 正文与声明的字符区间不一致")


def _ensure_blob(session: Session, request: IndexingRequest) -> None:
    blob = session.get(ContentBlob, request.stored_blob.blob_id)
    if blob is None:
        session.add(
            ContentBlob(
                blob_id=request.stored_blob.blob_id,
                byte_size=request.stored_blob.byte_size,
                media_type=request.media_type,
                storage_uri=request.stored_blob.storage_uri,
            )
        )
        return
    if blob.byte_size != request.stored_blob.byte_size:
        raise IndexContractConflict("相同 blob_id 的字节数不一致")
    if blob.storage_uri != request.stored_blob.storage_uri:
        raise IndexContractConflict("相同 blob_id 的持久 URI 不一致")


def _get_or_create_document(
    session: Session,
    request: IndexingRequest,
    *,
    document_id_factory: Callable[[], uuid.UUID],
) -> IndexedDocument:
    document = session.scalar(
        select(IndexedDocument).where(
            IndexedDocument.user_id == request.user_id,
            IndexedDocument.source_type == request.source_type,
            IndexedDocument.display_name == request.display_name,
        )
    )
    if document is not None:
        return document

    document = IndexedDocument(
        document_id=str(document_id_factory()),
        user_id=request.user_id,
        source_type=request.source_type,
        display_name=request.display_name,
    )
    session.add(document)
    session.flush()
    return document


def _get_or_create_revision(
    session: Session,
    document_id: str,
    blob_id: str,
    normalized_text_sha256: str,
    normalized_text_uri: str,
) -> DocumentRevision:
    latest = session.scalar(
        select(DocumentRevision)
        .where(DocumentRevision.document_id == document_id)
        .order_by(DocumentRevision.revision.desc())
        .limit(1)
    )
    if (
        latest is not None
        and latest.blob_id == blob_id
        and latest.normalized_text_sha256 == normalized_text_sha256
    ):
        if latest.normalized_text_uri != normalized_text_uri:
            raise IndexContractConflict("相同 document revision 的规范化文本 URI 不一致")
        return latest

    revision = DocumentRevision(
        document_id=document_id,
        revision=next_document_revision(None if latest is None else latest.revision),
        blob_id=blob_id,
        normalized_text_sha256=normalized_text_sha256,
        normalized_text_uri=normalized_text_uri,
        legacy_index_only=False,
    )
    session.add(revision)
    session.flush()
    return revision


def _ensure_index_version(
    session: Session,
    index_version: str,
    index_config: Mapping[str, object],
) -> None:
    canonical_config = json.loads(canonical_index_config(index_config))
    existing = session.get(IndexVersion, index_version)
    if existing is None:
        session.add(
            IndexVersion(
                index_version=index_version,
                contract_version=1,
                config_json=canonical_config,
            )
        )
        return
    if existing.config_json != canonical_config:
        raise IndexContractConflict("相同 index_version 的规范配置不一致")


def _build_persisted_chunks(
    request: IndexingRequest,
    *,
    document_id: str,
    document_revision: int,
    index_version: str,
    normalized_text_sha256: str,
) -> tuple[PersistedChunk, ...]:
    chunks = []
    for ordinal, chunk in enumerate(request.chunks):
        provenance = ChunkProvenance(
            document_id=document_id,
            document_revision=document_revision,
            index_version=index_version,
            chunk_ordinal=ordinal,
            blob_id=request.stored_blob.blob_id,
            normalized_text_sha256=normalized_text_sha256,
            character_span=chunk.character_span,
            page_number=chunk.page_number,
        )
        chunks.append(
            PersistedChunk(
                chunk_id=compute_chunk_id(chunk.content, provenance),
                chunk_ordinal=ordinal,
                content_sha256=compute_text_sha256(chunk.content),
                character_span=chunk.character_span,
                page_number=chunk.page_number,
            )
        )
    return tuple(chunks)


def _ensure_chunks(
    session: Session,
    chunks: tuple[PersistedChunk, ...],
    *,
    document_id: str,
    document_revision: int,
    index_version: str,
) -> None:
    existing = tuple(
        session.scalars(
            select(IndexChunk)
            .where(
                IndexChunk.document_id == document_id,
                IndexChunk.document_revision == document_revision,
                IndexChunk.index_version == index_version,
            )
            .order_by(IndexChunk.chunk_ordinal)
        )
    )
    if existing:
        expected = tuple(
            (
                chunk.chunk_id,
                chunk.chunk_ordinal,
                chunk.content_sha256,
                chunk.character_span.start,
                chunk.character_span.end,
                chunk.page_number,
            )
            for chunk in chunks
        )
        actual = tuple(
            (
                chunk.chunk_id,
                chunk.chunk_ordinal,
                chunk.content_sha256,
                chunk.char_start,
                chunk.char_end,
                chunk.page_number,
            )
            for chunk in existing
        )
        if actual != expected:
            raise IndexContractConflict("既有 chunk 事实与稳定身份输入不一致")
        return

    session.add_all(
        [
            IndexChunk(
                chunk_id=chunk.chunk_id,
                document_id=document_id,
                document_revision=document_revision,
                index_version=index_version,
                chunk_ordinal=chunk.chunk_ordinal,
                content_sha256=chunk.content_sha256,
                page_number=chunk.page_number,
                char_start=chunk.character_span.start,
                char_end=chunk.character_span.end,
                legacy_index_only=False,
            )
            for chunk in chunks
        ]
    )
