"""新上传原件、MySQL 索引事实与 Chroma 派生写入编排。"""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.indexing.blob_store import LocalBlobStore, StoredBlob
from app.indexing.provenance import (
    anchor_source_documents,
    apply_verified_metadata,
    build_chunk_drafts,
    runtime_index_config,
)
from app.indexing.repository import (
    IndexingRequest,
    PersistedIndexFacts,
    persist_index_facts,
)
from app.utils.config import chroma_config
from app.utils.path_tool import get_abstract_path


@dataclass(frozen=True, slots=True)
class StagedUpload:
    user_id: str
    display_name: str
    media_type: str | None
    legacy_md5: str
    stored_blob: StoredBlob


@dataclass(frozen=True, slots=True)
class PreparedUpload:
    request: IndexingRequest
    documents: tuple[Any, ...]
    legacy_md5: str


@dataclass(frozen=True, slots=True)
class IndexingOutcome:
    duplicate: bool
    facts: PersistedIndexFacts | None = None


class UploadIndexingService:
    """维持原件 → SQL 事实 → Chroma → MD5 的明确写入顺序。"""

    def __init__(
        self,
        vector_store: Any,
        *,
        blob_store: LocalBlobStore | None = None,
        session_factory: Any = None,
        index_config: dict[str, object] | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._blob_store = blob_store or _default_blob_store()
        self._session_factory_override = session_factory
        self._index_config = index_config or runtime_index_config(chroma_config)

    def stage_upload(
        self,
        content: bytes,
        *,
        filename: str,
        user_id: str,
        media_type: str | None,
    ) -> StagedUpload:
        """先原子持久化原件，再返回可供解析的稳定输入。"""
        if not isinstance(content, bytes) or not content:
            raise ValueError("上传内容必须是非空 bytes")
        if not filename or not user_id:
            raise ValueError("filename 和 user_id 不能为空")
        stored_blob = self._blob_store.put(content)
        legacy_md5 = hashlib.md5(content, usedforsecurity=False).hexdigest()
        return StagedUpload(
            user_id=user_id,
            display_name=filename,
            media_type=media_type,
            legacy_md5=legacy_md5,
            stored_blob=stored_blob,
        )

    async def prepare_upload(self, staged: StagedUpload) -> PreparedUpload:
        """从持久 blob 异步解析、切片并建立字符锚点。"""
        sources = await self._vector_store.get_file_document(
            str(staged.stored_blob.path),
            staged.legacy_md5,
            staged.user_id,
            source_filename=staged.display_name,
        )
        return await asyncio.to_thread(self._prepare_sources, staged, sources)

    def prepare_upload_sync(self, staged: StagedUpload) -> PreparedUpload:
        """在线程池中从持久 blob 解析、切片并建立字符锚点。"""
        sources = self._vector_store.get_file_document_sync(
            str(staged.stored_blob.path),
            staged.legacy_md5,
            staged.user_id,
            source_filename=staged.display_name,
        )
        return self._prepare_sources(staged, sources)

    async def index_upload(
        self,
        content: bytes,
        *,
        filename: str,
        user_id: str,
        media_type: str | None,
    ) -> IndexingOutcome:
        """普通上传入口：durable stage 后执行完整事实与派生写入。"""
        staged = await asyncio.to_thread(
            self.stage_upload,
            content,
            filename=filename,
            user_id=user_id,
            media_type=media_type,
        )
        if await self._is_same_document_duplicate(
            staged.legacy_md5,
            staged.display_name,
            staged.user_id,
        ):
            return IndexingOutcome(duplicate=True)
        prepared = await self.prepare_upload(staged)
        return await self.persist_and_index(prepared)

    async def persist_and_index(self, prepared: PreparedUpload) -> IndexingOutcome:
        """提交 MySQL 事实后，以稳定 ID 幂等写入 Chroma。"""
        request = prepared.request
        if await self._is_same_document_duplicate(
            prepared.legacy_md5,
            request.display_name,
            request.user_id,
        ):
            return IndexingOutcome(duplicate=True)

        session_factory = self._session_factory()
        async with session_factory() as session:
            async with session.begin():
                facts = await session.run_sync(
                    lambda sync_session: persist_index_facts(sync_session, request)
                )

        documents = list(prepared.documents)
        ids = apply_verified_metadata(
            documents,
            request,
            facts,
            legacy_md5=prepared.legacy_md5,
        )
        await asyncio.to_thread(
            self._vector_store.vectors_store.add_documents,
            documents,
            ids=ids,
        )
        await self._vector_store.save_md5_hex(
            prepared.legacy_md5,
            request.display_name,
            request.display_name,
            request.user_id,
        )
        return IndexingOutcome(duplicate=False, facts=facts)

    def _prepare_sources(
        self,
        staged: StagedUpload,
        sources: list[Any],
    ) -> PreparedUpload:
        if not sources:
            raise ValueError("文件加载为空")
        normalized_text = anchor_source_documents(sources)
        normalized_text_blob = self._blob_store.put(normalized_text.encode("utf-8"))
        documents = tuple(self._vector_store.split_documents_sync(sources))
        chunks = build_chunk_drafts(documents, normalized_text)
        request = IndexingRequest(
            user_id=staged.user_id,
            display_name=staged.display_name,
            source_type="knowledge_base",
            media_type=staged.media_type,
            stored_blob=staged.stored_blob,
            normalized_text=normalized_text,
            normalized_text_blob=normalized_text_blob,
            chunks=chunks,
            index_config=self._index_config,
        )
        return PreparedUpload(
            request=request,
            documents=documents,
            legacy_md5=staged.legacy_md5,
        )

    def _session_factory(self) -> Any:
        if self._session_factory_override is not None:
            return self._session_factory_override
        from app.db.db_config import AsyncSessionLocal

        return AsyncSessionLocal

    async def _is_same_document_duplicate(
        self,
        legacy_md5: str,
        display_name: str,
        user_id: str,
    ) -> bool:
        existing_md5 = await self._vector_store.get_md5_by_filename(
            user_id,
            display_name,
        )
        return existing_md5 == legacy_md5


def _default_blob_store() -> LocalBlobStore:
    configured = os.getenv(
        "RAG_BLOB_STORE_PATH",
        str(chroma_config.get("blob_store_path", "data/blobs")),
    )
    root = Path(configured)
    if not root.is_absolute():
        root = Path(get_abstract_path(configured))
    return LocalBlobStore(root)
