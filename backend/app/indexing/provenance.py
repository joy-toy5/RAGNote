"""解析文档、chunk 字符位置与持久事实之间的适配。"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from app.indexing.contracts import CharacterSpan
from app.indexing.repository import (
    ChunkDraft,
    IndexingRequest,
    PersistedIndexFacts,
)

_SOURCE_CHAR_BASE = "_rag_note_source_char_base"
_SPLITTER_START_INDEX = "start_index"


def anchor_source_documents(documents: Sequence[Any]) -> str:
    """用单个换行拼接规范化全文，并给每个源 Document 写入全局基址。"""
    if not documents:
        raise ValueError("源文档不能为空")

    parts: list[str] = []
    cursor = 0
    for index, document in enumerate(documents):
        content = _document_content(document)
        metadata = _document_metadata(document)
        metadata[_SOURCE_CHAR_BASE] = cursor
        parts.append(content)
        cursor += len(content)
        if index < len(documents) - 1:
            cursor += 1

    normalized_text = "\n".join(parts)
    if not normalized_text:
        raise ValueError("规范化全文不能为空")
    return normalized_text


def build_chunk_drafts(
    documents: Sequence[Any],
    normalized_text: str,
) -> tuple[ChunkDraft, ...]:
    """把 splitter 的局部 start_index 转成规范化全文的全局字符区间。"""
    drafts = []
    for document in documents:
        content = _document_content(document)
        metadata = _document_metadata(document)
        source_base = _required_nonnegative_int(metadata, _SOURCE_CHAR_BASE)
        local_start = _required_nonnegative_int(metadata, _SPLITTER_START_INDEX)
        start = source_base + local_start
        span = CharacterSpan(start=start, end=start + len(content))
        if span.end > len(normalized_text):
            raise ValueError("chunk 字符区间超出规范化全文")
        if normalized_text[span.start : span.end] != content:
            raise ValueError("splitter start_index 无法锚定 chunk 正文")
        drafts.append(
            ChunkDraft(
                content=content,
                character_span=span,
                page_number=_page_number(metadata),
            )
        )
    if not drafts:
        raise ValueError("切片结果不能为空")
    return tuple(drafts)


def apply_verified_metadata(
    documents: Sequence[Any],
    request: IndexingRequest,
    facts: PersistedIndexFacts,
    *,
    legacy_md5: str,
) -> list[str]:
    """把 SQL 事实写入 Chroma metadata，并返回同序稳定 ID。"""
    if len(documents) != len(facts.chunks):
        raise ValueError("Document 数量与持久 chunk 事实不一致")

    ids = []
    for document, chunk in zip(documents, facts.chunks, strict=True):
        metadata = _document_metadata(document)
        metadata.pop(_SOURCE_CHAR_BASE, None)
        metadata.pop(_SPLITTER_START_INDEX, None)
        metadata.update(
            {
                "blob_id": facts.blob_id,
                "char_end": chunk.character_span.end,
                "char_start": chunk.character_span.start,
                "chunk_id": chunk.chunk_id,
                "chunk_ordinal": chunk.chunk_ordinal,
                "document_id": facts.document_id,
                "document_revision": facts.document_revision,
                "index_version": facts.index_version,
                "md5": legacy_md5,
                "original_filename": request.display_name,
                "provenance_status": "verified",
                "source_text_sha256": facts.normalized_text_sha256,
                "source_text_uri": facts.normalized_text_uri,
                "source_type": request.source_type,
                "source": facts.source_uri,
                "source_uri": facts.source_uri,
                "user_id": request.user_id,
            }
        )
        if chunk.page_number is not None:
            metadata["page_number"] = chunk.page_number
        ids.append(chunk.chunk_id)
    return ids


def runtime_index_config(
    chroma_settings: Mapping[str, object],
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """提取会改变索引结果的非密钥运行配置。"""
    values = os.environ if environment is None else environment
    embedding_provider = values.get("EMBED_MODEL_TYPE", "OLLAMA").upper()
    if embedding_provider == "ALIYUN":
        embedding_model = values.get("ALIYUN_EMBED_MODEL_NAME", "qwen3-embedding")
    else:
        embedding_model = values.get(
            "TEXT_EMBEDDING_MODEL_NAME",
            "qwen3-embedding:0.6b",
        )

    vision_provider = (
        values.get("VISION_MODEL_TYPE", "").upper()
        or values.get("LLM_TYPE", "ALIYUN").upper()
    )
    if vision_provider == "OLLAMA":
        vision_model = values.get("VISION_OLLAMA_MODEL_NAME", "qwen-vl:7b")
    else:
        vision_model = values.get(
            "VISION_CHAT_MODEL_NAME",
            values.get("CHAT_MODEL_NAME", "qwen3-max"),
        )

    return {
        "contract": "rag-note.index-pipeline.v1",
        "embedding": {
            "model": embedding_model,
            "provider": embedding_provider,
        },
        "parser": {
            "pipeline": "document-processor.v1",
            "vision_model": vision_model,
            "vision_provider": vision_provider,
        },
        "splitter": {
            "chunk_overlap": chroma_settings["chunk_overlap"],
            "chunk_size": chroma_settings["chunk_size"],
            "length_unit": "unicode_codepoint",
            "separators": chroma_settings["separators"],
            "type": "recursive_character",
        },
    }


def _document_content(document: Any) -> str:
    content = getattr(document, "page_content", None)
    if not isinstance(content, str):
        raise TypeError("Document.page_content 必须是字符串")
    return content


def _document_metadata(document: Any) -> MutableMapping[str, Any]:
    metadata = getattr(document, "metadata", None)
    if not isinstance(metadata, MutableMapping):
        raise TypeError("Document.metadata 必须是可变映射")
    return metadata


def _required_nonnegative_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Document.metadata 缺少有效的 {key}")
    return value


def _page_number(metadata: Mapping[str, Any]) -> int | None:
    page_number = metadata.get("page_number")
    if page_number is not None:
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            raise TypeError("page_number 必须是整数")
        if page_number < 1:
            raise ValueError("page_number 必须从 1 开始")
        return page_number

    zero_based_page = metadata.get("page")
    if zero_based_page is None:
        return None
    if isinstance(zero_based_page, bool) or not isinstance(zero_based_page, int):
        raise TypeError("page 必须是 0-based 整数")
    if zero_based_page < 0:
        raise ValueError("page 必须大于等于 0")
    return zero_based_page + 1
