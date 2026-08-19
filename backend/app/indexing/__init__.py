"""索引身份、来源契约与持久化组件。"""

from app.indexing.contracts import (
    CharacterSpan,
    ChunkProvenance,
    canonical_index_config,
    compute_blob_id,
    compute_chunk_id,
    compute_index_version,
    compute_text_sha256,
    next_document_revision,
    validate_blob_id,
    validate_document_id,
    validate_document_revision,
    validate_index_version,
)

__all__ = [
    "CharacterSpan",
    "ChunkProvenance",
    "canonical_index_config",
    "compute_blob_id",
    "compute_chunk_id",
    "compute_index_version",
    "compute_text_sha256",
    "next_document_revision",
    "validate_blob_id",
    "validate_document_id",
    "validate_document_revision",
    "validate_index_version",
]
