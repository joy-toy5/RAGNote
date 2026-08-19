from __future__ import annotations

import math
import uuid
from dataclasses import replace

import pytest

from app.indexing.contracts import (
    CharacterSpan,
    ChunkProvenance,
    canonical_index_config,
    compute_blob_id,
    compute_chunk_id,
    compute_index_version,
    compute_text_sha256,
    next_document_revision,
    validate_document_id,
    validate_document_revision,
)

DOCUMENT_ID = "123e4567-e89b-42d3-a456-426614174000"
OTHER_DOCUMENT_ID = "123e4567-e89b-42d3-a456-426614174001"
INDEX_CONFIG = {
    "embedding": {"dimensions": 1024, "model": "text-embedding-v3"},
    "parser": "unstructured",
    "splitter": {"chunk_overlap": 100, "chunk_size": 500},
}
INDEX_VERSION = "99d8efedcb395f9b3de96424a258c1dbb35fd546e6bbbc2dd12986fcecb8deca"
BLOB_ID = "41cf6794ba4200b839c53531555f0f3998df4cbb01a4d5cb0b94e3ca5e23947d"
TEXT_SHA256 = "8a51b12c38b4fe10e5ad5222b112943b3d57c6661b34965e83e82a6bd48b3fb3"


def _provenance(**overrides: object) -> ChunkProvenance:
    values = {
        "document_id": DOCUMENT_ID,
        "document_revision": 1,
        "index_version": INDEX_VERSION,
        "chunk_ordinal": 0,
        "blob_id": BLOB_ID,
        "normalized_text_sha256": TEXT_SHA256,
        "character_span": CharacterSpan(0, 5),
        "page_number": 1,
    }
    values.update(overrides)
    return ChunkProvenance(**values)  # type: ignore[arg-type]


def test_identity_golden_values_are_stable() -> None:
    assert compute_blob_id(b"source") == BLOB_ID
    assert compute_text_sha256("Alpha beta") == TEXT_SHA256
    assert canonical_index_config(INDEX_CONFIG) == (
        b'{"config":{"embedding":{"dimensions":1024,"model":"text-embedding-v3"},'
        b'"parser":"unstructured","splitter":{"chunk_overlap":100,"chunk_size":500}},'
        b'"schema":"rag-note.index-config.v1"}'
    )
    assert compute_index_version(INDEX_CONFIG) == INDEX_VERSION
    assert compute_chunk_id("Alpha", _provenance()) == (
        "b30452c1b857e17e5e1ef49b45c8298c8fa5fb71ffd152f59ad174a59ac41402"
    )


def test_index_version_is_order_independent_and_unicode_canonical() -> None:
    reordered = {
        "splitter": {"chunk_size": 500, "chunk_overlap": 100},
        "parser": "unstructured",
        "embedding": {"model": "text-embedding-v3", "dimensions": 1024},
    }
    assert compute_index_version(reordered) == INDEX_VERSION
    assert compute_index_version({"label": "e\u0301"}) == compute_index_version({"label": "é"})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_index_config_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="NaN 或 Infinity"):
        compute_index_version({"value": value})


def test_index_config_rejects_non_json_values_and_normalized_duplicate_keys() -> None:
    with pytest.raises(TypeError, match="非 JSON 类型"):
        compute_index_version({"value": (1, 2)})
    with pytest.raises(ValueError, match="重复的键"):
        compute_index_version({"é": 1, "e\u0301": 2})


def test_persisted_document_id_must_be_canonical_nonzero_rfc4122_uuid() -> None:
    assert validate_document_id(DOCUMENT_ID) == DOCUMENT_ID
    assert validate_document_id(uuid.UUID(DOCUMENT_ID)) == DOCUMENT_ID
    for invalid in (DOCUMENT_ID.upper(), DOCUMENT_ID.replace("-", ""), str(uuid.UUID(int=0)), "not-a-uuid"):
        with pytest.raises(ValueError):
            validate_document_id(invalid)


def test_document_revision_starts_at_one_and_is_monotonic() -> None:
    assert next_document_revision(None) == 1
    assert next_document_revision(1) == 2
    assert validate_document_revision(4_294_967_295) == 4_294_967_295
    with pytest.raises(ValueError):
        validate_document_revision(0)
    with pytest.raises(TypeError):
        validate_document_revision(True)
    with pytest.raises(OverflowError):
        next_document_revision(4_294_967_295)


def test_same_blob_can_back_distinct_logical_documents() -> None:
    first = _provenance(document_id=DOCUMENT_ID)
    second = _provenance(document_id=OTHER_DOCUMENT_ID)
    assert first.blob_id == second.blob_id == compute_blob_id(b"source")
    assert compute_chunk_id("Alpha", first) != compute_chunk_id("Alpha", second)


@pytest.mark.parametrize(
    "changed",
    [
        replace(_provenance(), document_revision=2),
        replace(_provenance(), index_version="f" * 64),
        replace(_provenance(), chunk_ordinal=1),
        replace(_provenance(), character_span=CharacterSpan(1, 6)),
        replace(_provenance(), page_number=2),
        replace(_provenance(), blob_id="e" * 64),
    ],
)
def test_chunk_id_changes_when_identity_or_provenance_changes(changed: ChunkProvenance) -> None:
    assert compute_chunk_id("Alpha", changed) != compute_chunk_id("Alpha", _provenance())


def test_chunk_id_changes_when_content_changes() -> None:
    assert compute_chunk_id("Alpha!", _provenance()) != compute_chunk_id("Alpha", _provenance())


def test_page_and_character_coordinates_have_explicit_bases() -> None:
    assert CharacterSpan(0, 1) == CharacterSpan(start=0, end=1)
    assert _provenance(page_number=1).page_number == 1
    with pytest.raises(ValueError, match="0 <= start < end"):
        CharacterSpan(-1, 1)
    with pytest.raises(ValueError, match="0 <= start < end"):
        CharacterSpan(1, 1)
    with pytest.raises(ValueError, match="从 1 开始"):
        _provenance(page_number=0)


def test_legacy_index_only_allows_missing_blob_and_span_without_fabrication() -> None:
    legacy = _provenance(
        blob_id=None,
        normalized_text_sha256=None,
        character_span=None,
        page_number=None,
        legacy_index_only=True,
    )
    assert legacy.blob_id is None
    assert legacy.character_span is None
    assert len(compute_chunk_id("historical chunk", legacy)) == 64

    with pytest.raises(ValueError, match="非 legacy chunk"):
        _provenance(blob_id=None, normalized_text_sha256=None, character_span=None)
    with pytest.raises(ValueError, match="规范化全文摘要"):
        _provenance(
            normalized_text_sha256=None,
            character_span=CharacterSpan(0, 1),
            legacy_index_only=True,
        )
