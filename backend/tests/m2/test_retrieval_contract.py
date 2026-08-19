from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from app.rag.retrieval_contract import (
    EvidenceSpan,
    RetrievalTrace,
    StageObservation,
    candidate_from_document,
    single_index_version,
    validate_rerank_scores,
)

BLOB_ID = "a" * 64
CHUNK_ID = "b" * 64
INDEX_VERSION = "c" * 64
TEXT_SHA256 = "d" * 64
DOCUMENT_ID = "123e4567-e89b-42d3-a456-426614174000"


def _verified_metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "user_id": "user-a",
        "source_type": "knowledge_base",
        "original_filename": "guide.txt",
        "blob_id": BLOB_ID,
        "document_id": DOCUMENT_ID,
        "document_revision": 1,
        "chunk_id": CHUNK_ID,
        "index_version": INDEX_VERSION,
        "source_uri": f"cas+file://sha256/aa/aa/{BLOB_ID}",
        "page_number": 1,
        "char_start": 4,
        "char_end": 9,
        "source_text_sha256": TEXT_SHA256,
        "source_text_uri": f"cas+file://sha256/dd/dd/{TEXT_SHA256}",
    }
    metadata.update(overrides)
    return metadata


def _document(**metadata_overrides: object) -> SimpleNamespace:
    return SimpleNamespace(
        id="legacy-chroma-id",
        page_content="retry",
        metadata=_verified_metadata(**metadata_overrides),
    )


def test_verified_candidate_preserves_stable_identity_and_evidence() -> None:
    candidate = candidate_from_document(_document(), user_id="user-a", rank=2)

    assert candidate.provenance_status == "verified"
    assert candidate.candidate_id == CHUNK_ID
    assert candidate.document_id == DOCUMENT_ID
    assert candidate.evidence_spans == (
        EvidenceSpan(
            page_number=1,
            char_start=4,
            char_end=9,
            text_sha256=TEXT_SHA256,
            text_uri=f"cas+file://sha256/dd/dd/{TEXT_SHA256}",
        ),
    )
    assert candidate.stages == (
        StageObservation(stage="retrieval", route="legacy_hybrid", rank=2),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("blob_id", None),
        ("document_id", None),
        ("document_revision", None),
        ("chunk_id", None),
        ("index_version", None),
        ("source_uri", None),
        ("char_start", None),
        ("source_text_sha256", None),
        ("source_text_uri", None),
    ],
)
def test_partial_legacy_metadata_never_claims_verified_provenance(
    field: str,
    value: object,
) -> None:
    metadata = _verified_metadata()
    metadata[field] = value
    if field == "char_start":
        metadata["char_end"] = None

    candidate = candidate_from_document(
        SimpleNamespace(id="legacy-chroma-id", page_content="retry", metadata=metadata),
        user_id="user-a",
        rank=1,
    )

    assert candidate.provenance_status == "legacy_index_only"
    assert candidate.legacy_chunk_id == "legacy-chroma-id"
    assert candidate.chunk_id is None


def test_legacy_document_does_not_fabricate_blob_or_chunk_identity() -> None:
    document = SimpleNamespace(
        id="historical-random-id",
        page_content="legacy text",
        metadata={"user_id": "user-a", "original_filename": "lost.pdf"},
    )

    candidate = candidate_from_document(document, user_id="user-a", rank=1)

    assert candidate.provenance_status == "legacy_index_only"
    assert candidate.candidate_id == "historical-random-id"
    assert candidate.legacy_chunk_id == "historical-random-id"
    assert candidate.blob_id is None
    assert candidate.chunk_id is None


def test_legacy_zero_based_page_is_normalized_without_claiming_stable_id() -> None:
    document = SimpleNamespace(
        id=None,
        page_content="legacy text",
        metadata={
            "user_id": "user-a",
            "original_filename": "lost.pdf",
            "page": 0,
            "chunk_id": "f" * 64,
        },
    )

    candidate = candidate_from_document(document, user_id="user-a", rank=3)

    assert candidate.provenance_status == "legacy_index_only"
    assert candidate.candidate_id == "query-local:3"
    assert candidate.chunk_id is None
    assert candidate.evidence_spans == (EvidenceSpan(page_number=1),)


def test_candidate_rejects_cross_user_metadata() -> None:
    with pytest.raises(ValueError, match="用户与请求用户不一致"):
        candidate_from_document(_document(user_id="user-b"), user_id="user-a", rank=1)


def test_verified_candidate_rejects_empty_persistent_uri() -> None:
    with pytest.raises(ValueError, match="非空持久 URI"):
        candidate_from_document(_document(source_uri=""), user_id="user-a", rank=1)


@pytest.mark.parametrize(
    "span",
    [
        {"page_number": 0},
        {"page_number": True},
        {"char_start": 1, "char_end": None},
        {"char_start": -1, "char_end": 1},
        {"char_start": 1, "char_end": 1},
        {"char_start": True, "char_end": 1},
    ],
)
def test_evidence_span_rejects_ambiguous_coordinates(span: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        EvidenceSpan(**span)  # type: ignore[arg-type]


def test_stage_score_requires_direction_and_finite_value() -> None:
    with pytest.raises(ValueError, match="必须同时提供"):
        StageObservation(stage="rerank", route="cross_encoder", rank=1, raw_score=0.5)
    with pytest.raises(ValueError, match="有限值"):
        StageObservation(
            stage="rerank",
            route="cross_encoder",
            rank=1,
            raw_score=math.nan,
            score_direction="higher_is_better",
        )


def test_rerank_scores_reject_partial_and_nonfinite_results() -> None:
    with pytest.raises(ValueError, match="数量"):
        validate_rerank_scores([0.1], expected_count=2)
    with pytest.raises(ValueError, match="有限实数"):
        validate_rerank_scores([math.nan], expected_count=1)
    assert validate_rerank_scores([1, 0.25], expected_count=2) == (1.0, 0.25)


def test_candidate_and_trace_updates_are_immutable() -> None:
    candidate = candidate_from_document(_document(), user_id="user-a", rank=1)
    reranked = candidate.observe(
        StageObservation(
            stage="rerank",
            route="cross_encoder",
            rank=1,
            raw_score=0.75,
            score_direction="higher_is_better",
        )
    )
    selected = reranked.select_for_context()
    trace = RetrievalTrace(
        query_id="query-1",
        user_id="user-a",
        candidates=(selected,),
        index_version=INDEX_VERSION,
    )

    assert candidate.stages != reranked.stages
    assert not candidate.selected_for_context
    assert selected.selected_for_context
    assert trace.to_dict()["candidates"][0]["chunk_id"] == CHUNK_ID
    with pytest.raises(FrozenInstanceError):
        selected.cited = True  # type: ignore[misc]


def test_trace_rejects_cross_user_candidates_and_does_not_claim_mixed_version() -> None:
    first = candidate_from_document(_document(), user_id="user-a", rank=1)
    second = candidate_from_document(
        _document(index_version="e" * 64, chunk_id="f" * 64),
        user_id="user-a",
        rank=2,
    )
    assert single_index_version([first, second]) is None

    cross_user = candidate_from_document(
        _document(user_id="user-b"),
        user_id="user-b",
        rank=1,
    )
    with pytest.raises(ValueError, match="其他用户"):
        RetrievalTrace(
            query_id="query-1",
            user_id="user-a",
            candidates=(cross_user,),
        )
