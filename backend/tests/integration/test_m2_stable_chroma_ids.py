from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.rag.retrieval_contract import candidate_from_document

USER_ID = "user-a"
BLOB_ID = "a" * 64
CHUNK_ID = "b" * 64
INDEX_VERSION = "c" * 64
TEXT_SHA256 = "d" * 64
DOCUMENT_ID = "123e4567-e89b-42d3-a456-426614174000"


class DeterministicEmbeddings:
    @staticmethod
    def _embed(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255 for byte in digest[:16]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.mark.integration
def test_real_chroma_round_trips_stable_id_and_verified_provenance(
    tmp_path: Path,
) -> None:
    store = Chroma(
        collection_name="m2_stable_ids",
        embedding_function=DeterministicEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
    )
    document = Document(
        page_content="HTTP 429 should use bounded exponential backoff.",
        metadata={
            "blob_id": BLOB_ID,
            "char_end": 48,
            "char_start": 0,
            "chunk_id": CHUNK_ID,
            "document_id": DOCUMENT_ID,
            "document_revision": 1,
            "index_version": INDEX_VERSION,
            "original_filename": "retry-guide.txt",
            "page_number": 1,
            "source_text_sha256": TEXT_SHA256,
            "source_text_uri": f"cas+file://sha256/dd/dd/{TEXT_SHA256}",
            "source_type": "knowledge_base",
            "source_uri": f"cas+file://sha256/aa/aa/{BLOB_ID}",
            "user_id": USER_ID,
        },
    )

    store.add_documents([document], ids=[CHUNK_ID])
    store.add_documents([document], ids=[CHUNK_ID])

    stored = store.get(ids=[CHUNK_ID], include=["documents", "metadatas"])
    assert stored["ids"] == [CHUNK_ID]
    assert stored["metadatas"][0]["chunk_id"] == CHUNK_ID
    results = store.similarity_search(
        "bounded HTTP retry",
        k=3,
        filter={"user_id": USER_ID},
    )
    assert len(results) == 1
    assert results[0].id == CHUNK_ID

    candidate = candidate_from_document(results[0], user_id=USER_ID, rank=1)
    assert candidate.provenance_status == "verified"
    assert candidate.chunk_id == CHUNK_ID
    assert candidate.evidence_spans[0].char_start == 0
    assert candidate.evidence_spans[0].char_end == 48
