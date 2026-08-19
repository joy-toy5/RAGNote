from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.indexing.blob_store import StoredBlob
from app.indexing.provenance import (
    anchor_source_documents,
    apply_verified_metadata,
    build_chunk_drafts,
    runtime_index_config,
)
from app.indexing.repository import (
    IndexingRequest,
    PersistedChunk,
    PersistedIndexFacts,
)
from app.rag.text_spliter import AsyncTextSplitter


def _document(content: str, **metadata: object) -> SimpleNamespace:
    return SimpleNamespace(page_content=content, metadata=dict(metadata))


def test_splitter_chunks_map_to_global_right_open_character_spans() -> None:
    sources = [_document("alpha beta", page=0), _document("gamma", page=1)]
    normalized_text = anchor_source_documents(sources)
    splitter = AsyncTextSplitter(
        chunk_size=5,
        chunk_overlap=0,
        separators=[" ", ""],
    )

    chunks = splitter.split_documents_sync(sources)
    drafts = build_chunk_drafts(chunks, normalized_text)

    assert normalized_text == "alpha beta\ngamma"
    assert [draft.content for draft in drafts] == ["alpha", "beta", "gamma"]
    assert [
        (draft.character_span.start, draft.character_span.end)
        for draft in drafts
    ] == [(0, 5), (6, 10), (11, 16)]
    assert [draft.page_number for draft in drafts] == [1, 1, 2]


def test_verified_metadata_uses_persisted_facts_and_explicit_chunk_ids() -> None:
    documents = [_document("alpha", start_index=0, _rag_note_source_char_base=0)]
    stored_blob = StoredBlob(
        blob_id="a" * 64,
        byte_size=5,
        storage_uri=f"cas+file://sha256/aa/aa/{'a' * 64}",
        path=Path("/unused/test/blob"),
    )
    request = IndexingRequest(
        user_id="user-a",
        display_name="guide.txt",
        source_type="knowledge_base",
        media_type="text/plain",
        stored_blob=stored_blob,
        normalized_text="alpha",
        normalized_text_blob=stored_blob,
        chunks=build_chunk_drafts(documents, "alpha"),
        index_config={"splitter": "test"},
    )
    facts = PersistedIndexFacts(
        blob_id=stored_blob.blob_id,
        document_id="123e4567-e89b-42d3-a456-426614174000",
        document_revision=1,
        index_version="b" * 64,
        normalized_text_sha256="c" * 64,
        normalized_text_uri=f"cas+file://sha256/cc/cc/{'c' * 64}",
        source_uri=stored_blob.storage_uri,
        chunks=(
            PersistedChunk(
                chunk_id="d" * 64,
                chunk_ordinal=0,
                content_sha256="e" * 64,
                character_span=request.chunks[0].character_span,
                page_number=None,
            ),
        ),
    )

    ids = apply_verified_metadata(documents, request, facts, legacy_md5="legacy")

    assert ids == ["d" * 64]
    assert documents[0].metadata == {
        "blob_id": "a" * 64,
        "char_end": 5,
        "char_start": 0,
        "chunk_id": "d" * 64,
        "chunk_ordinal": 0,
        "document_id": "123e4567-e89b-42d3-a456-426614174000",
        "document_revision": 1,
        "index_version": "b" * 64,
        "md5": "legacy",
        "original_filename": "guide.txt",
        "provenance_status": "verified",
        "source": stored_blob.storage_uri,
        "source_text_sha256": "c" * 64,
        "source_text_uri": f"cas+file://sha256/cc/cc/{'c' * 64}",
        "source_type": "knowledge_base",
        "source_uri": stored_blob.storage_uri,
        "user_id": "user-a",
    }


def test_runtime_index_config_excludes_secrets_and_tracks_model_choices() -> None:
    config = runtime_index_config(
        {"chunk_size": 200, "chunk_overlap": 20, "separators": ["\n", ""]},
        {
            "EMBED_MODEL_TYPE": "ALIYUN",
            "ALIYUN_EMBED_MODEL_NAME": "embed-v2",
            "ALIYUN_ACCESS_KEY_SECRET": "must-not-leak",
            "VISION_MODEL_TYPE": "OLLAMA",
            "VISION_OLLAMA_MODEL_NAME": "vision-v3",
        },
    )

    assert config["embedding"] == {"model": "embed-v2", "provider": "ALIYUN"}
    assert config["parser"] == {
        "pipeline": "document-processor.v1",
        "vision_model": "vision-v3",
        "vision_provider": "OLLAMA",
    }
    assert "must-not-leak" not in repr(config)
