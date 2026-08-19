from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.evaluation.contracts import deterministic_document_id
from app.evaluation.dataset import DatasetValidationError, dataset_fingerprint, load_dataset
from app.indexing.blob_store import LocalBlobStore
from app.indexing.contracts import CharacterSpan
from app.indexing.repository import ChunkDraft, IndexingRequest, persist_index_facts


def _smoke_manifest(backend_root: Path) -> Path:
    return backend_root / "evals/datasets/m3_smoke_v1/manifest.json"


def _jsonl_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_smoke_dataset_has_deterministic_verified_identity(backend_root: Path) -> None:
    dataset = load_dataset(_smoke_manifest(backend_root))

    assert dataset.split == "smoke"
    assert len(dataset.corpus) == 2
    assert len(dataset.chunks) == 2
    assert len(dataset.queries) == 4
    assert len(dataset.source_qrels) == 2
    assert len(dataset_fingerprint(dataset)) == 64

    for item in dataset.corpus:
        assert item.document_id == deterministic_document_id(
            dataset.document_namespace,
            item.corpus_item_id,
        )


def test_document_identity_is_stable_but_namespaced() -> None:
    first_namespace = uuid.UUID("7ee74bb2-3da8-4e70-9a60-ed1b90f3288a")
    second_namespace = uuid.UUID("0fb8995c-0168-418b-8b30-dbc47d27e850")

    first = deterministic_document_id(first_namespace, "retry-guide")

    assert first == deterministic_document_id(first_namespace, "retry-guide")
    assert first != deterministic_document_id(first_namespace, "other-guide")
    assert first != deterministic_document_id(second_namespace, "retry-guide")


def test_dataset_rejects_tampered_normalized_text(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    source = _smoke_manifest(backend_root).parent
    target = tmp_path / source.name
    shutil.copytree(source, target)
    normalized_path = target / "corpus/user_a_retry_guide.txt"
    normalized_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="字节数|SHA-256"):
        load_dataset(target / "manifest.json")


def test_dataset_rejects_arbitrary_document_id(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    source = _smoke_manifest(backend_root).parent
    target = tmp_path / source.name
    shutil.copytree(source, target)
    corpus_path = target / "corpus.jsonl"
    rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["document_id"] = str(uuid.uuid4())
    corpus_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="确定性 document_id"):
        load_dataset(target / "manifest.json")


def test_dataset_rejects_non_integer_relevance(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    source = _smoke_manifest(backend_root).parent
    target = tmp_path / source.name
    shutil.copytree(source, target)
    qrels_path = target / "source_qrels.jsonl"
    rows = [json.loads(line) for line in qrels_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["relevance"] = 2.5
    qrels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="relevance"):
        load_dataset(target / "manifest.json")


def test_dataset_rejects_path_escape(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    source = _smoke_manifest(backend_root).parent
    target = tmp_path / source.name
    shutil.copytree(source, target)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["corpus"] = "../outside.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="路径越界"):
        load_dataset(manifest_path)


def test_dataset_rejects_absolute_internal_path(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    source = _smoke_manifest(backend_root).parent
    target = tmp_path / source.name
    shutil.copytree(source, target)
    corpus_path = target / "corpus.jsonl"
    rows = _jsonl_rows(corpus_path)
    rows[0]["blob_path"] = str(target / "corpus/user_a_retry_guide.txt")
    _write_jsonl(corpus_path, rows)

    with pytest.raises(DatasetValidationError, match="相对路径"):
        load_dataset(target / "manifest.json")


def test_dataset_rejects_duplicate_repository_document_key(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    source = _smoke_manifest(backend_root).parent
    target = tmp_path / source.name
    shutil.copytree(source, target)
    corpus_path = target / "corpus.jsonl"
    rows = _jsonl_rows(corpus_path)
    for field in ("user_id", "source_type", "display_name"):
        rows[1][field] = rows[0][field]
    _write_jsonl(corpus_path, rows)

    with pytest.raises(DatasetValidationError, match="source_type, display_name"):
        load_dataset(target / "manifest.json")


def test_dataset_rejects_revision_without_frozen_history(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    source = _smoke_manifest(backend_root).parent
    target = tmp_path / source.name
    shutil.copytree(source, target)
    corpus_path = target / "corpus.jsonl"
    rows = _jsonl_rows(corpus_path)
    rows[0]["document_revision"] = 2
    _write_jsonl(corpus_path, rows)

    with pytest.raises(DatasetValidationError, match="revision=1"):
        load_dataset(target / "manifest.json")


def test_dataset_rejects_empty_query_id_at_dataset_boundary(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    source = _smoke_manifest(backend_root).parent
    target = tmp_path / source.name
    shutil.copytree(source, target)
    queries_path = target / "queries.jsonl"
    queries = _jsonl_rows(queries_path)
    queries[0]["query_id"] = ""
    _write_jsonl(queries_path, queries)
    qrels_path = target / "source_qrels.jsonl"
    qrels = _jsonl_rows(qrels_path)
    qrels[0]["query_id"] = ""
    _write_jsonl(qrels_path, qrels)

    with pytest.raises(DatasetValidationError, match="query_id"):
        load_dataset(target / "manifest.json")


def test_answerable_query_rejects_background_only_evidence(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    source = _smoke_manifest(backend_root).parent
    target = tmp_path / source.name
    shutil.copytree(source, target)
    qrels_path = target / "source_qrels.jsonl"
    qrels = _jsonl_rows(qrels_path)
    qrels[0]["relevance"] = 1
    _write_jsonl(qrels_path, qrels)

    with pytest.raises(DatasetValidationError, match="完整 relevance=2"):
        load_dataset(target / "manifest.json")


def test_relevance_two_requires_complete_evidence_set(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    source = _smoke_manifest(backend_root).parent
    target = tmp_path / source.name
    shutil.copytree(source, target)
    qrels_path = target / "source_qrels.jsonl"
    qrels = _jsonl_rows(qrels_path)
    qrels[0]["relevance"] = 2
    qrels[0]["evidence_set_id"] = "compound-a"
    _write_jsonl(qrels_path, qrels)

    with pytest.raises(DatasetValidationError, match="至少需要两个"):
        load_dataset(target / "manifest.json")


def test_dataset_rejects_page_number_without_page_map(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    source = _smoke_manifest(backend_root).parent
    target = tmp_path / source.name
    shutil.copytree(source, target)
    chunks_path = target / "chunks.jsonl"
    chunks = _jsonl_rows(chunks_path)
    chunks[0]["page_number"] = 1
    _write_jsonl(chunks_path, chunks)

    with pytest.raises(DatasetValidationError, match="page map"):
        load_dataset(target / "manifest.json")


def test_evaluation_document_id_uses_repository_factory_seam(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    dataset = load_dataset(_smoke_manifest(backend_root))
    item = next(item for item in dataset.corpus if item.corpus_item_id == "retry-guide-a")
    text = (dataset.root / item.normalized_text_path).read_text(encoding="utf-8")
    blob_store = LocalBlobStore(tmp_path / "blobs")
    stored = blob_store.put(text.encode("utf-8"))
    request = IndexingRequest(
        user_id=item.user_id,
        display_name=item.display_name,
        source_type=item.source_type,
        media_type="text/plain",
        stored_blob=stored,
        normalized_text=text,
        normalized_text_blob=stored,
        chunks=(ChunkDraft(text, CharacterSpan(0, len(text))),),
        index_config=dataset.index_config,
    )
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        config = Config(str(backend_root / "alembic.ini"))
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    with Session(engine) as session:
        facts = persist_index_facts(
            session,
            request,
            document_id_factory=lambda: uuid.UUID(item.document_id),
        )
        assert facts.document_id == item.document_id
        assert facts.chunks[0].chunk_id == dataset.chunks[0].chunk_id
    engine.dispose()
