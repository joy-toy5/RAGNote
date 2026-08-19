from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.evaluation.dataset import load_dataset
from app.evaluation.qrels import QrelsCompilationError, compile_qrels
from app.indexing.contracts import (
    CharacterSpan,
    ChunkProvenance,
    compute_chunk_id,
    compute_text_sha256,
)


def _smoke_manifest(backend_root: Path) -> Path:
    return backend_root / "evals/datasets/m3_smoke_v1/manifest.json"


def test_qrels_compile_from_stable_evidence_without_content_lookup(
    backend_root: Path,
) -> None:
    compiled = compile_qrels(load_dataset(_smoke_manifest(backend_root)))
    by_query = {entry.query_id: entry for entry in compiled.entries}

    assert set(by_query) == {"smoke-a-429", "smoke-b-worker"}
    assert by_query["smoke-a-429"].relevance == 3
    assert by_query["smoke-b-worker"].relevance == 3
    assert by_query["smoke-a-429"].anchor_id == "anchor-a-429"
    assert len(by_query["smoke-a-429"].chunk_ids) == 1
    assert len(compiled.fingerprint) == 64


def test_qrels_rejects_anchor_not_contained_by_any_chunk(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    source = _smoke_manifest(backend_root).parent
    target = tmp_path / source.name
    shutil.copytree(source, target)
    chunks_path = target / "chunks.jsonl"
    rows = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    text = (target / "corpus/user_a_retry_guide.txt").read_text(encoding="utf-8")
    content = text[:4]
    rows[0]["char_end"] = 4
    rows[0]["content_sha256"] = compute_text_sha256(content)
    rows[0]["chunk_id"] = compute_chunk_id(
        content,
        ChunkProvenance(
            document_id=rows[0]["document_id"],
            document_revision=rows[0]["document_revision"],
            index_version=rows[0]["index_version"],
            chunk_ordinal=rows[0]["chunk_ordinal"],
            blob_id="73010e054d920a65781c7483a9580b0e9743c9e89cfc18602ccc7f94c34ac1a4",
            normalized_text_sha256="73010e054d920a65781c7483a9580b0e9743c9e89cfc18602ccc7f94c34ac1a4",
            character_span=CharacterSpan(0, 4),
        ),
    )
    chunks_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(QrelsCompilationError, match="UNRESOLVED_ANCHOR"):
        compile_qrels(load_dataset(target / "manifest.json"))


def test_compiled_qrels_are_byte_deterministic(backend_root: Path) -> None:
    dataset = load_dataset(_smoke_manifest(backend_root))

    first = compile_qrels(dataset).to_json()
    second = compile_qrels(dataset).to_json()

    assert first == second


def test_qrels_group_overlapping_chunks_as_anchor_alternatives(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    source = _smoke_manifest(backend_root).parent
    target = tmp_path / source.name
    shutil.copytree(source, target)
    chunks_path = target / "chunks.jsonl"
    rows = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    overlapping = dict(rows[0])
    overlapping["chunk_ordinal"] = 1
    text = (target / "corpus/user_a_retry_guide.txt").read_text(encoding="utf-8")
    content = text[overlapping["char_start"] : overlapping["char_end"]]
    overlapping["chunk_id"] = compute_chunk_id(
        content,
        ChunkProvenance(
            document_id=overlapping["document_id"],
            document_revision=overlapping["document_revision"],
            index_version=overlapping["index_version"],
            chunk_ordinal=overlapping["chunk_ordinal"],
            blob_id="73010e054d920a65781c7483a9580b0e9743c9e89cfc18602ccc7f94c34ac1a4",
            normalized_text_sha256="73010e054d920a65781c7483a9580b0e9743c9e89cfc18602ccc7f94c34ac1a4",
            character_span=CharacterSpan(
                overlapping["char_start"], overlapping["char_end"]
            ),
        ),
    )
    rows.append(overlapping)
    chunks_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    compiled = compile_qrels(load_dataset(target / "manifest.json"))
    entry = next(item for item in compiled.entries if item.query_id == "smoke-a-429")

    assert entry.anchor_id == "anchor-a-429"
    assert len(entry.chunk_ids) == 2


def test_qrels_preserve_compound_evidence_set(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    source = _smoke_manifest(backend_root).parent
    target = tmp_path / source.name
    shutil.copytree(source, target)
    text = (target / "corpus/user_a_retry_guide.txt").read_text(encoding="utf-8")
    anchors_path = target / "evidence_anchors.jsonl"
    anchors = [
        json.loads(line) for line in anchors_path.read_text(encoding="utf-8").splitlines()
    ]
    second = dict(anchors[0])
    second["anchor_id"] = "anchor-a-429-second"
    second["char_start"] = 0
    second["char_end"] = 10
    second["evidence_text_sha256"] = compute_text_sha256(text[:10])
    anchors.append(second)
    anchors_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in anchors),
        encoding="utf-8",
    )
    qrels_path = target / "source_qrels.jsonl"
    qrels = [json.loads(line) for line in qrels_path.read_text(encoding="utf-8").splitlines()]
    qrels[0]["relevance"] = 2
    qrels[0]["evidence_set_id"] = "compound-a"
    qrels.insert(
        1,
        {
            "query_id": "smoke-a-429",
            "anchor_id": "anchor-a-429-second",
            "relevance": 2,
            "evidence_set_id": "compound-a",
        },
    )
    qrels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in qrels),
        encoding="utf-8",
    )

    compiled = compile_qrels(load_dataset(target / "manifest.json"))
    compound = [
        entry for entry in compiled.entries if entry.evidence_set_id == "compound-a"
    ]

    assert [entry.anchor_id for entry in compound] == [
        "anchor-a-429",
        "anchor-a-429-second",
    ]
    assert all(entry.relevance == 2 for entry in compound)
