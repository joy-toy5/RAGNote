from __future__ import annotations

import json
from pathlib import Path

from scripts.m2_legacy_inventory import (
    build_legacy_inventory,
    read_image_directories,
    read_md5_records,
)


def test_legacy_inventory_hashes_content_without_fabricating_identity() -> None:
    report = build_legacy_inventory(
        {
            "ids": ["random-chroma-id"],
            "documents": ["private legacy body"],
            "metadatas": [
                {
                    "md5": "a" * 32,
                    "original_filename": "legacy.pdf",
                    "page": 0,
                    "user_id": "user-a",
                }
            ],
        },
        [{"md5": "a" * 32, "filename": "legacy.pdf", "user_id": "user-a"}],
        [
            {"file_count": 2, "md5": "a" * 32, "user_id": "user-a"},
            {"file_count": 1, "md5": "b" * 32, "user_id": "user-a"},
        ],
    )

    chunk = report["chunks"][0]
    assert chunk["legacy_chunk_id"] == "random-chroma-id"
    assert chunk["provenance_status"] == "legacy_index_only"
    assert chunk["blob_id"] is None
    assert chunk["document_id"] is None
    assert chunk["document_revision"] is None
    assert chunk["chunk_id"] is None
    assert chunk["index_version"] is None
    assert "private legacy body" not in json.dumps(report)
    assert report["summary"]["legacy_unresolved_image_directories"] == 1
    assert report["unresolved_image_directories"][0]["md5"] == "b" * 32


def test_legacy_inventory_reads_only_explicit_md5_and_image_roots(
    tmp_path: Path,
) -> None:
    md5_root = tmp_path / "md5"
    record_path = md5_root / "user_md5" / "user-a" / "md5_hex_store.txt"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps({"md5": "a" * 32, "filename": "guide.txt"}) + "\n",
        encoding="utf-8",
    )
    image_dir = tmp_path / "images" / "user-a" / ("a" * 32)
    image_dir.mkdir(parents=True)
    (image_dir / "p0_i0.png").write_bytes(b"image")

    records = read_md5_records(md5_root)
    images = read_image_directories(tmp_path / "images")

    assert records == [
        {"filename": "guide.txt", "md5": "a" * 32, "user_id": "user-a"}
    ]
    assert images == [{"file_count": 1, "md5": "a" * 32, "user_id": "user-a"}]
