from __future__ import annotations

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.indexing.blob_store import BlobIntegrityError, LocalBlobStore
from app.indexing.contracts import compute_blob_id


def test_blob_store_publishes_content_at_digest_only_path(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "blobs")
    content = b"immutable source bytes"
    digest = compute_blob_id(content)

    stored = store.put(content)

    assert stored.blob_id == digest
    assert stored.byte_size == len(content)
    assert stored.path == tmp_path / "blobs" / "sha256" / digest[:2] / digest[2:4] / digest
    assert stored.storage_uri == f"cas+file://sha256/{digest[:2]}/{digest[2:4]}/{digest}"
    assert stored.path.read_bytes() == content
    assert stat.S_IMODE(stored.path.stat().st_mode) == 0o600


def test_blob_store_is_idempotent_and_leaves_no_staging_files(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "blobs")

    first = store.put(b"same content")
    second = store.put(bytearray(b"same content"))

    assert first == second
    assert list(first.path.parent.glob(".blob-*")) == []


def test_blob_store_concurrent_publication_is_atomic(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "blobs")
    content = b"concurrent immutable content" * 1024

    with ThreadPoolExecutor(max_workers=8) as executor:
        stored = list(executor.map(store.put, [content] * 32))

    assert {item.path for item in stored} == {stored[0].path}
    assert stored[0].path.read_bytes() == content
    assert list(stored[0].path.parent.glob(".blob-*")) == []


def test_blob_store_never_overwrites_corrupted_existing_content(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "blobs")
    stored = store.put(b"original")
    stored.path.write_bytes(b"tampered")

    with pytest.raises(BlobIntegrityError, match="摘要不一致"):
        store.put(b"original")

    assert stored.path.read_bytes() == b"tampered"


def test_blob_store_rejects_path_like_ids_and_existing_symlinks(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "blobs")
    with pytest.raises(ValueError, match="64 位小写"):
        store.path_for("../outside")

    content = b"symlink target"
    digest = compute_blob_id(content)
    target = store.path_for(digest)
    target.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(content)
    target.symlink_to(outside)

    with pytest.raises(BlobIntegrityError, match="安全读取"):
        store.put(content)
    assert os.path.islink(target)


def test_blob_store_rejects_non_bytes_payload(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "blobs")
    with pytest.raises(TypeError, match="bytes-like"):
        store.put("not bytes")  # type: ignore[arg-type]
