"""本地 content-addressed blob store。"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.indexing.contracts import compute_blob_id, validate_blob_id


class BlobIntegrityError(RuntimeError):
    """持久化内容与其声明摘要不一致。"""


@dataclass(frozen=True, slots=True)
class StoredBlob:
    """已经原子发布的本地 blob。"""

    blob_id: str
    byte_size: int
    storage_uri: str
    path: Path


class LocalBlobStore:
    """只使用内容摘要派生相对路径的本地不可变 blob store。"""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise TypeError("root 必须是 pathlib.Path")
        self._root = root.resolve(strict=False)

    @property
    def root(self) -> Path:
        return self._root

    def put(self, content: bytes | bytearray | memoryview) -> StoredBlob:
        """落盘并原子发布；相同内容重复写入返回同一对象。"""
        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise TypeError("content 必须是 bytes-like 对象")
        payload = bytes(content)
        blob_id = compute_blob_id(payload)
        target = self.path_for(blob_id)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

        if target.exists():
            self._verify_target(target, blob_id, len(payload))
            return self._stored_blob(blob_id, len(payload), target)

        file_descriptor, temporary_name = tempfile.mkstemp(prefix=".blob-", dir=target.parent)
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(file_descriptor, 0o600)
            with os.fdopen(file_descriptor, "wb", closefd=True) as temporary_file:
                file_descriptor = -1
                temporary_file.write(payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            try:
                os.link(temporary_path, target)
                self._fsync_directory(target.parent)
            except FileExistsError:
                self._verify_target(target, blob_id, len(payload))
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            temporary_path.unlink(missing_ok=True)

        return self._stored_blob(blob_id, len(payload), target)

    def path_for(self, blob_id: str) -> Path:
        """仅从已校验摘要构造路径，不接受文件名或相对路径。"""
        digest = validate_blob_id(blob_id)
        return self._root / "sha256" / digest[:2] / digest[2:4] / digest

    @staticmethod
    def _stored_blob(blob_id: str, byte_size: int, path: Path) -> StoredBlob:
        storage_uri = f"cas+file://sha256/{blob_id[:2]}/{blob_id[2:4]}/{blob_id}"
        return StoredBlob(blob_id=blob_id, byte_size=byte_size, storage_uri=storage_uri, path=path)

    @staticmethod
    def _verify_target(target: Path, blob_id: str, byte_size: int) -> None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            file_descriptor = os.open(target, flags)
        except OSError as exc:
            raise BlobIntegrityError(f"无法安全读取既有 blob: {blob_id}") from exc

        digest = hashlib.sha256()
        try:
            metadata = os.fstat(file_descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != byte_size:
                raise BlobIntegrityError(f"既有 blob 大小或类型不一致: {blob_id}")
            with os.fdopen(file_descriptor, "rb", closefd=True) as existing_file:
                file_descriptor = -1
                for block in iter(lambda: existing_file.read(1024 * 1024), b""):
                    digest.update(block)
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)

        if digest.hexdigest() != blob_id:
            raise BlobIntegrityError(f"既有 blob 摘要不一致: {blob_id}")

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        file_descriptor = os.open(directory, flags)
        try:
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
