"""单宿主存储目录护栏：只在临时目录验证真实锁与失败关闭。"""

from __future__ import annotations

import errno
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.core import storage_ownership as ownership
from app.core.storage_ownership import (
    StorageBusyError,
    StorageOwnership,
    StorageOwnershipError,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="仅支持 Linux 本地目录锁")


@pytest.fixture
def directory(tmp_path):
    directory = tmp_path / "store"
    directory.mkdir()
    (directory / "sentinel").write_bytes(b"must survive")
    return directory


@pytest.mark.parametrize("path", ["", ".", "relative/storage", Path("store")])
def test_relative_directory_is_rejected(path):
    with pytest.raises(ValueError):
        StorageOwnership(path)


def test_constructor_does_not_access_or_create_directory(tmp_path, monkeypatch):
    directory = tmp_path / "missing"
    with monkeypatch.context() as patch:
        patch.setattr(ownership.os, "open", lambda *a, **k: pytest.fail("构造不能访问存储"))
        guard = StorageOwnership(directory)
        assert not guard.acquired
    assert not directory.exists()
    with pytest.raises(FileNotFoundError):
        guard.acquire()
    assert not guard.acquired
    assert not directory.exists()


def test_file_is_not_a_storage_directory(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"unchanged")
    with pytest.raises(NotADirectoryError):
        StorageOwnership(path).acquire()
    assert path.read_bytes() == b"unchanged"


def test_explicit_lifecycle_and_reacquisition(directory):
    guard = StorageOwnership(directory)
    assert not guard.acquired
    with pytest.raises(StorageOwnershipError):
        guard.assert_owned()
    guard.release()
    assert guard.acquire() is guard
    assert guard.acquired
    guard.assert_owned()
    guard.release()
    guard.release()
    assert not guard.acquired
    with pytest.raises(StorageOwnershipError):
        guard.assert_owned()
    with guard as acquired:
        assert acquired is guard
        guard.assert_owned()
    assert not guard.acquired
    assert sorted(p.name for p in directory.iterdir()) == ["sentinel"]
    assert (directory / "sentinel").read_bytes() == b"must survive"


def test_reentrant_acquire_does_not_release_outer_ownership(directory):
    guard = StorageOwnership(directory)
    with guard:
        with pytest.raises(StorageOwnershipError):
            guard.acquire()
        with pytest.raises(StorageOwnershipError):
            with guard:
                pytest.fail("不能重复进入同一所有权上下文")
        guard.assert_owned()
        with pytest.raises(StorageBusyError):
            StorageOwnership(directory).acquire()


def test_distinct_instances_contend_without_initializing(directory):
    first, second = StorageOwnership(directory), StorageOwnership(directory)
    with first:
        with pytest.raises(StorageBusyError):
            with second:
                pytest.fail("未获锁不能初始化存储")
        assert not second.acquired
        first.assert_owned()
    with second:
        second.assert_owned()


def test_context_exception_releases_lock_without_swallowing(directory):
    with pytest.raises(ValueError, match="业务异常"):
        with StorageOwnership(directory):
            raise ValueError("业务异常")
    with StorageOwnership(directory) as guard:
        guard.assert_owned()


@pytest.mark.parametrize("alias_kind", ["symlink", "dotdot"])
def test_directory_alias_uses_same_inode(directory, alias_kind):
    if alias_kind == "symlink":
        alias = directory.parent / "alias"
        alias.symlink_to(directory, target_is_directory=True)
    else:
        alias = directory / ".." / directory.name
    with StorageOwnership(directory):
        with pytest.raises(StorageBusyError):
            StorageOwnership(alias).acquire()


def test_independent_directories_do_not_contend(directory, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    with StorageOwnership(directory), StorageOwnership(other) as guard:
        guard.assert_owned()


def test_replaced_directory_is_rejected_but_original_lock_is_retained(directory):
    moved = directory.with_name("moved")
    with StorageOwnership(directory) as guard:
        directory.rename(moved)
        directory.mkdir()
        with pytest.raises(StorageOwnershipError):
            guard.assert_owned()
        with pytest.raises(StorageBusyError):
            StorageOwnership(moved).acquire()
    with StorageOwnership(moved) as guard:
        guard.assert_owned()
    assert (moved / "sentinel").read_bytes() == b"must survive"


def test_missing_directory_is_rejected_without_releasing_lock(directory):
    moved = directory.with_name("moved")
    with StorageOwnership(directory) as guard:
        directory.rename(moved)
        with pytest.raises(OSError):
            guard.assert_owned()
        with pytest.raises(StorageBusyError):
            StorageOwnership(moved).acquire()


@pytest.mark.parametrize("error_code", [errno.EACCES, errno.EAGAIN])
def test_contention_closes_unowned_descriptor(directory, monkeypatch, error_code):
    descriptors = []
    real_open = ownership.os.open

    def open_directory(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    def busy(*args):
        raise OSError(error_code, "不可泄露的路径或输入")

    guard = StorageOwnership(directory)
    with monkeypatch.context() as patch:
        patch.setattr(ownership.os, "open", open_directory)
        patch.setattr(ownership.fcntl, "flock", busy)
        with pytest.raises(StorageBusyError) as error:
            guard.acquire()
        assert "不可泄露" not in str(error.value)
    assert not guard.acquired
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    with guard:
        guard.assert_owned()


@pytest.mark.parametrize("failure_point", ["flock", "fstat", "stat"])
def test_unexpected_acquire_failure_closes_descriptor(directory, monkeypatch, failure_point):
    descriptors = []
    real_open = ownership.os.open
    expected = OSError(errno.EIO, "受控IO失败")

    def open_directory(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    def fail(*args, **kwargs):
        raise expected

    target = ownership.fcntl if failure_point == "flock" else ownership.os
    guard = StorageOwnership(directory)
    with monkeypatch.context() as patch:
        patch.setattr(ownership.os, "open", open_directory)
        patch.setattr(target, failure_point, fail)
        with pytest.raises(OSError) as error:
            guard.acquire()
        assert error.value is expected
    assert not guard.acquired
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    with guard:
        guard.assert_owned()


def test_descriptor_is_non_inheritable_and_release_does_not_unlock(directory, monkeypatch):
    descriptors, operations = [], []
    real_open, real_flock = ownership.os.open, ownership.fcntl.flock

    def open_directory(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        descriptors.append(descriptor)
        assert not os.get_inheritable(descriptor)
        return descriptor

    def lock_directory(descriptor, operation):
        operations.append(operation)
        return real_flock(descriptor, operation)

    with monkeypatch.context() as patch:
        patch.setattr(ownership.os, "open", open_directory)
        patch.setattr(ownership.fcntl, "flock", lock_directory)
        with StorageOwnership(directory):
            pass
    assert operations == [ownership.fcntl.LOCK_EX | ownership.fcntl.LOCK_NB]
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_cold_import_has_no_storage_or_production_dependencies(backend_root):
    code = """
import os
import sys

def forbidden(*args, **kwargs):
    raise AssertionError("导入不能打开存储")

os.open = forbidden
from app.core.storage_ownership import StorageOwnership
assert not StorageOwnership("/does-not-exist").acquired
assert not {"dotenv", "chromadb", "sqlalchemy", "app.db.db_config"}.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=backend_root,
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_directory_replaced_during_acquire_is_rejected_and_descriptor_closed(directory, monkeypatch):
    moved = directory.with_name("moved")
    real_flock = ownership.fcntl.flock

    def swap_after_lock(descriptor, operation):
        real_flock(descriptor, operation)
        directory.rename(moved)
        directory.mkdir()

    guard = StorageOwnership(directory)
    with monkeypatch.context() as patch:
        patch.setattr(ownership.fcntl, "flock", swap_after_lock)
        with pytest.raises(StorageOwnershipError):
            guard.acquire()
    assert not guard.acquired
    with StorageOwnership(moved), guard:
        guard.assert_owned()


def test_open_permission_failure_is_not_misclassified_as_contention(directory, monkeypatch):
    expected = PermissionError(errno.EACCES, "受控目录权限错误")

    def deny(*args, **kwargs):
        raise expected

    guard = StorageOwnership(directory)
    with monkeypatch.context() as patch:
        patch.setattr(ownership.os, "open", deny)
        with pytest.raises(PermissionError) as error:
            guard.acquire()
        assert error.value is expected
    assert not guard.acquired
    with guard:
        guard.assert_owned()


def test_unsupported_platform_refuses_before_opening_directory(directory, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(ownership, "fcntl", None)
        patch.setattr(ownership.os, "open", lambda *a, **k: pytest.fail("不能退化为无锁访问"))
        guard = StorageOwnership(directory)
        with pytest.raises(NotImplementedError):
            guard.acquire()
        assert not guard.acquired
