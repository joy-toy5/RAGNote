"""Linux 单宿主本地存储目录的协作式独占护栏，尚未接入生产。

组合根应先 acquire，再初始化存储；全部真实写入、缓存失效和存储关闭后才
release。护栏不终止线程、不证明外部执行停止，也不能约束未遵守协议的入口。
目录及父路径必须由可信部署维护，持有期间不得替换或删除。
"""

from __future__ import annotations

import errno
import os
import sys
import threading
from pathlib import Path
from typing import Self

if sys.platform == "linux":
    import fcntl
else:
    fcntl = None


class StorageOwnershipError(RuntimeError):
    """当前实例不能证明持有指定存储目录。"""


class StorageBusyError(StorageOwnershipError):
    """目录已有协作式所有者；不得初始化或抢占存储。"""


class StorageOwnership:
    """显式、不可重入的目录所有权；无锁文件、无构造期存储 I/O。

    仅支持 Linux 单宿主本地目录；不是 NFS/SMB 分布式锁。构造进程内的操作
    由互斥量串行化，fork 后必须使用新实例。遗忘 release 时保守持锁到进程退出，
    不以析构、PID 文件或超时推断写者已结束。
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = Path(directory)
        if not self._directory.is_absolute():
            raise ValueError("存储目录必须是绝对路径")
        self._pid = os.getpid()
        self._fd: int | None = None
        self._mutex = threading.Lock()

    @property
    def acquired(self) -> bool:
        """只描述本进程的句柄状态；实际使用前仍需 assert_owned。"""
        if os.getpid() != self._pid:
            return False
        with self._mutex:
            return self._fd is not None

    def acquire(self) -> Self:
        """非阻塞取锁；目录须预先存在，竞争或异常不进入初始化阶段。"""
        self._check_process()
        if fcntl is None:
            raise NotImplementedError("存储所有权护栏仅支持 Linux 本地目录")
        with self._mutex:
            if self._fd is not None:
                raise StorageOwnershipError("存储所有权不可重复获取")
            self._fd = self._open_locked_directory()
        return self

    def assert_owned(self) -> None:
        """拒绝未持有、跨进程或已更换目录；不是路径替换的原子防护。"""
        self._check_process()
        with self._mutex:
            if self._fd is None:
                raise StorageOwnershipError("尚未取得存储所有权")
            self._check_directory(self._fd)

    def release(self) -> None:
        """调用方证明执行已停止后显式关闭；不得对共享描述符执行 LOCK_UN。"""
        self._check_process()
        with self._mutex:
            descriptor, self._fd = self._fd, None
            if descriptor is not None:
                # fork 子进程仍持有同一打开文件描述时，保守保留内核锁。
                os.close(descriptor)

    def __enter__(self) -> Self:
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        self.release()

    def _check_process(self) -> None:
        # 必须在访问 mutex 之前检查，避免 fork 复制了其他线程持有的锁。
        if os.getpid() != self._pid:
            raise StorageOwnershipError("fork 后不能使用父进程的存储所有权实例")

    def _open_locked_directory(self) -> int:
        descriptor = os.open(self._directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            self._lock_directory(descriptor)
            self._check_directory(descriptor)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    @staticmethod
    def _lock_directory(descriptor: int) -> None:
        assert fcntl is not None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN):
                raise StorageBusyError("存储目录已有所有者，禁止初始化") from None
            raise

    def _check_directory(self, descriptor: int) -> None:
        locked = os.fstat(descriptor)
        current = os.stat(self._directory)
        if (locked.st_dev, locked.st_ino) != (current.st_dev, current.st_ino):
            raise StorageOwnershipError("存储目录已更换，不能继续使用旧所有权")
