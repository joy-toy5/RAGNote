"""进程内上传执行边界；保留未结束线程的所有权，不提供持久恢复。"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from app.rag.task_queue import TaskQueue

logger = logging.getLogger(__name__)


class UploadBusy(RuntimeError):
    """上传容量已满或正在关闭。"""


class UploadLease:
    """请求退出后，仍持有实际执行中的解析线程和索引写入。"""

    def __init__(self, runtime: UploadRuntime):
        self._runtime = runtime
        self.queue = TaskQueue(maxsize=2)
        self._futures: set[Future] = set()
        self._writes: set[asyncio.Task] = set()
        self._closed = False

    def submit(self, function: Callable, *args) -> Future:
        with self._runtime._lock:
            if self._closed or self._runtime._closing:
                raise UploadBusy("上传执行器正在关闭")
            future = self._runtime._get_executor().submit(function, *args)
            self._futures.add(future)
            future.add_done_callback(self._thread_done)
            return future

    def track_write(self, task: asyncio.Task) -> None:
        with self._runtime._lock:
            self._writes.add(task)
            task.add_done_callback(self._write_done)

    def _thread_done(self, future: Future) -> None:
        with self._runtime._lock:
            if not future.cancelled() and future.exception() is not None:
                logger.error("上传解析线程异常结束；请求消费者将报告失败")
            self._futures.discard(future)
            self._release_if_done()

    def _write_done(self, task: asyncio.Task) -> None:
        with self._runtime._lock:
            if task.cancelled():
                # 无法证明 to_thread 已结束：隔离到进程退出，不能回收写入容量。
                logger.warning("索引协程被外部取消，底层写入状态未知；执行容量保持隔离")
                return
            if task.exception() is not None:
                logger.error("上传索引写入异常结束；已提交事实不会自动回滚或重试")
            self._writes.discard(task)
            self._release_if_done()

    def _release_if_done(self) -> None:
        if self._closed and not self._futures and not self._writes:
            self._runtime._leases.discard(self)

    def close(self) -> None:
        with self._runtime._lock:
            self._closed = True
            self.queue.close()
            for future in tuple(self._futures):
                future.cancel()
            # 正在写入的协程不取消：其中的 to_thread 无法被协程取消终止。
            self._release_if_done()


class UploadRuntime:
    """共享惰性线程池；接入容量涵盖读取、解析以及断开后的残留执行。"""

    def __init__(self, *, max_workers: int = 4, max_uploads: int = 2):
        if min(max_workers, max_uploads) < 1:
            raise ValueError("上传线程和请求容量必须大于零")
        self._max_workers = max_workers
        self._max_uploads = max_uploads
        self._executor: ThreadPoolExecutor | None = None
        self._leases: set[UploadLease] = set()
        self._lock = threading.RLock()
        self._closing = False
        self._accepting = True

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._leases)

    def start(self) -> None:
        with self._lock:
            if self._leases:
                raise RuntimeError("旧上传执行未结束，不能重开执行器")
            self._closing = False
            self._accepting = True

    def stop_accepting(self) -> None:
        """仅停止新批次；已接受批次仍可完成解析/写入。"""
        with self._lock:
            self._accepting = False

    def acquire(self) -> UploadLease:
        with self._lock:
            if not self._accepting or self._closing or len(self._leases) >= self._max_uploads:
                raise UploadBusy("上传处理容量已满或正在关闭，请稍后重试")
            lease = UploadLease(self)
            self._leases.add(lease)
            return lease

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="upload-slice"
            )
        return self._executor

    async def shutdown(self, *, timeout: float = 5.0) -> int:
        with self._lock:
            self._accepting = False
            self._closing = True
            for lease in tuple(self._leases):
                lease.close()
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
        deadline = asyncio.get_running_loop().time() + max(0, timeout)
        while self.active_count and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        remaining = self.active_count
        if remaining:
            logger.warning(
                "上传关闭等待超时，仍有 %s 个执行占用容量；未声明线程已终止", remaining
            )
        return remaining


upload_runtime = UploadRuntime()
