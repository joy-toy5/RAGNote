"""同步生产者到异步消费者的有界桥接；关闭表示放弃剩余结果。"""

import asyncio
import queue
import threading
import time
from typing import Any


class QueueClosed(RuntimeError):
    """消费者已经退出，不再接收或等待结果。"""


class TaskQueue:
    def __init__(self, maxsize: int = 2):
        if maxsize < 1:
            raise ValueError("队列容量必须大于零")
        self._queue = queue.Queue(maxsize=maxsize)
        self._closed = threading.Event()
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def put(self, item: Any, block: bool = True, timeout: float | None = None):
        """有界等待空间，消费者关闭后唤醒生产线程。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                if self.closed:
                    raise QueueClosed("上传结果队列已关闭")
                try:
                    self._queue.put_nowait(item)
                    return
                except queue.Full:
                    if not block or (
                        deadline is not None and time.monotonic() >= deadline
                    ):
                        raise
            self._closed.wait(0.01)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                if self.closed:
                    raise QueueClosed("上传结果队列已关闭")
                try:
                    return self._queue.get_nowait()
                except queue.Empty:
                    if not block or (
                        deadline is not None and time.monotonic() >= deadline
                    ):
                        raise
            self._closed.wait(0.01)

    async def get_async(self, timeout: float | None = None) -> Any:
        """只捕获暂时无结果，不在线程池里留下不可取消的 get。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                return self.get(block=False)
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("等待上传解析结果超时") from None
                await asyncio.sleep(0.01)

    def close(self) -> None:
        """幂等关闭，丢弃未领取结果；已领取结果仍由消费者 task_done。"""
        with self._lock:
            self._closed.set()
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                self._queue.task_done()

    def task_done(self) -> None:
        self._queue.task_done()

    def join(self) -> None:
        self._queue.join()

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        return self._queue.full()
