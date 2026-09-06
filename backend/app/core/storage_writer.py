"""受管本地存储单写通道；尚未装配到生产入口。

单线程按顺序拥有初始化、写入、缓存失效和关闭。异步调用者只等待真实
concurrent Future，不拥有取消底层操作的权利。回调必须同步等待全部子工作；
本模块不提供租约/代次校验、不终止阻塞线程，也不约束绕过通道的写入口。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from app.core.storage_ownership import StorageOwnership

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class StorageWriterError(RuntimeError):
    """写通道当前不能安全接受或执行操作。"""


class StorageWriterFullError(StorageWriterError):
    """运行中和排队操作已占满容量；本次操作未被接受。"""


class StorageWriterState(str, Enum):
    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StorageShutdown:
    """一次关闭等待后的事实快照；失败或仍持锁不等于安全关闭。"""

    state: StorageWriterState
    pending: bool
    ownership_held: bool
    failed: bool


def _sync_callback(callback: Callable[[], Any]) -> None:
    target = getattr(callback, "__call__", callback)
    if not callable(callback) or any(
        inspect.iscoroutinefunction(item) or inspect.isasyncgenfunction(item)
        for item in (callback, target)
    ):
        raise TypeError("存储回调必须是同步可调用对象")


def _timeout_seconds(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("关闭等待时间必须是数值（不含布尔值）")
    try:
        seconds = float(value)
    except OverflowError:
        raise ValueError("关闭等待时间必须是可表示的有限非负数") from None
    if not 0 <= seconds < float("inf"):
        raise ValueError("关闭等待时间必须是有限非负数")
    return seconds


def _observe(future: asyncio.Future[Any]) -> None:
    """即使等待者已取消，也取走晚到异常；实际await仍能收到原异常。"""
    if not future.cancelled():
        future.exception()


def _wrap(future: Future[_T]) -> asyncio.Future[_T]:
    wrapped = asyncio.wrap_future(future)
    wrapped.add_done_callback(_observe)
    return wrapped


class StorageWriter:
    """单次生命期、有限接单；真实工作由线程和内部Future持续持有。

    构造无存储I/O或线程。所有公开操作先检查构造进程；内部状态由互斥量
    保护，但绝不持锁调用用户回调。shutdown不能强杀线程，必须在上层预算内
    检查返回值；未能关闭时由后续进程级停机协议处理，不用析构强行释放锁。
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        initialize: Callable[[], Any],
        invalidate: Callable[[], Any],
        close: Callable[[], Any],
        max_pending: int = 1,
    ) -> None:
        for callback in (initialize, invalidate, close):
            _sync_callback(callback)
        if isinstance(max_pending, bool) or not isinstance(max_pending, int):
            raise TypeError("写通道容量必须是整数（不含布尔值）")
        if max_pending < 1:
            raise ValueError("写通道容量必须大于零")
        self._ownership = StorageOwnership(directory)
        self._initialize_callback = initialize
        self._invalidate_callback = invalidate
        self._close_callback = close
        self._max_pending = max_pending
        self._pid = os.getpid()
        self._mutex = threading.RLock()
        self._queue: queue.Queue[tuple[Callable[[], Any], Future[Any]]] = queue.Queue()
        self._futures: set[Future[Any]] = set()
        self._writes: set[Future[Any]] = set()
        self._thread: threading.Thread | None = None
        self._closing: Future[None] | None = None
        self._state = StorageWriterState.NEW
        self._failed = False
        self._unmanaged: list[Any] = []

    @property
    def state(self) -> StorageWriterState:
        self._check_process()
        with self._mutex:
            return self._current_state()

    @property
    def pending(self) -> int:
        """真实操作数，不把已取消的异步等待者当作完成。"""
        self._check_process()
        with self._mutex:
            return sum(not future.done() for future in self._futures)

    @property
    def accepting(self) -> bool:
        return self.state == StorageWriterState.READY

    @property
    def ownership_held(self) -> bool:
        self._check_process()
        return self._ownership.acquired

    async def start(self) -> None:
        """先登记初始化Future再启动线程；取消等待不撤销已开始的初始化。"""
        self._check_waiter()
        with self._mutex:
            if self._state != StorageWriterState.NEW:
                raise StorageWriterError("写通道只能启动一次")
            self._state = StorageWriterState.STARTING
            future = self._enqueue(self._initialize)
            self._start_thread(future)
        await asyncio.shield(_wrap(future))

    async def run(self, write: Callable[[], _T]) -> _T:
        """非阻塞接单；已接受的write及失效始终作为一个真实操作完成。"""
        self._check_waiter()
        _sync_callback(write)
        with self._mutex:
            if self._state != StorageWriterState.READY:
                raise StorageWriterError("写通道未就绪或已停止接单")
            if sum(not future.done() for future in self._writes) >= self._max_pending:
                raise StorageWriterFullError("写通道容量已满，本次操作未接受")
            future = self._enqueue(lambda: self._write(write), writing=True)
        return await asyncio.shield(_wrap(future))

    async def shutdown(self, *, timeout: float = 5.0) -> StorageShutdown:
        """共享期限内等待真实收尾及线程退出；不取消排队/运行中的工作。"""
        self._check_waiter()
        seconds = _timeout_seconds(timeout)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + seconds
        future = self._begin_shutdown()
        if future is not None and not future.done():
            await asyncio.wait({_wrap(future)}, timeout=seconds)
        # Future完成早于线程最终返回；不在事件循环里调用阻塞join。
        while self._thread_alive() and loop.time() < deadline:
            await asyncio.sleep(min(0.001, max(0, deadline - loop.time())))
        result = self._shutdown_result()
        if result.pending:
            logger.warning("存储写通道关闭等待未收敛，仍有执行未证明停止")
        return result

    def _check_process(self) -> None:
        if os.getpid() != self._pid:
            raise StorageWriterError("fork后不能使用父进程的存储写通道")

    def _check_waiter(self) -> None:
        self._check_process()
        if threading.current_thread() is self._thread:
            raise StorageWriterError("存储回调不能递归等待自己的写通道")

    def _thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _current_state(self) -> StorageWriterState:
        if self._state == StorageWriterState.CLOSED and self._thread_alive():
            return StorageWriterState.DRAINING
        return self._state

    def _enqueue(
        self, callback: Callable[[], _T], *, writing: bool = False
    ) -> Future[_T]:
        """调用方持mutex；在入队前创建并登记唯一的真实执行凭据。"""
        future: Future[_T] = Future()
        self._futures.add(future)
        if writing:
            self._writes.add(future)
        future.add_done_callback(self._completed)
        self._queue.put_nowait((callback, future))
        return future

    def _start_thread(self, future: Future[None]) -> None:
        try:
            self._thread = threading.Thread(
                target=self._worker, name="storage-writer", daemon=False
            )
            self._thread.start()
        except BaseException as error:
            self._mark_failed()
            # 若原生线程已经开始，不伪造Future已完成或丢弃线程句柄。
            if self._thread is None or self._thread.ident is None:
                self._thread = None
                future.set_exception(error)
            raise

    def _worker(self) -> None:
        while True:
            callback, future = self._queue.get()
            if future.set_running_or_notify_cancel():
                self._execute(callback, future)
            if future is self._closing:
                return

    @staticmethod
    def _execute(callback: Callable[[], Any], future: Future[Any]) -> None:
        try:
            value = callback()
        except BaseException as error:
            future.set_exception(error)
        else:
            future.set_result(value)

    def _completed(self, future: Future[Any]) -> None:
        with self._mutex:
            self._futures.discard(future)
            self._writes.discard(future)
        if not future.cancelled() and future.exception() is not None:
            logger.error("存储写通道操作失败")

    def _mark_failed(self) -> None:
        with self._mutex:
            self._failed = True
            self._state = StorageWriterState.FAILED

    def _call(self, callback: Callable[[], _T]) -> _T:
        result = callback()
        if (
            inspect.iscoroutine(result)
            and inspect.getcoroutinestate(result) == inspect.CORO_CREATED
        ):
            result.close()
            self._mark_failed()
            raise TypeError("存储同步回调不能返回协程")
        if (
            inspect.isawaitable(result)
            or inspect.isasyncgen(result)
            or isinstance(result, Future)
        ):
            self._quarantine(result)
            raise TypeError("存储回调留下未受管执行，不能证明停止")
        return result

    def _quarantine(self, result: Any) -> None:
        """保留外部执行并观察可识别的异常；观察不是停止或交权证明。"""
        with self._mutex:
            self._unmanaged.append(result)
            self._mark_failed()
        if isinstance(result, asyncio.Future):
            try:
                # asyncio Future 的回调登记必须回到它自己的事件循环。
                result.get_loop().call_soon_threadsafe(
                    result.add_done_callback, _observe
                )
            except RuntimeError:
                # 循环已关闭时只读取已完成结果；未完成对象仍保守持有。
                if result.done():
                    _observe(result)

    def _initialize(self) -> None:
        try:
            self._ownership.acquire()
            self._call(self._initialize_callback)
        except BaseException:
            self._mark_failed()
            raise
        with self._mutex:
            if self._state == StorageWriterState.STARTING:
                self._state = StorageWriterState.READY

    def _write(self, write: Callable[[], _T]) -> _T:
        with self._mutex:
            if self._failed:
                raise StorageWriterError("写通道已有生命周期故障，拒绝排队写入")
        try:
            self._ownership.assert_owned()
        except BaseException:
            self._mark_failed()
            raise
        try:
            return self._call(write)
        finally:
            self._invalidate()

    def _invalidate(self) -> None:
        with self._mutex:
            if self._unmanaged:
                return
        try:
            self._call(self._invalidate_callback)
        except BaseException:
            self._mark_failed()
            raise

    def _begin_shutdown(self) -> Future[None] | None:
        with self._mutex:
            if self._closing is not None:
                return self._closing
            if self._thread is None:
                self._state = StorageWriterState.CLOSED
                return None
            self._state = StorageWriterState.DRAINING
            self._closing = self._enqueue(self._close_storage)
            return self._closing

    def _close_storage(self) -> None:
        try:
            with self._mutex:
                if self._unmanaged:
                    raise StorageWriterError("仍有未受管执行，禁止关闭资源及释放所有权")
            if self._ownership.acquired:
                self._call(self._close_callback)
                self._ownership.release()
        except BaseException:
            self._mark_failed()
            raise
        with self._mutex:
            self._state = StorageWriterState.CLOSED

    def _shutdown_result(self) -> StorageShutdown:
        with self._mutex:
            pending = self.pending > 0 or self._thread_alive() or bool(self._unmanaged)
            return StorageShutdown(
                self._current_state(), pending, self._ownership.acquired, self._failed
            )
