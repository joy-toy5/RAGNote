"""受管持久任务 polling 生命周期；不执行业务或授予外部存储写权。

只负责 claim_task → commit → dispatch。dispatch 必须拥有并等待其本次执行，
不得仅创建失管后台任务后返回；输入、心跳、结算与真实执行停止由执行层负责。
没有 dispatch 时禁止领取；过期 processing 不自动恢复。本模块尚未装配到生产。
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.exc import DBAPIError, DisconnectionError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeout
from sqlalchemy.ext.asyncio import AsyncSession

from app.tasking import leases
from app.tasking.lease_contracts import (
    TaskLease,
    validate_lease_owner,
    validate_lease_seconds,
    validate_task_kinds,
)

logger = logging.getLogger(__name__)
Dispatch = Callable[[TaskLease], Awaitable[None]]
SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]
MAX_POLL_SECONDS = 3600
# 连接数耗尽、锁超时/死锁、连接建立失败或已断开；不按错误文本猜测。
_MYSQL_TRANSIENT_ERRNOS = frozenset({1040, 1205, 1213, 2002, 2003, 2006, 2013})


class PollingState(str, Enum):
    """polling 状态；STOPPED 只表示循环结束，不是外部执行停止证明。"""

    NEW = "new"
    UNCONFIGURED = "unconfigured"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


class PollingConfigurationError(RuntimeError):
    """polling 缺少安全启动所需的 dispatch。"""


@dataclass(frozen=True, slots=True)
class PollingShutdown:
    """有界等待的事实结果；pending 只跟踪本实例的 polling task。"""

    state: PollingState
    pending: bool


def _seconds(value: float, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} 必须是数值（不含布尔值）")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError(f"{label} 必须是可表示的有限非负数") from None
    if not 0 <= value < float("inf"):
        raise ValueError(f"{label} 必须是有限非负数")
    if not allow_zero and value == 0:
        raise ValueError(f"{label} 必须大于零")
    return value


def _poll_seconds(value: float, label: str) -> float:
    value = _seconds(value, label)
    if value > MAX_POLL_SECONDS:
        raise ValueError(f"{label} 不能超过 {MAX_POLL_SECONDS} 秒")
    return value


def _is_transient_storage_error(error: BaseException) -> bool:
    """只重试已辨识的存储故障；权限/schema 和未知错误立即停止。"""
    if isinstance(error, (DisconnectionError, PoolTimeout)):
        return True
    if isinstance(error, DBAPIError) and error.connection_invalidated:
        return True
    if not isinstance(error, OperationalError):
        return False
    sqlite_code = getattr(error.orig, "sqlite_errorcode", None)
    if type(sqlite_code) is int:
        return sqlite_code & 0xFF in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
    args = getattr(error.orig, "args", ())
    return bool(args) and type(args[0]) is int and args[0] in _MYSQL_TRANSIENT_ERRNOS


class PollingRuntime:
    """单事件循环、单实例、单次启动；持有串行 claim/dispatch 的任务句柄。"""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        owner: str,
        kinds: tuple[str, ...],
        dispatch: Dispatch | None,
        lease_seconds: int = 90,
        idle_initial: float = 1.0,
        idle_max: float = 30.0,
        error_initial: float = 1.0,
        error_max: float = 30.0,
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory 必须可调用")
        if dispatch is not None and not callable(dispatch):
            raise TypeError("dispatch 必须可调用或为 None")
        validate_lease_owner(owner)
        validate_task_kinds(kinds)
        validate_lease_seconds(lease_seconds)
        self._idle_initial = _poll_seconds(idle_initial, "idle_initial")
        self._idle_max = _poll_seconds(idle_max, "idle_max")
        self._error_initial = _poll_seconds(error_initial, "error_initial")
        self._error_max = _poll_seconds(error_max, "error_max")
        if self._idle_max < self._idle_initial:
            raise ValueError("idle_max 不能小于 idle_initial")
        if self._error_max < self._error_initial:
            raise ValueError("error_max 不能小于 error_initial")
        self._session_factory = session_factory
        self._owner = owner
        self._kinds = kinds
        self._dispatch_callback = dispatch
        self._lease_seconds = lease_seconds
        self._state = (
            PollingState.NEW if dispatch is not None else PollingState.UNCONFIGURED
        )
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    @property
    def state(self) -> PollingState:
        task = self._task
        # 首步前取消不会进入协程 finally；不能依赖 done callback 已被调度。
        if task is not None and task.done() and self._state != PollingState.FAILED:
            if not task.cancelled() and task.exception() is not None:
                return PollingState.FAILED
            return PollingState.STOPPED
        return self._state

    @property
    def accepting(self) -> bool:
        return self.state == PollingState.RUNNING

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    def start(self) -> None:
        """失败时关闭尚未移交的协程；支持立即执行的 task factory。"""
        if self.state == PollingState.UNCONFIGURED:
            raise PollingConfigurationError("没有 dispatch，禁止领取持久任务")
        if self.state != PollingState.NEW:
            raise RuntimeError(f"polling 不能从 {self.state.value} 启动")
        asyncio.get_running_loop()
        coroutine = self._run_loop()
        self._state = PollingState.RUNNING
        try:
            self._task = asyncio.create_task(coroutine, name="task-polling")
        except BaseException:
            coroutine.close()
            self._state = PollingState.NEW
            raise
        self._task.add_done_callback(self._on_done)

    def stop_accepting(self) -> None:
        """关闭领取闸门并唤醒退避；不取消已经进入的 claim 或 dispatch。"""
        if self.state == PollingState.NEW:
            self._state = PollingState.STOPPED
        elif self.state == PollingState.RUNNING:
            self._state = PollingState.DRAINING
        self._stop_event.set()

    async def shutdown(self, *, timeout: float = 5.0) -> PollingShutdown:
        """共享调用方期限等待循环；超时/调用者取消都不取消在途工作。"""
        timeout = _seconds(timeout, "shutdown timeout", allow_zero=True)
        self.stop_accepting()
        task = self._task
        if task is not None and not task.done():
            await asyncio.wait({task}, timeout=timeout)
        pending = task is not None and not task.done()
        if pending:
            logger.warning("持久任务 polling 关闭等待超时，仍有 1 个受管循环未结束")
        return PollingShutdown(self.state, pending=pending)

    def _on_done(self, task: asyncio.Task[None]) -> None:
        """只在句柄结束后收敛状态并回收异常，不输出原始异常或任务输入。"""
        error = None if task.cancelled() else task.exception()
        if error is not None:
            self._state = PollingState.FAILED
            logger.error("持久任务 polling 异常退出，已停止领取")
        elif self._state != PollingState.FAILED:
            self._state = PollingState.STOPPED

    async def _run_loop(self) -> None:
        try:
            # 即使 eager task factory 也先让出，使 start 完成句柄登记。
            await asyncio.sleep(0)
            await self._poll()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._state = PollingState.FAILED
            logger.error("持久任务 polling 或 dispatch 失败，已停止领取")

    async def _poll(self) -> None:
        idle_delay = self._idle_initial
        error_delay = self._error_initial
        dispatch = self._dispatch_callback
        assert dispatch is not None
        while self.accepting:
            try:
                lease = await self._claim_once()
            except Exception as error:
                if not _is_transient_storage_error(error):
                    raise
                logger.warning("持久任务 polling 暂时失败，进入错误退避")
                await self._wait_for_stop(error_delay)
                error_delay = min(self._error_max, error_delay * 2)
                continue
            error_delay = self._error_initial
            if lease is None:
                await self._wait_for_stop(idle_delay)
                idle_delay = min(self._idle_max, idle_delay * 2)
                continue
            idle_delay = self._idle_initial
            # 优雅 stop 不丢弃已提交 lease；外部强制取消不提供此保证。
            await dispatch(lease)
            # await 本身不一定挂起；满队列和立即返回的回调也要允许 stop 调度。
            await asyncio.sleep(0)

    async def _claim_once(self) -> TaskLease | None:
        """领取提交后且 session 已退出才返回 lease；提交结果未知不做补偿。"""
        async with self._session_factory() as session:
            lease = await leases.claim_task(
                session,
                owner=self._owner,
                kinds=self._kinds,
                lease_seconds=self._lease_seconds,
            )
            await session.commit()
            return lease

    async def _wait_for_stop(self, delay: float) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
        except TimeoutError:
            pass
