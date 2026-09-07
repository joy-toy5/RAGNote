"""同一事件循环内的后台任务登记；不管理线程、持久化或自动重试。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class TaskRegistry:
    """默认接单，持有任务强引用，直到任务真实结束并回收异常。"""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()
        self._accepting = True

    @property
    def accepting(self) -> bool:
        return self._accepting

    def stop_accepting(self) -> None:
        """停止新任务，不取消当前执行。"""
        self._accepting = False

    async def drain(self, *, timeout: float = 5.0) -> frozenset[asyncio.Task[Any]]:
        """先给当前任务自然完成的时间；超时仍由登记器持有。"""
        self.stop_accepting()
        tasks = self.tasks
        if not tasks:
            return frozenset()
        _, pending = await asyncio.wait(tasks, timeout=timeout)
        return frozenset(pending)

    @property
    def tasks(self) -> frozenset[asyncio.Task[Any]]:
        """返回尚未结束任务的不可变快照，不暴露内部登记集合。"""
        return frozenset(task for task in self._tasks if not task.done())

    def start(self) -> None:
        """仅在旧任务全部结束后重新接单；不需要运行中的事件循环。"""
        if self.tasks:
            raise RuntimeError("仍有未结束的后台任务，不能重新接单")
        self._accepting = True

    def create(
        self, coroutine: Coroutine[Any, Any, _T], *, name: str
    ) -> asyncio.Task[_T]:
        """接管未调度的 coroutine；拒绝或调度失败时关闭它，避免泄漏。"""
        if not self._accepting:
            coroutine.close()
            raise RuntimeError("后台任务登记已停止接单")
        try:
            task = asyncio.create_task(coroutine, name=name)
        except BaseException:
            coroutine.close()
            raise
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
        return task

    async def cancel_and_wait(
        self, *, timeout: float = 5.0
    ) -> frozenset[asyncio.Task[Any]]:
        """先停止接单，再在共享时限内等待取消；超时任务继续保留登记。"""
        self.stop_accepting()
        tasks = self.tasks
        if not tasks:
            return frozenset()
        for task in tasks:
            task.cancel()
        _, pending = await asyncio.wait(tasks, timeout=timeout)
        if pending:
            logger.warning("后台任务取消等待超时，仍有 %d 个任务未结束", len(pending))
        return frozenset(pending)

    def _on_done(self, task: asyncio.Task[Any]) -> None:
        """只有真实结束才移除；取走异常，但不记录可能含用户内容的上下文。"""
        self._tasks.discard(task)
        if task.cancelled():
            return
        if task.exception() is not None:
            logger.error("后台任务执行失败")


async def cancel_task(task: asyncio.Task[Any], *, timeout: float = 1.0) -> bool:
    """有界取消单个任务；调用者取消直接传播，不改变任务登记或接单状态。"""
    if task.done():
        return True
    task.cancel()
    _, pending = await asyncio.wait({task}, timeout=timeout)
    return not pending


background_tasks = TaskRegistry()
