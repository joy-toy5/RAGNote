from __future__ import annotations

import asyncio
import builtins
import gc
import inspect
import io
import logging
import sys
import threading
import weakref
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from app.core import task_registry
from app.core.task_registry import TaskRegistry, background_tasks, cancel_task

pytestmark = pytest.mark.p0


async def _succeed() -> str:
    return "合成结果"


def _assert_rejected(registry: TaskRegistry) -> None:
    coroutine = _succeed()
    try:
        with pytest.raises(RuntimeError):
            registry.create(coroutine, name="late-task")
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
    finally:
        coroutine.close()


@asynccontextmanager
async def _resisting_task(
    registry: TaskRegistry,
) -> AsyncIterator[tuple[asyncio.Task[None], asyncio.Event, asyncio.Event]]:
    """抗取消替身只由释放事件退出；兜底定时器防止错误实现拖住测试。"""
    release = asyncio.Event()
    cancelled = asyncio.Event()

    async def work() -> None:
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()

    task = registry.create(work(), name="cancellation-resistant")
    watchdog = asyncio.get_running_loop().call_later(1.0, release.set)
    try:
        await asyncio.sleep(0)
        yield task, cancelled, release
    finally:
        release.set()
        watchdog.cancel()
        _, pending = await asyncio.wait({task}, timeout=1.0)
        assert not pending, "抗取消替身必须由测试明确释放"
        task.result()


async def _cancel(
    registry: TaskRegistry, task: asyncio.Task[Any], scope: str, *, timeout: float
) -> frozenset[asyncio.Task[Any]] | bool:
    if scope == "registry":
        return await registry.cancel_and_wait(timeout=timeout)
    return await cancel_task(task, timeout=timeout)


def test_import_creates_only_an_idle_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    source = Path(task_registry.__file__).read_text(encoding="utf-8")
    code = compile(source, task_registry.__file__, "exec")
    original_import = builtins.__import__

    def stdlib_import(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name.split(".", 1)[0] in sys.stdlib_module_names
        return original_import(name, *args, **kwargs)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("任务登记模块导入时不得启动线程、调度任务或执行 I/O")

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "__import__", stdlib_import)
        guard.setattr(builtins, "open", forbidden)
        guard.setattr(io, "open", forbidden)
        guard.setattr(threading.Thread, "start", forbidden)
        guard.setattr(logging, "FileHandler", forbidden)
        guard.setattr(asyncio, "create_task", forbidden)
        guard.setattr(asyncio, "get_running_loop", forbidden)
        guard.setattr(asyncio, "new_event_loop", forbidden)
        namespace = {"__name__": "m5_isolated_task_registry"}
        exec(code, namespace)

    singleton = namespace["background_tasks"]
    assert isinstance(singleton, namespace["TaskRegistry"])
    assert singleton.tasks == frozenset()
    assert isinstance(background_tasks, TaskRegistry)
    assert background_tasks is task_registry.background_tasks


def test_create_names_task_and_returns_an_immutable_pending_snapshot() -> None:
    async def scenario() -> None:
        registry = TaskRegistry()
        task = registry.create(_succeed(), name="successful-task")
        snapshot = registry.tasks
        assert isinstance(task, asyncio.Task)
        assert task.get_name() == "successful-task"
        assert isinstance(snapshot, frozenset)
        assert snapshot == frozenset({task})
        with pytest.raises(AttributeError):
            snapshot.clear()
        assert await task == "合成结果"
        assert registry.tasks == frozenset()
        assert snapshot == frozenset({task})

    asyncio.run(scenario())


@pytest.mark.parametrize("eager", [False, True])
def test_done_tasks_are_hidden_before_the_done_callback_runs(eager: bool) -> None:
    async def scenario() -> None:
        registry = TaskRegistry()
        loop = asyncio.get_running_loop()
        previous_factory = loop.get_task_factory()
        try:
            if eager:
                loop.set_task_factory(asyncio.eager_task_factory)
            task = registry.create(_succeed(), name="already-finished")
            if not eager:
                await asyncio.sleep(0)
            assert task.done()
            assert registry.tasks == frozenset()
            registry.start()
            await asyncio.sleep(0)
            assert task.result() == "合成结果"
        finally:
            loop.set_task_factory(previous_factory)

    asyncio.run(scenario())


def test_registry_holds_pending_tasks_and_releases_finished_tasks() -> None:
    async def scenario() -> None:
        registry = TaskRegistry()

        async def work() -> None:
            await asyncio.Future()

        task = registry.create(work(), name="no-external-owner")
        reference = weakref.ref(task)
        del task
        await asyncio.sleep(0)
        gc.collect()
        assert reference() is not None
        assert registry.tasks == frozenset({reference()})
        assert await registry.cancel_and_wait(timeout=0.5) == frozenset()
        await asyncio.sleep(0)
        gc.collect()
        assert reference() is None
        assert registry.tasks == frozenset()

    asyncio.run(scenario())


def test_failure_is_retrieved_and_logged_without_user_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        registry = TaskRegistry()
        loop = asyncio.get_running_loop()
        errors = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        finished = asyncio.Event()

        async def fail() -> None:
            raise RuntimeError("合成私密异常内容") from ValueError("合成私密异常原因")

        try:
            task = registry.create(fail(), name="合成私密任务名称")
            reference = weakref.ref(task)
            task.add_done_callback(lambda _task: finished.set())
            await finished.wait()
            assert registry.tasks == frozenset()
            del task
            gc.collect()
            assert reference() is None
            assert errors == []
        finally:
            loop.set_exception_handler(previous_handler)

    caplog.set_level(logging.ERROR, logger=task_registry.__name__)
    asyncio.run(scenario())
    records = [
        record for record in caplog.records if record.name == task_registry.__name__
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].exc_info is None
    assert records[0].stack_info is None
    assert "合成私密" not in caplog.text


def test_cancel_and_wait_cancels_all_tasks_without_warning_or_failure_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        registry = TaskRegistry()
        cleaned = []

        async def work() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.append(True)

        tasks = [registry.create(work(), name=f"worker-{index}") for index in range(2)]
        await asyncio.sleep(0)
        remaining = await registry.cancel_and_wait()
        assert isinstance(remaining, frozenset)
        assert remaining == frozenset()
        assert all(task.cancelled() for task in tasks)
        assert cleaned == [True, True]
        assert registry.tasks == frozenset()
        _assert_rejected(registry)

    caplog.set_level(logging.WARNING, logger=task_registry.__name__)
    asyncio.run(scenario())
    assert caplog.records == []


def test_shutdown_closes_admission_before_cancelling_tasks() -> None:
    async def scenario() -> None:
        registry = TaskRegistry()
        rejected = asyncio.Event()

        async def work() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                _assert_rejected(registry)
                rejected.set()

        task = registry.create(work(), name="late-submission-on-cancel")
        await asyncio.sleep(0)
        assert await registry.cancel_and_wait(timeout=0.5) == frozenset()
        assert task.cancelled()
        assert rejected.is_set()

    asyncio.run(scenario())


def test_empty_shutdown_is_idempotent_and_start_reopens_admission(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=task_registry.__name__)
    registry = TaskRegistry()
    assert asyncio.run(registry.cancel_and_wait()) == frozenset()
    assert asyncio.run(registry.cancel_and_wait(timeout=0.0)) == frozenset()
    _assert_rejected(registry)
    assert registry.start() is None

    async def scenario() -> None:
        assert await registry.create(_succeed(), name="restarted") == "合成结果"
        assert await registry.cancel_and_wait() == frozenset()

    asyncio.run(scenario())
    assert caplog.records == []


def test_create_without_a_running_loop_closes_the_coroutine() -> None:
    registry = TaskRegistry()
    coroutine = _succeed()
    try:
        with pytest.raises(RuntimeError):
            registry.create(coroutine, name="no-running-loop")
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
        assert registry.tasks == frozenset()
    finally:
        coroutine.close()


def test_create_closes_the_coroutine_if_scheduling_fails() -> None:
    async def scenario() -> None:
        registry = TaskRegistry()
        loop = asyncio.get_running_loop()
        previous_factory = loop.get_task_factory()
        coroutine = _succeed()

        def fail_factory(*args: Any, **kwargs: Any) -> asyncio.Task[Any]:
            raise RuntimeError("合成调度失败")

        try:
            loop.set_task_factory(fail_factory)
            with pytest.raises(RuntimeError, match="合成调度失败"):
                registry.create(coroutine, name="factory-failure")
            assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
            assert registry.tasks == frozenset()
        finally:
            loop.set_task_factory(previous_factory)
            coroutine.close()

    asyncio.run(scenario())


def test_start_refuses_pending_tasks() -> None:
    async def scenario() -> None:
        registry = TaskRegistry()
        async with _resisting_task(registry) as (task, _cancelled, _release):
            with pytest.raises(RuntimeError):
                registry.start()
            assert registry.tasks == frozenset({task})
        registry.start()
        assert await registry.create(_succeed(), name="after-old-task") == "合成结果"

    asyncio.run(scenario())


@pytest.mark.parametrize("scope", ["registry", "task"])
@pytest.mark.parametrize("timeout", [0.0, 0.01])
def test_timeout_keeps_a_strong_reference_until_the_task_really_finishes(
    scope: str, timeout: float
) -> None:
    async def scenario() -> None:
        registry = TaskRegistry()
        async with _resisting_task(registry) as (task, cancelled, release):
            outcome = await _cancel(registry, task, scope, timeout=timeout)
            assert outcome == (frozenset({task}) if scope == "registry" else False)
            assert cancelled.is_set()
            assert not release.is_set(), "超时必须先于兜底释放返回"
            assert not task.done()
            assert registry.tasks == frozenset({task})
            with pytest.raises(RuntimeError):
                registry.start()
            if scope == "registry":
                _assert_rejected(registry)
        assert task.done()
        assert registry.tasks == frozenset()
        registry.start()
        assert await registry.create(_succeed(), name="after-timeout") == "合成结果"

    asyncio.run(scenario())


@pytest.mark.parametrize("scope", ["registry", "task"])
def test_cancellation_of_the_caller_propagates_and_keeps_pending_tasks(
    scope: str,
) -> None:
    async def scenario() -> None:
        registry = TaskRegistry()
        async with _resisting_task(registry) as (task, cancelled, _release):
            caller = asyncio.create_task(_cancel(registry, task, scope, timeout=5.0))
            try:
                async with asyncio.timeout(0.5):
                    await cancelled.wait()
                caller.cancel()
                _, pending = await asyncio.wait({caller}, timeout=0.5)
                assert not pending, "调用者取消必须立即向上传播"
                with pytest.raises(asyncio.CancelledError):
                    caller.result()
                assert not task.done()
                assert registry.tasks == frozenset({task})
                if scope == "registry":
                    _assert_rejected(registry)
            finally:
                caller.cancel()
                await asyncio.wait({caller}, timeout=0.5)
        assert registry.tasks == frozenset()

    asyncio.run(scenario())


@pytest.mark.parametrize("pending_count", [1, 2])
@pytest.mark.parametrize("timeout", [0.0, 0.01])
def test_shutdown_timeout_warns_only_about_the_remaining_task_count(
    caplog: pytest.LogCaptureFixture, pending_count: int, timeout: float
) -> None:
    async def scenario() -> None:
        registry = TaskRegistry()
        async with AsyncExitStack() as stack:
            tasks = []
            releases = []
            for index in range(pending_count):
                task, _cancelled, release = await stack.enter_async_context(
                    _resisting_task(registry)
                )
                task.set_name(f"合成私密任务名称-{index}")
                tasks.append(task)
                releases.append(release)
            cooperative = registry.create(asyncio.Event().wait(), name="cooperative")
            remaining = await registry.cancel_and_wait(timeout=timeout)
            assert remaining == frozenset(tasks)
            assert registry.tasks == remaining
            assert cooperative.cancelled()
            assert not any(release.is_set() for release in releases)
        assert registry.tasks == frozenset()

    caplog.set_level(logging.WARNING, logger=task_registry.__name__)
    asyncio.run(scenario())
    records = [
        record for record in caplog.records if record.name == task_registry.__name__
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert (
        records[0].getMessage()
        == f"后台任务取消等待超时，仍有 {pending_count} 个任务未结束"
    )
    assert records[0].args == (pending_count,)
    assert records[0].exc_info is None
    assert records[0].stack_info is None
    assert "合成私密" not in caplog.text


def test_cancel_task_only_cancels_the_selected_task_and_keeps_admission_open() -> None:
    async def scenario() -> None:
        registry = TaskRegistry()
        first = registry.create(asyncio.Event().wait(), name="first")
        second = registry.create(asyncio.Event().wait(), name="second")
        assert await cancel_task(first) is True
        assert first.cancelled()
        assert not second.done()
        assert registry.tasks == frozenset({second})
        assert await registry.create(_succeed(), name="still-open") == "合成结果"
        assert await registry.cancel_and_wait(timeout=0.5) == frozenset()

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_cancel_task_accepts_already_finished_tasks(outcome: str) -> None:
    async def scenario() -> None:
        registry = TaskRegistry()

        async def work() -> None:
            if outcome == "failure":
                raise RuntimeError("合成失败")
            if outcome == "cancelled":
                raise asyncio.CancelledError

        task = registry.create(work(), name="already-finished")
        await asyncio.wait({task}, timeout=0.5)
        assert task.done()
        assert await cancel_task(task, timeout=0.0) is True
        assert registry.tasks == frozenset()

    asyncio.run(scenario())
