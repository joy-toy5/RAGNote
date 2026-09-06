"""polling 启动、取消与错误退避边界；仅用确定性替身，不访问真实存储。"""

from __future__ import annotations

import asyncio
import inspect
import logging
import sqlite3
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import DBAPIError, DisconnectionError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeout

from app.tasking import polling
from app.tasking.lease_contracts import TaskLease
from app.tasking.polling import PollingRuntime, PollingState

OWNER = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
LEASE = TaskLease("11111111-1111-4111-8111-111111111111", OWNER, 1, 1)


@pytest.fixture(autouse=True)
def enable_polling_logger(monkeypatch):
    # 先前 Alembic 回归可能禁用了 collection 时已经创建的 logger。
    monkeypatch.setattr(polling.logger, "disabled", False)
    monkeypatch.setattr(polling.logger, "propagate", True)


def _forbidden_factory():
    raise AssertionError("边界用例不得打开数据库会话")


def _runtime(dispatch, **overrides):
    return PollingRuntime(
        _forbidden_factory,
        owner=OWNER,
        kinds=("note.index",),
        dispatch=dispatch,
        **overrides,
    )


def _run(coroutine):
    asyncio.run(asyncio.wait_for(coroutine, timeout=2))


@asynccontextmanager
async def _running(runtime):
    runtime.start()
    try:
        yield runtime
    finally:
        runtime.stop_accepting()
        if not runtime.task.done():
            runtime.task.cancel()
        await asyncio.gather(runtime.task, return_exceptions=True)


@pytest.mark.parametrize("dispatch", [False, 0, 1, "", "dispatch", object()])
def test_invalid_dispatch_is_rejected_before_io(dispatch):
    with pytest.raises(TypeError, match="dispatch"):
        _runtime(dispatch)


def test_start_without_running_loop_preserves_new_state():
    runtime = _runtime(AsyncMock())
    with pytest.raises(RuntimeError, match="no running event loop"):
        runtime.start()
    assert runtime.state == PollingState.NEW
    assert runtime.task is None


def test_task_factory_failure_closes_coroutine_and_allows_start_retry():
    async def scenario():
        loop = asyncio.get_running_loop()
        original_factory = loop.get_task_factory()
        rejected = []

        def reject(_loop, coroutine, **_kwargs):
            rejected.append(coroutine)
            raise RuntimeError("测试工厂拒绝创建")

        runtime = _runtime(AsyncMock())
        loop.set_task_factory(reject)
        try:
            with pytest.raises(RuntimeError, match="测试工厂拒绝创建"):
                runtime.start()
        finally:
            loop.set_task_factory(original_factory)
        assert len(rejected) == 1
        assert inspect.getcoroutinestate(rejected[0]) == inspect.CORO_CLOSED
        assert runtime.state == PollingState.NEW
        assert runtime.task is None
        async with _running(runtime):
            runtime.stop_accepting()
            result = await runtime.shutdown(timeout=1)
            assert result.state == PollingState.STOPPED
            assert not result.pending

    _run(scenario())


@pytest.mark.skipif(
    not hasattr(asyncio, "eager_task_factory"), reason="需要 Python 3.12 eager factory"
)
def test_eager_task_factory_registers_handle_before_polling(monkeypatch):
    async def scenario():
        runtime = _runtime(AsyncMock())
        claim = AsyncMock(return_value=LEASE)
        monkeypatch.setattr(runtime, "_claim_once", claim)
        runtime._dispatch_callback.side_effect = lambda _lease: runtime.stop_accepting()
        loop = asyncio.get_running_loop()
        original_factory = loop.get_task_factory()
        loop.set_task_factory(asyncio.eager_task_factory)
        try:
            async with _running(runtime):
                assert runtime.state == PollingState.RUNNING
                assert runtime.task is not None
                with pytest.raises(RuntimeError):
                    runtime.start()
                await runtime.task
                assert runtime.state == PollingState.STOPPED
                claim.assert_awaited_once()
                runtime._dispatch_callback.assert_awaited_once_with(LEASE)
                with pytest.raises(RuntimeError):
                    runtime.start()
        finally:
            loop.set_task_factory(original_factory)

    _run(scenario())


@pytest.mark.parametrize("before_first_step", [False, True])
def test_external_cancellation_converges_even_before_first_step(
    monkeypatch, before_first_step
):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def dispatch(_lease):
            entered.set()
            await release.wait()

        runtime = _runtime(dispatch)
        claim = AsyncMock(return_value=LEASE)
        monkeypatch.setattr(runtime, "_claim_once", claim)
        async with _running(runtime):
            if not before_first_step:
                await entered.wait()
            runtime.task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await runtime.task
            assert runtime.task.cancelled()
            assert runtime.state == PollingState.STOPPED
            assert not runtime.accepting
            result = await runtime.shutdown(timeout=0)
            assert result.state == PollingState.STOPPED
            assert not result.pending
            assert claim.await_count == (0 if before_first_step else 1)

    _run(scenario())


def test_shutdown_zero_and_cancelled_waiter_do_not_cancel_dispatch(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=polling.logger.name)

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        waiting = asyncio.Event()

        async def dispatch(_lease):
            entered.set()
            await release.wait()

        runtime = _runtime(dispatch)
        monkeypatch.setattr(runtime, "_claim_once", AsyncMock(return_value=LEASE))

        async def shutdown_waiter():
            waiting.set()
            return await runtime.shutdown(timeout=1)

        async with _running(runtime):
            await entered.wait()
            pending = await runtime.shutdown(timeout=0)
            assert pending.state == PollingState.DRAINING
            assert pending.pending
            waiter = asyncio.create_task(shutdown_waiter())
            try:
                await waiting.wait()
                waiter.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiter
                assert not runtime.task.done()
                assert runtime.task.cancelling() == 0
                assert runtime.state == PollingState.DRAINING
                release.set()
                finished = await runtime.shutdown(timeout=1)
                assert finished.state == PollingState.STOPPED
                assert not finished.pending
            finally:
                release.set()
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)

    _run(scenario())
    assert caplog.messages == ["持久任务 polling 关闭等待超时，仍有 1 个受管循环未结束"]
    assert all(record.exc_info is None for record in caplog.records)


def test_stop_wakes_long_idle_backoff(monkeypatch):
    async def scenario():
        waiting = asyncio.Event()
        runtime = _runtime(AsyncMock(), idle_initial=3600, idle_max=3600)
        monkeypatch.setattr(runtime, "_claim_once", AsyncMock(return_value=None))
        original_wait = runtime._wait_for_stop

        async def wait(delay):
            assert delay == 3600
            waiting.set()
            await original_wait(delay)

        monkeypatch.setattr(runtime, "_wait_for_stop", wait)
        async with _running(runtime):
            await waiting.wait()
            result = await runtime.shutdown(timeout=1)
            assert result.state == PollingState.STOPPED
            assert not result.pending

    _run(scenario())


def test_backoff_caps_resets_and_never_discovers_expired_leases(monkeypatch):
    async def scenario():
        transient = OperationalError("脱敏占位", {}, RuntimeError(1213))
        outcomes = [None] * 4 + [LEASE] + [transient] * 4 + [None, transient, LEASE, None]
        delays = []
        dispatch = AsyncMock()
        runtime = _runtime(
            dispatch, idle_initial=1, idle_max=3, error_initial=2, error_max=5
        )

        async def claim():
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        async def wait(delay):
            delays.append(delay)
            if not outcomes:
                runtime.stop_accepting()

        discover = AsyncMock(side_effect=AssertionError("polling 不得自动发现/恢复过期执行"))
        monkeypatch.setattr(polling.leases, "list_expired_leases", discover)
        monkeypatch.setattr(runtime, "_claim_once", claim)
        monkeypatch.setattr(runtime, "_wait_for_stop", wait)
        async with _running(runtime):
            await runtime.task
            assert runtime.state == PollingState.STOPPED
        assert delays == [1, 2, 3, 3, 2, 4, 5, 5, 1, 2, 1]
        assert dispatch.await_count == 2
        discover.assert_not_awaited()

    _run(scenario())


def _sqlite_error(code):
    original = sqlite3.OperationalError("脱敏占位")
    original.sqlite_errorcode = code
    return OperationalError("脱敏占位", {}, original)


@pytest.mark.parametrize(
    "error,expected",
    [
        pytest.param(DisconnectionError(), True, id="disconnect"),
        pytest.param(PoolTimeout(), True, id="pool-timeout"),
        pytest.param(
            DBAPIError("脱敏占位", {}, RuntimeError(), connection_invalidated=True),
            True,
            id="invalidated-connection",
        ),
        pytest.param(DBAPIError("脱敏占位", {}, RuntimeError()), False, id="unknown-dbapi"),
        pytest.param(TimeoutError(), False, id="ordinary-timeout"),
        pytest.param(ValueError(), False, id="programming-error"),
    ],
)
def test_storage_exception_categories_are_explicit(error, expected):
    assert polling._is_transient_storage_error(error) is expected


@pytest.mark.parametrize("code", [1040, 1205, 1213, 2002, 2003, 2006, 2013])
def test_mysql_transient_numeric_codes_are_retryable(code):
    error = OperationalError("脱敏占位", {}, RuntimeError(code))
    assert polling._is_transient_storage_error(error)


@pytest.mark.parametrize("code", [1045, 1049, 1054, 1146, 9999, "1213", True, None])
def test_mysql_permission_schema_unknown_and_untyped_codes_are_fatal(code):
    error = OperationalError("脱敏占位", {}, RuntimeError(code))
    assert not polling._is_transient_storage_error(error)


@pytest.mark.parametrize(
    "code,expected",
    [(5, True), (6, True), (261, True), (262, True), (1, False), (True, False), ("5", False)],
)
def test_sqlite_busy_locked_extended_codes_are_retryable(code, expected):
    assert polling._is_transient_storage_error(_sqlite_error(code)) is expected


def test_fatal_storage_error_is_not_retried_or_logged_with_secrets(monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger=polling.logger.name)

    async def scenario():
        secret = "不得输出的输入与驱动详情"
        error = OperationalError(secret, {"private": secret}, RuntimeError(1045, secret))
        claim = AsyncMock(side_effect=error)
        dispatch = AsyncMock()
        runtime = _runtime(dispatch)
        monkeypatch.setattr(runtime, "_claim_once", claim)
        async with _running(runtime):
            await runtime.task
            assert runtime.state == PollingState.FAILED
            claim.assert_awaited_once()
            dispatch.assert_not_awaited()
            result = await runtime.shutdown(timeout=0)
            assert result.state == PollingState.FAILED
            assert not result.pending
        assert secret not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)

    _run(scenario())
    assert caplog.messages == ["持久任务 polling 或 dispatch 失败，已停止领取"]
