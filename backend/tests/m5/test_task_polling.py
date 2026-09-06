"""P2-B polling 生命周期离线契约；不接入生产 handler 或外部存储。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import OperationalError

from app.tasking.lease_contracts import TaskLease
from app.tasking.polling import (
    PollingConfigurationError,
    PollingRuntime,
    PollingState,
)

OWNER = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
LEASE = TaskLease(
    "11111111-1111-4111-8111-111111111111",
    OWNER,
    1,
    1,
)


class _Session:
    def __init__(self, events):
        self.events = events

    async def __aenter__(self):
        self.events.append("session-open")
        return self

    async def __aexit__(self, *_args):
        self.events.append("session-close")

    async def commit(self):
        self.events.append("commit")


class _SessionFactory:
    def __init__(self, events):
        self.events = events

    def __call__(self):
        return _Session(self.events)


@asynccontextmanager
async def _noop_session():
    yield AsyncMock()


def _runtime(session_factory, dispatch, **overrides):
    values = {
        "owner": OWNER,
        "kinds": ("note.index",),
        "dispatch": dispatch,
        "lease_seconds": 2,
        "idle_initial": 0.01,
        "idle_max": 0.02,
        "error_initial": 0.01,
        "error_max": 0.02,
    }
    values.update(overrides)
    return PollingRuntime(session_factory, **values)


def test_without_dispatch_runtime_is_unconfigured_and_never_opens_session():
    calls = []

    def forbidden_factory():
        calls.append("opened")
        raise AssertionError("未配置 dispatch 不得访问数据库")

    async def scenario():
        runtime = _runtime(forbidden_factory, None)
        assert runtime.state == PollingState.UNCONFIGURED
        assert not runtime.accepting
        with pytest.raises(PollingConfigurationError):
            runtime.start()
        result = await runtime.shutdown(timeout=0.01)
        assert result.state == PollingState.UNCONFIGURED
        assert not result.pending
        assert calls == []

    asyncio.run(scenario())


def test_claim_commit_dispatch_order_and_single_start():
    async def scenario():
        events = []
        factory = _SessionFactory(events)
        dispatch = AsyncMock(side_effect=lambda _lease: events.append("dispatch"))

        async def claim(_session, **_kwargs):
            events.append("claim")
            return LEASE if events.count("claim") == 1 else None

        runtime = _runtime(factory, dispatch)
        with pytest.MonkeyPatch.context() as patch:
            from app.tasking import polling

            patch.setattr(polling.leases, "claim_task", claim)
            runtime.start()
            with pytest.raises(RuntimeError):
                runtime.start()
            await asyncio.sleep(0.01)
            result = await runtime.shutdown(timeout=0.2)

        assert result.state == PollingState.STOPPED
        assert not result.pending
        assert events[:4] == ["session-open", "claim", "commit", "session-close"]
        assert events[4] == "dispatch"
        dispatch.assert_awaited_once_with(LEASE)

    asyncio.run(scenario())


def test_claim_commit_failure_never_reaches_dispatch():
    async def scenario():
        events = []

        class FailingSession(_Session):
            async def commit(self):
                events.append("commit")
                raise ValueError("commit result is not a dispatch signal")

        class FailingFactory(_SessionFactory):
            def __call__(self):
                return FailingSession(self.events)

        async def claim(_session, **_kwargs):
            events.append("claim")
            return LEASE

        dispatch = AsyncMock()
        with pytest.MonkeyPatch.context() as patch:
            from app.tasking import polling

            patch.setattr(polling.leases, "claim_task", claim)
            runtime = _runtime(FailingFactory(events), dispatch)
            runtime.start()
            await asyncio.wait_for(runtime.task, timeout=0.2)

        assert runtime.state == PollingState.FAILED
        assert events == ["session-open", "claim", "commit", "session-close"]
        dispatch.assert_not_awaited()

    asyncio.run(scenario())


def test_dispatch_failure_stops_polling_after_committed_claim():
    async def scenario():
        calls = 0

        async def claim(_session, **_kwargs):
            nonlocal calls
            calls += 1
            return LEASE

        async def dispatch(_lease):
            raise ValueError("dispatch detail must not escape")

        with pytest.MonkeyPatch.context() as patch:
            from app.tasking import polling

            patch.setattr(polling.leases, "claim_task", claim)
            runtime = _runtime(_noop_session, dispatch)
            runtime.start()
            await asyncio.wait_for(runtime.task, timeout=0.2)

        assert runtime.state == PollingState.FAILED
        assert calls == 1

    asyncio.run(scenario())


def test_stop_during_claim_still_dispatches_the_committed_lease():
    async def scenario():
        claim_started = asyncio.Event()
        release_claim = asyncio.Event()
        dispatched = asyncio.Event()

        async def claim(_session, **_kwargs):
            claim_started.set()
            await release_claim.wait()
            return LEASE

        async def dispatch(_lease):
            dispatched.set()

        with pytest.MonkeyPatch.context() as patch:
            from app.tasking import polling

            patch.setattr(polling.leases, "claim_task", claim)
            runtime = _runtime(_noop_session, dispatch)
            runtime.start()
            await asyncio.wait_for(claim_started.wait(), timeout=0.2)
            runtime.stop_accepting()
            release_claim.set()
            await asyncio.wait_for(dispatched.wait(), timeout=0.2)
            result = await runtime.shutdown(timeout=0.2)

        assert result.state == PollingState.STOPPED
        assert not result.pending

    asyncio.run(scenario())


def test_transient_storage_error_uses_error_backoff_and_keeps_running():
    async def scenario():
        calls = 0
        second_call = asyncio.Event()

        async def claim(_session, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OperationalError("临时 SQL", {}, RuntimeError(1213, "脱敏占位"))
            second_call.set()
            return None

        async def dispatch(_lease):
            raise AssertionError("无任务不应 dispatch")

        with pytest.MonkeyPatch.context() as patch:
            from app.tasking import polling

            patch.setattr(polling.leases, "claim_task", claim)
            runtime = _runtime(_noop_session, dispatch)
            runtime.start()
            await asyncio.wait_for(second_call.wait(), timeout=0.2)
            assert runtime.state == PollingState.RUNNING
            result = await runtime.shutdown(timeout=0.2)

        assert result.state == PollingState.STOPPED
        assert calls >= 2

    asyncio.run(scenario())


def test_unknown_polling_error_fails_without_retry_or_dispatch():
    async def scenario():
        calls = 0
        dispatch = AsyncMock()

        async def claim(_session, **_kwargs):
            nonlocal calls
            calls += 1
            raise ValueError("programming detail must not be logged")

        with pytest.MonkeyPatch.context() as patch:
            from app.tasking import polling

            patch.setattr(polling.leases, "claim_task", claim)
            runtime = _runtime(_noop_session, dispatch)
            runtime.start()
            await asyncio.wait_for(runtime.task, timeout=0.2)
            result = await runtime.shutdown(timeout=0.01)

        assert calls == 1
        assert dispatch.await_count == 0
        assert result.state == PollingState.FAILED
        assert not result.pending

    asyncio.run(scenario())


def test_shutdown_timeout_keeps_draining_until_dispatch_finishes():
    async def scenario():
        claimed = asyncio.Event()
        release = asyncio.Event()

        async def claim(_session, **_kwargs):
            return LEASE

        async def dispatch(_lease):
            claimed.set()
            await release.wait()

        with pytest.MonkeyPatch.context() as patch:
            from app.tasking import polling

            patch.setattr(polling.leases, "claim_task", claim)
            runtime = _runtime(_noop_session, dispatch)
            runtime.start()
            await asyncio.wait_for(claimed.wait(), timeout=0.2)
            pending = await runtime.shutdown(timeout=0.01)
            assert pending.state == PollingState.DRAINING
            assert pending.pending
            assert not runtime.task.done()
            release.set()
            finished = await runtime.shutdown(timeout=0.2)

        assert finished.state == PollingState.STOPPED
        assert not finished.pending

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changes",
    [
        {"idle_initial": 0},
        {"idle_initial": float("nan")},
        {"idle_max": 0.1, "idle_initial": 0.2},
        {"error_initial": True},
        {"idle_max": 3601},
    ],
)
def test_polling_backoff_bounds_are_strict(changes):
    with pytest.raises((TypeError, ValueError)):
        _runtime(_noop_session, AsyncMock(), **changes)


@pytest.mark.parametrize("started", [False, True], ids=["new", "running"])
@pytest.mark.parametrize(
    "timeout,expected_error",
    [
        pytest.param(10**1000, ValueError, id="huge-positive"),
        pytest.param(-(10**1000), ValueError, id="huge-negative"),
        pytest.param(float("nan"), ValueError, id="nan"),
        pytest.param(float("inf"), ValueError, id="positive-infinity"),
        pytest.param(float("-inf"), ValueError, id="negative-infinity"),
        pytest.param(-1, ValueError, id="negative"),
        pytest.param(True, TypeError, id="true"),
        pytest.param(False, TypeError, id="false"),
        pytest.param("1", TypeError, id="string"),
        pytest.param(None, TypeError, id="none"),
    ],
)
def test_shutdown_rejects_invalid_timeout_before_state_change(
    started, timeout, expected_error
):
    async def scenario():
        def forbidden_factory():
            raise AssertionError("参数拒绝前和停止后均不得访问数据库")

        runtime = _runtime(forbidden_factory, AsyncMock())
        if started:
            runtime.start()
        initial_state = runtime.state
        try:
            with pytest.raises(expected_error):
                await runtime.shutdown(timeout=timeout)
            assert runtime.state == initial_state
        finally:
            result = await runtime.shutdown(timeout=1)
            assert result.state == PollingState.STOPPED
            assert not result.pending

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field", ["idle_initial", "idle_max", "error_initial", "error_max"]
)
def test_polling_rejects_unrepresentable_integer_seconds(field):
    with pytest.raises(ValueError):
        _runtime(_noop_session, AsyncMock(), **{field: 10**1000})
