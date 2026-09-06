"""单写通道只接触一次性目录和真实受控线程，不加载生产存储。"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future

import pytest

from app.core.storage_ownership import StorageBusyError, StorageOwnership
from app.core.storage_writer import (
    StorageWriter,
    StorageWriterError,
    StorageWriterFullError,
    StorageWriterState,
)


def _run(coroutine):
    return asyncio.run(asyncio.wait_for(coroutine, timeout=10))


async def _until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), timeout=2)


def _block(entered, release):
    entered.set()
    assert release.wait(5), "测试必须在finally中释放写线程"


def _writer(path, **overrides):
    options = dict(initialize=lambda: None, invalidate=lambda: None, close=lambda: None)
    options.update(overrides)
    return StorageWriter(path, **options)


def _assert_busy(path):
    with pytest.raises(StorageBusyError):
        StorageOwnership(path).acquire()


def _assert_available(path):
    with StorageOwnership(path):
        pass


def test_single_thread_orders_initialization_write_invalidation_and_close(tmp_path):
    calls = []

    def record(name):
        _assert_busy(tmp_path)
        calls.append((name, threading.get_ident()))
        return name

    async def scenario():
        writer = _writer(
            tmp_path,
            initialize=lambda: record("init"),
            invalidate=lambda: record("invalidate"),
            close=lambda: record("close"),
        )
        assert writer.state == StorageWriterState.NEW
        assert (
            not writer.accepting and not writer.ownership_held and writer.pending == 0
        )
        try:
            with pytest.raises(StorageWriterError):
                await writer.run(lambda: record("forbidden"))
            await writer.start()
            assert writer.state == StorageWriterState.READY
            assert writer.accepting and writer.ownership_held
            assert await writer.run(lambda: record("write")) == "write"
        finally:
            result = await writer.shutdown(timeout=2)
        assert result.state == StorageWriterState.CLOSED
        assert not result.pending and not result.ownership_held and not result.failed
        assert writer.pending == 0
        with pytest.raises(StorageWriterError):
            await writer.start()
        with pytest.raises(StorageWriterError):
            await writer.run(lambda: record("forbidden"))
        assert await writer.shutdown(timeout=0) == result

    _run(scenario())
    assert [name for name, _ in calls] == ["init", "write", "invalidate", "close"]
    assert len({ident for _, ident in calls}) == 1
    assert calls[0][1] != threading.get_ident()
    _assert_available(tmp_path)


@pytest.mark.parametrize("max_pending", [0, -1, True, 1.5, "2", None])
def test_capacity_is_validated_without_storage_io(tmp_path, max_pending):
    with pytest.raises((TypeError, ValueError)):
        _writer(tmp_path / "absent", max_pending=max_pending)
    assert not (tmp_path / "absent").exists()


def test_cancelled_waiter_and_queued_waiter_keep_real_work_and_capacity(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def first():
        calls.append("first")
        _block(entered, release)

    async def scenario():
        writer = _writer(
            tmp_path, max_pending=2, invalidate=lambda: calls.append("invalidate")
        )
        tasks = []
        try:
            await writer.start()
            tasks.append(asyncio.create_task(writer.run(first)))
            await _until(entered.is_set)
            tasks.append(
                asyncio.create_task(writer.run(lambda: calls.append("second")))
            )
            await _until(lambda: writer.pending == 2)
            for task in tasks:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert writer.pending == 2
            with pytest.raises(StorageWriterFullError):
                await writer.run(lambda: calls.append("overflow"))
            report = await writer.shutdown(timeout=0)
            assert report.pending and report.ownership_held
            assert report.state == StorageWriterState.DRAINING
            _assert_busy(tmp_path)
            with pytest.raises(StorageWriterError):
                await writer.run(lambda: calls.append("late"))
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            report = await writer.shutdown(timeout=2)
        assert report.state == StorageWriterState.CLOSED and not report.pending

    _run(scenario())
    assert calls == ["first", "invalidate", "second", "invalidate"]
    _assert_available(tmp_path)


@pytest.mark.parametrize("blocked_phase", ["write", "invalidate", "close"])
def test_shutdown_does_not_release_storage_while_real_callback_is_running(
    tmp_path, blocked_phase
):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def callback(phase):
        calls.append(phase)
        if phase == blocked_phase:
            _block(entered, release)

    async def scenario():
        writer = _writer(
            tmp_path,
            invalidate=lambda: callback("invalidate"),
            close=lambda: callback("close"),
        )
        work = None
        try:
            await writer.start()
            work = asyncio.create_task(writer.run(lambda: callback("write")))
            if blocked_phase == "close":
                await work
                await writer.shutdown(timeout=0)
            await _until(entered.is_set)
            report = await writer.shutdown(timeout=0.005)
            assert report.pending and report.ownership_held
            assert writer.state == StorageWriterState.DRAINING
            _assert_busy(tmp_path)
            if blocked_phase != "close":
                assert "close" not in calls
        finally:
            release.set()
            if work:
                await asyncio.gather(work, return_exceptions=True)
            report = await writer.shutdown(timeout=2)
        assert report.state == StorageWriterState.CLOSED
        assert not report.pending and not report.ownership_held

    _run(scenario())
    assert calls == ["write", "invalidate", "close"]
    _assert_available(tmp_path)


def test_initialization_failure_closes_partial_resources_before_releasing(tmp_path):
    calls = []
    failure = ValueError("初始化失败")

    def initialize():
        calls.append("init")
        raise failure

    async def scenario():
        writer = _writer(
            tmp_path, initialize=initialize, close=lambda: calls.append("close")
        )
        try:
            with pytest.raises(ValueError) as caught:
                await writer.start()
            assert caught.value is failure
            assert writer.state == StorageWriterState.FAILED
            assert not writer.accepting and writer.ownership_held
            _assert_busy(tmp_path)
            with pytest.raises(StorageWriterError):
                await writer.run(lambda: calls.append("forbidden"))
        finally:
            report = await writer.shutdown(timeout=2)
        assert report.state == StorageWriterState.CLOSED and report.failed

    _run(scenario())
    assert calls == ["init", "close"]
    _assert_available(tmp_path)


def test_losing_owner_never_initializes_or_closes_the_winner(tmp_path):
    calls = []

    async def scenario():
        writer = _writer(
            tmp_path,
            initialize=lambda: calls.append("init"),
            close=lambda: calls.append("close"),
        )
        try:
            with pytest.raises(StorageBusyError):
                await writer.start()
            assert not writer.ownership_held
        finally:
            report = await writer.shutdown(timeout=2)
        assert report.state == StorageWriterState.CLOSED and report.failed

    with StorageOwnership(tmp_path):
        _run(scenario())
        _assert_busy(tmp_path)
    assert calls == []


def test_write_failure_still_invalidates_and_does_not_poison_safe_successor(tmp_path):
    calls = []
    failure = RuntimeError("测试写入失败")

    def write():
        calls.append("write")
        raise failure

    async def scenario():
        writer = _writer(tmp_path, invalidate=lambda: calls.append("invalidate"))
        try:
            await writer.start()
            with pytest.raises(RuntimeError) as caught:
                await writer.run(write)
            assert caught.value is failure
            assert writer.accepting
            assert await writer.run(lambda: 17) == 17
        finally:
            report = await writer.shutdown(timeout=2)
        assert not report.failed

    _run(scenario())
    assert calls == ["write", "invalidate", "invalidate"]


def test_invalidation_failure_rejects_already_queued_writes(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []
    failure = RuntimeError("失效失败")

    def invalidate():
        _block(entered, release)
        raise failure

    async def scenario():
        writer = _writer(
            tmp_path,
            max_pending=2,
            invalidate=invalidate,
            close=lambda: calls.append("close"),
        )
        tasks = []
        try:
            await writer.start()
            tasks.append(asyncio.create_task(writer.run(lambda: calls.append("first"))))
            await _until(entered.is_set)
            tasks.append(
                asyncio.create_task(writer.run(lambda: calls.append("forbidden")))
            )
            await _until(lambda: writer.pending == 2)
            release.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert results[0] is failure
            assert isinstance(results[1], StorageWriterError)
            assert writer.state == StorageWriterState.FAILED
            assert not writer.accepting
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            report = await writer.shutdown(timeout=2)
        assert report.state == StorageWriterState.CLOSED and report.failed

    _run(scenario())
    assert calls == ["first", "close"]


def test_close_failure_keeps_lock_and_is_not_retried(tmp_path):
    calls = []

    def close():
        calls.append("close")
        raise RuntimeError("关闭失败")

    async def scenario():
        writer = _writer(tmp_path, close=close)
        try:
            await writer.start()
            report = await writer.shutdown(timeout=2)
            assert report.state == StorageWriterState.FAILED
            assert report.failed and report.ownership_held and not report.pending
            _assert_busy(tmp_path)
            assert await writer.shutdown(timeout=0) == report
            assert calls == ["close"]
        finally:
            # 合成close没有遗留外部工作，已确认线程退出后仅为测试回收fd。
            writer._ownership.release()

    _run(scenario())


def test_unmanaged_future_is_not_mistaken_for_completed_storage_work(tmp_path):
    escaped = Future()
    calls = []

    async def scenario():
        writer = _writer(
            tmp_path,
            invalidate=lambda: calls.append("invalidate"),
            close=lambda: calls.append("close"),
        )
        try:
            await writer.start()
            with pytest.raises(TypeError):
                await writer.run(lambda: escaped)
            report = await writer.shutdown(timeout=2)
            assert report.state == StorageWriterState.FAILED
            assert report.pending and report.failed and report.ownership_held
            assert calls == []
            _assert_busy(tmp_path)
        finally:
            # 此future是无执行体的合成对象；明确完成后方可回收测试锁。
            escaped.set_result(None)
            writer._ownership.release()

    _run(scenario())


def test_replaced_directory_fails_before_writing(tmp_path):
    directory = tmp_path / "store"
    directory.mkdir()
    calls = []

    async def scenario():
        writer = _writer(directory, close=lambda: calls.append("close"))
        try:
            await writer.start()
            directory.rename(tmp_path / "old")
            directory.mkdir()
            with pytest.raises(RuntimeError):
                await writer.run(lambda: calls.append("forbidden"))
            assert writer.state == StorageWriterState.FAILED
        finally:
            report = await writer.shutdown(timeout=2)
        assert report.state == StorageWriterState.CLOSED and report.failed

    _run(scenario())
    assert calls == ["close"]


def test_shutdown_before_start_creates_no_resources(tmp_path):
    calls = []

    async def scenario():
        writer = _writer(
            tmp_path / "absent",
            initialize=lambda: calls.append("init"),
            close=lambda: calls.append("close"),
        )
        report = await writer.shutdown(timeout=0)
        assert report.state == StorageWriterState.CLOSED
        assert not report.pending and not report.ownership_held
        with pytest.raises(StorageWriterError):
            await writer.start()

    _run(scenario())
    assert calls == []


def test_started_coroutine_is_quarantined_without_running_its_cleanup(tmp_path):
    calls = []

    class Suspender:
        def __await__(self):
            yield None

    async def external():
        try:
            await Suspender()
        finally:
            calls.append("external-close")

    escaped = external()
    escaped.send(None)

    async def scenario():
        writer = _writer(
            tmp_path,
            invalidate=lambda: calls.append("invalidate"),
            close=lambda: calls.append("close"),
        )
        try:
            await writer.start()
            with pytest.raises(TypeError):
                await writer.run(lambda: escaped)
            report = await writer.shutdown(timeout=2)
            assert report.failed and report.pending and report.ownership_held
            assert calls == []
            _assert_busy(tmp_path)
        finally:
            # 外部生命期由测试拥有；通道不应替外部关闭已开始的协程。
            escaped.close()
            writer._ownership.release()

    _run(scenario())
    assert calls == ["external-close"]


@pytest.mark.parametrize("entry", ["run", "shutdown"])
def test_callback_cannot_recursively_wait_for_its_own_lane(tmp_path, entry):
    async def scenario():
        writer = _writer(tmp_path, max_pending=2)

        def write():
            if entry == "run":
                coroutine = writer.run(lambda: None)
            else:
                coroutine = writer.shutdown(timeout=0)
            with pytest.raises(StorageWriterError):
                asyncio.run(asyncio.wait_for(coroutine, timeout=0.05))

        try:
            await writer.start()
            await writer.run(write)
            assert writer.accepting
        finally:
            assert not (await writer.shutdown(timeout=2)).pending

    _run(scenario())


def test_closed_future_does_not_claim_the_thread_has_exited(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = StorageWriter._worker

    def delayed_exit(writer):
        original(writer)
        _block(entered, release)

    monkeypatch.setattr(StorageWriter, "_worker", delayed_exit)

    async def scenario():
        writer = _writer(tmp_path)
        try:
            await writer.start()
            await writer.shutdown(timeout=0)
            await _until(entered.is_set)
            report = await writer.shutdown(timeout=0.005)
            assert writer.pending == 0
            assert report.pending and report.state == StorageWriterState.DRAINING
            assert not report.ownership_held
        finally:
            release.set()
            report = await writer.shutdown(timeout=2)
        assert report.state == StorageWriterState.CLOSED and not report.pending

    _run(scenario())
