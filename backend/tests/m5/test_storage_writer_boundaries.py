"""P2-C2 独立边界：仅使用合成回调、一次性目录和有界受控解释器。"""

from __future__ import annotations

import asyncio
import gc
import inspect
import logging
import subprocess
import sys
import threading
import warnings
import weakref
from dataclasses import FrozenInstanceError

import pytest

from app.core import storage_writer as storage
from app.core.storage_ownership import StorageBusyError, StorageOwnership
from app.core.storage_writer import (
    StorageWriter,
    StorageWriterError,
    StorageWriterState,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="仅验证 Linux 本地存储所有权"
)


# ==================== 有界等待与合成资源回收 ====================


def _run(coroutine):
    return asyncio.run(asyncio.wait_for(coroutine, timeout=12), debug=True)


async def _until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), timeout=2)


def _block(entered, release):
    entered.set()
    assert release.wait(5), "测试必须在 finally 中释放回调"


def _writer(directory, **overrides):
    callbacks = dict(
        initialize=lambda: None, invalidate=lambda: None, close=lambda: None
    )
    callbacks.update(overrides)
    return StorageWriter(directory, **callbacks)


def _assert_busy(directory):
    contender = StorageOwnership(directory)
    try:
        with pytest.raises(StorageBusyError):
            contender.acquire()
    finally:
        contender.release()


def _assert_closed(report, *, failed=False):
    assert report.state == StorageWriterState.CLOSED
    assert report.pending is False
    assert report.ownership_held is False
    assert report.failed is failed


async def _finish(writer, *, releases=(), waiters=(), threads=()):
    for release in releases:
        release.set()
    try:
        return await asyncio.wait_for(writer.shutdown(timeout=2), timeout=3)
    finally:
        for waiter in waiters:
            waiter.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(*waiters, return_exceptions=True), 2)
        finally:
            for thread in {*threads, writer._thread} - {None}:
                if thread.ident is not None:
                    thread.join(timeout=2)
                    assert not thread.is_alive(), "测试写线程未在期限内退出"
            # 外部 Future 替身由测试显式完成；线程退出后才回收合成故障留下的 fd。
            writer._ownership.release()


# ==================== 生命周期等待者不拥有真实工作的取消权 ====================


def test_cancelled_initialization_waiter_does_not_cancel_startup(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def initialize():
        calls.append("初始化进入")
        _block(entered, release)
        calls.append("初始化完成")

    async def scenario():
        writer = _writer(
            tmp_path, initialize=initialize, close=lambda: calls.append("关闭")
        )
        waiter = asyncio.create_task(writer.start())
        try:
            await _until(entered.is_set)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert writer.state == StorageWriterState.STARTING
            assert writer.pending == 1 and writer.ownership_held
            assert not writer.accepting and calls == ["初始化进入"]
            _assert_busy(tmp_path)
            with pytest.raises(StorageWriterError):
                await writer.start()
            with pytest.raises(StorageWriterError):
                await writer.run(lambda: calls.append("非法写入"))
            release.set()
            await _until(lambda: writer.accepting and writer.pending == 0)
            assert calls == ["初始化进入", "初始化完成"]
        finally:
            report = await _finish(writer, releases=(release,), waiters=(waiter,))
        _assert_closed(report)

    _run(scenario())
    assert calls == ["初始化进入", "初始化完成", "关闭"]


def test_shutdown_during_initialization_never_reopens_admission(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def initialize():
        calls.append("初始化进入")
        _block(entered, release)
        calls.append("初始化完成")

    def close():
        _assert_busy(tmp_path)
        calls.append("关闭")

    async def scenario():
        writer = _writer(tmp_path, initialize=initialize, close=close)
        waiter = asyncio.create_task(writer.start())
        try:
            await _until(entered.is_set)
            snapshot = await writer.shutdown(timeout=0)
            assert snapshot.state == StorageWriterState.DRAINING
            assert snapshot.pending is True and snapshot.ownership_held is True
            assert snapshot.failed is False and not writer.accepting
            assert calls == ["初始化进入"]
            with pytest.raises(FrozenInstanceError):
                snapshot.pending = False
            with pytest.raises(StorageWriterError):
                await writer.run(lambda: calls.append("非法写入"))
            release.set()
            await asyncio.wait_for(waiter, timeout=2)
            assert not writer.accepting
        finally:
            report = await _finish(writer, releases=(release,), waiters=(waiter,))
        _assert_closed(report)
        assert (
            snapshot.state == StorageWriterState.DRAINING and snapshot.pending is True
        )

    _run(scenario())
    assert calls == ["初始化进入", "初始化完成", "关闭"]


@pytest.mark.parametrize("blocked_phase", ["write", "close"])
def test_cancelled_shutdown_waiter_preserves_the_accepted_drain(
    tmp_path, blocked_phase
):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def first():
        calls.append("首个写入")
        if blocked_phase == "write":
            _block(entered, release)

    def close():
        calls.append("关闭")
        if blocked_phase == "close":
            _block(entered, release)

    async def scenario():
        writer = _writer(
            tmp_path,
            max_pending=2,
            invalidate=lambda: calls.append("失效"),
            close=close,
        )
        waiters = []
        try:
            await writer.start()
            waiters.append(asyncio.create_task(writer.run(first)))
            if blocked_phase == "write":
                await _until(entered.is_set)
            waiters.append(
                asyncio.create_task(writer.run(lambda: calls.append("排队写入")))
            )
            if blocked_phase == "write":
                await _until(lambda: writer.pending == 2)
            else:
                await asyncio.gather(*waiters)
            shutdown = asyncio.create_task(writer.shutdown(timeout=5))
            waiters.append(shutdown)
            await _until(
                lambda: entered.is_set() and writer.state == StorageWriterState.DRAINING
            )
            shutdown.cancel()
            with pytest.raises(asyncio.CancelledError):
                await shutdown
            assert not writer.accepting and writer.ownership_held
            _assert_busy(tmp_path)
            with pytest.raises(StorageWriterError):
                await writer.run(lambda: calls.append("非法写入"))
            report = await writer.shutdown(timeout=0)
            assert report.pending is True and report.ownership_held is True
        finally:
            report = await _finish(writer, releases=(release,), waiters=waiters)
        _assert_closed(report)

    _run(scenario())
    assert calls == ["首个写入", "失效", "排队写入", "失效", "关闭"]


# ==================== 失败必须发生在不可逆状态变化之前 ====================


@pytest.mark.parametrize(
    ("invalid_values", "error_type"),
    [
        pytest.param((-1,), ValueError, id="negative"),
        pytest.param(
            (float("nan"), float("inf"), float("-inf")), ValueError, id="non-finite"
        ),
        pytest.param((True, False), TypeError, id="booleans"),
        pytest.param((None, "1"), TypeError, id="non-numeric"),
        pytest.param((1 << 20000,), ValueError, id="oversized-integer"),
    ],
)
def test_invalid_shutdown_timeout_leaves_new_and_ready_states_unchanged(
    tmp_path, invalid_values, error_type
):
    async def scenario():
        writer = _writer(tmp_path)
        try:
            for started in (False, True):
                if started:
                    await writer.start()
                snapshot = (
                    writer.state,
                    writer.pending,
                    writer.ownership_held,
                    writer.accepting,
                )
                for value in invalid_values:
                    with pytest.raises(error_type):
                        await writer.shutdown(timeout=value)
                    assert (
                        writer.state,
                        writer.pending,
                        writer.ownership_held,
                        writer.accepting,
                    ) == snapshot
            assert await writer.run(lambda: "仍可接单") == "仍可接单"
        finally:
            report = await _finish(writer)
        _assert_closed(report)

    _run(scenario())


def test_thread_start_failure_before_native_start_never_initializes(
    tmp_path, monkeypatch
):
    failure = RuntimeError("原生线程启动前失败")
    threads, calls = [], []
    directory = tmp_path / "not-created"

    def refuse_start(thread):
        threads.append(thread)
        raise failure

    async def scenario():
        writer = _writer(
            directory,
            initialize=lambda: calls.append("初始化"),
            close=lambda: calls.append("关闭"),
        )
        try:
            with monkeypatch.context() as patch:
                patch.setattr(threading.Thread, "start", refuse_start)
                with pytest.raises(RuntimeError) as caught:
                    await writer.start()
            assert caught.value is failure
            assert writer.state == StorageWriterState.FAILED
            assert writer.pending == 0 and not writer.ownership_held
            assert not writer.accepting and not directory.exists()
            assert len(threads) == 1 and threads[0].ident is None
            with pytest.raises(StorageWriterError):
                await writer.start()
        finally:
            report = await _finish(writer, threads=threads)
        _assert_closed(report, failed=True)

    _run(scenario())
    assert calls == [] and not directory.exists()


def test_thread_start_failure_after_native_start_keeps_the_live_owner(
    tmp_path, monkeypatch
):
    failure = RuntimeError("原生线程启动后失败")
    entered, release = threading.Event(), threading.Event()
    threads, calls = [], []
    original_start = threading.Thread.start

    def start_then_fail(thread):
        threads.append(thread)
        original_start(thread)
        raise failure

    def initialize():
        calls.append("初始化进入")
        _block(entered, release)
        calls.append("初始化完成")

    async def scenario():
        writer = _writer(
            tmp_path, initialize=initialize, close=lambda: calls.append("关闭")
        )
        try:
            with monkeypatch.context() as patch:
                patch.setattr(threading.Thread, "start", start_then_fail)
                with pytest.raises(RuntimeError) as caught:
                    await writer.start()
            assert caught.value is failure
            await _until(entered.is_set)
            assert len(threads) == 1 and threads[0].is_alive()
            assert writer.state == StorageWriterState.FAILED
            assert writer.pending == 1 and writer.ownership_held
            assert not writer.accepting and calls == ["初始化进入"]
            _assert_busy(tmp_path)
            with pytest.raises(StorageWriterError):
                await writer.start()
            report = await writer.shutdown(timeout=0)
            assert report.pending is True and report.ownership_held is True
            assert report.failed is True and "关闭" not in calls
        finally:
            report = await _finish(writer, releases=(release,), threads=threads)
        _assert_closed(report, failed=True)

    _run(scenario())
    assert calls == ["初始化进入", "初始化完成", "关闭"]
    with StorageOwnership(tmp_path):
        pass


# ==================== 同步协议既检查声明，也检查动态返回值 ====================


def _async_callback(kind, calls):
    async def coroutine():
        calls.append("协程函数执行")

    async def generator():
        calls.append("异步生成器执行")
        yield None

    class CoroutineCallable:
        async def __call__(self):
            calls.append("异步 callable 执行")

    class GeneratorCallable:
        async def __call__(self):
            calls.append("异步生成器 callable 执行")
            yield None

    return {
        "coroutine-function": coroutine,
        "coroutine-callable": CoroutineCallable(),
        "asyncgen-function": generator,
        "asyncgen-callable": GeneratorCallable(),
    }[kind]


@pytest.mark.parametrize(
    "kind",
    [
        "coroutine-function",
        "coroutine-callable",
        "asyncgen-function",
        "asyncgen-callable",
    ],
)
def test_declared_async_callbacks_are_rejected_before_admission(tmp_path, kind):
    bodies, calls = [], []
    callback = _async_callback(kind, bodies)
    absent = tmp_path / "not-created"
    for slot in ("initialize", "invalidate", "close"):
        with pytest.raises(TypeError):
            _writer(absent, **{slot: callback})
    assert not absent.exists()

    async def scenario():
        writer = _writer(tmp_path, invalidate=lambda: calls.append("失效"))
        try:
            await writer.start()
            with pytest.raises(TypeError):
                await writer.run(callback)
            assert writer.state == StorageWriterState.READY
            assert writer.pending == 0 and calls == []
            assert await writer.run(lambda: "同步写入") == "同步写入"
        finally:
            report = await _finish(writer)
        _assert_closed(report)

    _run(scenario())
    assert bodies == [] and calls == ["失效"]


@pytest.mark.parametrize("phase", ["initialize", "write", "invalidate", "close"])
def test_dynamic_unstarted_coroutine_is_closed_without_running_its_body(
    tmp_path, phase
):
    created, bodies, calls = [], [], []

    async def unstarted():
        bodies.append("协程体不能启动")
        await asyncio.sleep(0)

    def callback(name):
        calls.append(name)
        if name == phase:
            coroutine = unstarted()
            created.append(coroutine)
            return coroutine
        return None

    async def scenario():
        writer = _writer(
            tmp_path,
            initialize=lambda: callback("initialize"),
            invalidate=lambda: callback("invalidate"),
            close=lambda: callback("close"),
        )
        try:
            if phase == "initialize":
                with pytest.raises(TypeError):
                    await writer.start()
            else:
                await writer.start()
            if phase in ("write", "invalidate"):
                with pytest.raises(TypeError):
                    await writer.run(lambda: callback("write"))
            if phase != "close":
                assert writer.state == StorageWriterState.FAILED
                assert writer.ownership_held and "close" not in calls
            report = await writer.shutdown(timeout=2)
            if phase == "close":
                assert report.state == StorageWriterState.FAILED
                assert report.failed is True and report.ownership_held is True
                assert report.pending is False
                _assert_busy(tmp_path)
                assert await writer.shutdown(timeout=0) == report
            else:
                _assert_closed(report, failed=True)
            assert len(created) == 1
            assert inspect.getcoroutinestate(created[0]) == inspect.CORO_CLOSED
            assert bodies == []
        finally:
            try:
                await _finish(writer)
            finally:
                for coroutine in created:
                    coroutine.close()

    _run(scenario())
    expected = ["initialize", "close"]
    if phase in ("write", "invalidate"):
        expected = ["initialize", "write", "invalidate", "close"]
    assert calls == expected


# ==================== 真实操作的晚到异常必须被观察 ====================


@pytest.mark.parametrize("phase", ["initialize", "write", "close"])
def test_late_real_future_failure_is_observed_after_waiter_cancellation(
    tmp_path, monkeypatch, caplog, phase
):
    failure = RuntimeError("合成真实操作晚到异常")
    logger = storage.logger
    # Alembic fileConfig 可能禁用已收集的 logger，只恢复本用例需要的实例。
    monkeypatch.setattr(logger, "disabled", False)
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level(logging.ERROR, logger=logger.name)

    async def exercise():
        entered, release = threading.Event(), threading.Event()
        wrapped_pairs, observed, waiters = [], [], []
        original_wrap = asyncio.wrap_future
        source = None

        def capture(future, *args, **kwargs):
            wrapped = original_wrap(future, *args, **kwargs)
            wrapped_pairs.append((future, wrapped))
            return wrapped

        def callback(name):
            if name == phase:
                _block(entered, release)
                raise failure

        writer = _writer(
            tmp_path,
            initialize=lambda: callback("initialize"),
            close=lambda: callback("close"),
        )
        try:
            with monkeypatch.context() as patch:
                patch.setattr(asyncio, "wrap_future", capture)
                if phase != "initialize":
                    await writer.start()
                operation = {
                    "initialize": writer.start,
                    "write": lambda: writer.run(lambda: callback("write")),
                    "close": lambda: writer.shutdown(timeout=5),
                }[phase]
                waiter = asyncio.create_task(operation())
                waiters.append(waiter)
                await _until(entered.is_set)
                source, wrapped = next(
                    pair for pair in wrapped_pairs if pair[0].running()
                )
                original_exception = source.exception

                def observe_exception(*args, **kwargs):
                    error = original_exception(*args, **kwargs)
                    if error is failure:
                        observed.append(error)
                    return error

                # 只记录生产代码的 exception() 调用，测试本身不替它取走异常。
                source.exception = observe_exception
                waiter.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiter
                assert not source.cancelled() and not source.done()
                assert not wrapped.cancelled() and not wrapped.done()
                release.set()
                await _until(lambda: source.done() and wrapped.done())
                await asyncio.sleep(0)
                assert observed and not wrapped.cancelled()
                assert not wrapped._log_traceback, "asyncio 包装 Future 的异常尚未回收"
                report = await writer.shutdown(timeout=2)
                if phase == "close":
                    assert report.failed is True and report.ownership_held is True
                    assert report.pending is False
                else:
                    _assert_closed(report, failed=phase == "initialize")
                return [weakref.ref(pair[1]) for pair in wrapped_pairs]
        finally:
            if source is not None:
                del source.exception
            await _finish(writer, releases=(release,), waiters=waiters)

    async def scenario():
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        unhandled = []
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        try:
            references = await exercise()
            failure.__traceback__ = None
            for _ in range(3):
                gc.collect()
                await asyncio.sleep(0)
            assert all(reference() is None for reference in references)
            assert unhandled == [], "事件循环收到未回收异常或遗留任务"
        finally:
            loop.set_exception_handler(previous)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        _run(scenario())
    assert not any("never awaited" in str(warning.message) for warning in caught)
    assert any(
        record.name == logger.name
        and record.levelno == logging.ERROR
        and "操作失败" in record.getMessage()
        for record in caplog.records
    )
    assert not any(
        "exception was never retrieved" in message.lower()
        for message in caplog.messages
    )


def test_quarantined_asyncio_future_late_failure_is_observed(tmp_path):
    async def exercise():
        external = asyncio.get_running_loop().create_future()
        reference = weakref.ref(external)
        writer = _writer(tmp_path)
        try:
            await writer.start()
            with pytest.raises(TypeError):
                await writer.run(lambda: external)
            # 已隔离的外部对象晚于受管操作失败；保留引用不能替代异常观察。
            external.set_exception(RuntimeError("隔离 Future 的晚到合成异常"))
            for _ in range(3):
                await asyncio.sleep(0)
            return reference, not external._log_traceback
        finally:
            # 这是无执行体的 Future，不把真实外部工作伪装为已停止。
            if not external.done():
                external.set_result(None)
            await _finish(writer)

    async def scenario():
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        unhandled = []
        # 不保存 context/future，否则测试的观察器自身会阻止对象被回收。
        loop.set_exception_handler(
            lambda _loop, context: unhandled.append(context["message"])
        )
        try:
            reference, observed = await exercise()
            for _ in range(3):
                gc.collect()
                await asyncio.sleep(0)
            assert reference() is None, "必须实际回收隔离对象，而非靠强引用隐藏异常"
            assert observed, (
                f"隔离的 asyncio.Future 异常未被观察，回收诊断：{unhandled}"
            )
            assert unhandled == [], "外部 Future 的晚到异常不应成为未回收警告"
        finally:
            loop.set_exception_handler(previous)

    _run(scenario())


@pytest.mark.parametrize("completed", [False, True])
def test_external_future_with_closed_loop_stays_quarantined(tmp_path, completed):
    owner_loop = asyncio.new_event_loop()
    external = owner_loop.create_future()
    if completed:
        external.set_exception(RuntimeError("已关闭循环的合成异常"))
    owner_loop.close()
    calls = []

    async def scenario():
        writer = _writer(
            tmp_path,
            invalidate=lambda: calls.append("失效"),
            close=lambda: calls.append("关闭"),
        )
        try:
            await writer.start()
            with pytest.raises(TypeError):
                await writer.run(lambda: external)
            report = await writer.shutdown(timeout=2)
            assert report.failed and report.pending and report.ownership_held
            assert calls == [] and external.done() is completed
            _assert_busy(tmp_path)
            if completed:
                assert not external._log_traceback
        finally:
            # 合成 Future 无执行体；完成并观察后才回收本测试的目录锁。
            if not external.done():
                external.set_result(None)
            external.exception()
            await _finish(writer)

    _run(scenario())


# ==================== 冷导入与 fork 仅在无站点包的受控解释器中执行 ====================


_PROCESS_GUARD = r"""
import os
import sys

sys.path.insert(0, sys.argv[1])
blocked_roots = {
    "dotenv", "chromadb", "sqlalchemy", "redis", "torch", "transformers",
    "sentence_transformers", "llama_index", "langchain", "langchain_core",
    "langchain_openai", "pymysql", "mysql", "httpx", "requests", "aiohttp",
}
blocked_prefixes = ("app.db", "app.rag", "app.core.config")

class ImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked_roots or any(
            fullname == prefix or fullname.startswith(prefix + ".")
            for prefix in blocked_prefixes
        ):
            raise AssertionError("受控解释器禁止导入重资源依赖：" + fullname)

sys.meta_path.insert(0, ImportGuard())

def audit(event, args):
    if event in {"socket.connect", "socket.bind", "socket.getaddrinfo", "socket.sendto"}:
        raise AssertionError("受控解释器禁止访问网络")
    if event != "open" or not isinstance(args[0], (str, bytes)):
        return
    path = os.path.abspath(os.fsdecode(args[0]))
    data = os.path.join(sys.argv[1], "data")
    if os.path.basename(path).startswith(".env") or path == data or path.startswith(data + os.sep):
        raise AssertionError("禁止读取环境文件或真实数据")
    mode, flags = args[1:3]
    if (isinstance(mode, str) and any(char in mode for char in "wax+")) or flags & (
        os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    ):
        raise AssertionError("受控解释器禁止写入文件")

sys.addaudithook(audit)
"""


def _interpreter(code, backend_root, directory):
    result = subprocess.run(
        [
            "timeout",
            "--kill-after=5s",
            "12s",
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            _PROCESS_GUARD + code,
            str(backend_root),
            str(directory),
        ],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def test_cold_import_and_constructor_create_no_threads_or_storage_io(
    tmp_path, backend_root
):
    code = r"""
import _thread
import threading

before = {thread.ident for thread in threading.enumerate()}

def forbidden(*args, **kwargs):
    raise AssertionError("冷导入与构造不得启动线程、回调或访问存储")

threading.Thread.__init__ = forbidden
threading.Thread.start = forbidden
_thread.start_new_thread = forbidden
os.open = forbidden

from app.core.storage_writer import StorageWriter, StorageWriterState

original_stat = os.stat
os.stat = forbidden
os.mkdir = forbidden
os.listdir = forbidden
os.scandir = forbidden
path = os.path.join(sys.argv[2], "not-created")
try:
    writer = StorageWriter(path, initialize=forbidden, invalidate=forbidden, close=forbidden)
    assert writer.state == StorageWriterState.NEW
    assert writer.pending == 0 and not writer.accepting and not writer.ownership_held
finally:
    os.stat = original_stat
assert not os.path.exists(path)
assert {thread.ident for thread in threading.enumerate()} == before
assert not blocked_roots.intersection(sys.modules)
print("cold-safe")
"""
    assert _interpreter(code, backend_root, tmp_path) == "cold-safe"


def test_fork_rejects_all_public_members_before_inherited_mutex_access(
    tmp_path, backend_root
):
    code = r"""
import asyncio
import signal
import threading
import time
import traceback
import warnings
from app.core.storage_ownership import StorageOwnership
from app.core.storage_writer import StorageWriter, StorageWriterError, StorageWriterState

calls = []
writer = StorageWriter(
    sys.argv[2], initialize=lambda: calls.append("初始化"),
    invalidate=lambda: calls.append("失效"), close=lambda: calls.append("关闭"),
)
entered, release = threading.Event(), threading.Event()
child, reaped = None, False

def hold_mutexes():
    with writer._mutex, writer._ownership._mutex:
        entered.set()
        assert release.wait(8), "测试必须释放父解释器的互斥量"

holder = threading.Thread(target=hold_mutexes)

def reject(access):
    try:
        access()
    except StorageWriterError:
        return
    raise AssertionError("fork 后公开入口必须拒绝父对象")

def wait_child(pid, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        found, status = os.waitpid(pid, os.WNOHANG)
        if found:
            return status
        time.sleep(0.01)
    raise AssertionError("fork 子进程未在期限内退出")

try:
    asyncio.run(asyncio.wait_for(writer.start(), timeout=2))
    holder.start()
    assert entered.wait(2), "互斥量持有线程未就绪"
    # Python 3.12 会警告多线程 fork；此处恰好验证继承已锁互斥量的拒绝顺序。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        child = os.fork()
    if child == 0:
        signal.alarm(3)
        status = 1
        try:
            for name in ("state", "pending", "accepting", "ownership_held"):
                reject(lambda: getattr(writer, name))
            for operation in (
                writer.start,
                lambda: writer.run(lambda: calls.append("非法写入")),
                lambda: writer.shutdown(timeout=0),
            ):
                reject(lambda: asyncio.run(operation()))
            assert calls == ["初始化"]
            status = 0
        except BaseException:
            traceback.print_exc()
        finally:
            os._exit(status)
    status = wait_child(child, 5)
    reaped = True
    assert os.waitstatus_to_exitcode(status) == 0, "子进程未正确拒绝继承对象"
    release.set()
    holder.join(timeout=2)
    assert not holder.is_alive()
    assert asyncio.run(asyncio.wait_for(writer.run(lambda: "父通道正常"), 2)) == "父通道正常"
finally:
    try:
        if child is not None and child > 0 and not reaped:
            os.kill(child, signal.SIGKILL)
            wait_child(child, 2)
    finally:
        release.set()
        if holder.ident is not None:
            holder.join(timeout=2)
            assert not holder.is_alive(), "互斥量持有线程未退出"
        try:
            report = asyncio.run(asyncio.wait_for(writer.shutdown(timeout=2), 3))
        finally:
            if writer._thread is not None and writer._thread.ident is not None:
                writer._thread.join(timeout=2)
                assert not writer._thread.is_alive(), "存储写线程未退出"
            writer._ownership.release()
assert report.state == StorageWriterState.CLOSED
assert report.pending is False and report.ownership_held is False and report.failed is False
assert calls == ["初始化", "失效", "关闭"]
with StorageOwnership(sys.argv[2]):
    pass
print("fork-safe")
"""
    assert _interpreter(code, backend_root, tmp_path) == "fork-safe"
