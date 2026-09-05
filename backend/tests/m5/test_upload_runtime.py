"""只用受控线程验证上传桥接与执行所有权，不打开生产存储。"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Empty

import pytest

from app.rag.task_queue import TaskQueue


def test_empty_queue_wait_yields_to_event_loop() -> None:
    queue = TaskQueue(maxsize=1)

    async def scenario() -> None:
        waiting = asyncio.create_task(queue.get_async())
        await asyncio.sleep(0)
        assert not waiting.done()
        queue.put("result")
        assert await asyncio.wait_for(waiting, 1) == "result"
        queue.task_done()

    asyncio.run(scenario())


def test_close_releases_full_queue_producer_and_rejects_late_put() -> None:
    from app.rag.task_queue import QueueClosed

    queue = TaskQueue(maxsize=1)
    queue.put("first")
    started = threading.Event()

    def produce() -> None:
        started.set()
        queue.put("second")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(produce)
        try:
            assert started.wait(1)
            queue.close()
            with pytest.raises(QueueClosed):
                future.result(timeout=1)
            with pytest.raises(QueueClosed):
                queue.put("late", block=False)
            assert queue.empty()
            queue.join()
        finally:
            queue.close()


def test_empty_queue_timeout_and_cancellation_do_not_consume_results() -> None:
    queue = TaskQueue(maxsize=1)

    async def scenario() -> None:
        with pytest.raises(TimeoutError):
            await queue.get_async(timeout=0.01)
        waiting = asyncio.create_task(queue.get_async())
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        queue.put("kept")
        assert queue.get(block=False) == "kept"
        queue.task_done()
        with pytest.raises(Empty):
            queue.get(block=False)

    asyncio.run(scenario())


def test_lease_retains_running_thread_after_close_and_bounds_admission() -> None:
    from app.rag.upload_runtime import UploadBusy, UploadRuntime

    runtime = UploadRuntime(max_workers=1, max_uploads=1)
    lease = runtime.acquire()
    started, release = threading.Event(), threading.Event()

    def work() -> str:
        started.set()
        assert release.wait(2)
        return "done"

    future = lease.submit(work)
    try:
        assert started.wait(1)
        queued = lease.submit(lambda: "must-not-start")
        lease.close()
        assert queued.cancelled()
        assert not future.done()
        assert runtime.active_count == 1
        with pytest.raises(UploadBusy):
            runtime.acquire()
        pending = asyncio.run(runtime.shutdown(timeout=0.01))
        assert pending == 1
        assert runtime.active_count == 1
    finally:
        release.set()
        future.result(timeout=1)
        assert asyncio.run(runtime.shutdown(timeout=1)) == 0
    runtime.start()
    runtime.acquire().close()
    assert asyncio.run(runtime.shutdown(timeout=1)) == 0


def test_write_task_is_retained_not_cancelled_when_client_leaves() -> None:
    from app.rag.upload_runtime import UploadRuntime

    async def scenario() -> None:
        runtime = UploadRuntime(max_workers=1, max_uploads=1)
        lease = runtime.acquire()
        release = asyncio.Event()
        task = asyncio.create_task(release.wait())
        lease.track_write(task)
        lease.close()
        assert runtime.active_count == 1
        assert await runtime.shutdown(timeout=0.01) == 1
        assert not task.cancelled()
        release.set()
        await task
        assert await runtime.shutdown(timeout=1) == 0

    asyncio.run(scenario())


def test_externally_cancelled_write_is_quarantined_not_reported_stopped() -> None:
    from app.rag.upload_runtime import UploadBusy, UploadRuntime

    async def scenario() -> None:
        runtime = UploadRuntime(max_workers=1, max_uploads=1)
        lease = runtime.acquire()
        task = asyncio.create_task(asyncio.Event().wait())
        lease.track_write(task)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        lease.close()
        assert await runtime.shutdown(timeout=0) == 1
        with pytest.raises(RuntimeError, match="未结束"):
            runtime.start()
        with pytest.raises(UploadBusy):
            runtime.acquire()

    asyncio.run(scenario())
