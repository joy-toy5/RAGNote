"""隔离加载真实上传编排，禁止导入 Chroma、配置或真实索引后端。"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import logging
import sys
import threading
import types
from pathlib import Path

from fastapi import FastAPI
import pytest
from starlette.datastructures import UploadFile
from starlette.middleware.base import BaseHTTPMiddleware

from app.rag.upload_runtime import UploadRuntime


@pytest.fixture
def service_module(monkeypatch, backend_root: Path):
    for name, attributes in {
        "app.core.logger_handler": {"logger": logging.getLogger("test-upload")},
        "app.rag.vector_store": {"VectorStoreService": lambda: object()},
        "app.rag.indexing_service": {"UploadIndexingService": object},
        "magic": {
            "Magic": lambda **_: types.SimpleNamespace(
                from_buffer=lambda _: "text/plain"
            )
        },
    }.items():
        stub = types.ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
    spec = importlib.util.spec_from_file_location(
        "_m5_knowledge_service", backend_root / "app/router/knowledge_service.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "upload_runtime",
        UploadRuntime(max_workers=1, max_uploads=1),
        raising=False,
    )
    return module


def _file(content: bytes = b"note") -> UploadFile:
    return UploadFile(io.BytesIO(content), filename="test.txt")


def _events(raw: list[str]) -> list[dict]:
    return [json.loads(event.split("data: ", 1)[1]) for event in raw]


def _indexer(module, monkeypatch, *, parse=None, write=None):
    class Indexer:
        def __init__(self, store):
            pass

        def stage_upload(self, content, **kwargs):
            return content

        def prepare_upload_sync(self, staged):
            if parse:
                parse()
            return types.SimpleNamespace(documents=[], legacy_md5="fake")

        async def persist_and_index(self, prepared):
            if write:
                await write()

    monkeypatch.setattr(module, "UploadIndexingService", Indexer)


def test_parser_failure_is_reported_and_batch_has_terminal_event(
    service_module, monkeypatch
):
    module = service_module

    def fail():
        raise ValueError("受控解析失败")

    _indexer(module, monkeypatch, parse=fail)

    async def scenario():
        raw = [
            event
            async for event in module.KnowledgeService().handle_add_vector_multiple_stream(
                [_file()], "u"
            )
        ]
        events = _events(raw)
        assert any("受控解析失败" in event.get("error_message", "") for event in events)
        assert events[-1]["event_type"] == "finish"
        assert events[-1]["failed_count"] == 1
        assert await module.upload_runtime.shutdown(timeout=1) == 0

    asyncio.run(scenario())


def test_disconnect_during_parse_retains_thread_and_does_not_start_write(
    service_module, monkeypatch
):
    module = service_module
    started, release = threading.Event(), threading.Event()
    writes = []

    def parse():
        started.set()
        assert release.wait(2)

    async def write():
        writes.append(True)

    _indexer(module, monkeypatch, parse=parse, write=write)

    async def scenario():
        stream = module.KnowledgeService().handle_add_vector_multiple_stream(
            [_file()], "u"
        )
        await anext(stream)
        waiter = asyncio.create_task(anext(stream))
        try:
            async with asyncio.timeout(1):
                while not started.is_set():
                    await asyncio.sleep(0.001)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert module.upload_runtime.active_count == 1
        finally:
            release.set()
            await stream.aclose()
            assert await module.upload_runtime.shutdown(timeout=1) == 0
        assert not writes

    asyncio.run(scenario())


def test_parse_timeout_is_visible_but_not_a_false_thread_cancellation(
    service_module, monkeypatch
):
    module = service_module
    release = threading.Event()
    _indexer(module, monkeypatch, parse=lambda: release.wait(2))
    monkeypatch.setattr(module, "UPLOAD_WAIT_TIMEOUT", 0.02, raising=False)

    async def scenario():
        try:
            async with asyncio.timeout(1):
                events = _events(
                    [
                        e
                        async for e in module.KnowledgeService().handle_add_vector_multiple_stream(
                            [_file()], "u"
                        )
                    ]
                )
            assert events[-1]["event_type"] == "error"
            assert "超时" in events[-1]["error_message"]
            assert module.upload_runtime.active_count == 1
        finally:
            release.set()
            assert await module.upload_runtime.shutdown(timeout=1) == 0

    asyncio.run(scenario())


def test_disconnect_during_write_keeps_write_owned_until_completion(
    service_module, monkeypatch
):
    module = service_module

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        completed = []

        async def write():
            started.set()
            await release.wait()
            completed.append(True)

        _indexer(module, monkeypatch, write=write)
        stream = module.KnowledgeService().handle_add_vector_multiple_stream(
            [_file()], "u"
        )

        async def consume():
            async for _ in stream:
                pass

        consumer = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(started.wait(), 1)
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await consumer
            assert module.upload_runtime.active_count == 1
            assert not completed
        finally:
            release.set()
            await stream.aclose()
            assert await module.upload_runtime.shutdown(timeout=1) == 0
        assert completed == [True]

    asyncio.run(scenario())


def test_validation_reads_bounded_bytes_even_when_size_is_unknown(
    service_module, monkeypatch
):
    module = service_module
    monkeypatch.setattr(module, "MAX_FILE_SIZE", 4)
    reads = []

    class File:
        filename = "big.txt"
        size = None

        async def read(self, size=-1):
            reads.append(size)
            return b"0123456789"[:size] if size >= 0 else b"0123456789"

        async def seek(self, offset):
            pass

    async def scenario():
        valid, errors, _ = await module.KnowledgeService()._validate_and_read_files(
            [File()]
        )
        assert not valid and errors
        assert reads == [5]

    asyncio.run(scenario())


def test_bounded_queue_drains_all_files_and_releases_capacity(
    service_module, monkeypatch
):
    module = service_module
    writes = []

    async def write():
        await asyncio.sleep(0.002)
        writes.append(True)

    _indexer(module, monkeypatch, write=write)

    async def scenario():
        try:
            async with asyncio.timeout(2):
                events = _events(
                    [
                        e
                        async for e in module.KnowledgeService().handle_add_vector_multiple_stream(
                            [_file() for _ in range(12)], "u"
                        )
                    ]
                )
            assert events[-1]["event_type"] == "finish"
            assert events[-1]["success_count"] == 12
            assert events[-1]["failed_count"] == 0
            assert len(writes) == 12
            assert module.upload_runtime.active_count == 0
        finally:
            assert await module.upload_runtime.shutdown(timeout=1) == 0

    asyncio.run(scenario())


def test_unexpected_thread_error_does_not_spin_on_empty_queue(
    service_module, monkeypatch
):
    module = service_module
    _indexer(module, monkeypatch)

    def fail(*args):
        raise RuntimeError("线程未投递结果")

    monkeypatch.setattr(module, "_sync_slice_file", fail)

    async def scenario():
        try:
            async with asyncio.timeout(1):
                events = _events(
                    [
                        e
                        async for e in module.KnowledgeService().handle_add_vector_multiple_stream(
                            [_file()], "u"
                        )
                    ]
                )
            assert events[-1]["event_type"] == "error"
            assert "线程未投递结果" in events[-1]["error_message"]
            assert module.upload_runtime.active_count == 0
        finally:
            assert await module.upload_runtime.shutdown(timeout=1) == 0

    asyncio.run(scenario())


def test_disconnect_after_slice_event_accounts_for_claimed_queue_item(
    service_module, monkeypatch
):
    module = service_module
    _indexer(module, monkeypatch)

    async def scenario():
        service = module.KnowledgeService()
        lease = module.upload_runtime.acquire()
        try:
            files, _, _ = await service._validate_and_read_files([_file()])
            futures = service._start_slicing(files, "u", lease, object())
            stream = service._process_slice_results(
                lease, futures, object(), module.ProcessingState(total_valid=1)
            )
            await asyncio.wait_for(anext(stream), 1)
            lease.close()
            await stream.aclose()
            assert lease.queue._queue.unfinished_tasks == 0
        finally:
            lease.close()
            assert await module.upload_runtime.shutdown(timeout=1) == 0

    asyncio.run(scenario())


def test_capacity_rejection_does_not_read_additional_uploads(service_module):
    module = service_module
    lease = module.upload_runtime.acquire()

    class File:
        async def read(self, *args):
            pytest.fail("满载后不应继续读文件")

    async def scenario():
        try:
            events = _events(
                [
                    e
                    async for e in module.KnowledgeService().handle_add_vector_multiple_stream(
                        [File()], "u"
                    )
                ]
            )
            assert events[-1]["event_type"] == "error"
            assert "容量" in events[-1]["error_message"]
            assert module.upload_runtime.active_count == 1
        finally:
            lease.close()
            assert await module.upload_runtime.shutdown(timeout=1) == 0

    asyncio.run(scenario())


def test_file_limit_rejects_without_allocating_lease(service_module, monkeypatch):
    module = service_module
    monkeypatch.setattr(module, "MAX_UPLOAD_FILES", 1)

    async def scenario():
        events = _events(
            [
                e
                async for e in module.KnowledgeService().handle_add_vector_multiple_stream(
                    [_file(), _file()], "u"
                )
            ]
        )
        assert events[-1]["event_type"] == "error"
        assert "1个文件" in events[-1]["error_message"]
        assert module.upload_runtime.active_count == 0

    asyncio.run(scenario())


def test_write_failure_is_not_swallowed_and_validation_failures_are_counted(
    service_module, monkeypatch
):
    module = service_module
    monkeypatch.setattr(module, "MAX_FILE_SIZE", 4)

    async def write():
        raise ValueError("受控索引失败")

    _indexer(module, monkeypatch, write=write)

    async def scenario():
        try:
            events = _events(
                [
                    e
                    async for e in module.KnowledgeService().handle_add_vector_multiple_stream(
                        [_file(b"too large"), _file()], "u"
                    )
                ]
            )
            assert any("受控索引失败" in e.get("error_message", "") for e in events)
            assert events[-1]["event_type"] == "finish"
            assert events[-1]["failed_count"] == 2
            assert events[-1]["success_count"] == 0
        finally:
            assert await module.upload_runtime.shutdown(timeout=1) == 0

    asyncio.run(scenario())


def test_write_timeout_error_is_per_file_failure_and_batch_continues(
    service_module, monkeypatch
):
    module = service_module
    write_calls = []

    async def write():
        write_calls.append(True)
        if len(write_calls) == 1:
            raise TimeoutError("受控底层写入超时")

    _indexer(module, monkeypatch, write=write)

    async def scenario():
        try:
            events = _events(
                [
                    e
                    async for e in module.KnowledgeService().handle_add_vector_multiple_stream(
                        [_file(), _file()], "u"
                    )
                ]
            )
            write_errors = [
                event
                for event in events
                if event.get("event_type") == "error"
                and event.get("step") == "writing"
            ]
            assert len(write_errors) == 1
            assert write_errors[0]["error_message"] == "受控底层写入超时"
            assert events[-1]["event_type"] == "finish"
            assert events[-1]["failed_count"] == 1
            assert events[-1]["success_count"] == 1
            assert write_calls == [True, True]
            assert module.upload_runtime.active_count == 0
        finally:
            assert await module.upload_runtime.shutdown(timeout=1) == 0

    asyncio.run(scenario())


def test_write_timeout_keeps_ownership_until_actual_completion(
    service_module, monkeypatch
):
    module = service_module
    monkeypatch.setattr(module, "UPLOAD_WAIT_TIMEOUT", 0.02)

    async def scenario():
        release = asyncio.Event()
        completed = []

        async def write():
            await release.wait()
            completed.append(True)

        _indexer(module, monkeypatch, write=write)
        try:
            async with asyncio.timeout(1):
                events = _events(
                    [
                        e
                        async for e in module.KnowledgeService().handle_add_vector_multiple_stream(
                            [_file()], "u"
                        )
                    ]
                )
            assert events[-1]["event_type"] == "error"
            assert "索引超时" in events[-1]["error_message"]
            assert module.upload_runtime.active_count == 1
            assert not completed
        finally:
            release.set()
            assert await module.upload_runtime.shutdown(timeout=1) == 0
        assert completed == [True]

    asyncio.run(scenario())


def test_folder_limit_stops_reading_at_first_overflow(service_module, monkeypatch):
    module = service_module
    monkeypatch.setattr(module, "MAX_FILE_SIZE", 4)
    monkeypatch.setattr(module, "MAX_FOLDER_SIZE", 6)
    reads = []

    class File:
        filename = "test.txt"

        async def read(self, size=-1):
            reads.append(size)
            return b"data"[:size]

        async def seek(self, offset):
            pass

    async def scenario():
        valid, errors, _ = await module.KnowledgeService()._validate_and_read_files(
            [File(), File(), File()]
        )
        assert not valid and len(errors) == 1
        assert reads == [5, 3]
        assert "文件总大小" in errors[0]

    asyncio.run(scenario())


@pytest.mark.parametrize("contents", [[], [b"too large"]])
def test_no_valid_files_still_has_terminal_counts(
    service_module, monkeypatch, contents
):
    module = service_module
    monkeypatch.setattr(module, "MAX_FILE_SIZE", 4)

    async def scenario():
        events = _events(
            [
                e
                async for e in module.KnowledgeService().handle_add_vector_multiple_stream(
                    [_file(c) for c in contents], "u"
                )
            ]
        )
        assert events[-1]["event_type"] == "finish"
        assert events[-1]["failed_count"] == len(contents)
        assert events[-1]["success_count"] == 0
        assert module.upload_runtime.active_count == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("with_middleware", [False, True])
def test_real_fastapi_send_disconnect_has_no_unmanaged_wait_task(
    service_module, monkeypatch, with_middleware
):
    module = service_module
    monkeypatch.setitem(sys.modules, "app.router.knowledge_service", module)
    # 本例保留原P0低层写入/响应拥有权断言；持久接单由demo集成测试单独覆盖。
    async def submit_upload(service, files, user_id):
        return "p0-upload-test", service.handle_add_vector_multiple_stream(files, user_id)

    upload = types.ModuleType("app.tasking.upload")
    upload.submit_upload = submit_upload
    monkeypatch.setitem(sys.modules, upload.__name__, upload)
    # 路由只保留真实响应装配，鉴权/限流等依赖不得读取本机 .env。
    for name, attributes in {
        "app.utils.auth_utils": {"get_current_user_id": lambda: "u"},
        "app.utils.image_extractor": {"get_image_path": lambda *_: None},
        "app.core.success_response": {"success_response": lambda **kwargs: kwargs},
        "app.core.rate_limit": {"rate_limit": lambda **_: lambda: None},
    }.items():
        stub = types.ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)

    route_path = Path(module.__file__).with_name("knowledge_router.py")
    spec = importlib.util.spec_from_file_location("_m5_knowledge_router", route_path)
    route_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, route_module)
    spec.loader.exec_module(route_module)

    app = FastAPI()
    if with_middleware:

        class PassThroughMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                return await call_next(request)

        app.add_middleware(PassThroughMiddleware)
    app.include_router(route_module.knowledge_router)
    app.dependency_overrides[route_module.get_current_user_id] = lambda: "u"
    app.dependency_overrides[route_module.get_knowledge_service] = (
        lambda: module.KnowledgeService()
    )

    boundary = "m5-upload-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="files"; filename="test.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
        "note\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    write_started = asyncio.Event()
    release_write = asyncio.Event()

    async def write():
        write_started.set()
        await release_write.wait()

    _indexer(module, monkeypatch, write=write)

    async def scenario():
        request_sent = False

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            await write_started.wait()
            return {"type": "http.disconnect"}

        async def send(_message):
            return None

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/knowledge/add/multiple/stream",
            "raw_path": b"/knowledge/add/multiple/stream",
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (
                    b"content-type",
                    f"multipart/form-data; boundary={boundary}".encode(),
                ),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }

        try:
            async with asyncio.timeout(1):
                await app(scope, receive, send)

            leases = tuple(module.upload_runtime._leases)
            assert len(leases) == 1
            lease = leases[0]
            assert lease._closed is True
            write_tasks = tuple(lease._writes)
            assert len(write_tasks) == 1
            write_task = write_tasks[0]
            assert not write_task.done()

            current = asyncio.current_task()
            live_tasks = {
                task
                for task in asyncio.all_tasks()
                if not task.done() and task is not current
            }
            assert live_tasks == {write_task}
        finally:
            release_write.set()
            assert await module.upload_runtime.shutdown(timeout=1) == 0

    asyncio.run(scenario())
