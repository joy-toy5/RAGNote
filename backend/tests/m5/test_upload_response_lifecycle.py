"""真实 FastAPI 上传路由发送断开；共享离线替身，不监听 HTTP 端口。"""

from __future__ import annotations

import asyncio
import gc
import importlib.util
import json
import sys
import types

import httpx
import pytest
from fastapi import FastAPI
from starlette.requests import ClientDisconnect

from app.core.streaming_response import ClosingStreamingResponse
from test_upload_stream_lifecycle import _indexer, service_module


def _load_router(monkeypatch, backend_root, service):
    monkeypatch.setitem(sys.modules, "app.router.knowledge_service", service)
    stubs = {
        "app.utils.auth_utils": {"get_current_user_id": lambda: "test-user"},
        "app.utils.image_extractor": {"get_image_path": lambda *_: None},
        "app.core.success_response": {"success_response": lambda **kwargs: kwargs},
        "app.core.rate_limit": {"rate_limit": lambda **_: lambda: None},
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "_m5_upload_router", backend_root / "app/router/knowledge_router.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("with_middleware", [False, True])
@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_send_failure_closes_upload_before_gc_and_releases_capacity(
    service_module, monkeypatch, backend_root, with_middleware, spec_version
):
    module = service_module
    _indexer(module, monkeypatch)
    router = _load_router(monkeypatch, backend_root, module)
    runtime = module.upload_runtime
    acquired = []
    responses = []
    acquire = runtime.acquire

    def tracked_acquire():
        lease = acquire()
        acquired.append(lease)
        return lease

    class TrackedResponse(ClosingStreamingResponse):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 强引用响应和生成器，禁止测试靠引用计数/GC 触发关闭。
            responses.append(self)

    assert router.ClosingStreamingResponse is ClosingStreamingResponse
    monkeypatch.setattr(router, "ClosingStreamingResponse", TrackedResponse)
    monkeypatch.setattr(runtime, "acquire", tracked_acquire)
    app = FastAPI()
    app.include_router(router.knowledge_router)
    if with_middleware:
        @app.middleware("http")
        async def timing_header(request, call_next):
            response = await call_next(request)
            response.headers["X-Test-Time"] = "0"
            return response

    request = httpx.Request(
        "POST", "http://test/knowledge/add/multiple/stream",
        files=[("files", (f"{index}.txt", b"note", "text/plain")) for index in range(12)],
    )
    body = request.read()

    async def scenario():
        received = False
        failed_send = False

        async def receive():
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            nonlocal failed_send
            if message["type"] == "http.response.start":
                assert message["status"] == 200
            if message["type"] != "http.response.body" or not message.get("body"):
                return
            event = json.loads(message["body"].decode().split("data: ", 1)[1])
            if event.get("event_type") == "slicing_completed":
                # 在队列满、还有未消费解析任务时断开，而非空闲时直接关闭。
                async with asyncio.timeout(1):
                    while not acquired[0].queue.full():
                        await asyncio.sleep(0.001)
                failed_send = True
                raise OSError("受控 SSE 发送断开")

        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": spec_version},
            "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": "/knowledge/add/multiple/stream", "raw_path": b"/knowledge/add/multiple/stream",
            "query_string": b"", "root_path": "",
            "headers": [(name.lower(), value) for name, value in request.headers.raw],
            "client": ("127.0.0.1", 1), "server": ("test", 80),
        }
        automatic_gc = gc.isenabled()
        gc.disable()
        try:
            async with asyncio.timeout(2):
                with pytest.raises((OSError, ClientDisconnect)):
                    await app(scope, receive, send)
                assert failed_send
                assert len(acquired) == 1
                lease = acquired[0]
                assert lease._closed
                assert lease.queue.closed
                assert lease.queue.empty()
                assert responses[0]._stream.ag_frame is None
                while runtime.active_count:
                    await asyncio.sleep(0.001)
                assert not lease._futures and not lease._writes
                # 以下断言先于测试 finally 的 shutdown/aclose，不能靠善后修复泄漏。
                replacement = runtime.acquire()
                replacement.close()
                assert runtime.active_count == 0
        finally:
            # 即使断言失败也释放测试线程；不把失败变成 pytest 永久等待。
            assert await runtime.shutdown(timeout=1) == 0
            for response in responses:
                await response._stream.aclose()
            if automatic_gc:
                gc.enable()

    asyncio.run(scenario())
