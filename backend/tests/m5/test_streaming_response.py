"""真实 Starlette 响应的发送取消边界；不监听端口，不借助 GC 清理。"""

from __future__ import annotations

import asyncio

import anyio
import pytest
from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse

from app.core.streaming_response import ClosingStreamingResponse


def test_send_disconnect_closes_generator_inside_cancel_scope():
    async def scenario():
        entered, closed = asyncio.Event(), asyncio.Event()

        async def stream():
            try:
                yield "data: started\n\n"
            finally:
                entered.set()
                await asyncio.sleep(0)
                closed.set()

        response = ClosingStreamingResponse(stream(), media_type="text/event-stream")

        async def receive():
            await entered.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                entered.set()
                await asyncio.Event().wait()

        async with asyncio.timeout(1):
            await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
        assert closed.is_set()
        assert response._stream.ag_frame is None

    asyncio.run(scenario())


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_caller_cancel_at_send_closes_generator_and_propagates(spec_version):
    async def scenario():
        sending, closed = asyncio.Event(), asyncio.Event()

        async def stream():
            try:
                yield b"event"
            finally:
                await asyncio.sleep(0)
                closed.set()

        response = ClosingStreamingResponse(stream())

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.body":
                sending.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(response(
            {"type": "http", "asgi": {"spec_version": spec_version}}, receive, send
        ))
        try:
            async with asyncio.timeout(1):
                await sending.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert closed.is_set()
                assert response._stream.ag_frame is None
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await response._stream.aclose()

    asyncio.run(scenario())


def test_outer_anyio_cancel_at_send_finishes_async_cleanup():
    async def scenario():
        closed = asyncio.Event()

        async def stream():
            try:
                yield "event"
            finally:
                await asyncio.sleep(0)
                closed.set()

        response = ClosingStreamingResponse(stream())

        async def receive():
            await asyncio.Event().wait()

        with anyio.CancelScope() as scope:
            async def send(message):
                if message["type"] == "http.response.body":
                    scope.cancel()
                    await anyio.sleep(0)

            await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
        assert closed.is_set()
        assert response._stream.ag_frame is None

    asyncio.run(scenario())


def test_asgi_24_send_failure_closes_generator_before_client_disconnect():
    async def scenario():
        closed = asyncio.Event()

        async def stream():
            try:
                yield "event"
            finally:
                await asyncio.sleep(0)
                closed.set()

        response = ClosingStreamingResponse(stream())

        async def receive():
            raise AssertionError("ASGI 2.4 通过 send 异常判断断开")

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("受控发送断开")

        with pytest.raises(ClientDisconnect):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        assert closed.is_set()
        assert response._stream.ag_frame is None

    asyncio.run(scenario())


def test_normal_response_preserves_chunks_headers_and_closes_once():
    async def scenario():
        closes = []

        async def stream():
            try:
                yield "中文"
                yield b"bytes"
            finally:
                await asyncio.sleep(0)
                closes.append(True)

        raw = []
        response = ClosingStreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )
        assert isinstance(response, StreamingResponse)

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            raw.append(message)

        await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
        assert raw[0]["status"] == 200
        assert (b"cache-control", b"no-cache") in raw[0]["headers"]
        assert [part["body"] for part in raw[1:]] == ["中文".encode(), b"bytes", b""]
        assert raw[-1]["more_body"] is False
        assert closes == [True]
        assert response._stream.ag_frame is None

    asyncio.run(scenario())
