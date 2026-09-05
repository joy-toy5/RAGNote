"""由 ASGI 响应拥有异步生成器，发送失败也显式释放其资源。"""

from collections.abc import AsyncGenerator
from typing import Any

import anyio
from starlette.responses import StreamingResponse
from starlette.types import Send


class ClosingStreamingResponse(StreamingResponse):
    """仅接收可关闭的异步生成器；生成器自身负责有界的取消清理。"""

    def __init__(self, content: AsyncGenerator[str | bytes, None], **kwargs: Any):
        super().__init__(content, **kwargs)
        self._stream = content

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            # 断开发生在 send 而不是 anext 时，也不能依赖 GC 触发 finally。
            # AnyIO 会在取消域的每个等待点再次取消，关闭过程需局部屏蔽。
            with anyio.CancelScope(shield=True):
                await self._stream.aclose()
