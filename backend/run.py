"""Demo单进程入口：在backend目录执行 `.venv/bin/python run.py`。

保留Uvicorn；SIGTERM先等待任务，超时取消，最多10秒后退出。
强制退出不保证在途写入原子性，不提供重启恢复。
"""
from __future__ import annotations

import asyncio
import math
import os
import threading

import uvicorn


class DemoServer(uvicorn.Server):
    def __init__(self, config, *, begin_shutdown, grace_period=5.0, exit_timeout=10.0):
        super().__init__(config)
        if not math.isfinite(grace_period) or grace_period < 0:
            raise ValueError("关闭等待时间必须是有限非负数")
        if not math.isfinite(exit_timeout) or exit_timeout <= grace_period:
            raise ValueError("退出上限必须大于任务等待时间")
        self._begin_shutdown = begin_shutdown
        self._grace_period = grace_period
        self._exit_timeout = exit_timeout
        self._exit_timer = None
        self._request_deadline = None

    def handle_exit(self, sig, frame):
        first_signal = not self.should_exit
        super().handle_exit(sig, frame)
        if not first_signal:
            return
        # 不可中断线程可能卡住asyncio.run的executor收尾；到总上限后退出进程。
        self._exit_timer = threading.Timer(self._exit_timeout, os._exit, args=(1,))
        self._exit_timer.daemon = True
        self._exit_timer.start()
        self._begin_shutdown()
        self._request_deadline = asyncio.create_task(self._cancel_request_waiters(), name="demo-http-drain")

    async def _cancel_request_waiters(self):
        await asyncio.sleep(self._grace_period)
        # 锁定的0.21.1会无限等待ASGI任务；只取消等待，不复制其关闭流程。
        for task in tuple(self.server_state.tasks):
            task.cancel()

    def run(self, sockets=None):
        try:
            super().run(sockets=sockets)
        finally:
            # 必须等asyncio.run完成（含默认线程池），不能在serve返回时提前撤掉上限。
            if self._exit_timer is not None:
                self._exit_timer.cancel()


def main():
    from main import SHUTDOWN_GRACE_SECONDS, app, begin_shutdown

    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    DemoServer(
        config, begin_shutdown=begin_shutdown,
        grace_period=SHUTDOWN_GRACE_SECONDS, exit_timeout=SHUTDOWN_GRACE_SECONDS + 5.0,
    ).run()


if __name__ == "__main__":
    main()
