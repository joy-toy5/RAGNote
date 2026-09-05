"""锁定版本信号基线；只给自行创建的无监听、无数据库子进程发信号。"""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
from importlib.metadata import version

import pytest

# 驱动真实 Server.install_signal_handlers/shutdown，不启动 HTTP 或生产应用。
PROBE = r"""
import asyncio, json, logging, sys, time
from types import SimpleNamespace
from uvicorn import Config, Server
logging.disable(logging.CRITICAL)
busy = sys.argv[1] == "busy"
started = time.monotonic()
def emit(phase):
    print(json.dumps({"phase": phase, "elapsed": time.monotonic() - started}), flush=True)
async def app(scope, receive, send):
    pass
async def run():
    server = Server(Config(app, lifespan="off", log_config=None))
    server.servers = []
    async def lifespan_shutdown():
        emit("lifespan")
    server.lifespan = SimpleNamespace(shutdown=lifespan_shutdown)
    release = asyncio.Event()
    if busy:
        task = asyncio.create_task(release.wait())
        server.server_state.tasks.add(task)
        task.add_done_callback(server.server_state.tasks.discard)
    server.install_signal_handlers()
    emit("ready")
    async with asyncio.timeout(4):
        while not server.should_exit:
            await asyncio.sleep(0.001)
        emit("signal")
        shutdown = asyncio.create_task(server.shutdown())
        if busy:
            await asyncio.sleep(0.25)
            if shutdown.done():
                raise AssertionError("活动 ASGI 任务未阻止 teardown")
            emit("blocked_before_lifespan")
            release.set()
        await shutdown
    emit("exit")
asyncio.run(run())
"""


def _read_ready(process: subprocess.Popen) -> dict:
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(timeout=5), "受控子进程未就绪"
        return json.loads(process.stdout.readline())


@pytest.mark.skipif(os.name != "posix", reason="基线仅针对 POSIX SIGTERM")
@pytest.mark.parametrize("mode", ["idle", "busy"])
def test_real_sigterm_shutdown_order_without_production_app(mode, tmp_path):
    assert version("uvicorn") == "0.21.1", "升级 Uvicorn 时必须重新审查该基线"
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", PROBE, mode],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "HOME", "LANG"}
        },
    )
    try:
        assert _read_ready(process)["phase"] == "ready"
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=6)
        assert process.returncode == 0, stderr
        events = [json.loads(line) for line in stdout.splitlines()]
        expected = ["signal", "lifespan", "exit"]
        if mode == "busy":
            expected.insert(1, "blocked_before_lifespan")
        assert [event["phase"] for event in events] == expected
        assert events[-1]["elapsed"] < 5
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=2)


def test_second_sigterm_does_not_force_exit_in_locked_version():
    from uvicorn import Config, Server

    server = Server(Config(app=lambda *args: None, log_config=None))
    server.handle_exit(signal.SIGTERM, None)
    server.handle_exit(signal.SIGTERM, None)
    assert server.should_exit and not server.force_exit
    server.handle_exit(signal.SIGINT, None)
    assert server.force_exit
