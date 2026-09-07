"""只给自行创建、无HTTP监听/数据库/模型的子进程发送SIGTERM。"""
from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys

import pytest

PROBE = r'''
import asyncio, importlib.util, json, logging, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
from uvicorn import Config
root, mode = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root))
from app.core.task_registry import TaskRegistry
spec = importlib.util.spec_from_file_location("demo_runner", root / "run.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
logging.disable(logging.CRITICAL)
registry = TaskRegistry()
started = time.monotonic()
release = None
draining = None
def emit(phase):
    print(json.dumps(dict(phase=phase, elapsed=time.monotonic()-started)), flush=True)
async def work():
    try:
        await release.wait()
        emit("work_done")
    except asyncio.CancelledError:
        emit("work_cancelled")
        raise
async def drain():
    pending = await registry.drain(timeout=0.08)
    if pending:
        await registry.cancel_and_wait(timeout=0.1)
    emit("drained")
def begin_shutdown():
    global draining
    registry.stop_accepting()
    assert not registry.accepting
    emit("stop")
    if mode == "normal":
        asyncio.get_running_loop().call_later(0.02, release.set)
    draining = asyncio.create_task(drain())
async def cleanup():
    await draining
    emit("cleanup")
class ProbeServer(module.DemoServer):
    async def serve(self, sockets=None):
        global release
        release = asyncio.Event()
        self.servers = []
        self.lifespan = SimpleNamespace(shutdown=cleanup)
        if mode == "thread":
            thread_started = threading.Event()
            def blocking():
                thread_started.set()
                time.sleep(10)
            registry.create(asyncio.to_thread(blocking), name="synthetic-thread")
            while not thread_started.is_set():
                await asyncio.sleep(0.001)
        else:
            registry.create(work(), name="synthetic-work")
        if mode == "http":
            task = asyncio.create_task(asyncio.Event().wait())
            self.server_state.tasks.add(task)
            task.add_done_callback(self.server_state.tasks.discard)
        self.install_signal_handlers()
        emit("ready")
        while not self.should_exit:
            await asyncio.sleep(0.001)
        await self.shutdown()
        emit("serve_done")
server = ProbeServer(Config(lambda *args: None, loop="asyncio", lifespan="off", log_config=None),
                     begin_shutdown=begin_shutdown, grace_period=0.08, exit_timeout=0.7)
server.run()
emit("process_done")
'''


@pytest.mark.skipif(os.name != "posix", reason="受控信号用例仅适用于POSIX")
@pytest.mark.parametrize("mode", ["normal", "timeout", "http", "thread"])
def test_demo_sigterm_stops_waits_cleans_and_exits(mode, backend_root, tmp_path):
    assert (backend_root / "run.py").is_file(), "缺少demo基础关闭入口"
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", PROBE, str(backend_root), mode],
        cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={key: os.environ[key] for key in ("PATH", "HOME", "LANG") if key in os.environ},
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=5), "受控子进程未就绪"
            assert json.loads(process.stdout.readline())["phase"] == "ready"
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == (1 if mode == "thread" else 0), stderr
        events = [json.loads(line) for line in stdout.splitlines()]
        phases = [event['phase'] for event in events]
        assert phases[0] == "stop"
        assert phases.index("drained") < phases.index("cleanup") < phases.index("serve_done")
        if mode == "normal":
            assert "work_done" in phases and "work_cancelled" not in phases
        elif mode != "thread":
            assert "work_cancelled" in phases
        assert ("process_done" in phases) is (mode != "thread")
        assert events[-1]['elapsed'] < 2
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=2)
