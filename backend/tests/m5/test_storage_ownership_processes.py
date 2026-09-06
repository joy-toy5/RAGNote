"""只在一次性目录和受控解释器中验证内核所有权，不 fork pytest 主进程。"""

from __future__ import annotations

import select
import subprocess
import sys
from contextlib import contextmanager

import pytest

from app.core.storage_ownership import StorageBusyError, StorageOwnership

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="仅支持 Linux 本地目录锁")

_PRELUDE = """
import os
import select
import signal
import sys
from pathlib import Path
from app.core.storage_ownership import StorageBusyError, StorageOwnership, StorageOwnershipError

directory = Path(sys.argv[1])
signal.alarm(10)

def expect_busy():
    contender = StorageOwnership(directory)
    try:
        contender.acquire()
    except StorageBusyError:
        return
    finally:
        contender.release()
    raise AssertionError("仍有旧描述符时不应交出写权")
"""


def _run(code, directory, backend_root):
    result = subprocess.run(
        [sys.executable, "-B", "-c", _PRELUDE + code, str(directory)],
        cwd=backend_root, capture_output=True, text=True, timeout=12, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


@contextmanager
def _process(code, directory, backend_root):
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", _PRELUDE + code, str(directory)],
        cwd=backend_root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        yield process
    finally:
        try:
            process.communicate("\n" if process.poll() is None else None, timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=3)


def _readline(process):
    assert select.select([process.stdout], [], [], 5)[0], "受控子进程未在期限内回应"
    return process.stdout.readline().strip()


def _sendline(process):
    process.stdin.write("\n")
    process.stdin.flush()


def test_other_interpreter_cannot_initialize_before_acquiring(tmp_path, backend_root):
    code = """
try:
    with StorageOwnership(directory):
        (directory / "unexpected-initialization").write_text("禁止执行")
except StorageBusyError:
    print("busy")
else:
    print("initialized")
"""
    with StorageOwnership(tmp_path):
        assert _run(code, tmp_path, backend_root) == "busy"
    assert not (tmp_path / "unexpected-initialization").exists()
    assert _run(code, tmp_path, backend_root) == "initialized"


def test_abrupt_owner_exit_releases_lock_without_changing_assets(tmp_path, backend_root):
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"must survive")
    code = """
guard = StorageOwnership(directory).acquire()
print("owned", flush=True)
assert select.select([sys.stdin], [], [], 6)[0]
os._exit(23)
"""
    with _process(code, tmp_path, backend_root) as process:
        assert _readline(process) == "owned"
        with pytest.raises(StorageBusyError):
            StorageOwnership(tmp_path).acquire()
        _sendline(process)
        assert process.wait(timeout=3) == 23
        with StorageOwnership(tmp_path) as guard:
            guard.assert_owned()
    assert sentinel.read_bytes() == b"must survive"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["sentinel"]


def test_two_ready_processes_have_exactly_one_owner(tmp_path, backend_root):
    code = """
print("ready", flush=True)
assert select.select([sys.stdin], [], [], 6)[0]
sys.stdin.readline()
try:
    guard = StorageOwnership(directory).acquire()
except StorageBusyError:
    print("busy", flush=True)
else:
    print("owned", flush=True)
    assert select.select([sys.stdin], [], [], 6)[0]
    sys.stdin.readline()
    guard.release()
"""
    with _process(code, tmp_path, backend_root) as first:
        with _process(code, tmp_path, backend_root) as second:
            assert _readline(first) == _readline(second) == "ready"
            _sendline(first)
            _sendline(second)
            assert sorted([_readline(first), _readline(second)]) == ["busy", "owned"]
    assert first.returncode == second.returncode == 0
    with StorageOwnership(tmp_path) as guard:
        guard.assert_owned()


_FORK_INHERITANCE = """
guard = StorageOwnership(directory).acquire()
ready_read, ready_write = os.pipe()
stop_read, stop_write = os.pipe()
# 模拟 fork 时其他线程正占用对象互斥量；子进程必须在访问它之前拒绝。
guard._mutex.acquire()
child = os.fork()
if child == 0:
    signal.alarm(6)
    os.close(ready_read)
    os.close(stop_write)
    status = 1
    try:
        assert not guard.acquired
        for operation in (guard.acquire, guard.assert_owned, guard.release):
            try:
                operation()
            except StorageOwnershipError:
                pass
            else:
                raise AssertionError("不能跨 fork 使用父对象")
        os.write(ready_write, b"ok")
        assert select.select([stop_read], [], [], 5)[0]
        os.read(stop_read, 1)
        status = 0
    finally:
        os._exit(status)
guard._mutex.release()
os.close(ready_write)
os.close(stop_read)
try:
    assert select.select([ready_read], [], [], 5)[0]
    assert os.read(ready_read, 2) == b"ok"
    guard.assert_owned()
    expect_busy()
    guard.release()
    # 父对象已释放，但 fork 子进程仍持有同一打开文件描述。
    expect_busy()
finally:
    os.close(stop_write)
    os.close(ready_read)
    _, status = os.waitpid(child, 0)
    guard.release()
assert os.waitstatus_to_exitcode(status) == 0
with StorageOwnership(directory) as successor:
    successor.assert_owned()
print("fork-safe")
"""


def test_fork_cannot_release_parent_or_bypass_inherited_descriptor(tmp_path, backend_root):
    assert _run(_FORK_INHERITANCE, tmp_path, backend_root) == "fork-safe"


_EXEC_INHERITANCE = """
guard = StorageOwnership(directory).acquire()
ready_read, ready_write = os.pipe()
stop_read, stop_write = os.pipe()
os.set_inheritable(ready_write, True)
os.set_inheritable(stop_read, True)
child = os.fork()
if child == 0:
    signal.alarm(6)
    os.close(ready_read)
    os.close(stop_write)
    program = '''
import os, select, sys
ready, stop = map(int, sys.argv[1:])
os.write(ready, b"ok")
assert select.select([stop], [], [], 5)[0]
os.read(stop, 1)
'''
    os.execv(sys.executable, [sys.executable, "-B", "-c", program, str(ready_write), str(stop_read)])
os.close(ready_write)
os.close(stop_read)
try:
    assert select.select([ready_read], [], [], 5)[0]
    assert os.read(ready_read, 2) == b"ok"
    guard.release()
    # 收到 exec 后的就绪信号，且子进程仍存活，证明并非退出才释放了描述符。
    assert os.waitpid(child, os.WNOHANG) == (0, 0)
    with StorageOwnership(directory) as successor:
        successor.assert_owned()
finally:
    os.close(stop_write)
    os.close(ready_read)
    _, status = os.waitpid(child, 0)
    guard.release()
assert os.waitstatus_to_exitcode(status) == 0
print("exec-safe")
"""


def test_exec_closes_ownership_descriptor_while_child_remains_alive(tmp_path, backend_root):
    assert _run(_EXEC_INHERITANCE, tmp_path, backend_root) == "exec-safe"
