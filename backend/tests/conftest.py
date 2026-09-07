from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BACKEND_ROOT.parent
ONLINE_MARKERS = {"integration", "external", "destructive"}
OFFLINE_ENV = {
    "PIP_NO_INDEX": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "LANGSMITH_TRACING": "false",
    "PYTHON_DOTENV_DISABLED": "1",
}
_ORIGINAL_ENV: dict[str, str | None] = {}
_ORIGINAL_SOCKET: dict[tuple[object, str], Any] = {}
_ORIGINAL_CHROMA: dict[tuple[object, str], Any] = {}
_AUDIT_INSTALLED = False
_OFFLINE_ACTIVE = False
_PATH_EVENTS = {
    "open", "os.listdir", "os.scandir", "os.mkdir", "os.remove", "os.rmdir", "os.chdir", "sqlite3.connect",
}
_TWO_PATH_EVENTS = {"os.rename", "os.link", "os.symlink"}


def _explicit_online_suite(config: pytest.Config) -> bool:
    """显式选择外部、集成或破坏性 marker 时，不安装默认离线守卫。"""
    arguments = list(config.invocation_params.args)
    expressions: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == "-m" and index + 1 < len(arguments):
            expressions.append(arguments[index + 1])
        elif argument.startswith("-m="):
            expressions.append(argument.removeprefix("-m="))
    return any(
        marker in expression and f"not {marker}" not in expression
        for expression in expressions
        for marker in ONLINE_MARKERS
    )


def _blocked_network(*_: object, **__: object) -> None:
    raise RuntimeError("M0 默认离线测试禁止建立网络连接")


def _blocked_chroma(*_: object, **__: object) -> None:
    raise RuntimeError("默认离线测试禁止构造真实Chroma客户端，请显式注入替身")


def _guard_runtime_files(event: str, arguments: tuple[object, ...]) -> None:
    """审计Python层路径操作；原生存储入口另由构造守卫阻断。"""
    if not _OFFLINE_ACTIVE or event not in _PATH_EVENTS | _TWO_PATH_EVENTS:
        return
    paths = arguments[:2] if event in _TWO_PATH_EVENTS else arguments[:1]
    for value in paths:
        if not isinstance(value, (str, bytes, os.PathLike)):
            continue
        path = Path(os.path.abspath(os.fsdecode(value)))
        data_root = BACKEND_ROOT / "data"
        if path == data_root or data_root in path.parents:
            raise RuntimeError("默认离线测试禁止访问运行数据目录")
        if path.name == ".env" and REPOSITORY_ROOT in path.parents:
            raise RuntimeError("默认离线测试禁止读取项目.env")


def _guard_chroma_construction() -> None:
    from chromadb.api.shared_system_client import SharedSystemClient
    from langchain_chroma import Chroma

    for owner in (SharedSystemClient, Chroma):
        _ORIGINAL_CHROMA[(owner, "__init__")] = owner.__init__
        owner.__init__ = _blocked_chroma


def pytest_configure(config: pytest.Config) -> None:
    """在collection前保护网络、运行目录、私有环境与真实存储构造。"""
    global _AUDIT_INSTALLED, _OFFLINE_ACTIVE
    if _explicit_online_suite(config):
        return

    for name, value in OFFLINE_ENV.items():
        _ORIGINAL_ENV[name] = os.environ.get(name)
        os.environ[name] = value

    targets = [
        (socket, "create_connection"),
        (socket.socket, "connect"),
        (socket.socket, "connect_ex"),
        (socket.socket, "sendto"),
    ]
    if hasattr(socket.socket, "sendmsg"):
        targets.append((socket.socket, "sendmsg"))
    for owner, name in targets:
        _ORIGINAL_SOCKET[(owner, name)] = getattr(owner, name)
        setattr(owner, name, _blocked_network)

    _OFFLINE_ACTIVE = True
    if not _AUDIT_INSTALLED:
        sys.addaudithook(_guard_runtime_files)
        _AUDIT_INSTALLED = True
    _guard_chroma_construction()


def pytest_unconfigure(config: pytest.Config) -> None:
    del config
    global _OFFLINE_ACTIVE
    # audit hook无法卸载；退出pytest后先失活，再恢复本次替换的构造函数。
    _OFFLINE_ACTIVE = False
    for (owner, name), value in _ORIGINAL_CHROMA.items():
        setattr(owner, name, value)
    _ORIGINAL_CHROMA.clear()

    for (owner, name), value in _ORIGINAL_SOCKET.items():
        setattr(owner, name, value)
    _ORIGINAL_SOCKET.clear()

    for name, value in _ORIGINAL_ENV.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    _ORIGINAL_ENV.clear()


@pytest.fixture
def backend_root() -> Path:
    """返回后端根目录，只用于读取源码和测试资产。"""
    return BACKEND_ROOT


@pytest.fixture
def repository_root() -> Path:
    """返回仓库根目录，只用于只读基线检查。"""
    return REPOSITORY_ROOT


@pytest.fixture
def isolated_chroma_dir(tmp_path: Path) -> tuple[Path, Path]:
    """创建带哨兵文件的一次性 Chroma 目录。"""
    persist_dir = tmp_path / "chromadb"
    persist_dir.mkdir()
    sentinel = persist_dir / "must-survive.txt"
    sentinel.write_text("disposable-test-data", encoding="utf-8")
    return persist_dir, sentinel
