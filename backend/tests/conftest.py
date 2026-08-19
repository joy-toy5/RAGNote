from __future__ import annotations

import os
import socket
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
}
_ORIGINAL_ENV: dict[str, str | None] = {}
_ORIGINAL_SOCKET: dict[tuple[object, str], Any] = {}


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


def pytest_configure(config: pytest.Config) -> None:
    """在测试模块收集前阻断常见 Python socket 出口。"""
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


def pytest_unconfigure(config: pytest.Config) -> None:
    del config
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
