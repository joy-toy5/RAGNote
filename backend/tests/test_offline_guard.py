import socket
import sys

import pytest


def test_default_suite_blocks_common_socket_connections() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        with pytest.raises(RuntimeError, match="默认离线测试禁止建立网络连接"):
            client.connect_ex(("127.0.0.1", 9))


@pytest.mark.parametrize("kind", ["langchain", "native"])
def test_default_suite_blocks_real_chroma_construction(kind, tmp_path):
    from chromadb.api.shared_system_client import SharedSystemClient
    from langchain_chroma import Chroma

    owner = Chroma if kind == "langchain" else SharedSystemClient
    # 先确认守卫存在，红灯/变异运行也绝不调用未拦截的真实构造函数。
    assert owner.__init__.__name__ == "_blocked_chroma"
    with pytest.raises(RuntimeError, match="默认离线测试禁止构造真实Chroma"):
        owner.__init__(object())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("event", ["open", "os.mkdir", "os.remove", "os.rename", "sqlite3.connect"])
def test_default_suite_blocks_runtime_data_audit_events(event, backend_root, tmp_path):
    path = str(backend_root / "data" / "must-not-touch")
    args = {
        "open": (path, "r", 0),
        "sqlite3.connect": (path,),
        "os.mkdir": (path, 0o700, -1),
        "os.remove": (path, -1),
        "os.rename": (str(tmp_path / "safe-source"), path, -1, -1),
    }[event]
    # 合成audit事件只验证拦截器，不打开/创建/删除真实业务路径。
    with pytest.raises(RuntimeError, match="默认离线测试禁止访问运行数据"):
        sys.audit(event, *args)


def test_default_suite_blocks_private_dotenv_audit_event(repository_root):
    with pytest.raises(RuntimeError, match="默认离线测试禁止读取项目.env"):
        sys.audit("open", str(repository_root / ".env"), "r", 0)


def test_default_suite_allows_disposable_files(tmp_path):
    path = tmp_path / "synthetic.txt"
    path.write_text("合成离线内容", encoding="utf-8")
    assert path.read_text(encoding="utf-8") == "合成离线内容"
