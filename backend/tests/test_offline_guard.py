import socket

import pytest


def test_default_suite_blocks_common_socket_connections() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        with pytest.raises(RuntimeError, match="默认离线测试禁止建立网络连接"):
            client.connect_ex(("127.0.0.1", 9))
