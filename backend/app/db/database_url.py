from __future__ import annotations

import os
from collections.abc import Mapping

from sqlalchemy.engine import URL


class DatabaseConfigurationError(RuntimeError):
    """数据库连接配置无效。"""


def build_database_url(
    environment: Mapping[str, str] | None = None,
    *,
    drivername: str = "mysql+aiomysql",
) -> URL:
    """从环境映射构造不会误解析特殊字符的 SQLAlchemy URL。"""
    values = os.environ if environment is None else environment
    raw_port = values.get("MYSQL_PORT", "3306")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise DatabaseConfigurationError("MYSQL_PORT 必须是整数") from exc
    if not 1 <= port <= 65535:
        raise DatabaseConfigurationError("MYSQL_PORT 必须位于 1 到 65535")

    database = values.get("MYSQL_DATABASE", "chat_history").strip()
    if not database:
        raise DatabaseConfigurationError("MYSQL_DATABASE 不能为空")

    return URL.create(
        drivername=drivername,
        username=values.get("MYSQL_USER", "root"),
        password=values.get("MYSQL_PASSWORD", ""),
        host=values.get("MYSQL_HOST", "localhost"),
        port=port,
        database=database,
        query={"charset": "utf8mb4"},
    )
