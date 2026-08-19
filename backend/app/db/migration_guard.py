"""Alembic CLI 的显式目标、legacy preflight 与破坏性操作门禁。"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from sqlalchemy import inspect
from sqlalchemy.engine import Connection, URL

from app.db.schema_gate import (
    LEGACY_TABLE_NAMES,
    VERSION_TABLE,
    validate_schema_compatibility,
)

NON_TEST_ACKNOWLEDGEMENT = "I_UNDERSTAND_THIS_MODIFIES_NON_TEST_DATA"
DOWNGRADE_ACKNOWLEDGEMENT = "I_UNDERSTAND_DOWNGRADE_CAN_DELETE_DATA"


class MigrationSafetyError(RuntimeError):
    """迁移目标或命令未满足显式安全前提。"""


def validate_migration_target(
    url: URL,
    environment: Mapping[str, str] | None = None,
) -> None:
    """要求 CLI 目标与环境显式一致，非测试库需要二次确认。"""
    values = os.environ if environment is None else environment
    database = url.database
    configured_database = values.get("MYSQL_DATABASE", "").strip()
    acknowledged_target = values.get("RAG_MIGRATION_TARGET", "").strip()

    if not database or configured_database != database:
        raise MigrationSafetyError("Alembic 必须显式设置 MYSQL_DATABASE")
    if acknowledged_target != database:
        raise MigrationSafetyError(
            "RAG_MIGRATION_TARGET 必须与 MYSQL_DATABASE 完全一致"
        )
    if not database.endswith("_test") and values.get(
        "RAG_ALLOW_NON_TEST_MIGRATIONS"
    ) != NON_TEST_ACKNOWLEDGEMENT:
        raise MigrationSafetyError("非 _test 数据库迁移缺少显式二次确认")


def validate_migration_command(
    connection: Connection,
    command_name: str,
    revisions: Sequence[str] = (),
    environment: Mapping[str, str] | None = None,
) -> None:
    """在 Alembic 创建版本表或执行 DDL 前检查命令前置条件。"""
    values = os.environ if environment is None else environment
    tables = set(inspect(connection).get_table_names())

    if command_name == "stamp":
        if VERSION_TABLE in tables:
            raise MigrationSafetyError("已版本化数据库禁止重新 stamp legacy revision")
        if tuple(revisions) != ("0001_legacy",):
            raise MigrationSafetyError("只允许在 legacy preflight 后 stamp 0001_legacy")
        validate_schema_compatibility(
            connection,
            allow_unversioned_legacy=True,
        )
        return

    if command_name == "upgrade" and VERSION_TABLE not in tables:
        application_tables = tables - {VERSION_TABLE}
        if application_tables:
            if application_tables == set(LEGACY_TABLE_NAMES):
                validate_schema_compatibility(
                    connection,
                    allow_unversioned_legacy=True,
                )
            raise MigrationSafetyError(
                "无版本既有库禁止直接 upgrade；必须先完成 legacy preflight 与 stamp"
            )

    if command_name == "downgrade" and values.get(
        "RAG_ALLOW_DESTRUCTIVE_DOWNGRADE"
    ) != DOWNGRADE_ACKNOWLEDGEMENT:
        raise MigrationSafetyError("downgrade 缺少独立破坏性确认")
