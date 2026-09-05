from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import CheckConstraint, UniqueConstraint, inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from app.indexing.models import (
    ContentBlob,
    DocumentRevision,
    IndexChunk,
    IndexedDocument,
    IndexVersion,
)
from app.models import Base
from app.tasking.models import BackgroundTask, TaskAttempt

EXPECTED_SCHEMA_REVISION = "0003_task_lifecycle"
VERSION_TABLE = "alembic_version"
LEGACY_TABLE_NAMES = frozenset(
    {"chat_messages", "chat_sessions", "notes", "review_records"}
)
TASKING_MODELS = (BackgroundTask, TaskAttempt)
INDEXING_MODELS = (
    ContentBlob,
    DocumentRevision,
    IndexChunk,
    IndexedDocument,
    IndexVersion,
)


class SchemaCompatibilityError(RuntimeError):
    """数据库结构或迁移版本与当前应用不兼容。"""


def _foreign_key_signatures(table_name: str) -> set[tuple[object, ...]]:
    table = Base.metadata.tables[table_name]
    return {
        (
            tuple(element.parent.name for element in constraint.elements),
            constraint.elements[0].column.table.name,
            tuple(element.column.name for element in constraint.elements),
            (constraint.ondelete or "").upper(),
        )
        for constraint in table.foreign_key_constraints
    }


def _unique_constraint_signatures(table_name: str) -> set[tuple[object, ...]]:
    table = Base.metadata.tables[table_name]
    return {
        (constraint.name, tuple(column.name for column in constraint.columns))
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }


def _check_constraint_names(table_name: str) -> set[str]:
    table = Base.metadata.tables[table_name]
    return {
        constraint.name
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint) and constraint.name is not None
    }


def _validate_table_contract(
    connection: Connection,
    expected_table_names: set[str] | frozenset[str],
) -> None:
    inspector = inspect(connection)
    expected_tables = set(expected_table_names)
    actual_tables = set(inspector.get_table_names())
    application_tables = actual_tables - {VERSION_TABLE}

    missing_tables = expected_tables - application_tables
    unexpected_tables = application_tables - expected_tables
    if missing_tables or unexpected_tables:
        raise SchemaCompatibilityError(
            "数据库表集合不兼容："
            f"缺少={sorted(missing_tables)}，额外={sorted(unexpected_tables)}"
        )

    for table_name in sorted(expected_tables):
        table = Base.metadata.tables[table_name]
        expected_columns = {column.name: column for column in table.columns}
        actual_columns = {
            column["name"]: column for column in inspector.get_columns(table_name)
        }
        if set(actual_columns) != set(expected_columns):
            raise SchemaCompatibilityError(
                f"表 {table_name} 的列集合不兼容："
                f"期望={sorted(expected_columns)}，实际={sorted(actual_columns)}"
            )

        invalid_nullability = sorted(
            name
            for name, column in expected_columns.items()
            if bool(actual_columns[name]["nullable"]) != bool(column.nullable)
        )
        if invalid_nullability:
            raise SchemaCompatibilityError(
                f"表 {table_name} 的可空约束不兼容：{invalid_nullability}"
            )

        expected_pk = tuple(column.name for column in table.primary_key.columns)
        actual_pk = tuple(
            inspector.get_pk_constraint(table_name).get("constrained_columns") or ()
        )
        if actual_pk != expected_pk:
            raise SchemaCompatibilityError(
                f"表 {table_name} 的主键不兼容：期望={expected_pk}，实际={actual_pk}"
            )

        actual_foreign_keys = {
            (
                tuple(item.get("constrained_columns") or ()),
                item.get("referred_table"),
                tuple(item.get("referred_columns") or ()),
                str((item.get("options") or {}).get("ondelete") or "").upper(),
            )
            for item in inspector.get_foreign_keys(table_name)
        }
        expected_foreign_keys = _foreign_key_signatures(table_name)
        if actual_foreign_keys != expected_foreign_keys:
            raise SchemaCompatibilityError(
                f"表 {table_name} 的外键不兼容："
                f"期望={sorted(expected_foreign_keys)!r}，"
                f"实际={sorted(actual_foreign_keys)!r}"
            )

        expected_indexes = {
            (index.name, tuple(column.name for column in index.columns), index.unique)
            for index in table.indexes
        }
        actual_indexes = {
            (
                item.get("name"),
                tuple(item.get("column_names") or ()),
                bool(item.get("unique")),
            )
            for item in inspector.get_indexes(table_name)
        }
        missing_indexes = expected_indexes - actual_indexes
        if missing_indexes:
            raise SchemaCompatibilityError(
                f"表 {table_name} 缺少索引：{sorted(missing_indexes)!r}"
            )

        expected_unique_constraints = _unique_constraint_signatures(table_name)
        actual_unique_constraints = {
            (
                item.get("name"),
                tuple(item.get("column_names") or ()),
            )
            for item in inspector.get_unique_constraints(table_name)
        }
        missing_unique_constraints = (
            expected_unique_constraints - actual_unique_constraints
        )
        if missing_unique_constraints:
            raise SchemaCompatibilityError(
                f"表 {table_name} 缺少唯一约束："
                f"{sorted(missing_unique_constraints)!r}"
            )

        expected_check_names = _check_constraint_names(table_name)
        actual_check_names = {
            item.get("name")
            for item in inspector.get_check_constraints(table_name)
            if item.get("name") is not None
        }
        missing_check_names = expected_check_names - actual_check_names
        if missing_check_names:
            raise SchemaCompatibilityError(
                f"表 {table_name} 缺少 CHECK 约束：{sorted(missing_check_names)!r}"
            )


def _validate_database_capabilities(connection: Connection) -> None:
    if connection.dialect.name != "mysql":
        return
    version = connection.dialect.server_version_info
    if version is None or tuple(version[:3]) < (8, 0, 16):
        raise SchemaCompatibilityError(
            "MySQL 服务端必须 >= 8.0.16 以执行 M2 CHECK 约束"
        )


def validate_schema_compatibility(
    connection: Connection,
    *,
    expected_revision: str = EXPECTED_SCHEMA_REVISION,
    allow_unversioned_legacy: bool = False,
) -> None:
    """只读验证 legacy 预检或当前 head 契约。"""
    table_names = set(inspect(connection).get_table_names())
    if VERSION_TABLE not in table_names:
        if allow_unversioned_legacy:
            _validate_table_contract(connection, LEGACY_TABLE_NAMES)
            return
        raise SchemaCompatibilityError(
            "数据库缺少 alembic_version；既有库必须先通过 legacy 预检再显式 stamp"
        )

    revisions = list(
        connection.execute(text("SELECT version_num FROM alembic_version")).scalars()
    )
    if revisions != [expected_revision]:
        raise SchemaCompatibilityError(
            f"数据库 revision 不兼容：期望={[expected_revision]!r}，实际={revisions!r}"
        )
    _validate_database_capabilities(connection)
    _validate_table_contract(connection, set(Base.metadata.tables))


async def validate_schema_on_engine(
    engine: AsyncEngine,
    *,
    validator: Callable[[Connection], None] = validate_schema_compatibility,
) -> None:
    """在注入的异步引擎上运行只读门禁。"""
    async with engine.connect() as connection:
        await connection.run_sync(validator)
