"""任务事实的离线约束、迁移往返和 MySQL DDL 契约。"""

from __future__ import annotations

from collections.abc import Iterator
from io import StringIO
from pathlib import Path
import re
from typing import Any
from uuid import UUID

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
import pytest
from sqlalchemy import (
    CheckConstraint,
    Table,
    UniqueConstraint,
    create_engine,
    event,
    inspect,
    select,
    text,
)
from sqlalchemy.dialects import mysql
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateIndex, CreateTable

from app.db.schema_gate import SchemaCompatibilityError, validate_schema_compatibility
from app.indexing.models import ContentBlob
from app.models import Base
from app.tasking.contracts import ATTEMPT_STATUSES, TASK_KINDS, TASK_STATUSES
from app.tasking.models import BackgroundTask, TaskAttempt

TASK_ID = str(UUID(int=1))
OTHER_TASK_ID = str(UUID(int=2))
BLOB_ID = "a" * 64
TASK_TABLE = BackgroundTask.__table__
ATTEMPT_TABLE = TaskAttempt.__table__
TASK_TABLES = (TASK_TABLE, ATTEMPT_TABLE)


def _sqlite_engine(url: str = "sqlite:///:memory:") -> Engine:
    engine = create_engine(url)

    @event.listens_for(engine, "connect")
    def configure(connection: Any, _record: Any) -> None:
        connection.execute("PRAGMA foreign_keys=ON")
        # 旧索引 ORM 仍声明 MySQL 排序规则；SQLite 中只模拟二进制比较。
        for name in ("ascii_bin", "utf8mb4_bin"):
            connection.create_collation(
                name, lambda left, right: (left > right) - (left < right)
            )

    return engine


def _config(backend_root: Path, connection: Connection) -> Config:
    config = Config(str(backend_root / "alembic.ini"))
    config.attributes["connection"] = connection
    return config


@pytest.fixture(params=("orm", "migration"))
def connection(
    request: pytest.FixtureRequest, backend_root: Path
) -> Iterator[Connection]:
    engine = _sqlite_engine()
    try:
        with engine.begin() as connection:
            assert connection.scalar(text("PRAGMA foreign_keys")) == 1
            if request.param == "orm":
                Base.metadata.create_all(connection)
            else:
                command.upgrade(_config(backend_root, connection), "head")
                validate_schema_compatibility(connection)
            yield connection
    finally:
        engine.dispose()


def _task_values(**changes: Any) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "user_id": "schema-user",
        "kind": "note.index",
        "idempotency_key": "schema-key",
        "input_fingerprint": "b" * 64,
        "input_ref": "snapshot://schema-test",
        **changes,
    }


def _attempt_values(**changes: Any) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "attempt_no": 1,
        "lease_owner": "schema-worker",
        "lease_token": 1,
        **changes,
    }


def _insert_blob(connection: Connection) -> None:
    connection.execute(
        ContentBlob.__table__.insert().values(
            blob_id=BLOB_ID,
            byte_size=1,
            storage_uri="blob://schema-test",
        )
    )


def _normalize_sql(value: object) -> str:
    return " ".join(str(value).split())


@pytest.mark.parametrize("table", TASK_TABLES, ids=lambda table: table.name)
def test_task_columns_match_orm(connection: Connection, table: Table) -> None:
    columns = inspect(connection).get_columns(table.name)
    assert [column["name"] for column in columns] == list(table.columns.keys())
    for actual, expected in zip(columns, table.columns, strict=True):
        assert str(actual["type"]) == expected.type.compile(dialect=connection.dialect)
        assert actual["nullable"] is expected.nullable
        default = expected.server_default
        expected_default = (
            None
            if default is None
            else str(default.arg.compile(dialect=connection.dialect))
        )
        assert actual["default"] == expected_default


@pytest.mark.parametrize("table", TASK_TABLES, ids=lambda table: table.name)
def test_task_constraints_and_indexes_match_orm(
    connection: Connection, table: Table
) -> None:
    inspector = inspect(connection)
    assert inspector.get_pk_constraint(table.name)["constrained_columns"] == list(
        table.primary_key.columns.keys()
    )
    assert {
        item["name"]: _normalize_sql(item["sqltext"])
        for item in inspector.get_check_constraints(table.name)
    } == {
        constraint.name: _normalize_sql(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert {
        (item["name"], tuple(item["column_names"]))
        for item in inspector.get_unique_constraints(table.name)
    } == {
        (constraint.name, tuple(constraint.columns.keys()))
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assert {
        (item["name"], tuple(item["column_names"]), bool(item["unique"]))
        for item in inspector.get_indexes(table.name)
    } == {
        (index.name, tuple(index.columns.keys()), index.unique)
        for index in table.indexes
    }
    assert {
        (
            tuple(item["constrained_columns"]),
            item["referred_table"],
            tuple(item["referred_columns"]),
            item["options"].get("ondelete"),
        )
        for item in inspector.get_foreign_keys(table.name)
    } == {
        (
            tuple(fk.parent.name for fk in constraint.elements),
            constraint.elements[0].column.table.name,
            tuple(fk.column.name for fk in constraint.elements),
            constraint.ondelete,
        )
        for constraint in table.foreign_key_constraints
    }


def test_task_and_attempt_defaults_are_database_owned(connection: Connection) -> None:
    connection.execute(TASK_TABLE.insert().values(**_task_values()))
    connection.execute(ATTEMPT_TABLE.insert().values(**_attempt_values()))
    task = connection.execute(select(TASK_TABLE)).mappings().one()
    attempt = connection.execute(select(ATTEMPT_TABLE)).mappings().one()

    assert {
        name: task[name]
        for name in (
            "status",
            "progress",
            "attempt_count",
            "max_attempts",
            "input_schema_version",
            "lease_token",
        )
    } == {
        "status": "pending",
        "progress": 0,
        "attempt_count": 0,
        "max_attempts": 3,
        "input_schema_version": 1,
        "lease_token": 0,
    }
    assert task["created_at"] is not None and task["updated_at"] is not None
    assert attempt["status"] == "running" and attempt["started_at"] is not None


@pytest.mark.parametrize(
    ("column", "value", "constraint"),
    [
        ("kind", "unsupported", "kind"),
        ("kind", "NOTE.INDEX", "kind"),
        ("status", "running", "status"),
        ("status", "unknown", "status"),
        ("progress", -1, "progress"),
        ("progress", 101, "progress"),
        ("attempt_count", -1, "attempt_count"),
        ("max_attempts", 0, "max_attempts"),
        ("target_generation", 0, "generation"),
        ("input_schema_version", 0, "input_schema_version"),
        ("input_ref", "", "input_ref"),
        ("lease_token", -1, "lease_token"),
        ("result_version", 0, "result_version"),
    ],
)
def test_task_checks_reject_invalid_values(
    connection: Connection,
    column: str,
    value: Any,
    constraint: str,
) -> None:
    with pytest.raises(IntegrityError, match=f"ck_background_tasks_{constraint}"):
        with connection.begin_nested():
            connection.execute(
                TASK_TABLE.insert().values(**_task_values(**{column: value}))
            )


@pytest.mark.parametrize(
    ("column", "value", "constraint"),
    [
        ("status", "pending", "status"),
        ("status", "unknown", "status"),
        ("attempt_no", 0, "attempt_no"),
        ("attempt_no", -1, "attempt_no"),
        ("lease_token", 0, "lease_token"),
        ("lease_token", -1, "lease_token"),
    ],
)
def test_attempt_checks_reject_invalid_values(
    connection: Connection,
    column: str,
    value: Any,
    constraint: str,
) -> None:
    connection.execute(TASK_TABLE.insert().values(**_task_values()))
    with pytest.raises(IntegrityError, match=f"ck_task_attempts_{constraint}"):
        with connection.begin_nested():
            connection.execute(
                ATTEMPT_TABLE.insert().values(**_attempt_values(**{column: value}))
            )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        *(("kind", kind) for kind in sorted(TASK_KINDS)),
        *(("status", status) for status in sorted(TASK_STATUSES)),
    ],
)
def test_task_checks_accept_existing_contract_enums(
    connection: Connection,
    column: str,
    value: str,
) -> None:
    connection.execute(TASK_TABLE.insert().values(**_task_values(**{column: value})))
    assert connection.scalar(select(TASK_TABLE.c[column])) == value


@pytest.mark.parametrize("status", sorted(ATTEMPT_STATUSES))
def test_attempt_checks_accept_existing_contract_enums(
    connection: Connection, status: str
) -> None:
    connection.execute(TASK_TABLE.insert().values(**_task_values()))
    connection.execute(ATTEMPT_TABLE.insert().values(**_attempt_values(status=status)))
    assert connection.scalar(select(ATTEMPT_TABLE.c.status)) == status


def test_numeric_boundaries_preserve_tokens_above_32_bits(
    connection: Connection,
) -> None:
    token = 2**40 + 1
    connection.execute(
        TASK_TABLE.insert().values(
            **_task_values(
                progress=100,
                attempt_count=1,
                max_attempts=1,
                target_generation=1,
                input_schema_version=1,
                result_version=1,
                lease_token=token,
            )
        )
    )
    connection.execute(
        ATTEMPT_TABLE.insert().values(**_attempt_values(lease_token=token))
    )
    assert connection.scalar(select(TASK_TABLE.c.lease_token)) == token
    assert connection.scalar(select(ATTEMPT_TABLE.c.lease_token)) == token
    connection.execute(
        TASK_TABLE.update().values(
            progress=0,
            attempt_count=0,
            target_generation=None,
            result_version=None,
            lease_token=0,
        )
    )


def test_idempotency_is_unique_per_user_even_across_kinds(
    connection: Connection,
) -> None:
    connection.execute(TASK_TABLE.insert().values(**_task_values()))
    with pytest.raises(IntegrityError, match="UNIQUE constraint"):
        with connection.begin_nested():
            connection.execute(
                TASK_TABLE.insert().values(
                    **_task_values(task_id=OTHER_TASK_ID, kind="note.delete")
                )
            )
    connection.execute(
        TASK_TABLE.insert().values(
            **_task_values(task_id=OTHER_TASK_ID, user_id="another-user")
        )
    )
    connection.execute(
        TASK_TABLE.insert().values(
            **_task_values(task_id=str(UUID(int=3)), idempotency_key="another-key")
        )
    )
    assert len(connection.execute(select(TASK_TABLE.c.task_id)).all()) == 3


def test_attempt_primary_key_is_task_and_attempt_number(connection: Connection) -> None:
    connection.execute(
        TASK_TABLE.insert(),
        [
            _task_values(),
            _task_values(task_id=OTHER_TASK_ID, idempotency_key="another-key"),
        ],
    )
    connection.execute(
        ATTEMPT_TABLE.insert(),
        [
            _attempt_values(),
            _attempt_values(attempt_no=2, lease_token=2),
            _attempt_values(task_id=OTHER_TASK_ID),
        ],
    )
    with pytest.raises(IntegrityError, match="UNIQUE constraint"):
        with connection.begin_nested():
            connection.execute(
                ATTEMPT_TABLE.insert().values(**_attempt_values(lease_token=3))
            )


def test_blob_foreign_key_rejects_orphans_and_restricts_delete(
    connection: Connection,
) -> None:
    with pytest.raises(IntegrityError, match="FOREIGN KEY constraint"):
        with connection.begin_nested():
            connection.execute(
                TASK_TABLE.insert().values(**_task_values(input_blob_id=BLOB_ID))
            )
    _insert_blob(connection)
    connection.execute(
        TASK_TABLE.insert().values(**_task_values(input_blob_id=BLOB_ID, input_ref=""))
    )
    with pytest.raises(IntegrityError, match="FOREIGN KEY constraint"):
        with connection.begin_nested():
            connection.execute(ContentBlob.__table__.delete())


def test_retry_foreign_key_rejects_orphans_and_restricts_delete(
    connection: Connection,
) -> None:
    with pytest.raises(IntegrityError, match="FOREIGN KEY constraint"):
        with connection.begin_nested():
            connection.execute(
                TASK_TABLE.insert().values(
                    **_task_values(retry_of_task_id=OTHER_TASK_ID)
                )
            )
    connection.execute(TASK_TABLE.insert().values(**_task_values()))
    connection.execute(
        TASK_TABLE.insert().values(
            **_task_values(
                task_id=OTHER_TASK_ID,
                idempotency_key="retry-key",
                retry_of_task_id=TASK_ID,
            )
        )
    )
    with pytest.raises(IntegrityError, match="FOREIGN KEY constraint"):
        with connection.begin_nested():
            connection.execute(
                TASK_TABLE.delete().where(TASK_TABLE.c.task_id == TASK_ID)
            )


def test_attempt_foreign_key_rejects_orphans_and_restricts_delete(
    connection: Connection,
) -> None:
    with pytest.raises(IntegrityError, match="FOREIGN KEY constraint"):
        with connection.begin_nested():
            connection.execute(ATTEMPT_TABLE.insert().values(**_attempt_values()))
    connection.execute(TASK_TABLE.insert().values(**_task_values()))
    connection.execute(ATTEMPT_TABLE.insert().values(**_attempt_values()))
    with pytest.raises(IntegrityError, match="FOREIGN KEY constraint"):
        with connection.begin_nested():
            connection.execute(TASK_TABLE.delete())


def test_task_migration_round_trip_preserves_existing_blobs(
    backend_root: Path, tmp_path: Path
) -> None:
    engine = _sqlite_engine(f"sqlite:///{tmp_path / 'tasking_test.sqlite'}")
    try:
        with engine.begin() as connection:
            config = _config(backend_root, connection)
            command.upgrade(config, "0002_index_contract")
            _insert_blob(connection)
            original_blob = connection.execute(select(ContentBlob.__table__)).one()
        # 各阶段分别提交；后续连接验证的是已持久化的结构、版本和合成事实。
        with engine.begin() as connection:
            config = _config(backend_root, connection)
            command.upgrade(config, "head")
            connection.execute(
                TASK_TABLE.insert().values(**_task_values(input_blob_id=BLOB_ID))
            )
            connection.execute(
                TASK_TABLE.insert().values(
                    **_task_values(
                        task_id=OTHER_TASK_ID,
                        idempotency_key="retry-key",
                        retry_of_task_id=TASK_ID,
                    )
                )
            )
            connection.execute(ATTEMPT_TABLE.insert().values(**_attempt_values()))
            validate_schema_compatibility(connection)
        with engine.begin() as connection:
            config = _config(backend_root, connection)
            command.downgrade(config, "0002_index_contract")
        with engine.begin() as connection:
            config = _config(backend_root, connection)
            assert not {table.name for table in TASK_TABLES} & set(
                inspect(connection).get_table_names()
            )
            assert (
                connection.scalar(text("SELECT version_num FROM alembic_version"))
                == "0002_index_contract"
            )
            assert (
                connection.execute(select(ContentBlob.__table__)).one() == original_blob
            )
            with pytest.raises(SchemaCompatibilityError, match="revision 不兼容"):
                validate_schema_compatibility(connection)
            command.upgrade(config, "head")
        with engine.connect() as connection:
            validate_schema_compatibility(connection)
            assert connection.execute(select(TASK_TABLE)).all() == []
            assert connection.execute(select(ATTEMPT_TABLE)).all() == []
            assert (
                connection.execute(select(ContentBlob.__table__)).one() == original_blob
            )
            assert connection.scalar(text("PRAGMA foreign_keys")) == 1
            assert connection.execute(text("PRAGMA foreign_key_check")).all() == []
    finally:
        engine.dispose()


def _mysql_migration_ddl(backend_root: Path, *, downgrade: bool = False) -> str:
    """直接运行 revision 的离线上下文，完全绕开 URL、dotenv 和连接创建。"""
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="mysql", opts={"as_sql": True, "output_buffer": output}
    )
    scripts = ScriptDirectory.from_config(Config(str(backend_root / "alembic.ini")))
    revisions = list(scripts.walk_revisions())
    if not downgrade:
        revisions.reverse()
    with Operations.context(context):
        for revision in revisions:
            migrate = (
                revision.module.downgrade if downgrade else revision.module.upgrade
            )
            migrate()
    return output.getvalue()


def _ddl_parts(statement: str) -> tuple[list[str], set[str]]:
    """只忽略约束声明及表选项的顺序，不掩盖类型、默认值或约束内容差异。"""
    body, _, options = statement.partition("(")[2].rpartition(")")
    return sorted(
        _normalize_sql(line.rstrip(",")) for line in body.splitlines() if line.strip()
    ), set(options.split())


def test_mysql_offline_migrations_match_task_orm_and_keep_unsigned_fencing(
    backend_root: Path,
) -> None:
    ddl = _mysql_migration_ddl(backend_root)
    dialect = mysql.dialect()
    assert ddl.count("CREATE TABLE ") == 11
    assert ddl.count("lease_token BIGINT UNSIGNED NOT NULL") == 2
    assert (
        "FOREIGN KEY(input_blob_id) REFERENCES content_blobs (blob_id) ON DELETE RESTRICT"
        in ddl
    )
    for table in TASK_TABLES:
        migration_ddl = next(
            statement.strip()
            for statement in ddl.split(";")
            if statement.strip().startswith(f"CREATE TABLE {table.name} (")
        )
        orm_ddl = str(CreateTable(table).compile(dialect=dialect))
        assert _ddl_parts(migration_ddl) == _ddl_parts(orm_ddl)
    assert set(
        re.findall(
            r"CREATE INDEX [^;]+ ON (?:background_tasks|task_attempts) \([^;]+\)", ddl
        )
    ) == {
        str(CreateIndex(index).compile(dialect=dialect))
        for table in TASK_TABLES
        for index in table.indexes
    }


def test_mysql_offline_downgrade_drops_children_before_parents(
    backend_root: Path,
) -> None:
    ddl = _mysql_migration_ddl(backend_root, downgrade=True)

    assert ddl.count("DROP TABLE ") == 11
    assert ddl.index("DROP TABLE task_attempts") < ddl.index(
        "DROP TABLE background_tasks"
    )
    assert ddl.index("DROP TABLE background_tasks") < ddl.index(
        "DROP TABLE content_blobs"
    )
    assert "UPDATE background_tasks" not in ddl
    assert "PRAGMA" not in ddl
