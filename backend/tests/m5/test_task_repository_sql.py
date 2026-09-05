"""SQL 与错误分类契约；不以编译或模拟错误替代 MySQL 隔离并发测试。"""

import asyncio
import sqlite3
from unittest.mock import AsyncMock

import pytest
from pymysql.err import IntegrityError as MysqlIntegrityError
from sqlalchemy.dialects import mysql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.tasking import repository
from app.tasking.errors import TaskNotFound


@pytest.mark.parametrize(
    "code,message,expected",
    [
        (
            1062,
            "Duplicate entry 'x' for key 'uq_background_tasks_user_idempotency'",
            True,
        ),
        (
            1062,
            "Duplicate entry 'x' for key 'background_tasks.uq_background_tasks_user_idempotency'",
            True,
        ),
        (
            1062,
            "Duplicate entry 'x' for key `uq_background_tasks_user_idempotency`",
            True,
        ),
        (1062, "Duplicate entry 'x' for key 'PRIMARY'", False),
        (
            1062,
            "Duplicate entry 'uq_background_tasks_user_idempotency' for key 'PRIMARY'",
            False,
        ),
        (
            1062,
            "Duplicate entry 'x' for key 'uq_background_tasks_user_idempotency_suffix'",
            False,
        ),
        (
            1062,
            "Duplicate entry 'x for key 'uq_background_tasks_user_idempotency'' for key 'PRIMARY'",
            False,
        ),
        (
            1452,
            "Cannot add or update a child row: a foreign key constraint fails",
            False,
        ),
        (3819, "Check constraint 'ck_background_tasks_status' is violated", False),
        (1213, "Deadlock found when trying to get lock", False),
        (1062, "Duplicate entry without a reported constraint", False),
        (1062, 123, False),
    ],
)
def test_only_mysql_target_unique_key_is_recoverable(code, message, expected):
    error = IntegrityError("INSERT", {}, MysqlIntegrityError(code, message))
    assert repository._is_idempotency_conflict(error, "mysql") is expected
    assert repository._is_idempotency_conflict(error, "postgresql") is False


@pytest.mark.parametrize(
    "code,message,expected",
    [
        (
            sqlite3.SQLITE_CONSTRAINT_UNIQUE,
            "UNIQUE constraint failed: background_tasks.user_id, background_tasks.idempotency_key",
            True,
        ),
        (
            sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY,
            "UNIQUE constraint failed: background_tasks.task_id",
            False,
        ),
        (
            sqlite3.SQLITE_CONSTRAINT_UNIQUE,
            "UNIQUE constraint failed: content_blobs.blob_id",
            False,
        ),
        (sqlite3.SQLITE_CONSTRAINT_FOREIGNKEY, "FOREIGN KEY constraint failed", False),
        (
            sqlite3.SQLITE_CONSTRAINT_CHECK,
            "CHECK constraint failed: ck_background_tasks_status",
            False,
        ),
    ],
)
def test_sqlite_recovery_also_requires_exact_unique_constraint(code, message, expected):
    original = sqlite3.IntegrityError(message)
    original.sqlite_errorcode = code
    error = IntegrityError("INSERT", {}, original)
    assert repository._is_idempotency_conflict(error, "sqlite") is expected


def test_prelookup_avoids_gap_lock_and_competition_lookup_uses_current_read():
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = None
    asyncio.run(repository._get_by_idempotency(session, "user-a", "key"))
    initial = session.scalar.call_args.args[0]
    initial_sql = str(initial.compile(dialect=mysql.dialect()))
    assert "FOR UPDATE" not in initial_sql
    assert "LOCK IN SHARE MODE" not in initial_sql
    asyncio.run(repository._get_by_idempotency(session, "user-a", "key", lock=True))
    current = session.scalar.call_args.args[0]
    compiled = current.compile(dialect=mysql.dialect())
    assert str(compiled).endswith("LOCK IN SHARE MODE")
    assert compiled.params == {"user_id_1": "user-a", "idempotency_key_1": "key"}
    assert current.get_execution_options()["populate_existing"] is True


@pytest.mark.parametrize("operation", ["cancel", "retry"])
def test_mutation_lock_is_user_scoped_and_refreshes_identity_map(operation):
    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = None
    with pytest.raises(TaskNotFound):
        if operation == "cancel":
            asyncio.run(repository.request_cancel(session, "task-id", "user-a"))
        else:
            asyncio.run(
                repository.retry_task(
                    session, "task-id", "user-a", idempotency_key="retry-key"
                )
            )
    query = session.scalar.call_args.args[0]
    compiled = query.compile(dialect=mysql.dialect())
    assert str(compiled).endswith("FOR UPDATE")
    assert compiled.params == {"task_id_1": "task-id", "user_id_1": "user-a"}
    assert query.get_execution_options()["populate_existing"] is True


@pytest.mark.parametrize(
    "field,value", [("limit", True), ("limit", 1.5), ("offset", "1"), ("offset", False)]
)
def test_pagination_rejects_non_integer_values_before_io(field, value):
    session = AsyncMock(spec=AsyncSession)
    with pytest.raises(TypeError):
        asyncio.run(repository.list_tasks(session, "user-a", **{field: value}))
    assert session.mock_calls == []
