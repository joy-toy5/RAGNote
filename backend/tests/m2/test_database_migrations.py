from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event, inspect, text

from app.db.database_url import build_database_url
from app.db.migration_guard import (
    DOWNGRADE_ACKNOWLEDGEMENT,
    NON_TEST_ACKNOWLEDGEMENT,
    MigrationSafetyError,
    validate_migration_command,
    validate_migration_target,
)
from app.db.schema_gate import (
    EXPECTED_SCHEMA_REVISION,
    SchemaCompatibilityError,
    validate_schema_compatibility,
    validate_schema_on_engine,
)
from app.models import Base
import app.db.schema_gate as schema_gate


def _alembic_config(backend_root: Path, connection: Any) -> Config:
    config = Config(str(backend_root / "alembic.ini"))
    config.attributes["connection"] = connection
    return config


def test_database_url_escapes_credentials_without_persisting_them(
    backend_root: Path,
) -> None:
    environment = {
        "MYSQL_USER": "rag-user",
        "MYSQL_PASSWORD": "p@ss:/%word",
        "MYSQL_HOST": "db.internal",
        "MYSQL_PORT": "3307",
        "MYSQL_DATABASE": "rag_note",
    }

    url = build_database_url(environment)

    assert url.password == environment["MYSQL_PASSWORD"]
    assert url.render_as_string(hide_password=True).startswith(
        "mysql+aiomysql://rag-user:***@db.internal:3307/rag_note"
    )
    assert "p%40ss%3A%2F%25word" in url.render_as_string(hide_password=False)
    assert "sqlalchemy.url" not in (backend_root / "alembic.ini").read_text()
    assert environment["MYSQL_PASSWORD"] not in (
        backend_root / "alembic.ini"
    ).read_text()


def test_migration_target_requires_exact_database_and_non_test_acknowledgement() -> None:
    test_environment = {
        "MYSQL_DATABASE": "rag_note_test",
        "RAG_MIGRATION_TARGET": "rag_note_test",
    }
    validate_migration_target(build_database_url(test_environment), test_environment)

    production_environment = {
        "MYSQL_DATABASE": "rag_note",
        "RAG_MIGRATION_TARGET": "rag_note",
    }
    with pytest.raises(MigrationSafetyError, match="二次确认"):
        validate_migration_target(
            build_database_url(production_environment),
            production_environment,
        )
    production_environment["RAG_ALLOW_NON_TEST_MIGRATIONS"] = (
        NON_TEST_ACKNOWLEDGEMENT
    )
    validate_migration_target(
        build_database_url(production_environment),
        production_environment,
    )


def test_migration_guard_binds_legacy_preflight_stamp_and_downgrade(
    backend_root: Path,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        command.upgrade(_alembic_config(backend_root, connection), "0001_legacy")
        connection.execute(text("DROP TABLE alembic_version"))

        with pytest.raises(MigrationSafetyError, match="禁止直接 upgrade"):
            validate_migration_command(connection, "upgrade", ("head",), {})
        validate_migration_command(connection, "stamp", ("0001_legacy",), {})
        with pytest.raises(MigrationSafetyError, match="只允许"):
            validate_migration_command(connection, "stamp", ("head",), {})
        with pytest.raises(MigrationSafetyError, match="破坏性确认"):
            validate_migration_command(connection, "downgrade", ("base",), {})
        validate_migration_command(
            connection,
            "downgrade",
            ("base",),
            {"RAG_ALLOW_DESTRUCTIVE_DOWNGRADE": DOWNGRADE_ACKNOWLEDGEMENT},
        )


def test_models_register_one_complete_metadata_contract() -> None:
    assert set(Base.metadata.tables) == {
        "chat_messages",
        "chat_sessions",
        "content_blobs",
        "document_revisions",
        "documents",
        "index_chunks",
        "index_versions",
        "notes",
        "review_records",
    }


def test_legacy_baseline_upgrades_downgrades_and_reupgrades(
    backend_root: Path,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        config = _alembic_config(backend_root, connection)

        command.upgrade(config, "head")
        validate_schema_compatibility(connection)
        assert set(inspect(connection).get_table_names()) == {
            "alembic_version",
            "chat_messages",
            "chat_sessions",
            "content_blobs",
            "document_revisions",
            "documents",
            "index_chunks",
            "index_versions",
            "notes",
            "review_records",
        }

        command.downgrade(config, "base")
        assert inspect(connection).get_table_names() == ["alembic_version"]

        command.upgrade(config, "head")
        validate_schema_compatibility(connection)


def test_unversioned_legacy_schema_requires_explicit_preflight_mode() -> None:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        backend_root = Path(__file__).resolve().parents[2]
        command.upgrade(_alembic_config(backend_root, connection), "0001_legacy")
        connection.execute(text("DROP TABLE alembic_version"))

        validate_schema_compatibility(
            connection,
            allow_unversioned_legacy=True,
        )
        with pytest.raises(SchemaCompatibilityError, match="alembic_version"):
            validate_schema_compatibility(connection)


def test_migration_guard_rejects_restamping_versioned_database(
    backend_root: Path,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        command.upgrade(_alembic_config(backend_root, connection), "head")

        with pytest.raises(MigrationSafetyError, match="禁止重新 stamp"):
            validate_migration_command(
                connection,
                "stamp",
                ("0001_legacy",),
                {},
            )


def test_schema_gate_rejects_unknown_revision(backend_root: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        command.upgrade(_alembic_config(backend_root, connection), "head")
        connection.execute(
            text("UPDATE alembic_version SET version_num = '9999_unknown'")
        )

        with pytest.raises(SchemaCompatibilityError, match="revision 不兼容"):
            validate_schema_compatibility(connection)


@pytest.mark.parametrize(
    ("method_name", "table_name", "message"),
    [
        ("get_unique_constraints", "documents", "唯一约束"),
        ("get_check_constraints", "content_blobs", "CHECK 约束"),
    ],
)
def test_schema_gate_rejects_missing_fact_constraints(
    backend_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    table_name: str,
    message: str,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        command.upgrade(_alembic_config(backend_root, connection), "head")
        real_inspector = inspect(connection)

        class InspectorProxy:
            def __getattr__(self, name: str) -> Any:
                if name == method_name:
                    return lambda requested_table: [] if requested_table == table_name else getattr(
                        real_inspector, name
                    )(requested_table)
                return getattr(real_inspector, name)

        monkeypatch.setattr(
            schema_gate,
            "inspect",
            lambda _connection: InspectorProxy(),
        )

        with pytest.raises(SchemaCompatibilityError, match=message):
            validate_schema_compatibility(connection)


def test_schema_gate_requires_mysql_check_enforcement() -> None:
    unsupported = SimpleNamespace(
        dialect=SimpleNamespace(name="mysql", server_version_info=(8, 0, 15))
    )

    with pytest.raises(SchemaCompatibilityError, match="8.0.16"):
        schema_gate._validate_database_capabilities(unsupported)  # type: ignore[arg-type]


def test_schema_gate_only_emits_read_statements(backend_root: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    statements: list[str] = []
    with engine.begin() as connection:
        command.upgrade(_alembic_config(backend_root, connection), "head")

        def record_statement(
            _connection: Any,
            _cursor: Any,
            statement: str,
            _parameters: Any,
            _context: Any,
            _executemany: bool,
        ) -> None:
            statements.append(statement.strip().upper())

        event.listen(engine, "before_cursor_execute", record_statement)
        try:
            validate_schema_compatibility(connection)
        finally:
            event.remove(engine, "before_cursor_execute", record_statement)

    forbidden_prefixes = ("ALTER ", "CREATE ", "DELETE ", "DROP ", "INSERT ", "UPDATE ")
    assert statements
    assert not any(
        statement.startswith(forbidden_prefixes) for statement in statements
    )


def test_async_schema_gate_accepts_an_injected_engine() -> None:
    events: list[str] = []

    class FakeConnection:
        async def __aenter__(self) -> "FakeConnection":
            events.append("connect")
            return self

        async def __aexit__(self, *_: object) -> None:
            events.append("close")

        async def run_sync(self, validator: Any) -> None:
            events.append("validate")
            validator(object())

    class FakeEngine:
        def connect(self) -> FakeConnection:
            return FakeConnection()

    def validator(_connection: object) -> None:
        events.append(EXPECTED_SCHEMA_REVISION)

    asyncio.run(
        validate_schema_on_engine(FakeEngine(), validator=validator)  # type: ignore[arg-type]
    )

    assert events == ["connect", "validate", EXPECTED_SCHEMA_REVISION, "close"]
