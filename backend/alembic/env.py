from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.db.database_url import build_database_url
from app.db.migration_guard import (
    validate_migration_command,
    validate_migration_target,
)
from app.indexing.models import (
    ContentBlob,
    DocumentRevision,
    IndexChunk,
    IndexedDocument,
    IndexVersion,
)
from app.models import Base
from app.tasking.models import BackgroundTask, TaskAttempt

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

TASKING_MODELS = (BackgroundTask, TaskAttempt)

INDEXING_MODELS = (
    ContentBlob,
    DocumentRevision,
    IndexChunk,
    IndexedDocument,
    IndexVersion,
)
target_metadata = Base.metadata


def _database_url() -> str:
    load_dotenv()
    url = build_database_url()
    validate_migration_target(url)
    return url.render_as_string(hide_password=False)


def _cli_command() -> tuple[str | None, tuple[str, ...]]:
    options = getattr(config, "cmd_opts", None)
    command = getattr(options, "cmd", None)
    if not command:
        return None, ()
    command_name = command[0].__name__
    raw_revisions = getattr(options, "revisions", None)
    if raw_revisions is None:
        raw_revision = getattr(options, "revision", None)
        raw_revisions = () if raw_revision is None else (raw_revision,)
    return command_name, tuple(raw_revisions)


def run_migrations_offline() -> None:
    """生成离线 SQL，不创建数据库连接。"""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """在显式提供的连接上执行迁移。"""
    command_name, revisions = _cli_command()
    if command_name is not None:
        validate_migration_command(
            connection,
            command_name,
            revisions,
            os.environ,
        )
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """仅由显式 Alembic 命令创建短生命周期异步引擎。"""
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _database_url()
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """优先使用测试或运维方注入的连接，否则创建异步连接。"""
    injected_connection = config.attributes.get("connection")
    if injected_connection is not None:
        do_run_migrations(injected_connection)
        return
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
