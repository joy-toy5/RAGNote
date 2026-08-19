from __future__ import annotations

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.database_url import build_database_url
from app.db.schema_gate import validate_schema_on_engine

load_dotenv()

ASYNC_DATABASE_URL = build_database_url()

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    echo=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def check_database_schema(engine: AsyncEngine | None = None) -> None:
    """验证数据库兼容性，不执行任何 DDL 或迁移。"""
    database_engine = async_engine if engine is None else engine
    await validate_schema_on_engine(database_engine)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_mysql_connection() -> bool:
    """检查 MySQL 连接。"""
    try:
        async with async_engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        print(f"MySQL 连接失败: {exc}")
        return False
