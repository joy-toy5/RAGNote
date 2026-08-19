"""建立 FastAPI 现有四表的遗留结构基线。

Revision ID: 0001_legacy
Revises:
Create Date: 2026-08-17
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "0001_legacy"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """在空库中创建与 legacy ORM 一致的四张表。"""
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_sessions_id", "chat_sessions", ["id"], unique=False)
    op.create_index(
        "ix_chat_sessions_user_id", "chat_sessions", ["user_id"], unique=False
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_messages_id", "chat_messages", ["id"], unique=False)

    op.create_table(
        "notes",
        sa.Column("id", sa.String(length=36), nullable=False, comment="UUID"),
        sa.Column(
            "user_id",
            sa.String(length=36),
            nullable=False,
            comment="用户ID",
        ),
        sa.Column(
            "title", sa.String(length=200), nullable=False, comment="笔记标题"
        ),
        sa.Column("content", sa.Text(), nullable=False, comment="Markdown原文"),
        sa.Column(
            "tags", sa.JSON(), nullable=True, comment='标签列表 ["AI", "FastAPI"]'
        ),
        sa.Column(
            "category",
            sa.String(length=50),
            nullable=True,
            comment="分类 work/study/life/project",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
            comment="创建时间",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
            comment="更新时间",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_notes_user_id", "notes", ["user_id"], unique=False)

    op.create_table(
        "review_records",
        sa.Column("id", sa.String(length=36), nullable=False, comment="UUID"),
        sa.Column(
            "note_id", sa.String(length=36), nullable=False, comment="笔记ID"
        ),
        sa.Column(
            "user_id",
            sa.String(length=36),
            nullable=False,
            comment="用户ID",
        ),
        sa.Column(
            "last_reviewed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="上次回顾时间",
        ),
        sa.Column(
            "review_count", sa.Integer(), nullable=True, comment="回顾次数"
        ),
        sa.Column(
            "next_review_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="下次回顾时间",
        ),
        sa.Column(
            "interval_days", sa.Integer(), nullable=True, comment="当前间隔天数"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
            comment="创建时间",
        ),
        sa.ForeignKeyConstraint(
            ["note_id"], ["notes.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_review_records_user_id", "review_records", ["user_id"], unique=False
    )


def downgrade() -> None:
    """仅在隔离库中按依赖逆序删除 legacy 表。"""
    op.drop_index("ix_review_records_user_id", table_name="review_records")
    op.drop_table("review_records")
    op.drop_index("ix_notes_user_id", table_name="notes")
    op.drop_table("notes")
    op.drop_index("ix_chat_messages_id", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index("ix_chat_sessions_user_id", table_name="chat_sessions")
    op.drop_index("ix_chat_sessions_id", table_name="chat_sessions")
    op.drop_table("chat_sessions")
