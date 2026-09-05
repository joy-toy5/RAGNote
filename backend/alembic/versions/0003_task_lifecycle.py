"""增加持久任务事实与执行尝试历史。

Revision ID: 0003_task_lifecycle
Revises: 0002_index_contract
Create Date: 2026-09-06
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

revision: str = "0003_task_lifecycle"
down_revision: str | Sequence[str] | None = "0002_index_contract"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ASCII_SHA256 = sa.CHAR(64).with_variant(
    mysql.CHAR(64, collation="ascii_bin"),
    "mysql",
)
ASCII_UUID = sa.CHAR(36).with_variant(
    mysql.CHAR(36, collation="ascii_bin"),
    "mysql",
)
ASCII_USER_ID = sa.String(64).with_variant(
    mysql.VARCHAR(64, collation="ascii_bin"),
    "mysql",
)
ASCII_KIND = sa.String(64).with_variant(
    mysql.VARCHAR(64, collation="ascii_bin"),
    "mysql",
)
ASCII_KEY = sa.String(128).with_variant(
    mysql.VARCHAR(128, collation="ascii_bin"),
    "mysql",
)
ASCII_CODE = sa.String(64).with_variant(
    mysql.VARCHAR(64, collation="ascii_bin"),
    "mysql",
)
UTF8_REF = sa.String(1024).with_variant(
    mysql.VARCHAR(1024, collation="utf8mb4_bin"),
    "mysql",
)
ASCII_TEXT = sa.String(32).with_variant(
    mysql.VARCHAR(32, collation="ascii_bin"),
    "mysql",
)
ASCII_PHASE = sa.String(64).with_variant(
    mysql.VARCHAR(64, collation="ascii_bin"),
    "mysql",
)
UTF8_SUMMARY = sa.String(1024).with_variant(
    mysql.VARCHAR(1024, collation="utf8mb4_bin"),
    "mysql",
)
UNSIGNED_INTEGER = sa.Integer().with_variant(mysql.INTEGER(unsigned=True), "mysql")
UNSIGNED_BIGINT = sa.BigInteger().with_variant(mysql.BIGINT(unsigned=True), "mysql")
MYSQL_TABLE_OPTIONS = {
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_bin",
    "mysql_engine": "InnoDB",
}


def upgrade() -> None:
    """创建任务事实和不可覆盖的执行尝试历史。"""
    op.create_table(
        "background_tasks",
        sa.Column("task_id", ASCII_UUID, nullable=False, comment="持久任务 UUID"),
        sa.Column(
            "user_id", ASCII_USER_ID, nullable=False, comment="不透明认证主体 ID"
        ),
        sa.Column("kind", ASCII_KIND, nullable=False, comment="任务类型"),
        sa.Column("resource_id", ASCII_UUID, nullable=True, comment="目标资源 UUID"),
        sa.Column(
            "target_generation", UNSIGNED_INTEGER, nullable=True, comment="业务目标代次"
        ),
        sa.Column(
            "idempotency_key", ASCII_KEY, nullable=False, comment="用户作用域幂等键"
        ),
        sa.Column(
            "input_fingerprint", ASCII_SHA256, nullable=False, comment="不可变输入摘要"
        ),
        sa.Column(
            "input_schema_version",
            UNSIGNED_INTEGER,
            server_default=sa.text("1"),
            nullable=False,
            comment="任务输入 schema 版本",
        ),
        sa.Column(
            "input_blob_id",
            ASCII_SHA256,
            sa.ForeignKey("content_blobs.blob_id", ondelete="RESTRICT"),
            nullable=True,
            comment="可选原始 blob 摘要",
        ),
        sa.Column(
            "input_ref", UTF8_REF, nullable=False, comment="不可变输入引用或快照引用"
        ),
        sa.Column("input_metadata", sa.JSON(), nullable=True, comment="非敏感任务参数"),
        sa.Column(
            "status",
            ASCII_TEXT,
            server_default=sa.text("'pending'"),
            nullable=False,
            comment="任务状态",
        ),
        sa.Column("phase", ASCII_PHASE, nullable=True, comment="当前执行阶段"),
        sa.Column(
            "progress",
            UNSIGNED_INTEGER,
            server_default=sa.text("0"),
            nullable=False,
            comment="粗粒度百分比",
        ),
        sa.Column(
            "attempt_count",
            UNSIGNED_INTEGER,
            server_default=sa.text("0"),
            nullable=False,
            comment="已开始的尝试数",
        ),
        sa.Column(
            "max_attempts",
            UNSIGNED_INTEGER,
            server_default=sa.text("3"),
            nullable=False,
            comment="最大尝试数",
        ),
        sa.Column(
            "next_run_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="下一次可领取时间",
        ),
        sa.Column(
            "lease_owner",
            sa.String(128).with_variant(
                mysql.VARCHAR(128, collation="ascii_bin"), "mysql"
            ),
            nullable=True,
            comment="Worker 实例 UUID",
        ),
        sa.Column(
            "lease_token",
            UNSIGNED_BIGINT,
            server_default=sa.text("0"),
            nullable=False,
            comment="递增执行所有权代次",
        ),
        sa.Column(
            "lease_until",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="租约截止时间",
        ),
        sa.Column(
            "heartbeat_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="最近心跳时间",
        ),
        sa.Column("result_ref", UTF8_REF, nullable=True, comment="业务结果引用"),
        sa.Column(
            "result_version", UNSIGNED_INTEGER, nullable=True, comment="结果版本"
        ),
        sa.Column("error_code", ASCII_CODE, nullable=True, comment="脱敏错误码"),
        sa.Column("error_summary", UTF8_SUMMARY, nullable=True, comment="脱敏错误摘要"),
        sa.Column(
            "cancel_requested_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="取消请求时间",
        ),
        sa.Column(
            "retry_of_task_id",
            ASCII_UUID,
            sa.ForeignKey("background_tasks.task_id", ondelete="RESTRICT"),
            nullable=True,
            comment="人工重试来源任务",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "kind IN ('knowledge.index', 'note.index', 'note.delete')",
            name="ck_background_tasks_kind",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'retry_wait', 'succeeded', 'failed', 'superseded', 'cancelled')",
            name="ck_background_tasks_status",
        ),
        sa.CheckConstraint(
            "progress >= 0 AND progress <= 100", name="ck_background_tasks_progress"
        ),
        sa.CheckConstraint(
            "attempt_count >= 0", name="ck_background_tasks_attempt_count"
        ),
        sa.CheckConstraint(
            "max_attempts >= 1", name="ck_background_tasks_max_attempts"
        ),
        sa.CheckConstraint("lease_token >= 0", name="ck_background_tasks_lease_token"),
        sa.CheckConstraint(
            "result_version IS NULL OR result_version >= 1",
            name="ck_background_tasks_result_version",
        ),
        sa.CheckConstraint(
            "target_generation IS NULL OR target_generation >= 1",
            name="ck_background_tasks_generation",
        ),
        sa.CheckConstraint(
            "input_schema_version >= 1", name="ck_background_tasks_input_schema_version"
        ),
        sa.CheckConstraint(
            "input_blob_id IS NOT NULL OR input_ref <> ''",
            name="ck_background_tasks_input_ref",
        ),
        sa.PrimaryKeyConstraint("task_id"),
        sa.UniqueConstraint(
            "user_id", "idempotency_key", name="uq_background_tasks_user_idempotency"
        ),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_bin",
    )
    op.create_index(
        "ix_background_tasks_claim",
        "background_tasks",
        ["status", "next_run_at", "lease_until"],
    )
    op.create_index(
        "ix_background_tasks_user_updated",
        "background_tasks",
        ["user_id", "updated_at"],
    )
    op.create_index(
        "ix_background_tasks_resource_generation",
        "background_tasks",
        ["user_id", "resource_id", "target_generation"],
    )

    op.create_table(
        "task_attempts",
        sa.Column("task_id", ASCII_UUID, nullable=False, comment="所属任务 UUID"),
        sa.Column(
            "attempt_no",
            UNSIGNED_INTEGER,
            nullable=False,
            comment="任务内从 1 开始的尝试序号",
        ),
        sa.Column(
            "lease_owner",
            sa.String(128).with_variant(
                mysql.VARCHAR(128, collation="ascii_bin"), "mysql"
            ),
            nullable=False,
            comment="Worker 实例 UUID",
        ),
        sa.Column(
            "lease_token", UNSIGNED_BIGINT, nullable=False, comment="对应任务租约代次"
        ),
        sa.Column(
            "status",
            ASCII_TEXT,
            server_default=sa.text("'running'"),
            nullable=False,
            comment="尝试状态",
        ),
        sa.Column("phase", ASCII_PHASE, nullable=True, comment="失败或结束阶段"),
        sa.Column("error_code", ASCII_CODE, nullable=True, comment="脱敏错误码"),
        sa.Column("error_summary", UTF8_SUMMARY, nullable=True, comment="脱敏错误摘要"),
        sa.Column(
            "result_summary", UTF8_SUMMARY, nullable=True, comment="脱敏结果摘要"
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', 'abandoned', 'cancelled')",
            name="ck_task_attempts_status",
        ),
        sa.CheckConstraint("attempt_no >= 1", name="ck_task_attempts_attempt_no"),
        sa.CheckConstraint("lease_token >= 1", name="ck_task_attempts_lease_token"),
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["background_tasks.task_id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("task_id", "attempt_no"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_bin",
    )
    op.create_index(
        "ix_task_attempts_task_started", "task_attempts", ["task_id", "started_at"]
    )


def downgrade() -> None:
    """按依赖逆序删除任务事实表。"""
    op.drop_index("ix_task_attempts_task_started", table_name="task_attempts")
    op.drop_table("task_attempts")
    if op.get_context().dialect.name == "sqlite":
        # SQLite 的 DROP TABLE 会隐式删除行，自引用 RESTRICT 会阻止该删除。
        # 只拆除即将销毁表的内部链接，不关闭外键或修改保留的索引事实。
        op.execute(
            sa.text(
                "UPDATE background_tasks SET retry_of_task_id = NULL "
                "WHERE retry_of_task_id IS NOT NULL"
            )
        )
    op.drop_index(
        "ix_background_tasks_resource_generation", table_name="background_tasks"
    )
    op.drop_index("ix_background_tasks_user_updated", table_name="background_tasks")
    op.drop_index("ix_background_tasks_claim", table_name="background_tasks")
    op.drop_table("background_tasks")
