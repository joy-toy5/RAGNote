"""M5 持久任务与执行尝试的 SQLAlchemy 模型。"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    CHAR,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects import mysql
from sqlalchemy.sql import func

from app.models.chat_history import Base

ASCII_SHA256 = CHAR(64).with_variant(mysql.CHAR(64, collation="ascii_bin"), "mysql")
ASCII_UUID = CHAR(36).with_variant(mysql.CHAR(36, collation="ascii_bin"), "mysql")
ASCII_USER_ID = String(64).with_variant(
    mysql.VARCHAR(64, collation="ascii_bin"), "mysql"
)
ASCII_KIND = String(64).with_variant(mysql.VARCHAR(64, collation="ascii_bin"), "mysql")
ASCII_KEY = String(128).with_variant(mysql.VARCHAR(128, collation="ascii_bin"), "mysql")
ASCII_CODE = String(64).with_variant(mysql.VARCHAR(64, collation="ascii_bin"), "mysql")
ASCII_TEXT = String(32).with_variant(mysql.VARCHAR(32, collation="ascii_bin"), "mysql")
ASCII_PHASE = String(64).with_variant(mysql.VARCHAR(64, collation="ascii_bin"), "mysql")
ASCII_OWNER = String(128).with_variant(
    mysql.VARCHAR(128, collation="ascii_bin"), "mysql"
)
UTF8_REF = String(1024).with_variant(
    mysql.VARCHAR(1024, collation="utf8mb4_bin"), "mysql"
)
UTF8_SUMMARY = String(1024).with_variant(
    mysql.VARCHAR(1024, collation="utf8mb4_bin"), "mysql"
)
UNSIGNED_INTEGER = Integer().with_variant(mysql.INTEGER(unsigned=True), "mysql")
UNSIGNED_BIGINT = BigInteger().with_variant(mysql.BIGINT(unsigned=True), "mysql")
TABLE_OPTIONS = {
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_bin",
    "mysql_engine": "InnoDB",
}


class BackgroundTask(Base):
    """脱离 HTTP 连接存在的任务接单意图和当前状态。"""

    __tablename__ = "background_tasks"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "idempotency_key",
            name="uq_background_tasks_user_idempotency",
        ),
        Index(
            "ix_background_tasks_claim",
            "status",
            "next_run_at",
            "lease_until",
        ),
        Index("ix_background_tasks_user_updated", "user_id", "updated_at"),
        Index(
            "ix_background_tasks_resource_generation",
            "user_id",
            "resource_id",
            "target_generation",
        ),
        CheckConstraint(
            "kind IN ('knowledge.index', 'note.index', 'note.delete')",
            name="ck_background_tasks_kind",
        ),
        CheckConstraint(
            "status IN ('pending', 'processing', 'retry_wait', 'succeeded', 'failed', 'superseded', 'cancelled')",
            name="ck_background_tasks_status",
        ),
        CheckConstraint(
            "progress >= 0 AND progress <= 100", name="ck_background_tasks_progress"
        ),
        CheckConstraint("attempt_count >= 0", name="ck_background_tasks_attempt_count"),
        CheckConstraint("max_attempts >= 1", name="ck_background_tasks_max_attempts"),
        CheckConstraint("lease_token >= 0", name="ck_background_tasks_lease_token"),
        CheckConstraint(
            "result_version IS NULL OR result_version >= 1",
            name="ck_background_tasks_result_version",
        ),
        CheckConstraint(
            "target_generation IS NULL OR target_generation >= 1",
            name="ck_background_tasks_generation",
        ),
        CheckConstraint(
            "input_schema_version >= 1",
            name="ck_background_tasks_input_schema_version",
        ),
        CheckConstraint(
            "input_blob_id IS NOT NULL OR input_ref <> ''",
            name="ck_background_tasks_input_ref",
        ),
        TABLE_OPTIONS,
    )

    task_id = Column(ASCII_UUID, primary_key=True, comment="持久任务 UUID")
    user_id = Column(ASCII_USER_ID, nullable=False, comment="不透明认证主体 ID")
    kind = Column(ASCII_KIND, nullable=False, comment="任务类型")
    resource_id = Column(ASCII_UUID, nullable=True, comment="目标资源 UUID")
    target_generation = Column(UNSIGNED_INTEGER, nullable=True, comment="业务目标代次")
    idempotency_key = Column(ASCII_KEY, nullable=False, comment="用户作用域幂等键")
    input_fingerprint = Column(ASCII_SHA256, nullable=False, comment="不可变输入摘要")
    input_schema_version = Column(
        UNSIGNED_INTEGER,
        nullable=False,
        server_default=text("1"),
        comment="任务输入 schema 版本",
    )
    input_blob_id = Column(
        ASCII_SHA256,
        ForeignKey("content_blobs.blob_id", ondelete="RESTRICT"),
        nullable=True,
        comment="可选原始 blob 摘要",
    )
    input_ref = Column(UTF8_REF, nullable=False, comment="不可变输入引用或快照引用")
    input_metadata = Column(JSON, nullable=True, comment="非敏感任务参数")

    status = Column(
        ASCII_TEXT,
        nullable=False,
        server_default=text("'pending'"),
        comment="任务状态",
    )
    phase = Column(ASCII_PHASE, nullable=True, comment="当前执行阶段")
    progress = Column(
        UNSIGNED_INTEGER,
        nullable=False,
        server_default=text("0"),
        comment="粗粒度百分比",
    )
    attempt_count = Column(
        UNSIGNED_INTEGER,
        nullable=False,
        server_default=text("0"),
        comment="已开始的尝试数",
    )
    max_attempts = Column(
        UNSIGNED_INTEGER,
        nullable=False,
        server_default=text("3"),
        comment="最大尝试数",
    )
    next_run_at = Column(
        DateTime(timezone=True), nullable=True, comment="下一次可领取时间"
    )

    lease_owner = Column(ASCII_OWNER, nullable=True, comment="Worker 实例 UUID")
    lease_token = Column(
        UNSIGNED_BIGINT,
        nullable=False,
        server_default=text("0"),
        comment="递增执行所有权代次",
    )
    lease_until = Column(DateTime(timezone=True), nullable=True, comment="租约截止时间")
    heartbeat_at = Column(
        DateTime(timezone=True), nullable=True, comment="最近心跳时间"
    )

    result_ref = Column(UTF8_REF, nullable=True, comment="业务结果引用")
    result_version = Column(UNSIGNED_INTEGER, nullable=True, comment="结果版本")
    error_code = Column(ASCII_CODE, nullable=True, comment="脱敏错误码")
    error_summary = Column(UTF8_SUMMARY, nullable=True, comment="脱敏错误摘要")
    cancel_requested_at = Column(
        DateTime(timezone=True), nullable=True, comment="取消请求时间"
    )
    retry_of_task_id = Column(
        ASCII_UUID,
        ForeignKey("background_tasks.task_id", ondelete="RESTRICT"),
        nullable=True,
        comment="人工重试来源任务",
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    started_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=func.now(),
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)


class TaskAttempt(Base):
    """一次任务执行尝试的不可覆盖历史。"""

    __tablename__ = "task_attempts"
    __table_args__ = (
        Index("ix_task_attempts_task_started", "task_id", "started_at"),
        CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', 'abandoned', 'cancelled')",
            name="ck_task_attempts_status",
        ),
        CheckConstraint("attempt_no >= 1", name="ck_task_attempts_attempt_no"),
        CheckConstraint("lease_token >= 1", name="ck_task_attempts_lease_token"),
        TABLE_OPTIONS,
    )

    task_id = Column(
        ASCII_UUID,
        ForeignKey("background_tasks.task_id", ondelete="RESTRICT"),
        primary_key=True,
        comment="所属任务 UUID",
    )
    attempt_no = Column(
        UNSIGNED_INTEGER,
        primary_key=True,
        comment="任务内从 1 开始的尝试序号",
    )
    lease_owner = Column(ASCII_OWNER, nullable=False, comment="Worker 实例 UUID")
    lease_token = Column(UNSIGNED_BIGINT, nullable=False, comment="对应任务租约代次")
    status = Column(
        ASCII_TEXT,
        nullable=False,
        server_default=text("'running'"),
        comment="尝试状态",
    )
    phase = Column(ASCII_PHASE, nullable=True, comment="失败或结束阶段")
    error_code = Column(ASCII_CODE, nullable=True, comment="脱敏错误码")
    error_summary = Column(UTF8_SUMMARY, nullable=True, comment="脱敏错误摘要")
    result_summary = Column(UTF8_SUMMARY, nullable=True, comment="脱敏结果摘要")
    started_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    finished_at = Column(DateTime(timezone=True), nullable=True)
