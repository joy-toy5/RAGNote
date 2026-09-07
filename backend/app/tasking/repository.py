"""持久任务的事务内接单、查询、取消与人工重试。"""

from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.tasking.contracts import (
    TERMINAL_TASK_STATUSES,
    TaskSubmission,
    canonical_task_metadata,
    snapshot_submission,
    validate_task_key,
    validate_task_user_id,
)
from app.tasking.errors import (
    TaskIdempotencyConflict,
    TaskNotFound,
    TaskStateConflict,
)
from app.tasking.models import BackgroundTask

MAX_TASK_PAGE_SIZE = 100
DEFAULT_TASK_PAGE_SIZE = 20
_RETRYABLE_MANUAL_STATES = frozenset(("failed", "cancelled"))
_MYSQL_IDEMPOTENCY_KEY = re.compile(
    r"for key ['`](?:[\w$]+\.)*uq_background_tasks_user_idempotency['`]$",
)
_SQLITE_IDEMPOTENCY_ERROR = "UNIQUE constraint failed: background_tasks.user_id, background_tasks.idempotency_key"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _is_idempotency_conflict(error: IntegrityError, dialect: str) -> bool:
    """只认目标唯一键，不把 FK/CHECK/主键错误或死锁当成幂等竞争。"""
    if dialect in {"mysql", "mariadb"}:
        args = getattr(error.orig, "args", ())
        return (
            len(args) >= 2
            and args[0] == 1062
            and isinstance(args[1], str)
            and _MYSQL_IDEMPOTENCY_KEY.search(args[1]) is not None
        )
    return (
        dialect == "sqlite"
        and getattr(error.orig, "sqlite_errorcode", None)
        == sqlite3.SQLITE_CONSTRAINT_UNIQUE
        and str(error.orig) == _SQLITE_IDEMPOTENCY_ERROR
    )


def _submission_matches(task: BackgroundTask, submission: TaskSubmission) -> bool:
    """比较幂等任务的不可变业务输入，不比较数据库生成字段。"""
    return (
        task.user_id == submission.user_id
        and task.kind == submission.kind
        and task.resource_id == submission.resource_id
        and task.target_generation == submission.target_generation
        and task.input_fingerprint == submission.input_fingerprint
        and task.input_schema_version == submission.input_schema_version
        and task.input_blob_id == submission.input_blob_id
        and task.input_ref == submission.input_ref
        and canonical_task_metadata(task.input_metadata)
        == canonical_task_metadata(submission.input_metadata)
        and task.max_attempts == submission.max_attempts
        and task.retry_of_task_id == submission.retry_of_task_id
    )


async def _get_by_idempotency(
    session: AsyncSession,
    user_id: str,
    idempotency_key: str,
    *,
    lock: bool = False,
) -> BackgroundTask | None:
    query = select(BackgroundTask).where(
        BackgroundTask.user_id == user_id,
        BackgroundTask.idempotency_key == idempotency_key,
    )
    if lock:
        # 重复键竞争可能已持有共享锁；当前读不再额外升级为独占锁。
        query = query.with_for_update(read=True).execution_options(
            populate_existing=True
        )
    return await session.scalar(query)


async def create_or_get_task(
    session: AsyncSession,
    submission: TaskSubmission,
) -> tuple[BackgroundTask, bool]:
    """在调用方事务中登记任务；返回 ``(任务, 是否新建)``，不提交事务。"""
    submission = snapshot_submission(submission)
    # begin_nested 自身也会 flush；调用方工作必须在捕获候选 INSERT 错误之前落盘。
    await session.flush()
    existing = await _get_by_idempotency(
        session,
        submission.user_id,
        submission.idempotency_key,
    )
    if existing is not None:
        if not _submission_matches(existing, submission):
            raise TaskIdempotencyConflict("幂等键已对应不同任务输入")
        return existing, False

    task = BackgroundTask(
        task_id=str(uuid.uuid4()),
        user_id=submission.user_id,
        kind=submission.kind,
        resource_id=submission.resource_id,
        target_generation=submission.target_generation,
        idempotency_key=submission.idempotency_key,
        input_fingerprint=submission.input_fingerprint,
        input_schema_version=submission.input_schema_version,
        input_blob_id=submission.input_blob_id,
        input_ref=submission.input_ref,
        input_metadata=submission.input_metadata,
        max_attempts=submission.max_attempts,
        retry_of_task_id=submission.retry_of_task_id,
    )
    try:
        # SAVEPOINT 只包住候选 INSERT，不能因幂等竞争回滚调用方事务。
        async with session.begin_nested():
            session.add(task)
            await session.flush()
    except IntegrityError as exc:
        if not _is_idempotency_conflict(exc, session.get_bind().dialect.name):
            raise
        # MySQL REPEATABLE READ 下普通 SELECT 可能仍看旧快照；竞争后用当前锁定读。
        winner = await _get_by_idempotency(
            session,
            submission.user_id,
            submission.idempotency_key,
            lock=True,
        )
        if winner is None:
            raise
        if not _submission_matches(winner, submission):
            raise TaskIdempotencyConflict("幂等键已对应不同任务输入")
        return winner, False
    return task, True


async def get_task(
    session: AsyncSession,
    task_id: str,
    user_id: str,
) -> BackgroundTask:
    """按任务 ID 和用户同时过滤，避免泄露跨用户存在性。"""
    return await _get_task(session, task_id, user_id, lock=False)


async def _get_task(
    session: AsyncSession,
    task_id: str,
    user_id: str,
    *,
    lock: bool,
) -> BackgroundTask:
    query = select(BackgroundTask).where(
        BackgroundTask.task_id == task_id,
        BackgroundTask.user_id == user_id,
    )
    if lock:
        query = query.with_for_update().execution_options(populate_existing=True)
    task = await session.scalar(query)
    if task is None:
        raise TaskNotFound("任务不存在")
    return task


async def list_tasks(
    session: AsyncSession,
    user_id: str,
    limit: int = DEFAULT_TASK_PAGE_SIZE,
    *,
    offset: int = 0,
    idempotency_key: str | None = None,
) -> list[BackgroundTask]:
    """只返回当前用户任务，并强制限制单页大小。"""
    validate_task_user_id(user_id)
    for label, value in (("limit", limit), ("offset", offset)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} 必须是整数")
    bounded_limit = min(max(limit, 1), MAX_TASK_PAGE_SIZE)
    bounded_offset = max(offset, 0)
    filters = [BackgroundTask.user_id == user_id]
    if idempotency_key is not None:
        validate_task_key(idempotency_key)
        filters.append(BackgroundTask.idempotency_key == idempotency_key)
    result = await session.scalars(
        select(BackgroundTask)
        .where(*filters)
        .order_by(
            BackgroundTask.updated_at.desc(),
            BackgroundTask.created_at.desc(),
            BackgroundTask.task_id.desc(),
        )
        .offset(bounded_offset)
        .limit(bounded_limit)
    )
    return list(result)


async def request_cancel(
    session: AsyncSession,
    task_id: str,
    user_id: str,
) -> BackgroundTask:
    """请求取消；处理中任务只记录请求，不伪造已停止。"""
    task = await _get_task(session, task_id, user_id, lock=True)
    if task.status in TERMINAL_TASK_STATUSES:
        raise TaskStateConflict("终态任务不能重复取消")

    requested_at = task.cancel_requested_at or _now()
    if task.status in {"pending", "retry_wait"}:
        task.status = "cancelled"
        task.phase = "cancelled"
        task.lease_owner = None
        task.lease_until = None
        task.cancel_requested_at = requested_at
        task.completed_at = requested_at
    elif task.status == "processing":
        task.cancel_requested_at = requested_at
    else:
        raise TaskStateConflict("当前任务状态不允许取消")
    await session.flush()
    return task


async def retry_task(
    session: AsyncSession,
    task_id: str,
    user_id: str,
    *,
    idempotency_key: str,
) -> BackgroundTask:
    """为失败/取消任务创建新的待执行任务，保留原任务终态。"""
    validate_task_key(idempotency_key)
    source = await _get_task(session, task_id, user_id, lock=True)
    if source.status not in _RETRYABLE_MANUAL_STATES:
        raise TaskStateConflict("当前任务状态不允许人工重试")

    submission = TaskSubmission(
        user_id=source.user_id,
        kind=source.kind,
        idempotency_key=idempotency_key,
        input_fingerprint=source.input_fingerprint,
        input_ref=source.input_ref,
        resource_id=source.resource_id,
        target_generation=source.target_generation,
        input_blob_id=source.input_blob_id,
        input_schema_version=source.input_schema_version,
        input_metadata=source.input_metadata,
        max_attempts=source.max_attempts,
        retry_of_task_id=source.task_id,
    )
    task, _ = await create_or_get_task(session, submission)
    return task


async def start_local_task(session: AsyncSession, task_id: str, user_id: str) -> bool:
    """Demo本进程已接单任务开始执行；不领取租约或自动重放。"""
    task = await _get_task(session, task_id, user_id, lock=True)
    if task.status != "pending":
        return False
    task.status = "processing"
    task.phase = "processing"
    task.started_at = _now()
    await session.flush()
    return True


async def finish_local_task(
    session: AsyncSession, task_id: str, user_id: str, *, status: str,
    result_ref: str | None = None, error_code: str | None = None,
    error_summary: str | None = None,
) -> None:
    """复用已有结果/错误字段；调用方负责事务提交与错误脱敏。"""
    if status not in {"succeeded", "failed"}:
        raise ValueError("本地执行只结算成功或失败")
    task = await _get_task(session, task_id, user_id, lock=True)
    if task.status in TERMINAL_TASK_STATUSES:
        return
    task.status = status
    task.phase = status
    task.progress = 100 if status == "succeeded" else task.progress
    task.result_ref = result_ref
    task.result_version = 1 if result_ref else None
    task.error_code = error_code
    task.error_summary = error_summary
    task.completed_at = _now()
    await session.flush()
