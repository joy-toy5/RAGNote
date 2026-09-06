"""事务内租约与尝试状态机；不启动 Worker，不授予外部存储写权。

调用方使用独立短事务，并在领取提交后才开始执行。结算/重试的前提是
真实执行已经结束；取消协程、请求超时或租约过期都不是线程停止证明。
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, case, func, literal_column, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.tasking.errors import TaskLeaseLost, TaskStateConflict
from app.tasking.lease_contracts import (
    MAX_ATTEMPT_COUNT,
    MAX_LEASE_TOKEN,
    RetryPolicy,
    TaskLease,
    validate_code,
    validate_count,
    validate_lease_owner,
    validate_lease_seconds,
    validate_summary,
    validate_task_kinds,
)
from app.tasking.models import BackgroundTask, TaskAttempt

_CLAIMABLE_STATUSES = ("pending", "retry_wait")


def _utc(value: datetime) -> datetime:
    """MySQL DATETIME 不携带时区，执行事实统一使用无时区 UTC。"""
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _clock_expression(session: AsyncSession):
    dialect = session.get_bind().dialect.name
    if dialect in {"mysql", "mariadb"}:
        return func.utc_timestamp(type_=DateTime())
    if dialect == "sqlite":
        return func.current_timestamp(type_=DateTime())
    raise RuntimeError("租约时钟尚未支持当前数据库方言")


def _after_seconds(session: AsyncSession, seconds: int):
    clock = _clock_expression(session)
    if session.get_bind().dialect.name == "sqlite":
        return func.datetime(clock, f"+{seconds} seconds", type_=DateTime())
    return func.timestampadd(literal_column("SECOND"), seconds, clock, type_=DateTime())


def _instant(session: AsyncSession, value):
    # SQLite DATETIME 可有不同小数位文本，必须按时间比较而非字符串比较。
    return (
        func.julianday(value) if session.get_bind().dialect.name == "sqlite" else value
    )


async def _database_now(session: AsyncSession) -> datetime:
    value = await session.scalar(select(_clock_expression(session)))
    if not isinstance(value, datetime):
        raise TypeError("数据库未返回有效 UTC 时间")
    return _utc(value)


def _claim_filters(session: AsyncSession, kinds: tuple[str, ...], now):
    return (
        BackgroundTask.status.in_(_CLAIMABLE_STATUSES),
        BackgroundTask.kind.in_(kinds),
        BackgroundTask.cancel_requested_at.is_(None),
        BackgroundTask.attempt_count < BackgroundTask.max_attempts,
        or_(
            BackgroundTask.next_run_at.is_(None),
            _instant(session, BackgroundTask.next_run_at) <= _instant(session, now),
        ),
    )


def _ownership_filters(lease: TaskLease):
    return (
        BackgroundTask.task_id == lease.task_id,
        BackgroundTask.lease_owner == lease.owner,
        BackgroundTask.lease_token == lease.token,
        BackgroundTask.attempt_count == lease.attempt_no,
        BackgroundTask.status == "processing",
    )


async def _fenced_update(
    session: AsyncSession, task: BackgroundTask, lease: TaskLease, values: dict
) -> None:
    # 先用实际 UPDATE 的时钟做 fencing；不能预先改脏 ORM 对象触发 autoflush。
    result = await session.execute(
        update(BackgroundTask)
        .where(
            *_ownership_filters(lease),
            _instant(session, BackgroundTask.lease_until)
            > _instant(session, _clock_expression(session)),
        )
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        raise TaskLeaseLost("任务在状态写入前已失去有效租约")
    await session.refresh(task, with_for_update=True)


def _receipt(task: BackgroundTask) -> TaskLease:
    return TaskLease(
        task.task_id, task.lease_owner, task.lease_token, task.attempt_count
    )


def _claimable(task: BackgroundTask, now: datetime, kinds: tuple[str, ...]) -> bool:
    return (
        task.status in _CLAIMABLE_STATUSES
        and task.kind in kinds
        and task.cancel_requested_at is None
        and task.attempt_count < task.max_attempts
        and (task.next_run_at is None or _utc(task.next_run_at) <= now)
    )


async def claim_task(
    session: AsyncSession,
    *,
    owner: str,
    kinds: tuple[str, ...],
    lease_seconds: int = 90,
) -> TaskLease | None:
    """领取一项支持的到期任务；processing（包括过期项）绝不自动重领。"""
    validate_lease_owner(owner)
    validate_task_kinds(kinds)
    validate_lease_seconds(lease_seconds)
    now = await _database_now(session)
    task = await session.scalar(
        select(BackgroundTask)
        .where(*_claim_filters(session, kinds, now))
        .order_by(BackgroundTask.created_at, BackgroundTask.task_id)
        .limit(1)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if task is None:
        return None
    # 行锁可能等待；不能拿加锁前的时间开始租约或接受已经失效的状态。
    now = await _database_now(session)
    if not _claimable(task, now, kinds):
        return None
    if task.lease_token >= MAX_LEASE_TOKEN or task.attempt_count >= MAX_ATTEMPT_COUNT:
        raise TaskStateConflict("任务执行计数已达到存储上限")
    if not await _start_attempt(session, task, owner, kinds, lease_seconds):
        return None
    await session.flush()
    return _receipt(task)


async def _start_attempt(
    session: AsyncSession,
    task: BackgroundTask,
    owner: str,
    kinds: tuple[str, ...],
    lease_seconds: int,
) -> bool:
    clock = _clock_expression(session)
    result = await session.execute(
        update(BackgroundTask)
        .where(
            BackgroundTask.task_id == task.task_id,
            BackgroundTask.lease_token == task.lease_token,
            BackgroundTask.attempt_count == task.attempt_count,
            *_claim_filters(session, kinds, clock),
        )
        .values(
            status="processing",
            phase="claimed",
            progress=0,
            attempt_count=task.attempt_count + 1,
            lease_token=task.lease_token + 1,
            lease_owner=owner,
            lease_until=_after_seconds(session, lease_seconds),
            heartbeat_at=clock,
            started_at=func.coalesce(BackgroundTask.started_at, clock),
            updated_at=clock,
            next_run_at=None,
            completed_at=None,
            error_code=None,
            error_summary=None,
            result_ref=None,
            result_version=None,
        )
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        return False
    await session.refresh(task, with_for_update=True)
    session.add(
        TaskAttempt(
            task_id=task.task_id,
            attempt_no=task.attempt_count,
            lease_owner=owner,
            lease_token=task.lease_token,
            status="running",
            phase=task.phase,
            started_at=task.heartbeat_at,
        )
    )
    return True


async def _owned_attempt(
    session: AsyncSession, lease: TaskLease
) -> tuple[BackgroundTask, TaskAttempt, datetime]:
    if not isinstance(lease, TaskLease):
        raise TypeError("lease 必须是 TaskLease")
    task = await session.scalar(
        select(BackgroundTask)
        .where(*_ownership_filters(lease))
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if task is None:
        raise TaskLeaseLost("任务租约已失效或所有权已变更")
    # 所有操作固定 task → attempt 的锁顺序，避免相反顺序引入死锁。
    attempt = await session.scalar(
        select(TaskAttempt)
        .where(
            TaskAttempt.task_id == lease.task_id,
            TaskAttempt.attempt_no == lease.attempt_no,
            TaskAttempt.lease_owner == lease.owner,
            TaskAttempt.lease_token == lease.token,
            TaskAttempt.status == "running",
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if attempt is None:
        raise TaskStateConflict("运行中尝试与任务租约不一致")
    now = await _database_now(session)
    if task.lease_until is None or _utc(task.lease_until) <= now:
        raise TaskLeaseLost("任务租约已到期，禁止续约或结算")
    return task, attempt, now


async def heartbeat_task(
    session: AsyncSession,
    lease: TaskLease,
    *,
    lease_seconds: int = 90,
    phase: str | None = None,
    progress: int | None = None,
) -> bool:
    """续约并返回取消请求；不能复活到期租约，也不表示执行已经停止。"""
    validate_lease_seconds(lease_seconds)
    if phase is not None:
        validate_code(phase, "执行阶段")
    if progress is not None:
        validate_count(progress, "进度", 100, minimum=0)
    task, attempt, _ = await _owned_attempt(session, lease)
    deadline = _after_seconds(session, lease_seconds)
    clock = _clock_expression(session)
    values = dict(
        lease_until=case(
            (
                _instant(session, BackgroundTask.lease_until)
                > _instant(session, deadline),
                BackgroundTask.lease_until,
            ),
            else_=deadline,
        ),
        heartbeat_at=clock,
        updated_at=clock,
    )
    if phase is not None:
        values["phase"] = phase
    if progress is not None:
        values["progress"] = progress
    await _fenced_update(session, task, lease, values)
    if phase is not None:
        attempt.phase = phase
    await session.flush()
    return task.cancel_requested_at is not None


async def _finish(
    session: AsyncSession,
    lease: TaskLease,
    task: BackgroundTask,
    attempt: TaskAttempt,
    *,
    status: str,
    attempt_status: str,
    values: dict | None = None,
) -> None:
    phase = task.phase
    clock = _clock_expression(session)
    changes = dict(
        status=status,
        phase=status,
        lease_owner=None,
        lease_until=None,
        next_run_at=None,
        completed_at=None if status == "retry_wait" else clock,
        updated_at=clock,
    )
    changes.update(values or {})
    await _fenced_update(session, task, lease, changes)
    attempt.status = attempt_status
    attempt.phase = phase
    attempt.finished_at = task.updated_at


async def succeed_task(
    session: AsyncSession,
    lease: TaskLease,
    *,
    result_ref: str,
    result_version: int = 1,
    result_summary: str | None = None,
) -> BackgroundTask:
    """结算真实成功；调用方须先完成业务发布，取消请求不伪造执行失败。"""
    validate_summary(result_ref, "结果引用")
    validate_count(result_version, "结果版本", MAX_ATTEMPT_COUNT)
    if result_summary is not None:
        validate_summary(result_summary, "结果摘要")
    task, attempt, _ = await _owned_attempt(session, lease)
    await _finish(
        session,
        lease,
        task,
        attempt,
        status="succeeded",
        attempt_status="succeeded",
        values=dict(
            progress=100,
            result_ref=result_ref,
            result_version=result_version,
            error_code=None,
            error_summary=None,
        ),
    )
    attempt.result_summary = result_summary
    await session.flush()
    return task


async def fail_task(
    session: AsyncSession,
    lease: TaskLease,
    *,
    error_code: str,
    error_summary: str,
    retryable: bool = False,
    retry_policy: RetryPolicy = RetryPolicy(),
) -> BackgroundTask:
    """只在真实执行已经结束后结算失败；取消请求或预算耗尽均禁止重试。"""
    validate_code(error_code, "错误码")
    validate_summary(error_summary, "错误摘要")
    if not isinstance(retryable, bool):
        raise TypeError("retryable 必须是布尔值")
    if not isinstance(retry_policy, RetryPolicy):
        raise TypeError("retry_policy 必须是 RetryPolicy")
    task, attempt, _ = await _owned_attempt(session, lease)
    retry = (
        retryable
        and task.attempt_count < task.max_attempts
        and task.cancel_requested_at is None
    )
    values = dict(error_code=error_code, error_summary=error_summary)
    if retry:
        values["next_run_at"] = _after_seconds(
            session, retry_policy.delay_seconds(lease.attempt_no)
        )
    await _finish(
        session,
        lease,
        task,
        attempt,
        status="retry_wait" if retry else "failed",
        attempt_status="failed",
        values=values,
    )
    attempt.error_code = error_code
    attempt.error_summary = error_summary
    await session.flush()
    return task


async def acknowledge_cancellation(
    session: AsyncSession, lease: TaskLease
) -> BackgroundTask:
    """仅供已经确认真实执行停止的调用方确认取消，不能用于取消等待者。"""
    task, attempt, _ = await _owned_attempt(session, lease)
    if task.cancel_requested_at is None:
        raise TaskStateConflict("任务尚未请求取消")
    await _finish(
        session, lease, task, attempt, status="cancelled", attempt_status="cancelled"
    )
    await session.flush()
    return task


async def list_expired_leases(
    session: AsyncSession, *, kinds: tuple[str, ...], limit: int = 20
) -> list[TaskLease]:
    """只读发现，不改变尝试或移交写权；恢复须等待存储所有权与停止证明。"""
    validate_task_kinds(kinds)
    validate_count(limit, "过期发现页大小", 100)
    now = await _database_now(session)
    tasks = await session.scalars(
        select(BackgroundTask)
        .where(
            BackgroundTask.kind.in_(kinds),
            BackgroundTask.status == "processing",
            _instant(session, BackgroundTask.lease_until) <= _instant(session, now),
        )
        .order_by(BackgroundTask.lease_until, BackgroundTask.task_id)
        .limit(limit)
        .execution_options(populate_existing=True)
    )
    return [_receipt(task) for task in tasks]
