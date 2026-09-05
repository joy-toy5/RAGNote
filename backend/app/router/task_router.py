"""持久任务的用户隔离查询、取消与幂等人工重试，不负责执行任务。"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.success_response import success_response
from app.db.db_config import get_db
from app.schemas.task import TaskIdempotencyKey, TaskListResponse, TaskResponse, TaskRetryRequest
from app.tasking import repository
from app.tasking.errors import TaskIdempotencyConflict, TaskNotFound, TaskStateConflict
from app.utils.auth_utils import get_current_user_id


task_router = APIRouter(prefix="/tasks", tags=["tasks"])


@contextmanager
def _task_errors() -> Iterator[None]:
    """只转换已知异常；固定文案不透传 SQL、输入引用或跨用户存在性。"""
    try:
        yield
    except TaskNotFound as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    except (TaskIdempotencyConflict, TaskStateConflict) as exc:
        raise HTTPException(status_code=409, detail="任务状态或幂等键与请求冲突") from exc
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="任务存储暂不可用，请稍后重试") from exc


async def _commit_task_response(db: AsyncSession, task_id: str, user_id: str) -> JSONResponse:
    # flush 后的服务端时间可能已过期；按用户重读，禁止同步懒加载或无作用域 refresh。
    task = await repository.get_task(db, task_id, user_id)
    data = TaskResponse.model_validate(task)
    # JSON 编码也可能失败；完整响应先构造，保证此类失败仍可回滚写入。
    response = success_response(data=data)
    await db.commit()
    return response


@task_router.get("")
async def list_tasks(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(repository.DEFAULT_TASK_PAGE_SIZE, ge=1, le=repository.MAX_TASK_PAGE_SIZE),
    offset: int = Query(0, ge=0),
    idempotency_key: Annotated[TaskIdempotencyKey | None, Query()] = None,
) -> JSONResponse:
    """仅查询当前用户的任务；支持分页和按幂等键找回丢失的任务 ID。"""
    with _task_errors():
        tasks = await repository.list_tasks(
            db, user_id, limit=limit, offset=offset, idempotency_key=idempotency_key,
        )
        data = TaskListResponse(
            tasks=[TaskResponse.model_validate(task) for task in tasks], limit=limit, offset=offset,
        )
        return success_response(data=data)


@task_router.get("/{task_id}")
async def get_task(
    task_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """不存在与不属于当前用户的任务统一返回 404。"""
    with _task_errors():
        task = await repository.get_task(db, task_id, user_id)
        return success_response(data=TaskResponse.model_validate(task))


@task_router.post("/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """提交取消请求；处理中任务是否已停止由 repository 状态事实决定。"""
    with _task_errors():
        task = await repository.request_cancel(db, task_id, user_id)
        return await _commit_task_response(db, task.task_id, user_id)


@task_router.post("/{task_id}/retry")
async def retry_task(
    task_id: str,
    payload: TaskRetryRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """提交已有任务的人工重试；重发使用相同客户端幂等键，不随机新建意图。"""
    with _task_errors():
        task = await repository.retry_task(
            db, task_id, user_id, idempotency_key=payload.idempotency_key,
        )
        return await _commit_task_response(db, task.task_id, user_id)
