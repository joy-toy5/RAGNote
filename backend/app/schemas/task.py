"""任务查询的公开白名单与人工重试请求契约。"""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.tasking.contracts import TaskKind, TaskStatus


TaskIdempotencyKey = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=128, pattern=r"^[!-~]+$"),
]


class TaskRetryRequest(BaseModel):
    """客户端为一次重试意图生成新键；同一意图重发时必须复用该键。"""

    model_config = ConfigDict(extra="forbid")

    idempotency_key: TaskIdempotencyKey


class TaskResponse(BaseModel):
    """只公开状态事实；错误字段必须由上游脱敏，不包含输入或执行凭据。"""

    model_config = ConfigDict(from_attributes=True)

    task_id: str
    kind: TaskKind
    status: TaskStatus
    resource_id: str | None = None
    target_generation: int | None = None
    progress: int = Field(ge=0, le=100)
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    next_run_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    error_code: str | None = None
    error_summary: str | None = None
    retry_of_task_id: str | None = None


class TaskListResponse(BaseModel):
    """当前页和请求边界；不把本页长度冒充任务总数。"""

    tasks: list[TaskResponse]
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
