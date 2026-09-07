"""持久任务的稳定枚举与输入契约。"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Literal

from app.indexing.contracts import (
    validate_blob_id,
    validate_document_id,
    validate_document_revision,
)

TaskKind = Literal["knowledge.index", "note.index", "note.delete"]
TaskStatus = Literal[
    "pending",
    "processing",
    "retry_wait",
    "succeeded",
    "failed",
    "superseded",
    "cancelled",
]
AttemptStatus = Literal["running", "succeeded", "failed", "abandoned", "cancelled"]

TASK_KINDS = frozenset(("knowledge.index", "note.index", "note.delete"))
TASK_STATUSES = frozenset(
    (
        "pending",
        "processing",
        "retry_wait",
        "succeeded",
        "failed",
        "superseded",
        "cancelled",
    )
)
ATTEMPT_STATUSES = frozenset(
    ("running", "succeeded", "failed", "abandoned", "cancelled")
)
TERMINAL_TASK_STATUSES = frozenset(("succeeded", "failed", "superseded", "cancelled"))


@dataclass(frozen=True, slots=True)
class TaskSubmission:
    """事务内登记的任务意图；输入能否重放由提交入口明确保证。"""

    user_id: str
    kind: TaskKind
    idempotency_key: str
    input_fingerprint: str
    input_ref: str
    resource_id: str | None = None
    target_generation: int | None = None
    input_blob_id: str | None = None
    input_schema_version: int = 1
    input_metadata: dict[str, object] | None = None
    max_attempts: int = 3
    retry_of_task_id: str | None = None


def _ascii_token(value: str, label: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} 必须是字符串")
    if not 1 <= len(value) <= max_length or any(
        not "!" <= char <= "~" for char in value
    ):
        raise ValueError(f"{label} 必须是 1 到 {max_length} 位可见 ASCII（不含空格）")
    return value


def validate_task_user_id(user_id: str) -> str:
    """限制认证主体的存储边界，不改写其大小写或内容。"""
    return _ascii_token(user_id, "user_id", 64)


def validate_task_key(idempotency_key: str) -> str:
    """拒绝空白和尾空格，避免 MySQL PAD SPACE 排序规则折叠不同键。"""
    return _ascii_token(idempotency_key, "幂等键", 128)


def _validate_json_types(value: object) -> None:
    if value is None or isinstance(value, (str, bool, float)):
        return
    if isinstance(value, int):
        # MySQL JSON 越界整数可能转为 DOUBLE，破坏值和幂等比较中的类型。
        if not -(2**63) <= value <= 2**64 - 1:
            raise ValueError("任务 metadata 整数必须在 [-2**63, 2**64-1] 范围内")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("任务 metadata 的键必须是字符串")
            _validate_json_types(item)
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_types(item)
        return
    raise TypeError(f"任务 metadata 包含非 JSON 类型：{type(value).__name__}")


def canonical_task_metadata(metadata: dict[str, object] | None) -> str:
    """严格 JSON 编码；键顺序无关，保留布尔/整数/浮点和业务文本的区别。"""
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("任务 metadata 必须是 JSON 对象或 None")
    # 编码器先拒绝循环容器和非有限数，再禁止其默许的键/tuple 隐式转换。
    encoded = json.dumps(
        metadata,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    _validate_json_types(metadata)
    encoded.encode("utf-8")
    return encoded


def _positive_count(value: int, label: str) -> int:
    try:
        return validate_document_revision(value)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{label} 必须是 1 到 4294967295 的整数（不含布尔值）") from exc


def _optional_uuid(value: str | None) -> str | None:
    return None if value is None else validate_document_id(value)


def snapshot_submission(submission: TaskSubmission) -> TaskSubmission:
    """在首个数据库 await 前校验所有输入，并解除嵌套 JSON 与调用方的别名。"""
    if not isinstance(submission, TaskSubmission):
        raise TypeError("submission 必须是 TaskSubmission")
    if not isinstance(submission.kind, str) or submission.kind not in TASK_KINDS:
        raise ValueError("任务类型不受支持")
    if not isinstance(submission.input_ref, str):
        raise TypeError("input_ref 必须是字符串")
    if len(submission.input_ref) > 1024:
        raise ValueError("input_ref 不能超过 1024 个字符")
    submission.input_ref.encode("utf-8")
    if not submission.input_ref.strip() and submission.input_blob_id is None:
        raise ValueError("必须提供不可变输入引用或原始 blob")
    return replace(
        submission,
        user_id=validate_task_user_id(submission.user_id),
        idempotency_key=validate_task_key(submission.idempotency_key),
        input_fingerprint=validate_blob_id(submission.input_fingerprint),
        input_blob_id=None
        if submission.input_blob_id is None
        else validate_blob_id(submission.input_blob_id),
        resource_id=_optional_uuid(submission.resource_id),
        retry_of_task_id=_optional_uuid(submission.retry_of_task_id),
        input_schema_version=_positive_count(
            submission.input_schema_version, "输入 schema 版本"
        ),
        target_generation=None
        if submission.target_generation is None
        else _positive_count(submission.target_generation, "目标代次"),
        max_attempts=_positive_count(submission.max_attempts, "最大尝试数"),
        input_metadata=json.loads(canonical_task_metadata(submission.input_metadata)),
    )
