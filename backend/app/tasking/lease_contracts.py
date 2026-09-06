"""执行租约的不可变凭据、存储边界与有限退避；不创建运行时资源。"""

from __future__ import annotations

from dataclasses import dataclass

from app.indexing.contracts import validate_document_id
from app.tasking.contracts import TASK_KINDS

MAX_ATTEMPT_COUNT = 2**32 - 1
MAX_LEASE_TOKEN = 2**64 - 1


def validate_count(value: int, label: str, maximum: int, *, minimum: int = 1) -> int:
    """布尔不是计数；在数据库 await 前拒绝列边界外的输入。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} 必须是整数")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} 必须在 {minimum} 到 {maximum} 之间")
    return value


def _uuid(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} 必须是 UUID 字符串")
    try:
        return validate_document_id(value)
    except ValueError as exc:
        raise ValueError(f"{label} 必须是规范的非零 RFC 4122 UUID") from exc


def validate_lease_owner(owner: str) -> str:
    """所有者是每次 Worker 启动生成的 UUID，不是可复用 PID。"""
    return _uuid(owner, "lease_owner")


def validate_lease_seconds(lease_seconds: int) -> int:
    return validate_count(lease_seconds, "租约秒数", 3600)


def validate_task_kinds(kinds: tuple[str, ...]) -> tuple[str, ...]:
    """仅领取调用方确实支持的种类，不隐式认领所有任务。"""
    if not isinstance(kinds, tuple):
        raise TypeError("任务种类必须是不可变 tuple")
    if not kinds or any(
        not isinstance(kind, str) or kind not in TASK_KINDS for kind in kinds
    ):
        raise ValueError("任务种类必须非空且受支持")
    if len(set(kinds)) != len(kinds):
        raise ValueError("任务种类不能重复")
    return kinds


def validate_code(value: str, label: str) -> str:
    """机器阶段和错误码只接受可见 ASCII，避免排序规则折叠空白。"""
    if not isinstance(value, str):
        raise TypeError(f"{label} 必须是字符串")
    if not 1 <= len(value) <= 64 or any(not "!" <= char <= "~" for char in value):
        raise ValueError(f"{label} 必须是 1 到 64 位可见 ASCII（不含空格）")
    return value


def validate_summary(value: str, label: str) -> str:
    """只校验存储格式；调用方仍须先脱敏，不得直接传入原始异常或输入。"""
    if not isinstance(value, str):
        raise TypeError(f"{label} 必须是字符串")
    if not value.strip() or len(value) > 1024:
        raise ValueError(f"{label} 必须非空且不超过 1024 个字符")
    value.encode("utf-8")
    return value


@dataclass(frozen=True, slots=True)
class TaskLease:
    """一次已领取尝试的身份；返回凭据不等于领取事务已提交。"""

    task_id: str
    owner: str
    token: int
    attempt_no: int

    def __post_init__(self) -> None:
        _uuid(self.task_id, "task_id")
        validate_lease_owner(self.owner)
        validate_count(self.token, "lease_token", MAX_LEASE_TOKEN)
        validate_count(self.attempt_no, "attempt_no", MAX_ATTEMPT_COUNT)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """单次已停止执行的有限指数退避；总次数由任务 max_attempts 控制。"""

    base_seconds: int = 5
    max_seconds: int = 300

    def __post_init__(self) -> None:
        validate_count(self.base_seconds, "初始退避秒数", 86400)
        validate_count(self.max_seconds, "最大退避秒数", 86400)
        if self.max_seconds < self.base_seconds:
            raise ValueError("最大退避不能小于初始退避")

    def delay_seconds(self, attempt_no: int) -> int:
        validate_count(attempt_no, "attempt_no", MAX_ATTEMPT_COUNT)
        # 先截断指数，避免异常大尝试数导致巨大整数分配。
        exponent = min(
            attempt_no - 1, (self.max_seconds // self.base_seconds).bit_length()
        )
        return min(self.max_seconds, self.base_seconds * (1 << exponent))
