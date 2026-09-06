"""任务状态操作的领域异常。"""


class TaskError(RuntimeError):
    """任务领域异常基类。"""


class TaskNotFound(TaskError):
    """任务不存在，或不属于当前用户。"""


class TaskIdempotencyConflict(TaskError):
    """幂等键已被不同输入占用。"""


class TaskStateConflict(TaskError):
    """请求与任务当前状态不兼容。"""


class TaskLeaseLost(TaskStateConflict):
    """执行凭据陈旧或租约到期，旧执行不得续约或结算。"""
