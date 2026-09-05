"""生命周期用例的局部日志隔离，不修改应用或迁移的日志配置。"""

import logging

import pytest


@pytest.fixture(autouse=True)
def isolate_lifecycle_loggers(monkeypatch: pytest.MonkeyPatch) -> None:
    # 先执行的 Alembic 回归会 fileConfig，默认禁用 collection 时已有的 logger。
    # 仅在本用例内恢复本目录 logger，退出后由 monkeypatch 还原。
    for name in (
        "app.core.task_registry",
        "app.rag.upload_runtime",
        "m5-note-task-lifecycle",
        "m5-agent-lifecycle",
        "test-upload",
    ):
        logger = logging.getLogger(name)
        monkeypatch.setattr(logger, "disabled", False)
        monkeypatch.setattr(logger, "propagate", True)
