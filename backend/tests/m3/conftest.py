"""构造期模型替身显式按用例启用，不影响provider本身的离线契约测试。"""
from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda


@pytest.fixture
def isolated_rag_model(monkeypatch: pytest.MonkeyPatch):
    """只允许构造真实编排；各用例必须自行替换会实际运行的生成链。"""
    import app.rag.rag_service as module

    calls = []

    def unexpected_model_call(value):
        calls.append(value)
        raise AssertionError("本组离线用例不应调用构造期模型替身")

    model = RunnableLambda(unexpected_model_call)
    monkeypatch.setattr(module, "get_chat_model", lambda: model)
    yield
    assert calls == [], "被业务异常处理捕获的意外模型调用也必须失败"
