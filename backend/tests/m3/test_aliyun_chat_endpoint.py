"""`RAG-019`：阿里云百炼对话客户端的端点门禁。

要锁住的缺陷：`langchain-community` 0.4.1 的 `ChatTongyi` 没有 `base_url` 字段，
且 `model_config` 为 `extra='ignore'`，因此 `base_url=...` 被静默接受再丢弃——
无警告、无异常、`model_kwargs` 仍为空。后果是 `.env` 里配好的
`ALIYUN_BASE_URL=.../compatible-mode/v1` 在整条生产链路上无效，生产永远打原生
DashScope 端点，而 `CHAT_MODEL_NAME=qwen3.8-max` 只在兼容模式端点存在（原生端点
返回 400 `InvalidParameter: url error, please check url`，报错指向 URL 而不是模型）。

因此本文件断言三件事，缺一不可：
1. `base_url` 真的落到底层 HTTP 客户端上（不是只存在 LangChain 包装字段里）；
2. `ALIYUN_BASE_URL` 缺失时兜到兼容模式端点，**绝不能**落到 `api.openai.com`；
3. 三处调用点（ChatModel / VisionModel / Agent）共用同一个构造函数，
   且各自的 `streaming` 语义不回退。

全部离线：只构造客户端并读取其配置，不发请求。默认离线守卫仍然生效。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.utils.factory import (
    ALIYUN_COMPATIBLE_BASE_URL,
    ChatModelFactory,
    VisionModelFactory,
    build_aliyun_chat_model,
)

CUSTOM_BASE_URL = "https://gateway.internal.invalid/v1"


@pytest.fixture
def aliyun_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """给出确定的阿里云环境，避免测试读到真实 `.env` 的模型名或密钥。"""
    monkeypatch.setenv("ALIYUN_ACCESS_KEY_SECRET", "sk-test-not-a-real-key")
    monkeypatch.setenv("LLM_TYPE", "ALIYUN")
    monkeypatch.setenv("VISION_MODEL_TYPE", "ALIYUN")
    monkeypatch.delenv("ALIYUN_BASE_URL", raising=False)
    monkeypatch.delenv("ALIYUN_MODEL_NAME", raising=False)
    monkeypatch.delenv("VISION_CHAT_MODEL_NAME", raising=False)
    monkeypatch.setenv("CHAT_MODEL_NAME", "qwen3.8-max")
    return monkeypatch


def _effective_base_urls(model: object) -> set[str]:
    """底层 openai 客户端实际使用的 base_url（同步 + 异步），去掉尾斜杠。"""
    return {
        str(getattr(model, name).base_url).rstrip("/")
        for name in ("root_client", "root_async_client")
    }


def test_base_url_reaches_underlying_http_client(aliyun_env: pytest.MonkeyPatch) -> None:
    """`ALIYUN_BASE_URL` 必须落到真正发请求的客户端上，而不是只存在包装字段里。

    这是 `ChatTongyi` 缺陷的正面反证：断言同步与异步 openai 客户端的 base_url
    都等于环境变量的值。只断言 LangChain 字段是不够的——`ChatTongyi` 连字段都没有。
    """
    aliyun_env.setenv("ALIYUN_BASE_URL", CUSTOM_BASE_URL)

    model = build_aliyun_chat_model(model_name="qwen3.8-max", streaming=True)

    assert _effective_base_urls(model) == {CUSTOM_BASE_URL}
    assert str(model.openai_api_base).rstrip("/") == CUSTOM_BASE_URL


def test_missing_base_url_falls_back_to_compatible_mode(
    aliyun_env: pytest.MonkeyPatch,
) -> None:
    """`ALIYUN_BASE_URL` 未设置时兜到百炼兼容模式端点。

    没有这条兜底，`ChatOpenAI` 会用它自己的默认值打 `api.openai.com`——
    用阿里云密钥打 OpenAI，是比原缺陷更隐蔽的错误，所以显式断言 host。
    """
    del aliyun_env  # fixture 已删除 ALIYUN_BASE_URL

    model = build_aliyun_chat_model(model_name="qwen3.8-max", streaming=True)

    assert _effective_base_urls(model) == {ALIYUN_COMPATIBLE_BASE_URL}
    for base_url in _effective_base_urls(model):
        assert "api.openai.com" not in base_url


def test_empty_base_url_also_falls_back(aliyun_env: pytest.MonkeyPatch) -> None:
    """`ALIYUN_BASE_URL=`（空串）与未设置同等处理，同样不能落到 OpenAI。"""
    aliyun_env.setenv("ALIYUN_BASE_URL", "")

    model = build_aliyun_chat_model(model_name="qwen3.8-max", streaming=True)

    assert _effective_base_urls(model) == {ALIYUN_COMPATIBLE_BASE_URL}


def test_compatible_base_url_constant_points_at_dashscope() -> None:
    """兜底常量必须是百炼兼容模式端点本身，改错了这里其余断言都会被带偏。"""
    assert ALIYUN_COMPATIBLE_BASE_URL == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


@pytest.mark.parametrize("streaming", [True, False])
def test_streaming_and_top_p_are_preserved(
    aliyun_env: pytest.MonkeyPatch, streaming: bool
) -> None:
    """`streaming` 按调用点传入的值生效，`top_p` 保持切换客户端前的 0.7。"""
    del aliyun_env

    model = build_aliyun_chat_model(
        model_name="qwen3.8-max", streaming=streaming
    )

    assert model.streaming is streaming
    assert model.top_p == 0.7
    assert model.model_name == "qwen3.8-max"


def test_chat_model_factory_uses_compatible_endpoint(
    aliyun_env: pytest.MonkeyPatch,
) -> None:
    """ChatModel 调用点：端点与流式行为都要正确，模型名来自 `CHAT_MODEL_NAME`。"""
    del aliyun_env

    model = ChatModelFactory().generator()

    assert _effective_base_urls(model) == {ALIYUN_COMPATIBLE_BASE_URL}
    assert model.model_name == "qwen3.8-max"
    assert model.streaming is True


def test_vision_model_factory_uses_compatible_endpoint_without_streaming(
    aliyun_env: pytest.MonkeyPatch,
) -> None:
    """VisionModel 调用点：同一端点，但必须保持 `streaming=False`。

    视觉推理要在完整上下文上做，`streaming=True` 会改变行为，属于回退。
    """
    aliyun_env.setenv("VISION_CHAT_MODEL_NAME", "qwen3.8-max")

    model = VisionModelFactory().generator()

    assert _effective_base_urls(model) == {ALIYUN_COMPATIBLE_BASE_URL}
    assert model.streaming is False


def test_aliyun_base_url_switches_every_call_site(
    aliyun_env: pytest.MonkeyPatch,
) -> None:
    """改一个环境变量必须同时切走两个工厂的端点，证明没有旁路硬编码。"""
    aliyun_env.setenv("VISION_CHAT_MODEL_NAME", "qwen3.8-max")
    aliyun_env.setenv("ALIYUN_BASE_URL", CUSTOM_BASE_URL)

    chat_urls = _effective_base_urls(ChatModelFactory().generator())
    vision_urls = _effective_base_urls(VisionModelFactory().generator())

    assert chat_urls == {CUSTOM_BASE_URL}
    assert vision_urls == {CUSTOM_BASE_URL}


def test_chat_tongyi_still_drops_base_url() -> None:
    """记录缺陷成因本身：`ChatTongyi` 没有 `base_url` 字段且 `extra='ignore'`。

    这条是升级哨兵，不是要求 `ChatTongyi` 一直有缺陷：如果将来
    `langchain-community` 补上了 `base_url`，本测试会失败，提示回来重新评估
    `build_aliyun_chat_model` 是否还需要换客户端。
    """
    from langchain_community.chat_models.tongyi import ChatTongyi

    assert "base_url" not in ChatTongyi.model_fields
    assert ChatTongyi.model_config.get("extra") == "ignore"

    model = ChatTongyi(
        model="qwen3.8-max",
        api_key="sk-test-not-a-real-key",
        base_url=CUSTOM_BASE_URL,
        streaming=True,
        top_p=0.7,
    )

    assert not hasattr(model, "base_url")
    assert model.model_kwargs == {}


def _aliyun_chat_construction_sites(source: str) -> list[str]:
    """源码里构造阿里云对话客户端的调用名（AST，不靠字符串匹配注释）。"""
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = (
            function.id
            if isinstance(function, ast.Name)
            else function.attr
            if isinstance(function, ast.Attribute)
            else None
        )
        if name in {"ChatTongyi", "ChatOpenAI", "build_aliyun_chat_model"}:
            names.append(name)
    return names


@pytest.mark.parametrize(
    "relative_path",
    [
        "app/agent/agent.py",
        "app/utils/factory.py",
        "scripts/rag008_answer_eval.py",
        "scripts/rag008_branch_trace.py",
    ],
)
def test_no_call_site_builds_its_own_aliyun_client(
    backend_root: Path, relative_path: str
) -> None:
    """除工厂本体外，任何调用点都不得自建客户端。

    RAG-019 排查时发现的是三处调用点而不是两处；评测脚本还自建了第四处。
    只改其中几处、留下另几处，是这个缺陷最容易复发的形态，所以用 AST 门禁盯住。
    """
    source = (backend_root / relative_path).read_text(encoding="utf-8")
    constructions = _aliyun_chat_construction_sites(source)

    assert "ChatTongyi" not in constructions
    if relative_path == "app/utils/factory.py":
        # 工厂本体是唯一允许出现 ChatOpenAI 的地方，且只应出现一次。
        assert constructions.count("ChatOpenAI") == 1
    else:
        assert "ChatOpenAI" not in constructions
        assert "build_aliyun_chat_model" in constructions
