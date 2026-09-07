"""模型工厂的冷导入、并发懒加载和 provider 契约；所有上游均为内存替身。"""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys
from threading import Barrier, Event
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


MODEL_FACTORIES = [
    ("chat", "ChatModelFactory"),
    ("embed", "EmbedModelFactory"),
    ("vision", "VisionModelFactory"),
]
CONFIG_KEYS = (
    "LLM_TYPE", "EMBED_MODEL_TYPE", "VISION_MODEL_TYPE",
    "ALIYUN_ACCESS_KEY_SECRET", "ALIYUN_BASE_URL", "ALIYUN_MODEL_NAME",
    "CHAT_MODEL_NAME", "VISION_CHAT_MODEL_NAME", "ALIYUN_EMBED_MODEL_NAME",
    "OLLAMA_MODEL_NAME", "OLLAMA_CHAT_MODEL_NAME", "TEXT_EMBEDDING_MODEL_NAME",
    "VISION_OLLAMA_MODEL_NAME", "OLLAMA_BASE_URL",
)


@pytest.fixture
def isolated_factory(monkeypatch):
    """以独立模块执行源码，不加载真实 app、dotenv 或模型依赖。"""
    for name in CONFIG_KEYS:
        monkeypatch.delenv(name, raising=False)
    clients = {
        name: Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        for name in ("ChatOpenAI", "ChatOllama", "OllamaEmbeddings")
    }
    load_dotenv = Mock(return_value=False)
    logger = SimpleNamespace(info=Mock())
    embedding_call = Mock(return_value=SimpleNamespace(
        status_code=200, output={"embeddings": [{"embedding": [1, 2]}]},
    ))
    upstreams = {
        "app": {"__path__": []},
        "app.core": {"__path__": []},
        "app.core.logger_handler": {"logger": logger},
        "dotenv": {"load_dotenv": load_dotenv},
        "langchain_core": {"__path__": []},
        "langchain_core.embeddings": {"Embeddings": type("Embeddings", (), {})},
        "langchain_core.language_models": {
            "BaseChatModel": type("BaseChatModel", (), {}),
        },
        "langchain_ollama": {
            name: clients[name] for name in ("ChatOllama", "OllamaEmbeddings")
        },
        "langchain_openai": {"ChatOpenAI": clients["ChatOpenAI"]},
        "dashscope": {"TextEmbedding": SimpleNamespace(call=embedding_call)},
    }
    for name, attributes in upstreams.items():
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)

    path = Path(__file__).resolve().parents[2] / "app/utils/factory.py"
    spec = importlib.util.spec_from_file_location("_m5_model_factory", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return SimpleNamespace(
        module=module, clients=clients, load_dotenv=load_dotenv, logger=logger,
        dashscope=sys.modules["dashscope"],
    )


def _replace_factory(monkeypatch, module, factory_name, generator):
    factory = Mock(return_value=SimpleNamespace(generator=generator))
    monkeypatch.setattr(module, factory_name, factory)
    return factory


def test_import_does_not_construct_models(isolated_factory):
    assert {
        name: client.call_count for name, client in isolated_factory.clients.items()
    } == {"ChatOpenAI": 0, "ChatOllama": 0, "OllamaEmbeddings": 0}
    isolated_factory.logger.info.assert_not_called()


def test_import_does_not_load_dotenv(isolated_factory):
    isolated_factory.load_dotenv.assert_not_called()


def test_no_implicit_legacy_model_exports(isolated_factory):
    module = isolated_factory.module
    assert {"chat_model", "embed_model", "vision_model", "__getattr__"}.isdisjoint(
        vars(module),
    )
    assert module.reranker_model is None
    assert module.RerankerModelFactory().generator() is None


@pytest.mark.parametrize("kind,factory_name", MODEL_FACTORIES)
def test_getter_caches_one_instance(isolated_factory, monkeypatch, kind, factory_name):
    module = isolated_factory.module
    instance = object()
    generator = Mock(return_value=instance)
    factory = _replace_factory(monkeypatch, module, factory_name, generator)
    getter = getattr(module, f"get_{kind}_model")

    assert getter() is instance
    assert getter() is instance
    factory.assert_called_once_with()
    generator.assert_called_once_with()
    for client in isolated_factory.clients.values():
        client.assert_not_called()


def test_model_types_have_independent_caches(isolated_factory, monkeypatch):
    module = isolated_factory.module
    instances = {kind: object() for kind, _ in MODEL_FACTORIES}
    for kind, factory_name in MODEL_FACTORIES:
        _replace_factory(
            monkeypatch, module, factory_name, Mock(return_value=instances[kind]),
        )

    for _ in range(2):
        for kind, _ in MODEL_FACTORIES:
            assert getattr(module, f"get_{kind}_model")() is instances[kind]


@pytest.mark.parametrize("kind,factory_name", MODEL_FACTORIES)
def test_concurrent_getter_constructs_once(
    isolated_factory, monkeypatch, kind, factory_name,
):
    module = isolated_factory.module
    workers = 8
    start = Barrier(workers + 1, timeout=5)
    entered = Event()
    release = Event()

    def construct():
        entered.set()
        assert release.wait(5), "测试必须释放构造线程"
        return object()

    generator = Mock(side_effect=construct)
    factory = _replace_factory(monkeypatch, module, factory_name, generator)
    getter = getattr(module, f"get_{kind}_model")

    def invoke():
        start.wait()
        return getter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(invoke) for _ in range(workers)]
        try:
            start.wait()
            assert entered.wait(5), "构造必须在调用 getter 后开始"
        finally:
            release.set()
        instances = [future.result(timeout=5) for future in futures]

    assert all(instance is instances[0] for instance in instances)
    factory.assert_called_once_with()
    generator.assert_called_once_with()


@pytest.mark.parametrize("kind,factory_name", MODEL_FACTORIES)
@pytest.mark.parametrize("failure_stage", ["factory", "generator"])
def test_failed_initialization_can_retry(
    isolated_factory, monkeypatch, kind, factory_name, failure_stage,
):
    module = isolated_factory.module
    failure = RuntimeError("合成初始化失败")
    instance = object()
    generator = Mock(return_value=instance)
    factory = _replace_factory(monkeypatch, module, factory_name, generator)
    target = factory if failure_stage == "factory" else generator
    successful_result = factory.return_value if failure_stage == "factory" else instance
    target.side_effect = [failure, successful_result]
    getter = getattr(module, f"get_{kind}_model")

    with pytest.raises(RuntimeError) as caught:
        getter()
    assert caught.value is failure
    assert getter() is instance
    assert getter() is instance
    assert factory.call_count == 2
    assert generator.call_count == (1 if failure_stage == "factory" else 2)


@pytest.mark.parametrize("kind,provider,model_key,client_name", [
    ("chat", "ALIYUN", "ALIYUN_MODEL_NAME", "ChatOpenAI"),
    ("chat", "OLLAMA", "OLLAMA_MODEL_NAME", "ChatOllama"),
    ("vision", "ALIYUN", "VISION_CHAT_MODEL_NAME", "ChatOpenAI"),
    ("vision", "OLLAMA", "VISION_OLLAMA_MODEL_NAME", "ChatOllama"),
])
def test_chat_and_vision_read_current_provider_configuration(
    isolated_factory, monkeypatch, kind, provider, model_key, client_name,
):
    provider_key = "LLM_TYPE" if kind == "chat" else "VISION_MODEL_TYPE"
    monkeypatch.setenv(provider_key, provider.lower())
    monkeypatch.setenv(model_key, "synthetic-model")
    monkeypatch.setenv("ALIYUN_ACCESS_KEY_SECRET", "offline-key")
    monkeypatch.setenv("ALIYUN_BASE_URL", "https://aliyun.invalid/v1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.invalid:11434")
    expected = dict(model="synthetic-model", streaming=kind == "chat", top_p=0.7)
    if provider == "ALIYUN":
        expected.update(api_key="offline-key", base_url="https://aliyun.invalid/v1")
    else:
        expected.update(base_url="http://ollama.invalid:11434")

    getter = getattr(isolated_factory.module, f"get_{kind}_model")
    model = getter()
    isolated_factory.clients[client_name].assert_called_once_with(**expected)
    assert vars(model) == expected
    monkeypatch.setenv(model_key, "must-not-replace-cached-model")
    assert getter() is model


@pytest.mark.parametrize("provider,model_key", [
    ("ALIYUN", "ALIYUN_EMBED_MODEL_NAME"),
    ("OLLAMA", "TEXT_EMBEDDING_MODEL_NAME"),
])
def test_embed_getter_preserves_provider_and_validation(
    isolated_factory, monkeypatch, provider, model_key,
):
    module = isolated_factory.module
    monkeypatch.setenv("EMBED_MODEL_TYPE", provider.lower())
    monkeypatch.setenv(model_key, "synthetic-embedding")
    monkeypatch.setenv("ALIYUN_ACCESS_KEY_SECRET", "offline-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.invalid:11434")
    model = module.get_embed_model()

    assert isinstance(model, module.ValidatedEmbeddings)
    assert model.provider == provider
    if provider == "OLLAMA":
        isolated_factory.clients["OllamaEmbeddings"].assert_called_once_with(
            model="synthetic-embedding", base_url="http://ollama.invalid:11434",
        )
    else:
        assert isinstance(model.delegate, module.DashScopeEmbeddingsWrapper)
        assert model.delegate.model_name == "synthetic-embedding"
        assert isolated_factory.dashscope.api_key == "offline-key"
        assert model.embed_query("合成查询") == [1.0, 2.0]
        isolated_factory.dashscope.TextEmbedding.call.assert_called_once_with(
            model="synthetic-embedding", input="合成查询",
        )
    _assert_embedding_validation(module, model, provider)
    assert module.get_embed_model() is model


def _assert_embedding_validation(module, model, provider):
    """验证 getter 保留向量校验及错误的 provider、cause 和可重试标记。"""
    model.delegate.embed_query = Mock(return_value=[1, 2])
    assert model.embed_query("合成查询") == [1.0, 2.0]
    model.delegate.embed_query.return_value = []
    with pytest.raises(module.EmbeddingServiceError) as invalid:
        model.embed_query("合成查询")
    assert invalid.value.provider == provider
    failure = TimeoutError("合成超时")
    model.delegate.embed_query.side_effect = failure
    with pytest.raises(module.EmbeddingServiceError) as failed:
        model.embed_query("合成查询")
    assert failed.value.provider == provider
    assert failed.value.__cause__ is failure
    assert failed.value.retryable is True
    model.delegate.embed_documents = Mock(return_value=[[1, 2]])
    with pytest.raises(module.EmbeddingServiceError, match="返回数量与输入不一致"):
        model.embed_documents(["合成甲", "合成乙"])


@pytest.mark.parametrize("kind,provider_key", [
    ("chat", "LLM_TYPE"),
    ("embed", "EMBED_MODEL_TYPE"),
    ("vision", "VISION_MODEL_TYPE"),
])
def test_invalid_provider_can_be_corrected_after_import(
    isolated_factory, monkeypatch, kind, provider_key,
):
    monkeypatch.setenv(provider_key, "unsupported")
    getter = getattr(isolated_factory.module, f"get_{kind}_model")
    with pytest.raises(ValueError, match=provider_key):
        getter()
    for client in isolated_factory.clients.values():
        client.assert_not_called()
    monkeypatch.setenv(provider_key, "ollama")
    model = getter()
    assert getter() is model


@pytest.mark.parametrize("base_url", [None, "", "https://gateway.invalid/v1"])
@pytest.mark.parametrize("streaming", [True, False])
def test_build_aliyun_chat_model_keeps_endpoint_and_options(
    isolated_factory, monkeypatch, base_url, streaming,
):
    module = isolated_factory.module
    monkeypatch.setenv("ALIYUN_ACCESS_KEY_SECRET", "offline-key")
    if base_url is not None:
        monkeypatch.setenv("ALIYUN_BASE_URL", base_url)
    model = module.build_aliyun_chat_model(
        model_name="synthetic-chat", streaming=streaming, top_p=0.35,
    )
    expected = dict(
        model="synthetic-chat", api_key="offline-key", streaming=streaming,
        base_url=base_url or module.ALIYUN_COMPATIBLE_BASE_URL, top_p=0.35,
    )
    isolated_factory.clients["ChatOpenAI"].assert_called_once_with(**expected)
    assert vars(model) == expected


@pytest.mark.parametrize("kind,settings,client_name,model_name", [
    ("chat", {}, "ChatOpenAI", "qwen3-max"),
    ("chat", {"CHAT_MODEL_NAME": "legacy-chat"}, "ChatOpenAI", "legacy-chat"),
    ("chat", {"LLM_TYPE": "ollama", "OLLAMA_CHAT_MODEL_NAME": "legacy-ollama"},
     "ChatOllama", "legacy-ollama"),
    ("vision", {}, "ChatOpenAI", "qwen3-max"),
    ("vision", {"CHAT_MODEL_NAME": "fallback-vision"}, "ChatOpenAI", "fallback-vision"),
    ("vision", {"LLM_TYPE": "ollama"}, "ChatOllama", "qwen-vl:7b"),
    ("vision", {"LLM_TYPE": "ollama", "OLLAMA_MODEL_NAME": "fallback-ollama"},
     "ChatOllama", "fallback-ollama"),
])
def test_chat_and_vision_keep_configuration_fallbacks(
    isolated_factory, monkeypatch, kind, settings, client_name, model_name,
):
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    model = getattr(isolated_factory.module, f"get_{kind}_model")()
    assert model.model == model_name
    assert model.streaming is (kind == "chat")
    assert isolated_factory.clients[client_name].call_count == 1
