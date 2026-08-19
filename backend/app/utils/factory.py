from abc import ABC, abstractmethod
import math
from numbers import Real
from typing import Optional, List
import os
from dotenv import load_dotenv

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import OllamaEmbeddings, ChatOllama

from app.core.logger_handler import logger

# 加载环境变量
load_dotenv()


class EmbeddingServiceError(RuntimeError):
    """Embedding 调用失败或返回不满足向量契约。"""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable if retryable is not None else (
            status_code == 429 or (status_code is not None and status_code >= 500)
        )


class ValidatedEmbeddings(Embeddings):
    """拒绝空、非有限或数量不匹配向量的 provider 适配器。"""

    def __init__(self, delegate: Embeddings, provider: str) -> None:
        self.delegate = delegate
        self.provider = provider

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            vectors = self.delegate.embed_documents(texts)
        except EmbeddingServiceError:
            raise
        except Exception as exc:
            raise EmbeddingServiceError(
                "Embedding 批量调用失败",
                provider=self.provider,
                retryable=isinstance(exc, (TimeoutError, ConnectionError)),
            ) from exc
        if not isinstance(vectors, list):
            raise EmbeddingServiceError(
                "Embedding 批量返回类型无效",
                provider=self.provider,
            )
        if len(vectors) != len(texts):
            raise EmbeddingServiceError(
                "Embedding 返回数量与输入不一致",
                provider=self.provider,
            )
        validated = [self._validate_vector(vector) for vector in vectors]
        if len({len(vector) for vector in validated}) > 1:
            raise EmbeddingServiceError(
                "Embedding 批次向量维度不一致",
                provider=self.provider,
            )
        return validated

    def embed_query(self, text: str) -> List[float]:
        try:
            vector = self.delegate.embed_query(text)
        except EmbeddingServiceError:
            raise
        except Exception as exc:
            raise EmbeddingServiceError(
                "Embedding 查询调用失败",
                provider=self.provider,
                retryable=isinstance(exc, (TimeoutError, ConnectionError)),
            ) from exc
        return self._validate_vector(vector)

    def _validate_vector(self, vector: object) -> List[float]:
        if not isinstance(vector, list) or not vector:
            raise EmbeddingServiceError(
                "Embedding 返回空向量或非法类型",
                provider=self.provider,
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in vector
        ):
            raise EmbeddingServiceError(
                "Embedding 向量包含非有限或非数值元素",
                provider=self.provider,
            )
        return [float(value) for value in vector]


class DashScopeEmbeddingsWrapper(Embeddings):
    """阿里云DashScope嵌入模型封装"""
    
    def __init__(self, model_name: str = "qwen3-embedding", api_key: str = None):
        try:
            import dashscope
            self.dashscope = dashscope
            self.dashscope.api_key = api_key or os.getenv("ALIYUN_ACCESS_KEY_SECRET")
            self.model_name = model_name
        except ImportError:
            raise ImportError("需要安装 dashscope 库: pip install dashscope")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        return [self._embed(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        try:
            response = self.dashscope.TextEmbedding.call(
                model=self.model_name,
                input=text,
            )
        except Exception as exc:
            raise EmbeddingServiceError(
                "DashScope Embedding 调用失败",
                provider="ALIYUN",
                retryable=isinstance(exc, (TimeoutError, ConnectionError)),
            ) from exc
        raw_status_code = getattr(response, "status_code", None)
        status_code = (
            raw_status_code
            if isinstance(raw_status_code, int) and not isinstance(raw_status_code, bool)
            else None
        )
        if status_code != 200:
            raise EmbeddingServiceError(
                f"DashScope Embedding 返回状态 {status_code}",
                provider="ALIYUN",
                status_code=status_code,
            )
        try:
            return response.output["embeddings"][0]["embedding"]
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise EmbeddingServiceError(
                "DashScope Embedding 响应结构无效",
                provider="ALIYUN",
                status_code=status_code,
            ) from exc


class BaseModelFactory(ABC):
    """基础模型工厂"""

    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """生成模型"""
        pass


class ChatModelFactory(BaseModelFactory):
    """聊天模型工厂 - 支持阿里云百炼和Ollama"""
    
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """根据LLM_TYPE生成对应的聊天模型"""
        llm_type = os.getenv("LLM_TYPE", "ALIYUN").upper()
        
        if llm_type == "OLLAMA":
            model_name = os.getenv("OLLAMA_MODEL_NAME", os.getenv("OLLAMA_CHAT_MODEL_NAME", "qwen3:7b"))
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
            logger.info(f"📦 ChatModel 使用Ollama模型: {model_name}, 地址: {base_url}")
            
            return ChatOllama(
                model=model_name,
                base_url=base_url,
                streaming=True,
                top_p=0.7,
            )
        
        elif llm_type == "ALIYUN":
            model_name = os.getenv("ALIYUN_MODEL_NAME", os.getenv("CHAT_MODEL_NAME", "qwen3-max"))
            api_key = os.getenv("ALIYUN_ACCESS_KEY_SECRET")
            base_url = os.getenv("ALIYUN_BASE_URL")
            
            logger.info(f"📦 ChatModel 使用阿里云百炼模型: {model_name}")
            
            return ChatTongyi(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                streaming=True,
                top_p=0.7,
            )
        
        else:
            raise ValueError(f"不支持的LLM_TYPE: {llm_type}，可选值: ALIYUN, OLLAMA")


class EmbedModelFactory(BaseModelFactory):
    """嵌入模型工厂 - 支持Ollama和阿里云百炼"""
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """根据EMBED_MODEL_TYPE生成对应的嵌入模型"""
        embed_type = os.getenv("EMBED_MODEL_TYPE", "OLLAMA").upper()
        
        if embed_type == "OLLAMA":
            model_name = os.getenv("TEXT_EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
            logger.info(f"📦 EmbedModel 使用Ollama嵌入模型: {model_name}, 地址: {base_url}")
            
            return ValidatedEmbeddings(
                OllamaEmbeddings(
                    model=model_name,
                    base_url=base_url,
                ),
                provider="OLLAMA",
            )
        
        elif embed_type == "ALIYUN":
            model_name = os.getenv("ALIYUN_EMBED_MODEL_NAME", "qwen3-embedding")
            api_key = os.getenv("ALIYUN_ACCESS_KEY_SECRET")
            
            logger.info(f"📦 EmbedModel 使用阿里云嵌入模型: {model_name}")
            
            return ValidatedEmbeddings(
                DashScopeEmbeddingsWrapper(
                    model_name=model_name,
                    api_key=api_key,
                ),
                provider="ALIYUN",
            )
        
        else:
            raise ValueError(f"不支持的EMBED_MODEL_TYPE: {embed_type}，可选值: OLLAMA, ALIYUN")


class VisionModelFactory(BaseModelFactory):
    """
    视觉模型工厂 - 支持阿里云百炼和Ollama多模态模型。
    用于 PDF 多模态加载场景：将 PDF 页面渲染为图片，然后调用视觉模型进行图片理解，
    提取纯文本提取难以获取的图表、表格、流程图等视觉信息。

    之所以单独为一个视觉模型工厂而不是复用 ChatModelFactory，是因为：
    1. ChatModel 使用 streaming=True（流式输出），而视觉模型只能用 streaming=False
       （图片理解不适合流式）
    2. 视觉模型可能有独立的模型配置（如 VISION_OLLAMA_MODEL_NAME 区分于 OLLAMA_MODEL_NAME）
    3. 部分用户可能希望视觉模型使用更大的参数量或专门的多模态模型（如 qwen-vl 系列）
    """

    def generator(self) -> Optional[BaseChatModel]:
        """根据VISION_MODEL_TYPE生成对应的视觉模型"""
        # 未设置 VISION_MODEL_TYPE 时，默认跟随 LLM_TYPE（保持向后兼容）
        vision_type = os.getenv("VISION_MODEL_TYPE", "").upper() or os.getenv("LLM_TYPE", "ALIYUN").upper()

        if vision_type == "OLLAMA":
            model_name = os.getenv("VISION_OLLAMA_MODEL_NAME") or os.getenv("OLLAMA_MODEL_NAME") or "qwen-vl:7b"
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

            logger.info(f"🎨 VisionModel 使用Ollama多模态模型: {model_name}, 地址: {base_url}")

            return ChatOllama(
                model=model_name,
                base_url=base_url,
                # 视觉模型禁用 streaming，因为图片理解需要在完整的上下文上做推理
                streaming=False,
                top_p=0.7,
            )

        elif vision_type == "ALIYUN":
            model_name = os.getenv("VISION_CHAT_MODEL_NAME") or os.getenv("CHAT_MODEL_NAME") or "qwen3-max"
            api_key = os.getenv("ALIYUN_ACCESS_KEY_SECRET")
            base_url = os.getenv("ALIYUN_BASE_URL")

            logger.info(f"🎨 VisionModel 使用阿里云百炼多模态模型: {model_name}")

            return ChatTongyi(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                streaming=False,
                top_p=0.7,
            )

        else:
            raise ValueError(f"不支持的VISION_MODEL_TYPE: {vision_type}，可选值: ALIYUN, OLLAMA")


class RerankerModelFactory(BaseModelFactory):
    """使用CrossEncoder模型"""
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """生成模型"""
        return None


chat_model = ChatModelFactory().generator()
embed_model = EmbedModelFactory().generator()
reranker_model = None
vision_model = VisionModelFactory().generator()
