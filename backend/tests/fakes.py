from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable


@dataclass(frozen=True)
class FakeMessage:
    content: str


class DeterministicEmbeddings:
    """离线且可重复的嵌入替身，不表达真实语义质量。"""

    def __init__(self, dimension: int = 8) -> None:
        if dimension <= 0:
            raise ValueError("dimension 必须大于 0")
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def _embed(self, text: str) -> list[float]:
        digest = sha256(text.encode("utf-8")).digest()
        values = [digest[index % len(digest)] / 255 for index in range(self.dimension)]
        return values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls.append([text])
        return self._embed(text)


class FakeChatModel:
    """按顺序返回固定响应，并记录调用输入。"""

    def __init__(self, responses: Iterable[str] = ("fake-response",)) -> None:
        self._responses = list(responses)
        self.calls: list[object] = []

    def _next(self) -> FakeMessage:
        if not self._responses:
            raise AssertionError("FakeChatModel 没有剩余响应")
        return FakeMessage(self._responses.pop(0))

    def invoke(self, value: object) -> FakeMessage:
        self.calls.append(value)
        return self._next()

    async def ainvoke(self, value: object) -> FakeMessage:
        self.calls.append(value)
        return self._next()


class FakeReranker:
    """使用显式分数表排序，避免测试依赖本地模型。"""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def rank(self, query: str, documents: list[str]) -> list[dict[str, object]]:
        self.calls.append((query, list(documents)))
        ranked = sorted(
            documents, key=lambda document: self.scores[document], reverse=True
        )
        return [
            {"document": document, "score": self.scores[document], "rank": rank}
            for rank, document in enumerate(ranked, start=1)
        ]
