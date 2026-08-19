from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from app.utils.factory import (
    DashScopeEmbeddingsWrapper,
    EmbeddingServiceError,
    ValidatedEmbeddings,
)


class FakeEmbeddings:
    def __init__(self, *, documents: object = None, query: object = None) -> None:
        self.documents = documents
        self.query = query

    def embed_documents(self, texts: list[str]) -> object:
        del texts
        if isinstance(self.documents, Exception):
            raise self.documents
        return self.documents

    def embed_query(self, text: str) -> object:
        del text
        if isinstance(self.query, Exception):
            raise self.query
        return self.query


@pytest.mark.parametrize(
    "vectors",
    [
        [[]],
        [[math.nan]],
        [[math.inf]],
        [[True]],
        [["not-a-number"]],
    ],
)
def test_embedding_adapter_rejects_illegal_vectors(vectors: object) -> None:
    adapter = ValidatedEmbeddings(
        FakeEmbeddings(documents=vectors),  # type: ignore[arg-type]
        provider="TEST",
    )

    with pytest.raises(EmbeddingServiceError):
        adapter.embed_documents(["text"])


def test_embedding_adapter_rejects_partial_batch_and_classifies_exceptions() -> None:
    partial = ValidatedEmbeddings(
        FakeEmbeddings(documents=[[0.1, 0.2]]),  # type: ignore[arg-type]
        provider="TEST",
    )
    failed = ValidatedEmbeddings(
        FakeEmbeddings(query=TimeoutError("timeout")),  # type: ignore[arg-type]
        provider="TEST",
    )

    with pytest.raises(EmbeddingServiceError, match="数量"):
        partial.embed_documents(["first", "second"])
    with pytest.raises(EmbeddingServiceError, match="查询调用失败") as captured:
        failed.embed_query("query")
    assert captured.value.provider == "TEST"
    assert captured.value.retryable


def test_embedding_adapter_wraps_invalid_batch_container() -> None:
    adapter = ValidatedEmbeddings(
        FakeEmbeddings(documents=None),  # type: ignore[arg-type]
        provider="TEST",
    )

    with pytest.raises(EmbeddingServiceError, match="返回类型"):
        adapter.embed_documents(["text"])


def test_embedding_adapter_rejects_mixed_batch_dimensions() -> None:
    adapter = ValidatedEmbeddings(
        FakeEmbeddings(documents=[[0.1], [0.2, 0.3]]),  # type: ignore[arg-type]
        provider="TEST",
    )

    with pytest.raises(EmbeddingServiceError, match="维度不一致"):
        adapter.embed_documents(["first", "second"])


@pytest.mark.parametrize("status_code", [429, 503])
def test_dashscope_status_failure_is_retryable_without_returning_empty_vector(
    status_code: int,
) -> None:
    wrapper = object.__new__(DashScopeEmbeddingsWrapper)
    wrapper.model_name = "test-model"
    wrapper.dashscope = SimpleNamespace(
        TextEmbedding=SimpleNamespace(
            call=lambda **_: SimpleNamespace(status_code=status_code, output={})
        )
    )

    with pytest.raises(EmbeddingServiceError) as captured:
        wrapper.embed_query("query")

    assert captured.value.status_code == status_code
    assert captured.value.retryable


def test_validated_embedding_returns_finite_float_vectors() -> None:
    adapter = ValidatedEmbeddings(
        FakeEmbeddings(documents=[[1, 2.5]], query=[3, 4]),  # type: ignore[arg-type]
        provider="TEST",
    )

    assert adapter.embed_documents(["text"]) == [[1.0, 2.5]]
    assert adapter.embed_query("query") == [3.0, 4.0]
