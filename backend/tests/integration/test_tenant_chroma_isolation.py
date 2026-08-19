from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

USER_A = "user-a"
USER_B = "user-b"


class DeterministicEmbeddings:
    """不访问模型或网络的固定维度测试 embedding。"""

    @staticmethod
    def _embed(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255 for byte in digest[:16]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.mark.integration
def test_real_chroma_keeps_tenants_isolated_during_search_and_delete(
    tmp_path: Path,
) -> None:
    store = Chroma(
        collection_name="m1_tenant_isolation",
        embedding_function=DeterministicEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
    )
    store.add_documents(
        [
            Document(
                page_content="共享主题：用户 A 的私有部署步骤",
                metadata={"user_id": USER_A, "document_id": "doc-a"},
            ),
            Document(
                page_content="共享主题：用户 B 的私有恢复步骤",
                metadata={"user_id": USER_B, "document_id": "doc-b"},
            ),
        ],
        ids=["chunk-a", "chunk-b"],
    )

    user_a_results = store.similarity_search(
        "用户 B 的私有恢复步骤",
        k=10,
        filter={"user_id": USER_A},
    )
    assert user_a_results
    assert {document.metadata["user_id"] for document in user_a_results} == {USER_A}
    assert {document.metadata["document_id"] for document in user_a_results} == {
        "doc-a"
    }

    store.delete(where={"user_id": USER_A})

    assert store.get(where={"user_id": USER_A})["ids"] == []
    assert store.get(where={"user_id": USER_B})["ids"] == ["chunk-b"]
    user_b_results = store.similarity_search(
        "用户 B 的私有恢复步骤",
        k=10,
        filter={"user_id": USER_B},
    )
    assert {document.metadata["document_id"] for document in user_b_results} == {
        "doc-b"
    }
