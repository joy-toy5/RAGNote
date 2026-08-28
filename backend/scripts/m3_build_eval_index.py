"""为冻结数据集构建隔离的离线评测 Chroma 索引。

索引只允许写进 evals/indexes 下的目录，绝不碰生产 persist_directory。
chunk 正文不存在数据集里，必须从 *.normalized.txt 按 [char_start, char_end)
重新切片，并用 content_sha256 逐条校验，切错一个字符就会失败。

每个 chunk 必须写全 retrieval_contract.candidate_from_document 判定
provenance_status="verified" 所需的元数据；缺任何一项都会让
execution_from_trace 以 "正式评测不接受 legacy 候选" 拒绝整条 Query。

用法::

    PYTHONPATH=. .venv/bin/python scripts/m3_build_eval_index.py \
        --dataset evals/datasets/m3_dev_v2/manifest.json --dry-run
    PYTHONPATH=. .venv/bin/python scripts/m3_build_eval_index.py \
        --dataset evals/datasets/m3_dev_v2/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from app.indexing.contracts import (
    validate_blob_id,
    validate_document_id,
    validate_document_revision,
    validate_index_version,
)

# 评测索引只能落在这个根目录下，避免任何一次手滑写进生产库。
ALLOWED_INDEX_ROOT = Path("evals/indexes").resolve()
# Chroma 只接受 3-512 位 [a-zA-Z0-9._-]，且首尾必须是字母或数字。
COLLECTION_NAME = "m3-dev-v2-eval"
URI_SCHEME = "eval-dataset://rag-note-m3-dev"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_documents(dataset_dir: Path, manifest: dict[str, Any]) -> list[Any]:
    """把 108 个 chunk 还原成带完整 provenance 元数据的 Document。"""
    from langchain_core.documents import Document

    index_version = manifest["index_version"]
    validate_index_version(index_version)
    dataset_version = manifest["dataset_version"]

    corpus = {
        item["corpus_item_id"]: item
        for item in load_jsonl(dataset_dir / manifest["files"]["corpus"])
    }
    chunks = load_jsonl(dataset_dir / manifest["files"]["chunks"])

    # 规范化文本一次读入，后续所有切片都从同一份字符串来，避免重复 IO 造成偏移漂移。
    normalized: dict[str, str] = {}
    for item_id, item in corpus.items():
        text = (dataset_dir / item["normalized_text_path"]).read_text(encoding="utf-8")
        actual = sha256_text(text)
        if actual != item["normalized_text_sha256"]:
            raise ValueError(
                f"{item_id} 规范化文本摘要不一致: "
                f"声明 {item['normalized_text_sha256']}, 实际 {actual}"
            )
        normalized[item_id] = text

    documents: list[Any] = []
    seen_chunk_ids: set[str] = set()
    for chunk in chunks:
        item_id = chunk["corpus_item_id"]
        item = corpus[item_id]
        if chunk["index_version"] != index_version:
            raise ValueError(f"{chunk['chunk_id']} index_version 与 manifest 不一致")
        validate_document_revision(chunk["document_revision"])
        if chunk["document_revision"] != 1:
            raise ValueError("clean-load 数据集只允许 document_revision=1")
        if chunk["page_number"] is not None:
            raise ValueError("page map 未冻结，page_number 必须为 null")
        if chunk["document_id"] != item["document_id"]:
            raise ValueError(f"{chunk['chunk_id']} document_id 与 corpus 不一致")
        if chunk["user_id"] != item["user_id"]:
            raise ValueError(f"{chunk['chunk_id']} user_id 与 corpus 不一致")
        validate_document_id(chunk["document_id"])
        validate_blob_id(chunk["chunk_id"])
        validate_blob_id(chunk["content_sha256"])

        char_start = chunk["char_start"]
        char_end = chunk["char_end"]
        if char_start < 0 or char_end <= char_start:
            raise ValueError(f"{chunk['chunk_id']} 字符区间必须是非空 0-based 右开区间")
        text = normalized[item_id]
        if char_end > len(text):
            raise ValueError(f"{chunk['chunk_id']} 字符区间越过规范化文本末尾")
        content = text[char_start:char_end]
        actual = sha256_text(content)
        if actual != chunk["content_sha256"]:
            raise ValueError(
                f"{chunk['chunk_id']} 正文摘要不一致: "
                f"声明 {chunk['content_sha256']}, 实际 {actual}"
            )
        if chunk["chunk_id"] in seen_chunk_ids:
            raise ValueError(f"chunk_id 重复: {chunk['chunk_id']}")
        seen_chunk_ids.add(chunk["chunk_id"])

        base_uri = f"{URI_SCHEME}/{dataset_version}"
        # page_number 故意不写：Chroma 不接受 None，而 _evidence_spans_from_metadata
        # 用 .get() 读取，键缺失即等价于 None。
        metadata = {
            "user_id": chunk["user_id"],
            "source_type": "knowledge_base",
            "original_filename": item["display_name"],
            "blob_id": item["blob_id"],
            "document_id": chunk["document_id"],
            "document_revision": chunk["document_revision"],
            "chunk_id": chunk["chunk_id"],
            "chunk_ordinal": chunk["chunk_ordinal"],
            "index_version": chunk["index_version"],
            "source_uri": f"{base_uri}/{item['blob_path']}",
            # char 偏移是相对规范化文本的，所以摘要必须是规范化文本的摘要，
            # 不是 chunk 正文自己的摘要。
            "char_start": char_start,
            "char_end": char_end,
            "source_text_sha256": item["normalized_text_sha256"],
            "source_text_uri": f"{base_uri}/{item['normalized_text_path']}",
        }
        if any(value is None for value in metadata.values()):
            raise ValueError(f"{chunk['chunk_id']} 元数据存在 None，Chroma 会拒绝写入")
        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def assert_verified(documents: list[Any]) -> None:
    """用生产契约本身确认每个 chunk 都会被判成 verified。"""
    from app.rag.retrieval_contract import candidate_from_document

    for rank, document in enumerate(documents, 1):
        candidate = candidate_from_document(
            document,
            user_id=document.metadata["user_id"],
            rank=rank,
        )
        if candidate.provenance_status != "verified":
            raise ValueError(
                f"{document.metadata['chunk_id']} 未被判定为 verified，"
                "评测会整条 Query 失败"
            )
        if candidate.chunk_id != document.metadata["chunk_id"]:
            raise ValueError("候选 chunk_id 与元数据不一致")


def resolve_index_dir(raw: str) -> Path:
    index_dir = Path(raw).resolve()
    if index_dir != ALLOWED_INDEX_ROOT and ALLOWED_INDEX_ROOT not in index_dir.parents:
        raise ValueError(f"评测索引目录必须位于 {ALLOWED_INDEX_ROOT} 之下")
    return index_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="构建隔离的 M3 离线评测索引")
    parser.add_argument("--dataset", required=True, help="manifest.json 路径")
    parser.add_argument("--index-dir", default="evals/indexes/m3_dev_v2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验切片和 provenance，不调用 embedding、不写盘",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="删除已存在的索引目录后重建",
    )
    args = parser.parse_args()

    manifest_path = Path(args.dataset)
    if manifest_path.is_dir():
        raise SystemExit("--dataset 必须指向 manifest.json，不是数据集目录")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_dir = manifest_path.parent

    documents = build_documents(dataset_dir, manifest)
    assert_verified(documents)

    users = sorted({document.metadata["user_id"] for document in documents})
    print(f"dataset        : {manifest['dataset_id']} {manifest['dataset_version']}")
    print(f"index_version  : {manifest['index_version']}")
    print(f"chunks         : {len(documents)}")
    print(f"users          : {users}")
    print("provenance     : 全部 verified")

    if args.dry_run:
        print("dry-run: 未调用 embedding，未写盘")
        return 0

    index_dir = resolve_index_dir(args.index_dir)
    if index_dir.exists():
        if not args.rebuild:
            raise SystemExit(f"{index_dir} 已存在；要重建请显式加 --rebuild")
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True)

    from langchain_chroma import Chroma

    from app.utils.factory import embed_model

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embed_model,
        persist_directory=str(index_dir),
    )
    try:
        total = len(documents)
        for start in range(0, total, args.batch_size):
            batch = documents[start : start + args.batch_size]
            store.add_documents(
                documents=batch,
                ids=[document.metadata["chunk_id"] for document in batch],
            )
            print(f"embedded {min(start + len(batch), total)}/{total}")
        stored = store.get(include=["metadatas"])
        count = len(stored["ids"])
        if count != total:
            raise SystemExit(f"写入数量不一致: 期望 {total}, 实际 {count}")
        if set(stored["ids"]) != {d.metadata["chunk_id"] for d in documents}:
            raise SystemExit("写入的 id 集合与 chunk_id 集合不一致")
        print(f"索引已写入 {index_dir} | collection={COLLECTION_NAME} | count={count}")
    finally:
        client = getattr(store, "_client", None)
        if client is not None:
            client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
