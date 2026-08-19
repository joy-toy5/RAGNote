"""从显式只读导出生成 legacy 索引映射，不修改任何事实或派生数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def build_legacy_inventory(
    chroma_export: Mapping[str, object],
    md5_records: Iterable[Mapping[str, object]],
    image_directories: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """把 Chroma、MD5 和图片清单映射为保守的 legacy 报告。"""
    ids = _list_field(chroma_export, "ids")
    documents = _list_field(chroma_export, "documents")
    metadatas = _list_field(chroma_export, "metadatas")
    if not (len(ids) == len(documents) == len(metadatas)):
        raise ValueError("Chroma 导出的 ids/documents/metadatas 长度不一致")

    chunks = []
    document_keys: set[tuple[str | None, str | None, str | None]] = set()
    chroma_keys: set[tuple[str | None, str | None]] = set()
    for legacy_id, content, raw_metadata in zip(
        ids,
        documents,
        metadatas,
        strict=True,
    ):
        if not isinstance(legacy_id, str) or not isinstance(content, str):
            raise TypeError("Chroma ID 和正文必须是字符串")
        if not isinstance(raw_metadata, Mapping):
            raise TypeError("Chroma metadata 必须是映射")
        metadata = dict(raw_metadata)
        user_id = _optional_string(metadata.get("user_id"))
        md5_value = _optional_string(metadata.get("md5"))
        filename = _filename(metadata)
        document_keys.add((user_id, md5_value, filename))
        chroma_keys.add((user_id, md5_value))
        chunks.append(
            {
                "blob_id": None,
                "chunk_id": None,
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "document_id": None,
                "document_revision": None,
                "filename": filename,
                "index_version": None,
                "legacy_chunk_id": legacy_id,
                "md5": md5_value,
                "observed_page": metadata.get("page_number", metadata.get("page")),
                "provenance_status": "legacy_index_only",
                "reason": "历史原件或稳定字符区间缺失，禁止推导 verified 身份",
                "user_id": user_id,
            }
        )

    md5_items = [_normalize_md5_record(record) for record in md5_records]
    md5_keys = {(item["user_id"], item["md5"]) for item in md5_items}
    images = [dict(item) for item in image_directories]
    unresolved_images = [
        {
            **item,
            "provenance_status": "legacy_unresolved",
            "reason": "图片目录无法同时关联 Chroma metadata 与 MD5 台账",
        }
        for item in images
        if (
            _optional_string(item.get("user_id")),
            _optional_string(item.get("md5")),
        )
        not in chroma_keys & md5_keys
    ]

    return {
        "schema_version": 1,
        "report_type": "m2_legacy_index_inventory",
        "migration_policy": {
            "in_place_metadata_rewrite": False,
            "legacy_qrels_eligible": False,
            "original_bytes_required_for_verified_blob_id": True,
        },
        "summary": {
            "chroma_chunks": len(chunks),
            "legacy_documents": len(document_keys),
            "md5_records": len(md5_items),
            "image_directories": len(images),
            "legacy_unresolved_image_directories": len(unresolved_images),
        },
        "chunks": chunks,
        "md5_records": md5_items,
        "unresolved_image_directories": unresolved_images,
    }


def read_chroma_export(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Chroma 导出必须是 JSON object")
    return payload


def read_md5_records(root: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(root.rglob("md5_hex_store.txt")):
        user_id = _user_id_from_md5_path(root, path)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            value = line.strip()
            if not value:
                continue
            if value.startswith("{"):
                record = json.loads(value)
                if not isinstance(record, dict):
                    raise TypeError(f"{path}:{line_number} 不是 JSON object")
            else:
                record = {"md5": value}
            records.append({**record, "user_id": user_id})
    return records


def read_image_directories(root: Path) -> list[dict[str, object]]:
    directories = []
    if not root.exists():
        return directories
    for user_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for md5_dir in sorted(path for path in user_dir.iterdir() if path.is_dir()):
            files = sorted(path.name for path in md5_dir.iterdir() if path.is_file())
            directories.append(
                {
                    "file_count": len(files),
                    "md5": md5_dir.name,
                    "user_id": user_dir.name,
                }
            )
    return directories


def _list_field(payload: Mapping[str, object], name: str) -> list[Any]:
    value = payload.get(name)
    if not isinstance(value, list):
        raise TypeError(f"Chroma 导出字段 {name} 必须是列表")
    return value


def _filename(metadata: Mapping[str, object]) -> str | None:
    return _optional_string(
        metadata.get("original_filename")
        or metadata.get("filename")
        or metadata.get("source")
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _normalize_md5_record(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "filename": _optional_string(
            record.get("original_filename") or record.get("filename")
        ),
        "md5": _optional_string(record.get("md5")),
        "user_id": _optional_string(record.get("user_id")),
    }


def _user_id_from_md5_path(root: Path, path: Path) -> str | None:
    relative_parts = path.relative_to(root).parts
    if "user_md5" in relative_parts:
        index = relative_parts.index("user_md5")
        if index + 1 < len(relative_parts):
            return relative_parts[index + 1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 M2 legacy 索引只读映射报告")
    parser.add_argument("--chroma-export", type=Path, required=True)
    parser.add_argument("--md5-root", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    report = build_legacy_inventory(
        read_chroma_export(arguments.chroma_export),
        read_md5_records(arguments.md5_root),
        read_image_directories(arguments.images_root),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if arguments.output is None:
        print(rendered, end="")
        return
    if arguments.output.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {arguments.output}")
    arguments.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
