"""稳定索引身份和来源信息的纯函数契约。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_INDEX_CONFIG_SCHEMA = "rag-note.index-config.v1"
_CHUNK_ID_SCHEMA = "rag-note.chunk-id.v1"
_MAX_REVISION = 4_294_967_295

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def compute_blob_id(content: bytes | bytearray | memoryview) -> str:
    """以原始字节的 SHA-256 标识不可变内容。"""
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError("content 必须是 bytes-like 对象")
    return hashlib.sha256(content).hexdigest()


def compute_text_sha256(text: str) -> str:
    """计算已由上游规范化的 UTF-8 文本摘要。"""
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_blob_id(blob_id: str) -> str:
    """校验并返回规范的小写 SHA-256 blob ID。"""
    return _validate_sha256(blob_id, "blob_id")


def validate_index_version(index_version: str) -> str:
    """校验并返回规范的小写 SHA-256 index version。"""
    return _validate_sha256(index_version, "index_version")


def validate_document_id(document_id: str | uuid.UUID) -> str:
    """校验已经持久化的 RFC 4122 UUID，不在此处生成业务身份。"""
    if isinstance(document_id, uuid.UUID):
        parsed = document_id
        canonical = str(document_id)
    elif isinstance(document_id, str):
        try:
            parsed = uuid.UUID(document_id)
        except (ValueError, AttributeError) as exc:
            raise ValueError("document_id 必须是规范 RFC 4122 UUID") from exc
        canonical = str(parsed)
        if document_id != canonical:
            raise ValueError("document_id 必须使用小写连字符 UUID 格式")
    else:
        raise TypeError("document_id 必须是字符串或 UUID")

    if parsed.int == 0 or parsed.variant != uuid.RFC_4122:
        raise ValueError("document_id 必须是非零 RFC 4122 UUID")
    return canonical


def validate_document_revision(revision: int) -> int:
    """校验从 1 开始、可落入无符号 32 位列的文档版本号。"""
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise TypeError("document_revision 必须是整数")
    if not 1 <= revision <= _MAX_REVISION:
        raise ValueError("document_revision 必须在 1 到 4294967295 之间")
    return revision


def next_document_revision(current: int | None) -> int:
    """从当前持久化版本计算下一单调版本；首个版本固定为 1。"""
    if current is None:
        return 1
    current = validate_document_revision(current)
    if current == _MAX_REVISION:
        raise OverflowError("document_revision 已达到上限")
    return current + 1


def canonical_index_config(config: Mapping[str, object]) -> bytes:
    """将 JSON 配置规范化为稳定 UTF-8 字节。"""
    if not isinstance(config, Mapping):
        raise TypeError("index config 必须是映射")
    normalized = _normalize_json(config, path="config")
    payload = {"config": normalized, "schema": _INDEX_CONFIG_SCHEMA}
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def compute_index_version(config: Mapping[str, object]) -> str:
    """以规范配置摘要标识可重建的一套索引行为。"""
    return hashlib.sha256(canonical_index_config(config)).hexdigest()


@dataclass(frozen=True, slots=True)
class CharacterSpan:
    """规范化全文中的 0-based、右开字符区间。"""

    start: int
    end: int

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.start, self.end)):
            raise TypeError("字符区间必须由整数构成")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("字符区间必须满足 0 <= start < end")


@dataclass(frozen=True, slots=True)
class ChunkProvenance:
    """生成 chunk 身份所需的不可变来源锚点。"""

    document_id: str | uuid.UUID
    document_revision: int
    index_version: str
    chunk_ordinal: int
    blob_id: str | None
    normalized_text_sha256: str | None
    character_span: CharacterSpan | None
    page_number: int | None = None
    legacy_index_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", validate_document_id(self.document_id))
        object.__setattr__(
            self,
            "document_revision",
            validate_document_revision(self.document_revision),
        )
        object.__setattr__(self, "index_version", validate_index_version(self.index_version))

        if isinstance(self.chunk_ordinal, bool) or not isinstance(self.chunk_ordinal, int):
            raise TypeError("chunk_ordinal 必须是整数")
        if self.chunk_ordinal < 0:
            raise ValueError("chunk_ordinal 必须从 0 开始")
        if self.page_number is not None:
            if isinstance(self.page_number, bool) or not isinstance(self.page_number, int):
                raise TypeError("page_number 必须是整数")
            if self.page_number < 1:
                raise ValueError("page_number 必须从 1 开始")
        if not isinstance(self.legacy_index_only, bool):
            raise TypeError("legacy_index_only 必须是布尔值")

        if self.blob_id is not None:
            object.__setattr__(self, "blob_id", validate_blob_id(self.blob_id))
        if self.normalized_text_sha256 is not None:
            object.__setattr__(
                self,
                "normalized_text_sha256",
                _validate_sha256(self.normalized_text_sha256, "normalized_text_sha256"),
            )
        if self.character_span is not None and not isinstance(self.character_span, CharacterSpan):
            raise TypeError("character_span 必须是 CharacterSpan")

        if not self.legacy_index_only and (
            self.blob_id is None
            or self.normalized_text_sha256 is None
            or self.character_span is None
        ):
            raise ValueError("非 legacy chunk 必须绑定 blob、规范化全文摘要和字符区间")
        if self.character_span is not None and self.normalized_text_sha256 is None:
            raise ValueError("字符区间必须绑定规范化全文摘要")


def compute_chunk_id(content: str, provenance: ChunkProvenance) -> str:
    """根据版本化来源锚点和 chunk 内容生成确定性 SHA-256 ID。"""
    if not isinstance(provenance, ChunkProvenance):
        raise TypeError("provenance 必须是 ChunkProvenance")
    span = provenance.character_span
    payload: dict[str, JsonValue] = {
        "blob_id": provenance.blob_id,
        "character_span": None if span is None else {"end": span.end, "start": span.start},
        "chunk_ordinal": provenance.chunk_ordinal,
        "content_sha256": compute_text_sha256(content),
        "document_id": str(provenance.document_id),
        "document_revision": provenance.document_revision,
        "index_version": provenance.index_version,
        "legacy_index_only": provenance.legacy_index_only,
        "normalized_text_sha256": provenance.normalized_text_sha256,
        "page_number": provenance.page_number,
        "schema": _CHUNK_ID_SCHEMA,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} 必须是字符串")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} 必须是 64 位小写十六进制 SHA-256")
    return value


def _normalize_json(value: object, *, path: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return unicodedata.normalize("NFC", value) if isinstance(value, str) else value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} 不能包含 NaN 或 Infinity")
        return 0.0 if value == 0 else value
    if isinstance(value, list):
        return [_normalize_json(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} 的键必须是字符串")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise ValueError(f"{path} 包含规范化后重复的键: {normalized_key}")
            normalized[normalized_key] = _normalize_json(item, path=f"{path}.{normalized_key}")
        return normalized
    raise TypeError(f"{path} 包含非 JSON 类型: {type(value).__name__}")
