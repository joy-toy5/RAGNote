"""冻结评测语料、Query、evidence anchor 与 source qrels 装载。"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from app.evaluation.contracts import (
    BINARY_RELEVANCE_THRESHOLD,
    deterministic_document_id,
    sha256_json,
)
from app.indexing.contracts import (
    CharacterSpan,
    ChunkProvenance,
    compute_chunk_id,
    compute_index_version,
    compute_text_sha256,
    validate_blob_id,
    validate_document_id,
    validate_document_revision,
    validate_index_version,
)


class DatasetValidationError(ValueError):
    """冻结数据文件或引用完整性不满足评测契约。"""


@dataclass(frozen=True, slots=True)
class CorpusItem:
    corpus_item_id: str
    user_id: str
    document_id: str
    document_revision: int
    source_type: Literal["knowledge_base", "note"]
    display_name: str
    blob_path: str
    blob_id: str
    blob_byte_size: int
    normalized_text_path: str
    normalized_text_sha256: str


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str
    corpus_item_id: str
    user_id: str
    document_id: str
    document_revision: int
    index_version: str
    chunk_ordinal: int
    content_sha256: str
    char_start: int
    char_end: int
    page_number: int | None


@dataclass(frozen=True, slots=True)
class QueryCase:
    query_id: str
    user_id: str
    query: str
    query_type: str
    answerability: Literal["answerable", "unanswerable"]
    no_answer_reason: str | None


@dataclass(frozen=True, slots=True)
class EvidenceAnchor:
    anchor_id: str
    corpus_item_id: str
    user_id: str
    document_id: str
    document_revision: int
    blob_id: str
    normalized_text_sha256: str
    char_start: int
    char_end: int
    page_number: int | None
    evidence_text_sha256: str


@dataclass(frozen=True, slots=True)
class SourceQrel:
    query_id: str
    anchor_id: str
    relevance: int
    evidence_set_id: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    schema_version: int
    dataset_id: str
    dataset_version: str
    qrels_version: str
    split: Literal["smoke", "development", "held_out"]
    document_namespace: uuid.UUID
    index_config: dict[str, Any]
    index_version: str
    corpus: tuple[CorpusItem, ...]
    chunks: tuple[ChunkRecord, ...]
    queries: tuple[QueryCase, ...]
    anchors: tuple[EvidenceAnchor, ...]
    source_qrels: tuple[SourceQrel, ...]
    root: Path

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "qrels_version": self.qrels_version,
            "split": self.split,
            "document_namespace": str(self.document_namespace),
            "index_config": self.index_config,
            "index_version": self.index_version,
            "corpus": [asdict(item) for item in self.corpus],
            "chunks": [asdict(item) for item in self.chunks],
            "queries": [asdict(item) for item in self.queries],
            "anchors": [asdict(item) for item in self.anchors],
            "source_qrels": [asdict(item) for item in self.source_qrels],
        }


def load_dataset(manifest_path: Path) -> EvaluationDataset:
    """从显式 manifest 装载并复核全部冻结事实。"""
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = _read_json_object(manifest_path)
    files = _mapping(manifest.get("files"), "manifest.files")
    dataset = EvaluationDataset(
        schema_version=_integer(manifest.get("schema_version"), "schema_version"),
        dataset_id=_text(manifest.get("dataset_id"), "dataset_id"),
        dataset_version=_text(manifest.get("dataset_version"), "dataset_version"),
        qrels_version=_text(manifest.get("qrels_version"), "qrels_version"),
        split=_split(manifest.get("split")),
        document_namespace=_namespace(manifest.get("document_namespace")),
        index_config=dict(_mapping(manifest.get("index_config"), "index_config")),
        index_version=_text(manifest.get("index_version"), "index_version"),
        corpus=_load_records(root, files, "corpus", CorpusItem),
        chunks=_load_records(root, files, "chunks", ChunkRecord),
        queries=_load_records(root, files, "queries", QueryCase),
        anchors=_load_records(root, files, "evidence_anchors", EvidenceAnchor),
        source_qrels=_load_records(root, files, "source_qrels", SourceQrel),
        root=root,
    )
    _validate_dataset(dataset)
    return dataset


def dataset_fingerprint(dataset: EvaluationDataset) -> str:
    return sha256_json(dataset.semantic_payload())


def _validate_dataset(dataset: EvaluationDataset) -> None:
    if dataset.schema_version != 1:
        raise DatasetValidationError("dataset schema_version 必须为 1")
    validate_index_version(dataset.index_version)
    if compute_index_version(dataset.index_config) != dataset.index_version:
        raise DatasetValidationError("index_config 与 index_version 不一致")
    corpus = _unique_map(dataset.corpus, "corpus_item_id")
    _unique_map(dataset.corpus, "document_id")
    _unique_document_natural_keys(dataset.corpus)
    queries = _unique_map(dataset.queries, "query_id")
    anchors = _unique_map(dataset.anchors, "anchor_id")
    _unique_anchor_spans(dataset.anchors)
    _unique_map(dataset.chunks, "chunk_id")
    _validate_corpus(dataset, corpus)
    _validate_chunks(dataset, corpus)
    _validate_anchors(dataset, corpus)
    _validate_qrels(dataset, queries, anchors)


def _validate_corpus(
    dataset: EvaluationDataset,
    corpus: dict[str, CorpusItem],
) -> None:
    for item in corpus.values():
        _text(item.corpus_item_id, "corpus_item_id")
        _text(item.user_id, "corpus user_id")
        _text(item.display_name, "corpus display_name")
        if item.source_type not in {"knowledge_base", "note"}:
            raise DatasetValidationError("corpus source_type 不受支持")
        validate_document_id(item.document_id)
        validate_document_revision(item.document_revision)
        if item.document_revision != 1:
            raise DatasetValidationError(
                "当前 clean-load dataset 只支持 document_revision=1"
            )
        expected = deterministic_document_id(
            dataset.document_namespace,
            item.corpus_item_id,
        )
        if item.document_id != expected:
            raise DatasetValidationError("corpus 未使用确定性 document_id")
        validate_blob_id(item.blob_id)
        validate_blob_id(item.normalized_text_sha256)
        if (
            isinstance(item.blob_byte_size, bool)
            or not isinstance(item.blob_byte_size, int)
            or item.blob_byte_size < 0
        ):
            raise DatasetValidationError("corpus blob_byte_size 必须是非负整数")
        blob = _safe_file(dataset.root, item.blob_path).read_bytes()
        normalized = _safe_file(dataset.root, item.normalized_text_path).read_bytes()
        if len(blob) != item.blob_byte_size:
            raise DatasetValidationError("corpus 原件字节数不一致")
        if hashlib.sha256(blob).hexdigest() != item.blob_id:
            raise DatasetValidationError("corpus 原件 SHA-256 不一致")
        if hashlib.sha256(normalized).hexdigest() != item.normalized_text_sha256:
            raise DatasetValidationError("corpus 规范化文本 SHA-256 不一致")
        normalized.decode("utf-8")


def _validate_chunks(
    dataset: EvaluationDataset,
    corpus: dict[str, CorpusItem],
) -> None:
    by_item: dict[str, list[ChunkRecord]] = {}
    for chunk in dataset.chunks:
        item = _related_item(corpus, chunk.corpus_item_id, "chunk")
        _same_identity(item, chunk, "chunk")
        validate_blob_id(chunk.chunk_id)
        validate_blob_id(chunk.content_sha256)
        if chunk.index_version != dataset.index_version:
            raise DatasetValidationError("chunk index_version 与数据集不一致")
        span = CharacterSpan(chunk.char_start, chunk.char_end)
        if chunk.page_number is not None:
            raise DatasetValidationError("缺少 page map 时 chunk page_number 必须为 null")
        text = _normalized_text(dataset, item)
        content = text[span.start : span.end]
        if len(content) != span.end - span.start:
            raise DatasetValidationError("chunk 字符区间超出规范化文本")
        if compute_text_sha256(content) != chunk.content_sha256:
            raise DatasetValidationError("chunk 正文 SHA-256 不一致")
        provenance = ChunkProvenance(
            document_id=item.document_id,
            document_revision=item.document_revision,
            index_version=dataset.index_version,
            chunk_ordinal=chunk.chunk_ordinal,
            blob_id=item.blob_id,
            normalized_text_sha256=item.normalized_text_sha256,
            character_span=span,
            page_number=chunk.page_number,
        )
        if compute_chunk_id(content, provenance) != chunk.chunk_id:
            raise DatasetValidationError("chunk_id 与 M2 稳定身份规则不一致")
        by_item.setdefault(item.corpus_item_id, []).append(chunk)
    for chunks in by_item.values():
        ordinals = sorted(chunk.chunk_ordinal for chunk in chunks)
        if ordinals != list(range(len(ordinals))):
            raise DatasetValidationError("chunk_ordinal 必须从 0 连续递增")
    if set(by_item) != set(corpus):
        raise DatasetValidationError("每个 corpus item 必须至少具有一个 chunk")


def _validate_anchors(
    dataset: EvaluationDataset,
    corpus: dict[str, CorpusItem],
) -> None:
    for anchor in dataset.anchors:
        item = _related_item(corpus, anchor.corpus_item_id, "anchor")
        _same_identity(item, anchor, "anchor")
        _text(anchor.anchor_id, "anchor_id")
        validate_blob_id(anchor.evidence_text_sha256)
        _optional_page(anchor.page_number, "anchor page_number")
        if anchor.page_number is not None:
            raise DatasetValidationError("缺少 page map 时 anchor page_number 必须为 null")
        if anchor.blob_id != item.blob_id:
            raise DatasetValidationError("anchor blob_id 与 corpus 不一致")
        if anchor.normalized_text_sha256 != item.normalized_text_sha256:
            raise DatasetValidationError("anchor 规范化文本摘要与 corpus 不一致")
        span = CharacterSpan(anchor.char_start, anchor.char_end)
        evidence = _normalized_text(dataset, item)[span.start : span.end]
        if len(evidence) != span.end - span.start:
            raise DatasetValidationError("anchor 字符区间超出规范化文本")
        if compute_text_sha256(evidence) != anchor.evidence_text_sha256:
            raise DatasetValidationError("anchor evidence SHA-256 不一致")


def _validate_qrels(
    dataset: EvaluationDataset,
    queries: dict[str, QueryCase],
    anchors: dict[str, EvidenceAnchor],
) -> None:
    qrels_by_query: dict[str, list[SourceQrel]] = {}
    evidence_sets: dict[tuple[str, str], list[SourceQrel]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    referenced_anchors: set[str] = set()
    for qrel in dataset.source_qrels:
        query = _related_item(queries, qrel.query_id, "source qrel query")
        anchor = _related_item(anchors, qrel.anchor_id, "source qrel anchor")
        if (
            isinstance(qrel.relevance, bool)
            or not isinstance(qrel.relevance, int)
            or not 1 <= qrel.relevance <= 3
        ):
            raise DatasetValidationError("relevance 必须在 1 到 3 之间")
        if query.user_id != anchor.user_id:
            raise DatasetValidationError("source qrel 不能跨用户")
        if qrel.relevance == BINARY_RELEVANCE_THRESHOLD:
            evidence_set_id = _text(qrel.evidence_set_id, "evidence_set_id")
            evidence_sets.setdefault(
                (qrel.query_id, evidence_set_id), []
            ).append(qrel)
        elif qrel.evidence_set_id is not None:
            raise DatasetValidationError(
                "只有 relevance=2 的必要证据可以声明 evidence_set_id"
            )
        pair = (qrel.query_id, qrel.anchor_id)
        if pair in seen_pairs:
            raise DatasetValidationError("source qrel 不能重复")
        seen_pairs.add(pair)
        referenced_anchors.add(qrel.anchor_id)
        qrels_by_query.setdefault(qrel.query_id, []).append(qrel)
    if any(len(members) < 2 for members in evidence_sets.values()):
        raise DatasetValidationError("relevance=2 evidence set 至少需要两个证据锚点")
    if referenced_anchors != set(anchors):
        raise DatasetValidationError("evidence anchor 必须且只能被 source qrel 引用")
    answerability = {query.answerability for query in dataset.queries}
    if answerability != {"answerable", "unanswerable"}:
        raise DatasetValidationError("数据集必须同时包含有答案和无答案 Query")
    corpus_users = {item.user_id for item in dataset.corpus}
    for query in dataset.queries:
        _text(query.query_id, "query_id")
        _text(query.user_id, "query user_id")
        _text(query.query, "query")
        _text(query.query_type, "query_type")
        if query.user_id not in corpus_users:
            raise DatasetValidationError("query user_id 不属于冻结 corpus")
        related = qrels_by_query.get(query.query_id, [])
        has_complete_evidence = any(qrel.relevance == 3 for qrel in related) or any(
            query_id == query.query_id for query_id, _ in evidence_sets
        )
        if query.answerability == "answerable" and not has_complete_evidence:
            raise DatasetValidationError(
                "有答案 Query 必须具有 relevance=3 证据或完整 relevance=2 evidence set"
            )
        if query.answerability == "answerable" and query.no_answer_reason is not None:
            raise DatasetValidationError("有答案 Query 不能携带 no_answer_reason")
        if query.answerability == "unanswerable":
            if related:
                raise DatasetValidationError("无答案 Query 不能具有正向 source qrel")
            if query.no_answer_reason not in {
                "absent_from_corpus",
                "tenant_excluded",
                "insufficient_evidence",
            }:
                raise DatasetValidationError("无答案 Query 缺少受支持的原因")


def _load_records(
    root: Path,
    files: dict[str, Any],
    name: str,
    record_type: type[Any],
) -> tuple[Any, ...]:
    path = _safe_file(root, _text(files.get(name), f"files.{name}"))
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            records.append(record_type(**payload))
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError(f"{path.name}:{line_number} 无效: {exc}") from exc
    if not records:
        raise DatasetValidationError(f"{path.name} 不能为空")
    return tuple(records)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"无法读取数据集 manifest: {exc}") from exc
    return _mapping(payload, "manifest")


def _safe_file(root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise DatasetValidationError(f"数据集路径必须是相对路径: {relative_path}")
    path = (root / candidate).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise DatasetValidationError(f"数据集路径越界或不是文件: {relative_path}")
    return path


def _normalized_text(dataset: EvaluationDataset, item: CorpusItem) -> str:
    return _safe_file(dataset.root, item.normalized_text_path).read_text(encoding="utf-8")


def _unique_map(records: tuple[Any, ...], field: str) -> dict[str, Any]:
    values = [getattr(record, field) for record in records]
    if len(values) != len(set(values)):
        raise DatasetValidationError(f"{field} 必须唯一")
    return dict(zip(values, records, strict=True))


def _unique_document_natural_keys(corpus: tuple[CorpusItem, ...]) -> None:
    keys = [
        (item.user_id, item.source_type, item.display_name)
        for item in corpus
    ]
    if len(keys) != len(set(keys)):
        raise DatasetValidationError(
            "corpus 的 (user_id, source_type, display_name) 必须唯一"
        )


def _unique_anchor_spans(anchors: tuple[EvidenceAnchor, ...]) -> None:
    keys = [
        (
            anchor.user_id,
            anchor.document_id,
            anchor.document_revision,
            anchor.page_number,
            anchor.char_start,
            anchor.char_end,
        )
        for anchor in anchors
    ]
    if len(keys) != len(set(keys)):
        raise DatasetValidationError("evidence anchor 的 provenance span 必须唯一")


def _related_item(records: dict[str, Any], key: str, label: str) -> Any:
    try:
        return records[key]
    except KeyError as exc:
        raise DatasetValidationError(f"{label} 引用了未知身份: {key}") from exc


def _same_identity(item: CorpusItem, record: Any, label: str) -> None:
    actual = (record.user_id, record.document_id, record.document_revision)
    expected = (item.user_id, item.document_id, item.document_revision)
    if actual != expected:
        raise DatasetValidationError(f"{label} 与 corpus 身份不一致")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetValidationError(f"{label} 必须是 JSON object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{label} 必须是非空字符串")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetValidationError(f"{label} 必须是整数")
    return value


def _namespace(value: object) -> uuid.UUID:
    try:
        return uuid.UUID(_text(value, "document_namespace"))
    except ValueError as exc:
        raise DatasetValidationError("document_namespace 必须是 UUID") from exc


def _split(value: object) -> Literal["smoke", "development", "held_out"]:
    if value not in {"smoke", "development", "held_out"}:
        raise DatasetValidationError("split 不受支持")
    return value


def _optional_page(value: object, label: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatasetValidationError(f"{label} 必须是 1-based 正整数或 null")
