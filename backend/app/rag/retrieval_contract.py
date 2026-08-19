from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from numbers import Real
from typing import Any, Literal

from app.indexing.contracts import (
    validate_blob_id,
    validate_document_id,
    validate_document_revision,
    validate_index_version,
)


ProvenanceStatus = Literal["verified", "legacy_index_only"]
ScoreDirection = Literal["higher_is_better", "lower_is_better"]
StageOutcome = Literal["success", "degraded", "skipped"]


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """候选证据在规范化提取文本中的稳定位置。"""

    page_number: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    text_sha256: str | None = None
    text_uri: str | None = None

    def __post_init__(self) -> None:
        if self.page_number is not None:
            if isinstance(self.page_number, bool) or not isinstance(
                self.page_number, int
            ):
                raise TypeError("页码必须是整数")
            if self.page_number < 1:
                raise ValueError("页码必须从 1 开始")
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("字符区间必须同时提供起点和终点")
        if self.char_start is not None:
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (self.char_start, self.char_end)
            ):
                raise TypeError("字符区间必须由整数组成")
            if self.char_start < 0 or self.char_end <= self.char_start:
                raise ValueError("字符区间必须是非空的 0-based 右开区间")
        if self.text_sha256 is not None:
            validate_blob_id(self.text_sha256)
        if (self.text_sha256 is None) != (self.text_uri is None):
            raise ValueError("规范化文本摘要与持久 URI 必须同时提供")
        if self.text_uri is not None and (
            not isinstance(self.text_uri, str) or not self.text_uri.strip()
        ):
            raise ValueError("规范化文本持久 URI 不能为空")


@dataclass(frozen=True, slots=True)
class StageObservation:
    """候选在一个检索阶段中的原始排名和分数。"""

    stage: str
    route: str
    rank: int
    raw_score: float | None = None
    score_direction: ScoreDirection | None = None
    outcome: StageOutcome = "success"
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not self.stage or not self.route:
            raise ValueError("阶段和检索路由不能为空")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("排名必须是整数")
        if self.rank < 1:
            raise ValueError("排名必须从 1 开始")
        if (self.raw_score is None) != (self.score_direction is None):
            raise ValueError("原始分数与分数方向必须同时提供")
        if self.raw_score is not None:
            if isinstance(self.raw_score, bool) or not isinstance(
                self.raw_score, Real
            ):
                raise TypeError("原始分数必须是实数")
            if not math.isfinite(float(self.raw_score)):
                raise ValueError("原始分数必须是有限值")
        if self.outcome == "success" and self.error_code is not None:
            raise ValueError("成功阶段不能携带错误码")
        if self.outcome not in {"success", "degraded", "skipped"}:
            raise ValueError("阶段结果不受支持")
        if self.score_direction not in {
            None,
            "higher_is_better",
            "lower_is_better",
        }:
            raise ValueError("分数方向不受支持")


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """贯穿召回、重排和上下文选择的不可变候选。"""

    candidate_id: str
    user_id: str
    source_type: Literal["knowledge_base", "note"]
    content: str
    display_name: str
    provenance_status: ProvenanceStatus
    blob_id: str | None = None
    document_id: str | None = None
    document_revision: int | None = None
    chunk_id: str | None = None
    index_version: str | None = None
    legacy_chunk_id: str | None = None
    source_uri: str | None = None
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    image_paths: tuple[str, ...] = ()
    stages: tuple[StageObservation, ...] = ()
    selected_for_context: bool = False
    cited: bool = False

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.candidate_id, self.user_id, self.content)
        ):
            raise ValueError("候选身份、用户和正文不能为空")
        if self.source_type not in {"knowledge_base", "note"}:
            raise ValueError("候选来源类型不受支持")
        if self.provenance_status not in {"verified", "legacy_index_only"}:
            raise ValueError("候选 provenance 状态不受支持")
        if self.document_revision is not None:
            validate_document_revision(self.document_revision)
        if self.provenance_status == "verified":
            required = (
                self.blob_id,
                self.document_id,
                self.document_revision,
                self.chunk_id,
                self.index_version,
                self.source_uri,
            )
            if any(value is None for value in required):
                raise ValueError("已验证候选必须携带完整稳定身份")
            if not isinstance(self.source_uri, str) or not self.source_uri.strip():
                raise ValueError("已验证候选必须携带非空持久 URI")
            validate_blob_id(self.blob_id)
            validate_document_id(self.document_id)
            validate_document_revision(self.document_revision)
            validate_blob_id(self.chunk_id)
            validate_index_version(self.index_version)
            if not any(
                span.char_start is not None
                and span.text_sha256 is not None
                and span.text_uri is not None
                for span in self.evidence_spans
            ):
                raise ValueError("已验证候选必须携带可校验证据区间")

    @property
    def reranker_text(self) -> str:
        if self.source_type == "note":
            return f"[来源：笔记《{self.display_name}》]\n{self.content}"
        return f"[来源：知识库《{self.display_name}》]\n{self.content}"

    def observe(self, observation: StageObservation) -> RetrievalCandidate:
        return replace(self, stages=(*self.stages, observation))

    def select_for_context(self) -> RetrievalCandidate:
        return replace(self, selected_for_context=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """一次查询的结构化候选轨迹。"""

    query_id: str
    user_id: str
    candidates: tuple[RetrievalCandidate, ...]
    index_version: str | None = None
    no_answer: bool = False

    def __post_init__(self) -> None:
        if not self.query_id or not self.user_id:
            raise ValueError("查询和用户身份不能为空")
        if any(candidate.user_id != self.user_id for candidate in self.candidates):
            raise ValueError("查询轨迹不能包含其他用户的候选")
        candidate_versions = {
            candidate.index_version
            for candidate in self.candidates
            if candidate.index_version is not None
        }
        if self.index_version is not None:
            validate_index_version(self.index_version)
            if candidate_versions and candidate_versions != {self.index_version}:
                raise ValueError("查询级 index_version 与候选版本不一致")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def candidate_from_document(
    document: Any,
    *,
    user_id: str,
    rank: int,
) -> RetrievalCandidate:
    """把 LangChain Document 转成不丢身份的候选。"""

    metadata = dict(getattr(document, "metadata", {}) or {})
    metadata_user_id = metadata.get("user_id")
    if metadata_user_id is not None and str(metadata_user_id) != user_id:
        raise ValueError("候选 metadata 用户与请求用户不一致")
    source_type = metadata.get("source_type", "knowledge_base")
    if source_type not in {"knowledge_base", "note"}:
        raise ValueError(f"不支持的候选来源类型: {source_type}")

    raw_chunk_id = metadata.get("chunk_id")
    legacy_chunk_id = getattr(document, "id", None) or metadata.get(
        "legacy_chunk_id"
    )
    display_name = (
        metadata.get("title", "无标题")
        if source_type == "note"
        else metadata.get("original_filename", "知识库文档")
    )

    evidence_spans = _evidence_spans_from_metadata(metadata)

    stable_values = (
        metadata.get("blob_id"),
        metadata.get("document_id"),
        metadata.get("document_revision"),
        raw_chunk_id,
        metadata.get("index_version"),
        metadata.get("source_uri"),
    )
    has_verified_span = any(
        span.char_start is not None
        and span.text_sha256 is not None
        and span.text_uri is not None
        for span in evidence_spans
    )
    verified = all(value is not None for value in stable_values) and has_verified_span
    provenance_status: ProvenanceStatus = (
        "verified" if verified else "legacy_index_only"
    )
    chunk_id = raw_chunk_id if verified else None
    candidate_id = chunk_id or legacy_chunk_id or f"query-local:{rank}"
    image_paths = metadata.get("image_paths") or ()
    if isinstance(image_paths, str):
        image_paths = (image_paths,)

    return RetrievalCandidate(
        candidate_id=str(candidate_id),
        user_id=user_id,
        source_type=source_type,
        content=str(getattr(document, "page_content", "")),
        display_name=str(display_name),
        provenance_status=provenance_status,
        blob_id=metadata.get("blob_id") if verified else None,
        document_id=metadata.get("document_id") if verified else None,
        document_revision=metadata.get("document_revision") if verified else None,
        chunk_id=chunk_id,
        index_version=metadata.get("index_version") if verified else None,
        legacy_chunk_id=legacy_chunk_id,
        source_uri=metadata.get("source_uri") if verified else None,
        evidence_spans=evidence_spans,
        image_paths=tuple(str(path) for path in image_paths),
        stages=(
            StageObservation(
                stage="retrieval",
                route="legacy_hybrid",
                rank=rank,
            ),
        ),
    )


def _evidence_spans_from_metadata(
    metadata: dict[str, Any],
) -> tuple[EvidenceSpan, ...]:
    page_number = metadata.get("page_number")
    if page_number is None:
        legacy_page = metadata.get("page")
        if (
            isinstance(legacy_page, int)
            and not isinstance(legacy_page, bool)
            and legacy_page >= 0
        ):
            page_number = legacy_page + 1

    values = {
        "page_number": page_number,
        "char_start": metadata.get("char_start"),
        "char_end": metadata.get("char_end"),
        "text_sha256": metadata.get("source_text_sha256"),
        "text_uri": metadata.get("source_text_uri"),
    }
    if all(value is None for value in values.values()):
        return ()
    try:
        return (EvidenceSpan(**values),)
    except (TypeError, ValueError):
        try:
            return (EvidenceSpan(page_number=page_number),)
        except (TypeError, ValueError):
            return ()


def single_index_version(
    candidates: list[RetrievalCandidate],
) -> str | None:
    """仅当全部版本化候选属于同一索引版本时返回查询级版本。"""
    versions = {
        candidate.index_version
        for candidate in candidates
        if candidate.index_version is not None
    }
    return next(iter(versions)) if len(versions) == 1 else None


def validate_rerank_scores(scores: Any, expected_count: int) -> tuple[float, ...]:
    """验证 Reranker 对每个候选都返回一个有限分数。"""
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise TypeError("候选数量必须是整数")
    if expected_count < 0:
        raise ValueError("候选数量不能为负数")
    try:
        raw_scores = tuple(scores)
    except TypeError as exc:
        raise ValueError("Reranker 分数必须是可迭代序列") from exc
    if len(raw_scores) != expected_count:
        raise ValueError("Reranker 返回数量与候选数量不一致")

    validated = []
    for score in raw_scores:
        if isinstance(score, bool):
            raise ValueError("Reranker 分数必须是有限实数")
        try:
            value = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError("Reranker 分数必须是有限实数") from exc
        if not math.isfinite(value):
            raise ValueError("Reranker 分数必须是有限实数")
        validated.append(value)
    return tuple(validated)
