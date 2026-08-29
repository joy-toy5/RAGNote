"""评测运行、阶段观测和确定性身份的纯契约。"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Literal

from app.indexing.contracts import validate_blob_id, validate_index_version

StageOutcome = Literal[
    "success",
    "degraded",
    "skipped",
    "failed",
    "not_instrumented",
]

BINARY_RELEVANCE_THRESHOLD = 2


def canonical_json_bytes(value: object) -> bytes:
    """生成不接受 NaN/Infinity 的稳定 JSON 字节。"""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def deterministic_document_id(
    namespace: str | uuid.UUID,
    corpus_item_id: str,
) -> str:
    """仅为冻结评测装载生成稳定 UUID，不改变生产上传身份。"""
    parsed_namespace = _uuid(namespace, "document namespace")
    normalized_item_id = _nonempty(corpus_item_id, "corpus_item_id")
    return str(uuid.uuid5(parsed_namespace, unicodedata.normalize("NFC", normalized_item_id)))


def evaluation_run_id(
    *,
    dataset_sha256: str,
    index_version: str,
    retrieval_config_sha256: str,
    source_fingerprint: str,
    source_revision: str,
    executor_id: str,
) -> str:
    """从冻结输入身份计算 run ID，不包含波动的时延结果。"""
    return sha256_json(
        {
            "contract": "rag-note.eval-run.v2",
            "dataset_sha256": dataset_sha256,
            "index_version": index_version,
            "retrieval_config_sha256": retrieval_config_sha256,
            "source_fingerprint": source_fingerprint,
            "source_revision": source_revision,
            "executor_id": executor_id,
        }
    )


ANSWER_PATHS = ("retrieval_only", "generated")


@dataclass(frozen=True, slots=True)
class ExecutorDescriptor:
    """执行器对本次索引和检索配置的显式声明。"""

    executor_id: str
    index_version: str
    retrieval_config_sha256: str
    answer_path: str = "retrieval_only"
    """本次执行是否真的跑了生成层（RAG-008）。

    离线 run 只到检索为止，`predicted_no_answer` 只能来自检索层门禁，所以
    `false_answer_rate` 是「门禁单独的漏判率」，不是系统答错率。不标出来，
    这个数会被读成后者。默认 retrieval_only：旧 run 的 JSON 里没有这个键，
    反序列化时取默认值，与它们的实际执行方式一致。
    """

    def __post_init__(self) -> None:
        _nonempty(self.executor_id, "executor_id")
        validate_index_version(self.index_version)
        validate_blob_id(self.retrieval_config_sha256)
        if self.answer_path not in ANSWER_PATHS:
            raise ValueError(f"answer_path 必须是 {ANSWER_PATHS} 之一")

    def execution_identity_v1(self) -> dict[str, str]:
        """`rag-note.eval-execution.v1` 摘要用的字段集，**冻结不再增补**。

        `execution_sha256` 摘的就是这一份。v1 已经有三个冻结 run 依赖，往里加
        字段会让它们全部加载失败（`__post_init__` 会重算校验），所以后加的
        `answer_path` 显式落在 v1 之外 —— 契约串带版本号，v1 就该一直是 v1。
        报告元数据走 `asdict`，仍然会带上新字段，可见性不受影响。
        """
        return {
            "executor_id": self.executor_id,
            "index_version": self.index_version,
            "retrieval_config_sha256": self.retrieval_config_sha256,
        }


@dataclass(frozen=True, slots=True)
class CandidateStage:
    stage: str
    route: str
    rank: int
    raw_score: float | None = None
    score_direction: Literal["higher_is_better", "lower_is_better"] | None = None
    outcome: StageOutcome = "success"
    error_code: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.stage, "candidate stage")
        _nonempty(self.route, "candidate route")
        _positive_int(self.rank, "candidate rank")
        _optional_finite(self.raw_score, "candidate raw_score")
        if (self.raw_score is None) != (self.score_direction is None):
            raise ValueError("候选分数与方向必须同时提供")
        if self.score_direction not in {
            None,
            "higher_is_better",
            "lower_is_better",
        }:
            raise ValueError("候选分数方向不受支持")
        _stage_outcome(self.outcome)
        _validate_outcome_error(self.outcome, self.error_code, "候选阶段")


@dataclass(frozen=True, slots=True)
class CandidateHit:
    chunk_id: str
    rank: int
    stages: tuple[CandidateStage, ...] = ()

    def __post_init__(self) -> None:
        validate_blob_id(self.chunk_id)
        _positive_int(self.rank, "candidate rank")
        _unique_stage_keys(self.stages, "同一候选")


@dataclass(frozen=True, slots=True)
class StageRun:
    stage: str
    outcome: StageOutcome
    route: str | None = None
    wall_time_ms: float | None = None
    call_count: int | None = None
    llm_input_tokens: int | None = None
    llm_output_tokens: int | None = None
    embedding_calls: int | None = None
    embedding_items: int | None = None
    reranker_calls: int | None = None
    reranker_pairs: int | None = None
    gpu_inference_ms: float | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.stage, "stage")
        if self.route is not None:
            _nonempty(self.route, "stage route")
        _stage_outcome(self.outcome)
        _validate_outcome_error(self.outcome, self.error_code, "运行阶段")
        _optional_nonnegative_float(self.wall_time_ms, "wall_time_ms")
        _optional_nonnegative_float(self.gpu_inference_ms, "gpu_inference_ms")
        for name in (
            "call_count",
            "llm_input_tokens",
            "llm_output_tokens",
            "embedding_calls",
            "embedding_items",
            "reranker_calls",
            "reranker_pairs",
        ):
            _optional_nonnegative_int(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class QueryExecution:
    candidates: tuple[CandidateHit, ...]
    predicted_no_answer: bool
    stages: tuple[StageRun, ...] = ()
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.predicted_no_answer, bool):
            raise TypeError("predicted_no_answer 必须是布尔值")
        ranks = [candidate.rank for candidate in self.candidates]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("候选 rank 必须从 1 连续递增")
        chunk_ids = [candidate.chunk_id for candidate in self.candidates]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("同一 Query 不能包含重复 chunk_id")
        _unique_stage_keys(self.stages, "同一 Query")
        candidate_stage_failed = any(
            stage.outcome == "failed"
            for candidate in self.candidates
            for stage in candidate.stages
        )
        if self.error_code is not None:
            _nonempty(self.error_code, "error_code")
            if self.predicted_no_answer:
                raise ValueError("运行错误不能声明为正确拒答")
        if any(stage.outcome == "failed" for stage in self.stages) or candidate_stage_failed:
            if self.error_code is None:
                raise ValueError("失败阶段必须提升为 Query error_code")


@dataclass(frozen=True, slots=True)
class QueryRun:
    query_id: str
    candidates: tuple[CandidateHit, ...]
    predicted_no_answer: bool
    stages: tuple[StageRun, ...] = ()
    error_code: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.query_id, "query_id")
        QueryExecution(
            candidates=self.candidates,
            predicted_no_answer=self.predicted_no_answer,
            stages=self.stages,
            error_code=self.error_code,
        )


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    schema_version: int
    run_id: str
    dataset_sha256: str
    index_version: str
    source_revision: str
    source_fingerprint: str
    retrieval_config: dict[str, Any]
    retrieval_config_sha256: str
    executor: ExecutorDescriptor
    queries: tuple[QueryRun, ...]
    execution_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("run schema_version 必须为 2")
        validate_blob_id(self.run_id)
        validate_blob_id(self.dataset_sha256)
        validate_index_version(self.index_version)
        _nonempty(self.source_revision, "source_revision")
        validate_blob_id(self.source_fingerprint)
        validate_blob_id(self.retrieval_config_sha256)
        if sha256_json(self.retrieval_config) != self.retrieval_config_sha256:
            raise ValueError("retrieval_config SHA-256 不一致")
        if self.executor.index_version != self.index_version:
            raise ValueError("执行器声明的 index_version 不一致")
        if self.executor.retrieval_config_sha256 != self.retrieval_config_sha256:
            raise ValueError("执行器声明的检索配置摘要不一致")
        expected_run_id = evaluation_run_id(
            dataset_sha256=self.dataset_sha256,
            index_version=self.index_version,
            retrieval_config_sha256=self.retrieval_config_sha256,
            source_fingerprint=self.source_fingerprint,
            source_revision=self.source_revision,
            executor_id=self.executor.executor_id,
        )
        if self.run_id != expected_run_id:
            raise ValueError("run_id 与冻结输入身份不一致")
        query_ids = [query.query_id for query in self.queries]
        if query_ids != sorted(query_ids) or len(query_ids) != len(set(query_ids)):
            raise ValueError("run Query 必须唯一并按 query_id 排序")
        validate_blob_id(self.execution_sha256)
        expected_execution = sha256_json(
            {
                "contract": "rag-note.eval-execution.v1",
                "executor": self.executor.execution_identity_v1(),
                "queries": [asdict(query) for query in self.queries],
            }
        )
        if self.execution_sha256 != expected_execution:
            raise ValueError("execution_sha256 与执行载荷不一致")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"


def _uuid(value: str | uuid.UUID, label: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是 UUID") from exc


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 不能为空")
    return value


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} 必须是正整数")
    return value


def _optional_nonnegative_int(value: int | None, label: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} 必须是非负整数或 null")


def _optional_finite(value: Real | None, label: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{label} 必须是有限实数或 null")


def _optional_nonnegative_float(value: Real | None, label: str) -> None:
    _optional_finite(value, label)
    if value is not None and value < 0:
        raise ValueError(f"{label} 必须是非负实数或 null")


def _stage_outcome(value: str) -> None:
    if value not in {
        "success",
        "degraded",
        "skipped",
        "failed",
        "not_instrumented",
    }:
        raise ValueError("阶段 outcome 不受支持")


def _validate_outcome_error(
    outcome: StageOutcome,
    error_code: str | None,
    label: str,
) -> None:
    if outcome == "failed":
        if error_code is None:
            raise ValueError(f"{label}失败时必须携带 error_code")
        _nonempty(error_code, f"{label} error_code")
        return
    if outcome == "success" and error_code is not None:
        raise ValueError(f"{label}成功时不能携带 error_code")
    if error_code is not None:
        _nonempty(error_code, f"{label} error_code")


def _unique_stage_keys(stages: tuple[Any, ...], label: str) -> None:
    keys = [(stage.stage, stage.route) for stage in stages]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label}的 (stage, route) 必须唯一")
