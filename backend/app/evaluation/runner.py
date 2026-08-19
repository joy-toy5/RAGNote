"""按冻结 query_id 串行执行可注入评测适配器。"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from app.evaluation.contracts import (
    CandidateHit,
    CandidateStage,
    EvaluationRun,
    ExecutorDescriptor,
    QueryExecution,
    QueryRun,
    StageRun,
    evaluation_run_id,
    sha256_json,
)
from app.evaluation.dataset import EvaluationDataset, dataset_fingerprint
from app.indexing.contracts import validate_blob_id, validate_index_version
from app.rag.retrieval_contract import RetrievalTrace


class QueryExecutor(Protocol):
    @property
    def descriptor(self) -> ExecutorDescriptor: ...

    async def execute(self, request: EvalRequest) -> QueryExecution: ...


@dataclass(frozen=True, slots=True)
class EvalRequest:
    query_id: str
    user_id: str
    query: str
    index_version: str
    retrieval_config_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.query_id, "query_id"),
            (self.user_id, "user_id"),
            (self.query, "query"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} 不能为空")
        validate_index_version(self.index_version)
        validate_blob_id(self.retrieval_config_sha256)


async def run_dataset(
    dataset: EvaluationDataset,
    executor: QueryExecutor,
    *,
    source_revision: str,
    source_fingerprint: str,
    retrieval_config: dict[str, Any],
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> EvaluationRun:
    """默认并发度为 1，避免当前 retriever 缓存造成顺序污染。"""
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise ValueError("source_revision 不能为空")
    validate_blob_id(source_fingerprint)
    _validate_top_k(retrieval_config)
    config_sha256 = sha256_json(retrieval_config)
    descriptor = executor.descriptor
    _validate_executor_descriptor(descriptor, dataset.index_version, config_sha256)
    query_runs = []
    for query in sorted(dataset.queries, key=lambda item: item.query_id):
        started_ns = clock_ns()
        execution = await _execute_query(
            executor,
            EvalRequest(
                query.query_id,
                query.user_id,
                query.query,
                dataset.index_version,
                config_sha256,
            ),
        )
        elapsed_ms = (clock_ns() - started_ns) / 1_000_000
        query_runs.append(_query_run(query.query_id, execution, elapsed_ms))
    dataset_sha256 = dataset_fingerprint(dataset)
    run_id = evaluation_run_id(
        dataset_sha256=dataset_sha256,
        index_version=dataset.index_version,
        retrieval_config_sha256=config_sha256,
        source_fingerprint=source_fingerprint,
        source_revision=source_revision,
        executor_id=descriptor.executor_id,
    )
    queries = tuple(query_runs)
    execution_sha256 = sha256_json(
        {
            "contract": "rag-note.eval-execution.v1",
            "executor": asdict(descriptor),
            "queries": [asdict(query) for query in queries],
        }
    )
    return EvaluationRun(
        schema_version=2,
        run_id=run_id,
        dataset_sha256=dataset_sha256,
        index_version=dataset.index_version,
        source_revision=source_revision,
        source_fingerprint=source_fingerprint,
        retrieval_config=dict(retrieval_config),
        retrieval_config_sha256=config_sha256,
        executor=descriptor,
        queries=queries,
        execution_sha256=execution_sha256,
    )


def execution_from_trace(
    trace: RetrievalTrace,
    *,
    expected_query_id: str,
    expected_user_id: str,
    expected_index_version: str,
    stages: tuple[StageRun, ...] = (),
    error_code: str | None = None,
) -> QueryExecution:
    """把 M2 trace 严格收敛为可评分候选，不接受 legacy。"""
    if trace.query_id != expected_query_id:
        raise ValueError("trace query_id 与冻结 Query 不一致")
    if trace.user_id != expected_user_id:
        raise ValueError("trace user_id 与冻结 Query 不一致")
    if trace.index_version != expected_index_version:
        raise ValueError("trace index_version 与评测版本不一致")
    candidates = []
    for rank, candidate in enumerate(trace.candidates, 1):
        if candidate.user_id != expected_user_id:
            raise ValueError("trace 包含跨用户候选")
        if candidate.provenance_status != "verified" or candidate.chunk_id is None:
            raise ValueError("正式评测不接受 legacy 候选")
        if candidate.index_version != expected_index_version:
            raise ValueError("候选 index_version 与评测版本不一致")
        candidate_stages = tuple(
            CandidateStage(
                stage=observation.stage,
                route=observation.route,
                rank=observation.rank,
                raw_score=observation.raw_score,
                score_direction=observation.score_direction,
                outcome=observation.outcome,
                error_code=observation.error_code,
            )
            for observation in candidate.stages
        )
        candidates.append(CandidateHit(candidate.chunk_id, rank, candidate_stages))
    return QueryExecution(
        candidates=tuple(candidates),
        predicted_no_answer=trace.no_answer,
        stages=stages,
        error_code=error_code,
    )


def load_run(path: Path) -> EvaluationRun:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("run 文件必须是 JSON object")
    values = dict(payload)
    queries = tuple(_query_run_from_dict(item) for item in values.pop("queries"))
    descriptor = ExecutorDescriptor(**values.pop("executor"))
    return EvaluationRun(queries=queries, executor=descriptor, **values)


def write_run(run: EvaluationRun, path: Path) -> None:
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(run.to_json())


def _query_run(query_id: str, execution: QueryExecution, elapsed_ms: float) -> QueryRun:
    end_to_end = StageRun(
        stage="end_to_end",
        route="evaluation_wrapper",
        outcome="failed" if execution.error_code else "success",
        wall_time_ms=elapsed_ms,
        error_code=execution.error_code,
    )
    return QueryRun(
        query_id=query_id,
        candidates=execution.candidates,
        predicted_no_answer=execution.predicted_no_answer,
        stages=(*execution.stages, end_to_end),
        error_code=execution.error_code,
    )


async def _execute_query(
    executor: QueryExecutor,
    request: EvalRequest,
) -> QueryExecution:
    try:
        return await executor.execute(request)
    except (TypeError, ValueError):
        raise
    except Exception as exc:
        return QueryExecution(
            candidates=(),
            predicted_no_answer=False,
            error_code=f"EXECUTOR_EXCEPTION:{type(exc).__name__}",
        )


def _query_run_from_dict(payload: dict[str, Any]) -> QueryRun:
    values = dict(payload)
    values["candidates"] = tuple(
        _candidate_from_dict(item) for item in values.get("candidates", [])
    )
    values["stages"] = tuple(StageRun(**item) for item in values.get("stages", []))
    return QueryRun(**values)


def _candidate_from_dict(payload: dict[str, Any]) -> CandidateHit:
    values = dict(payload)
    values["stages"] = tuple(
        CandidateStage(**item) for item in values.get("stages", [])
    )
    return CandidateHit(**values)


def _validate_top_k(config: dict[str, Any]) -> None:
    top_k = config.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 20:
        raise ValueError("正式 M3 run 的 top_k 必须至少为 20")


def _validate_executor_descriptor(
    descriptor: ExecutorDescriptor,
    index_version: str,
    retrieval_config_sha256: str,
) -> None:
    if not isinstance(descriptor, ExecutorDescriptor):
        raise TypeError("执行器必须提供 ExecutorDescriptor")
    if descriptor.index_version != index_version:
        raise ValueError("执行器 index_version attestation 不一致")
    if descriptor.retrieval_config_sha256 != retrieval_config_sha256:
        raise ValueError("执行器 retrieval config attestation 不一致")
