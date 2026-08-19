"""从冻结 dataset、qrels 与 run 生成确定性 JSON/Markdown 报告。"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.evaluation.contracts import (
    BINARY_RELEVANCE_THRESHOLD,
    EvaluationRun,
    QueryRun,
    sha256_json,
)
from app.evaluation.dataset import EvaluationDataset, QueryCase, dataset_fingerprint
from app.evaluation.metrics import EvidenceJudgment, no_answer_metrics, ranking_metrics
from app.evaluation.qrels import CompiledQrel, CompiledQrels


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    schema_version: int
    metadata: dict[str, Any]
    aggregate: dict[str, Any]
    hard_gate: dict[str, Any]
    queries: tuple[dict[str, Any], ...]
    failures: tuple[dict[str, Any], ...]
    quality_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_run(
    dataset: EvaluationDataset,
    qrels: CompiledQrels,
    run: EvaluationRun,
    *,
    run_file_sha256: str | None = None,
) -> EvaluationReport:
    """严格校验运行身份后计算宏平均质量和逐 Query 诊断。"""
    _validate_run_identity(dataset, qrels, run)
    chunks = {chunk.chunk_id: chunk for chunk in dataset.chunks}
    qrels_by_query = _qrels_by_query(qrels)
    query_specs = {query.query_id: query for query in dataset.queries}
    details = []
    failures = []
    ranking_results = []
    no_answer_outcomes = []
    cross_user_hit_count = 0
    for query_run in run.queries:
        spec = query_specs[query_run.query_id]
        detail, query_failures, ranking, cross_hits = _evaluate_query(
            spec,
            query_run,
            qrels_by_query.get(spec.query_id, ()),
            chunks,
        )
        details.append(detail)
        failures.extend(query_failures)
        cross_user_hit_count += cross_hits
        if ranking is not None:
            ranking_results.append(ranking)
        no_answer_outcomes.append(
            (
                spec.answerability == "unanswerable",
                query_run.predicted_no_answer,
                query_run.error_code is not None,
            )
        )
    aggregate = _aggregate(
        ranking_results,
        no_answer_outcomes,
        run,
        cross_user_hit_count,
    )
    hard_gate = _hard_gate(aggregate)
    metadata = {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "dataset_sha256": dataset_fingerprint(dataset),
        "qrels_version": qrels.qrels_version,
        "qrels_sha256": qrels.fingerprint,
        "run_id": run.run_id,
        "execution_sha256": run.execution_sha256,
        "run_payload_sha256": sha256_json(run.to_dict()),
        "run_file_sha256": run_file_sha256,
        "index_version": run.index_version,
        "executor": asdict(run.executor),
        "source_revision": run.source_revision,
        "source_fingerprint": run.source_fingerprint,
        "retrieval_config_sha256": run.retrieval_config_sha256,
        "metrics_version": "rag-note.metrics.v2-evidence-unit",
        "binary_relevance_threshold": BINARY_RELEVANCE_THRESHOLD,
        "usage_semantics": "stage_incremental",
    }
    quality_aggregate = {
        key: value
        for key, value in aggregate.items()
        if key not in {"stage_latency_ms", "usage_totals"}
    }
    quality_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in {"run_file_sha256", "run_payload_sha256", "execution_sha256"}
    }
    quality_queries = [
        {key: value for key, value in detail.items() if key != "stages"}
        for detail in details
    ]
    quality_fingerprint = sha256_json(
        {
            "metadata": quality_metadata,
            "aggregate": quality_aggregate,
            "hard_gate": hard_gate,
            "queries": quality_queries,
        }
    )
    return EvaluationReport(
        schema_version=2,
        metadata=metadata,
        aggregate=aggregate,
        hard_gate=hard_gate,
        queries=tuple(details),
        failures=tuple(failures),
        quality_fingerprint=quality_fingerprint,
    )


def render_report_json(report: EvaluationReport) -> str:
    payload = report.to_dict()
    payload["report_sha256"] = sha256_json(payload)
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def render_report_markdown(report: EvaluationReport) -> str:
    aggregate = report.aggregate
    metadata = report.metadata
    lines = [
        "# RAG 离线评测报告",
        "",
        f"- dataset: {_md_literal(metadata['dataset_id'])}@{_md_literal(metadata['dataset_version'])}",
        f"- index_version: {_md_literal(metadata['index_version'])}",
        f"- source_revision: {_md_literal(metadata['source_revision'])}",
        f"- source_fingerprint: {_md_literal(metadata['source_fingerprint'])}",
        f"- quality_fingerprint: {_md_literal(report.quality_fingerprint)}",
        f"- hard_gate: {_md_literal(report.hard_gate['status'])}",
        "",
        "## 聚合指标",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
    ]
    for key in (
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "mrr_at_10",
        "ndcg_at_10",
        "precision_at_3",
        "no_answer_precision",
        "no_answer_recall",
        "no_answer_f1",
        "false_answer_rate",
        "cross_user_hit_count",
    ):
        lines.append(f"| `{key}` | {aggregate[key]} |")
    lines.extend(["", "## 失败诊断", ""])
    if not report.failures:
        lines.append("无。")
    else:
        for failure in report.failures:
            lines.append(
                f"- query_id={_md_literal(failure['query_id'])}; "
                f"code={_md_literal(failure['code'])}"
            )
    return "\n".join(lines) + "\n"


def write_report_bundle(
    report: EvaluationReport,
    qrels: CompiledQrels,
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(output_dir.parent)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        (staging / "report.json").write_text(
            render_report_json(report), encoding="utf-8"
        )
        (staging / "report.md").write_text(
            render_report_markdown(report), encoding="utf-8"
        )
        (staging / "qrels.json").write_text(qrels.to_json(), encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError(output_dir)
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_run_identity(
    dataset: EvaluationDataset,
    qrels: CompiledQrels,
    run: EvaluationRun,
) -> None:
    expected_dataset = dataset_fingerprint(dataset)
    if run.dataset_sha256 != expected_dataset or qrels.dataset_sha256 != expected_dataset:
        raise ValueError("dataset SHA-256 不一致")
    if qrels.schema_version != 2 or qrels.qrels_version != dataset.qrels_version:
        raise ValueError("qrels 版本与冻结数据集不一致")
    if run.index_version != dataset.index_version or qrels.index_version != dataset.index_version:
        raise ValueError("index_version 不一致")
    top_k = run.retrieval_config.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 20:
        raise ValueError("正式 M3 run 的 top_k 必须至少为 20")
    expected_queries = sorted(query.query_id for query in dataset.queries)
    if [query.query_id for query in run.queries] != expected_queries:
        raise ValueError("run Query 集合与冻结数据集不一致")


def _qrels_by_query(qrels: CompiledQrels) -> dict[str, tuple[CompiledQrel, ...]]:
    grouped: dict[str, list[CompiledQrel]] = {}
    for entry in qrels.entries:
        grouped.setdefault(entry.query_id, []).append(entry)
    return {query_id: tuple(entries) for query_id, entries in grouped.items()}


def _evaluate_query(
    spec: QueryCase,
    run: QueryRun,
    query_qrels: tuple[CompiledQrel, ...],
    chunks: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]], Any, int]:
    chunk_relevance = _chunk_relevance(query_qrels)
    candidates = []
    cross_hits = 0
    for candidate in run.candidates:
        if candidate.chunk_id not in chunks:
            raise ValueError(f"候选不属于冻结 corpus: {candidate.chunk_id}")
        chunk = chunks[candidate.chunk_id]
        cross_user = chunk.user_id != spec.user_id
        cross_hits += int(cross_user)
        candidates.append(
            {
                "chunk_id": candidate.chunk_id,
                "rank": candidate.rank,
                "relevance": chunk_relevance.get(candidate.chunk_id, 0),
                "cross_user": cross_user,
                "stages": [asdict(stage) for stage in candidate.stages],
            }
        )
    ranked_ids = [candidate.chunk_id for candidate in run.candidates]
    judgments = tuple(
        EvidenceJudgment(
            evidence_id=entry.anchor_id,
            relevance=entry.relevance,
            chunk_ids=frozenset(entry.chunk_ids),
        )
        for entry in query_qrels
    )
    ranking = (
        ranking_metrics(judgments, ranked_ids)
        if spec.answerability == "answerable"
        else None
    )
    ranked_set = set(ranked_ids)
    missing = sorted(
        entry.anchor_id
        for entry in query_qrels
        if entry.relevance >= BINARY_RELEVANCE_THRESHOLD
        and not ranked_set.intersection(entry.chunk_ids)
    )
    failures = _query_failures(spec, run, missing, cross_hits)
    detail = {
        "query_id": spec.query_id,
        "answerability": spec.answerability,
        "predicted_no_answer": run.predicted_no_answer,
        "error_code": run.error_code,
        "stages": [asdict(stage) for stage in run.stages],
        "metrics": None if ranking is None else _ranking_payload(ranking),
        "missing_relevant_anchor_ids": missing,
        "candidates": candidates,
    }
    return detail, failures, ranking, cross_hits


def _query_failures(
    spec: QueryCase,
    run: QueryRun,
    missing: list[str],
    cross_hits: int,
) -> list[dict[str, str]]:
    codes = []
    if missing:
        codes.append("MISSED_RELEVANT")
    if cross_hits:
        codes.append("CROSS_USER_HIT")
    if spec.answerability == "unanswerable" and not run.predicted_no_answer:
        codes.append("FALSE_ANSWER")
    if spec.answerability == "answerable" and run.predicted_no_answer:
        codes.append("FALSE_REJECTION")
    if run.error_code is not None:
        codes.append("RUN_ERROR")
    return [{"query_id": spec.query_id, "code": code} for code in codes]


def _ranking_payload(result: Any) -> dict[str, float]:
    return {
        **{f"recall_at_{key}": _rounded(value) for key, value in result.recall_at.items()},
        "precision_at_3": _rounded(result.precision_at_3),
        "mrr_at_10": _rounded(result.mrr_at_10),
        "ndcg_at_10": _rounded(result.ndcg_at_10),
    }


def _aggregate(
    rankings: list[Any],
    no_answer_outcomes: list[tuple[bool, bool, bool]],
    run: EvaluationRun,
    cross_user_hit_count: int,
) -> dict[str, Any]:
    no_answer = no_answer_metrics(no_answer_outcomes)
    aggregate = {
        f"recall_at_{cutoff}": _macro(
            [result.recall_at[cutoff] for result in rankings]
        )
        for cutoff in (3, 5, 10, 20)
    }
    aggregate.update(
        {
            "precision_at_3": _macro([item.precision_at_3 for item in rankings]),
            "mrr_at_10": _macro([item.mrr_at_10 for item in rankings]),
            "ndcg_at_10": _macro([item.ndcg_at_10 for item in rankings]),
            "no_answer_precision": _rounded(no_answer.precision),
            "no_answer_recall": _rounded(no_answer.recall),
            "no_answer_f1": _rounded(no_answer.f1),
            "false_answer_rate": _rounded(no_answer.false_answer_rate),
            "no_answer_run_error_count": no_answer.run_error_count,
            "cross_user_hit_count": cross_user_hit_count,
            "stage_latency_ms": _stage_latency(run),
            "usage_totals": _usage_totals(run),
        }
    )
    return aggregate


def _stage_latency(run: EvaluationRun) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = {}
    for query in run.queries:
        for stage in query.stages:
            if stage.wall_time_ms is not None:
                key = "::".join((stage.stage, stage.route or "", stage.outcome))
                grouped.setdefault(key, []).append(stage.wall_time_ms)
    return {
        stage: {
            "count": len(values),
            "p50": _rounded(_nearest_rank(values, 0.50)),
            "p95": _rounded(_nearest_rank(values, 0.95)),
            "p99": _rounded(_nearest_rank(values, 0.99)),
        }
        for stage, values in sorted(grouped.items())
    }


def _usage_totals(run: EvaluationRun) -> dict[str, int | float | None]:
    integer_names = (
        "call_count",
        "llm_input_tokens",
        "llm_output_tokens",
        "embedding_calls",
        "embedding_items",
        "reranker_calls",
        "reranker_pairs",
    )
    result: dict[str, int | float | None] = {}
    stages = [
        stage
        for query in run.queries
        for stage in query.stages
        if stage.stage != "end_to_end"
    ]
    for name in integer_names:
        observed = [getattr(stage, name) for stage in stages if getattr(stage, name) is not None]
        result[name] = None if not observed else sum(observed)
    gpu_values = [
        stage.gpu_inference_ms
        for stage in stages
        if stage.gpu_inference_ms is not None
    ]
    result["gpu_inference_ms"] = (
        None if not gpu_values else _rounded(math.fsum(gpu_values))
    )
    return result


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _macro(values: list[float]) -> float:
    if not values:
        raise ValueError("排名聚合至少需要一个有答案 Query")
    return _rounded(math.fsum(values) / len(values))


def _rounded(value: float) -> float:
    return round(float(value), 12)


def _chunk_relevance(qrels: tuple[CompiledQrel, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for entry in qrels:
        for chunk_id in entry.chunk_ids:
            result[chunk_id] = max(result.get(chunk_id, 0), entry.relevance)
    return result


def _hard_gate(aggregate: dict[str, Any]) -> dict[str, Any]:
    violations = []
    if aggregate["cross_user_hit_count"]:
        violations.append("CROSS_USER_HIT")
    if aggregate["no_answer_run_error_count"]:
        violations.append("RUN_ERROR")
    return {
        "status": "PASS" if not violations else "FAIL",
        "passed": not violations,
        "violations": violations,
    }


def _md_literal(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
