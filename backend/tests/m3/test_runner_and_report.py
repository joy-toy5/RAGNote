from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

import app.evaluation.reporting as reporting_module
from app.evaluation.contracts import (
    CandidateHit,
    CandidateStage,
    ExecutorDescriptor,
    QueryExecution,
    StageRun,
    sha256_json,
)
from app.evaluation.dataset import EvaluationDataset, load_dataset
from app.evaluation.qrels import CompiledQrels, compile_qrels
from app.evaluation.reporting import evaluate_run, render_report_json, render_report_markdown
from app.evaluation.runner import (
    EvalRequest,
    execution_from_trace,
    load_run,
    run_dataset,
    write_run,
)
from app.rag.retrieval_contract import RetrievalCandidate, RetrievalTrace
from scripts.m3_evaluate import main as cli_main


def _smoke_manifest(backend_root: Path) -> Path:
    return backend_root / "evals/datasets/m3_smoke_v1/manifest.json"


RETRIEVAL_CONFIG = {"name": "fixture", "top_k": 20}


class FixtureExecutor:
    def __init__(
        self,
        relevant_by_query: dict[str, str],
        descriptor: ExecutorDescriptor,
    ) -> None:
        self.relevant_by_query = relevant_by_query
        self.descriptor = descriptor
        self.requests: list[EvalRequest] = []

    async def execute(self, request: EvalRequest) -> QueryExecution:
        self.requests.append(request)
        chunk_id = self.relevant_by_query.get(request.query_id)
        candidates = () if chunk_id is None else (CandidateHit(chunk_id, rank=1),)
        return QueryExecution(
            candidates=candidates,
            predicted_no_answer=chunk_id is None,
            stages=(
                StageRun(
                    stage="retrieval",
                    route="fixture",
                    outcome="success",
                    wall_time_ms=1.0,
                    call_count=1,
                ),
            ),
        )


class FailingExecutor(FixtureExecutor):
    async def execute(self, request: EvalRequest) -> QueryExecution:
        if request.query_id == "smoke-a-absent":
            raise RuntimeError("injected executor failure")
        return await super().execute(request)


def _executor(
    dataset: EvaluationDataset,
    relevant: dict[str, str],
    *,
    config: dict[str, object] = RETRIEVAL_CONFIG,
    executor_type: type[FixtureExecutor] = FixtureExecutor,
) -> FixtureExecutor:
    descriptor = ExecutorDescriptor(
        executor_id="fixture-executor-v1",
        index_version=dataset.index_version,
        retrieval_config_sha256=sha256_json(config),
    )
    return executor_type(relevant, descriptor)


def _relevant_chunks(compiled: CompiledQrels) -> dict[str, str]:
    return {entry.query_id: entry.chunk_ids[0] for entry in compiled.entries}


def _clock(values: list[int]) -> callable:
    iterator: Iterator[int] = iter(values)
    return lambda: next(iterator)


def test_runner_uses_frozen_query_ids_and_sorted_serial_execution(
    backend_root: Path,
) -> None:
    dataset = load_dataset(_smoke_manifest(backend_root))
    compiled = compile_qrels(dataset)
    relevant = _relevant_chunks(compiled)
    executor = _executor(dataset, relevant)
    clock_values = [value * 1_000_000 for value in range(len(dataset.queries) * 2)]

    run = asyncio.run(
        run_dataset(
            dataset,
            executor,
            source_revision="cb0d3d3",
            source_fingerprint="a" * 64,
            retrieval_config=RETRIEVAL_CONFIG,
            clock_ns=_clock(clock_values),
        )
    )

    expected_ids = sorted(query.query_id for query in dataset.queries)
    assert [request.query_id for request in executor.requests] == expected_ids
    assert all(
        request.index_version == dataset.index_version
        and request.retrieval_config_sha256 == sha256_json(RETRIEVAL_CONFIG)
        for request in executor.requests
    )
    assert [query.query_id for query in run.queries] == expected_ids
    assert all(query.stages[-1].stage == "end_to_end" for query in run.queries)
    assert run.executor == executor.descriptor
    assert len(run.execution_sha256) == 64


def test_runner_rejects_executor_attestation_mismatch(backend_root: Path) -> None:
    dataset = load_dataset(_smoke_manifest(backend_root))
    executor = _executor(dataset, {})
    executor.descriptor = replace(executor.descriptor, index_version="b" * 64)

    with pytest.raises(ValueError, match="attestation"):
        asyncio.run(
            run_dataset(
                dataset,
                executor,
                source_revision="cb0d3d3",
                source_fingerprint="a" * 64,
                retrieval_config=RETRIEVAL_CONFIG,
            )
        )

    assert executor.requests == []


def test_runner_rejects_top_k_below_metrics_cutoff(backend_root: Path) -> None:
    dataset = load_dataset(_smoke_manifest(backend_root))
    config = {"name": "fixture", "top_k": 10}

    with pytest.raises(ValueError, match="至少为 20"):
        asyncio.run(
            run_dataset(
                dataset,
                _executor(dataset, {}, config=config),
                source_revision="cb0d3d3",
                source_fingerprint="a" * 64,
                retrieval_config=config,
            )
        )


def test_report_is_deterministic_and_exposes_failure_diagnostics(
    backend_root: Path,
) -> None:
    dataset = load_dataset(_smoke_manifest(backend_root))
    compiled = compile_qrels(dataset)
    relevant = _relevant_chunks(compiled)
    executor = _executor(dataset, relevant)
    run = asyncio.run(
        run_dataset(
            dataset,
            executor,
            source_revision="cb0d3d3",
            source_fingerprint="b" * 64,
            retrieval_config=RETRIEVAL_CONFIG,
            clock_ns=_clock([value * 1_000_000 for value in range(8)]),
        )
    )

    report = evaluate_run(dataset, compiled, run)

    assert report.aggregate["recall_at_20"] == 1
    assert report.aggregate["cross_user_hit_count"] == 0
    assert report.hard_gate["status"] == "PASS"
    assert all(detail["stages"] for detail in report.queries)
    assert render_report_json(report) == render_report_json(report)
    assert render_report_markdown(report) == render_report_markdown(report)


def test_report_rejects_unknown_chunk(backend_root: Path) -> None:
    dataset = load_dataset(_smoke_manifest(backend_root))
    compiled = compile_qrels(dataset)
    relevant = _relevant_chunks(compiled)
    executor = _executor(dataset, relevant)
    executor.relevant_by_query["smoke-a-429"] = "f" * 64
    run = asyncio.run(
        run_dataset(
            dataset,
            executor,
            source_revision="cb0d3d3",
            source_fingerprint="c" * 64,
            retrieval_config=RETRIEVAL_CONFIG,
            clock_ns=_clock([value * 1_000_000 for value in range(8)]),
        )
    )

    with pytest.raises(ValueError, match="冻结 corpus"):
        evaluate_run(dataset, compiled, run)


def test_trace_adapter_rejects_legacy_candidate() -> None:
    trace = RetrievalTrace(
        query_id="runtime-random-id",
        user_id="eval-user-a",
        candidates=(
            RetrievalCandidate(
                candidate_id="legacy-1",
                user_id="eval-user-a",
                source_type="note",
                content="legacy note",
                display_name="note",
                provenance_status="legacy_index_only",
                legacy_chunk_id="legacy-1",
            ),
        ),
        index_version="a" * 64,
    )

    with pytest.raises(ValueError, match="legacy"):
        execution_from_trace(
            trace,
            expected_query_id="runtime-random-id",
            expected_user_id="eval-user-a",
            expected_index_version="a" * 64,
        )


def test_trace_adapter_rejects_runtime_random_query_id() -> None:
    trace = RetrievalTrace(
        query_id="runtime-random-id",
        user_id="eval-user-a",
        candidates=(),
        no_answer=True,
    )

    with pytest.raises(ValueError, match="冻结 Query"):
        execution_from_trace(
            trace,
            expected_query_id="smoke-a-absent",
            expected_user_id="eval-user-a",
            expected_index_version="a" * 64,
        )


def test_trace_adapter_rejects_trace_level_user_mismatch() -> None:
    trace = RetrievalTrace(
        query_id="smoke-a-absent",
        user_id="eval-user-b",
        candidates=(),
        no_answer=True,
    )

    with pytest.raises(ValueError, match="trace user_id"):
        execution_from_trace(
            trace,
            expected_query_id="smoke-a-absent",
            expected_user_id="eval-user-a",
            expected_index_version="a" * 64,
        )


def test_cross_user_candidate_is_reported_as_security_failure(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    dataset = load_dataset(_smoke_manifest(backend_root))
    compiled = compile_qrels(dataset)
    relevant = _relevant_chunks(compiled)
    foreign_chunk = next(
        chunk.chunk_id for chunk in dataset.chunks if chunk.user_id == "eval-user-b"
    )
    relevant["smoke-a-tenant-excluded"] = foreign_chunk
    run = asyncio.run(
        run_dataset(
            dataset,
            _executor(dataset, relevant),
            source_revision="cb0d3d3",
            source_fingerprint="d" * 64,
            retrieval_config=RETRIEVAL_CONFIG,
            clock_ns=_clock([value * 1_000_000 for value in range(8)]),
        )
    )

    report = evaluate_run(dataset, compiled, run)

    assert report.aggregate["cross_user_hit_count"] == 1
    assert report.hard_gate == {
        "status": "FAIL",
        "passed": False,
        "violations": ["CROSS_USER_HIT"],
    }
    assert {failure["code"] for failure in report.failures} >= {
        "CROSS_USER_HIT",
        "FALSE_ANSWER",
    }
    run_path = tmp_path / "cross-user-run.json"
    write_run(run, run_path)
    assert cli_main(
        [
            "evaluate",
            "--dataset",
            str(_smoke_manifest(backend_root)),
            "--run",
            str(run_path),
            "--output-dir",
            str(tmp_path / "cross-user-report"),
        ]
    ) == 1


def test_executor_failure_is_recorded_and_not_counted_as_rejection(
    backend_root: Path,
) -> None:
    dataset = load_dataset(_smoke_manifest(backend_root))
    compiled = compile_qrels(dataset)
    relevant = _relevant_chunks(compiled)
    run = asyncio.run(
        run_dataset(
            dataset,
            _executor(dataset, relevant, executor_type=FailingExecutor),
            source_revision="cb0d3d3",
            source_fingerprint="9" * 64,
            retrieval_config=RETRIEVAL_CONFIG,
            clock_ns=_clock([value * 1_000_000 for value in range(8)]),
        )
    )

    failed = next(query for query in run.queries if query.query_id == "smoke-a-absent")
    report = evaluate_run(dataset, compiled, run)

    assert failed.error_code == "EXECUTOR_EXCEPTION:RuntimeError"
    assert failed.predicted_no_answer is False
    assert report.aggregate["no_answer_run_error_count"] == 1
    assert report.hard_gate["status"] == "FAIL"
    assert {failure["code"] for failure in report.failures} >= {
        "FALSE_ANSWER",
        "RUN_ERROR",
    }


def test_quality_fingerprint_excludes_timing_noise(backend_root: Path) -> None:
    dataset = load_dataset(_smoke_manifest(backend_root))
    compiled = compile_qrels(dataset)
    relevant = _relevant_chunks(compiled)

    first = asyncio.run(
        run_dataset(
            dataset,
            _executor(dataset, relevant),
            source_revision="cb0d3d3",
            source_fingerprint="e" * 64,
            retrieval_config=RETRIEVAL_CONFIG,
            clock_ns=_clock([value * 1_000_000 for value in range(8)]),
        )
    )
    second = asyncio.run(
        run_dataset(
            dataset,
            _executor(dataset, relevant),
            source_revision="cb0d3d3",
            source_fingerprint="e" * 64,
            retrieval_config=RETRIEVAL_CONFIG,
            clock_ns=_clock([value * 2_000_000 for value in range(8)]),
        )
    )

    first_report = evaluate_run(dataset, compiled, first)
    second_report = evaluate_run(dataset, compiled, second)
    assert first_report.quality_fingerprint == second_report.quality_fingerprint
    assert render_report_json(first_report) != render_report_json(second_report)


def test_cli_compiles_and_scores_without_overwriting(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    manifest = _smoke_manifest(backend_root)
    dataset = load_dataset(manifest)
    compiled = compile_qrels(dataset)
    relevant = _relevant_chunks(compiled)
    run = asyncio.run(
        run_dataset(
            dataset,
            _executor(dataset, relevant),
            source_revision="cb0d3d3",
            source_fingerprint="f" * 64,
            retrieval_config=RETRIEVAL_CONFIG,
            clock_ns=_clock([value * 1_000_000 for value in range(8)]),
        )
    )
    run_path = tmp_path / "run.json"
    qrels_path = tmp_path / "qrels.json"
    report_dir = tmp_path / "report"
    write_run(run, run_path)

    assert cli_main(["compile-qrels", "--dataset", str(manifest), "--output", str(qrels_path)]) == 0
    assert cli_main(
        [
            "evaluate",
            "--dataset",
            str(manifest),
            "--run",
            str(run_path),
            "--output-dir",
            str(report_dir),
        ]
    ) == 0
    assert {path.name for path in report_dir.iterdir()} == {
        "qrels.json",
        "report.json",
        "report.md",
    }
    report_payload = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert report_payload["metadata"]["run_file_sha256"]
    assert report_payload["metadata"]["execution_sha256"] == run.execution_sha256
    with pytest.raises(FileExistsError):
        cli_main(["compile-qrels", "--dataset", str(manifest), "--output", str(qrels_path)])


def test_load_run_rejects_tampered_execution_payload(
    backend_root: Path,
    tmp_path: Path,
) -> None:
    dataset = load_dataset(_smoke_manifest(backend_root))
    compiled = compile_qrels(dataset)
    run = asyncio.run(
        run_dataset(
            dataset,
            _executor(dataset, _relevant_chunks(compiled)),
            source_revision="cb0d3d3",
            source_fingerprint="1" * 64,
            retrieval_config=RETRIEVAL_CONFIG,
            clock_ns=_clock([value * 1_000_000 for value in range(8)]),
        )
    )
    payload = json.loads(run.to_json())
    payload["queries"][0]["predicted_no_answer"] = not payload["queries"][0][
        "predicted_no_answer"
    ]
    path = tmp_path / "tampered-run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="execution_sha256"):
        load_run(path)


def test_report_bundle_does_not_publish_partial_directory(
    backend_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_dataset(_smoke_manifest(backend_root))
    compiled = compile_qrels(dataset)
    run = asyncio.run(
        run_dataset(
            dataset,
            _executor(dataset, _relevant_chunks(compiled)),
            source_revision="cb0d3d3",
            source_fingerprint="2" * 64,
            retrieval_config=RETRIEVAL_CONFIG,
            clock_ns=_clock([value * 1_000_000 for value in range(8)]),
        )
    )
    report = evaluate_run(dataset, compiled, run)
    output_dir = tmp_path / "report"

    def fail_render(_: object) -> str:
        raise RuntimeError("injected render failure")

    monkeypatch.setattr(reporting_module, "render_report_markdown", fail_render)
    with pytest.raises(RuntimeError, match="injected render failure"):
        reporting_module.write_report_bundle(report, compiled, output_dir)

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".report.tmp-*"))


def test_failed_stage_requires_error_and_query_failure() -> None:
    with pytest.raises(ValueError, match="必须携带 error_code"):
        StageRun(stage="retrieval", route="fixture", outcome="failed")

    failed = StageRun(
        stage="retrieval",
        route="fixture",
        outcome="failed",
        error_code="RETRIEVAL_FAILED",
    )
    with pytest.raises(ValueError, match="Query error_code"):
        QueryExecution(candidates=(), predicted_no_answer=False, stages=(failed,))

    candidate_failed = CandidateStage(
        stage="reranker",
        route="fixture",
        rank=1,
        outcome="failed",
        error_code="RERANK_FAILED",
    )
    with pytest.raises(ValueError, match="Query error_code"):
        QueryExecution(
            candidates=(CandidateHit("a" * 64, 1, (candidate_failed,)),),
            predicted_no_answer=False,
        )


def test_duplicate_stage_route_is_rejected() -> None:
    stage = StageRun(stage="retrieval", route="fixture", outcome="success")

    with pytest.raises(ValueError, match="必须唯一"):
        QueryExecution(
            candidates=(),
            predicted_no_answer=True,
            stages=(stage, stage),
        )
