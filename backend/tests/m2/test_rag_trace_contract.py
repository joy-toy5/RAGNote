from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def traces() -> dict[str, dict[str, object]]:
    backend_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "tests/baseline/fake_rag_harness.py",
            "--scenario",
            "all",
            "--samples",
            "1",
            "--warmup",
            "0",
        ],
        cwd=backend_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
        env={
            **os.environ,
            "PIP_NO_INDEX": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "LANGSMITH_TRACING": "false",
        },
    )
    payload = json.loads(result.stdout)
    return {item["scenario"]: item for item in payload["scenarios"]}


def test_success_trace_preserves_verified_and_legacy_candidate_identity(
    traces: dict[str, dict[str, object]],
) -> None:
    trace = traces["success_3_docs"]["retrieval_trace"]
    candidates = trace["candidates"]

    assert trace["user_id"] == "eval-user-a"
    assert trace["index_version"] == "f" * 64
    assert [candidate["provenance_status"] for candidate in candidates] == [
        "verified",
        "verified",
        "legacy_index_only",
    ]
    assert all(candidate["selected_for_context"] for candidate in candidates)
    assert all(
        [stage["stage"] for stage in candidate["stages"]]
        == ["retrieval", "rerank"]
        for candidate in candidates
    )


def test_rerank_degradation_keeps_candidate_ids_and_records_failure(
    traces: dict[str, dict[str, object]],
) -> None:
    success_candidates = traces["success_3_docs"]["retrieval_trace"]["candidates"]
    degraded_candidates = traces["rerank_failure"]["retrieval_trace"]["candidates"]

    assert [candidate["candidate_id"] for candidate in degraded_candidates] == [
        candidate["candidate_id"] for candidate in reversed(success_candidates)
    ]
    assert all(
        candidate["stages"][-1]["outcome"] == "degraded"
        and candidate["stages"][-1]["error_code"] == "RERANK_FAILED"
        for candidate in degraded_candidates
    )


def test_summary_timeout_returns_the_same_verified_trace(
    traces: dict[str, dict[str, object]],
) -> None:
    scenario = traces["summary_timeout"]
    candidate = scenario["retrieval_trace"]["candidates"][0]

    assert scenario["outcome"] == "summary_timeout"
    assert candidate["provenance_status"] == "verified"
    assert candidate["chunk_id"] == candidate["candidate_id"]
    assert candidate["selected_for_context"]
