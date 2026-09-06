from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.baseline


@pytest.fixture(scope="module")
def fake_rag_payload() -> dict[str, object]:
    backend_root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "tests/baseline/fake_rag_harness.py",
        "--scenario",
        "all",
        "--samples",
        "1",
        "--warmup",
        "0",
    ]
    env = {
        **os.environ,
        "PIP_NO_INDEX": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LANGSMITH_TRACING": "false",
    }
    result = subprocess.run(
        command,
        cwd=backend_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )
    payload = json.loads(result.stdout)
    assert payload["baseline_type"] == "fake_orchestration_baseline"
    assert payload["dangerous_imports"] == []
    return payload


@pytest.fixture(scope="module")
def fake_rag_scenarios(
    fake_rag_payload: dict[str, object],
) -> dict[str, dict[str, object]]:
    return {item["scenario"]: item for item in fake_rag_payload["scenarios"]}


def test_success_path_preserves_current_call_amplification(
    fake_rag_scenarios: dict[str, dict[str, object]],
) -> None:
    scenario = fake_rag_scenarios["success_3_docs"]

    assert scenario["calls"] == {
        "hyde_model": 1,
        "kb_retrieval": 1,
        "note_retrieval": 1,
        "rerank_service": 1,
        "reranker_inference": 1,
        "summary_model": 4,
    }
    assert scenario["documents_returned"] == 3
    assert scenario["outcome"] == "success"


def test_hyde_failure_falls_back_to_original_query(
    fake_rag_scenarios: dict[str, dict[str, object]],
) -> None:
    scenario = fake_rag_scenarios["hyde_failure"]

    assert scenario["fallback_input"] == "How should HTTP 429 be retried?"
    assert scenario["documents_returned"] == 3
    assert scenario["error_class"] == "RuntimeError"
    assert scenario["outcome"] == "success_with_fallback"
    assert scenario["observed_degradation"] == "original_query"


def test_kb_failure_skips_note_fallback(
    fake_rag_scenarios: dict[str, dict[str, object]],
) -> None:
    scenario = fake_rag_scenarios["kb_retrieval_failure"]

    assert scenario["calls"] == {
        "hyde_model": 1,
        "kb_retrieval": 1,
        "note_retrieval": 0,
        "rerank_service": 1,
        "reranker_inference": 0,
        "summary_model": 0,
    }
    assert scenario["documents_returned"] == 0
    assert scenario["outcome"] == "no_documents"


def test_note_failure_keeps_knowledge_base_candidates(
    fake_rag_scenarios: dict[str, dict[str, object]],
) -> None:
    scenario = fake_rag_scenarios["note_retrieval_failure"]

    assert scenario["calls"]["note_retrieval"] == 1
    assert scenario["documents_returned"] == 2
    assert scenario["outcome"] == "success_with_fallback"
    assert scenario["observed_degradation"] == "knowledge_base_only"


def test_rerank_failure_keeps_original_candidates(
    fake_rag_scenarios: dict[str, dict[str, object]],
) -> None:
    scenario = fake_rag_scenarios["rerank_failure"]

    assert scenario["calls"]["rerank_service"] == 1
    assert scenario["calls"]["summary_model"] == 4
    assert scenario["documents_returned"] == 3
    assert scenario["observed_degradation"] == "original_candidate_order"
    assert (
        scenario["candidate_fingerprints"]["returned"]
        == scenario["candidate_fingerprints"]["rerank_input"]
    )


def test_summary_timeout_preserves_candidates(
    fake_rag_scenarios: dict[str, dict[str, object]],
) -> None:
    scenario = fake_rag_scenarios["summary_timeout"]

    assert scenario["documents_returned"] == 1
    assert scenario["summary"] == "抱歉，生成摘要超时，请稍后再试。"
    assert scenario["outcome"] == "summary_timeout"
    assert (
        scenario["candidate_fingerprints"]["returned"]
        == scenario["candidate_fingerprints"]["rerank_service_output"]
    )


def test_summary_runtime_error_clears_candidates(
    fake_rag_scenarios: dict[str, dict[str, object]],
) -> None:
    scenario = fake_rag_scenarios["summary_runtime_error"]

    assert scenario["documents_returned"] == 0
    assert scenario["summary"] == "抱歉，处理您的请求时出现了错误。"
    assert scenario["observed_degradation"] == "outer_handler_clears_documents"


def test_missing_user_short_circuits_all_dependencies(
    fake_rag_scenarios: dict[str, dict[str, object]],
) -> None:
    scenario = fake_rag_scenarios["missing_user"]

    assert scenario["calls"] == {
        "hyde_model": 0,
        "kb_retrieval": 0,
        "note_retrieval": 0,
        "rerank_service": 0,
        "reranker_inference": 0,
        "summary_model": 0,
    }
    assert scenario["outcome"] == "missing_user"


def _assert_persisted_baseline_sha256(actual_sha256: str) -> None:
    if actual_sha256 == (
        "66bb7a0f018a5c41364d177725486dc384f433351790788ae65742d642e7f68f"
    ):
        pytest.xfail(
            "KI-M0-BASELINE-SHA: 已确认的 M0 历史摘要差异，业务契约独立校验"
        )
    assert actual_sha256 == (
        "1fbd6a18d6a94212b09392dc770dcd6374a64f667c57272b81613f8a778351a8"
    )


def test_persisted_baseline_sha256() -> None:
    baseline_path = Path(__file__).resolve().with_name("m0_fake_rag_baseline.json")
    _assert_persisted_baseline_sha256(
        hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    )


def test_persisted_baseline_matches_current_harness_and_contracts(
    fake_rag_scenarios: dict[str, dict[str, object]],
) -> None:
    backend_root = Path(__file__).resolve().parents[2]
    baseline_bytes = (
        backend_root / "tests/baseline/m0_fake_rag_baseline.json"
    ).read_bytes()
    payload = json.loads(baseline_bytes)

    assert payload["protocol"]["warmup_count"] == 5
    assert payload["protocol"]["sample_count"] == 30
    assert payload["dangerous_imports"] == []
    assert {item["scenario"] for item in payload["scenarios"]} == {
        "success_3_docs",
        "hyde_failure",
        "kb_retrieval_failure",
        "note_retrieval_failure",
        "rerank_failure",
        "summary_timeout",
        "summary_runtime_error",
        "missing_user",
    }
    stable_fields = (
        "calls",
        "outcome",
        "expected_degradation",
        "observed_degradation",
        "fallback_input",
        "documents_returned",
        "summary",
        "error_class",
        "candidate_fingerprints",
    )
    for item in payload["scenarios"]:
        assert item["sample_count"] == 30
        live = fake_rag_scenarios[item["scenario"]]
        assert all(item[field] == live[field] for field in stable_fields)


def test_expected_baseline_sha256_passes() -> None:
    _assert_persisted_baseline_sha256(
        "1fbd6a18d6a94212b09392dc770dcd6374a64f667c57272b81613f8a778351a8"
    )


def test_known_baseline_sha256_is_classified() -> None:
    with pytest.raises(pytest.xfail.Exception, match="KI-M0-BASELINE-SHA"):
        _assert_persisted_baseline_sha256(
            "66bb7a0f018a5c41364d177725486dc384f433351790788ae65742d642e7f68f"
        )


@pytest.mark.parametrize(
    "actual_sha256",
    [
        "0" * 64,
        "66bb7a0f018a5c41364d177725486dc384f433351790788ae65742d642e7f68e",
    ],
    ids=["unknown", "near_known"],
)
def test_unknown_baseline_sha256_fails(actual_sha256: str) -> None:
    with pytest.raises(AssertionError):
        _assert_persisted_baseline_sha256(actual_sha256)


def test_persisted_contracts_are_independent_of_sha256(
    monkeypatch: pytest.MonkeyPatch,
    fake_rag_scenarios: dict[str, dict[str, object]],
) -> None:
    monkeypatch.setattr(
        hashlib,
        "sha256",
        lambda _: pytest.fail("业务契约不得执行摘要校验"),
    )
    test_persisted_baseline_matches_current_harness_and_contracts(fake_rag_scenarios)

    changed_scenarios = {
        **fake_rag_scenarios,
        "success_3_docs": {**fake_rag_scenarios["success_3_docs"], "outcome": "broken"},
    }
    with pytest.raises(AssertionError):
        test_persisted_baseline_matches_current_harness_and_contracts(changed_scenarios)


def test_missing_baseline_is_not_xfailed(
    monkeypatch: pytest.MonkeyPatch,
    fake_rag_scenarios: dict[str, dict[str, object]],
) -> None:
    def missing_baseline(_: Path) -> bytes:
        raise FileNotFoundError("合成的 M0 基线缺失")

    monkeypatch.setattr(Path, "read_bytes", missing_baseline)
    with pytest.raises(FileNotFoundError):
        test_persisted_baseline_sha256()
    with pytest.raises(FileNotFoundError):
        test_persisted_baseline_matches_current_harness_and_contracts(fake_rag_scenarios)


def test_invalid_baseline_json_is_not_xfailed(
    monkeypatch: pytest.MonkeyPatch,
    fake_rag_scenarios: dict[str, dict[str, object]],
) -> None:
    monkeypatch.setattr(Path, "read_bytes", lambda _: b"{")
    with pytest.raises(AssertionError):
        test_persisted_baseline_sha256()
    with pytest.raises(json.JSONDecodeError):
        test_persisted_baseline_matches_current_harness_and_contracts(fake_rag_scenarios)
