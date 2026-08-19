from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.evaluation.contracts import ExecutorDescriptor, QueryExecution, sha256_json
from app.evaluation.dataset import load_dataset
from app.evaluation.runner import EvalRequest, run_dataset
from app.rag.evaluation_adapter import RagServiceQueryExecutor
from app.rag.retrieval_contract import (
    EvidenceSpan,
    RetrievalCandidate,
    RetrievalTrace,
    StageObservation,
)


RETRIEVAL_CONFIG = {"name": "production-contract", "top_k": 20}
INDEX_VERSION = "a" * 64
CONFIG_SHA256 = sha256_json(RETRIEVAL_CONFIG)
class RecordingService:
    def __init__(
        self,
        user_id: str,
        *,
        descriptor: ExecutorDescriptor | None = None,
        fail: bool = False,
    ) -> None:
        self.user_id = user_id
        self.descriptor = descriptor or _descriptor()
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    async def get_retrieval_trace(
        self,
        query: str,
        *,
        query_id: str,
    ) -> RetrievalTrace:
        self.calls.append((query_id, query))
        if self.fail:
            raise RuntimeError("injected retrieval failure")
        return RetrievalTrace(
            query_id=query_id,
            user_id=self.user_id,
            candidates=(),
            index_version=self.descriptor.index_version,
            no_answer=True,
        )


class RecordingFactory:
    def __init__(
        self,
        descriptor: ExecutorDescriptor | None = None,
        *,
        fail: bool = False,
        service_descriptor: ExecutorDescriptor | None = None,
    ) -> None:
        self.descriptor = descriptor or _descriptor()
        self.service_descriptor = service_descriptor or self.descriptor
        self.fail = fail
        self.services: list[RecordingService] = []
        self.requests: list[EvalRequest] = []

    def __call__(self, request: EvalRequest) -> RecordingService:
        self.requests.append(request)
        service = RecordingService(
            request.user_id,
            descriptor=self.service_descriptor,
            fail=self.fail,
        )
        self.services.append(service)
        return service


def _descriptor() -> ExecutorDescriptor:
    return ExecutorDescriptor(
        executor_id="rag-service-production-contract-v1",
        index_version=INDEX_VERSION,
        retrieval_config_sha256=CONFIG_SHA256,
    )


def _request(**overrides: str) -> EvalRequest:
    values = {
        "query_id": "frozen-query-id",
        "user_id": "eval-user-a",
        "query": "How should HTTP 429 be retried?",
        "index_version": INDEX_VERSION,
        "retrieval_config_sha256": CONFIG_SHA256,
    }
    values.update(overrides)
    return EvalRequest(**values)


def _smoke_manifest(backend_root: Path) -> Path:
    return backend_root / "evals/datasets/m3_smoke_v1/manifest.json"


class StaticFactory:
    def __init__(
        self,
        builder: Callable[[EvalRequest], RecordingService],
    ) -> None:
        self.descriptor = _descriptor()
        self._builder = builder

    def __call__(self, request: EvalRequest) -> RecordingService:
        return self._builder(request)


def test_executor_forwards_frozen_identity_and_returns_query_execution() -> None:
    factory = RecordingFactory()
    executor = RagServiceQueryExecutor(factory)

    execution = asyncio.run(executor.execute(_request()))

    assert isinstance(execution, QueryExecution)
    assert execution.predicted_no_answer is True
    assert execution.error_code is None
    assert len(factory.services) == 1
    assert factory.requests[0].index_version == INDEX_VERSION
    assert factory.requests[0].retrieval_config_sha256 == CONFIG_SHA256
    assert factory.services[0].calls == [
        ("frozen-query-id", "How should HTTP 429 be retried?")
    ]
    assert executor.descriptor == _descriptor()


def test_executor_converts_verified_candidate_and_stage_observation() -> None:
    chunk_id = "c" * 64

    class VerifiedService(RecordingService):
        async def get_retrieval_trace(
            self,
            query: str,
            *,
            query_id: str,
        ) -> RetrievalTrace:
            candidate = RetrievalCandidate(
                candidate_id=chunk_id,
                user_id=self.user_id,
                source_type="knowledge_base",
                content="Retry HTTP 429 with bounded exponential backoff.",
                display_name="retry-guide.txt",
                provenance_status="verified",
                blob_id="b" * 64,
                document_id="12345678-1234-5678-9234-567812345678",
                document_revision=1,
                chunk_id=chunk_id,
                index_version=INDEX_VERSION,
                source_uri="blob://normalized/retry-guide.txt",
                evidence_spans=(
                    EvidenceSpan(
                        char_start=0,
                        char_end=47,
                        text_sha256="d" * 64,
                        text_uri="blob://normalized/retry-guide.txt",
                    ),
                ),
                stages=(
                    StageObservation(
                        stage="retrieval",
                        route="legacy_hybrid",
                        rank=1,
                    ),
                ),
            )
            return RetrievalTrace(
                query_id=query_id,
                user_id=self.user_id,
                candidates=(candidate,),
                index_version=INDEX_VERSION,
            )

    executor = RagServiceQueryExecutor(
        StaticFactory(lambda request: VerifiedService(request.user_id))
    )

    execution = asyncio.run(executor.execute(_request()))

    assert execution.candidates[0].chunk_id == chunk_id
    assert execution.candidates[0].stages[0].stage == "retrieval"
    assert execution.candidates[0].stages[0].route == "legacy_hybrid"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("index_version", "b" * 64, "index_version"),
        ("retrieval_config_sha256", "c" * 64, "retrieval config"),
    ),
)
def test_executor_rejects_request_attestation_before_service_creation(
    field: str,
    value: str,
    message: str,
) -> None:
    factory = RecordingFactory()
    executor = RagServiceQueryExecutor(factory)

    with pytest.raises(ValueError, match=message):
        asyncio.run(executor.execute(_request(**{field: value})))

    assert factory.services == []


def test_executor_rejects_runtime_random_trace_identity() -> None:
    class RandomIdentityService(RecordingService):
        async def get_retrieval_trace(
            self,
            query: str,
            *,
            query_id: str,
        ) -> RetrievalTrace:
            return RetrievalTrace(
                query_id="runtime-random-id",
                user_id=self.user_id,
                candidates=(),
                index_version=self.descriptor.index_version,
                no_answer=True,
            )

    executor = RagServiceQueryExecutor(
        StaticFactory(lambda request: RandomIdentityService(request.user_id))
    )

    with pytest.raises(ValueError, match="冻结 Query"):
        asyncio.run(executor.execute(_request()))


def test_executor_rejects_unversioned_empty_trace() -> None:
    class UnboundService(RecordingService):
        async def get_retrieval_trace(
            self,
            query: str,
            *,
            query_id: str,
        ) -> RetrievalTrace:
            return RetrievalTrace(
                query_id=query_id,
                user_id=self.user_id,
                candidates=(),
                no_answer=True,
            )

    executor = RagServiceQueryExecutor(
        StaticFactory(lambda request: UnboundService(request.user_id))
    )

    with pytest.raises(ValueError, match="trace index_version"):
        asyncio.run(executor.execute(_request()))


def test_executor_rejects_reused_service_instance() -> None:
    service = RecordingService("eval-user-a")
    executor = RagServiceQueryExecutor(StaticFactory(lambda _: service))

    asyncio.run(executor.execute(_request(query_id="frozen-query-1")))
    with pytest.raises(RuntimeError, match="独立服务实例"):
        asyncio.run(executor.execute(_request(query_id="frozen-query-2")))


def test_executor_rejects_service_descriptor_mismatch_before_retrieval() -> None:
    factory = RecordingFactory(
        service_descriptor=ExecutorDescriptor(
            executor_id="rag-service-production-contract-v1",
            index_version="b" * 64,
            retrieval_config_sha256=CONFIG_SHA256,
        )
    )
    executor = RagServiceQueryExecutor(factory)

    with pytest.raises(ValueError, match="服务 descriptor"):
        asyncio.run(executor.execute(_request()))

    assert factory.services[0].calls == []


def test_factory_cannot_mutate_frozen_eval_request() -> None:
    def mutate_request(request: EvalRequest) -> RecordingService:
        request.query_id = "mutated-query-id"  # type: ignore[misc]
        return RecordingService(request.user_id)

    executor = RagServiceQueryExecutor(StaticFactory(mutate_request))

    with pytest.raises(FrozenInstanceError):
        asyncio.run(executor.execute(_request()))


def test_retrieval_failure_is_a_run_error_not_a_correct_rejection(
    backend_root: Path,
) -> None:
    dataset = load_dataset(_smoke_manifest(backend_root))
    descriptor = ExecutorDescriptor(
        executor_id="rag-service-production-contract-v1",
        index_version=dataset.index_version,
        retrieval_config_sha256=CONFIG_SHA256,
    )
    factory = RecordingFactory(descriptor, fail=True)
    executor = RagServiceQueryExecutor(factory)

    run = asyncio.run(
        run_dataset(
            dataset,
            executor,
            source_revision="cb0d3d3",
            source_fingerprint="f" * 64,
            retrieval_config=RETRIEVAL_CONFIG,
        )
    )

    assert len(factory.services) == len(dataset.queries)
    assert len({id(service) for service in factory.services}) == len(dataset.queries)
    assert all(query.predicted_no_answer is False for query in run.queries)
    assert all(
        query.error_code == "EXECUTOR_EXCEPTION:RuntimeError"
        for query in run.queries
    )


def test_rag_service_accepts_frozen_id_and_strict_errors(
    backend_root: Path,
) -> None:
    script = r'''
import asyncio
import json
import uuid

from tests.baseline import fake_rag_harness as harness


async def main():
    harness._install_offline_guards()
    module = harness._load_rag_module()

    harness._ACTIVE_STATE = harness.RunState(harness.SCENARIOS["success_3_docs"])
    service = module.RagService(user_id="eval-user-a")
    frozen = await service.get_documents_and_summary(
        "How should HTTP 429 be retried?",
        query_id="frozen-query-id",
    )

    harness._ACTIVE_STATE = harness.RunState(harness.SCENARIOS["success_3_docs"])
    service = module.RagService(user_id="eval-user-a")
    generated = await service.get_documents_and_summary(
        "How should HTTP 429 be retried?"
    )
    uuid.UUID(generated["retrieval_trace"]["query_id"])

    strict_errors = {}
    for scenario_name in ("kb_retrieval_failure", "note_retrieval_failure"):
        harness._ACTIVE_STATE = harness.RunState(harness.SCENARIOS[scenario_name])
        service = module.RagService(user_id="eval-user-a")
        try:
            await service.get_retrieval_trace(
                "How should HTTP 429 be retried?",
                query_id=f"strict-{scenario_name}",
            )
        except Exception as exc:
            strict_errors[scenario_name] = type(exc).__name__
        else:
            strict_errors[scenario_name] = None

    print(json.dumps({
        "frozen_query_id": frozen["retrieval_trace"]["query_id"],
        "generated_query_id": generated["retrieval_trace"]["query_id"],
        "strict_errors": strict_errors,
    }))


asyncio.run(main())
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
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

    assert payload["frozen_query_id"] == "frozen-query-id"
    assert payload["generated_query_id"] != "frozen-query-id"
    assert payload["strict_errors"] == {
        "kb_retrieval_failure": "RuntimeError",
        "note_retrieval_failure": "RuntimeError",
    }
