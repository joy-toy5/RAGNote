from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import platform
import socket
import sys
import time
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
RAG_SOURCE = BACKEND_ROOT / "app/rag/rag_service.py"
RETRIEVAL_CONTRACT_SOURCE = BACKEND_ROOT / "app/rag/retrieval_contract.py"
INDEXING_CONTRACT_SOURCE = BACKEND_ROOT / "app/indexing/contracts.py"
HARNESS_SOURCE = Path(__file__).resolve()
DANGEROUS_IMPORTS = (
    "chromadb",
    "langchain_chroma",
    "modelscope",
    "sentence_transformers",
    "torch",
)
CALL_NAMES = (
    "hyde_model",
    "kb_retrieval",
    "note_retrieval",
    "rerank_service",
    "reranker_inference",
    "summary_model",
)
STAGE_NAMES = (
    "hyde",
    "retrieval",
    "rerank",
    "orchestration_remainder",
    "end_to_end",
)


@dataclass(frozen=True)
class Scenario:
    name: str
    user_id: str | None = "eval-user-a"
    kb_documents: int = 2
    note_documents: int = 1
    failure: str | None = None
    expected_degradation: str = "none"


SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        Scenario("success_3_docs"),
        Scenario(
            "hyde_failure",
            failure="hyde",
            expected_degradation="original_query",
        ),
        Scenario(
            "kb_retrieval_failure",
            failure="kb_retrieval",
            expected_degradation="note_fallback_skipped",
        ),
        Scenario(
            "note_retrieval_failure",
            failure="note_retrieval",
            expected_degradation="knowledge_base_only",
        ),
        Scenario(
            "rerank_failure",
            failure="rerank",
            expected_degradation="original_candidate_order",
        ),
        Scenario(
            "summary_timeout",
            kb_documents=1,
            note_documents=0,
            failure="summary_timeout",
            expected_degradation="timeout_message",
        ),
        Scenario(
            "summary_runtime_error",
            kb_documents=1,
            note_documents=0,
            failure="summary_runtime_error",
            expected_degradation="outer_handler_clears_documents",
        ),
        Scenario(
            "missing_user",
            user_id=None,
            kb_documents=0,
            note_documents=0,
            expected_degradation="request_rejected",
        ),
    )
}


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict[str, Any]
    id: str | None = None


@dataclass
class RunState:
    scenario: Scenario
    calls: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in CALL_NAMES}
    )
    inputs: dict[str, list[Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    timings: dict[str, list[float]] = field(default_factory=dict)

    def record_input(self, stage: str, value: Any) -> None:
        self.inputs.setdefault(stage, []).append(value)

    def record_error(self, error: BaseException) -> None:
        self.errors.append(type(error).__name__)

    def record_timing(self, stage: str, started_ns: int) -> None:
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        self.timings.setdefault(stage, []).append(elapsed_ms)


_ACTIVE_STATE: RunState | None = None


def _state() -> RunState:
    if _ACTIVE_STATE is None:
        raise RuntimeError("Fake RAG 场景尚未初始化")
    return _ACTIVE_STATE


def _documents(source: str, count: int) -> list[FakeDocument]:
    documents = []
    for index in range(1, count + 1):
        content = f"{source}-document-{index}"
        metadata = {"original_filename": f"kb-{index}.txt"}
        document_id = None
        if source == "note":
            metadata = {"title": f"note-{index}"}
            document_id = f"note-{index}"
        else:
            blob_id = hashlib.sha256(f"blob-{index}".encode()).hexdigest()
            chunk_id = hashlib.sha256(f"chunk-{index}".encode()).hexdigest()
            metadata.update(
                {
                    "blob_id": blob_id,
                    "char_end": len(content),
                    "char_start": 0,
                    "chunk_id": chunk_id,
                    "document_id": str(
                        uuid.uuid5(uuid.NAMESPACE_URL, f"fake-document-{index}")
                    ),
                    "document_revision": 1,
                    "index_version": "f" * 64,
                    "page_number": 1,
                    "source_text_sha256": hashlib.sha256(
                        content.encode()
                    ).hexdigest(),
                    "source_text_uri": "cas+file://sha256/fake/normalized-text",
                    "source_uri": f"cas+file://sha256/{blob_id[:2]}/{blob_id[2:4]}/{blob_id}",
                }
            )
            document_id = chunk_id
        user_id = _state().scenario.user_id
        if user_id is not None:
            metadata["user_id"] = user_id
        documents.append(
            FakeDocument(
                page_content=content,
                metadata=metadata,
                id=document_id,
            )
        )
    return documents


class FakePromptTemplate:
    def __init__(self, template: str) -> None:
        self.kind = "hyde" if "假设性回答" in template else "summary"

    @classmethod
    def from_template(cls, template: str) -> FakePromptTemplate:
        return cls(template)

    def __or__(self, other: object) -> FakePipeline:
        return FakePipeline([self, other])

    async def ainvoke(self, value: Any) -> dict[str, Any]:
        return {"kind": self.kind, "value": value}


class FakePipeline:
    def __init__(self, parts: list[object]) -> None:
        self.parts = parts

    def __or__(self, other: object) -> FakePipeline:
        return FakePipeline([*self.parts, other])

    async def ainvoke(self, value: Any) -> Any:
        current = value
        for part in self.parts:
            current = await part.ainvoke(current)
        return current


class FakeOutputParser:
    async def ainvoke(self, value: Any) -> str:
        return str(value)


class FakeChatModel:
    async def ainvoke(self, value: dict[str, Any]) -> str:
        state = _state()
        kind = value["kind"]
        payload = value["value"]
        if kind == "hyde":
            state.calls["hyde_model"] += 1
            state.record_input("hyde", payload["query"])
            if state.scenario.failure == "hyde":
                error = RuntimeError("injected HyDE failure")
                state.record_error(error)
                raise error
            return f"hypothetical::{payload['query']}"

        state.calls["summary_model"] += 1
        state.record_input("summarize", payload)
        if state.scenario.failure == "summary_timeout":
            error = asyncio.TimeoutError("injected summary timeout")
            state.record_error(error)
            raise error
        if state.scenario.failure == "summary_runtime_error":
            error = RuntimeError("injected summary failure")
            state.record_error(error)
            raise error
        return f"summary-{state.calls['summary_model']}"


class FakeRetriever:
    async def ainvoke(self, query: str) -> list[FakeDocument]:
        state = _state()
        state.calls["kb_retrieval"] += 1
        state.record_input("kb_retrieval", query)
        if state.scenario.failure == "kb_retrieval":
            error = RuntimeError("injected knowledge-base retrieval failure")
            state.record_error(error)
            raise error
        return _documents("knowledge_base", state.scenario.kb_documents)


class FakeVectorStore:
    async def get_dynamic_weights(self, _: str) -> tuple[float, float]:
        return (0.5, 0.5)

    async def get_retriever(self, _: str, __: str) -> FakeRetriever:
        return FakeRetriever()


class FakeNoteStore:
    def similarity_search(self, query: str, **options: Any) -> list[FakeDocument]:
        state = _state()
        state.calls["note_retrieval"] += 1
        state.record_input("note_retrieval", {"query": query, **options})
        if state.scenario.failure == "note_retrieval":
            error = RuntimeError("injected note retrieval failure")
            state.record_error(error)
            raise error
        return _documents("note", state.scenario.note_documents)


class FakeNoteService:
    notes_store = FakeNoteStore()


class FakeReorderService:
    async def reorder_documents(
        self,
        query: str,
        documents: list[Any],
        thinking_callback: object = None,
    ) -> dict[str, Any]:
        del thinking_callback
        state = _state()
        state.calls["rerank_service"] += 1
        rendered_documents = [document.reranker_text for document in documents]
        state.record_input("rerank_input", rendered_documents)
        if documents:
            state.calls["reranker_inference"] += 1
        if state.scenario.failure == "rerank":
            error = RuntimeError("injected reranker failure")
            state.record_error(error)
            state.record_input("rerank_service_output", [])
            return {"success": False, "documents": [], "error": str(error)}
        from app.rag.retrieval_contract import StageObservation

        ranked = []
        for index, candidate in enumerate(reversed(documents), start=1):
            score = 1.0 - index / 100
            ranked.append(
                {
                    "candidate": candidate.observe(
                        StageObservation(
                            stage="rerank",
                            route="fake_cross_encoder",
                            rank=index,
                            raw_score=score,
                            score_direction="higher_is_better",
                        )
                    ),
                    "document": candidate.reranker_text,
                    "similarity": score,
                }
            )
        state.record_input(
            "rerank_service_output", [item["document"] for item in ranked]
        )
        return {"success": True, "documents": ranked, "error": ""}


class FakeLogger:
    def __getattr__(self, _: str) -> object:
        return lambda *args, **kwargs: None


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _identity_traceable(function: object = None, **_: object) -> object:
    if function is None:
        return lambda target: target
    return function


def _install_offline_guards() -> None:
    def blocked(*_: object, **__: object) -> None:
        raise RuntimeError("Fake RAG harness 禁止建立网络连接")

    socket.socket.connect = blocked
    socket.create_connection = blocked
    os.environ.update(
        {
            "PIP_NO_INDEX": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "LANGSMITH_TRACING": "false",
        }
    )


def _load_rag_module() -> types.ModuleType:
    modules = {
        "app": _package("app"),
        "app.indexing": _package("app.indexing"),
        "app.rag": _package("app.rag"),
        "app.utils": _package("app.utils"),
        "app.core": _package("app.core"),
        "app.services": _package("app.services"),
        "app.rag.vector_store": _module(
            "app.rag.vector_store", VectorStoreService=FakeVectorStore
        ),
        "app.rag.reorder_service": _module(
            "app.rag.reorder_service", reorder_service=FakeReorderService()
        ),
        "app.utils.factory": _module("app.utils.factory", chat_model=FakeChatModel()),
        "app.utils.prompt_loader": _module(
            "app.utils.prompt_loader", load_prompt=lambda **_: "summary prompt"
        ),
        "app.core.logger_handler": _module(
            "app.core.logger_handler", logger=FakeLogger()
        ),
        "app.services.note_service": _module(
            "app.services.note_service", note_service=FakeNoteService()
        ),
        "langchain_core": _package("langchain_core"),
        "langchain_core.output_parsers": _module(
            "langchain_core.output_parsers", StrOutputParser=FakeOutputParser
        ),
        "langchain_core.prompts": _module(
            "langchain_core.prompts", PromptTemplate=FakePromptTemplate
        ),
        "langsmith": _module("langsmith", traceable=_identity_traceable),
    }
    sys.modules.update(modules)

    indexing_contract_name = "app.indexing.contracts"
    indexing_contract_spec = importlib.util.spec_from_file_location(
        indexing_contract_name,
        INDEXING_CONTRACT_SOURCE,
    )
    if indexing_contract_spec is None or indexing_contract_spec.loader is None:
        raise RuntimeError(f"无法加载 {INDEXING_CONTRACT_SOURCE}")
    indexing_contract_module = importlib.util.module_from_spec(
        indexing_contract_spec
    )
    sys.modules[indexing_contract_name] = indexing_contract_module
    indexing_contract_spec.loader.exec_module(indexing_contract_module)

    contract_name = "app.rag.retrieval_contract"
    contract_spec = importlib.util.spec_from_file_location(
        contract_name,
        RETRIEVAL_CONTRACT_SOURCE,
    )
    if contract_spec is None or contract_spec.loader is None:
        raise RuntimeError(f"无法加载 {RETRIEVAL_CONTRACT_SOURCE}")
    contract_module = importlib.util.module_from_spec(contract_spec)
    sys.modules[contract_name] = contract_module
    contract_spec.loader.exec_module(contract_module)

    module_name = "app.rag.rag_service"
    spec = importlib.util.spec_from_file_location(module_name, RAG_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {RAG_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _timed_service_class(base: type) -> type:
    class TimedRagService(base):
        async def generate_hypothetical_document(self, query: str) -> str:
            started = time.perf_counter_ns()
            try:
                return await super().generate_hypothetical_document(query)
            finally:
                _state().record_timing("hyde", started)

        async def retrieve_document(self, query: str) -> list[FakeDocument]:
            started = time.perf_counter_ns()
            try:
                return await super().retrieve_document(query)
            finally:
                _state().record_timing("retrieval_flow", started)

        async def reorder_documents(
            self, query: str, documents: list[Any]
        ) -> list[Any]:
            started = time.perf_counter_ns()
            try:
                return await super().reorder_documents(query, documents)
            finally:
                _state().record_timing("rerank", started)

        async def get_documents_and_summary(self, query: str) -> dict[str, Any]:
            started = time.perf_counter_ns()
            try:
                return await super().get_documents_and_summary(query)
            finally:
                _state().record_timing("end_to_end", started)

    return TimedRagService


def _single_timing(state: RunState, name: str) -> float | None:
    values = state.timings.get(name, [])
    return values[-1] if values else None


def _stage_timings(state: RunState) -> dict[str, float | None]:
    hyde = _single_timing(state, "hyde")
    retrieval_flow = _single_timing(state, "retrieval_flow")
    rerank = _single_timing(state, "rerank")
    end_to_end = _single_timing(state, "end_to_end")
    retrieval = None
    if retrieval_flow is not None:
        retrieval = max(0.0, retrieval_flow - (hyde or 0.0))
    remainder = None
    if state.calls["summary_model"] and end_to_end is not None:
        remainder = max(0.0, end_to_end - (retrieval_flow or 0.0) - (rerank or 0.0))
    return {
        "hyde": hyde,
        "retrieval": retrieval,
        "rerank": rerank,
        "orchestration_remainder": remainder,
        "end_to_end": end_to_end,
    }


def _outcome(scenario: Scenario, result: dict[str, Any]) -> str:
    summary = result["summary"]
    if scenario.user_id is None:
        return "missing_user"
    if summary == "抱歉，生成摘要超时，请稍后再试。":
        return "summary_timeout"
    if summary == "抱歉，处理您的请求时出现了错误。":
        return "request_error"
    if not result["documents"]:
        return "no_documents"
    if scenario.failure:
        return "success_with_fallback"
    return "success"


def _fingerprints(values: list[str]) -> list[str]:
    return [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in values]


def _observed_degradation(
    scenario: Scenario,
    state: RunState,
    result: dict[str, Any],
) -> str:
    returned = result["documents"]
    rerank_input = (state.inputs.get("rerank_input") or [[]])[-1]
    rerank_output = (state.inputs.get("rerank_service_output") or [[]])[-1]
    if scenario.name == "success_3_docs":
        return "none"
    if scenario.failure == "hyde":
        kb_input = (state.inputs.get("kb_retrieval") or [None])[-1]
        if kb_input == "How should HTTP 429 be retried?" and returned:
            return "original_query"
    if scenario.failure == "kb_retrieval":
        if state.calls["note_retrieval"] == 0 and not returned:
            return "note_fallback_skipped"
    if scenario.failure == "note_retrieval":
        if returned and all(value.startswith("[来源：知识库") for value in returned):
            return "knowledge_base_only"
    if scenario.failure == "rerank":
        if returned == rerank_input:
            return "original_candidate_order"
    if scenario.failure == "summary_timeout":
        if (
            returned == rerank_output
            and result["summary"] == "抱歉，生成摘要超时，请稍后再试。"
        ):
            return "timeout_message"
    if scenario.failure == "summary_runtime_error":
        if not returned and result["summary"] == "抱歉，处理您的请求时出现了错误。":
            return "outer_handler_clears_documents"
    if scenario.user_id is None and not any(state.calls.values()):
        return "request_rejected"
    return "unverified"


async def _run_once(service_class: type, scenario: Scenario) -> dict[str, Any]:
    global _ACTIVE_STATE
    state = RunState(scenario)
    _ACTIVE_STATE = state
    service = service_class(user_id=scenario.user_id)
    result = await service.get_documents_and_summary("How should HTTP 429 be retried?")
    observed_degradation = _observed_degradation(scenario, state, result)
    if observed_degradation != scenario.expected_degradation:
        raise RuntimeError(
            f"场景 {scenario.name} 的实际降级 {observed_degradation} "
            f"不符合预期 {scenario.expected_degradation}"
        )
    rerank_input = (state.inputs.get("rerank_input") or [[]])[-1]
    rerank_output = (state.inputs.get("rerank_service_output") or [[]])[-1]
    return {
        "calls": dict(state.calls),
        "outcome": _outcome(scenario, result),
        "expected_degradation": scenario.expected_degradation,
        "observed_degradation": observed_degradation,
        "fallback_input": (state.inputs.get("kb_retrieval") or [None])[-1],
        "documents_returned": len(result["documents"]),
        "summary": result["summary"],
        "error_class": state.errors[0] if state.errors else None,
        "candidate_fingerprints": {
            "rerank_input": _fingerprints(rerank_input),
            "rerank_service_output": _fingerprints(rerank_output),
            "returned": _fingerprints(result["documents"]),
        },
        "retrieval_trace": result.get("retrieval_trace"),
        "timings": _stage_timings(state),
    }


def _percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * ratio) - 1)
    return ordered[index]


def _metrics(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min_ms": round(min(values), 6),
        "p50_ms": round(_percentile(values, 0.50), 6),
        "p95_ms": round(_percentile(values, 0.95), 6),
        "max_ms": round(max(values), 6),
    }


async def _capture_scenario(
    service_class: type,
    scenario: Scenario,
    sample_count: int,
    warmup_count: int,
) -> dict[str, Any]:
    for _ in range(warmup_count):
        await _run_once(service_class, scenario)
    samples = [await _run_once(service_class, scenario) for _ in range(sample_count)]
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
    for field_name in stable_fields:
        if any(sample[field_name] != samples[0][field_name] for sample in samples[1:]):
            raise RuntimeError(f"场景 {scenario.name} 的 {field_name} 不稳定")

    return {
        "scenario": scenario.name,
        "sample_count": sample_count,
        "stage": {
            name: _metrics(
                [
                    sample["timings"][name]
                    for sample in samples
                    if sample["timings"][name] is not None
                ]
            )
            for name in STAGE_NAMES
        },
        **{field_name: samples[0][field_name] for field_name in stable_fields},
        "retrieval_trace": samples[0]["retrieval_trace"],
        "rag_source_sha256": hashlib.sha256(RAG_SOURCE.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(HARNESS_SOURCE.read_bytes()).hexdigest(),
        "limitations": [
            "只测真实 RagService 方法体上的替身编排开销。",
            "微秒级统计包含 harness 计时与记录开销。",
            "不代表模型、Chroma、网络、吞吐或线上延迟。",
        ],
    }


async def capture(
    scenario_names: list[str], sample_count: int, warmup_count: int
) -> dict[str, Any]:
    _install_offline_guards()
    module = _load_rag_module()
    service_class = _timed_service_class(module.RagService)
    scenarios = [
        await _capture_scenario(
            service_class, SCENARIOS[name], sample_count, warmup_count
        )
        for name in scenario_names
    ]
    imported = sorted(
        name
        for name in sys.modules
        if any(
            name == target or name.startswith(f"{target}.")
            for target in DANGEROUS_IMPORTS
        )
    )
    return {
        "schema_version": 2,
        "baseline_type": "fake_orchestration_baseline",
        "protocol": {
            "python": platform.python_version(),
            "runtime_platform": platform.platform(),
            "concurrency": 1,
            "warmup_count": warmup_count,
            "sample_count": sample_count,
            "cache": "每次请求创建新的 RagService 和场景状态；无模型或向量缓存。",
            "timeout": "超时场景直接注入 TimeoutError，只验证异常处理，不验证 30 秒 deadline。",
            "percentile_method": "nearest-rank",
        },
        "dangerous_imports": imported,
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="采集 M0 Fake RAG 编排基线")
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 0 or args.warmup < 0:
        parser.error("samples 必须大于 0，warmup 不能小于 0")

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    content = (
        json.dumps(
            asyncio.run(capture(names, args.samples, args.warmup)),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
