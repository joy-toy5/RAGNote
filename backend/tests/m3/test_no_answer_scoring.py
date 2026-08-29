"""锁住拒答在评测侧的收敛形状（RAG-008）。

拒答要能被打分，必须满足两件事，两件都容易写错：
1. 拒答不是错误。`predicted_no_answer=True` 且 `error_code is None` ——
   把拒答记成 run_error 会让 false_answer_rate 虚假下降（错误不参与分母），
   那是指标造假而不是修复。
2. 零候选拒答的身份由具体评测工厂补齐，裸 RagService 不自报 request 的版本
   （契约见 tests/m3/M3_EVALUATION_REPORT.md）。

全部离线：只构造 trace 对象，不跑检索。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.evaluation.runner import execution_from_trace
from app.rag.rag_service import NO_ROUTE_AGREEMENT_CODE
from app.rag.retrieval_contract import (
    EvidenceSpan,
    RetrievalCandidate,
    RetrievalTrace,
    StageObservation,
)

QUERY_ID = "dev-013"
USER = "eval-user-a"
INDEX_VERSION = "a" * 64


def _verified_candidate(*, refused: bool, rank: int = 1) -> RetrievalCandidate:
    chunk_id = str(rank).ljust(64, "c")
    stages = [StageObservation(stage="retrieval", route="hybrid", rank=rank)]
    if refused:
        stages.append(
            StageObservation(
                stage="no_answer_gate",
                route="route_agreement",
                rank=rank,
                outcome="degraded",
                error_code=NO_ROUTE_AGREEMENT_CODE,
            )
        )
    return RetrievalCandidate(
        candidate_id=chunk_id,
        user_id=USER,
        source_type="knowledge_base",
        content="缓存目录空间不足时先清理临时文件。",
        display_name="ops.txt",
        provenance_status="verified",
        blob_id="b" * 64,
        document_id="12345678-1234-5678-9234-567812345678",
        document_revision=1,
        chunk_id=chunk_id,
        index_version=INDEX_VERSION,
        source_uri="blob://normalized/ops.txt",
        evidence_spans=(
            EvidenceSpan(
                char_start=0,
                char_end=18,
                text_sha256="d" * 64,
                text_uri="blob://normalized/ops.txt",
            ),
        ),
        stages=tuple(stages),
    )


def _execution(trace: RetrievalTrace):
    return execution_from_trace(
        trace,
        expected_query_id=QUERY_ID,
        expected_user_id=USER,
        expected_index_version=INDEX_VERSION,
    )


# --- 拒答不是错误 -----------------------------------------------------------


def test_refusal_trace_scores_as_no_answer_not_as_error() -> None:
    """门禁拒答收敛为 predicted_no_answer=True 且 error_code is None。

    这是本轮最关键的一条：若拒答被记成 run_error，该条 Query 不进
    false_answer_rate 的分母，指标会「变好」而模型行为没有改善。
    """
    trace = RetrievalTrace(
        query_id=QUERY_ID,
        user_id=USER,
        candidates=(
            _verified_candidate(refused=True, rank=1),
            _verified_candidate(refused=True, rank=2),
        ),
        index_version=INDEX_VERSION,
        no_answer=True,
    )

    execution = _execution(trace)
    assert execution.predicted_no_answer is True
    assert execution.error_code is None


def test_gate_reason_survives_on_candidate_stages() -> None:
    """拒答理由必须留在候选阶段上，事后才能判断门禁是拒对还是拒错。"""
    trace = RetrievalTrace(
        query_id=QUERY_ID,
        user_id=USER,
        candidates=(_verified_candidate(refused=True),),
        index_version=INDEX_VERSION,
        no_answer=True,
    )

    execution = _execution(trace)
    stages = execution.candidates[0].stages
    assert [stage.stage for stage in stages] == ["retrieval", "no_answer_gate"]
    assert stages[-1].error_code == NO_ROUTE_AGREEMENT_CODE
    assert stages[-1].outcome == "degraded"


def test_refused_candidates_still_count_as_retrieved_for_recall() -> None:
    """拒答保留候选，因此检索指标仍可计算。

    「拒答了」与「检索到了什么」是两个正交事实：门禁拒答不应让 Recall 归零，
    否则无法区分「检索失败」与「检索到了但判定不足以回答」。
    """
    trace = RetrievalTrace(
        query_id=QUERY_ID,
        user_id=USER,
        candidates=(
            _verified_candidate(refused=True, rank=1),
            _verified_candidate(refused=True, rank=2),
        ),
        index_version=INDEX_VERSION,
        no_answer=True,
    )

    execution = _execution(trace)
    assert len(execution.candidates) == 2
    assert [hit.rank for hit in execution.candidates] == [1, 2]


def test_answering_trace_is_not_marked_no_answer() -> None:
    trace = RetrievalTrace(
        query_id=QUERY_ID,
        user_id=USER,
        candidates=(_verified_candidate(refused=False),),
        index_version=INDEX_VERSION,
        no_answer=False,
    )

    assert _execution(trace).predicted_no_answer is False


# --- 零候选拒答的身份补齐 ---------------------------------------------------


def _offline_service(trace: RetrievalTrace):
    """只装配 index_version 补齐这一条路径用到的字段。

    不走 __init__：那会真建 Chroma。这里测的是补齐条件本身，不是工厂装配。
    """
    import scripts.m3_run_eval as runner_script

    service = object.__new__(runner_script._RetrievalOnlyRagService)
    service._descriptor = SimpleNamespace(index_version=INDEX_VERSION)
    service._closed = True  # 让 close() 直接短路，不触碰 _vector_store

    class _Inner:
        async def get_retrieval_trace(self, query: str, *, query_id: str):
            return trace

    service._service = _Inner()
    return service


def test_factory_backfills_index_version_only_for_empty_traces() -> None:
    """零候选拒答必须带上身份，否则一次拒答会以 ValueError 终止整条 run。

    补齐放在工厂而不是 execution_from_trace：后者的严格性正是漂移检测本身，
    放宽它等于拆掉门禁。工厂的 index_version 来自 manifest，已在开跑前与
    dataset 比对过，属于「由具体工厂验证的身份」。
    """
    empty = RetrievalTrace(
        query_id=QUERY_ID, user_id=USER, candidates=(), no_answer=True
    )
    service = _offline_service(empty)

    result = asyncio.run(service.get_retrieval_trace("q", query_id=QUERY_ID))
    assert result.index_version == INDEX_VERSION
    assert result.no_answer is True
    assert result.candidates == ()


def test_factory_does_not_touch_traces_that_have_candidates() -> None:
    """有候选就不补齐 —— 候选自带版本，工厂不得覆盖观测到的事实。"""
    populated = RetrievalTrace(
        query_id=QUERY_ID,
        user_id=USER,
        candidates=(_verified_candidate(refused=True),),
        index_version=INDEX_VERSION,
        no_answer=True,
    )
    service = _offline_service(populated)

    result = asyncio.run(service.get_retrieval_trace("q", query_id=QUERY_ID))
    assert result is populated


def test_factory_does_not_overwrite_an_existing_index_version() -> None:
    """已有版本一律保留，哪怕与工厂声明不一致 —— 不一致要暴露给漂移检测。"""
    other_version = "9" * 64
    trace = RetrievalTrace(
        query_id=QUERY_ID,
        user_id=USER,
        candidates=(),
        index_version=other_version,
        no_answer=True,
    )
    service = _offline_service(trace)

    result = asyncio.run(service.get_retrieval_trace("q", query_id=QUERY_ID))
    assert result.index_version == other_version
