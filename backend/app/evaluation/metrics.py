"""无第三方依赖的 RAG 排名与无答案指标。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from app.evaluation.contracts import BINARY_RELEVANCE_THRESHOLD

RANK_CUTOFFS = (3, 5, 10, 20)


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    recall_at: dict[int, float]
    precision_at_3: float
    mrr_at_10: float
    ndcg_at_10: float


@dataclass(frozen=True, slots=True)
class EvidenceJudgment:
    """一个证据锚点及能够替代命中它的稳定 chunk。"""

    evidence_id: str
    relevance: int
    chunk_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class NoAnswerMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float
    false_answer_rate: float
    run_error_count: int
    zero_division_policy: str = "zero"


def ranking_metrics(
    judgments: tuple[EvidenceJudgment, ...],
    ranked_chunk_ids: list[str],
) -> RankingMetrics:
    """按 evidence anchor 去重计算二值召回与分级排序指标。"""
    _validate_ranking_inputs(judgments, ranked_chunk_ids)
    relevant = tuple(
        judgment
        for judgment in judgments
        if judgment.relevance >= BINARY_RELEVANCE_THRESHOLD
    )
    relevant_chunks = frozenset(
        chunk_id for judgment in relevant for chunk_id in judgment.chunk_ids
    )
    recall_at = {
        cutoff: _evidence_recall(relevant, ranked_chunk_ids[:cutoff])
        for cutoff in RANK_CUTOFFS
    }
    first_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(ranked_chunk_ids[:10], 1)
            if chunk_id in relevant_chunks
        ),
        None,
    )
    return RankingMetrics(
        recall_at=recall_at,
        precision_at_3=_deduplicated_precision_hits(
            relevant, ranked_chunk_ids[:3]
        )
        / 3,
        mrr_at_10=0.0 if first_rank is None else 1 / first_rank,
        ndcg_at_10=_ndcg_at_10(judgments, ranked_chunk_ids),
    )


def no_answer_metrics(
    outcomes: list[tuple[bool, bool, bool]],
) -> NoAnswerMetrics:
    """以系统明确拒答为正类；运行错误不能算正确拒答。"""
    if not outcomes or not any(actual for actual, _, _ in outcomes):
        raise ValueError("无答案指标至少需要一个 gold 无答案样本")
    tp = fp = fn = tn = errors = 0
    for actual_no_answer, predicted_no_answer, run_error in outcomes:
        errors += int(run_error)
        if actual_no_answer and predicted_no_answer and not run_error:
            tp += 1
        elif actual_no_answer:
            fn += 1
        elif predicted_no_answer:
            fp += 1
        else:
            tn += 1
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    return NoAnswerMetrics(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        precision=precision,
        recall=recall,
        f1=_divide(2 * tp, 2 * tp + fp + fn),
        false_answer_rate=_divide(fn, tp + fn),
        run_error_count=errors,
    )


def _validate_ranking_inputs(
    judgments: tuple[EvidenceJudgment, ...],
    ranked_chunk_ids: list[str],
) -> None:
    if not judgments or not any(
        judgment.relevance >= BINARY_RELEVANCE_THRESHOLD
        for judgment in judgments
    ):
        raise ValueError("有答案 Query 必须具有可回答 evidence qrel")
    evidence_ids = [judgment.evidence_id for judgment in judgments]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("evidence_id 不能重复")
    if any(not judgment.chunk_ids for judgment in judgments):
        raise ValueError("每个 evidence judgment 必须至少映射一个 chunk")
    if any(
        isinstance(judgment.relevance, bool)
        or not isinstance(judgment.relevance, int)
        or not 1 <= judgment.relevance <= 3
        for judgment in judgments
    ):
        raise ValueError("qrel relevance 必须在 1 到 3 之间")
    if len(ranked_chunk_ids) != len(set(ranked_chunk_ids)):
        raise ValueError("排名列表不能包含重复 chunk_id")


def _evidence_recall(
    judgments: tuple[EvidenceJudgment, ...],
    ranked_chunk_ids: list[str],
) -> float:
    retrieved = set(ranked_chunk_ids)
    hits = sum(bool(judgment.chunk_ids.intersection(retrieved)) for judgment in judgments)
    return hits / len(judgments)


def _deduplicated_precision_hits(
    judgments: tuple[EvidenceJudgment, ...],
    ranked_chunk_ids: list[str],
) -> int:
    unseen = {judgment.evidence_id: judgment for judgment in judgments}
    hits = 0
    for chunk_id in ranked_chunk_ids:
        matched = [
            evidence_id
            for evidence_id, judgment in unseen.items()
            if chunk_id in judgment.chunk_ids
        ]
        if matched:
            hits += 1
        for evidence_id in matched:
            unseen.pop(evidence_id)
    return hits


def _ndcg_at_10(
    judgments: tuple[EvidenceJudgment, ...],
    ranked_chunk_ids: list[str],
) -> float:
    actual = math.fsum(
        _discounted_gain(relevance, rank)
        for relevance, rank in _first_hit_relevances(judgments, ranked_chunk_ids)
    )
    ideal = _dcg(
        sorted((judgment.relevance for judgment in judgments), reverse=True)[:10]
    )
    return _divide(actual, ideal)


def _first_hit_relevances(
    judgments: tuple[EvidenceJudgment, ...],
    ranked_chunk_ids: list[str],
) -> Iterable[tuple[int, int]]:
    unseen = {judgment.evidence_id: judgment for judgment in judgments}
    next_virtual_rank = 1
    for candidate_rank, chunk_id in enumerate(ranked_chunk_ids[:10], 1):
        matched = sorted(
            (
                judgment
                for judgment in unseen.values()
                if chunk_id in judgment.chunk_ids
            ),
            key=lambda judgment: (-judgment.relevance, judgment.evidence_id),
        )
        virtual_rank = max(candidate_rank, next_virtual_rank)
        for judgment in matched:
            unseen.pop(judgment.evidence_id)
            if virtual_rank <= 10:
                yield judgment.relevance, virtual_rank
            virtual_rank += 1
        next_virtual_rank = virtual_rank


def _dcg(relevances: list[int]) -> float:
    return math.fsum(
        _discounted_gain(relevance, rank)
        for rank, relevance in enumerate(relevances, 1)
    )


def _discounted_gain(relevance: int, rank: int) -> float:
    return (2**relevance - 1) / math.log2(rank + 1)


def _divide(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
