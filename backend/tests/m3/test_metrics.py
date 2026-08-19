from __future__ import annotations

import math

from app.evaluation.metrics import EvidenceJudgment, no_answer_metrics, ranking_metrics


def _judgments(qrels: dict[str, int]) -> tuple[EvidenceJudgment, ...]:
    return tuple(
        EvidenceJudgment(
            evidence_id=chunk_id,
            relevance=relevance,
            chunk_ids=frozenset({chunk_id}),
        )
        for chunk_id, relevance in qrels.items()
    )


def test_ranking_metrics_match_binary_golden_vector() -> None:
    qrels = _judgments({"a": 2, "b": 2, "c": 2, "d": 2})
    result = ranking_metrics(qrels, ["x", "a", "y", "z", "b"])

    assert result.recall_at == {3: 0.25, 5: 0.5, 10: 0.5, 20: 0.5}
    assert result.precision_at_3 == 1 / 3
    assert result.mrr_at_10 == 0.5
    assert math.isclose(result.ndcg_at_10, 0.397322006969, abs_tol=1e-12)


def test_ndcg_uses_graded_exponential_gain() -> None:
    result = ranking_metrics(_judgments({"a": 2, "b": 1}), ["b", "x", "a"])

    assert math.isclose(result.ndcg_at_10, 0.68852888094, abs_tol=1e-12)


def test_precision_at_three_has_fixed_denominator() -> None:
    result = ranking_metrics(_judgments({"a": 3}), ["a"])

    assert result.recall_at[3] == 1
    assert result.mrr_at_10 == 1
    assert result.ndcg_at_10 == 1
    assert result.precision_at_3 == 1 / 3


def test_overlapping_chunks_count_one_evidence_anchor_once() -> None:
    result = ranking_metrics(
        (
            EvidenceJudgment(
                evidence_id="anchor-a",
                relevance=3,
                chunk_ids=frozenset({"overlap-left", "overlap-right"}),
            ),
        ),
        ["overlap-left", "overlap-right"],
    )

    assert result.recall_at[3] == 1
    assert result.precision_at_3 == 1 / 3
    assert result.mrr_at_10 == 1
    assert result.ndcg_at_10 == 1


def test_background_evidence_does_not_satisfy_binary_retrieval() -> None:
    result = ranking_metrics(
        _judgments({"direct": 3, "background": 1}),
        ["background"],
    )

    assert result.recall_at[20] == 0
    assert result.precision_at_3 == 0
    assert result.mrr_at_10 == 0
    assert result.ndcg_at_10 > 0


def test_no_answer_metrics_match_confusion_matrix() -> None:
    result = no_answer_metrics(
        [
            (True, True, False),
            (True, True, False),
            (False, True, False),
            (True, False, False),
            (True, False, False),
            (False, False, False),
            (False, False, False),
            (False, False, False),
        ]
    )

    assert (result.true_positive, result.false_positive) == (2, 1)
    assert (result.false_negative, result.true_negative) == (2, 3)
    assert math.isclose(result.precision, 2 / 3)
    assert result.recall == 0.5
    assert math.isclose(result.f1, 4 / 7)
    assert result.false_answer_rate == 0.5


def test_run_error_is_not_counted_as_correct_rejection() -> None:
    result = no_answer_metrics([(True, True, True)])

    assert result.true_positive == 0
    assert result.false_negative == 1
    assert result.run_error_count == 1
