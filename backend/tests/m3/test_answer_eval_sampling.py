"""锁住 `rag008_answer_eval.py` 的抽样构成（`RAG-021`）。

这个文件存在的理由：原抽样只按 `user_id` 分层，而 `m3_dev_v2` 里 `query_id` 的
编号顺序与 `query_type` 强相关（每个 user 的简单型排在前面）。轮转只取到每人前
3~4 个 index，于是 `cross_chunk` 9 条与 `cross_document` 6 条（占可回答的
37.5%）一条都进不了抽样 —— 抽到的 10 条可回答全是单文档、8 条单锚点。
`refusal_rate_answerable = 0.000` 因此只在「单文档单锚点查找」上成立，而
`RAG-016` 修的恰是最容易在跨块查询上触发的缺陷。缺陷本身不报错、不改指纹，只有
把抽样构成断言出来才拦得住。

三件事被锁住：
1. 每个 `query_type` 都有名额（原缺陷的直接签名）。
2. 样本里有多文档、多锚点查询（`RAG-018` 截断窗口要靠这类查询才测得到）。
3. 不可回答那侧是全取而非抽样 —— 这是 `false_answer_rate` 不受抽样偏差影响的
   全部依据。

全部离线：只读数据集文件与 qrels，不建索引、不调 LLM。
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from scripts.rag008_answer_eval import (
    ANSWERABLE_SAMPLE,
    allocate_by_type,
    select_queries,
)

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "evals/datasets/m3_dev_v2"
MANIFEST = DATASET_DIR / "manifest.json"

requires_dataset = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="m3_dev_v2 数据集不在工作树中（.gitignore 只发布 *.py）",
)


@pytest.fixture(scope="module")
def dataset():
    from app.evaluation.dataset import load_dataset

    return load_dataset(MANIFEST)


@pytest.fixture(scope="module")
def gold_documents() -> dict[str, set[str]]:
    """query_id -> 金标准证据涉及的 corpus_item 集合。anchor_id 形如 `文件#a01`。"""
    mapping: dict[str, set[str]] = collections.defaultdict(set)
    with (DATASET_DIR / "source_qrels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mapping[row["query_id"]].add(row["anchor_id"].split("#", 1)[0])
    return dict(mapping)


@pytest.fixture(scope="module")
def gold_anchor_counts() -> dict[str, int]:
    counter: collections.Counter[str] = collections.Counter()
    with (DATASET_DIR / "source_qrels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            counter[json.loads(line)["query_id"]] += 1
    return dict(counter)


# --- 配额分配：纯函数，不依赖数据集 ---------------------------------------


def test_every_type_gets_at_least_one_slot():
    sizes = {"a": 10, "b": 9, "c": 6, "d": 6, "e": 7, "f": 2}
    quota = allocate_by_type(sizes, 10)
    assert sum(quota.values()) == 10
    assert set(quota) == set(sizes)
    assert min(quota.values()) >= 1


def test_quota_never_exceeds_available_queries():
    # f 只有 2 条，配额不能给到 3。
    quota = allocate_by_type({"a": 30, "f": 2}, 10)
    assert quota["f"] <= 2
    assert sum(quota.values()) == 10


def test_quota_is_deterministic():
    sizes = {"a": 10, "b": 9, "c": 6, "d": 6, "e": 7, "f": 2}
    assert allocate_by_type(sizes, 10) == allocate_by_type(sizes, 10)


def test_allocation_rejects_total_smaller_than_type_count():
    with pytest.raises(ValueError, match="每型保底"):
        allocate_by_type({"a": 5, "b": 5, "c": 5}, 2)


def test_allocation_saturates_when_pool_is_exhausted():
    # 总量 10 但只有 6 条可选：配额合计不能虚报成 10。
    quota = allocate_by_type({"a": 3, "b": 2, "c": 1}, 10)
    assert quota == {"a": 3, "b": 2, "c": 1}


# --- 抽样构成：`RAG-021` 的直接门禁 --------------------------------------


@requires_dataset
def test_all_answerable_query_types_are_sampled(dataset):
    """原缺陷的直接签名：整型缺失。"""
    available = {q.query_type for q in dataset.queries if q.answerability == "answerable"}
    selected, quota = select_queries(dataset)
    sampled = {q.query_type for q in selected if q.answerability == "answerable"}
    assert sampled == available, f"缺失的型：{sorted(available - sampled)}"
    assert set(quota) == available


@requires_dataset
def test_sample_covers_multi_document_queries(dataset, gold_documents):
    """`RAG-018` 的截断只在必须跨多篇作答时才伤人，样本里必须有这类查询。"""
    selected = [q for q in select_queries(dataset)[0] if q.answerability == "answerable"]
    multi_document = [
        q.query_id for q in selected if len(gold_documents.get(q.query_id, ())) > 1
    ]
    assert multi_document, "样本里没有跨文档查询，测不到生成窗口截断"


@requires_dataset
def test_sample_covers_multi_anchor_queries(dataset, gold_anchor_counts):
    selected = [q for q in select_queries(dataset)[0] if q.answerability == "answerable"]
    multi_anchor = [
        q.query_id for q in selected if gold_anchor_counts.get(q.query_id, 0) > 1
    ]
    assert len(multi_anchor) >= 3, f"多锚点查询只有 {len(multi_anchor)} 条"


@requires_dataset
def test_every_user_is_represented(dataset):
    """型内错开轮转起点的理由：不错开时 10 条会全落在前两个 user 上。"""
    available = {q.user_id for q in dataset.queries if q.answerability == "answerable"}
    selected = [q for q in select_queries(dataset)[0] if q.answerability == "answerable"]
    assert {q.user_id for q in selected} == available


@requires_dataset
def test_unanswerable_side_is_exhaustive_not_sampled(dataset):
    """`false_answer_rate` 的分母是不可回答数。全取，才不受抽样偏差影响。"""
    everything = [q for q in dataset.queries if q.answerability == "unanswerable"]
    selected = [q for q in select_queries(dataset)[0] if q.answerability == "unanswerable"]
    assert [q.query_id for q in selected] == sorted(q.query_id for q in everything)


@requires_dataset
def test_answerable_count_matches_declared_sample_size(dataset):
    selected, quota = select_queries(dataset)
    answerable = [q for q in selected if q.answerability == "answerable"]
    assert len(answerable) == ANSWERABLE_SAMPLE
    assert sum(quota.values()) == ANSWERABLE_SAMPLE
    per_type = collections.Counter(q.query_type for q in answerable)
    assert dict(per_type) == quota


@requires_dataset
def test_selection_is_deterministic(dataset):
    """重跑之间的差异只该来自 LLM，不该来自样本。"""
    first, first_quota = select_queries(dataset)
    second, second_quota = select_queries(dataset)
    assert [q.query_id for q in first] == [q.query_id for q in second]
    assert first_quota == second_quota


@requires_dataset
def test_selection_is_sorted_and_unique(dataset):
    selected = select_queries(dataset)[0]
    ids = [q.query_id for q in selected]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
