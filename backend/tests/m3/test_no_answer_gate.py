"""锁住拒答门禁：检索层的双路无交集与生成层的标记解析（RAG-008）。

修复前的失效面：development 集 10 条无答案查询上 `false_answer_rate = 1.000`
—— 每一条都被编出了答案。原因是两层都没有拒答通路：检索层无论证据多弱都把
top-3 送进上下文，生成层的提示词也没有「答不了就说答不了」的出口。

本文件全部离线：不调用真 LLM、不调用 embedding、不调用重排服务。重排在测试里
被换成恒等函数（它需要真模型），门禁本体是真代码。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from langchain_core.documents import Document

from app.rag.rag_service import NO_ROUTE_AGREEMENT_CODE, RagService

USER = "gate-user"


@dataclass(frozen=True)
class _Routes:
    """双路结果的替身，只暴露门禁读的两个成员。

    不导入真 RouteRetrieval：它依赖 langchain_chroma。真实交集语义由
    tests/m3/test_route_visibility.py 覆盖，这里只驱动门禁分支。
    """

    both_routes_present: bool
    overlap_count: int | None


def _doc(text: str, *, source_type: str = "knowledge_base", **metadata) -> Document:
    return Document(
        page_content=text,
        metadata={
            "user_id": USER,
            "source_type": source_type,
            "chunk_id": metadata.pop("chunk_id", text[:8]),
            **metadata,
        },
    )


def _service(
    documents: list[Document],
    routes: _Routes | None,
    *,
    notes: list[Document] | None = None,
) -> RagService:
    """造一个只有检索与门禁是真的 RagService。

    笔记必须走 note_service：`retrieve_document` 会把向量路文档的 source_type
    覆写成 knowledge_base，所以笔记候选无法从两路检索注入。
    """

    async def _retrieve_with_routes(search_query, user_id, *, weight_query=None):
        if routes is None:
            raise AssertionError("本用例不应触达两路检索")
        return SimpleNamespace(
            fused=tuple(documents),
            vector_documents=tuple(documents),
            bm25_documents=(),
            weights=(0.6, 0.4),
            both_routes_present=routes.both_routes_present,
            overlap_count=routes.overlap_count,
        )

    service = RagService(
        USER,
        vector_store=SimpleNamespace(retrieve_with_routes=_retrieve_with_routes),
        note_service_override=SimpleNamespace(
            notes_store=SimpleNamespace(
                similarity_search=lambda query, k=3, filter=None: list(notes or [])
            )
        ),
    )
    # HyDE 与重排都需要真模型；门禁不关心它们的内容，换成恒等。
    async def _hyde(query):
        return query

    async def _reorder(query, candidates):
        return list(candidates)

    service.generate_hypothetical_document = _hyde
    service.reorder_documents = _reorder
    return service


def _trace(
    documents: list[Document],
    routes: _Routes | None,
    *,
    notes: list[Document] | None = None,
):
    service = _service(documents, routes, notes=notes)
    return asyncio.run(
        service._build_retrieval_trace(
            "缓存目录空间不足时怎么办",
            query_id="q-1",
            raise_errors=False,
        )
    )


# --- 检索层门禁 -------------------------------------------------------------


def test_zero_overlap_refuses_and_selects_nothing_for_context() -> None:
    """双路都跑了却毫无交集 → 拒答，且没有任何候选进生成上下文。"""
    trace = _trace(
        [_doc("量子色动力学的渐进自由"), _doc("日志轮转按天切分")],
        _Routes(both_routes_present=True, overlap_count=0),
    )

    assert trace.no_answer is True
    assert not any(c.selected_for_context for c in trace.candidates)


def test_refusal_keeps_candidates_for_auditability() -> None:
    """拒答保留候选：「拒答了什么」比「拒答了」信息量大。

    候选被清空的话，事后无法判断门禁是拒对了还是拒错了。
    """
    documents = [_doc("量子色动力学的渐进自由"), _doc("日志轮转按天切分")]
    trace = _trace(documents, _Routes(both_routes_present=True, overlap_count=0))

    assert len(trace.candidates) == len(documents)
    for candidate in trace.candidates:
        stages = [stage.stage for stage in candidate.stages]
        assert stages[-1] == "no_answer_gate"
        assert candidate.stages[-1].error_code == NO_ROUTE_AGREEMENT_CODE
        assert candidate.stages[-1].outcome == "degraded"


def test_positive_overlap_does_not_refuse() -> None:
    """有交集就不拒答 —— 门禁不能把正常查询一并截掉。"""
    trace = _trace(
        [_doc("缓存目录空间不足时先清理临时文件")],
        _Routes(both_routes_present=True, overlap_count=1),
    )

    assert trace.no_answer is False
    assert any(c.selected_for_context for c in trace.candidates)


def test_single_route_fallback_does_not_refuse() -> None:
    """单路兜底交集未定义（None），必须放行。

    把 None 当 0 会让向量兜底路径上的每条查询都被拒答 —— 这是本轮最大的
    误伤风险，所以单独锁一条。
    """
    trace = _trace(
        [_doc("缓存目录空间不足时先清理临时文件")],
        _Routes(both_routes_present=False, overlap_count=None),
    )

    assert trace.no_answer is False
    assert any(c.selected_for_context for c in trace.candidates)


def test_note_evidence_overrides_zero_overlap() -> None:
    """笔记候选不进 BM25 语料，交集为 0 对它没有判别力，因此不拒答。

    笔记走的是独立检索路径，用「词法路没命中」去否定笔记证据是拿错尺子量。
    """
    trace = _trace(
        [_doc("量子色动力学的渐进自由")],
        _Routes(both_routes_present=True, overlap_count=0),
        notes=[_doc("我的排查笔记：先看磁盘", title="排查")],
    )

    assert trace.no_answer is False
    assert any(c.source_type == "note" for c in trace.candidates)


def test_empty_candidates_still_refuse_without_index_version() -> None:
    """零候选仍是拒答，且裸 RagService 不自报 index_version。

    契约见 M3_EVALUATION_REPORT：空候选 trace 的身份必须由具体评测工厂补齐。
    """
    trace = _trace([], _Routes(both_routes_present=True, overlap_count=0))

    assert trace.no_answer is True
    assert trace.candidates == ()
    assert trace.index_version is None
