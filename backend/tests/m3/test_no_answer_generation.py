"""锁住生成层拒答的标记解析与传播（RAG-008）。

拒答靠提示词里的 `[[NO_ANSWER]]` 标记，解析型标记有两个经典失效面，本文件各
锁一组：
1. 标记泄漏 —— 解析漏掉某条路径时，用户直接看到 `[[NO_ANSWER]]`；
2. 判定过宽 —— 用自然语言关键词（「资料里没有提到」）判拒答，会把正常回答误判。

另外锁住分批总结的传播规则：逐文档拒答是正常现象（文档 1 没答案不代表文档 2
也没有），只有全部分支都拒答才是拒答。

全部离线：`chain` 换成计数用的假件，不调真 LLM。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from app.rag.rag_service import (
    GENERATION_ERROR_MESSAGE,
    GENERATION_TIMEOUT_MESSAGE,
    INFRASTRUCTURE_FAILURE_MESSAGES,
    NO_ANSWER_MARKER,
    RagService,
)

USER = "gen-user"
REFUSAL_TEXT = "抱歉，我在你的资料里没有找到能回答这个问题的内容。"


class _FakeChain:
    """按调用次序返回预置结果，并记录调用次数。"""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[str] = []

    async def ainvoke(self, payload: dict) -> str:
        self.calls.append(payload["context"])
        if not self.responses:
            raise AssertionError("假 chain 被调用的次数超过预置回复数")
        return self.responses.pop(0)


def _doc(text: str, index: int) -> Document:
    return Document(
        page_content=text,
        metadata={
            "user_id": USER,
            "source_type": "knowledge_base",
            "chunk_id": f"c{index}",
        },
    )


def _service(
    *,
    responses: list[str],
    documents: list[Document],
    overlap: int | None = 1,
    both_routes: bool = True,
) -> tuple[RagService, _FakeChain]:
    async def _retrieve_with_routes(search_query, user_id, *, weight_query=None):
        return SimpleNamespace(
            fused=tuple(documents),
            vector_documents=tuple(documents),
            bm25_documents=(),
            weights=(0.6, 0.4),
            both_routes_present=both_routes,
            overlap_count=overlap,
        )

    service = RagService(
        USER,
        vector_store=SimpleNamespace(retrieve_with_routes=_retrieve_with_routes),
        note_service_override=SimpleNamespace(
            notes_store=SimpleNamespace(
                similarity_search=lambda query, k=3, filter=None: []
            )
        ),
    )

    async def _hyde(query):
        return query

    async def _reorder(query, candidates):
        return list(candidates)

    service.generate_hypothetical_document = _hyde
    service.reorder_documents = _reorder
    chain = _FakeChain(responses)
    service.chain = chain
    return service, chain


def _run(service: RagService, query: str = "缓存目录空间不足时怎么办") -> dict:
    return asyncio.run(service.get_documents_and_summary(query, query_id="q-1"))


# --- 标记判定 ---------------------------------------------------------------


@pytest.mark.parametrize(
    "summary, expected",
    [
        (NO_ANSWER_MARKER, True),
        (f"{NO_ANSWER_MARKER}\n参考资料只提到日志轮转，没有讲缓存清理。", True),
        (f"  {NO_ANSWER_MARKER} 缺少相关章节", True),
        ("参考资料里没有提到缓存目录的处理方式。", False),
        ("我无法回答这个问题。", False),
        (f"按文档说明先清理临时文件。注意不要输出 {NO_ANSWER_MARKER}。", False),
        ("", False),
        (None, False),
        (123, False),
    ],
)
def test_refusal_detection_only_trusts_a_leading_marker(summary, expected) -> None:
    """只认开头的标记，不做自然语言判断。

    「资料里没有提到 X」既可能是拒答，也可能是正常答案的一部分（比如在解释
    文档覆盖范围）。靠关键词匹配会把后者误判成拒答，这比不拒答更糟。
    标记出现在正文中间同样不算：那更像模型在复述要求。
    """
    assert RagService._is_refusal(summary) is expected


def test_marker_never_reaches_the_user_visible_summary() -> None:
    """标记泄漏是解析型标记最常见的失效方式，单独锁一条。"""
    service, _ = _service(
        responses=[f"{NO_ANSWER_MARKER}\n资料只讲了日志轮转"],
        documents=[_doc("日志轮转按天切分", 1)],
    )
    result = _run(service)

    assert result["no_answer"] is True
    assert NO_ANSWER_MARKER not in result["summary"]
    assert result["summary"] == REFUSAL_TEXT


def test_strip_marker_keeps_only_the_explanation() -> None:
    detail = RagService._strip_marker(f"{NO_ANSWER_MARKER}\n资料只讲了日志轮转")
    assert detail == "资料只讲了日志轮转"
    assert NO_ANSWER_MARKER not in detail


# --- 单文档路径 -------------------------------------------------------------


def test_single_document_refusal_is_honoured() -> None:
    service, chain = _service(
        responses=[f"{NO_ANSWER_MARKER} 无相关内容"],
        documents=[_doc("日志轮转按天切分", 1)],
    )
    result = _run(service)

    assert result["no_answer"] is True
    assert len(chain.calls) == 1  # 单文档不进合并阶段


def test_single_document_answer_is_returned_verbatim() -> None:
    service, _ = _service(
        responses=["先清理临时文件再重启服务。"],
        documents=[_doc("缓存目录空间不足时先清理临时文件", 1)],
    )
    result = _run(service)

    assert result["no_answer"] is False
    assert result["summary"] == "先清理临时文件再重启服务。"


# --- 多文档分批总结的传播 ---------------------------------------------------


def test_partial_branch_refusals_are_dropped_before_combining() -> None:
    """逐文档拒答不是整体拒答：只把有内容的分支送进合并阶段。

    否则合并阶段会看到「无法回答」的字样并把它写进最终答案，稀释真实答案。
    """
    service, chain = _service(
        responses=[
            f"{NO_ANSWER_MARKER} 本文档只讲日志",
            "先清理临时文件再重启服务。",
            f"{NO_ANSWER_MARKER} 本文档只讲索引重建",
            "综合：先清理临时文件再重启服务。",
        ],
        documents=[_doc("日志轮转", 1), _doc("缓存清理", 2), _doc("索引重建", 3)],
    )
    result = _run(service)

    assert result["no_answer"] is False
    assert result["summary"] == "综合：先清理临时文件再重启服务。"
    combine_context = chain.calls[-1]
    assert "先清理临时文件再重启服务。" in combine_context
    assert NO_ANSWER_MARKER not in combine_context
    assert "本文档只讲日志" not in combine_context


def test_all_branches_refusing_skips_the_combine_call() -> None:
    """全部分支拒答 → 直接拒答，不再花一次 LLM 去合并一堆拒答。"""
    service, chain = _service(
        responses=[
            f"{NO_ANSWER_MARKER} a",
            f"{NO_ANSWER_MARKER} b",
            f"{NO_ANSWER_MARKER} c",
        ],
        documents=[_doc("日志轮转", 1), _doc("索引重建", 2), _doc("端口占用", 3)],
    )
    result = _run(service)

    assert result["no_answer"] is True
    assert result["summary"] == REFUSAL_TEXT
    assert len(chain.calls) == 3  # 三个分支，没有第四次合并调用


def test_refusal_in_the_combined_summary_is_honoured() -> None:
    """分支都有内容但合并阶段判定答不了，同样是拒答。"""
    service, _ = _service(
        responses=[
            "文档一提到磁盘。",
            "文档二提到端口。",
            "文档三提到日志。",
            f"{NO_ANSWER_MARKER} 三个摘要都没回答这个问题",
        ],
        documents=[_doc("磁盘", 1), _doc("端口", 2), _doc("日志", 3)],
    )
    result = _run(service)

    assert result["no_answer"] is True
    assert result["summary"] == REFUSAL_TEXT


def test_only_first_three_documents_are_summarised() -> None:
    """分批总结只取前 3 篇；改动不得放大 LLM 调用数。"""
    service, chain = _service(
        responses=["一", "二", "三", "综合"],
        documents=[_doc(f"文档{i}", i) for i in range(1, 6)],
    )
    result = _run(service)

    assert result["no_answer"] is False
    assert len(chain.calls) == 4  # 3 篇 + 1 次合并


# --- 与检索层门禁的衔接 -----------------------------------------------------


def test_retrieval_gate_short_circuits_generation_entirely() -> None:
    """检索层已判拒答时，生成层一次 LLM 都不该调。

    省掉的是每条被拒查询的 4 次调用；也保证两层判定不会互相覆盖。
    """
    service, chain = _service(
        responses=[],  # 任何调用都会让假 chain 抛断言错
        documents=[_doc("量子色动力学", 1), _doc("日志轮转", 2)],
        overlap=0,
    )
    result = _run(service)

    assert result["no_answer"] is True
    assert result["summary"] == REFUSAL_TEXT
    assert chain.calls == []
    assert result["documents"] == []


def test_single_route_fallback_still_generates() -> None:
    """单路兜底不触发门禁，生成照常进行。"""
    service, chain = _service(
        responses=["先清理临时文件。"],
        documents=[_doc("缓存目录空间不足时先清理临时文件", 1)],
        overlap=None,
        both_routes=False,
    )
    result = _run(service)

    assert result["no_answer"] is False
    assert len(chain.calls) == 1


# --- 失败不得伪装成拒答 -----------------------------------------------------


def test_timeout_is_not_reported_as_refusal() -> None:
    """超时是失败，不是拒答。

    评测里 `run_error` 与 `predicted_no_answer` 是两类；把错误记成拒答会让
    false_answer_rate 虚假下降 —— 那是指标造假，不是修复。
    """

    class _TimeoutChain:
        calls: list = []

        async def ainvoke(self, payload: dict):
            raise TimeoutError

    service, _ = _service(
        responses=[], documents=[_doc("缓存清理", 1)]
    )
    service.chain = _TimeoutChain()
    result = _run(service)

    assert result["no_answer"] is False
    # 断言到常量而非子串：答案层评测按这批文案识别基础设施失败，改文案会让
    # 评测把 LLM 故障当成真作答，所以这里必须是一处强耦合。
    assert result["summary"] == GENERATION_TIMEOUT_MESSAGE
    assert result["summary"] in INFRASTRUCTURE_FAILURE_MESSAGES


def test_unexpected_exception_is_not_reported_as_refusal() -> None:
    class _BoomChain:
        async def ainvoke(self, payload: dict):
            raise RuntimeError("模型连接失败")

    service, _ = _service(responses=[], documents=[_doc("缓存清理", 1)])
    service.chain = _BoomChain()
    result = _run(service)

    assert result["no_answer"] is False
    assert result["summary"] == GENERATION_ERROR_MESSAGE
    assert result["summary"] in INFRASTRUCTURE_FAILURE_MESSAGES


def test_infrastructure_failure_is_distinguishable_from_a_real_refusal() -> None:
    """基础设施失败与真拒答必须可分辨，否则 LLM 故障会被记成幻觉。

    两者的 `no_answer` 一个 False 一个 True，但**都返回文案**。评测拿不到异常
    （生产兜底不外抛），只能靠这批常量识别，所以拒答文案绝不能落进这个集合。
    """
    from app.rag.rag_service import NO_ANSWER_MARKER

    service, chain = _service(
        responses=[f"{NO_ANSWER_MARKER}\n资料里没有这个配置项"],
        documents=[_doc("缓存清理", 1)],
    )
    refusal = _run(service)

    assert refusal["no_answer"] is True
    assert refusal["summary"] not in INFRASTRUCTURE_FAILURE_MESSAGES
    assert len(chain.calls) == 1


def test_every_return_path_carries_a_no_answer_flag() -> None:
    """评测读的是这个键；任何一条路径漏掉它，该条 Query 就无法判分。"""
    empty_user = RagService(
        None,
        vector_store=SimpleNamespace(),
        note_service_override=SimpleNamespace(),
    )
    assert _run(empty_user)["no_answer"] is True

    no_documents, _ = _service(responses=[], documents=[])
    assert _run(no_documents)["no_answer"] is True
