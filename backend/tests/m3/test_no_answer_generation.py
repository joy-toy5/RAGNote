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
    EVIDENCE_FORBIDS_MARKER,
    GENERATION_ERROR_MESSAGE,
    GENERATION_TIMEOUT_MESSAGE,
    INFRASTRUCTURE_FAILURE_MESSAGES,
    NO_ANSWER_MARKER,
    REFUSAL_KIND_EVIDENCE_FORBIDS,
    REFUSAL_KIND_INFORMATION_MISSING,
    RagService,
)

pytestmark = pytest.mark.usefixtures("isolated_rag_model")

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


# --- 分支级失败隔离（RAG-024）-----------------------------------------------


class _PerBranchChain:
    """按上下文里的资料序号决定该分支是成功、拒答还是抛异常。

    不能按调用次序预置：三个分支是并发的，到达次序不保证。所以用上下文内容
    路由 —— `【参考资料N】` 里的 N 就是分支号，合并调用则没有这个前缀。
    """

    def __init__(self, *, branch_outcomes: dict[int, object], combined: object):
        self.branch_outcomes = branch_outcomes
        self.combined = combined
        self.calls: list[str] = []

    async def ainvoke(self, payload: dict):
        context = payload["context"]
        self.calls.append(context)
        if context.startswith("以下是多个文档的摘要"):
            outcome = self.combined
        else:
            branch = int(context.split("【参考资料", 1)[1].split("】", 1)[0])
            outcome = self.branch_outcomes[branch]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _three_doc_service(
    *, branch_outcomes: dict[int, object], combined: object = "综合回答。"
) -> tuple[RagService, _PerBranchChain]:
    service, _ = _service(
        responses=[],
        documents=[_doc("磁盘", 1), _doc("端口", 2), _doc("日志", 3)],
    )
    chain = _PerBranchChain(branch_outcomes=branch_outcomes, combined=combined)
    service.chain = chain
    return service, chain


def test_one_failed_branch_does_not_destroy_the_whole_query() -> None:
    """`RAG-024` 的本体：单个分支超时不得报废整条查询。

    修复前 `asyncio.gather` 没有 `return_exceptions=True`，任一分支抛异常就从
    gather 传出，连同**其余已经成功的分支**一起掉进超时兜底。分支数等于
    `max_documents`，所以窗口越宽越容易发生 —— 这正是 `RAG-018` 深度曲线上
    45%~65% 可回答查询报废的机制。
    """
    service, chain = _three_doc_service(
        branch_outcomes={
            1: "文档一提到磁盘。",
            2: TimeoutError(),
            3: "文档三提到日志。",
        }
    )
    result = _run(service)

    assert result["no_answer"] is False
    assert result["summary"] == "综合回答。"
    assert result["summary"] not in INFRASTRUCTURE_FAILURE_MESSAGES

    # 两个活分支的内容都进了合并上下文，死分支没有留下任何占位文本。
    combine_context = chain.calls[-1]
    assert "文档一提到磁盘。" in combine_context
    assert "文档三提到日志。" in combine_context


def test_a_failed_branch_is_never_counted_as_a_refusal() -> None:
    """失败分支与拒答分支必须可分辨 —— 这是本条的验收标准。

    两者都导致该分支没有可用摘要，但语义相反：拒答是模型读完了说没有，失败是
    根本没读到。若把失败并进拒答，一次基础设施故障会被记成"系统正确地拒绝
    回答"，拒答率虚高而 false_answer_rate 虚假下降。那是指标造假，不是修复。
    """
    # 两个分支失败，剩下那个给出真答案：结果必须是答案，且失败可见。
    service, _ = _three_doc_service(
        branch_outcomes={
            1: TimeoutError(),
            2: "文档二提到端口冲突时改监听端口。",
            3: RuntimeError("模型连接失败"),
        }
    )
    result = _run(service)

    assert result["no_answer"] is False
    health = result["generation_health"]
    assert health["branches_total"] == 3
    assert health["branches_succeeded"] == 1
    assert health["branches_failed"] == 2
    assert health["degraded"] is True
    assert {err["error_type"] for err in health["branch_errors"]} == {
        "TimeoutError",
        "RuntimeError",
    }
    assert {err["branch"] for err in health["branch_errors"]} == {1, 3}


def test_all_branches_failing_stays_an_error_not_a_refusal() -> None:
    """全线失败仍是失败。降级成拒答会把一次全线故障记成一次正确拒答。"""
    service, chain = _three_doc_service(
        branch_outcomes={1: TimeoutError(), 2: TimeoutError(), 3: TimeoutError()}
    )
    result = _run(service)

    assert result["no_answer"] is False
    assert result["summary"] == GENERATION_TIMEOUT_MESSAGE
    assert result["summary"] in INFRASTRUCTURE_FAILURE_MESSAGES
    assert len(chain.calls) == 3  # 没有第四次合并调用
    assert result["generation_health"]["branches_failed"] == 3


def test_all_branches_failing_keeps_the_original_failure_kind() -> None:
    """全线失败时不得把「连不上」一律写成「太慢」。

    隔离分支异常之后，全线失败要由本方法重抛。若无条件抛超时，一次连接故障会
    被记成超时 —— 两者都在 `INFRASTRUCTURE_FAILURE_MESSAGES` 里，不影响判分，
    但错因分类失真，而排查方向完全不同。
    """
    service, _ = _three_doc_service(
        branch_outcomes={
            1: RuntimeError("模型连接失败"),
            2: RuntimeError("模型连接失败"),
            3: TimeoutError(),
        }
    )
    result = _run(service)

    assert result["summary"] == GENERATION_ERROR_MESSAGE
    assert result["summary"] != GENERATION_TIMEOUT_MESSAGE
    assert result["no_answer"] is False
    assert result["generation_health"]["branches_failed"] == 3


def test_refusal_with_failed_branches_is_marked_degraded() -> None:
    """带失败分支的拒答不能和证据齐全时的拒答记成同一件事。

    「3 个分支都读完了说没有」与「2 个分支根本没读到、剩 1 个说没有」可信度
    完全不同。前者是拒答证据，后者是故障掩盖成的拒答。
    """
    service, _ = _three_doc_service(
        branch_outcomes={
            1: TimeoutError(),
            2: f"{NO_ANSWER_MARKER} 本文档只讲端口",
            3: TimeoutError(),
        }
    )
    result = _run(service)

    assert result["no_answer"] is True
    assert result["summary"] == REFUSAL_TEXT
    assert result["generation_health"]["degraded"] is True
    assert result["generation_health"]["branches_failed"] == 2


def test_healthy_run_is_not_marked_degraded() -> None:
    """没有失败时 degraded 必须是 False，否则这个信号没有区分力。"""
    service, _ = _three_doc_service(
        branch_outcomes={1: "一", 2: "二", 3: "三"}
    )
    result = _run(service)

    health = result["generation_health"]
    assert health["degraded"] is False
    assert health["branches_failed"] == 0
    assert health["branch_errors"] == []
    assert health["branches_succeeded"] == 3


def test_generation_health_is_present_on_every_return_path() -> None:
    """形状恒定，评测侧才能无条件读这个键。"""
    empty_user = RagService(
        None,
        vector_store=SimpleNamespace(),
        note_service_override=SimpleNamespace(),
    )
    paths = [_run(empty_user)]

    no_documents, _ = _service(responses=[], documents=[])
    paths.append(_run(no_documents))

    gated, _ = _service(
        responses=[], documents=[_doc("量子色动力学", 1)], overlap=0
    )
    paths.append(_run(gated))

    answered, _ = _service(
        responses=["先清理临时文件。"], documents=[_doc("缓存清理", 1)]
    )
    paths.append(_run(answered))

    expected_keys = {
        "branches_total",
        "branches_succeeded",
        "branches_failed",
        "branch_errors",
        "degraded",
    }
    for result in paths:
        assert set(result["generation_health"]) == expected_keys


def test_empty_health_instances_are_not_shared() -> None:
    """空 health 用工厂函数生成：共享一个 dict 会让 branch_errors 互相污染。"""
    first, _ = _service(responses=[], documents=[])
    second, _ = _service(responses=[], documents=[])
    health_a = _run(first)["generation_health"]
    health_b = _run(second)["generation_health"]

    assert health_a is not health_b
    health_a["branch_errors"].append({"branch": 1})
    assert health_b["branch_errors"] == []


# --- 拒答语义分层（RAG-026）-------------------------------------------------


def test_both_refusal_kinds_are_detected_but_only_one_is_a_veto() -> None:
    """两个标记都算拒答，但只有证据禁止型触发否决。

    `_is_refusal` 对两者都为真是刻意的：所有既有的「是否拒答」判断因此语义不变，
    新标记只增加可分辨性。区分只发生在**处置**那一处。
    """
    assert RagService._is_refusal(NO_ANSWER_MARKER) is True
    assert RagService._is_refusal(EVIDENCE_FORBIDS_MARKER) is True

    assert RagService._is_evidence_forbidden(EVIDENCE_FORBIDS_MARKER) is True
    assert RagService._is_evidence_forbidden(NO_ANSWER_MARKER) is False
    assert RagService._is_evidence_forbidden("旧记录不能据此判断错误码。") is False
    assert RagService._is_evidence_forbidden(None) is False


def test_strip_marker_covers_both_kinds() -> None:
    """漏剥一种标记，它就会原样泄漏进用户可见文案。

    `_strip_marker` 的返回值被当作纯说明文字使用，所以只处理
    `[[NO_ANSWER]]` 会让 `[[EVIDENCE_FORBIDS]]` 带着方括号跑出去。
    """
    detail = "旧记录只写了「中继变慢」，缺少上游响应类型与缓冲使用率。"
    assert RagService._strip_marker(f"{EVIDENCE_FORBIDS_MARKER}\n{detail}") == detail
    assert RagService._strip_marker(f"{NO_ANSWER_MARKER}\n{detail}") == detail
    for marker in (NO_ANSWER_MARKER, EVIDENCE_FORBIDS_MARKER):
        assert marker not in RagService._strip_marker(f"{marker} {detail}")


def test_evidence_forbidding_branch_vetoes_a_confident_branch() -> None:
    """`RAG-026` 的本体：一个分支说「不能据此推断」，整条查询就必须拒答。

    这是 dev-035 误答的精确形态 —— rank 1 块里既有「E2101 是缓冲溢出」又有
    「旧记录…不能据此判断错误码」，rank 5 块只有错误码定义、块内自洽。修复前
    rank 1 的异议被当成「这块没答案」剔掉，rank 5 独占合并输入，于是输出一个
    引用正确、事实可核对、只有推理那步错了的假答案。
    """
    service, chain = _three_doc_service(
        branch_outcomes={
            1: f"{EVIDENCE_FORBIDS_MARKER}\n旧记录不能据此判断错误码。",
            2: "属于错误码 E2101（中继缓冲区溢出）。",
            3: f"{NO_ANSWER_MARKER}\n本块只讲日志轮转。",
        }
    )
    result = _run(service)

    assert result["no_answer"] is True
    assert result["summary"] == REFUSAL_TEXT
    assert result["refusal_kind"] == REFUSAL_KIND_EVIDENCE_FORBIDS

    # 关键断言：合并阶段根本没被调用。那个自信分支没有机会独占输入。
    assert not any(
        call.startswith("以下是多个文档的摘要") for call in chain.calls
    )
    assert "E2101" not in result["summary"]


def test_a_veto_is_not_a_vote() -> None:
    """1 个禁止 vs 2 个自信作答，仍然拒答。

    不用「拒答过半」这类投票规则：dev-035 在深度 5 是 4 拒 1 答、深度 8 是
    7 拒 1 答，投票能盖住这一条，却盖不住「禁止性证据恰好也在一个自信块里」
    的情形 —— 那是把语义问题换成阈值问题，换来的达标是假的。
    """
    service, chain = _three_doc_service(
        branch_outcomes={
            1: "答案是 E2101。",
            2: "同样指向 E2101。",
            3: f"{EVIDENCE_FORBIDS_MARKER}\n证据不足以确定错误码。",
        }
    )
    result = _run(service)

    assert result["no_answer"] is True
    assert result["refusal_kind"] == REFUSAL_KIND_EVIDENCE_FORBIDS
    assert not any(
        call.startswith("以下是多个文档的摘要") for call in chain.calls
    )


def test_information_missing_branches_are_still_merged_not_vetoed() -> None:
    """信息缺失型的处置保持原样：剔除该分支，其余照常合并。

    这条守的是「修复没有过度扩张」—— 把两类都当否决会让「文档 1 没有答案」
    毁掉整条查询，那比原缺陷更糟。
    """
    service, chain = _three_doc_service(
        branch_outcomes={
            1: f"{NO_ANSWER_MARKER}\n本块没有提到。",
            2: "按文档说明先清理临时文件。",
            3: f"{NO_ANSWER_MARKER}\n本块讲的是端口。",
        }
    )
    result = _run(service)

    assert result["no_answer"] is False
    assert result["summary"] == "综合回答。"
    assert result["refusal_kind"] is None

    combine_context = chain.calls[-1]
    assert "先清理临时文件" in combine_context
    assert NO_ANSWER_MARKER not in combine_context


def test_refusal_kind_is_reported_on_the_all_refused_path() -> None:
    """全部分支拒答时也要报出类别，否则这条路径上的区分不可见。"""
    service, _ = _three_doc_service(
        branch_outcomes={
            1: f"{NO_ANSWER_MARKER}\n没有。",
            2: f"{NO_ANSWER_MARKER}\n也没有。",
            3: f"{NO_ANSWER_MARKER}\n同样没有。",
        }
    )
    result = _run(service)

    assert result["no_answer"] is True
    assert result["refusal_kind"] == REFUSAL_KIND_INFORMATION_MISSING


def test_single_document_veto_is_reported_as_evidence_forbidden() -> None:
    """单文档路径也要分类，不能只在多分支路径上分层。"""
    service, _ = _service(
        responses=[f"{EVIDENCE_FORBIDS_MARKER}\n该数据不足以推断结论。"],
        documents=[_doc("唯一一块", 1)],
    )
    result = _run(service)

    assert result["no_answer"] is True
    assert result["summary"] == REFUSAL_TEXT
    assert result["refusal_kind"] == REFUSAL_KIND_EVIDENCE_FORBIDS
    assert EVIDENCE_FORBIDS_MARKER not in result["summary"]


def test_combined_summary_veto_is_classified() -> None:
    """合并阶段自己吐禁止型标记时，类别同样要落到返回值里。"""
    service, _ = _service(
        responses=[
            "文档一的摘要。",
            "文档二的摘要。",
            f"{EVIDENCE_FORBIDS_MARKER}\n两块合起来仍不足以判断。",
        ],
        documents=[_doc("甲", 1), _doc("乙", 2)],
    )
    result = _run(service)

    assert result["no_answer"] is True
    assert result["refusal_kind"] == REFUSAL_KIND_EVIDENCE_FORBIDS


def test_refusal_kind_is_present_on_every_return_path() -> None:
    """形状恒定，评测侧才能无条件读 —— 与 generation_health 同一个道理。

    `RAG-024` 的教训：区分做在代码里但不进产物等于没做。所以这个键必须在
    每条返回路径上都存在，作答与失败路径取 None（无拒答语义），而不是缺键。
    """
    empty_user = RagService(
        None,
        vector_store=SimpleNamespace(),
        note_service_override=SimpleNamespace(),
    )
    paths = [_run(empty_user)]

    no_documents, _ = _service(responses=[], documents=[])
    paths.append(_run(no_documents))

    gated, _ = _service(
        responses=[], documents=[_doc("量子色动力学", 1)], overlap=0
    )
    paths.append(_run(gated))

    answered, _ = _service(
        responses=["先清理临时文件。"], documents=[_doc("缓存清理", 1)]
    )
    paths.append(_run(answered))

    refused, _ = _service(
        responses=[f"{NO_ANSWER_MARKER}\n没有。"],
        documents=[_doc("无关", 1)],
    )
    paths.append(_run(refused))

    vetoed, _ = _three_doc_service(
        branch_outcomes={
            1: f"{EVIDENCE_FORBIDS_MARKER}\n不能据此推断。",
            2: "自信作答。",
            3: "也自信作答。",
        }
    )
    paths.append(_run(vetoed))

    timed_out, _ = _service(
        responses=[], documents=[_doc("超时", 1)]
    )

    async def _always_timeout(payload):
        raise asyncio.TimeoutError()

    timed_out.chain = SimpleNamespace(ainvoke=_always_timeout)
    paths.append(_run(timed_out))

    allowed = {None, REFUSAL_KIND_INFORMATION_MISSING, REFUSAL_KIND_EVIDENCE_FORBIDS}
    for result in paths:
        assert "refusal_kind" in result
        assert result["refusal_kind"] in allowed

    # 失败路径不得带拒答语义：那会把基础设施故障记成"系统正确拒答"。
    assert paths[-1]["summary"] == GENERATION_TIMEOUT_MESSAGE
    assert paths[-1]["refusal_kind"] is None


def test_a_failed_branch_cannot_be_mistaken_for_a_veto() -> None:
    """失败分支不得触发否决 —— 否则一次超时会伪装成「证据禁止作答」。

    这是 `RAG-024` 那条「失败不是拒答」在新维度上的重演：否决是很强的动作，
    只能由模型读完证据后的明确表态触发，不能由「这个分支没读到」触发。
    """
    service, chain = _three_doc_service(
        branch_outcomes={
            1: TimeoutError(),
            2: "按文档说明先清理临时文件。",
            3: RuntimeError("连接被拒"),
        }
    )
    result = _run(service)

    assert result["no_answer"] is False
    assert result["refusal_kind"] is None
    assert result["generation_health"]["branches_failed"] == 2
    assert result["generation_health"]["degraded"] is True
    assert any(
        call.startswith("以下是多个文档的摘要") for call in chain.calls
    )


def test_prompt_teaches_both_markers_and_their_difference() -> None:
    """两类拒答必须在 prompt 里可分辨 —— 这是本条验收标准的前半句。

    代码能分辨但 prompt 从不产出第二种标记的话，分层只存在于类型系统里。
    prompt 是 source_fingerprint 的一部分（`RAG-017`），所以这条同时锁住
    「改 prompt 必然改指纹」这个约束。
    """
    from pathlib import Path

    import app.rag.rag_service as rag_module

    prompt_path = (
        Path(rag_module.__file__).resolve().parents[1] / "prompt" / "rag_summarize.txt"
    )
    text = prompt_path.read_text(encoding="utf-8")

    assert NO_ANSWER_MARKER in text
    assert EVIDENCE_FORBIDS_MARKER in text
    # 必须讲清区别，否则模型无从选择用哪一个。
    assert "第 7 条" in text and "本条" in text
