from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
import types
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import anyio
import pytest


# ==================== 离线依赖与生命周期替身 ====================

def _module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class TaskRegistry:
    """只实现冻结 API；超时任务必须继续持有，不能伪装已经结束。"""

    def __init__(self) -> None:
        self.tasks: set[asyncio.Task] = set()
        self.accepting = True
        self.rejected = []
        self.created: list[asyncio.Task] = []
        self.cancel_calls: list[tuple[asyncio.Task, float]] = []
        self.cancel_finished = asyncio.Event()

    def create(self, coro, *, name: str) -> asyncio.Task:
        if not self.accepting:
            self.rejected.append(coro)
            coro.close()
            raise RuntimeError("后台任务登记已停止接单")
        task = asyncio.create_task(coro, name=name)
        self.created.append(task)
        self.tasks.add(task)
        task.add_done_callback(self._finished)
        return task

    def _finished(self, task: asyncio.Task) -> None:
        self.tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def cancel_task(self, task: asyncio.Task, *, timeout: float = 1.0) -> bool:
        self.cancel_calls.append((task, timeout))
        if not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=timeout)
        self.cancel_finished.set()
        return task.done()


class AgentHarness:
    def __init__(self) -> None:
        self.registry = TaskRegistry()
        self.context = ContextVar("m5_agent_context", default=None)
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()
        self.release = asyncio.Event()
        self.cancel_seen = asyncio.Event()
        self.cleanup_awaited = asyncio.Event()
        self.queue_waiting = asyncio.Event()
        self.queue_full = asyncio.Event()
        self.save_waiting = asyncio.Event()
        self.save_release = asyncio.Event()
        self.queue_size = None
        self.queue = None
        self.unmanaged: list[asyncio.Task] = []
        self.consumers: list[asyncio.Task] = []
        self.inputs: list[dict] = []
        self.save_attempts: list[tuple] = []
        self.saved: list[tuple] = []
        self.before_commit = None
        self.after_commit = None
        self.produce = self.answer
        self.generator = None
        self.module = None

    async def answer(self):
        yield {"output": "答"}
        yield {"output": "案"}

    async def blocked_answer(self):
        await self.release.wait()
        yield {"output": "答案"}

    async def slow_cancel_answer(self):
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancel_seen.set()
            await self.release.wait()
        yield {"output": "答案"}

    async def cancel_then_finish_answer(self):
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancel_seen.set()
            await asyncio.sleep(0)
            self.cleanup_awaited.set()
        yield {"output": "答案"}

    async def astream(self, inputs):
        assert self.context.get().user_id == "user-test"
        self.inputs.append(inputs)  # 每次调用代表一轮可能有副作用的工具执行。
        self.started.set()
        try:
            async for chunk in self.produce():
                yield chunk
        finally:
            self.stopped.set()

    async def get_history(self, session_id, user_id):
        assert (session_id, user_id) == ("session-test", "user-test")
        return []

    async def add_message(self, *args):
        self.save_attempts.append(args)
        if self.before_commit is not None:
            await self.before_commit()
        self.saved.append(args)
        if self.after_commit is not None:
            await self.after_commit()

    async def block_commit(self):
        self.save_waiting.set()
        await self.save_release.wait()

    @contextmanager
    def bind_context(self, context):
        token = self.context.set(context)
        try:
            yield
        finally:
            self.context.reset(token)

    def make_queue(self, maxsize=0):
        harness = self

        class ThinkingQueue(asyncio.Queue):
            async def get(self):
                harness.queue_waiting.set()
                return await super().get()

            async def put(self, item):
                if self.full():
                    harness.queue_full.set()
                await super().put(item)

        self.queue = ThinkingQueue(
            maxsize=maxsize if self.queue_size is None else self.queue_size
        )
        return self.queue

    def create_unmanaged(self, coro, **kwargs):
        task = asyncio.create_task(coro, **kwargs)
        self.unmanaged.append(task)
        return task

    @property
    def child(self):
        children = self.registry.created + self.unmanaged
        assert len(children) == 1
        return children[0]

    def open(self):
        self.generator = self.module.get_agent_stream_response(
            "问题", "session-test", "user-test"
        )
        return self.generator

    def consume(self, coro):
        task = asyncio.create_task(coro)
        self.consumers.append(task)
        return task

    def assert_cleaned(self):
        assert self.registry.created == [self.child]
        assert self.registry.cancel_calls == [(self.child, 1.0)]
        assert self.child.done()
        assert self.child.get_name()
        assert not self.registry.tasks
        assert self.context.get() is None

    async def cleanup(self):
        """失败也主动释放替身，避免 asyncio.run 的退出清理无限等待。"""
        self.release.set()
        self.save_release.set()
        if self.queue is not None:
            while not self.queue.empty():
                self.queue.get_nowait()
        tasks = self.registry.created + self.unmanaged + self.consumers
        for _ in range(2):
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                _, pending = await asyncio.wait(tasks, timeout=0.5)
                tasks = list(pending)
        assert not tasks, "测试替身未在清理期限内结束"
        if self.generator is not None:
            await asyncio.wait_for(self.generator.aclose(), timeout=1.5)

    def run(self, scenario):
        async def bounded():
            try:
                async with asyncio.timeout(5):
                    await scenario()
            finally:
                await self.cleanup()

        asyncio.run(bounded())

    def run_anyio(self, scenario):
        async def bounded():
            try:
                async with asyncio.timeout(5):
                    await scenario()
            finally:
                await self.cleanup()

        anyio.run(bounded, backend="asyncio")


@pytest.fixture
def agent(backend_root: Path, monkeypatch: pytest.MonkeyPatch) -> AgentHarness:
    harness = AgentHarness()
    tools = {
        name: object()
        for name in (
            "rag_summary_tools", "what_time_is_now", "get_user_info_tools",
            "search_notes_tool", "get_note_stats_tool", "get_today_reviews_tool",
            "mark_reviewed_tool", "create_note_tool", "get_related_notes_tool",
        )
    }
    modules = {
        "langsmith": _module("langsmith", traceable=lambda function: function),
        "langchain_classic.agents": _module(
            "langchain_classic.agents", AgentExecutor=object,
            create_tool_calling_agent=object,
        ),
        "langchain_ollama": _module("langchain_ollama", ChatOllama=object),
        "langchain_core.messages": _module(
            "langchain_core.messages", BaseMessage=object,
        ),
        "langchain_core.prompts": _module(
            "langchain_core.prompts", ChatPromptTemplate=object,
            MessagesPlaceholder=object,
        ),
        "langchain_core.tools": _module("langchain_core.tools", BaseTool=object),
        "app.agent.agent_middleware": _module(
            "app.agent.agent_middleware", get_middleware=lambda: [],
        ),
        "app.agent.agent_tools": _module(
            "app.agent.agent_tools", **tools, bind_tool_context=harness.bind_context,
            ToolContext=types.SimpleNamespace,
        ),
        "app.core.logger_handler": _module(
            "app.core.logger_handler", logger=logging.getLogger("m5-agent-lifecycle"),
        ),
        "app.core.task_registry": _module(
            "app.core.task_registry", background_tasks=harness.registry,
            cancel_task=harness.registry.cancel_task,
        ),
        "app.services": _module(
            "app.services", session_manager=types.SimpleNamespace(session_manager=harness),
        ),
        "app.utils.factory": _module(
            "app.utils.factory", build_aliyun_chat_model=object,
        ),
        "app.utils.prompt_loader": _module(
            "app.utils.prompt_loader", load_prompt=lambda _: "测试提示词",
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    name = "app.agent._m5_agent_lifecycle"
    spec = importlib.util.spec_from_file_location(name, backend_root / "app/agent/agent.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.agent_factory, "create_agent_executor", lambda **_: harness)
    # 仅替换被测模块的 asyncio 引用，不污染其他测试或事件循环本身。
    async_module = _module("asyncio")
    async_module.__dict__.update(vars(asyncio))
    async_module.Queue = harness.make_queue
    async_module.create_task = harness.create_unmanaged
    monkeypatch.setattr(module, "asyncio", async_module)
    harness.module = module
    return harness


def _event(frame: str) -> dict:
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    return json.loads(frame[6:])


async def _collect(stream) -> list[dict]:
    return [_event(frame) async for frame in stream]


# ==================== 关闭、取消与异常 ====================

@pytest.mark.p0
def test_closed_registry_emits_error_done_without_starting_or_saving(agent):
    async def scenario():
        agent.registry.accepting = False
        events = await _collect(agent.open())
        assert [event["type"] for event in events] == ["error", "done"]
        assert "停止接单" in events[0]["content"]
        assert events[0]["session_id"] == "session-test"
        assert not agent.registry.created and not agent.registry.cancel_calls
        assert len(agent.registry.rejected) == 1
        assert agent.registry.rejected[0].cr_frame is None
        assert not agent.inputs and not agent.save_attempts and not agent.saved

    agent.run(scenario)


@pytest.mark.p0
@pytest.mark.parametrize("start_child", [False, True])
def test_first_yield_aclose_cleans_managed_child(agent, start_child):
    async def scenario():
        agent.produce = agent.blocked_answer
        stream = agent.open()
        assert _event(await anext(stream))["content"] == ""
        if start_child:
            await agent.started.wait()
        await asyncio.wait_for(stream.aclose(), timeout=1.5)
        agent.assert_cleaned()
        assert agent.child.cancelled()
        assert not agent.saved and not agent.save_attempts

    agent.run(scenario)


@pytest.mark.p0
def test_cancel_waiting_consumer_propagates_and_cleans_child(agent):
    async def scenario():
        agent.produce = agent.blocked_answer
        stream = agent.open()
        await anext(stream)
        consumer = agent.consume(anext(stream))
        await agent.queue_waiting.wait()
        await agent.started.wait()
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        agent.assert_cleaned()
        assert agent.child.cancelled()
        assert not agent.saved and not agent.save_attempts
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    agent.run(scenario)


@pytest.mark.p0
def test_agent_exception_reports_error_without_persistence_or_replay(agent):
    async def fail():
        raise RuntimeError("工具执行失败")
        yield  # 异步生成器替身，与 astream 的迭代协议一致。

    async def scenario():
        agent.produce = fail
        events = await _collect(agent.open())
        assert [event["type"] for event in events] == ["response", "error", "done"]
        assert "工具执行失败" in events[1]["content"]
        assert not agent.saved and not agent.save_attempts
        assert len(agent.inputs) == 1
        agent.assert_cleaned()

    agent.run(scenario)


@pytest.mark.p0
@pytest.mark.parametrize("start_child", [False, True])
def test_child_cancel_is_not_converted_to_saved_answer(agent, start_child):
    async def scenario():
        agent.produce = agent.blocked_answer
        stream = agent.open()
        await anext(stream)
        if start_child:
            await agent.started.wait()
        agent.child.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(anext(stream), timeout=0.5)
        assert not agent.save_attempts
        agent.assert_cleaned()

    agent.run(scenario)


@pytest.mark.p0
def test_outer_error_then_aclose_still_cleans_child(agent):
    async def scenario():
        agent.produce = agent.blocked_answer
        stream = agent.open()
        await anext(stream)
        await agent.started.wait()
        event = _event(await stream.athrow(RuntimeError("流处理失败")))
        assert event["type"] == "error" and "流处理失败" in event["content"]
        await asyncio.wait_for(stream.aclose(), timeout=1.5)
        agent.assert_cleaned()
        assert not agent.saved

    agent.run(scenario)


@pytest.mark.p0
@pytest.mark.parametrize("cleanup_thinking", [False, True])
def test_full_thinking_queue_disconnect_does_not_block_cleanup(agent, cleanup_thinking):
    async def produce():
        callback = agent.context.get().thinking_callback
        try:
            await callback({"type": "thinking", "content": "第一条"})
            await callback({"type": "thinking", "content": "第二条"})
        finally:
            if cleanup_thinking:
                await callback({"type": "thinking", "content": "工具清理"})
        yield {"output": "答案"}

    async def scenario():
        agent.queue_size = 1
        agent.produce = produce
        stream = agent.open()
        await anext(stream)
        await agent.queue_full.wait()
        assert agent.queue.full()
        await asyncio.wait_for(stream.aclose(), timeout=1.5)
        agent.assert_cleaned()
        assert agent.stopped.is_set()
        assert not agent.save_attempts

    agent.run(scenario)


@pytest.mark.p0
@pytest.mark.parametrize("disconnect", ["aclose", "cancel"])
def test_cancel_timeout_keeps_task_registered_and_does_not_claim_stopped(
    agent, caplog, disconnect,
):
    async def scenario():
        agent.produce = agent.slow_cancel_answer
        stream = agent.open()
        await anext(stream)
        await agent.started.wait()
        if disconnect == "aclose":
            await asyncio.wait_for(stream.aclose(), timeout=1.5)
        else:
            consumer = agent.consume(anext(stream))
            await agent.queue_waiting.wait()
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(consumer, timeout=1.5)
        assert agent.registry.created == [agent.child]
        assert agent.registry.cancel_calls == [(agent.child, 1.0)]
        assert agent.cancel_seen.is_set() and not agent.child.done()
        assert agent.child in agent.registry.tasks
        assert any("超时" in message and "尚未结束" in message for message in caplog.messages)
        assert not any("已停止" in message for message in caplog.messages)
        assert "【Agent流式响应】添加到会话历史成功" not in caplog.messages
        assert not agent.save_attempts
        agent.release.set()
        await asyncio.wait_for(agent.child, timeout=0.5)
        assert not agent.registry.tasks
        assert len(agent.inputs) == 1 and not agent.saved

    agent.run(scenario)


@pytest.mark.p0
@pytest.mark.parametrize("disconnect", ["aclose", "cancel"])
def test_parent_cancel_during_cleanup_is_not_swallowed(agent, disconnect):
    async def scenario():
        agent.produce = agent.slow_cancel_answer
        stream = agent.open()
        await anext(stream)
        await agent.started.wait()
        if disconnect == "aclose":
            closer = agent.consume(stream.aclose())
        else:
            closer = agent.consume(anext(stream))
            await agent.queue_waiting.wait()
            closer.cancel()
        await agent.cancel_seen.wait()
        closer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closer
        assert agent.registry.created == [agent.child]
        assert agent.child in agent.registry.tasks and not agent.child.done()
        agent.release.set()
        await asyncio.wait_for(agent.child, timeout=0.5)
        agent.assert_cleaned()
        assert not agent.save_attempts

    agent.run(scenario)


@pytest.mark.p0
def test_anyio_level_cancel_waits_for_agent_cleanup_and_propagates(agent):
    async def scenario():
        agent.produce = agent.cancel_then_finish_answer
        stream = agent.open()
        await anext(stream)
        await agent.started.wait()
        with anyio.CancelScope() as scope:
            scope.cancel()
            with pytest.raises(asyncio.CancelledError):
                await anext(stream)
        assert agent.cancel_seen.is_set()
        assert agent.cleanup_awaited.is_set()
        assert agent.registry.cancel_finished.is_set()
        agent.assert_cleaned()

    agent.run_anyio(scenario)


@pytest.mark.p0
def test_anyio_level_cancel_timeout_keeps_agent_registered(agent):
    async def scenario():
        agent.produce = agent.slow_cancel_answer
        stream = agent.open()
        await anext(stream)
        await agent.started.wait()
        with anyio.CancelScope() as scope:
            scope.cancel()
            with pytest.raises(asyncio.CancelledError):
                await anext(stream)
        assert agent.cancel_seen.is_set()
        assert agent.registry.cancel_finished.is_set()
        assert not agent.child.done()
        assert agent.child in agent.registry.tasks
        agent.release.set()
        await asyncio.wait_for(agent.child, timeout=0.5)
        assert not agent.registry.tasks
        assert not agent.save_attempts and not agent.saved

    agent.run_anyio(scenario)


@pytest.mark.p0
def test_thinking_events_preserve_order_under_queue_backpressure(agent):
    async def produce():
        callback = agent.context.get().thinking_callback
        for index in range(3):
            await callback({"type": "thinking", "content": str(index)})
        yield {"output": "答案"}

    async def scenario():
        agent.queue_size = 1
        agent.produce = produce
        events = await _collect(agent.open())
        assert [event["content"] for event in events if event["type"] == "thinking"] == [
            "0", "1", "2",
        ]
        assert agent.queue_full.is_set()
        await asyncio.wait_for(agent.queue.join(), timeout=0.5)
        assert agent.saved == [("session-test", "user-test", "问题", "答案")]
        assert len(agent.inputs) == 1 and events[-1]["type"] == "done"
        agent.assert_cleaned()

    agent.run(scenario)


# ==================== 持久化顺序与禁止整轮重放 ====================

@pytest.mark.p0
def test_success_saves_full_response_once_before_answer_characters(agent, caplog):
    caplog.set_level(logging.INFO, logger="m5-agent-lifecycle")

    async def scenario():
        events = []
        async for frame in agent.open():
            event = _event(frame)
            if event["type"] == "response" and event["content"]:
                assert agent.saved == [("session-test", "user-test", "问题", "答案")]
            events.append(event)
        assert [event["type"] for event in events] == [
            "response", "response", "response", "done",
        ]
        assert "".join(event.get("content", "") for event in events) == "答案"
        assert events[-1]["session_id"] == "session-test"
        assert agent.save_attempts == agent.saved and len(agent.saved) == 1
        assert len(agent.inputs) == 1
        assert agent.queue.maxsize > 0
        assert caplog.messages.count("【Agent流式响应】添加到会话历史成功") == 1
        agent.assert_cleaned()

    agent.run(scenario)


@pytest.mark.p0
@pytest.mark.parametrize("committed", [False, True])
def test_disconnect_during_save_does_not_claim_success_or_replay(agent, caplog, committed):
    caplog.set_level(logging.INFO, logger="m5-agent-lifecycle")

    async def scenario():
        if committed:
            agent.after_commit = agent.block_commit
        else:
            agent.before_commit = agent.block_commit
        stream = agent.open()
        assert _event(await anext(stream))["content"] == ""
        consumer = agent.consume(anext(stream))
        await agent.save_waiting.wait()
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert len(agent.save_attempts) == 1
        assert len(agent.saved) == int(committed)
        assert len(agent.inputs) == 1
        assert "【Agent流式响应】添加到会话历史成功" not in caplog.messages
        agent.assert_cleaned()
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    agent.run(scenario)


@pytest.mark.p0
def test_aclose_after_first_answer_keeps_single_committed_turn(agent):
    async def scenario():
        stream = agent.open()
        await anext(stream)
        assert _event(await anext(stream))["content"] == "答"
        await stream.aclose()
        assert agent.saved == [("session-test", "user-test", "问题", "答案")]
        assert agent.save_attempts == agent.saved
        assert len(agent.inputs) == 1
        agent.assert_cleaned()
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    agent.run(scenario)


@pytest.mark.p0
def test_persistence_failure_reports_error_without_retry(agent, caplog):
    caplog.set_level(logging.INFO, logger="m5-agent-lifecycle")

    async def fail():
        raise RuntimeError("提交失败")

    async def scenario():
        agent.before_commit = fail
        events = await _collect(agent.open())
        assert [event["type"] for event in events] == ["response", "error", "done"]
        assert "提交失败" in events[1]["content"]
        assert len(agent.save_attempts) == 1 and not agent.saved
        assert len(agent.inputs) == 1
        assert "【Agent流式响应】添加到会话历史成功" not in caplog.messages
        agent.assert_cleaned()

    agent.run(scenario)
