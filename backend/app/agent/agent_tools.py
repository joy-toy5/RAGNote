from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from langchain_core.tools import tool

from app.core.logger_handler import logger
from app.rag.rag_service import RagService
from app.services.note_service import note_service
from app.services.review_service import review_service
from app.db.db_config import AsyncSessionLocal

import datetime

ThinkingCallback = Callable[[dict], object]


@dataclass(frozen=True)
class ToolContext:
    """只能由已鉴权请求绑定的 Agent 工具上下文。"""

    user_id: str
    username: Optional[str] = None
    thinking_callback: Optional[ThinkingCallback] = None

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, str) or not self.user_id.strip():
            raise ValueError("工具上下文必须包含有效的用户 ID")
        object.__setattr__(self, "user_id", self.user_id.strip())


tool_context_var: ContextVar[Optional[ToolContext]] = ContextVar(
    "tool_context", default=None
)


@contextmanager
def bind_tool_context(context: ToolContext) -> Iterator[ToolContext]:
    """在当前执行上下文中绑定身份，并在退出时精确复位。"""
    token = tool_context_var.set(context)
    try:
        yield context
    finally:
        tool_context_var.reset(token)


def require_tool_context() -> ToolContext:
    """返回可信请求上下文；缺失身份时拒绝执行工具。"""
    context = tool_context_var.get()
    if context is None:
        raise RuntimeError("Agent 工具缺少可信用户上下文")
    return context


def get_current_user_id_from_context() -> str:
    """从可信上下文获取当前用户 ID。"""
    return require_tool_context().user_id


def get_thinking_callback_from_context() -> Optional[ThinkingCallback]:
    """从可信上下文获取思考过程回调。"""
    return require_tool_context().thinking_callback

@tool(description="用于从向量数据库里检索文档并生成摘要，返回包含文档列表和摘要的结果。返回格式为：'摘要: [摘要内容]\n\n检索到的文档列表:\n1. [文档1内容]\n2. [文档2内容]\n...'。注意：文档已经过自动重排序，无需再调用重排序工具")
async def rag_summary_tools(query: str) -> str:
    """RAG 摘要工具"""
    context = require_tool_context()
    result = await RagService(
        context.user_id,
        thinking_callback=context.thinking_callback,
    ).get_documents_and_summary(query)
    documents = result.get("documents", [])
    summary = result.get("summary", "")

    formatted_result = f"摘要: {summary}\n\n"
    formatted_result += "检索到的文档列表（已重排序）:\n"
    for i, doc in enumerate(documents, 1):
        formatted_result += f"{i}. {doc}\n"

    return formatted_result

@tool(description="当用户明确询问自己的身份时，返回当前已鉴权用户的信息")
async def get_user_info_tools() -> str:
    """从可信请求上下文获取用户信息。"""
    context = require_tool_context()
    username = context.username or "未知"
    return f"用户信息：\n- 用户ID: {context.user_id}\n- 用户名: {username}"

@tool(description="用于获取当前年月日时分的工具")
async def what_time_is_now() -> str:
    """获取当前年月日时分的工具"""
    return f"当前时间是：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"

@tool(description="语义搜索用户的笔记，根据关键词返回最相关的笔记列表。参数 query 为搜索关键词，top_k 为返回结果数量（默认5）。")
async def search_notes_tool(query: str, top_k: int = 5) -> str:
    """搜索笔记工具"""
    user_id = get_current_user_id_from_context()
    if not user_id:
        return "错误: 无法确定用户身份"
    async with AsyncSessionLocal() as db:
        try:
            results = await note_service.search_notes(db, user_id, query, top_k=top_k)
            if not results:
                return "未找到相关笔记"
            lines = [f"找到 {len(results)} 篇相关笔记：\n"]
            for i, note in enumerate(results, 1):
                lines.append(f"{i}. **{note.title}**")
                if note.category:
                    lines.append(f"   分类: {note.category}")
                if note.tags:
                    lines.append(f"   标签: {', '.join(note.tags)}")
                lines.append(f"   内容预览: {note.content[:200]}...\n")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"搜索笔记失败: {e}")
            return f"搜索笔记时出错: {str(e)}"

@tool(description="获取用户的笔记统计信息，包括笔记总数、各分类（工作/学习/生活/项目）的笔记数量。")
async def get_note_stats_tool() -> str:
    """笔记统计工具"""
    user_id = get_current_user_id_from_context()
    if not user_id:
        return "错误: 无法确定用户身份"
    async with AsyncSessionLocal() as db:
        try:
            stats = await note_service.get_category_stats(db, user_id)
            lines = ["📊 笔记统计\n"]
            lines.append(f"总笔记数: {stats['total']}\n")
            lines.append("各分类:")
            for cat in stats['categories']:
                emoji = {'work': '💼', 'study': '📖', 'life': '🏠', 'project': '🚀'}.get(cat['category'], '📄')
                lines.append(f"  {emoji} {cat['category']}: {cat['count']} 篇")
            if stats['uncategorized'] > 0:
                lines.append(f"  📄 未分类: {stats['uncategorized']} 篇")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"获取笔记统计失败: {e}")
            return f"获取笔记统计时出错: {str(e)}"

@tool(description="获取今日待回顾的笔记列表。返回每篇笔记的标题、内容预览和回顾次数，帮助用户进行间隔重复复习。")
async def get_today_reviews_tool() -> str:
    """获取今日回顾列表工具"""
    user_id = get_current_user_id_from_context()
    if not user_id:
        return "错误: 无法确定用户身份"
    async with AsyncSessionLocal() as db:
        try:
            reviews = await review_service.get_today_reviews(db, user_id)
            if not reviews:
                return "今日没有待回顾的笔记，继续保持！"
            lines = [f"📅 今日待回顾笔记（共 {len(reviews)} 篇）\n"]
            for i, rv in enumerate(reviews, 1):
                lines.append(f"{i}. **{rv['title']}**")
                lines.append(f"   回顾次数: 第 {rv['review_count'] + 1} 次")
                lines.append(f"   内容预览: {rv['content_preview'][:100]}...\n")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"获取今日回顾失败: {e}")
            return f"获取今日回顾时出错: {str(e)}"

@tool(description="标记一篇笔记为已回顾。参数 note_id 为笔记ID。调用成功后笔记的下次回顾时间会自动按艾宾浩斯遗忘曲线延后。")
async def mark_reviewed_tool(note_id: str) -> str:
    """标记回顾完成工具"""
    user_id = get_current_user_id_from_context()
    if not user_id:
        return "错误: 无法确定用户身份"
    async with AsyncSessionLocal() as db:
        try:
            result = await review_service.mark_reviewed(db, note_id, user_id)
            if result["success"]:
                return f"✅ 已标记回顾完成！第 {result['review_count']} 次回顾，下次回顾间隔 {result['interval_days']} 天。"
            else:
                return f"标记失败: {result['message']}"
        except Exception as e:
            logger.error(f"标记回顾失败: {e}")
            return f"标记回顾时出错: {str(e)}"

@tool(description="创建一篇新笔记。参数 title 为笔记标题，content 为笔记内容（支持Markdown格式，可选，不传则只创建标题）。创建后会自动生成向量索引和智能标签。")
async def create_note_tool(title: str, content: str = "") -> str:
    """创建笔记工具"""
    user_id = get_current_user_id_from_context()
    if not user_id:
        return "错误: 无法确定用户身份"
    from app.schemas.models import NoteCreate
    async with AsyncSessionLocal() as db:
        try:
            payload = NoteCreate(title=title, content=content)
            note = await note_service.create_note(db, user_id, payload)
            return f"✅ 笔记创建成功！\n- 标题: {note.title}\n- ID: {note.id}\n- 标签和分类正在后台生成中..."
        except Exception as e:
            logger.error(f"创建笔记失败: {e}")
            return f"创建笔记时出错: {str(e)}"

@tool(description="获取某篇笔记的关联推荐，包括语义相似的笔记和知识库文档。参数 note_id 为笔记ID，top_k 为返回数量（默认3）。")
async def get_related_notes_tool(note_id: str, top_k: int = 3) -> str:
    """关联笔记推荐工具"""
    user_id = get_current_user_id_from_context()
    if not user_id:
        return "错误: 无法确定用户身份"
    async with AsyncSessionLocal() as db:
        try:
            related = await note_service.get_related_notes(db, note_id, user_id, top_k=top_k)
            if not related:
                return "未找到关联笔记或知识库文档"
            lines = [f"🔗 关联推荐（共 {len(related)} 项）\n"]
            for i, item in enumerate(related, 1):
                source_label = "📝 笔记" if item['source'] == 'note' else "📚 知识库"
                lines.append(f"{i}. {source_label} — {item['title']}")
                lines.append(f"   相似度: {item['similarity']}")
                lines.append(f"   预览: {item['content_preview'][:100]}...\n")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"获取关联推荐失败: {e}")
            return f"获取关联推荐时出错: {str(e)}"
