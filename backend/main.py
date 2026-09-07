import asyncio
import time
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware

from app.core.task_registry import background_tasks
from app.rag.upload_runtime import upload_runtime
from app.rag.bootstrap import initialize_rag_resources
from app.db.db_config import async_engine, check_database_schema
from app.db.redis_config import connect_redis, close_redis
from app.router.chat import chat_router
from app.router.knowledge_router import knowledge_router
from app.router.health import health_router
from app.router.user import user_router
from app.router.note_router import note_router
from app.router.review_router import review_router
from app.router.task_router import task_router

from app.services.database_session_manager import init_database_session_manager

from app.core.failed_response_register import register_exception_handlers
from app.core.rate_limit import RateLimitMiddleware, validate_rate_limit_config
from app.core.logger_handler import logger
from app.utils.auth_utils import get_cors_origins, validate_auth_config

from app.rag.reorder_service import check_and_download_reranker_model

# 加载环境变量
load_dotenv()

app = FastAPI()
SHUTDOWN_GRACE_SECONDS = 5.0
_shutdown_task = None


# 中间件始终装配；开发环境可通过开关在内部放行，生产环境禁止关闭。
app.add_middleware(RateLimitMiddleware, limit=100, window=60)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    return response

# 集成API路由
app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(health_router)
app.include_router(user_router)
app.include_router(note_router)
app.include_router(review_router)
app.include_router(task_router)




app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# 注册异常处理函数
register_exception_handlers(app)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化会话管理器"""
    global _shutdown_task
    _shutdown_task = None
    validate_rate_limit_config()
    validate_auth_config()

    await check_database_schema()
    logger.info("数据库 schema/revision 兼容性检查完成")
    
    # 使用数据库版本的会话管理器
    await init_database_session_manager()
    logger.info("数据库会话管理器初始化完成")

    # 连接Redis
    await connect_redis()
    logger.info("Redis连接初始化完成")
    
    # 检查并重排序模型
    check_and_download_reranker_model()
    logger.info("重排序模型检查完成")
    initialize_rag_resources()
    logger.info("RAG资源初始化完成")
    background_tasks.start()
    upload_runtime.start()

def begin_shutdown():
    """信号入口只停止新任务并安排收尾，不在信号回调中关闭存储。"""
    global _shutdown_task
    background_tasks.stop_accepting()
    upload_runtime.stop_accepting()
    if _shutdown_task is None:
        _shutdown_task = asyncio.create_task(_drain_tasks(), name="application-drain")
    return _shutdown_task


async def _drain_tasks():
    pending = await background_tasks.drain(timeout=SHUTDOWN_GRACE_SECONDS)
    if pending:
        await background_tasks.cancel_and_wait(timeout=1.0)
    # 停止剩余解析排队，已运行线程仍保留原P0真实执行边界。
    return await upload_runtime.shutdown(timeout=1.0)


@app.on_event("shutdown")
async def shutdown_event():
    """先等待任务，超时取消，再有界清理连接；不声称能杀死Python线程。"""
    try:
        await asyncio.shield(begin_shutdown())
    finally:
        try:
            await asyncio.wait_for(close_redis(), timeout=1.0)
            logger.info("Redis连接已关闭")
        finally:
            await asyncio.wait_for(async_engine.dispose(), timeout=1.0)
            logger.info("数据库连接池已关闭")
