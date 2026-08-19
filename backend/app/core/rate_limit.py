import os

from fastapi import Request, HTTPException

from app.db.redis_config import connect_redis


PRODUCTION_ENVIRONMENTS = {"prod", "production"}
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def is_rate_limit_enabled() -> bool:
    """动态读取限流开关，生产环境缺省时保持启用。"""
    raw_value = os.getenv("RATE_LIMIT_ENABLED")
    if raw_value is None:
        return True
    value = raw_value.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise RuntimeError("RATE_LIMIT_ENABLED 必须是布尔值")


def validate_rate_limit_config() -> None:
    """拒绝以关闭限流的配置启动生产服务。"""
    environment = os.getenv("ENV", "dev").strip().lower()
    if environment in PRODUCTION_ENVIRONMENTS and not is_rate_limit_enabled():
        raise RuntimeError("生产环境禁止关闭限流")


def rate_limit(limit: int = 1, window: int = 60):
    """
    限流依赖函数
    :param limit: 时间窗口内的最大请求数
    :param window: 时间窗口大小（秒）
    :return: 依赖函数
    """
    async def dependency(request: Request):
        # 全局开关关闭时直接放行，不做任何限流检查
        if not is_rate_limit_enabled():
            return

        # 获取客户端IP
        client_ip = request.client.host
        if not client_ip:
            client_ip = request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or 'unknown'

        # 生成限流键
        key = f"rate_limit:aichat:{client_ip}"

        # 获取Redis连接
        redis = await connect_redis()
        
        # 获取当前计数
        current = await redis.get(key)
        current = int(current) if current else 0

        if current >= limit:
            # 限流触发
            raise HTTPException(
                status_code=429,
                detail="请求过于频繁，请稍后再试"
            )

        # 增加计数
        if current == 0:
            # 第一次请求，设置过期时间
            await redis.setex(key, window, 1)
        else:
            # 后续请求，增加计数
            await redis.incr(key)

    return dependency


class RateLimitMiddleware:
    """
    全局限流中间件
    """
    def __init__(self, app, limit: int = 100, window: int = 60):
        self.app = app
        self.limit = limit
        self.window = window

    async def __call__(self, scope, receive, send):
        # 全局开关关闭时直接放行
        if not is_rate_limit_enabled():
            await self.app(scope, receive, send)
            return

        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        # 构建请求对象
        from fastapi import Request
        request = Request(scope, receive)
        
        # 获取客户端IP
        client_ip = request.client.host
        if not client_ip:
            client_ip = request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or 'unknown'

        # 生成限流键
        key = f"rate_limit:global:{client_ip}"

        # 获取Redis连接
        redis = await connect_redis()
        
        # 获取当前计数
        current = await redis.get(key)
        current = int(current) if current else 0

        if current >= self.limit:
            # 限流触发
            from starlette.responses import JSONResponse
            response = JSONResponse(
                {"detail": "请求过于频繁，请稍后再试"},
                status_code=429
            )
            await response(scope, receive, send)
            return

        # 增加计数
        if current == 0:
            # 第一次请求，设置过期时间
            await redis.setex(key, self.window, 1)
        else:
            # 后续请求，增加计数
            await redis.incr(key)

        await self.app(scope, receive, send)
