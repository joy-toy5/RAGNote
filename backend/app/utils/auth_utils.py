import asyncio
import os
import json
from typing import Any, Callable, Dict, Optional, Protocol
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.failed_response import logger
from app.db.redis_config import connect_redis, set_redis_cache

load_dotenv()

# Django JWT配置
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

# 创建Bearer认证方案
security = HTTPBearer()

PRODUCTION_ENVIRONMENTS = {"prod", "production"}
DEVELOPMENT_CORS_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)


def get_cors_origins() -> list[str]:
    """返回显式 CORS allowlist；开发环境仅默认允许本地前端。"""
    raw_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
    if raw_origins.strip():
        return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    environment = os.getenv("ENV", "dev").strip().lower()
    return [] if environment in PRODUCTION_ENVIRONMENTS else list(DEVELOPMENT_CORS_ORIGINS)


def validate_auth_config() -> None:
    """验证 FastAPI 生产身份和跨域配置，缺少关键保护时拒绝启动。"""
    environment = os.getenv("ENV", "dev").strip().lower()
    if environment not in {"dev", "development", *PRODUCTION_ENVIRONMENTS}:
        raise RuntimeError("ENV 必须是 dev、development、prod 或 production")
    if environment not in PRODUCTION_ENVIRONMENTS:
        return

    secret = os.getenv("SECRET_KEY", "")
    if len(secret) < 50 or len(set(secret)) < 5 or "YOUR_" in secret.upper():
        raise RuntimeError("生产环境 SECRET_KEY 不满足强度要求")
    if os.getenv("ALGORITHM") != "HS256":
        raise RuntimeError("生产环境 ALGORITHM 必须显式设置为 HS256")

    parsed_url = urlparse(os.getenv("DJANGO_API_URL", ""))
    is_loopback_http = parsed_url.scheme == "http" and parsed_url.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }
    if not parsed_url.hostname or not (
        parsed_url.scheme == "https" or is_loopback_http
    ):
        raise RuntimeError("生产环境 DJANGO_API_URL 必须使用 HTTPS 或回环地址")

    origins = get_cors_origins()
    if not origins or "*" in origins:
        raise RuntimeError("生产环境必须配置非通配符 CORS allowlist")
    if any(urlparse(origin).scheme != "https" for origin in origins):
        raise RuntimeError("生产环境 CORS 来源必须使用 HTTPS")


class UserStatusVerifier(Protocol):
    """验证 JWT 对应账户当前仍然可用。"""

    async def is_active(self, user_id: str, token: str) -> bool:
        ...


class DjangoUserStatusVerifier:
    """通过 Django 的受保护用户详情接口验证账户状态。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        request_get: Callable[..., Any] = requests.get,
    ) -> None:
        self.base_url = (base_url or os.getenv("DJANGO_API_URL", "")).rstrip("/")
        self.request_get = request_get

    async def is_active(self, user_id: str, token: str) -> bool:
        if not self.base_url:
            raise RuntimeError("未配置 DJANGO_API_URL")

        response = await asyncio.to_thread(
            self.request_get,
            url=f"{self.base_url}/user/detail/",
            headers={"Authorization": f"Bearer {token}"},
            timeout=(2.0, 3.0),
        )
        if response.status_code != 200:
            return False

        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        return (
            payload.get("success") is True
            and isinstance(data, dict)
            and data.get("id") == user_id
        )


def get_user_status_verifier() -> UserStatusVerifier:
    """构造请求级账户状态验证器。"""
    return DjangoUserStatusVerifier()


def decode_django_jwt(token: str) -> Optional[Dict[str, Any]]:
    """解析Django生成的JWT token
    
    Args:
        token: JWT token字符串
        
    Returns:
        解析后的payload，如果解析失败返回None
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    verifier: UserStatusVerifier = Depends(get_user_status_verifier),
) -> str:
    """从Django JWT中获取当前用户UUID
    
    Args:
        credentials: HTTP认证凭据
        
    Returns:
        用户的UUID
        
    Raises:
        HTTPException: 认证失败时抛出
    """
    token = credentials.credentials
    payload = decode_django_jwt(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    jti = payload.get("jti")
    user_id = payload.get("user_id")
    if not isinstance(jti, str) or not jti:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing a valid identifier",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not find user ID in token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        redis_client = await connect_redis()
        revoked = await redis_client.get(f"jwt:revoked:{jti}")
    except Exception as exc:
        logger.error(
            f"检查 JWT 撤销状态失败: {exc}",
            extra={"path": "auth_utils.get_current_user_id"},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication state service unavailable",
        ) from exc
    if revoked is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        is_active = await verifier.is_active(user_id, token)
    except Exception as exc:
        logger.error(
            f"验证用户状态失败: {exc}",
            extra={"path": "auth_utils.get_current_user_id"},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User status service unavailable",
        ) from exc
    if not is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is not active",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_id


async def fetch_user_info_from_django_api(token: str, url: str) -> Optional[Dict[str, Any]]:
    """从Django API获取用户信息
    
    Args:
        token: JWT token字符串
        
    Returns:
        用户信息字典，如果获取失败返回None
    """

    try:
        # 构建请求头
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        response = await asyncio.to_thread(
            requests.get,
            url=url,
            headers=headers,
            timeout=(2.0, 3.0),
        )
        
        if response.status_code == 200:
            user_data = response.json()
            logger.info("【debug】 从Django API获取用户信息成功", extra={"path": "auth_utils.fetch_user_info_from_django_api"})
            return user_data
        else:
            logger.error(f"【debug】 从Django API获取用户信息失败，status_code: {response.status_code}", extra={"path": "auth_utils.fetch_user_info_from_django_api"})
            return None
    except Exception as e:
        logger.error(f"【debug】 调用Django API时出错: {str(e)}", extra={"path": "auth_utils.fetch_user_info_from_django_api"})
        return None


async def get_user_info_from_redis(user_id: str, credentials: HTTPAuthorizationCredentials):
    """从Redis中获取用户信息
    
    Args:
        user_id: 用户ID
        credentials: HTTP认证凭据
        
    Returns:
        用户信息
    """
    redis_client = await connect_redis()
    key = f":1:user:{user_id}"
    
    try:
        # 从Redis中获取用户信息
        user_info = await redis_client.get(key)
        if user_info is None:
            # 降级调用django查询用户信息
            user_data = await fetch_user_info_from_django_api(credentials.credentials, os.getenv("DJANGO_API_URL") + "/user/detail/")
            if user_data:
                # 将用户信息存入Redis，设置过期时间为1小时
                await set_redis_cache(
                    key,
                    user_data,
                    expire=3600
                )
                user_info = user_data
        else:
            # 如果从Redis中获取到数据，尝试将其解析为字典
            try:
                
                user_info = json.loads(user_info)
            except json.JSONDecodeError:
                # 如果解析失败，删除旧数据并重新获取
                await redis_client.delete(key)
                user_data = await fetch_user_info_from_django_api(credentials.credentials, os.getenv("DJANGO_API_URL") + "/user/detail/")
                if user_data:
                    await set_redis_cache(
                        key,
                        user_data,
                        expire=3600
                    )
                    user_info = user_data
                else:
                    user_info = None
    except UnicodeDecodeError:
        # 处理解码错误，删除旧数据并重新获取
        await redis_client.delete(key)
        user_data = await fetch_user_info_from_django_api(credentials.credentials, os.getenv("DJANGO_API_URL") + "/user/detail/")
        if user_data:
            await set_redis_cache(
                key,
                user_data,
                expire=3600
            )
            user_info = user_data
        else:
            user_info = None

    return user_info
