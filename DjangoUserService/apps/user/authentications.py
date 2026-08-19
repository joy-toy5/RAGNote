import logging
import time
import uuid

import jwt
from django.conf import settings
from django.core.cache import cache
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import APIException, AuthenticationFailed

from .models import User, UserStatusChoice

logger = logging.getLogger(__name__)
REQUIRED_CLAIMS = ("exp", "iat", "jti", "user_id")


class TokenRevocationUnavailable(APIException):
    status_code = 503
    default_detail = "认证状态服务暂不可用，请稍后重试"
    default_code = "token_revocation_unavailable"


def _revocation_key(jti: str) -> str:
    return f"jwt:revoked:{jti}"


def _decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": list(REQUIRED_CLAIMS)},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationFailed("Token已过期，请重新登录") from exc
    except jwt.PyJWTError as exc:
        raise AuthenticationFailed("无效的Token") from exc

    if not isinstance(payload.get("jti"), str) or not payload["jti"]:
        raise AuthenticationFailed("Token缺少有效的唯一标识")
    if not isinstance(payload.get("user_id"), str) or not payload["user_id"]:
        raise AuthenticationFailed("Token中未包含有效的用户信息")
    return payload


def _get_active_user(user_id: str) -> User:
    try:
        user = User.objects.get(uuid=user_id)
    except User.DoesNotExist as exc:
        raise AuthenticationFailed("用户不存在") from exc

    if user.status != UserStatusChoice.ACTIVE:
        raise AuthenticationFailed("用户状态异常")
    return user


def _token_ttl(payload: dict) -> int:
    ttl = int(payload["exp"]) - int(time.time())
    if ttl <= 0:
        raise AuthenticationFailed("Token已过期，请重新登录")
    return ttl


def _is_revoked(jti: str) -> bool:
    try:
        return bool(cache.get(_revocation_key(jti)))
    except Exception as exc:
        logger.exception("读取 JWT 撤销状态失败")
        raise TokenRevocationUnavailable() from exc


class JWTAuthentication(BaseAuthentication):
    """验证 Bearer JWT，并以数据库账户状态作为身份边界。"""

    def authenticate(self, request) -> tuple[User, str] | None:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None

        try:
            auth_type, token = auth_header.split(" ", 1)
        except ValueError as exc:
            raise AuthenticationFailed("认证头格式错误") from exc
        if auth_type.lower() != "bearer" or not token:
            raise AuthenticationFailed("认证类型错误，应为Bearer")

        payload = _decode_token(token)
        if _is_revoked(payload["jti"]):
            raise AuthenticationFailed("Token已被撤销")
        user = _get_active_user(payload["user_id"])
        return user, token

    def authenticate_header(self, request) -> str:
        return "Bearer"


class JWTTokenGenerator:
    """签发 JWT，并提供单次原子轮换和精确撤销。"""

    @staticmethod
    def generate_token(user: User) -> tuple[str, int]:
        if user.status != UserStatusChoice.ACTIVE:
            raise AuthenticationFailed("用户状态异常")

        issued_at = int(time.time())
        expire_time = issued_at + settings.JWT_TOKEN_TTL_SECONDS
        payload = {
            "user_id": str(user.uuid),
            "username": user.username,
            "email": user.email,
            "exp": expire_time,
            "iat": issued_at,
            "jti": str(uuid.uuid4()),
        }
        token = jwt.encode(
            payload,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )
        return token, expire_time

    @staticmethod
    def refresh_token(token: str) -> tuple[str, int]:
        payload = _decode_token(token)
        user = _get_active_user(payload["user_id"])
        ttl = _token_ttl(payload)

        try:
            consumed = cache.add(
                _revocation_key(payload["jti"]),
                "1",
                timeout=ttl,
            )
        except Exception as exc:
            logger.exception("原子消费 JWT 失败")
            raise TokenRevocationUnavailable() from exc
        if not consumed:
            raise AuthenticationFailed("Token已被撤销或已使用")

        return JWTTokenGenerator.generate_token(user)

    @staticmethod
    def blacklist_token(token: str) -> None:
        payload = _decode_token(token)
        ttl = _token_ttl(payload)
        try:
            cache.set(_revocation_key(payload["jti"]), "1", timeout=ttl)
        except Exception as exc:
            logger.exception("撤销 JWT 失败")
            raise TokenRevocationUnavailable() from exc
