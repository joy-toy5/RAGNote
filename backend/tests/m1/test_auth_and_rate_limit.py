from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.core import rate_limit
from app.utils import auth_utils


class RedisProbe:
    def __init__(self, value: str | None = None) -> None:
        self.value = value
        self.keys: list[str] = []

    async def get(self, key: str) -> str | None:
        self.keys.append(key)
        return self.value


class StatusVerifier:
    def __init__(self, active: bool = True, error: Exception | None = None) -> None:
        self.active = active
        self.error = error
        self.calls: list[tuple[str, str]] = []

    async def is_active(self, user_id: str, token: str) -> bool:
        self.calls.append((user_id, token))
        if self.error:
            raise self.error
        return self.active


@pytest.mark.p0
def test_auth_uses_exact_revocation_key_and_active_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = RedisProbe()
    verifier = StatusVerifier()

    async def connect() -> RedisProbe:
        return redis

    monkeypatch.setattr(
        auth_utils,
        "decode_django_jwt",
        lambda _: {"user_id": "user-a", "jti": "token-id"},
    )
    monkeypatch.setattr(auth_utils, "connect_redis", connect)

    user_id = asyncio.run(
        auth_utils.get_current_user_id(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials="signed-token"),
            verifier,
        )
    )
    assert user_id == "user-a"
    assert redis.keys == ["jwt:revoked:token-id"]
    assert verifier.calls == [("user-a", "signed-token")]


@pytest.mark.p0
def test_auth_rejects_revoked_token_before_status_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = RedisProbe(value="1")
    verifier = StatusVerifier()

    async def connect() -> RedisProbe:
        return redis

    monkeypatch.setattr(
        auth_utils,
        "decode_django_jwt",
        lambda _: {"user_id": "user-a", "jti": "revoked-id"},
    )
    monkeypatch.setattr(auth_utils, "connect_redis", connect)

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            auth_utils.get_current_user_id(
                HTTPAuthorizationCredentials(
                    scheme="Bearer", credentials="signed-token"
                ),
                verifier,
            )
        )
    assert error.value.status_code == 401
    assert redis.keys == ["jwt:revoked:revoked-id"]
    assert verifier.calls == []


@pytest.mark.p0
@pytest.mark.parametrize(
    ("verifier", "status_code"),
    [(StatusVerifier(active=False), 401), (StatusVerifier(error=RuntimeError()), 503)],
)
def test_auth_fails_closed_when_status_is_not_confirmed(
    monkeypatch: pytest.MonkeyPatch,
    verifier: StatusVerifier,
    status_code: int,
) -> None:
    async def connect() -> RedisProbe:
        return RedisProbe()

    monkeypatch.setattr(
        auth_utils,
        "decode_django_jwt",
        lambda _: {"user_id": "user-a", "jti": "token-id"},
    )
    monkeypatch.setattr(auth_utils, "connect_redis", connect)

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            auth_utils.get_current_user_id(
                HTTPAuthorizationCredentials(
                    scheme="Bearer", credentials="signed-token"
                ),
                verifier,
            )
        )
    assert error.value.status_code == status_code


@pytest.mark.p0
def test_django_status_verifier_runs_blocking_client_off_event_loop() -> None:
    caller_thread = threading.get_ident()
    request_threads: list[int] = []

    def request_get(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        assert kwargs["timeout"] == (2.0, 3.0)
        request_threads.append(threading.get_ident())
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"success": True, "data": {"id": "user-a"}},
        )

    verifier = auth_utils.DjangoUserStatusVerifier(
        base_url="http://django.invalid", request_get=request_get
    )
    assert asyncio.run(verifier.is_active("user-a", "token")) is True
    assert request_threads and request_threads[0] != caller_thread


@pytest.mark.p0
@pytest.mark.parametrize(
    "payload",
    [
        {"success": False, "data": {"id": "user-a"}},
        {"success": True, "data": {"id": "user-b"}},
        {"success": True, "data": None},
    ],
)
def test_django_status_verifier_rejects_unconfirmed_identity(payload: dict) -> None:
    def request_get(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(status_code=200, json=lambda: payload)

    verifier = auth_utils.DjangoUserStatusVerifier(
        base_url="http://django.invalid", request_get=request_get
    )
    assert asyncio.run(verifier.is_active("user-a", "token")) is False


@pytest.mark.p0
def test_production_rejects_disabled_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    with pytest.raises(RuntimeError, match="生产环境禁止关闭限流"):
        rate_limit.validate_rate_limit_config()


@pytest.mark.p0
def test_production_rate_limit_defaults_to_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.delenv("RATE_LIMIT_ENABLED", raising=False)
    rate_limit.validate_rate_limit_config()
    assert rate_limit.is_rate_limit_enabled() is True


@pytest.mark.p0
def test_development_cors_defaults_to_local_frontends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENV", "development")
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    assert auth_utils.get_cors_origins() == [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]


@pytest.mark.p0
def test_production_auth_and_cors_config_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("SECRET_KEY", "weak")
    monkeypatch.setenv("ALGORITHM", "HS256")
    monkeypatch.setenv("DJANGO_API_URL", "https://identity.example.invalid")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")

    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        auth_utils.validate_auth_config()

    monkeypatch.setenv(
        "SECRET_KEY", "m1-fastapi-production-secret-with-more-than-fifty-characters-42"
    )
    with pytest.raises(RuntimeError, match="allowlist"):
        auth_utils.validate_auth_config()

    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS", "https://rag-note.example.invalid"
    )
    auth_utils.validate_auth_config()
