import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time
from types import SimpleNamespace
from unittest.mock import patch

import jwt
from django.apps import apps
from django.conf import settings
from django.core.cache import cache
from django.db.migrations.autodetector import MigrationAutodetector
from django.db.migrations.loader import MigrationLoader
from django.db.migrations.questioner import MigrationQuestioner
from django.db.migrations.state import ProjectState
from django.test import SimpleTestCase, override_settings
from django.utils.translation import override
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.permissions import AllowAny
from rest_framework.test import APIRequestFactory

from .authentications import JWTTokenGenerator, TokenRevocationUnavailable
from .models import User, UserStatusChoice
from .serializers import LoginSerializer
from .views import LoginView, RegisterView, TokenRefreshView

TEST_CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "m0-jwt-refresh-tests",
        "KEY_FUNCTION": "DjangoUserService.settings.shared_cache_key",
    }
}


@override_settings(CACHES=TEST_CACHES)
class JWTRefreshTests(SimpleTestCase):
    """验证 JWT 轮换、撤销与账户状态边界。"""

    def setUp(self) -> None:
        cache.clear()
        self.user = SimpleNamespace(
            uuid="eval-user-a",
            username="eval-a",
            email="eval-a@example.invalid",
            status=UserStatusChoice.ACTIVE,
        )

    def make_token(
        self,
        *,
        expires_at: int,
        jti: str = "m1-refresh-token",
        missing_claim: str | None = None,
        signing_key: str | None = None,
    ) -> str:
        payload = {
            "user_id": self.user.uuid,
            "username": self.user.username,
            "email": self.user.email,
            "iat": int(time()) - 60,
            "exp": expires_at,
            "jti": jti,
        }
        if missing_claim:
            payload.pop(missing_claim)
        return jwt.encode(
            payload,
            signing_key or settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )

    def test_expired_token_cannot_be_refreshed(self) -> None:
        token = self.make_token(expires_at=int(time()) - 1)

        with patch(
            "apps.user.authentications.User.objects.get", return_value=self.user
        ):
            with self.assertRaises(AuthenticationFailed):
                JWTTokenGenerator.refresh_token(token)

    def test_revoked_token_cannot_be_refreshed(self) -> None:
        token = self.make_token(expires_at=int(time()) + 3600)
        cache.set("jwt:revoked:m1-refresh-token", "1", 3600)

        with patch(
            "apps.user.authentications.User.objects.get", return_value=self.user
        ):
            with self.assertRaises(AuthenticationFailed):
                JWTTokenGenerator.refresh_token(token)

    def test_non_active_user_cannot_refresh_token(self) -> None:
        token = self.make_token(expires_at=int(time()) + 3600)

        for user_status in (UserStatusChoice.DISABLED, UserStatusChoice.LOCKED):
            with self.subTest(user_status=user_status):
                inactive_user = SimpleNamespace(**vars(self.user))
                inactive_user.status = user_status
                with patch(
                    "apps.user.authentications.User.objects.get",
                    return_value=inactive_user,
                ):
                    with self.assertRaises(AuthenticationFailed):
                        JWTTokenGenerator.refresh_token(token)

    def test_refresh_token_is_single_use(self) -> None:
        token = self.make_token(expires_at=int(time()) + 3600)

        with patch(
            "apps.user.authentications.User.objects.get", return_value=self.user
        ):
            first = self.refresh_request(token)
            second = self.refresh_request(token)

        self.assertEqual(first, 200)
        self.assertEqual(second, 401)

    def test_concurrent_refresh_allows_only_one_success(self) -> None:
        token = self.make_token(expires_at=int(time()) + 3600)

        with patch(
            "apps.user.authentications.User.objects.get", return_value=self.user
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                statuses = list(
                    executor.map(lambda _: self.refresh_request(token), range(2))
                )

        self.assertEqual(statuses.count(200), 1)
        self.assertEqual(statuses.count(401), 1)

    def test_required_claims_cannot_be_omitted(self) -> None:
        for claim in ("exp", "iat", "jti", "user_id"):
            with self.subTest(claim=claim):
                token = self.make_token(
                    expires_at=int(time()) + 3600,
                    missing_claim=claim,
                )
                with patch(
                    "apps.user.authentications.User.objects.get",
                    return_value=self.user,
                ):
                    with self.assertRaises(AuthenticationFailed):
                        JWTTokenGenerator.refresh_token(token)

    def test_token_with_forged_signature_cannot_be_refreshed(self) -> None:
        token = self.make_token(
            expires_at=int(time()) + 3600,
            signing_key="forged-signing-key-with-no-production-authority",
        )

        with self.assertRaises(AuthenticationFailed):
            JWTTokenGenerator.refresh_token(token)

    def test_cache_failure_closes_refresh_with_service_unavailable(self) -> None:
        token = self.make_token(expires_at=int(time()) + 3600)

        with patch(
            "apps.user.authentications.User.objects.get", return_value=self.user
        ), patch(
            "apps.user.authentications.cache.add",
            side_effect=RuntimeError("cache unavailable"),
        ):
            with self.assertLogs("apps.user.authentications", level="ERROR"):
                self.assertEqual(self.refresh_request(token), 503)

    def test_token_is_consumed_before_replacement_is_issued(self) -> None:
        token = self.make_token(expires_at=int(time()) + 3600)
        events = []

        def consume(*args, **kwargs):
            events.append(("consume", args, kwargs))
            return True

        def issue(user):
            events.append(("issue", user))
            return "replacement", int(time()) + 3600

        with patch(
            "apps.user.authentications.User.objects.get", return_value=self.user
        ), patch(
            "apps.user.authentications.cache.add", side_effect=consume
        ), patch.object(JWTTokenGenerator, "generate_token", side_effect=issue):
            JWTTokenGenerator.refresh_token(token)

        self.assertEqual([event[0] for event in events], ["consume", "issue"])
        self.assertEqual(events[0][1][0], "jwt:revoked:m1-refresh-token")
        self.assertGreater(events[0][2]["timeout"], 0)

    def test_blacklist_uses_the_shared_exact_key(self) -> None:
        token = self.make_token(expires_at=int(time()) + 3600)

        JWTTokenGenerator.blacklist_token(token)

        self.assertEqual(cache.get("jwt:revoked:m1-refresh-token"), "1")

    def test_revocation_key_is_not_rewritten_by_django_cache(self) -> None:
        self.assertEqual(
            cache.make_key("jwt:revoked:contract-id"),
            "jwt:revoked:contract-id",
        )
        self.assertEqual(cache.make_key("user:contract-id"), ":1:user:contract-id")

    def test_cache_failure_exception_is_not_an_authentication_success(self) -> None:
        token = self.make_token(expires_at=int(time()) + 3600)

        with patch(
            "apps.user.authentications.cache.set",
            side_effect=RuntimeError("cache unavailable"),
        ):
            with self.assertLogs("apps.user.authentications", level="ERROR"):
                with self.assertRaises(TokenRevocationUnavailable):
                    JWTTokenGenerator.blacklist_token(token)

    def test_public_identity_endpoints_explicitly_allow_anonymous_access(self) -> None:
        for view in (LoginView, RegisterView, TokenRefreshView):
            with self.subTest(view=view.__name__):
                self.assertEqual(view.permission_classes, [AllowAny])
                self.assertEqual(view.authentication_classes, [])

    def test_legacy_test_credentials_do_not_bypass_account_lookup(self) -> None:
        user_query = SimpleNamespace(exists=lambda: False)
        serializer = LoginSerializer(
            data={"username": "test", "password": "666666"}
        )

        with patch(
            "apps.user.serializers.User.objects.filter",
            return_value=user_query,
        ):
            self.assertFalse(serializer.is_valid())

    @staticmethod
    def refresh_request(token: str) -> int:
        request = APIRequestFactory().post("/user/refresh-token/", {"token": token})
        response = TokenRefreshView.as_view()(request)
        return response.status_code


class ProductionSettingsTests(SimpleTestCase):
    """验证生产设置安全失败，本测试只导入设置，不连接外部服务。"""

    REQUIRED_ENV = {
        "DJANGO_SECRET_KEY": "m1-D9!django-production-secret-with-more-than-fifty-characters-48",
        "JWT_SECRET_KEY": "m1-K7!production-check-secret-with-more-than-fifty-characters-29",
        "DJANGO_ALLOWED_HOSTS": "rag-note.example.invalid",
        "DJANGO_CORS_ALLOWED_ORIGINS": "https://rag-note.example.invalid",
        "DJANGO_ANON_RATE": "30/minute",
        "DJANGO_USER_RATE": "300/minute",
        "REDIS_CACHE_URL": "redis://127.0.0.1:6379/15",
    }

    def production_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.REQUIRED_ENV)
        env["DJANGO_ENV"] = "production"
        env["DJANGO_LOAD_DOTENV"] = "false"
        return env

    def run_settings_import(
        self, env: dict[str, str], expression: str = "print('ok')"
    ) -> subprocess.CompletedProcess[str]:
        code = f"import DjangoUserService.settings as settings; {expression}"
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

    def test_production_rejects_each_missing_required_setting(self) -> None:
        for variable in self.REQUIRED_ENV:
            with self.subTest(variable=variable):
                env = self.production_env()
                env.pop(variable, None)
                result = self.run_settings_import(env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(variable, result.stderr)

    def test_production_enables_security_and_default_permissions(self) -> None:
        fields = (
            "DEBUG",
            "CORS_ALLOW_ALL_ORIGINS",
            "SECURE_SSL_REDIRECT",
            "SESSION_COOKIE_SECURE",
            "CSRF_COOKIE_SECURE",
            "SECURE_HSTS_SECONDS",
            "REST_FRAMEWORK",
        )
        expression = (
            "import json; "
            f"print(json.dumps({{{', '.join(f'{name!r}: settings.{name}' for name in fields)}}}))"
        )

        result = self.run_settings_import(self.production_env(), expression)

        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(result.stdout.strip())
        self.assertFalse(config["DEBUG"])
        self.assertFalse(config["CORS_ALLOW_ALL_ORIGINS"])
        self.assertTrue(config["SECURE_SSL_REDIRECT"])
        self.assertTrue(config["SESSION_COOKIE_SECURE"])
        self.assertTrue(config["CSRF_COOKIE_SECURE"])
        self.assertGreater(config["SECURE_HSTS_SECONDS"], 0)
        self.assertEqual(
            config["REST_FRAMEWORK"]["DEFAULT_PERMISSION_CLASSES"],
            ["rest_framework.permissions.IsAuthenticated"],
        )
        self.assertTrue(config["REST_FRAMEWORK"]["DEFAULT_THROTTLE_RATES"])

    def test_production_rejects_shared_django_and_jwt_secret(self) -> None:
        env = self.production_env()
        env["DJANGO_SECRET_KEY"] = env["JWT_SECRET_KEY"]

        result = self.run_settings_import(env)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("必须不同", result.stderr)

    def test_development_does_not_force_https(self) -> None:
        env = os.environ.copy()
        env["DJANGO_ENV"] = "development"
        env["DJANGO_LOAD_DOTENV"] = "false"
        env["DJANGO_SECRET_KEY"] = "m1-development-django-test-secret"
        env["JWT_SECRET_KEY"] = "m1-development-test-secret"
        expression = (
            "import json; print(json.dumps({"
            "'debug': settings.DEBUG, "
            "'ssl': settings.SECURE_SSL_REDIRECT, "
            "'session': settings.SESSION_COOKIE_SECURE, "
            "'csrf': settings.CSRF_COOKIE_SECURE, "
            "'hsts': settings.SECURE_HSTS_SECONDS}))"
        )

        result = self.run_settings_import(env, expression)

        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(result.stdout.strip())
        self.assertTrue(config["debug"])
        self.assertFalse(config["ssl"])
        self.assertFalse(config["session"])
        self.assertFalse(config["csrf"])
        self.assertEqual(config["hsts"], 0)


class UserMigrationBaselineTests(SimpleTestCase):
    """验证用户应用迁移基线可加载且与当前 ORM 结构一致。"""

    MIGRATION_KEY = ("user", "0001_initial")
    EXPECTED_FIELDS = (
        "password",
        "uuid",
        "username",
        "email",
        "telephone",
        "is_active",
        "status",
        "gender",
        "bio",
        "date_joined",
        "last_login",
        "avatar",
    )

    def setUp(self) -> None:
        self.loader = MigrationLoader(None, ignore_no_migrations=True)
        self.migration = self.loader.disk_migrations[self.MIGRATION_KEY]
        state = self.migration.mutate_state(ProjectState())
        self.model_state = state.models[("user", "user")]

    def test_initial_migration_file_is_loadable(self) -> None:
        migration_path = Path(__file__).with_name("migrations") / "0001_initial.py"

        self.assertTrue(migration_path.is_file())
        self.assertIn(self.MIGRATION_KEY, self.loader.disk_migrations)
        self.assertTrue(self.migration.initial)

    def test_initial_migration_is_the_user_graph_root(self) -> None:
        node = self.loader.graph.node_map[self.MIGRATION_KEY]

        self.assertEqual(self.migration.dependencies, [])
        self.assertEqual(set(node.parents), set())
        self.assertEqual(
            self.loader.graph.root_nodes("user"),
            [self.MIGRATION_KEY],
        )
        self.assertEqual(
            self.loader.graph.leaf_nodes("user"),
            [self.MIGRATION_KEY],
        )

    def test_migration_user_state_matches_current_orm_contract(self) -> None:
        state_fields = self.model_state.fields
        orm_fields = {field.name: field for field in User._meta.local_fields}

        self.assertEqual(self.model_state.options["db_table"], "user_service")
        self.assertEqual(self.model_state.options["db_table"], User._meta.db_table)
        self.assertEqual(tuple(state_fields), self.EXPECTED_FIELDS)
        self.assertEqual(tuple(orm_fields), self.EXPECTED_FIELDS)

        for field_name in self.EXPECTED_FIELDS:
            with self.subTest(field=field_name):
                self.assertEqual(
                    self.field_contract(state_fields[field_name]),
                    self.field_contract(orm_fields[field_name]),
                )

    def test_migration_state_has_no_orm_schema_drift(self) -> None:
        # Django 的迁移命令会禁用翻译，避免 verbose_name 产生 locale 假阳性。
        with override(None):
            changes = MigrationAutodetector(
                self.loader.project_state(),
                ProjectState.from_apps(apps),
                questioner=MigrationQuestioner(defaults={"ask_initial": False}),
            ).changes(graph=self.loader.graph, trim_to_apps={"user"})

        self.assertEqual(changes, {})

    @staticmethod
    def field_contract(field) -> tuple[str, list, dict]:
        _, path, args, kwargs = field.deconstruct()
        # verbose_name 会随激活的 Django locale 翻译，不属于数据库结构契约。
        kwargs.pop("verbose_name", None)
        return path, args, kwargs
