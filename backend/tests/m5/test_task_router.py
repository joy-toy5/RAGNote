"""任务 HTTP 契约：真实路由/repository 查询与 fake AsyncSession，不启动服务。"""

from __future__ import annotations

import ast
import asyncio
import importlib
import importlib.util
import inspect
import sys
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace
from typing import get_type_hints
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, Header, HTTPException
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession


USER_ID = "user-a"
TASK_ID = "11111111-1111-4111-8111-111111111111"
RETRY_ID = "22222222-2222-4222-8222-222222222222"
NOW = datetime(2026, 9, 5, 8, 0, tzinfo=timezone.utc)
PUBLIC_FIELDS = {
    "task_id", "kind", "status", "resource_id", "target_generation", "progress",
    "created_at", "updated_at", "started_at", "completed_at", "next_run_at",
    "cancel_requested_at", "error_code", "error_summary", "retry_of_task_id",
}
OPERATIONS = [
    ("GET", "/tasks", "list_tasks"),
    ("GET", f"/tasks/{TASK_ID}", "get_task"),
    ("POST", f"/tasks/{TASK_ID}/cancel", "request_cancel"),
    ("POST", f"/tasks/{TASK_ID}/retry", "retry_task"),
]


def _task(**overrides):
    values = dict(
        task_id=TASK_ID, user_id=USER_ID, kind="note.index", status="processing",
        resource_id="33333333-3333-4333-8333-333333333333", target_generation=2,
        phase="internal-phase", progress=35, created_at=NOW, updated_at=NOW,
        started_at=NOW, completed_at=None, next_run_at=None,
        cancel_requested_at=None, error_code="INDEX_TIMEOUT",
        error_summary="索引执行超时，请稍后重试", retry_of_task_id=None,
        input_ref="private://snapshots/source", input_metadata={"secret": "private-metadata"},
        input_blob_id="b" * 64, input_fingerprint="a" * 64,
        input_schema_version=1, idempotency_key="private-key", lease_owner="private-worker",
        lease_token=42, lease_until=NOW, heartbeat_at=NOW, result_ref="private-result",
        result_version=2, attempt_count=1, max_attempts=3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def api(monkeypatch, backend_root):
    db = AsyncMock(spec=AsyncSession)
    task = _task()
    db.scalar.return_value = task
    db.scalars.return_value = [task]

    async def identity(x_test_user: str | None = Header(None)):
        if x_test_user is None:
            raise HTTPException(status_code=401, detail="未登录")
        return x_test_user

    async def get_db():
        # 故意不在 yield 后提交，确保写入成功完全依赖路由的显式 commit。
        try:
            yield db
        except Exception:
            await db.rollback()
            raise

    for name, attributes in {
        "app.db.db_config": {"get_db": get_db},
        "app.utils.auth_utils": {"get_current_user_id": identity},
    }.items():
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)

    # 只加载纯 ORM 元数据，不让路由导入数据库引擎、.env 或身份服务。
    importlib.import_module("app.db.schema_gate")
    spec = importlib.util.spec_from_file_location(
        "_m5_task_router", backend_root / "app/router/task_router.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.task_router)
    return SimpleNamespace(app=app, module=module, db=db, task=task)


def _request(api, method="GET", path="/tasks", *, authenticated=True, **kwargs):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test",
        ) as client:
            headers = {"X-Test-User": USER_ID} if authenticated else {}
            return await client.request(method, path, headers=headers, **kwargs)

    return asyncio.run(scenario())


def _payload(path):
    return {"json": {"idempotency_key": "retry-client-key"}} if path.endswith("/retry") else {}


def _assert_public(data):
    assert set(data) == PUBLIC_FIELDS
    assert data["kind"] == "note.index"
    assert data["resource_id"] == "33333333-3333-4333-8333-333333333333"
    assert data["target_generation"] == 2
    assert datetime.fromisoformat(data["created_at"]) == NOW
    assert datetime.fromisoformat(data["updated_at"]) == NOW
    assert data["error_code"] == "INDEX_TIMEOUT"
    assert data["error_summary"] == "索引执行超时，请稍后重试"


def _assert_scope(statement, task_id=None):
    where = str(statement.whereclause)
    params = statement.compile().params
    assert "background_tasks.user_id = :user_id_1" in where
    assert params["user_id_1"] == USER_ID
    if task_id is not None:
        assert "background_tasks.task_id = :task_id_1" in where
        assert params["task_id_1"] == task_id


def test_router_exposes_only_four_authenticated_endpoints(api):
    routes = api.module.task_router.routes
    assert {(route.path, method) for route in routes for method in route.methods} == {
        ("/tasks", "GET"),
        ("/tasks/{task_id}", "GET"), ("/tasks/{task_id}/cancel", "POST"),
        ("/tasks/{task_id}/retry", "POST"),
    }
    for route in routes:
        parameters = inspect.signature(route.endpoint).parameters
        assert parameters["user_id"].default.dependency is api.module.get_current_user_id
        assert parameters["db"].default.dependency is api.module.get_db
        assert get_type_hints(route.endpoint)["db"] is AsyncSession


def test_main_imports_and_registers_task_router_once(backend_root):
    tree = ast.parse((backend_root / "main.py").read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert sum(
        node.module == "app.router.task_router"
        and [(alias.name, alias.asname) for alias in node.names] == [("task_router", None)]
        for node in imports
    ) == 1
    included = [
        ast.unparse(node.value.args[0]) for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "app.include_router"
    ]
    assert included.count("task_router") == 1
    assert {"chat_router", "knowledge_router", "health_router", "user_router",
            "note_router", "review_router"} <= set(included)


@pytest.mark.parametrize("limit,offset,key", [(20, 0, None), (1, 2, "recover-me"), (100, 7, None)])
def test_list_is_scoped_paginated_and_filterable(api, limit, offset, key):
    params = {"limit": limit, "offset": offset, "user_id": "user-b"}
    if key is not None:
        params["idempotency_key"] = key
    response = _request(api, params=params)
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200 and body["message"] == "success"
    assert set(body["data"]) == {"tasks", "limit", "offset"}
    assert body["data"]["limit"] == limit and body["data"]["offset"] == offset
    assert len(body["data"]["tasks"]) == 1
    _assert_public(body["data"]["tasks"][0])
    statement = api.db.scalars.await_args.args[0]
    _assert_scope(statement)
    sql = " ".join(str(statement.compile(compile_kwargs={"literal_binds": True})).split())
    assert f"LIMIT {limit} OFFSET {offset}" in sql
    if key is not None:
        assert statement.compile().params["idempotency_key_1"] == key
    api.db.commit.assert_not_awaited()


def test_empty_list_uses_default_page_without_claiming_total(api):
    api.db.scalars.return_value = []
    response = _request(api)
    assert response.json()["data"] == {"tasks": [], "limit": 20, "offset": 0}


def test_detail_uses_user_scope_and_public_fields(api):
    response = _request(api, path=f"/tasks/{TASK_ID}?user_id=user-b")
    assert response.status_code == 200
    _assert_public(response.json()["data"])
    _assert_scope(api.db.scalar.await_args.args[0], TASK_ID)
    api.db.commit.assert_not_awaited()


@pytest.mark.parametrize("method,suffix", [("GET", ""), ("POST", "/cancel"), ("POST", "/retry")])
def test_foreign_and_missing_tasks_have_identical_404(api, method, suffix):
    api.db.scalar.return_value = None
    responses = []
    for task_id in ("foreign-task", "absent-task"):
        path = f"/tasks/{task_id}{suffix}"
        response = _request(api, method, path, **_payload(path))
        assert response.status_code == 404
        _assert_scope(api.db.scalar.await_args.args[0], task_id)
        responses.append(response.json())
    assert responses[0] == responses[1] == {"detail": "任务不存在"}
    api.db.commit.assert_not_awaited()


@pytest.mark.parametrize("method,path,operation", OPERATIONS)
def test_authentication_cannot_be_bypassed_with_request_user_id(api, method, path, operation):
    response = _request(
        api, method, path, authenticated=False, params={"user_id": USER_ID}, **_payload(path),
    )
    assert response.status_code == 401
    api.db.scalar.assert_not_awaited()
    api.db.scalars.assert_not_awaited()
    api.db.commit.assert_not_awaited()


@pytest.mark.parametrize("method,path,operation", OPERATIONS)
@pytest.mark.parametrize("error_name,status", [
    ("TaskNotFound", 404), ("TaskIdempotencyConflict", 409), ("TaskStateConflict", 409),
])
def test_domain_error_mapping_never_echoes_internal_details(
    api, monkeypatch, method, path, operation, error_name, status,
):
    errors = importlib.import_module("app.tasking.errors")
    failure = getattr(errors, error_name)("private://source user-b SQL secret")
    monkeypatch.setattr(api.module.repository, operation, AsyncMock(side_effect=failure))
    response = _request(api, method, path, **_payload(path))
    assert response.status_code == status
    assert "private" not in response.text and "user-b" not in response.text
    assert "SQL" not in response.text
    api.db.commit.assert_not_awaited()
    api.db.rollback.assert_awaited_once()


def _database_failure():
    return OperationalError(
        "SELECT input_ref FROM background_tasks", {"input_ref": "private://source"},
        RuntimeError("private database connection"),
    )


@pytest.mark.parametrize("method,path,operation", OPERATIONS)
def test_repository_database_errors_become_sanitized_503(api, monkeypatch, method, path, operation):
    monkeypatch.setattr(api.module.repository, operation, AsyncMock(side_effect=_database_failure()))
    response = _request(api, method, path, **_payload(path))
    assert response.status_code == 503
    assert response.json() == {"detail": "任务存储暂不可用，请稍后重试"}
    api.db.commit.assert_not_awaited()


@pytest.mark.parametrize("operation,task_id,status", [
    ("cancel", TASK_ID, "processing"), ("retry", RETRY_ID, "pending"),
])
def test_writes_reread_scoped_snapshot_before_commit_and_return_dto(
    api, monkeypatch, operation, task_id, status,
):
    class ExpiringTask(SimpleNamespace):
        def __getattribute__(self, name):
            if name in PUBLIC_FIELDS and object.__getattribute__(self, "expired"):
                raise AssertionError("提交后不得读取 ORM 属性")
            return super().__getattribute__(name)

    retry_of = TASK_ID if operation == "retry" else None
    task = ExpiringTask(**vars(_task(
        task_id=task_id, status=status, retry_of_task_id=retry_of, cancel_requested_at=NOW,
    )), expired=False)
    api.db.scalar.return_value = task
    # 写入返回体故意只有主键：完整响应必须来自带用户作用域的重读。
    mutation = AsyncMock(return_value=SimpleNamespace(task_id=task_id))
    name = "request_cancel" if operation == "cancel" else "retry_task"
    monkeypatch.setattr(api.module.repository, name, mutation)

    async def commit():
        task.expired = True

    build_response = api.module.success_response

    def response_before_commit(**kwargs):
        assert not task.expired, "完整响应必须在提交前编码，失败时仍可回滚"
        return build_response(**kwargs)

    monkeypatch.setattr(api.module, "success_response", response_before_commit)
    api.db.commit.side_effect = commit
    path = f"/tasks/{TASK_ID}/{operation}"
    response = _request(api, "POST", path, **_payload(path))
    assert response.status_code == 200
    _assert_public(response.json()["data"])
    assert response.json()["data"]["status"] == status
    assert response.json()["data"]["retry_of_task_id"] == retry_of
    assert response.json()["data"]["task_id"] == task_id
    _assert_scope(api.db.scalar.await_args.args[0], task_id)
    kwargs = {"idempotency_key": "retry-client-key"} if operation == "retry" else {}
    mutation.assert_awaited_once_with(api.db, TASK_ID, USER_ID, **kwargs)
    api.db.commit.assert_awaited_once()
    api.db.rollback.assert_not_awaited()


@pytest.mark.parametrize("operation", ["cancel", "retry"])
@pytest.mark.parametrize("failure_stage", ["reread", "commit"])
def test_failed_write_never_returns_success(api, monkeypatch, operation, failure_stage):
    name = "request_cancel" if operation == "cancel" else "retry_task"
    monkeypatch.setattr(api.module.repository, name, AsyncMock(return_value=api.task))
    target = api.db.scalar if failure_stage == "reread" else api.db.commit
    target.side_effect = _database_failure()
    path = f"/tasks/{TASK_ID}/{operation}"
    response = _request(api, "POST", path, **_payload(path))
    assert response.status_code == 503
    assert "private" not in response.text and "input_ref" not in response.text
    assert "SELECT" not in response.text
    api.db.rollback.assert_awaited_once()
    assert api.db.commit.await_count == (failure_stage == "commit")


@pytest.mark.parametrize("key", [None, "", " ", "a b", "\ta", "a\n", "a\r", "a\x00", "a\x7f", "重试", "a" * 129, 1, True, []])
def test_retry_rejects_invalid_idempotency_keys(api, key):
    response = _request(api, "POST", f"/tasks/{TASK_ID}/retry", json={"idempotency_key": key})
    assert response.status_code == 422
    api.db.scalar.assert_not_awaited()
    api.db.commit.assert_not_awaited()


@pytest.mark.parametrize("payload", [
    None, {}, {"idempotency_key": "valid", "user_id": "user-b"},
    {"idempotency_key": "valid", "input_ref": "private://injected"},
])
def test_retry_requires_body_and_forbids_extra_fields(api, payload):
    response = _request(api, "POST", f"/tasks/{TASK_ID}/retry", json=payload)
    assert response.status_code == 422
    api.db.scalar.assert_not_awaited()
    api.db.commit.assert_not_awaited()


@pytest.mark.parametrize("key", ["!", "~" * 128, "".join(chr(code) for code in range(33, 127))])
def test_retry_accepts_visible_ascii_and_preserves_client_key(api, monkeypatch, key):
    mutation = AsyncMock(return_value=api.task)
    monkeypatch.setattr(api.module.repository, "retry_task", mutation)
    response = _request(api, "POST", f"/tasks/{TASK_ID}/retry", json={"idempotency_key": key})
    assert response.status_code == 200
    mutation.assert_awaited_once_with(api.db, TASK_ID, USER_ID, idempotency_key=key)


@pytest.mark.parametrize("params", [
    {"limit": 0}, {"limit": -1}, {"limit": 101}, {"limit": "abc"}, {"limit": "1.5"},
    {"offset": -1}, {"offset": "abc"}, {"offset": "1.5"},
    {"idempotency_key": ""}, {"idempotency_key": "a b"}, {"idempotency_key": "a\n"},
    {"idempotency_key": "重试"}, {"idempotency_key": "a" * 129},
])
def test_list_rejects_invalid_page_and_filter_before_querying(api, params):
    response = _request(api, params=params)
    assert response.status_code == 422
    api.db.scalars.assert_not_awaited()


def test_no_arbitrary_task_creation_endpoint(api):
    response = _request(api, "POST", "/tasks", json={"kind": "note.index", "input_ref": "anything"})
    assert response.status_code == 405
    api.db.commit.assert_not_awaited()


def test_openapi_publishes_strict_retry_and_pagination_contract(api):
    spec = api.app.openapi()
    retry = spec["paths"]["/tasks/{task_id}/retry"]["post"]["requestBody"]
    assert retry["required"] is True
    schema_name = retry["content"]["application/json"]["schema"]["$ref"].rsplit("/", 1)[1]
    schema = spec["components"]["schemas"][schema_name]
    assert schema["required"] == ["idempotency_key"]
    assert schema["additionalProperties"] is False
    key_schema = schema["properties"]["idempotency_key"]
    assert key_schema["minLength"] == 1 and key_schema["maxLength"] == 128
    parameters = {item["name"]: item["schema"] for item in spec["paths"]["/tasks"]["get"]["parameters"]}
    assert parameters["limit"]["minimum"] == 1 and parameters["limit"]["maximum"] == 100
    assert parameters["limit"]["default"] == 20
    assert parameters["offset"]["minimum"] == 0 and parameters["offset"]["default"] == 0
    assert "idempotency_key" in parameters
