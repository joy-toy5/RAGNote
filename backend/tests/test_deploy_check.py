"""部署检查只接触临时目录、合成Compose结果与HTTP替身。"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from urllib.error import HTTPError, URLError

import pytest
import yaml


@pytest.fixture
def checker(backend_root):
    path = backend_root.parent / "deploy" / "check.py"
    spec = importlib.util.spec_from_file_location("deployment_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    services = {name: {} for name in (
        "mysql", "redis", "backend", "userservice", "celery", "nginx", "migrate", "usermigrate",
    )}
    for name in ("mysql", "redis", "backend", "userservice"):
        services[name]["healthcheck"] = {"test": ["CMD", "true"]}
    services["mysql"]["environment"] = {"MYSQL_ROOT_PASSWORD": "private-root-secret"}
    database = {"MYSQL_USER": "demo", "MYSQL_PASSWORD": "private-db-secret", "MYSQL_DATABASE": "demo_test"}
    services["backend"]["environment"] = dict(
        database, SECRET_KEY="private-shared-jwt", RERANKER_MODEL_PATH="/models/reranker",
    )
    services["userservice"]["environment"] = dict(
        DB_USER="demo", DB_PASSWORD="private-db-secret", DB_NAME="user_service_test",
        DJANGO_SECRET_KEY="private-django-secret", JWT_SECRET_KEY="private-shared-jwt",
    )
    services["backend"]["volumes"] = [
        {"type": "bind", "source": str(model), "target": "/models/reranker", "read_only": True},
    ]
    services["nginx"]["ports"] = [{"target": 80, "published": "8080", "protocol": "tcp"}]
    return {"services": services}


@pytest.fixture
def rows():
    running = [
        {"Service": name, "State": "running", "Health": "healthy" if name in (
            "mysql", "redis", "backend", "userservice",
        ) else "", "ExitCode": 0}
        for name in ("mysql", "redis", "backend", "userservice", "celery", "nginx")
    ]
    return running + [
        {"Service": name, "State": "exited", "Health": "", "ExitCode": 0}
        for name in ("migrate", "usermigrate")
    ]


class Response:
    def __init__(self, body=b"<html>demo</html>", content_type="text/html", status=200):
        self.status = status
        self.headers = {"Content-Type": content_type}
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size=-1):
        return self.body if size < 0 else self.body[:size]


@pytest.fixture
def harness(checker, config, rows, tmp_path, monkeypatch):
    compose_file = tmp_path / "compose.yml"
    compose_file.write_text("services: {}\n")
    calls, requests = [], []
    responses = [Response(), Response(b'{"data":{"status":"ok"}}', "application/json")]
    outputs = {"config": json.dumps(config), "ps": json.dumps(rows)}

    def run(command, **kwargs):
        assert command[:4] == ["docker", "compose", "-f", compose_file.name]
        assert kwargs["cwd"] == compose_file.parent
        assert kwargs["capture_output"] is True
        assert kwargs["timeout"] > 0
        operation = command[4:]
        assert operation in (["config", "--format", "json"], ["ps", "--all", "--format", "json"])
        calls.append(operation)
        return subprocess.CompletedProcess(command, 0, outputs[operation[0]], "private-tool-stderr")

    def urlopen(url, **kwargs):
        assert kwargs["timeout"] > 0
        requests.append(url)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(checker.subprocess, "run", run)
    monkeypatch.setattr(checker, "urlopen", urlopen)
    return {
        "args": ["--compose-file", str(compose_file)], "outputs": outputs,
        "calls": calls, "requests": requests, "responses": responses,
    }


def test_preflight_is_read_only_and_does_not_check_http(checker, harness, capsys):
    assert checker.main(harness["args"] + ["--preflight"]) == 0
    assert harness["calls"] == [["config", "--format", "json"]]
    assert harness["requests"] == []
    output = capsys.readouterr().out
    assert "预检" in output
    assert "private-" not in output


@pytest.mark.parametrize("problem", ["variable", "jwt", "model", "readonly", "service"])
def test_preflight_rejects_missing_prerequisites(checker, harness, config, problem, capsys):
    services = config["services"]
    if problem == "variable":
        services["mysql"]["environment"]["MYSQL_ROOT_PASSWORD"] = " "
    elif problem == "jwt":
        services["backend"]["environment"]["SECRET_KEY"] = "private-other-key"
    elif problem == "model":
        services["backend"]["volumes"][0]["source"] += "/absent"
    elif problem == "readonly":
        services["backend"]["volumes"][0]["read_only"] = False
    else:
        services.pop("migrate")
    harness["outputs"]["config"] = json.dumps(config)
    assert checker.main(harness["args"]) == 1
    assert len(harness["calls"]) == 1
    assert not harness["requests"]
    output = capsys.readouterr().out
    assert "[失败]" in output and "private-" not in output


def test_live_check_accepts_six_services_and_two_completed_migrations(checker, harness, capsys):
    assert checker.main(harness["args"]) == 0
    assert harness["requests"] == ["http://127.0.0.1:8080/", "http://127.0.0.1:8080/health/ready"]
    assert "基础部署验收通过" in capsys.readouterr().out


@pytest.mark.parametrize("problem", ["missing", "duplicate", "exited", "unhealthy", "starting", "migration_failed", "migration_running", "exit_unknown"])
def test_bad_container_state_prevents_http_on_other_projects(checker, harness, rows, problem):
    if problem == "missing":
        rows.pop(0)
    elif problem == "duplicate":
        rows.append(dict(rows[0]))
    elif problem == "exited":
        rows[0]["State"] = "exited"
    elif problem in ("unhealthy", "starting"):
        rows[0]["Health"] = problem
    elif problem == "migration_failed":
        rows[-1]["ExitCode"] = 1
    elif problem == "migration_running":
        rows[-1]["State"] = "running"
    else:
        rows[-1].pop("ExitCode")
    harness["outputs"]["ps"] = json.dumps(rows)
    assert checker.main(harness["args"]) == 1
    assert not harness["requests"]


@pytest.mark.parametrize("format_", ["array", "lines", "single", "empty"])
def test_compose_ps_json_formats(checker, rows, format_):
    raw = {
        "array": json.dumps(rows), "lines": "\n".join(map(json.dumps, rows)),
        "single": json.dumps(rows[0]), "empty": "",
    }[format_]
    expected = rows if format_ in ("array", "lines") else [rows[0]] if format_ == "single" else []
    assert checker.parse_ps(raw) == expected


@pytest.mark.parametrize("error", ["missing_cli", "timeout", "nonzero", "json", "shape"])
def test_tool_failures_are_bounded_and_redacted(checker, harness, monkeypatch, error, capsys):
    if error in ("json", "shape"):
        harness["outputs"]["config"] = "private-invalid-json" if error == "json" else '{"services":[]}'
    else:
        def run(command, **kwargs):
            if error == "missing_cli":
                raise FileNotFoundError("private-cli-detail")
            if error == "timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"], output="private-output")
            return subprocess.CompletedProcess(command, 1, "private-output", "private-stderr")
        monkeypatch.setattr(checker.subprocess, "run", run)
    assert checker.main(harness["args"]) == 2
    assert "private-" not in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["html", "bad_json", "not_ready", "http_error", "network"])
def test_readiness_never_accepts_spa_or_leaks_response(checker, harness, failure, capsys):
    responses = {
        "html": Response(b"private-html", "text/html"),
        "bad_json": Response(b"private-invalid-json", "application/json"),
        "not_ready": Response(b'{"data":{"status":"private-not-ready"}}', "application/json"),
        "http_error": HTTPError("http://local/", 503, "private-detail", {}, None),
        "network": URLError("private-error"),
    }
    harness["responses"][1] = responses[failure]
    assert checker.main(harness["args"]) == 1
    assert "private-" not in capsys.readouterr().out


def test_url_uses_actual_published_port_or_explicit_override(checker, harness, config):
    config["services"]["nginx"]["ports"][0]["published"] = "18080"
    harness["outputs"]["config"] = json.dumps(config)
    assert checker.main(harness["args"]) == 0
    assert harness["requests"][0] == "http://127.0.0.1:18080/"
    harness["requests"].clear()
    harness["responses"][:] = [Response(), Response(b'{"data":{"status":"ok"}}', "application/json")]
    assert checker.main(harness["args"] + ["--url", "http://127.0.0.1:28080"]) == 0
    assert harness["requests"][0] == "http://127.0.0.1:28080/"


@pytest.mark.parametrize("arguments", [
    ["--timeout", "0"], ["--timeout", "nan"], ["--timeout", "inf"],
    ["--url", "file:///private"], ["--url", "http://user:private-password@localhost"],
])
def test_invalid_arguments_are_rejected_without_echoing_credentials(checker, arguments, capsys):
    with pytest.raises(SystemExit) as exc:
        checker.main(arguments)
    assert exc.value.code == 2
    assert "private-password" not in capsys.readouterr().err


def test_docker_reuses_runner_and_mysql_rejects_missing_databases(backend_root):
    root = backend_root.parent
    compose = yaml.safe_load((root / "docker-compose.yml").read_text())
    command = compose["services"]["mysql"]["healthcheck"]["test"]
    assert command[:2] == ["CMD", "mysql"]
    assert "--protocol=TCP" in command and "-h127.0.0.1" in command
    assert command[-1] == "USE chat_history; USE user_service; SELECT 1"
    assert compose["services"]["backend"]["stop_grace_period"] == "15s"
    assert compose["services"]["migrate"]["command"] == ["alembic", "upgrade", "head"]
    cmd = next(line for line in (backend_root / "Dockerfile").read_text().splitlines() if line.startswith("CMD "))
    assert json.loads(cmd[4:]) == ["python", "run.py"]


def test_required_variables_follow_existing_service_templates(checker, backend_root):
    root = backend_root.parent
    for service, template in (("backend", "backend.env.example"), ("userservice", "userservice.env.example")):
        names = {line.split("=", 1)[0].strip() for line in (root / "deploy" / template).read_text().splitlines()
                 if "=" in line and not line.lstrip().startswith("#")}
        assert set(checker.REQUIRED_ENV[service]) <= names
