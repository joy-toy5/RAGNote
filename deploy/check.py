#!/usr/bin/env python3
"""单机 Demo 的只读部署验收；不启动服务、不执行迁移、不调用模型。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen


SERVICES = ("mysql", "redis", "backend", "userservice", "celery", "nginx")
MIGRATIONS = ("migrate", "usermigrate")
REQUIRED_ENV = {
    "mysql": ("MYSQL_ROOT_PASSWORD",),
    "backend": ("MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE", "SECRET_KEY"),
    "userservice": ("DB_USER", "DB_PASSWORD", "DB_NAME", "DJANGO_SECRET_KEY", "JWT_SECRET_KEY"),
}
DEFAULT_COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yml"


class CheckError(Exception):
    """无法执行检查；消息只包含受控说明，不带工具原始输出。"""


def report(ok, message):
    print(f"[{'通过' if ok else '失败'}] {message}")
    return ok


def run_compose(compose_file, arguments, timeout):
    command = ["docker", "compose", "-f", compose_file.name, *arguments]
    try:
        result = subprocess.run(
            command, cwd=compose_file.parent, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CheckError("无法运行 Docker Compose 或命令超时；请检查 CLI、context 与超时设置。") from exc
    if result.returncode:
        raise CheckError(
            f"Compose {arguments[0]} 检查失败；请核对配置、环境文件和 Docker context（不回显原始错误）。"
        )
    return result.stdout


def read_config(compose_file, timeout):
    raw = run_compose(compose_file, ["config", "--format", "json"], timeout)
    try:
        config = json.loads(raw)
        services = config["services"]
        if not isinstance(services, dict) or not all(isinstance(value, dict) for value in services.values()):
            raise ValueError
    except (ValueError, KeyError, TypeError) as exc:
        raise CheckError("Compose 配置输出无效；需要支持 JSON 配置输出的 Docker Compose。") from exc
    return config


def check_environment(services):
    checks = []
    for name, required in REQUIRED_ENV.items():
        environment = services.get(name, {}).get("environment") or {}
        missing = [key for key in required if not str(environment.get(key) or "").strip()]
        detail = "缺少变量：" + ", ".join(missing) if missing else "关键变量已配置（不校验密钥有效性）"
        checks.append(report(not missing, f"{name}：{detail}"))
    backend = services.get("backend", {}).get("environment") or {}
    userservice = services.get("userservice", {}).get("environment") or {}
    shared = backend.get("SECRET_KEY")
    checks.append(report(bool(shared) and shared == userservice.get("JWT_SECRET_KEY"),
                         "FastAPI SECRET_KEY 与 Django JWT_SECRET_KEY 一致"))
    return all(checks)


def check_model(services):
    backend = services.get("backend", {})
    target = (backend.get("environment") or {}).get("RERANKER_MODEL_PATH")
    mounts = [volume for volume in backend.get("volumes", [])
              if volume.get("type") == "bind" and volume.get("target") == target]
    valid = len(mounts) == 1 and mounts[0].get("read_only") is True
    if valid:
        source = mounts[0].get("source")
        valid = isinstance(source, str) and Path(source).is_dir()
    return report(valid, "重排序模型只读挂载目录存在；失败时核对 backend 的模型路径与 bind source（不加载模型）")


def preflight(config):
    services = config["services"]
    missing = [name for name in SERVICES + MIGRATIONS if name not in services]
    definitions = report(not missing, "6 个常驻组件与 2 个迁移任务定义齐全" if not missing
                         else "缺少服务定义：" + ", ".join(missing))
    environment = check_environment(services)
    model = check_model(services)
    return definitions and environment and model


def parse_ps(raw):
    if not raw.strip():
        return []
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("容器状态结构无效")
    return rows


def check_services(config, rows):
    checks = []
    for name in SERVICES + MIGRATIONS:
        matches = [row for row in rows if row.get("Service") == name]
        if len(matches) != 1:
            checks.append(report(False, f"{name}：容器缺失或数量不符合单机单副本配置"))
            continue
        row = matches[0]
        if name in MIGRATIONS:
            ok = row.get("State") == "exited" and str(row.get("ExitCode")) == "0"
            checks.append(report(ok, f"{name}：迁移任务须已退出且退出码为 0"))
            continue
        health = config["services"][name].get("healthcheck") or {}
        monitored = bool(health) and not health.get("disable") and health.get("test") != ["NONE"]
        ok = row.get("State") == "running" and (not monitored or row.get("Health") == "healthy")
        requirement = "running / healthy" if monitored else "running（未配置健康检查）"
        checks.append(report(ok, f"{name}：要求 {requirement}；未通过时检查该服务状态/日志"))
    return all(checks)


def default_url(config):
    ports = config["services"]["nginx"].get("ports", [])
    ports = [port for port in ports if str(port.get("target")) == "80" and port.get("protocol", "tcp") == "tcp"]
    if len(ports) != 1:
        raise CheckError("无法确定 Nginx 发布端口；请显式指定 --url。")
    port = str(ports[0].get("published", ""))
    if not port.isdecimal() or not 0 < int(port) < 65536:
        raise CheckError("Nginx 不是固定发布端口；请显式指定 --url。")
    host = ports[0].get("host_ip") or "127.0.0.1"
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{port}"


def valid_url(value):
    try:
        parsed = urlsplit(value)
        return (parsed.scheme in ("http", "https") and bool(parsed.hostname)
                and not parsed.username and not parsed.password and not parsed.query
                and not parsed.fragment and (parsed.port is None or 0 < parsed.port < 65536))
    except ValueError:
        return False


def check_http(base_url, timeout):
    checks = []
    for path, label in (("/", "前端入口"), ("/health/ready", "后端 readiness")):
        try:
            with urlopen(base_url.rstrip("/") + path, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                ok = response.status == 200
                if path == "/":
                    ok = ok and content_type == "text/html"
                else:
                    payload = json.loads(response.read(65536))
                    data = payload.get("data") if isinstance(payload, dict) else None
                    ok = (ok and content_type == "application/json"
                          and isinstance(data, dict) and data.get("status") == "ok")
        except (OSError, URLError, ValueError):
            ok = False
        checks.append(report(ok, f"{label}：HTTP 与响应类型符合预期；失败时检查代理、端口或依赖服务"))
    return all(checks)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="只检查配置与模型目录，不访问容器状态或 HTTP")
    parser.add_argument("--compose-file", type=Path, default=DEFAULT_COMPOSE, help="Compose 文件，默认仓库根目录")
    parser.add_argument("--url", help="访问入口；默认取 Compose 中 Nginx 的发布端口")
    parser.add_argument("--timeout", type=float, default=5.0, help="每条外部命令或 HTTP 的超时秒数，默认 5")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout 必须是有限正数")
    if args.url and not valid_url(args.url):
        parser.error("--url 必须为不含用户名、密码、查询参数或片段的 HTTP(S) 地址")
    return args


def main(argv=None):
    args = parse_args(argv)
    compose_file = args.compose_file.resolve()
    try:
        if not compose_file.is_file():
            raise CheckError("Compose 文件不存在；请核对 --compose-file。")
        config = read_config(compose_file, args.timeout)
        if not preflight(config):
            return 1
        if args.preflight:
            print("[通过] 配置预检通过；尚未验证容器、端口可占用性或模型推理。")
            return 0
        raw = run_compose(compose_file, ["ps", "--all", "--format", "json"], args.timeout)
        try:
            rows = parse_ps(raw)
        except ValueError as exc:
            raise CheckError("Compose 容器状态不是有效 JSON；未回显原始输出。") from exc
        if not check_services(config, rows):
            print("[跳过] 服务或迁移未就绪，不探测 HTTP；脚本不会自动启动或修复服务。")
            return 1
        if not check_http(args.url or default_url(config), args.timeout):
            return 1
    except CheckError as exc:
        print(f"[无法检查] {exc}")
        return 2
    print("[通过] 基础部署验收通过；未执行模型推理、Celery业务任务或数据恢复。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
