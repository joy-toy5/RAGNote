from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
from pathlib import Path
from typing import Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
GENERATED_REPORTS = {
    Path("backend/tests/baseline/m0_baseline.json"),
    Path("backend/tests/baseline/m0_fake_rag_baseline.json"),
}


def run(command: list[str], cwd: Path = REPOSITORY_ROOT) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable ({type(error).__name__})"
    output = result.stdout.strip() or result.stderr.strip()
    return output if result.returncode == 0 else f"unavailable ({output})"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(REPOSITORY_ROOT)): digest_file(path)
        for path in sorted(paths)
        if path.is_file()
    }


def git_fingerprint() -> dict[str, object]:
    diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    untracked_raw = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    untracked = [
        Path(value.decode("utf-8"))
        for value in untracked_raw.split(b"\0")
        if value and Path(value.decode("utf-8")) not in GENERATED_REPORTS
    ]
    return {
        "head": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "branch", "--show-current"]),
        "status_sha256": digest_bytes(
            subprocess.run(
                ["git", "status", "--porcelain=v2", "-z"],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
            ).stdout
        ),
        "tracked_diff_sha256": digest_bytes(diff),
        "untracked": hashes(REPOSITORY_ROOT / path for path in untracked),
        "excluded_generated": sorted(str(path) for path in GENERATED_REPORTS),
    }


def data_inventory() -> dict[str, object]:
    data_root = BACKEND_ROOT / "data"
    groups: dict[str, dict[str, int]] = {}
    if not data_root.exists():
        return {"exists": False, "groups": groups}
    for child in sorted(data_root.iterdir()):
        files = [path for path in child.rglob("*") if path.is_file()]
        groups[child.name] = {
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
        }
    return {"exists": True, "groups": groups}


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(0.2)
        return client.connect_ex(("127.0.0.1", port)) == 0


def capture() -> dict[str, object]:
    config_files = [
        BACKEND_ROOT / "pyproject.toml",
        BACKEND_ROOT / "uv.lock",
        BACKEND_ROOT / ".env.example",
        *sorted((BACKEND_ROOT / "app/config").glob("*.yaml")),
    ]
    return {
        "schema_version": 1,
        "git": git_fingerprint(),
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "uv": run(["uv", "--version"]),
            "node": run(["node", "--version"]),
            "npm": run(["npm", "--version"]),
            "mysql_client": run(["mysql", "--version"]),
            "redis_server": run(["redis-server", "--version"]),
            "ollama": run(["ollama", "--version"]),
        },
        "local_ports": {
            "mysql_3306": port_open(3306),
            "redis_6379": port_open(6379),
            "fastapi_8000": port_open(8000),
            "django_8001": port_open(8001),
            "ollama_11434": port_open(11434),
        },
        "config_sha256": hashes(config_files),
        "data_inventory": data_inventory(),
        "privacy": "未采集环境变量值、数据库记录、用户路径或文档内容。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成脱敏的 M0 基线指纹")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    content = json.dumps(capture(), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
