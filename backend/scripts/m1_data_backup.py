"""创建、校验并恢复 M1 遗留数据冷备份。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Final, Iterator

SCHEMA_VERSION: Final = 1
BUFFER_SIZE: Final = 1024 * 1024
LABEL_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
ENV_NAME_PATTERN: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
MYSQL_IDENTIFIER_PATTERN: Final = re.compile(r"[A-Za-z0-9_]{1,64}")
HASH_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
MODE_PATTERN: Final = re.compile(r"[0-7]{4}")
RESTORE_DATABASE_PATTERN: Final = re.compile(r"[A-Za-z0-9_]+_restore_test")


class BackupError(RuntimeError):
    """备份契约被破坏时抛出的可预期错误。"""


@dataclass(frozen=True)
class TreeEntry:
    """备份 payload 中一个普通文件或目录的稳定描述。"""

    path: str
    kind: str
    mode: str
    size: int | None = None
    sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "path": self.path,
            "type": self.kind,
            "mode": self.mode,
        }
        if self.kind == "file":
            value["size"] = self.size
            value["sha256"] = self.sha256
        return value


@dataclass(frozen=True)
class MysqlTarget:
    """单个 MySQL 逻辑备份目标；密码只存在于进程内存。"""

    label: str
    host: str
    port: int
    user: str
    database: str
    password: str = field(repr=False)


Runner = Callable[..., subprocess.CompletedProcess[bytes]]


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _mode(value: os.stat_result) -> str:
    return f"{stat.S_IMODE(value.st_mode):04o}"


def _validate_manifest_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise BackupError("manifest 包含非法相对路径")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise BackupError(f"manifest 路径不是规范相对路径: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise BackupError(f"manifest 路径可能越界: {value!r}")
    return value


def _open_regular_file(path: Path, expected: os.stat_result) -> BinaryIO:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BackupError(f"无法安全读取普通文件: {path}") from exc

    current = os.fstat(descriptor)
    identity = (current.st_dev, current.st_ino, current.st_size)
    expected_identity = (expected.st_dev, expected.st_ino, expected.st_size)
    if not stat.S_ISREG(current.st_mode) or identity != expected_identity:
        os.close(descriptor)
        raise BackupError(f"扫描期间文件发生变化: {path}")
    return os.fdopen(descriptor, "rb")


def _hash_regular_file(path: Path, expected: os.stat_result) -> str:
    digest = hashlib.sha256()
    with _open_regular_file(path, expected) as source:
        before = os.fstat(source.fileno())
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
        after = os.fstat(source.fileno())
    before_state = (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after_state = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_state != after_state:
        raise BackupError(f"计算校验和期间文件发生变化: {path}")
    return digest.hexdigest()


def _scan_tree(root: Path) -> tuple[TreeEntry, ...]:
    if root.is_symlink() or not root.is_dir():
        raise BackupError(f"数据树必须是非符号链接目录: {root}")

    entries: list[TreeEntry] = []

    def visit(directory: Path) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise BackupError(f"无法扫描数据目录: {directory}") from exc

        for child in children:
            path = Path(child.path)
            try:
                info = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise BackupError(f"无法读取数据条目: {path}") from exc
            relative = path.relative_to(root).as_posix()
            _validate_manifest_path(relative)

            if stat.S_ISLNK(info.st_mode):
                raise BackupError(f"数据树禁止符号链接: {relative}")
            if stat.S_ISDIR(info.st_mode):
                entries.append(TreeEntry(relative, "directory", _mode(info)))
                visit(path)
                continue
            if stat.S_ISREG(info.st_mode):
                entries.append(
                    TreeEntry(
                        relative,
                        "file",
                        _mode(info),
                        info.st_size,
                        _hash_regular_file(path, info),
                    )
                )
                continue
            raise BackupError(f"数据树包含不支持的特殊文件: {relative}")

    visit(root)
    return tuple(sorted(entries, key=lambda item: item.path))


def _copy_regular_file(source: Path, target: Path, entry: TreeEntry) -> None:
    source_info = source.lstat()
    if not stat.S_ISREG(source_info.st_mode):
        raise BackupError(f"复制期间源文件类型发生变化: {entry.path}")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(target, flags, 0o600)
    digest = hashlib.sha256()
    size = 0
    try:
        with _open_regular_file(source, source_info) as source_file, os.fdopen(
            descriptor, "wb"
        ) as target_file:
            while chunk := source_file.read(BUFFER_SIZE):
                target_file.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            target_file.flush()
            os.fsync(target_file.fileno())
            os.fchmod(target_file.fileno(), int(entry.mode, 8))
    except Exception:
        if _path_exists(target):
            target.unlink()
        raise

    if size != entry.size or digest.hexdigest() != entry.sha256:
        target.unlink()
        raise BackupError(f"复制期间源文件内容发生变化: {entry.path}")


def _copy_tree(
    source: Path,
    target: Path,
    entries: Sequence[TreeEntry],
    root_mode: str,
) -> None:
    target.mkdir(mode=0o700)
    directories = [entry for entry in entries if entry.kind == "directory"]
    files = [entry for entry in entries if entry.kind == "file"]

    for entry in sorted(directories, key=lambda item: item.path.count("/")):
        (target / entry.path).mkdir(mode=0o700)
    for entry in files:
        _copy_regular_file(source / entry.path, target / entry.path, entry)
    for entry in sorted(
        directories, key=lambda item: item.path.count("/"), reverse=True
    ):
        os.chmod(target / entry.path, int(entry.mode, 8))
    os.chmod(target, int(root_mode, 8))


def _write_secure(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _resolve_config_value(
    field_name: str,
    value: object,
    environ: Mapping[str, str],
    *,
    secret: bool = False,
) -> str:
    if isinstance(value, dict):
        if set(value) != {"env"} or not isinstance(value["env"], str):
            raise BackupError(f"MySQL 字段 {field_name} 的 env 映射无效")
        env_name = value["env"]
        if ENV_NAME_PATTERN.fullmatch(env_name) is None:
            raise BackupError(f"MySQL 字段 {field_name} 的环境变量名无效")
        if env_name not in environ:
            raise BackupError(f"MySQL 字段 {field_name} 缺少环境变量 {env_name}")
        resolved = environ[env_name]
    elif secret:
        raise BackupError("MySQL 密码必须通过环境变量映射提供")
    elif isinstance(value, (str, int)) and not isinstance(value, bool):
        resolved = str(value)
    else:
        raise BackupError(f"MySQL 字段 {field_name} 的值无效")

    if field_name != "password" and not resolved:
        raise BackupError(f"MySQL 字段 {field_name} 不能为空")
    if any(character in resolved for character in ("\0", "\n", "\r")):
        raise BackupError(f"MySQL 字段 {field_name} 包含控制字符")
    return resolved


def _validate_mysql_database(value: object, label: str) -> str:
    if not isinstance(value, str) or MYSQL_IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise BackupError(f"MySQL target {label} 的数据库标识符无效")
    return value


def load_mysql_targets(
    config_path: Path,
    environ: Mapping[str, str] | None = None,
) -> tuple[MysqlTarget, ...]:
    """读取显式 JSON 配置；不会加载项目或当前目录中的 .env。"""
    environment = os.environ if environ is None else environ
    if config_path.is_symlink() or not config_path.is_file():
        raise BackupError("MySQL 配置必须是非符号链接普通文件")
    if config_path.stat().st_size > 1024 * 1024:
        raise BackupError("MySQL 配置文件超过 1 MiB")
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BackupError("MySQL 配置不是有效 UTF-8 JSON") from exc
    if not isinstance(value, dict) or set(value) != {"targets"}:
        raise BackupError("MySQL 配置顶层必须只包含 targets")
    if not isinstance(value["targets"], list) or not value["targets"]:
        raise BackupError("MySQL targets 必须是非空数组")

    expected_fields = {"label", "host", "port", "user", "database", "password"}
    targets: list[MysqlTarget] = []
    labels: set[str] = set()
    for raw_target in value["targets"]:
        if not isinstance(raw_target, dict) or set(raw_target) != expected_fields:
            raise BackupError(f"每个 MySQL target 必须包含 {sorted(expected_fields)}")
        label = _resolve_config_value("label", raw_target["label"], environment)
        if LABEL_PATTERN.fullmatch(label) is None or label in labels:
            raise BackupError(f"MySQL label 非法或重复: {label!r}")
        labels.add(label)
        port_text = _resolve_config_value("port", raw_target["port"], environment)
        try:
            port = int(port_text)
        except ValueError as exc:
            raise BackupError(f"MySQL target {label} 的端口不是整数") from exc
        if not 1 <= port <= 65535:
            raise BackupError(f"MySQL target {label} 的端口超出范围")
        database = _validate_mysql_database(
            _resolve_config_value("database", raw_target["database"], environment),
            label,
        )
        targets.append(
            MysqlTarget(
                label=label,
                host=_resolve_config_value("host", raw_target["host"], environment),
                port=port,
                user=_resolve_config_value("user", raw_target["user"], environment),
                database=database,
                password=_resolve_config_value(
                    "password", raw_target["password"], environment, secret=True
                ),
            )
        )
    return tuple(targets)


def _validate_mysql_targets(
    targets: Sequence[MysqlTarget],
) -> tuple[MysqlTarget, ...]:
    validated = tuple(targets)
    labels: set[str] = set()
    for target in validated:
        if not isinstance(target, MysqlTarget):
            raise BackupError("MySQL target 类型无效")
        if (
            not isinstance(target.label, str)
            or LABEL_PATTERN.fullmatch(target.label) is None
            or target.label in labels
        ):
            raise BackupError(f"MySQL label 非法或重复: {target.label!r}")
        labels.add(target.label)
        if (
            isinstance(target.port, bool)
            or not isinstance(target.port, int)
            or not 1 <= target.port <= 65535
        ):
            raise BackupError(f"MySQL target {target.label} 的端口超出范围")
        for field_name, value in (
            ("host", target.host),
            ("user", target.user),
        ):
            if (
                not isinstance(value, str)
                or not value
                or any(character in value for character in ("\0", "\n", "\r"))
            ):
                raise BackupError(f"MySQL target {target.label} 的 {field_name} 无效")
        _validate_mysql_database(target.database, target.label)
        if not isinstance(target.password, str) or any(
            character in target.password for character in ("\0", "\n", "\r")
        ):
            raise BackupError(f"MySQL target {target.label} 的密码包含控制字符")
    return validated


def _mysql_option_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


@contextmanager
def _mysql_defaults_file(target: MysqlTarget) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="rag-note-mysql-") as temp_dir:
        option_path = Path(temp_dir) / "client.cnf"
        option_text = "\n".join(
            (
                "[client]",
                f"host={_mysql_option_value(target.host)}",
                f"port={target.port}",
                f"user={_mysql_option_value(target.user)}",
                f"password={_mysql_option_value(target.password)}",
                "protocol=TCP",
                "default-character-set=utf8mb4",
                "",
            )
        )
        _write_secure(option_path, option_text.encode("utf-8"))
        try:
            yield option_path
        finally:
            if _path_exists(option_path):
                option_path.unlink()


def _database_hex(database: str) -> str:
    return database.encode("utf-8").hex()


def _run_mysql_query(
    target: MysqlTarget,
    query: str,
    *,
    mysql_bin: str,
    runner: Runner,
) -> str:
    with _mysql_defaults_file(target) as option_path:
        command = [
            mysql_bin,
            f"--defaults-extra-file={option_path}",
            "--batch",
            "--skip-column-names",
            "--raw",
            f"--execute={query}",
        ]
        result = runner(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode != 0:
        raise BackupError(f"MySQL 预检失败: {target.label}")
    try:
        return result.stdout.decode("utf-8").strip()
    except (AttributeError, UnicodeDecodeError) as exc:
        raise BackupError(f"MySQL 预检输出无效: {target.label}") from exc


def inspect_mysql_storage(
    target: MysqlTarget,
    *,
    mysql_bin: str = "mysql",
    runner: Runner = subprocess.run,
) -> int:
    """返回 base table 数，并拒绝任何非 InnoDB base table。"""
    database = f"0x{_database_hex(target.database)}"
    query = (
        "SELECT COUNT(*), "
        "COALESCE(SUM(CASE WHEN ENGINE = 'InnoDB' THEN 0 ELSE 1 END), 0) "
        "FROM information_schema.tables "
        f"WHERE table_schema = {database} AND table_type = 'BASE TABLE'"
    )
    output = _run_mysql_query(target, query, mysql_bin=mysql_bin, runner=runner)
    fields = output.split("\t")
    if len(fields) != 2:
        raise BackupError(f"MySQL 引擎预检输出无效: {target.label}")
    try:
        table_count, non_innodb_count = (int(field) for field in fields)
    except ValueError as exc:
        raise BackupError(f"MySQL 引擎预检输出无效: {target.label}") from exc
    if table_count < 0 or non_innodb_count < 0 or non_innodb_count > table_count:
        raise BackupError(f"MySQL 引擎预检计数无效: {target.label}")
    if non_innodb_count:
        raise BackupError(
            f"MySQL 数据库包含 {non_innodb_count} 个非 InnoDB base table: "
            f"{target.label}"
        )
    return table_count


def dump_mysql_target(
    target: MysqlTarget,
    output_path: Path,
    *,
    mysql_bin: str = "mysql",
    mysqldump_bin: str = "mysqldump",
    runner: Runner = subprocess.run,
) -> int:
    """使用 0600 临时 option file 执行单库一致性逻辑转储。"""
    if _path_exists(output_path):
        raise BackupError(f"MySQL dump 目标已存在: {output_path}")
    database = _validate_mysql_database(target.database, target.label)
    table_count = inspect_mysql_storage(target, mysql_bin=mysql_bin, runner=runner)
    try:
        with _mysql_defaults_file(target) as option_path:
            descriptor = os.open(
                output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            command = [
                mysqldump_bin,
                f"--defaults-extra-file={option_path}",
                "--single-transaction",
                "--quick",
                "--routines",
                "--events",
                "--triggers",
                "--hex-blob",
                "--no-tablespaces",
                "--set-gtid-purged=OFF",
                database,
            ]
            with os.fdopen(descriptor, "wb") as output:
                result = runner(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                output.flush()
                os.fsync(output.fileno())
            if result.returncode != 0:
                raise BackupError(f"mysqldump 执行失败: {target.label}")
        if output_path.stat().st_size == 0:
            raise BackupError(f"mysqldump 生成了空文件: {target.label}")
    except Exception:
        if _path_exists(output_path):
            output_path.unlink()
        raise
    return table_count


def _manifest_bytes(
    entries: Sequence[TreeEntry],
    table_counts: Mapping[str, int],
) -> bytes:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "mysql_labels": list(table_counts),
        "mysql_base_table_counts": dict(table_counts),
        "entries": [entry.to_dict() for entry in entries],
    }
    return (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _parse_manifest(path: Path) -> tuple[dict[str, object], tuple[TreeEntry, ...]]:
    if path.is_symlink() or not path.is_file():
        raise BackupError("备份缺少普通 manifest.json 文件")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BackupError("manifest.json 不是有效 UTF-8 JSON") from exc
    expected_keys = {
        "schema_version",
        "created_at",
        "mysql_labels",
        "mysql_base_table_counts",
        "entries",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise BackupError("manifest.json 字段集合无效")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise BackupError("不支持的备份 manifest 版本")
    if not isinstance(manifest["created_at"], str):
        raise BackupError("manifest created_at 无效")
    labels = manifest["mysql_labels"]
    if not isinstance(labels, list) or any(
        not isinstance(label, str) or LABEL_PATTERN.fullmatch(label) is None
        for label in labels
    ):
        raise BackupError("manifest mysql_labels 无效")
    if len(labels) != len(set(labels)):
        raise BackupError("manifest mysql_labels 重复")
    table_counts = manifest["mysql_base_table_counts"]
    if not isinstance(table_counts, dict) or set(table_counts) != set(labels):
        raise BackupError("manifest mysql_base_table_counts 与 labels 不一致")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in table_counts.values()
    ):
        raise BackupError("manifest mysql_base_table_counts 无效")
    raw_entries = manifest["entries"]
    if not isinstance(raw_entries, list):
        raise BackupError("manifest entries 必须是数组")

    entries: list[TreeEntry] = []
    paths: set[str] = set()
    for value in raw_entries:
        if not isinstance(value, dict):
            raise BackupError("manifest entry 必须是对象")
        kind = value.get("type")
        keys = {"path", "type", "mode"}
        if kind == "file":
            keys |= {"size", "sha256"}
        if set(value) != keys or kind not in {"file", "directory"}:
            raise BackupError("manifest entry 字段或类型无效")
        entry_path = _validate_manifest_path(value["path"])
        if entry_path in paths:
            raise BackupError(f"manifest 路径重复: {entry_path}")
        paths.add(entry_path)
        mode = value["mode"]
        if not isinstance(mode, str) or MODE_PATTERN.fullmatch(mode) is None:
            raise BackupError(f"manifest mode 无效: {entry_path}")
        if kind == "directory":
            entries.append(TreeEntry(entry_path, kind, mode))
            continue
        size = value["size"]
        sha256 = value["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BackupError(f"manifest size 无效: {entry_path}")
        if not isinstance(sha256, str) or HASH_PATTERN.fullmatch(sha256) is None:
            raise BackupError(f"manifest sha256 无效: {entry_path}")
        entries.append(TreeEntry(entry_path, kind, mode, size, sha256))

    result = tuple(sorted(entries, key=lambda item: item.path))
    if not any(entry.path == "data" and entry.kind == "directory" for entry in result):
        raise BackupError("manifest 缺少 payload/data 目录")
    for label in labels:
        expected_dump = f"mysql/{label}.sql"
        if not any(
            entry.path == expected_dump and entry.kind == "file" for entry in result
        ):
            raise BackupError(f"manifest 缺少 MySQL dump: {label}")
    return manifest, result


def _assert_private(path: Path, expected_type: str) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise BackupError(f"备份中的 {expected_type} 禁止符号链接")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise BackupError(f"备份中的 {expected_type} 权限过宽")


def verify_backup(backup_dir: Path) -> dict[str, object]:
    """验证备份布局、路径边界、条目集合、权限、大小与 SHA-256。"""
    if backup_dir.is_symlink():
        raise BackupError("备份根路径不能是符号链接")
    backup_dir = backup_dir.resolve(strict=True)
    if not backup_dir.is_dir():
        raise BackupError("备份路径不是目录")
    _assert_private(backup_dir, "根目录")
    root_names = {entry.name for entry in os.scandir(backup_dir)}
    if root_names != {"manifest.json", "payload"}:
        raise BackupError("备份根目录存在额外或缺失条目")

    manifest_path = backup_dir / "manifest.json"
    payload = backup_dir / "payload"
    _assert_private(manifest_path, "manifest")
    _assert_private(payload, "payload 目录")
    manifest, expected_entries = _parse_manifest(manifest_path)
    actual_entries = _scan_tree(payload)
    expected_by_path = {entry.path: entry for entry in expected_entries}
    actual_by_path = {entry.path: entry for entry in actual_entries}
    missing = sorted(expected_by_path.keys() - actual_by_path.keys())
    extra = sorted(actual_by_path.keys() - expected_by_path.keys())
    if missing or extra:
        raise BackupError(f"备份 payload 条目不一致: missing={missing}, extra={extra}")
    for entry_path, expected in expected_by_path.items():
        if actual_by_path[entry_path] != expected:
            raise BackupError(f"备份条目校验失败: {entry_path}")
    return manifest


def _resolved_new_path(path: Path) -> Path:
    if _path_exists(path):
        raise BackupError(f"目标已存在: {path}")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise BackupError(f"目标父路径不是目录: {parent}")
    return parent / path.name


def create_backup(
    source: Path,
    backup_dir: Path,
    *,
    mysql_targets: Sequence[MysqlTarget] = (),
    mysql_bin: str = "mysql",
    mysqldump_bin: str = "mysqldump",
    runner: Runner = subprocess.run,
) -> dict[str, object]:
    """从显式源目录创建不可覆盖、原子发布的冷备份包。"""
    if source.is_symlink():
        raise BackupError("源目录不能是符号链接")
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise BackupError("源路径不是目录")
    backup_dir = _resolved_new_path(backup_dir)
    if backup_dir.is_relative_to(source):
        raise BackupError("备份输出不能位于源目录内部")
    mysql_targets = _validate_mysql_targets(mysql_targets)

    source_root_mode = _mode(source.stat())
    source_entries = _scan_tree(source)
    staging = backup_dir.parent / f".{backup_dir.name}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir(mode=0o700)
        payload = staging / "payload"
        payload.mkdir(mode=0o700)
        _copy_tree(source, payload / "data", source_entries, source_root_mode)

        table_counts: dict[str, int] = {}
        if mysql_targets:
            mysql_dir = payload / "mysql"
            mysql_dir.mkdir(mode=0o700)
            for target in mysql_targets:
                table_counts[target.label] = dump_mysql_target(
                    target,
                    mysql_dir / f"{target.label}.sql",
                    mysql_bin=mysql_bin,
                    mysqldump_bin=mysqldump_bin,
                    runner=runner,
                )

        if source_root_mode != _mode(source.stat()) or source_entries != _scan_tree(
            source
        ):
            raise BackupError("备份期间源数据发生变化，拒绝发布快照")
        payload_entries = _scan_tree(payload)
        _write_secure(
            staging / "manifest.json", _manifest_bytes(payload_entries, table_counts)
        )
        verify_backup(staging)
        os.replace(staging, backup_dir)
        return verify_backup(backup_dir)
    except Exception:
        if _path_exists(staging):
            shutil.rmtree(staging)
        raise


def _parse_mysql_count(output: str, label: str) -> int:
    try:
        count = int(output)
    except ValueError as exc:
        raise BackupError(f"MySQL 计数查询输出无效: {label}") from exc
    if count < 0:
        raise BackupError(f"MySQL 计数查询输出无效: {label}")
    return count


def _mysql_schema_exists(
    target: MysqlTarget,
    *,
    mysql_bin: str,
    runner: Runner,
) -> bool:
    database = f"0x{_database_hex(target.database)}"
    query = (
        "SELECT COUNT(*) FROM information_schema.schemata "
        f"WHERE schema_name = {database}"
    )
    count = _parse_mysql_count(
        _run_mysql_query(target, query, mysql_bin=mysql_bin, runner=runner),
        target.label,
    )
    if count not in {0, 1}:
        raise BackupError(f"MySQL schema 查询返回异常计数: {target.label}")
    return count == 1


def _mysql_object_count(
    target: MysqlTarget,
    *,
    mysql_bin: str,
    runner: Runner,
) -> int:
    database = f"0x{_database_hex(target.database)}"
    query = (
        "SELECT "
        f"(SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = {database}) + "
        f"(SELECT COUNT(*) FROM information_schema.routines WHERE routine_schema = {database}) + "
        f"(SELECT COUNT(*) FROM information_schema.events WHERE event_schema = {database})"
    )
    return _parse_mysql_count(
        _run_mysql_query(target, query, mysql_bin=mysql_bin, runner=runner),
        target.label,
    )


def _create_mysql_restore_database(
    target: MysqlTarget,
    *,
    mysql_bin: str,
    runner: Runner,
) -> None:
    query = f"CREATE DATABASE IF NOT EXISTS `{target.database}`"
    _run_mysql_query(target, query, mysql_bin=mysql_bin, runner=runner)


def _restore_mysql_dump(
    target: MysqlTarget,
    dump_path: Path,
    *,
    mysql_bin: str,
    runner: Runner,
) -> None:
    with _mysql_defaults_file(target) as option_path, dump_path.open("rb") as source:
        command = [
            mysql_bin,
            f"--defaults-extra-file={option_path}",
            "--binary-mode=1",
            f"--database={target.database}",
        ]
        result = runner(
            command,
            stdin=source,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode != 0:
        raise BackupError(f"MySQL 恢复失败，测试库可能包含部分数据: {target.label}")


def restore_mysql_backup(
    backup_dir: Path,
    mysql_targets: Sequence[MysqlTarget],
    *,
    mysql_bin: str = "mysql",
    runner: Runner = subprocess.run,
) -> dict[str, int]:
    """将 dump 导入显式空 `_restore_test` 库并核对 base table 数。"""
    manifest = verify_backup(backup_dir)
    backup_dir = backup_dir.resolve(strict=True)
    targets = _validate_mysql_targets(mysql_targets)
    labels = manifest["mysql_labels"]
    table_counts = manifest["mysql_base_table_counts"]
    if not isinstance(labels, list) or not labels:
        raise BackupError("备份不包含 MySQL dump")
    if not isinstance(table_counts, dict):
        raise BackupError("备份缺少 MySQL base table 计数")
    if {target.label for target in targets} != set(labels):
        raise BackupError("恢复 target labels 必须与备份完全一致")
    databases = [target.database for target in targets]
    if len(databases) != len(set(databases)):
        raise BackupError("多个 MySQL target 不能恢复到同一个数据库")
    for target in targets:
        if RESTORE_DATABASE_PATTERN.fullmatch(target.database) is None:
            raise BackupError(
                f"MySQL 恢复数据库必须以 _restore_test 结尾: {target.label}"
            )

    existence: dict[str, bool] = {}
    for target in targets:
        exists = _mysql_schema_exists(target, mysql_bin=mysql_bin, runner=runner)
        existence[target.label] = exists
        if exists and _mysql_object_count(target, mysql_bin=mysql_bin, runner=runner):
            raise BackupError(f"MySQL 恢复目标数据库非空: {target.label}")

    restored_counts: dict[str, int] = {}
    for target in targets:
        if not existence[target.label]:
            _create_mysql_restore_database(target, mysql_bin=mysql_bin, runner=runner)
        dump_path = backup_dir / "payload/mysql" / f"{target.label}.sql"
        _restore_mysql_dump(target, dump_path, mysql_bin=mysql_bin, runner=runner)
        restored_count = inspect_mysql_storage(
            target, mysql_bin=mysql_bin, runner=runner
        )
        expected_count = table_counts[target.label]
        if restored_count != expected_count:
            raise BackupError(
                f"MySQL 恢复后 base table 数不一致: {target.label} "
                f"expected={expected_count}, actual={restored_count}"
            )
        restored_counts[target.label] = restored_count
    return restored_counts


def restore_backup(backup_dir: Path, target: Path) -> dict[str, object]:
    """将已验证 payload 原子恢复到不存在或空的目标目录。"""
    manifest = verify_backup(backup_dir)
    backup_dir = backup_dir.resolve(strict=True)
    target_was_empty = False
    if _path_exists(target):
        if target.is_symlink() or not target.is_dir():
            raise BackupError("恢复目标必须是非符号链接目录")
        if any(os.scandir(target)):
            raise BackupError("恢复目标非空，拒绝覆盖")
        target_was_empty = True
        target = target.resolve(strict=True)
    else:
        target = target.parent.resolve(strict=True) / target.name
    if target.is_relative_to(backup_dir):
        raise BackupError("恢复目标不能位于备份目录内部")

    _, expected_entries = _parse_manifest(backup_dir / "manifest.json")
    payload = backup_dir / "payload"
    staging = target.parent / f".{target.name}.restore-{uuid.uuid4().hex}"
    try:
        _copy_tree(payload, staging, expected_entries, _mode(payload.stat()))
        if _scan_tree(staging) != expected_entries:
            raise BackupError("恢复副本与 manifest 不一致")
        if target_was_empty:
            target.rmdir()
        os.rename(staging, target)
        return manifest
    except Exception:
        if _path_exists(staging):
            shutil.rmtree(staging)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="创建新的冷备份包")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--backup-dir", type=Path, required=True)
    create.add_argument("--mysql-config", type=Path)
    create.add_argument("--mysql-bin", default="mysql")
    create.add_argument("--mysqldump-bin", default="mysqldump")

    verify = subparsers.add_parser("verify", help="校验已有备份包")
    verify.add_argument("--backup-dir", type=Path, required=True)

    restore = subparsers.add_parser("restore", help="恢复到不存在或空目录")
    restore.add_argument("--backup-dir", type=Path, required=True)
    restore.add_argument("--target", type=Path, required=True)

    restore_mysql = subparsers.add_parser(
        "restore-mysql", help="恢复 MySQL dump 到空的 _restore_test 数据库"
    )
    restore_mysql.add_argument("--backup-dir", type=Path, required=True)
    restore_mysql.add_argument("--mysql-config", type=Path, required=True)
    restore_mysql.add_argument("--mysql-bin", default="mysql")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "create":
            targets = (
                load_mysql_targets(arguments.mysql_config)
                if arguments.mysql_config
                else ()
            )
            manifest = create_backup(
                arguments.source,
                arguments.backup_dir,
                mysql_targets=targets,
                mysql_bin=arguments.mysql_bin,
                mysqldump_bin=arguments.mysqldump_bin,
            )
            print(
                f"备份已创建并校验: {arguments.backup_dir} "
                f"({len(manifest['entries'])} 个条目)"
            )
        elif arguments.command == "verify":
            manifest = verify_backup(arguments.backup_dir)
            print(f"备份校验通过: {len(manifest['entries'])} 个条目")
        elif arguments.command == "restore":
            manifest = restore_backup(arguments.backup_dir, arguments.target)
            print(
                f"备份已恢复并校验: {arguments.target} "
                f"({len(manifest['entries'])} 个条目)"
            )
        else:
            targets = load_mysql_targets(arguments.mysql_config)
            restored = restore_mysql_backup(
                arguments.backup_dir,
                targets,
                mysql_bin=arguments.mysql_bin,
            )
            print(f"MySQL 恢复并校验通过: {restored}")
    except (BackupError, OSError) as exc:
        parser.exit(2, f"错误: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
