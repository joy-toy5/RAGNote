from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts/m1_data_backup.py"
SPEC = importlib.util.spec_from_file_location("_m1_data_backup", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BACKUP_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BACKUP_MODULE
SPEC.loader.exec_module(BACKUP_MODULE)

BackupError = BACKUP_MODULE.BackupError
MysqlTarget = BACKUP_MODULE.MysqlTarget
create_backup = BACKUP_MODULE.create_backup
load_mysql_targets = BACKUP_MODULE.load_mysql_targets
restore_backup = BACKUP_MODULE.restore_backup
restore_mysql_backup = BACKUP_MODULE.restore_mysql_backup
verify_backup = BACKUP_MODULE.verify_backup


@pytest.fixture
def source_tree(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    nested = source / "nested/empty"
    nested.mkdir(parents=True)
    (source / "document.txt").write_text("M1 cold backup\n", encoding="utf-8")
    (source / "nested/data.bin").write_bytes(bytes(range(64)))
    os.chmod(source, 0o750)
    os.chmod(source / "nested", 0o710)
    os.chmod(nested, 0o700)
    os.chmod(source / "document.txt", 0o640)
    os.chmod(source / "nested/data.bin", 0o600)
    return source


def test_create_verify_and_restore_preserve_manifest_contract(
    tmp_path: Path,
    source_tree: Path,
) -> None:
    backup = tmp_path / "backup"
    manifest = create_backup(source_tree, backup)

    assert verify_backup(backup) == manifest
    assert stat.S_IMODE(backup.stat().st_mode) == 0o700
    assert stat.S_IMODE((backup / "manifest.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((backup / "payload").stat().st_mode) == 0o700

    entries = {entry["path"]: entry for entry in manifest["entries"]}
    assert entries["data/document.txt"] == {
        "path": "data/document.txt",
        "type": "file",
        "mode": "0640",
        "size": 15,
        "sha256": "982aa54063ba00d49361cfe470599b8fce9b923c85d037199c7c40150697f178",
    }
    assert entries["data/nested/empty"] == {
        "path": "data/nested/empty",
        "type": "directory",
        "mode": "0700",
    }
    manifest_text = (backup / "manifest.json").read_text(encoding="utf-8")
    assert str(source_tree) not in manifest_text
    assert manifest["mysql_labels"] == []

    restored = tmp_path / "restored"
    assert restore_backup(backup, restored) == manifest
    assert (restored / "data/document.txt").read_text(encoding="utf-8") == (
        "M1 cold backup\n"
    )
    assert (restored / "data/nested/data.bin").read_bytes() == bytes(range(64))
    assert (restored / "data/nested/empty").is_dir()
    assert stat.S_IMODE((restored / "data").stat().st_mode) == 0o750
    assert stat.S_IMODE((restored / "data/document.txt").stat().st_mode) == 0o640


def test_restore_accepts_an_existing_empty_directory(
    tmp_path: Path,
    source_tree: Path,
) -> None:
    backup = tmp_path / "backup"
    target = tmp_path / "empty-target"
    target.mkdir()
    create_backup(source_tree, backup)

    restore_backup(backup, target)

    assert (target / "data/document.txt").is_file()


@pytest.mark.parametrize("damage", ["missing", "extra", "changed"])
def test_verify_rejects_missing_extra_and_changed_files(
    tmp_path: Path,
    source_tree: Path,
    damage: str,
) -> None:
    backup = tmp_path / f"backup-{damage}"
    create_backup(source_tree, backup)
    document = backup / "payload/data/document.txt"

    if damage == "missing":
        document.unlink()
    elif damage == "extra":
        (backup / "payload/data/extra.txt").write_text("extra", encoding="utf-8")
    else:
        document.write_text("changed", encoding="utf-8")

    with pytest.raises(BackupError, match="条目不一致|条目校验失败"):
        verify_backup(backup)


def test_verify_rejects_manifest_path_traversal(
    tmp_path: Path,
    source_tree: Path,
) -> None:
    backup = tmp_path / "backup"
    create_backup(source_tree, backup)
    manifest_path = backup / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["path"] = "../escape"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BackupError, match="路径可能越界"):
        verify_backup(backup)
    assert not (tmp_path / "escape").exists()


def test_create_rejects_symlink_source_entries(
    tmp_path: Path,
    source_tree: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("must not be copied", encoding="utf-8")
    (source_tree / "link").symlink_to(outside)

    with pytest.raises(BackupError, match="禁止符号链接"):
        create_backup(source_tree, tmp_path / "backup")
    assert not (tmp_path / "backup").exists()


def test_verify_rejects_symlink_backup_root(
    tmp_path: Path,
    source_tree: Path,
) -> None:
    backup = tmp_path / "backup"
    create_backup(source_tree, backup)
    backup_link = tmp_path / "backup-link"
    backup_link.symlink_to(backup, target_is_directory=True)

    with pytest.raises(BackupError, match="根路径不能是符号链接"):
        verify_backup(backup_link)


def test_create_rejects_output_inside_source_and_existing_target(
    tmp_path: Path,
    source_tree: Path,
) -> None:
    with pytest.raises(BackupError, match="不能位于源目录内部"):
        create_backup(source_tree, source_tree / "backup")

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(BackupError, match="目标已存在"):
        create_backup(source_tree, existing)


def test_restore_rejects_nonempty_target_and_target_inside_backup(
    tmp_path: Path,
    source_tree: Path,
) -> None:
    backup = tmp_path / "backup"
    create_backup(source_tree, backup)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    sentinel = nonempty / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(BackupError, match="恢复目标非空"):
        restore_backup(backup, nonempty)
    assert sentinel.read_text(encoding="utf-8") == "keep"

    with pytest.raises(BackupError, match="不能位于备份目录内部"):
        restore_backup(backup, backup / "empty-target")


def test_mysql_dump_uses_private_temporary_option_file(
    tmp_path: Path,
    source_tree: Path,
) -> None:
    config_path = tmp_path / "mysql.json"
    config_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "label": "fastapi",
                        "host": {"env": "M1_MYSQL_HOST"},
                        "port": 3307,
                        "user": "backup_user",
                        "database": {"env": "M1_MYSQL_DATABASE"},
                        "password": {"env": "M1_MYSQL_PASSWORD"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    secret = 'not-on-argv-"-or-manifest'
    targets = load_mysql_targets(
        config_path,
        {
            "M1_MYSQL_HOST": "127.0.0.1",
            "M1_MYSQL_DATABASE": "chat_history",
            "M1_MYSQL_PASSWORD": secret,
        },
    )
    observed: dict[str, object] = {}
    option_paths: list[Path] = []

    def fake_runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess:
        option_path = Path(command[1].split("=", 1)[1])
        option_paths.append(option_path)
        observed["command"] = command
        observed["option_path"] = option_path
        observed["option_mode"] = stat.S_IMODE(option_path.stat().st_mode)
        observed["option_text"] = option_path.read_text(encoding="utf-8")
        if command[0] == "mysql":
            return subprocess.CompletedProcess(command, 0, stdout=b"4\t0\n", stderr=b"")
        output = kwargs["stdout"]
        assert hasattr(output, "write")
        output.write(b"-- deterministic test dump\n")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    backup = tmp_path / "backup"
    manifest = create_backup(
        source_tree,
        backup,
        mysql_targets=targets,
        mysqldump_bin="/usr/bin/mysqldump",
        runner=fake_runner,
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert command[0] == "/usr/bin/mysqldump"
    assert command[1].startswith("--defaults-extra-file=")
    assert secret not in "\n".join(command)
    assert observed["option_mode"] == 0o600
    assert secret.replace('"', '\\"') in observed["option_text"]
    assert option_paths and all(not path.exists() for path in option_paths)
    assert stat.S_IMODE((backup / "payload/mysql/fastapi.sql").stat().st_mode) == 0o600
    assert manifest["mysql_labels"] == ["fastapi"]
    assert manifest["mysql_base_table_counts"] == {"fastapi": 4}
    assert secret not in (backup / "manifest.json").read_text(encoding="utf-8")


def test_mysql_config_rejects_literal_password(tmp_path: Path) -> None:
    config_path = tmp_path / "mysql.json"
    config_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "label": "unsafe",
                        "host": "localhost",
                        "port": 3306,
                        "user": "root",
                        "database": "chat_history",
                        "password": "literal-secret",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BackupError, match="密码必须通过环境变量映射"):
        load_mysql_targets(config_path, {})


@pytest.mark.parametrize(
    "database",
    [
        "--all-databases",
        "-A",
        "chat history",
        "chat.history",
        "`chat`",
        "chat/data",
        "a" * 65,
    ],
)
def test_mysql_config_rejects_unsafe_database_identifier(
    tmp_path: Path,
    database: str,
) -> None:
    config_path = tmp_path / "mysql.json"
    config_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "label": "unsafe",
                        "host": "localhost",
                        "port": 3306,
                        "user": "backup",
                        "database": {"env": "M1_DATABASE"},
                        "password": {"env": "M1_PASSWORD"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BackupError, match="数据库标识符无效"):
        load_mysql_targets(
            config_path,
            {"M1_DATABASE": database, "M1_PASSWORD": "secret"},
        )


@pytest.mark.parametrize(
    "database",
    [
        "--all-databases",
        "-A",
        "chat history",
        "chat.history",
        "`chat`",
        "chat/data",
        "a" * 65,
    ],
)
def test_direct_mysql_target_rejects_unsafe_database_before_subprocess(
    tmp_path: Path,
    source_tree: Path,
    database: str,
) -> None:
    target = MysqlTarget(
        label="unsafe",
        host="localhost",
        port=3306,
        user="backup",
        database=database,
        password="secret",
    )
    calls: list[list[str]] = []

    def should_not_run(command: list[str], **_: object) -> subprocess.CompletedProcess:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    with pytest.raises(BackupError, match="数据库标识符无效"):
        create_backup(
            source_tree,
            tmp_path / "backup",
            mysql_targets=[target],
            runner=should_not_run,
        )

    assert calls == []
    assert not (tmp_path / "backup").exists()


def test_failed_mysql_dump_publishes_nothing_and_removes_secret_file(
    tmp_path: Path,
    source_tree: Path,
) -> None:
    config_path = tmp_path / "mysql.json"
    config_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "label": "failed",
                        "host": "localhost",
                        "port": 3306,
                        "user": "backup",
                        "database": "chat_history",
                        "password": {"env": "M1_PASSWORD"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    targets = load_mysql_targets(config_path, {"M1_PASSWORD": "secret"})
    observed_path: list[Path] = []

    def failed_runner(command: list[str], **_: object) -> subprocess.CompletedProcess:
        observed_path.append(Path(command[1].split("=", 1)[1]))
        if command[0] == "mysql":
            return subprocess.CompletedProcess(command, 0, stdout=b"1\t0\n", stderr=b"")
        return subprocess.CompletedProcess(command, 2, stdout=b"", stderr=b"failed")

    backup = tmp_path / "backup"
    with pytest.raises(BackupError, match="mysqldump 执行失败: failed"):
        create_backup(
            source_tree,
            backup,
            mysql_targets=targets,
            runner=failed_runner,
        )

    assert not backup.exists()
    assert observed_path and all(not path.exists() for path in observed_path)
    assert not list(tmp_path.glob(".backup.staging-*"))


def test_mysql_dump_rejects_non_innodb_tables_before_dump(
    tmp_path: Path,
    source_tree: Path,
) -> None:
    target = MysqlTarget(
        label="legacy",
        host="localhost",
        port=3306,
        user="backup",
        database="legacy_data",
        password="secret",
    )
    commands: list[list[str]] = []

    def non_innodb_runner(
        command: list[str], **_: object
    ) -> subprocess.CompletedProcess:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"3\t1\n", stderr=b"")

    backup = tmp_path / "backup"
    with pytest.raises(BackupError, match="1 个非 InnoDB"):
        create_backup(
            source_tree,
            backup,
            mysql_targets=[target],
            runner=non_innodb_runner,
        )

    assert not backup.exists()
    assert commands and all(command[0] == "mysql" for command in commands)


def _create_mysql_test_backup(
    tmp_path: Path,
    source_tree: Path,
) -> tuple[Path, str]:
    secret = "source-secret"
    source_target = MysqlTarget(
        label="fastapi",
        host="localhost",
        port=3306,
        user="backup",
        database="chat_history",
        password=secret,
    )

    def backup_runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess:
        if command[0] == "mysql":
            return subprocess.CompletedProcess(command, 0, stdout=b"2\t0\n", stderr=b"")
        output = kwargs["stdout"]
        assert hasattr(output, "write")
        output.write(b"CREATE TABLE restored (id INT) ENGINE=InnoDB;\n")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    backup = tmp_path / "mysql-backup"
    create_backup(
        source_tree,
        backup,
        mysql_targets=[source_target],
        runner=backup_runner,
    )
    return backup, secret


def test_mysql_restore_requires_safe_empty_database_and_checks_table_count(
    tmp_path: Path,
    source_tree: Path,
) -> None:
    backup, _ = _create_mysql_test_backup(tmp_path, source_tree)
    restore_secret = "restore-secret"
    target = MysqlTarget(
        label="fastapi",
        host="localhost",
        port=3306,
        user="restore",
        database="fastapi_restore_test",
        password=restore_secret,
    )
    commands: list[list[str]] = []
    option_paths: list[Path] = []
    restored_sql: list[bytes] = []

    def restore_runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess:
        commands.append(command)
        option_path = Path(command[1].split("=", 1)[1])
        option_paths.append(option_path)
        assert stat.S_IMODE(option_path.stat().st_mode) == 0o600
        if any(argument.startswith("--database=") for argument in command):
            source = kwargs["stdin"]
            assert hasattr(source, "read")
            restored_sql.append(source.read())
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

        query = next(
            argument.removeprefix("--execute=")
            for argument in command
            if argument.startswith("--execute=")
        )
        if "information_schema.schemata" in query:
            stdout = b"0\n"
        elif query.startswith("CREATE DATABASE"):
            stdout = b""
        else:
            stdout = b"2\t0\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

    result = restore_mysql_backup(backup, [target], runner=restore_runner)

    assert result == {"fastapi": 2}
    assert restored_sql == [b"CREATE TABLE restored (id INT) ENGINE=InnoDB;\n"]
    assert any(
        "CREATE DATABASE IF NOT EXISTS `fastapi_restore_test`" in "\n".join(command)
        for command in commands
    )
    assert all(restore_secret not in "\n".join(command) for command in commands)
    assert option_paths and all(not path.exists() for path in option_paths)


def test_mysql_restore_rejects_unsafe_or_nonempty_target_before_import(
    tmp_path: Path,
    source_tree: Path,
) -> None:
    backup, _ = _create_mysql_test_backup(tmp_path, source_tree)
    unsafe = MysqlTarget(
        label="fastapi",
        host="localhost",
        port=3306,
        user="restore",
        database="production",
        password="secret",
    )
    calls: list[list[str]] = []

    def should_not_run(command: list[str], **_: object) -> subprocess.CompletedProcess:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    with pytest.raises(BackupError, match="必须以 _restore_test 结尾"):
        restore_mysql_backup(backup, [unsafe], runner=should_not_run)
    assert calls == []

    nonempty = MysqlTarget(
        label="fastapi",
        host="localhost",
        port=3306,
        user="restore",
        database="occupied_restore_test",
        password="secret",
    )

    def nonempty_runner(command: list[str], **_: object) -> subprocess.CompletedProcess:
        calls.append(command)
        query = next(
            argument.removeprefix("--execute=")
            for argument in command
            if argument.startswith("--execute=")
        )
        stdout = b"1\n"
        if "information_schema.schemata" not in query:
            stdout = b"1\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

    with pytest.raises(BackupError, match="目标数据库非空"):
        restore_mysql_backup(backup, [nonempty], runner=nonempty_runner)
    assert not any(
        any(argument.startswith("--database=") for argument in command)
        for command in calls
    )
