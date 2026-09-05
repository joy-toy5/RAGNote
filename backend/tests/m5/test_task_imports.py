"""在独立解释器中验证模型导入顺序，不依赖 pytest 已预热的 sys.modules。"""

import itertools
import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "order",
    list(
        itertools.permutations(
            ("app.models", "app.indexing.models", "app.tasking.models")
        )
    )
    + [("app.tasking.contracts", "app.tasking.repository", "app.db.schema_gate")],
)
def test_cold_model_registration_has_one_complete_metadata(
    backend_root, tmp_path, order
):
    script = r"""
import importlib
import importlib.abc
import json
import socket
import sys

sys.path.insert(0, sys.argv[1])

def forbidden(*args, **kwargs):
    raise AssertionError("冷导入禁止环境/网络副作用")

socket.create_connection = forbidden
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
import dotenv
dotenv.load_dotenv = forbidden

blocked = {"app.db.db_config", "app.db.redis_config", "chromadb", "modelscope", "sentence_transformers", "main"}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("冷导入触及运行时依赖：" + fullname)

sys.meta_path.insert(0, Guard())
for name in json.loads(sys.argv[2]):
    importlib.import_module(name)
from app.db import schema_gate
from app.models import Base
from app.models.chat_history import Base as LegacyBase
from app.indexing.models import ContentBlob
from app.tasking.models import BackgroundTask, TaskAttempt
assert Base is LegacyBase
assert all(model.metadata is Base.metadata for model in (ContentBlob, BackgroundTask, TaskAttempt))
assert set(Base.metadata.tables) == {
    "chat_messages", "chat_sessions", "notes", "review_records", "content_blobs",
    "documents", "document_revisions", "index_versions", "index_chunks",
    "background_tasks", "task_attempts",
}
assert len(Base.metadata.sorted_tables) == 11
assert schema_gate.EXPECTED_SCHEMA_REVISION == "0003_task_lifecycle"
assert blocked.isdisjoint(sys.modules)
print("cold-import-ok")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            script,
            str(backend_root),
            json.dumps(order),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "cold-import-ok"
