"""CLI 环境准备的 AST 与假 dotenv 契约，不执行脚本，也不代表真实存储验收。"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


SCRIPT_ROOT = Path(__file__).resolve().parents[2] / "scripts"
ENTRYPOINTS = (
    ("m3_build_eval_index.py", "main", "get_embed_model"),
    ("m3_run_eval.py", "run", "OfflineRagServiceFactory"),
    ("rag008_answer_eval.py", "run", "get_embed_model"),
    ("rag008_branch_trace.py", "main", "get_embed_model"),
    ("rag018_rank_probe.py", "main", "get_embed_model"),
    ("rag024_latency_probe.py", "main", "get_chat_model"),
)
MODEL_MODULES = (
    "app.utils.factory",
    "app.rag.rag_service",
    "app.rag.vector_store",
    "langchain_chroma",
)
MODEL_CALLS = {
    "get_embed_model", "get_chat_model", "get_vision_model",
    "build_aliyun_chat_model", "OfflineRagServiceFactory", "run_dataset",
    "RagService", "_build_service", "answer_one",
}
FUNCTION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _function(tree: ast.Module, name: str):
    return next(
        node for node in tree.body
        if isinstance(node, FUNCTION_TYPES) and node.name == name
    )


def _read_cli(filename: str, entry_name: str):
    path = SCRIPT_ROOT / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tree, _function(tree, entry_name)


def _is_call(node: ast.AST, names: set[str]) -> bool:
    return isinstance(node, ast.Call) and (
        isinstance(node.func, ast.Name) and node.func.id in names
        or isinstance(node.func, ast.Attribute) and node.func.attr in names
    )


def _is_model_dependency(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").startswith(MODEL_MODULES)
    if isinstance(node, ast.Import):
        return any(alias.name.startswith(MODEL_MODULES) for alias in node.names)
    return _is_call(node, MODEL_CALLS)


def _environment_preparation(entry):
    imports = [
        node for node in entry.body
        if isinstance(node, ast.ImportFrom) and node.module == "dotenv"
    ]
    calls = [
        node for node in entry.body
        if isinstance(node, ast.Expr) and _is_call(node.value, {"load_dotenv"})
    ]
    assert len(imports) == len(calls) == 1, "真实入口必须直接导入并调用一次 load_dotenv"
    imported, prepared = imports[0], calls[0]
    assert [(alias.name, alias.asname) for alias in imported.names] == [
        ("load_dotenv", None)
    ]
    assert imported.end_lineno < prepared.lineno
    call = prepared.value
    assert isinstance(call.func, ast.Name) and call.func.id == "load_dotenv"
    assert not call.args and len(call.keywords) == 1
    keyword = call.keywords[0]
    assert keyword.arg == "override"
    assert isinstance(keyword.value, ast.Constant) and keyword.value.value is False
    return imported, prepared


@pytest.mark.parametrize(("filename", "entry_name", "handoff"), ENTRYPOINTS)
def test_environment_precedes_model_dependencies(filename, entry_name, handoff):
    tree, entry = _read_cli(filename, entry_name)
    _, prepared = _environment_preparation(entry)
    dependencies = [node for node in ast.walk(entry) if _is_model_dependency(node)]
    assert any(_is_call(node, {handoff}) for node in dependencies)
    assert all(prepared.end_lineno < node.lineno for node in dependencies)
    # 模块导入和同步 CLI 分派不能抢在异步入口的准备块之前装配模型。
    module_body = ast.Module(
        body=[node for node in tree.body if not isinstance(node, (*FUNCTION_TYPES, ast.ClassDef))],
        type_ignores=[],
    )
    assert not any(_is_model_dependency(node) for node in ast.walk(module_body))
    if entry_name != "main":
        main = _function(tree, "main")
        assert any(_is_call(node, {entry_name}) for node in ast.walk(main))
        assert not any(_is_model_dependency(node) for node in ast.walk(main))


@pytest.mark.parametrize(("filename", "entry_name", "handoff"), ENTRYPOINTS)
def test_dotenv_call_exists_only_in_real_entrypoint(filename, entry_name, handoff):
    tree, entry = _read_cli(filename, entry_name)
    _, prepared = _environment_preparation(entry)
    names = {"load_dotenv"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "dotenv":
            names.update(
                alias.asname or alias.name for alias in node.names
                if alias.name == "load_dotenv"
            )
    calls = [node for node in ast.walk(tree) if _is_call(node, names)]
    assert calls == [prepared.value], "禁止模块级或其他辅助路径加载 dotenv"


@pytest.mark.parametrize(("filename", "entry_name", "handoff"), ENTRYPOINTS)
def test_preparation_passes_override_false_to_fake_dotenv(
    filename, entry_name, handoff, monkeypatch
):
    _, entry = _read_cli(filename, entry_name)
    imported, prepared = _environment_preparation(entry)
    fake = ModuleType("dotenv")
    fake.load_dotenv = Mock(return_value=True)
    monkeypatch.setitem(sys.modules, "dotenv", fake)
    # 只执行已校验为单一导入和常量参数调用的两条语句，不执行入口或真实配置。
    block = ast.Module(body=[imported, prepared], type_ignores=[])
    exec(compile(block, str(SCRIPT_ROOT / filename), "exec"), {})
    fake.load_dotenv.assert_called_once_with(override=False)


def test_build_index_dry_run_returns_before_environment_preparation():
    _, entry = _read_cli("m3_build_eval_index.py", "main")
    imported, prepared = _environment_preparation(entry)
    branches = [
        node for node in entry.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "args" and node.test.attr == "dry_run"
    ]
    assert len(branches) == 1
    dry_run = branches[0]
    assert dry_run.end_lineno < imported.lineno < prepared.lineno
    returned = dry_run.body[-1]
    assert isinstance(returned, ast.Return), "dry-run 必须在分支内直接返回"
    assert isinstance(returned.value, ast.Constant) and returned.value.value == 0
