"""`source_fingerprint` 的采集范围门禁（RAG-017）。

拒答与摘要行为由 `app/prompt/*.txt` 驱动，而非只由 `.py` / `.yaml` 驱动。
`rag_summarize.txt` 第 5 条就是拒答开关：改它可以把 `false_answer_rate`
从 0.000 推到 1.000。若该目录不在指纹内，冻结 run 之间无法区分这种变更，
质量门禁在这一维度上有盲区。本文件锁住"prompt 变则指纹变"这条耦合。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.m3_run_eval import FINGERPRINT_PATTERNS, source_fingerprint

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _tree(root: Path) -> None:
    """搭一棵最小的、能被全部 pattern 命中的源码树。"""
    (root / "app" / "config").mkdir(parents=True)
    (root / "app" / "prompt").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    (root / "app" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (root / "app" / "config" / "chroma.yaml").write_text("k: 5\n", encoding="utf-8")
    (root / "scripts" / "m3_run_eval.py").write_text("y = 2\n", encoding="utf-8")
    (root / "app" / "prompt" / "rag_summarize.txt").write_text(
        "要求：\n5. 证据不足时输出 [[NO_ANSWER]]\n", encoding="utf-8"
    )


def test_prompt_directory_is_in_the_collected_patterns() -> None:
    assert "app/prompt/*.txt" in FINGERPRINT_PATTERNS


def test_editing_the_refusal_prompt_changes_the_fingerprint(tmp_path: Path) -> None:
    """RAG-017 的验收条件：改 prompt 任一字节，指纹必变。"""
    _tree(tmp_path)
    before = source_fingerprint(tmp_path)

    prompt = tmp_path / "app" / "prompt" / "rag_summarize.txt"
    prompt.write_text(
        prompt.read_text(encoding="utf-8") + "6. 补一条\n", encoding="utf-8"
    )

    assert source_fingerprint(tmp_path) != before


def test_a_single_byte_in_any_prompt_file_changes_the_fingerprint(
    tmp_path: Path,
) -> None:
    """不止 rag_summarize.txt——目录下任一 prompt 都算行为源。"""
    _tree(tmp_path)
    other = tmp_path / "app" / "prompt" / "main_prompt.txt"
    other.write_text("a\n", encoding="utf-8")
    before = source_fingerprint(tmp_path)

    other.write_text("b\n", encoding="utf-8")

    assert source_fingerprint(tmp_path) != before


def test_fingerprint_is_stable_under_repetition(tmp_path: Path) -> None:
    _tree(tmp_path)
    assert source_fingerprint(tmp_path) == source_fingerprint(tmp_path)


def test_fingerprint_binds_path_not_only_content(tmp_path: Path) -> None:
    """同样内容换个文件名必须换指纹，否则改名可以绕过绑定。"""
    _tree(tmp_path)
    prompt = tmp_path / "app" / "prompt" / "rag_summarize.txt"
    body = prompt.read_text(encoding="utf-8")
    before = source_fingerprint(tmp_path)

    prompt.unlink()
    (tmp_path / "app" / "prompt" / "renamed.txt").write_text(body, encoding="utf-8")

    assert source_fingerprint(tmp_path) != before


def test_pycache_is_excluded(tmp_path: Path) -> None:
    _tree(tmp_path)
    before = source_fingerprint(tmp_path)
    cache = tmp_path / "app" / "__pycache__"
    cache.mkdir()
    (cache / "mod.cpython-312.pyc").write_bytes(b"\x00\x01")
    assert source_fingerprint(tmp_path) == before


def test_non_prompt_text_files_outside_the_patterns_are_ignored(
    tmp_path: Path,
) -> None:
    """采集范围必须是明确列举的，不能把任意 .txt 都吞进来。"""
    _tree(tmp_path)
    before = source_fingerprint(tmp_path)
    (tmp_path / "notes.txt").write_text("随手记\n", encoding="utf-8")
    (tmp_path / "app" / "readme.txt").write_text("说明\n", encoding="utf-8")
    assert source_fingerprint(tmp_path) == before


@pytest.mark.parametrize(
    "name",
    [
        "rag_summarize.txt",
        "main_prompt.txt",
        "reorder_prompt.txt",
    ],
)
def test_real_prompt_files_are_actually_collected(name: str) -> None:
    """对真实工作树断言：这些 prompt 确实落在采集范围内。"""
    assert (BACKEND_ROOT / "app" / "prompt" / name).is_file()
    collected = {
        path.name
        for pattern in FINGERPRINT_PATTERNS
        for path in BACKEND_ROOT.glob(pattern)
        if path.is_file()
    }
    assert name in collected


def test_real_tree_fingerprint_changes_when_prompt_changes(tmp_path: Path) -> None:
    """在真实 app/prompt 的副本上验证，避免只测构造树。"""
    root = tmp_path / "backend"
    (root / "app").mkdir(parents=True)
    shutil.copytree(BACKEND_ROOT / "app" / "prompt", root / "app" / "prompt")
    (root / "app" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    before = source_fingerprint(root)

    target = root / "app" / "prompt" / "rag_summarize.txt"
    target.write_bytes(target.read_bytes() + b"\n")

    assert source_fingerprint(root) != before
