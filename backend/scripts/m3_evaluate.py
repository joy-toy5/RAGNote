"""M3 冻结数据校验、qrels 编译与纯离线评分 CLI。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from app.evaluation.dataset import dataset_fingerprint, load_dataset
from app.evaluation.qrels import compile_qrels
from app.evaluation.reporting import evaluate_run, sha256_file, write_report_bundle
from app.evaluation.runner import load_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 M3 离线 RAG 评测工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="校验冻结数据与哈希")
    validate.add_argument("--dataset", type=Path, required=True)

    compile_command = subparsers.add_parser("compile-qrels", help="编译 chunk qrels")
    compile_command.add_argument("--dataset", type=Path, required=True)
    compile_command.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="评分冻结 run 产物")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    dataset = load_dataset(arguments.dataset)
    if arguments.command == "validate":
        print(dataset_fingerprint(dataset))
        return 0

    qrels = compile_qrels(dataset)
    if arguments.command == "compile-qrels":
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(qrels.to_json())
        return 0

    run = load_run(arguments.run)
    report = evaluate_run(
        dataset,
        qrels,
        run,
        run_file_sha256=sha256_file(arguments.run),
    )
    write_report_bundle(
        report,
        qrels,
        arguments.output_dir,
    )
    return 0 if report.hard_gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
