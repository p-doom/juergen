from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .aggregate import aggregate_results, load_jsonl
from .manifest import EvaluationManifest, load_evaluation_manifest
from .planning import build_plan
from .readiness import ConsumedReadiness, consume_executor_ready
from .runner import PairedEvaluationRunner, write_jsonl_atomic


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)


def _shard(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)


def _load_factory(
    value: str,
) -> Callable[[EvaluationManifest, ConsumedReadiness], Any]:
    module_name, separator, attribute = value.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("runtime factory must be module.path:callable")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("runtime factory is not callable")
    return factory


def _write_json(path: Path | None, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(payload, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Development-only paired proper-VM model evaluation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    _common(validate)

    plan_parser = commands.add_parser("plan")
    _common(plan_parser)
    _shard(plan_parser)
    plan_parser.add_argument("--output", type=Path)

    run = commands.add_parser("run")
    _common(run)
    _shard(run)
    run.add_argument("--executor-ready", type=Path, required=True)
    run.add_argument("--runtime-factory", required=True)
    run.add_argument("--output", type=Path, required=True)

    aggregate = commands.add_parser("aggregate")
    _common(aggregate)
    _shard(aggregate)
    aggregate.add_argument("--results", type=Path, action="append", required=True)
    aggregate.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    manifest = load_evaluation_manifest(
        args.evaluation_manifest,
        args.task_manifest,
    )
    if args.command == "validate":
        _write_json(
            None,
            {
                "status": "valid",
                "suite": manifest.suite,
                "tasks": len(manifest.tasks),
                "development_only": True,
                "comparison_label": manifest.comparison_label,
            },
        )
        return 0

    plan = build_plan(
        manifest,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    if args.command == "plan":
        _write_json(
            args.output,
            {
                "schema_version": 1,
                "suite": manifest.suite,
                "development_only": True,
                "scored_execution": False,
                "trials": [asdict(trial) for trial in plan],
            },
        )
        return 0
    if args.command == "run":
        # Consume and bind readiness before importing a runtime factory.  This
        # prevents a factory with VM/model startup side effects from bypassing
        # the pre-scoring integration gate.
        readiness = consume_executor_ready(
            args.executor_ready,
            expected_sha256=manifest.expected_executor_ready_sha256,
        )
        factory = _load_factory(args.runtime_factory)
        runtime = factory(manifest, readiness)
        rows = PairedEvaluationRunner(manifest, readiness, runtime).run(plan)
        write_jsonl_atomic(args.output, rows)
        return 0
    if args.command == "aggregate":
        report = aggregate_results(manifest, plan, load_jsonl(args.results))
        _write_json(args.output, report)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
