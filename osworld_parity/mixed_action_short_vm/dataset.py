from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Literal

from .manifest import load_authorized_tasks, payload_sha256
from .runtime import Episode
from .teacher import (
    NativeTeacherCollector,
    collect_compact_derivative,
    native_gold_actions,
)


class DatasetError(RuntimeError):
    pass


def _trace_row(trace: Any) -> dict[str, Any]:
    trace.verify()
    return {**trace.unsigned_payload(), "trace_sha256": trace.trace_sha256}


def build_teacher_pairs(
    output: Path, *, split: Literal["train", "development"]
) -> dict[str, Any]:
    """Build a CPU contract artifact: native teacher first, compact derivative.

    The frame references are deterministic stand-ins emitted by the contract
    backend. Real VM collection uses the same ``NativeTeacherCollector`` with
    screenshot-backed observations. This artifact is explicitly non-scientific
    and never authorizes training/model launch by itself.
    """
    if output.exists() and any(output.iterdir()):
        raise DatasetError(f"refusing to overwrite non-empty output: {output}")
    stage = output.with_name(
        f".{output.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    )
    stage.mkdir(parents=True, exist_ok=False)
    tasks = load_authorized_tasks(split)
    native_rows: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    try:
        for task in tasks:
            episode = Episode(task, "native_absolute_control")
            receipt = episode.reset()
            collector = NativeTeacherCollector(task, receipt)
            observation = receipt.observation
            for action in native_gold_actions(task):
                collector.record(observation, action)
                result = episode.step(action)
                observation = result.observation
            native = collector.finish()
            compact = collect_compact_derivative(task, native)
            if compact.source_native_trace_sha256 != native.trace_sha256:
                raise DatasetError(f"format provenance mismatch: {task.task_id}")
            native_rows.append(_trace_row(native))
            compact_rows.append(_trace_row(compact))
        artifacts = (
            ("native_absolute.jsonl", native_rows),
            ("compact_raw.jsonl", compact_rows),
        )
        for name, rows in artifacts:
            (stage / name).write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
        pair_fingerprints = [
            payload_sha256(
                {
                    "task_id": native["task_id"],
                    "task_sha256": native["task_sha256"],
                    "reset_fingerprint": native["reset_fingerprint"],
                    "native_trace_sha256": native["trace_sha256"],
                    "compact_trace_sha256": compact["trace_sha256"],
                }
            )
            for native, compact in zip(native_rows, compact_rows, strict=True)
        ]
        manifest = {
            "artifact_type": "roadmap_3_3_teacher_collection_contract",
            "schema_version": 1,
            "status": "cpu_contract_only_not_scientific_not_training_authorization",
            "split": split,
            "record_count_per_format": len(tasks),
            "formats": ["native_absolute_control", "compact_raw_phaseb"],
            "native_first_conversion": True,
            "matched_pair_fingerprints": pair_fingerprints,
            "sealed_evaluation_payload_accessed": False,
            "model_executed": False,
            "gpu_used": False,
        }
        (stage / "artifact_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output.mkdir(parents=True, exist_ok=True)
        for child in sorted(stage.iterdir()):
            os.replace(child, output / child.name)
        stage.rmdir()
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "development"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_teacher_pairs(args.output, split=args.split)
    except DatasetError as exc:
        print(f"FATAL teacher artifact: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
