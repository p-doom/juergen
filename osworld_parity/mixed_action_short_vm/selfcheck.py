from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .hidden_oracle import evaluate_in_fresh_process
from .manifest import (
    DEVELOPMENT_MANIFEST,
    SEALED_EVALUATION_MANIFEST,
    TRAIN_MANIFEST,
    load_manifest,
    materialize_tasks,
)
from .replay import replay
from .runtime import Episode
from .teacher import (
    NativeTeacherCollector,
    collect_compact_derivative,
    native_gold_actions,
)


class SelfcheckError(RuntimeError):
    pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_build_selfcheck() -> dict[str, Any]:
    train_manifest = load_manifest(TRAIN_MANIFEST)
    development_manifest = load_manifest(DEVELOPMENT_MANIFEST)
    # Only allocation metadata exists for evaluation. Loading this manifest
    # validates its commitments; no hidden payload exists to generate or open.
    sealed_metadata = load_manifest(SEALED_EVALUATION_MANIFEST)
    train = materialize_tasks(train_manifest)
    development = materialize_tasks(development_manifest)
    if {task.parameter_seed for task in train} & {
        task.parameter_seed for task in development
    }:
        raise SelfcheckError("train/development seed overlap")
    if {task.task_sha256 for task in train} & {
        task.task_sha256 for task in development
    }:
        raise SelfcheckError("train/development parameter overlap")

    reset_checks = 0
    replay_reports: list[dict[str, Any]] = []
    teacher_pairs = 0
    for task in development:
        for arm in ("native_absolute_control", "compact_raw_phaseb"):
            episode = Episode(task, arm)
            reset_a = episode.reset()
            first_action = native_gold_actions(task)[0]
            if arm == "compact_raw_phaseb":
                from .teacher import convert_native_actions

                first_action = convert_native_actions(
                    (first_action,), task.geometry.initial_cursor
                )[0]
            episode.step(first_action)
            reset_b = episode.reset()
            if reset_a.reset_fingerprint != reset_b.reset_fingerprint:
                raise SelfcheckError(f"reset fingerprint drift: {task.task_id}/{arm}")
            negative = evaluate_in_fresh_process(
                task, episode._trainer_hidden_snapshot()
            )
            if negative.oracle_status != "ok" or negative.MOUSE_SOLVED:
                raise SelfcheckError(f"reset oracle was not negative: {task.task_id}")
            reset_checks += 1
            for near_miss in (False, True):
                replay_reports.append(
                    replay(task, arm=arm, near_miss=near_miss).as_dict()
                )

        native_episode = Episode(task, "native_absolute_control")
        receipt = native_episode.reset()
        collector = NativeTeacherCollector(task, receipt)
        observation = receipt.observation
        for action in native_gold_actions(task):
            collector.record(observation, action)
            observation = native_episode.step(action).observation
        native_trace = collector.finish()
        compact_trace = collect_compact_derivative(task, native_trace)
        if compact_trace.source_native_trace_sha256 != native_trace.trace_sha256:
            raise SelfcheckError(f"teacher provenance mismatch: {task.task_id}")
        teacher_pairs += 1

    report = {
        "status": "pass",
        "schema_version": 1,
        "suite": "roadmap_3_3_mixed_action_short_vm",
        "execution_class": "cpu_contract_pre_gate_only",
        "scientific_evaluation_executed": False,
        "sealed_evaluation_payload_materialized_or_opened": False,
        "model_executed": False,
        "gpu_used": False,
        "manifests": {
            "train": {
                "records": len(train),
                "payload_sha256": train_manifest.manifest_payload_sha256,
            },
            "development": {
                "records": len(development),
                "payload_sha256": development_manifest.manifest_payload_sha256,
            },
            "sealed_evaluation_metadata": {
                "reserved_slots": len(sealed_metadata.cells),
                "materialized": sealed_metadata.materialized,
                "payload_sha256": sealed_metadata.manifest_payload_sha256,
            },
        },
        "reset_equivalence": {"passing": reset_checks, "total": reset_checks},
        "gold_replays": {
            "passing": sum(row["kind"] == "gold" for row in replay_reports),
            "total": sum(row["kind"] == "gold" for row in replay_reports),
        },
        "near_miss_rejections": {
            "passing": sum(row["kind"] == "near_miss" for row in replay_reports),
            "total": sum(row["kind"] == "near_miss" for row in replay_reports),
        },
        "native_to_compact_teacher_pairs": {
            "passing": teacher_pairs,
            "total": teacher_pairs,
        },
        "replays": replay_reports,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_build_selfcheck()
    except BaseException as exc:
        failure = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "scientific_evaluation_executed": False,
            "sealed_evaluation_payload_materialized_or_opened": False,
            "model_executed": False,
            "gpu_used": False,
        }
        _atomic_json(args.output / "selfcheck.json", failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 2
    _atomic_json(args.output / "selfcheck.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
