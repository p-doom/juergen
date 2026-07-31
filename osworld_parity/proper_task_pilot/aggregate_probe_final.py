#!/usr/bin/env python3
"""Fail-closed final aggregate for the preregistered best-of-8 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from .aggregate_probe import (
        CANONICAL_TASKS_SHA256,
        CHECKPOINT_MANIFEST_SHA256,
        RUNTIME_TREE_SHA256,
        atomic_json,
        load_object,
        payload_seal_valid,
        sha256,
        validate_seed,
    )
except ImportError:  # Direct script execution from its immutable snapshot.
    from aggregate_probe import (
        CANONICAL_TASKS_SHA256,
        CHECKPOINT_MANIFEST_SHA256,
        RUNTIME_TREE_SHA256,
        atomic_json,
        load_object,
        payload_seal_valid,
        sha256,
        validate_seed,
    )


ALL_SEEDS = (101, 211, 307, 401, 503, 601, 701, 809)
OPERATIONAL_CORRECTION_SHA256 = (
    "776120e37ca78e0bce6c8fefd00380b610e531878f57815de79510db948291ea"
)
RECOVERY_AMENDMENT_SHA256 = (
    "9218a63991037566599c41cecedfc12a9cd2765b58281d23b6173c1c8aa3042e"
)
FAILED_503_FAILURE_SHA256 = (
    "fba50a7f3351f5ace7e4132bd1266994f926832f7ea230d82d892e2481763331"
)
FAILED_503_RESULT_SHA256 = (
    "749685dc53b93dfe4b11cc5ddff9a34629904e06dfe21246885a22e1f894911b",
    "2ba27068e885c149f32d57ed15811b1e7c2ef61c61bc2ec3cecbabac9805300c",
)
FAILED_503_TRACE_SHA256 = (
    "3a97475279f13f21303bb3fd5c5a7cc3540a4c79fca203887b29c37da478efd6",
    "c45007c335adc99312d35afb1b6ff3e068363e77f8c1e63a4dad07c42c02bc5b",
)
EXPECTED_GPU_JOBS = {
    "135517": ("seed101", "COMPLETED", "0:0"),
    "135518": ("seed211", "COMPLETED", "0:0"),
    "135519": ("seed307", "COMPLETED", "0:0"),
    "135520": ("seed401", "COMPLETED", "0:0"),
    "135555": ("seed503_failed_infra", "FAILED", "1:0"),
    "135556": ("seed601", "COMPLETED", "0:0"),
    "135558": ("seed701", "COMPLETED", "0:0"),
    "135559": ("seed809", "COMPLETED", "0:0"),
    "135575": ("seed503_continuation", "COMPLETED", "0:0"),
}


def validate_gpu_accounting(path: Path) -> dict[str, Any]:
    payload = load_object(path)
    if not payload_seal_valid(payload):
        raise ValueError("GPU accounting payload seal mismatch")
    if payload.get("schema_version") != 1 or payload.get("status") != "complete":
        raise ValueError("GPU accounting status/schema mismatch")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(EXPECTED_GPU_JOBS):
        raise ValueError("GPU accounting job set mismatch")
    observed: dict[str, tuple[str, str, str]] = {}
    total = 0
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("GPU accounting row is not an object")
        job_id = str(job.get("job_id"))
        elapsed = job.get("elapsed_raw")
        if (
            job.get("gpu_count") != 1
            or not isinstance(elapsed, int)
            or isinstance(elapsed, bool)
            or elapsed < 0
        ):
            raise ValueError(f"invalid GPU accounting row for job {job_id}")
        observed[job_id] = (
            str(job.get("role")), str(job.get("state")), str(job.get("exit_code"))
        )
        total += elapsed
    if observed != EXPECTED_GPU_JOBS:
        raise ValueError(f"GPU accounting identity/state mismatch: {observed}")
    if payload.get("first_half_seconds") != 6400:
        raise ValueError("GPU accounting first-half total changed")
    if payload.get("total_gpu_seconds") != total or total > 14_400:
        raise ValueError(f"GPU cap violated or total mismatch: {total}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "payload_sha256": payload["payload_sha256"],
        "total_gpu_seconds": total,
        "cap_gpu_seconds": 14_400,
        "margin_gpu_seconds": 14_400 - total,
        "jobs": jobs,
    }


def _cell_from_artifacts(
    directory: Path,
    task: dict[str, Any],
    *,
    seed: int,
    task_id: str,
    canonical_index: int,
    session_index: int,
    mode: str,
) -> dict[str, Any]:
    if (
        task.get("mode") != mode
        or task.get("arm") != "relative_r256"
        or task.get("action_format") != "move_rel"
        or task.get("order_index") != canonical_index
        or task.get("task_id") != task_id
        or task.get("infra_valid") is not True
        or task.get("infra_error") is not None
        or task.get("trace_error") is not None
        or task.get("session_id") != f"session-{session_index:06d}"
        or task.get("sampling")
        != {"temperature": 0.7, "top_p": 0.95, "seed": seed}
    ):
        raise ValueError(f"seed {seed} invalid task record at canonical index {canonical_index}")
    result_path = directory / task["result"]
    trace_path = directory / task["trace"]
    task_json_path = Path(task["task_json"])
    if (
        sha256(result_path) != task["result_sha256"]
        or sha256(trace_path) != task["trace_sha256"]
        or sha256(task_json_path) != task["task_json_sha256"]
    ):
        raise ValueError(f"seed {seed} task artifact hash mismatch at {canonical_index}")
    result = load_object(result_path)
    expected_result = {
        key: value
        for key, value in task.items()
        if key not in {"result", "result_sha256"}
    }
    trace = load_object(trace_path)
    if (
        result != expected_result
        or trace.get("errors") != []
        or trace.get("is_completed") is not True
        or trace.get("state", {}).get("infra_valid") is not True
        or trace.get("state", {}).get("task_reward") != task.get("raw_reward")
    ):
        raise ValueError(f"seed {seed} task artifact content mismatch at {canonical_index}")
    return {
        "seed": seed,
        "task_id": task_id,
        "raw_reward": task["raw_reward"],
        "full_success": task["full_success"],
        "parse_errors": task["parse_errors"],
        "result_sha256": task["result_sha256"],
        "trace_sha256": task["trace_sha256"],
        "source_attempt": "failed_parent_valid_cell" if canonical_index < 2 else "continuation",
    }


def validate_seed503_recovery(
    parent: Path, recovery: Path, canonical_ids: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failure_path = parent / "failure.json"
    failure = load_object(failure_path)
    if (
        sha256(failure_path) != FAILED_503_FAILURE_SHA256
        or failure.get("status") != "failed"
        or failure.get("artifact_valid") is not False
        or failure.get("error_type") != "TaskError"
        or "infrastructure-invalid" not in str(failure.get("message"))
        or "no byte screenshot" not in str(failure.get("message"))
        or (parent / "run_manifest.json").exists()
    ):
        raise ValueError("seed503 parent failure evidence mismatch")
    parent_results = sorted((parent / "tasks").glob("*/result.json"))
    expected_parent_results = [
        parent / "tasks" / f"{index:02d}_{canonical_ids[index]}" / "result.json"
        for index in (0, 1)
    ]
    if parent_results != expected_parent_results:
        raise ValueError("seed503 parent published cells changed")
    cells: list[dict[str, Any]] = []
    for index in (0, 1):
        result_path = expected_parent_results[index]
        trace_path = result_path.with_name("trace.json")
        if (
            sha256(result_path) != FAILED_503_RESULT_SHA256[index]
            or sha256(trace_path) != FAILED_503_TRACE_SHA256[index]
        ):
            raise ValueError(f"seed503 parent cell {index} seal mismatch")
        result = load_object(result_path)
        task = {
            **result,
            "result": str(result_path.relative_to(parent)),
            "result_sha256": sha256(result_path),
        }
        cells.append(
            _cell_from_artifacts(
                parent,
                task,
                seed=503,
                task_id=canonical_ids[index],
                canonical_index=index,
                session_index=index + 1,
                mode="probe_seed",
            )
        )

    manifest_path = recovery / "run_manifest.json"
    manifest = load_object(manifest_path)
    if not payload_seal_valid(manifest):
        raise ValueError("seed503 continuation payload seal mismatch")
    expected = {
        "status": "complete",
        "artifact_valid": True,
        "all_infra_valid": True,
        "benchmark_data": True,
        "mode": "probe_continuation",
        "arm": "relative_r256",
        "action_format": "move_rel",
        "task_count_expected": 10,
        "task_count_completed": 10,
        "task_selection_sha256": CANONICAL_TASKS_SHA256,
        "reverse_tasks": False,
        "task_ids": canonical_ids[2:],
        "canonical_task_indices": list(range(2, 12)),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    continuation = manifest.get("continuation_contract", {})
    preflight_parent = manifest.get("preflight", {}).get("continuation_parent", {})
    rollout = manifest.get("rollout_config", {})
    provenance = manifest.get("runtime_provenance_verification", {})
    if (
        mismatches
        or continuation.get("start_index") != 2
        or preflight_parent.get("path") != str(parent.resolve())
        or preflight_parent.get("failure_sha256") != FAILED_503_FAILURE_SHA256
        or preflight_parent.get("reused_valid_canonical_indices") != [0, 1]
        or preflight_parent.get("excluded_infra_attempt_canonical_index") != 2
        or preflight_parent.get("result_sha256") != list(FAILED_503_RESULT_SHA256)
        or preflight_parent.get("trace_sha256") != list(FAILED_503_TRACE_SHA256)
        or rollout.get("temperature") != 0.7
        or rollout.get("top_p") != 0.95
        or rollout.get("sampling_seed") != 503
        or rollout.get("fresh_vm_per_rollout") is not True
        or provenance.get("expected_tree_sha256") != RUNTIME_TREE_SHA256
        or provenance.get("unchanged_before_write") is not True
        or provenance.get("unchanged_after_candidate_write") is not True
        or manifest.get("checkpoint", {}).get("manifest_sha256")
        != CHECKPOINT_MANIFEST_SHA256
        or manifest.get("preflight", {}).get("no_heldout_use") is not True
        or manifest.get("preflight", {}).get("probe_gate", {}).get(
            "materiality_gate_open"
        )
        is not True
    ):
        raise ValueError(f"seed503 continuation contract mismatch: {mismatches}")
    recovery_results = sorted(recovery.glob("tasks/*/result.json"))
    if len(recovery_results) != 10:
        raise ValueError("seed503 continuation does not contain exactly 10 results")
    for local_index, task in enumerate(manifest.get("tasks") or []):
        canonical_index = local_index + 2
        cells.append(
            _cell_from_artifacts(
                recovery,
                task,
                seed=503,
                task_id=canonical_ids[canonical_index],
                canonical_index=canonical_index,
                session_index=local_index + 1,
                mode="probe_continuation",
            )
        )
    if len(cells) != 12:
        raise ValueError(f"seed503 merged recovery has {len(cells)} cells")
    return manifest, cells


def _binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(
        math.comb(n, i) * p**i * (1.0 - p) ** (n - i)
        for i in range(k + 1)
    )


def _binomial_upper_tail(k: int, n: int, p: float) -> float:
    return sum(
        math.comb(n, i) * p**i * (1.0 - p) ** (n - i)
        for i in range(k, n + 1)
    )


def exact_binomial_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval without a scipy dependency."""
    if not (0 <= successes <= trials and trials > 0 and 0 < confidence < 1):
        raise ValueError("invalid exact-binomial interval arguments")
    tail = (1.0 - confidence) / 2.0
    if successes == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if _binomial_upper_tail(successes, trials, mid) < tail:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2.0
    if successes == trials:
        upper = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if _binomial_cdf(successes, trials, mid) > tail:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2.0
    return lower, upper


def _sealed(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "payload_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--operational-correction", type=Path, required=True)
    parser.add_argument("--gpu-accounting", type=Path)
    parser.add_argument("--recovery-amendment", type=Path)
    for seed in ALL_SEEDS:
        parser.add_argument(f"--seed{seed}", type=Path, required=True)
    parser.add_argument("--seed503-recovery", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reasons: list[str] = []
    cells: list[dict[str, Any]] = []
    manifests: dict[str, dict[str, Any]] = {}
    gpu_accounting: dict[str, Any] | None = None
    canonical_ids: list[str] = []
    try:
        if sha256(args.operational_correction) != OPERATIONAL_CORRECTION_SHA256:
            raise ValueError("operational-correction hash mismatch")
        if args.gpu_accounting is not None:
            gpu_accounting = validate_gpu_accounting(args.gpu_accounting)
        if args.seed503_recovery is not None:
            if (
                args.recovery_amendment is None
                or sha256(args.recovery_amendment) != RECOVERY_AMENDMENT_SHA256
            ):
                raise ValueError("seed503 recovery amendment hash mismatch")
        if sha256(args.tasks) != CANONICAL_TASKS_SHA256:
            raise ValueError("canonical task-list hash mismatch")
        canonical_ids = args.tasks.read_text(encoding="utf-8").splitlines()
        if len(canonical_ids) != 12 or len(set(canonical_ids)) != 12:
            raise ValueError("canonical task list is not 12 unique IDs")
        for seed in ALL_SEEDS:
            directory = getattr(args, f"seed{seed}")
            if seed == 503 and args.seed503_recovery is not None:
                manifest, seed_cells = validate_seed503_recovery(
                    directory, args.seed503_recovery, canonical_ids
                )
                manifests[str(seed)] = {
                    "merge_policy": "sealed_valid_parent_cells_0_1_plus_continuation_2_11",
                    "failed_parent_directory": str(directory.resolve()),
                    "failed_parent_failure_sha256": FAILED_503_FAILURE_SHA256,
                    "continuation_directory": str(args.seed503_recovery.resolve()),
                    "continuation_manifest_sha256": sha256(
                        args.seed503_recovery / "run_manifest.json"
                    ),
                    "continuation_payload_sha256": manifest["payload_sha256"],
                    "excluded_infrastructure_attempts": 1,
                }
            else:
                manifest, seed_cells = validate_seed(
                    directory, seed=seed, canonical_ids=canonical_ids
                )
                manifests[str(seed)] = {
                    "directory": str(directory.resolve()),
                    "manifest_sha256": sha256(directory / "run_manifest.json"),
                    "payload_sha256": manifest["payload_sha256"],
                }
            cells.extend(seed_cells)
        expected_keys = {
            (seed, task_id) for seed in ALL_SEEDS for task_id in canonical_ids
        }
        observed_keys = {(cell["seed"], cell["task_id"]) for cell in cells}
        if observed_keys != expected_keys or len(cells) != 96:
            raise ValueError("best-of-8 task/seed crossing is incomplete or duplicated")
    except Exception as exc:  # noqa: BLE001 - must publish incomplete
        reasons.append(f"{type(exc).__name__}: {exc}")

    positive = [cell for cell in cells if cell.get("full_success") is True]
    rejection = [
        cell
        for cell in cells
        if cell.get("raw_reward") == 0.0
        and cell.get("full_success") is False
        and cell.get("parse_errors") == 0
    ]
    per_task: list[dict[str, Any]] = []
    if not reasons and len(cells) == 96:
        for task_id in canonical_ids:
            task_cells = [cell for cell in cells if cell["task_id"] == task_id]
            successes = sum(cell["full_success"] is True for cell in task_cells)
            lower, upper = exact_binomial_interval(successes, len(ALL_SEEDS))
            per_task.append(
                {
                    "task_id": task_id,
                    "success_count": successes,
                    "attempt_count": len(ALL_SEEDS),
                    "success_rate": successes / len(ALL_SEEDS),
                    "clopper_pearson_95": {
                        "lower": round(lower, 12),
                        "upper": round(upper, 12),
                    },
                    "best_of_8_success": successes > 0,
                    "successful_seeds": [
                        cell["seed"]
                        for cell in task_cells
                        if cell["full_success"] is True
                    ],
                }
            )

    positive_tasks = sorted({cell["task_id"] for cell in positive})
    rejection_tasks = sorted({cell["task_id"] for cell in rejection})
    yield_gate_open = (
        not reasons
        and len(cells) == 96
        and len(positive) >= 12
        and len(positive_tasks) >= 4
        and len(rejection) >= 12
        and len(rejection_tasks) >= 4
    )
    status = "complete" if not reasons and len(cells) == 96 else "incomplete"
    payload = _sealed(
        {
            "schema_version": 1,
            "status": status,
            "probe_complete": status == "complete",
            "expected_cells": 96,
            "valid_cells": len(cells),
            "seeds": list(ALL_SEEDS),
            "canonical_tasks_sha256": CANONICAL_TASKS_SHA256,
            "operational_correction_sha256": OPERATIONAL_CORRECTION_SHA256,
            "recovery_amendment_sha256": (
                RECOVERY_AMENDMENT_SHA256
                if args.seed503_recovery is not None
                else None
            ),
            "runtime_tree_sha256": RUNTIME_TREE_SHA256,
            "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
            "gpu_accounting": gpu_accounting,
            "success_count": len(positive),
            "successful_task_count": len(positive_tasks),
            "successful_tasks": positive_tasks,
            "parse_valid_reward_zero_count": len(rejection),
            "parse_valid_reward_zero_task_count": len(rejection_tasks),
            "parse_valid_reward_zero_tasks": rejection_tasks,
            "best_of_8_success_task_count": sum(
                row["best_of_8_success"] for row in per_task
            ),
            "yield_gate": {
                "minimum_successes": 12,
                "minimum_success_tasks": 4,
                "minimum_parse_valid_reward_zero": 12,
                "minimum_parse_valid_reward_zero_tasks": 4,
                "open": yield_gate_open,
            },
            "per_task": per_task,
            "inputs": manifests,
            "cells": cells,
            "reasons": reasons,
            "infrastructure_attempt_policy": (
                "exclude failed seed503 screenshot-transport attempt before a result was "
                "published; require exactly one valid result for each intended seed/task cell"
            ),
            "gpu_training_authorized": False,
        }
    )
    atomic_json(args.output / "probe_manifest.json", payload)
    print(
        f"best-of-8 aggregate status={status} valid_cells={len(cells)} "
        f"successes={len(positive)} yield_gate_open={yield_gate_open}",
        flush=True,
    )
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
