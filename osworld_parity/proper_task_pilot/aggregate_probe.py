#!/usr/bin/env python3
"""Fail-closed first-half aggregate for the preregistered move_rel probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CANONICAL_TASKS_SHA256 = (
    "53c6750ec8bbc9d1705ea770bcc1c8216c028a88d008d45c040802e1fd96a100"
)
RUNTIME_TREE_SHA256 = (
    "abc961f70ad2278ab5edc2822759ba1f00e4e6106dcf0ee944e5e14283aa0bb3"
)
CHECKPOINT_MANIFEST_SHA256 = (
    "996adabfcf22cf537f5727aa330d2468497bb5513b71f5db18de89193eaed483"
)
FIRST_HALF_SEEDS = (101, 211, 307, 401)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def payload_seal_valid(payload: dict[str, Any]) -> bool:
    candidate = dict(payload)
    observed = candidate.pop("payload_sha256", None)
    canonical = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
    return observed == hashlib.sha256(canonical.encode()).hexdigest()


def validate_seed(
    directory: Path, *, seed: int, canonical_ids: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = directory / "run_manifest.json"
    manifest = load_object(manifest_path)
    if not payload_seal_valid(manifest):
        raise ValueError(f"payload seal mismatch: {manifest_path}")
    expected = {
        "status": "complete",
        "artifact_valid": True,
        "all_infra_valid": True,
        "benchmark_data": True,
        "mode": "probe_seed",
        "arm": "relative_r256",
        "action_format": "move_rel",
        "task_count_expected": 12,
        "task_count_completed": 12,
        "task_selection_sha256": CANONICAL_TASKS_SHA256,
        "reverse_tasks": False,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"seed {seed} manifest mismatch: {mismatches}")
    if manifest.get("task_ids") != canonical_ids:
        raise ValueError(f"seed {seed} task order mismatch")
    sampling = manifest.get("rollout_config", {})
    if (
        sampling.get("temperature") != 0.7
        or sampling.get("top_p") != 0.95
        or sampling.get("sampling_seed") != seed
        or sampling.get("fresh_vm_per_rollout") is not True
        or sampling.get("max_sessions") != 1
        or sampling.get("max_rollouts_per_session") != 1
    ):
        raise ValueError(f"seed {seed} sampling/fresh-VM contract mismatch")
    provenance = manifest.get("runtime_provenance_verification", {})
    if (
        provenance.get("expected_tree_sha256") != RUNTIME_TREE_SHA256
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
        raise ValueError(f"seed {seed} frozen provenance mismatch")

    cells: list[dict[str, Any]] = []
    for index, task in enumerate(manifest.get("tasks") or []):
        task_id = canonical_ids[index]
        if (
            task.get("order_index") != index
            or task.get("task_id") != task_id
            or task.get("infra_valid") is not True
            or task.get("infra_error") is not None
            or task.get("trace_error") is not None
            or task.get("session_id") != f"session-{index + 1:06d}"
            or task.get("sampling")
            != {"temperature": 0.7, "top_p": 0.95, "seed": seed}
        ):
            raise ValueError(f"seed {seed} invalid task record at index {index}")
        result_path = directory / task["result"]
        trace_path = directory / task["trace"]
        task_json_path = Path(task["task_json"])
        if (
            sha256(result_path) != task["result_sha256"]
            or sha256(trace_path) != task["trace_sha256"]
            or sha256(task_json_path) != task["task_json_sha256"]
        ):
            raise ValueError(f"seed {seed} task artifact hash mismatch at {index}")
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
            raise ValueError(f"seed {seed} task artifact content mismatch at {index}")
        cells.append(
            {
                "seed": seed,
                "task_id": task_id,
                "raw_reward": task["raw_reward"],
                "full_success": task["full_success"],
                "parse_errors": task["parse_errors"],
                "result_sha256": task["result_sha256"],
                "trace_sha256": task["trace_sha256"],
            }
        )
    if len(cells) != 12:
        raise ValueError(f"seed {seed} has {len(cells)} cells, expected 12")
    return manifest, cells


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--seed101", type=Path, required=True)
    parser.add_argument("--seed211", type=Path, required=True)
    parser.add_argument("--seed307", type=Path, required=True)
    parser.add_argument("--seed401", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reasons: list[str] = []
    cells: list[dict[str, Any]] = []
    manifests: dict[str, dict[str, Any]] = {}
    try:
        if sha256(args.tasks) != CANONICAL_TASKS_SHA256:
            raise ValueError("canonical task-list hash mismatch")
        canonical_ids = args.tasks.read_text(encoding="utf-8").splitlines()
        if len(canonical_ids) != 12 or len(set(canonical_ids)) != 12:
            raise ValueError("canonical task list is not 12 unique IDs")
        for seed in FIRST_HALF_SEEDS:
            directory = getattr(args, f"seed{seed}")
            manifest, seed_cells = validate_seed(
                directory, seed=seed, canonical_ids=canonical_ids
            )
            manifests[str(seed)] = {
                "directory": str(directory.resolve()),
                "manifest_sha256": sha256(directory / "run_manifest.json"),
                "payload_sha256": manifest["payload_sha256"],
            }
            cells.extend(seed_cells)
        cell_keys = {(cell["seed"], cell["task_id"]) for cell in cells}
        expected_keys = {
            (seed, task_id) for seed in FIRST_HALF_SEEDS for task_id in canonical_ids
        }
        if cell_keys != expected_keys or len(cells) != 48:
            raise ValueError("first-half task/seed crossing is incomplete or duplicated")
    except Exception as exc:  # noqa: BLE001 - aggregate must publish incomplete
        reasons.append(f"{type(exc).__name__}: {exc}")

    successes = [cell for cell in cells if cell["full_success"] is True]
    decision = (
        "incomplete"
        if reasons or len(cells) != 48
        else "continue"
        if successes
        else "stop_zero_success"
    )
    payload = {
        "schema_version": 1,
        "status": decision,
        "decision": decision,
        "expected_cells": 48,
        "valid_cells": len(cells),
        "successes": len(successes),
        "successful_tasks": sorted({cell["task_id"] for cell in successes}),
        "first_half_seeds": list(FIRST_HALF_SEEDS),
        "canonical_tasks_sha256": CANONICAL_TASKS_SHA256,
        "runtime_tree_sha256": RUNTIME_TREE_SHA256,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "reasons": reasons,
        "inputs": manifests,
        "cells": cells,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["payload_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    atomic_json(args.output / "decision.json", payload)
    print(
        f"probe first-half decision={decision} valid_cells={len(cells)} "
        f"successes={len(successes)}",
        flush=True,
    )
    return 0 if decision in {"continue", "stop_zero_success"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
