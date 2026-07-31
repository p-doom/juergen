"""Offline-only launch audit for the short-task pass@1/4/8 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any


PLACEHOLDER_PREFIX = "UNRESOLVED_PIN:"
ZERO_SHA256 = "0" * 64


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _resource_audit(path: Path, *, shard: bool) -> dict[str, Any]:
    value = tomllib.loads(path.read_text(encoding="utf-8"))
    resources = value.get("resources")
    if not isinstance(resources, dict):
        raise ValueError(f"{path}: resources table is missing")
    expected_gpus = 2 if shard else 0
    if resources.get("gpus") != expected_gpus:
        raise ValueError(f"{path}: expected gpus={expected_gpus}")
    if int(resources.get("cpus", 0)) < (16 if shard else 2):
        raise ValueError(f"{path}: CPU allocation is too small")
    extras = resources.get("sbatch_extra")
    if not isinstance(extras, list) or "--ntasks=1" not in extras or "--no-requeue" not in extras:
        raise ValueError(f"{path}: one-task/no-requeue Slurm contract is missing")
    command = value.get("command")
    source = json.dumps(command)
    if shard and not all(
        marker in source
        for marker in (
            "/dev/kvm",
            "CUDA_VISIBLE_DEVICES",
            "paired_eval run",
            "--shard-index",
            "--shard-count",
        )
    ):
        raise ValueError(f"{path}: GPU/model + CPU/KVM command contract drift")
    return {
        "recipe": str(path),
        "gpus": resources["gpus"],
        "cpus": resources["cpus"],
        "mem": resources.get("mem"),
        "time": resources.get("time"),
    }


def audit(
    evaluation_manifest: Path,
    recipes: list[Path],
    aggregate_recipe: Path,
) -> dict[str, Any]:
    manifest = _load_json(evaluation_manifest)
    unresolved: list[str] = []
    for key in (
        "expected_executor_ready_sha256",
        "expected_task_setup_validation_sha256",
    ):
        if manifest.get(key) == ZERO_SHA256:
            unresolved.append(key)
    for key in (
        "expected_executor_ready_artifact_id",
        "expected_task_setup_validation_artifact_id",
    ):
        if str(manifest.get(key, "")).startswith(PLACEHOLDER_PREFIX):
            unresolved.append(key)
    for arm in manifest.get("arms", []):
        name = arm.get("name", "unknown") if isinstance(arm, dict) else "unknown"
        if not isinstance(arm, dict) or str(arm.get("checkpoint", "")).startswith(
            PLACEHOLDER_PREFIX
        ):
            unresolved.append(f"arms.{name}.checkpoint")
        if not isinstance(arm, dict) or arm.get("checkpoint_sha256") == ZERO_SHA256:
            unresolved.append(f"arms.{name}.checkpoint_sha256")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("evaluation manifest runtime binding is missing")
    runtime_source = Path(__file__).with_name("runtime.py")
    observed_runtime_sha = hashlib.sha256(runtime_source.read_bytes()).hexdigest()
    if runtime.get("source_sha256") != observed_runtime_sha:
        raise ValueError("evaluation manifest runtime source hash drift")
    resource_rows = [_resource_audit(path, shard=True) for path in recipes]
    resource_rows.append(_resource_audit(aggregate_recipe, shard=False))
    return {
        "schema_version": 1,
        "status": "blocked_on_explicit_pins" if unresolved else "launch_ready",
        "heldout_access": False,
        "job_submission": False,
        "attempts_per_cell": manifest.get("attempts_per_cell"),
        "pass_at_k": [1, 4, 8],
        "shard_count": len(recipes),
        "runtime_source_sha256": observed_runtime_sha,
        "unresolved": sorted(unresolved),
        "resources": resource_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, action="append", required=True)
    parser.add_argument("--aggregate-recipe", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit(args.evaluation_manifest, args.recipe, args.aggregate_recipe)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["unresolved"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
