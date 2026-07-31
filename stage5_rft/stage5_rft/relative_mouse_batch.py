"""Seal fixed-budget relative-mouse rollout shards before learner conversion."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from stage5_rft.util import ContractError, atomic_write_json, read_json


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal_relative_mouse_batch(
    *,
    shard_dirs: list[str | Path],
    output_dir: str | Path,
    expected_tasks: int,
    expected_k: int,
    minimum_accepted: int,
    expected_train_seed: int,
) -> dict[str, Any]:
    if len(shard_dirs) < 2:
        raise ContractError("multi-GPU rollout batch requires at least two shards")
    if expected_tasks < 1 or expected_k < 1 or minimum_accepted < 1:
        raise ContractError("batch budget values must be positive")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    task_keys: set[str] = set()
    source_shards = []
    total_attempts = total_accepted = total_errors = total_tasks = 0
    expected_per_shard = expected_tasks // len(shard_dirs)
    if expected_per_shard * len(shard_dirs) != expected_tasks:
        raise ContractError("expected_tasks must divide evenly across shards")

    for expected_index, raw_dir in enumerate(shard_dirs):
        root = Path(raw_dir)
        stats_path = root / f"stats_shard{expected_index}.json"
        rollout_path = root / f"rollouts_shard{expected_index}.jsonl"
        stats = read_json(stats_path)
        summary = stats.get("summary")
        per_task = stats.get("per_task")
        if not isinstance(summary, dict) or not isinstance(per_task, list):
            raise ContractError(f"shard {expected_index} has invalid stats schema")
        required = {
            "status": "OK",
            "n_tasks": expected_per_shard,
            "k": expected_k,
            "n_rollouts_ok": expected_per_shard * expected_k,
            "n_rollouts_err": 0,
            "error_rate": 0.0,
            "pool": "train",
            "train_seed": expected_train_seed,
            "shard_id": expected_index,
            "n_shards": len(shard_dirs),
        }
        for name, expected in required.items():
            if summary.get(name) != expected:
                raise ContractError(
                    f"shard {expected_index} {name}={summary.get(name)!r}, expected {expected!r}"
                )
        if len(per_task) != expected_per_shard:
            raise ContractError(f"shard {expected_index} per_task count is inconsistent")
        for task in per_task:
            key = str(task.get("task_key", ""))
            if not key or key in task_keys:
                raise ContractError(f"duplicate or empty task key across shards: {key!r}")
            task_keys.add(key)
            if int(task.get("n", -1)) != expected_k:
                raise ContractError(f"task {key} does not have fixed k={expected_k}")
        if not rollout_path.is_file():
            raise ContractError(f"shard rollout file is missing: {rollout_path}")
        copied = out / f"rollouts_shard{expected_index}.jsonl"
        shutil.copyfile(rollout_path, copied)
        source_shards.append(
            {
                "shard_id": expected_index,
                "stats_path": str(stats_path.resolve()),
                "stats_sha256": _sha(stats_path),
                "rollouts_path": str(rollout_path.resolve()),
                "rollouts_sha256": _sha(rollout_path),
                "copied_rollouts_sha256": _sha(copied),
                "accepted": int(summary.get("n_accepted_rollouts", 0)),
            }
        )
        total_attempts += int(summary["n_rollouts_ok"])
        total_accepted += int(summary.get("n_accepted_rollouts", 0))
        total_errors += int(summary["n_rollouts_err"])
        total_tasks += int(summary["n_tasks"])

    if total_tasks != expected_tasks or total_attempts != expected_tasks * expected_k:
        raise ContractError("aggregate fixed task x k budget is inconsistent")
    if total_errors != 0:
        raise ContractError("rollout batch contains errors; zero is required")
    if total_accepted < minimum_accepted:
        raise ContractError(
            f"accepted trajectory count {total_accepted} is below preregistered {minimum_accepted}"
        )
    manifest = {
        "schema_version": "stage5.relative_mouse_rollout_batch.v1",
        "status": "complete",
        "data_class": "synthetic_training_only",
        "contains_official_heldout": False,
        "contains_real_vm_eval": False,
        "contains_crowd_cast": False,
        "adaptive_resampling": False,
        "train_seed": expected_train_seed,
        "tasks": total_tasks,
        "k": expected_k,
        "attempts": total_attempts,
        "errors": total_errors,
        "accepted": total_accepted,
        "minimum_accepted": minimum_accepted,
        "source_shards": source_shards,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_write_json(out / "manifest.json", manifest)
    return manifest

