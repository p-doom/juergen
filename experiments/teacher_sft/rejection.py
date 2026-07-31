"""Deterministic success/reward rejection sampling over raw teacher rollouts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.teacher_sft import SCHEMA_VERSION
from experiments.teacher_sft.contracts import (
    ContractError,
    ensure_empty_output,
    file_sha256,
    iter_jsonl,
    object_sha256,
    read_json,
    require_finite_score,
    require_train_split,
    write_json,
    write_jsonl,
)
from experiments.teacher_sft.task_sources import load_task_rows


def _rollout_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("rollout.json") if path.is_file())


def _validate_rollout(
    path: Path,
    tasks: dict[str, dict[str, Any]],
    *,
    min_reward: float,
    require_success_termination: bool,
) -> tuple[dict[str, Any], list[str]]:
    rollout = read_json(path)
    reasons: list[str] = []
    if not isinstance(rollout, dict) or rollout.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"invalid rollout schema: {path}")
    task = rollout.get("task")
    if not isinstance(task, dict):
        raise ContractError(f"rollout task is missing: {path}")
    task_key = task.get("task_key")
    if task_key not in tasks:
        raise ContractError(f"rollout references unknown task: {task_key!r}")
    manifest_task = tasks[task_key]
    require_train_split(task.get("source_split"), context=f"rollout {path}")
    if task.get("split") != manifest_task.get("split"):
        raise ContractError(f"rollout split differs from task manifest: {path}")
    if task.get("task_row_sha256") != manifest_task.get("task_row_sha256"):
        raise ContractError(f"rollout task row hash differs from task manifest: {path}")
    steps = rollout.get("steps")
    if not isinstance(steps, list) or not steps:
        reasons.append("no_steps")
    result = rollout.get("result")
    if not isinstance(result, dict):
        raise ContractError(f"rollout result is missing: {path}")
    reward = require_finite_score(
        result.get("reward"), context=f"rollout reward {path}"
    )
    if reward < min_reward:
        reasons.append("reward_below_threshold")
    if result.get("success") is not True:
        reasons.append("environment_not_successful")
    if require_success_termination and result.get("termination") != "success":
        reasons.append("no_success_termination")
    if result.get("error"):
        reasons.append("runtime_error")
    parse_errors = result.get("parse_errors")
    if not isinstance(parse_errors, int) or parse_errors < 0:
        raise ContractError(f"rollout parse_errors is invalid: {path}")
    if parse_errors:
        reasons.append("parse_errors")
    return rollout, sorted(set(reasons))


def reject_rollouts(
    task_manifest_dir: Path,
    rollouts_dir: Path,
    output_dir: Path,
    *,
    min_reward: float = 1.0,
    max_per_task: int = 1,
    require_success_termination: bool = True,
) -> dict[str, Any]:
    """Keep best successful candidates; ties are fully deterministic."""
    ensure_empty_output(output_dir)
    if not 0.0 <= min_reward <= 1.0 or max_per_task < 1:
        raise ContractError("invalid rejection policy")
    task_rows = load_task_rows(task_manifest_dir)
    tasks = {row["task_key"]: row for row in task_rows}
    collection_manifest_path = rollouts_dir / "manifest.json"
    collection_manifest = read_json(collection_manifest_path)
    if (
        not isinstance(collection_manifest, dict)
        or collection_manifest.get("construction_scope") != "train_only"
    ):
        raise ContractError("rollout collection manifest is not train-only")
    if collection_manifest.get("task_manifest_sha256") != file_sha256(
        task_manifest_dir / "manifest.json"
    ):
        raise ContractError("rollout collection used a different task manifest")
    index_path = rollouts_dir / "index.jsonl"
    if file_sha256(index_path) != collection_manifest.get("index_sha256"):
        raise ContractError("rollout collection index hash mismatch")
    indexed = {
        Path(str(row.get("path", ""))).resolve(): row for row in iter_jsonl(index_path)
    }
    discovered = _rollout_files(rollouts_dir)
    if set(indexed) != {path.resolve() for path in discovered}:
        raise ContractError("rollout files and collection index differ")
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in discovered:
        if file_sha256(path) != indexed[path.resolve()].get("sha256"):
            raise ContractError(f"rollout differs from collection index: {path}")
        rollout, reasons = _validate_rollout(
            path,
            tasks,
            min_reward=min_reward,
            require_success_termination=require_success_termination,
        )
        rollout_id = rollout.get("rollout_id")
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ContractError(f"rollout id is missing: {path}")
        if rollout_id in seen_ids:
            raise ContractError(f"duplicate rollout_id: {rollout_id}")
        seen_ids.add(rollout_id)
        result = rollout["result"]
        ref = {
            "schema_version": SCHEMA_VERSION,
            "rollout_id": rollout_id,
            "task_key": rollout["task"]["task_key"],
            "split": rollout["task"]["split"],
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "reward": float(result["reward"]),
            "n_steps": len(rollout.get("steps", [])),
            "parse_errors": result["parse_errors"],
        }
        if reasons:
            rejected.append({**ref, "reasons": reasons})
        else:
            candidates[ref["task_key"]].append(ref)

    accepted: list[dict[str, Any]] = []
    for task_key in sorted(candidates):
        ranked = sorted(
            candidates[task_key],
            key=lambda row: (
                -row["reward"],
                row["parse_errors"],
                row["n_steps"],
                row["rollout_id"],
            ),
        )
        accepted.extend(ranked[:max_per_task])
        for row in ranked[max_per_task:]:
            rejected.append({**row, "reasons": ["lower_ranked_success"]})
    accepted.sort(key=lambda row: (row["task_key"], row["rollout_id"]))
    rejected.sort(key=lambda row: (row["task_key"], row["rollout_id"]))
    write_jsonl(output_dir / "accepted.jsonl", accepted)
    write_jsonl(output_dir / "rejected.jsonl", rejected)
    if not accepted:
        raise ContractError("rejection policy accepted no teacher rollouts")
    policy = {
        "min_reward": min_reward,
        "max_per_task": max_per_task,
        "require_success_termination": require_success_termination,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "teacher_sft_rejection_sample",
        "construction_scope": "train_only",
        "task_manifest_sha256": file_sha256(task_manifest_dir / "manifest.json"),
        "rollout_collection_manifest_sha256": file_sha256(collection_manifest_path),
        "policy": policy,
        "policy_sha256": object_sha256(policy),
        "accepted_sha256": file_sha256(output_dir / "accepted.jsonl"),
        "rejected_sha256": file_sha256(output_dir / "rejected.jsonl"),
        "counts": {"accepted": len(accepted), "rejected": len(rejected)},
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
