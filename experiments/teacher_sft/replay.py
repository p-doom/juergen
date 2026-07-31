"""Fresh-environment replay verification for train-derived trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.teacher_sft import SCHEMA_VERSION
from experiments.teacher_sft.actions import parse_compact_sequence
from experiments.teacher_sft.contracts import (
    ContractError,
    ensure_empty_output,
    file_sha256,
    iter_jsonl,
    read_json,
    require_finite_score,
    write_json,
    write_jsonl,
)
from experiments.teacher_sft.rpc import JsonlRpcEnvironment, assert_observation


def replay_converted(
    converted_dir: Path,
    output_dir: Path,
    *,
    env_command: str,
    min_reward: float = 1.0,
    only_split: str = "train_validation",
) -> dict[str, Any]:
    """Replay compact actions; this is prepared infrastructure, not an official eval."""
    ensure_empty_output(output_dir)
    if only_split not in {"train", "train_validation"}:
        raise ContractError("live replay is restricted to train-derived splits")
    converted_manifest = read_json(converted_dir / "manifest.json")
    if (
        not isinstance(converted_manifest, dict)
        or converted_manifest.get("construction_scope") != "train_only"
    ):
        raise ContractError("converted replay input is not train-only")
    path = converted_dir / "converted.jsonl"
    if file_sha256(path) != converted_manifest.get("converted_sha256"):
        raise ContractError("converted input hash mismatch")
    results: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        task = row["task"]
        if task.get("source_split") != "train" or task.get("split") != only_split:
            continue
        with JsonlRpcEnvironment(env_command) as environment:
            observation = assert_observation(
                environment.call("reset", {"task": task, "replay": True}),
                context="replay reset",
            )
            for step in row["steps"]:
                sequence = step["compact_action"]
                parse_compact_sequence(sequence)
                observation = assert_observation(
                    environment.call("step_compact", {"sequence": sequence}),
                    context=f"replay {row['rollout_id']} step {step['step_index']}",
                )
            reward_result = environment.call("reward")
            if not isinstance(reward_result, dict):
                raise ContractError("reward adapter response must be an object")
            reward = require_finite_score(
                reward_result.get("reward"), context="replay reward"
            )
            success = reward >= min_reward and reward_result.get("success") is True
            results.append(
                {
                    "rollout_id": row["rollout_id"],
                    "task_key": task["task_key"],
                    "split": only_split,
                    "reward": reward,
                    "success": success,
                    "final_cursor": observation["cursor"],
                }
            )
    write_jsonl(output_dir / "results.jsonl", results)
    if not results:
        raise ContractError(f"live replay selected no rows from split {only_split!r}")
    failures = [result for result in results if not result["success"]]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "teacher_sft_live_replay",
        "evaluation_scope": "train_derived_only",
        "split": only_split,
        "converted_manifest_sha256": file_sha256(converted_dir / "manifest.json"),
        "results_sha256": file_sha256(output_dir / "results.jsonl"),
        "counts": {"total": len(results), "successful": len(results) - len(failures)},
        "all_successful": not failures,
    }
    if failures:
        write_jsonl(output_dir / "replay_quarantine.jsonl", failures)
        raise ContractError(
            f"{len(failures)} live replay(s) did not meet success policy"
        )
    write_json(output_dir / "manifest.json", manifest)
    return manifest
