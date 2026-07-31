"""Parallel VM workers for closed-loop absolute teacher collection."""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.teacher_sft import SCHEMA_VERSION
from experiments.teacher_sft.contracts import (
    ContractError,
    ensure_empty_output,
    file_sha256,
    object_sha256,
    require_finite_score,
    write_json,
    write_jsonl,
)
from experiments.teacher_sft.rpc import JsonlRpcEnvironment, assert_observation
from experiments.teacher_sft.task_sources import load_task_rows
from experiments.teacher_sft.teacher import OpenAIChatTeacher, load_teacher_spec


def _copy_observation(value: Any, destination: Path, *, context: str) -> dict[str, Any]:
    observation = assert_observation(value, context=context)
    source = Path(observation["image_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {
        "image_path": str(destination.resolve()),
        "image_sha256": file_sha256(destination),
        "cursor": observation["cursor"],
        "screen_size": observation["screen_size"],
    }


def _termination(actions: list[dict[str, Any]]) -> str | None:
    for action in actions:
        if action["action"] == "terminate":
            status = str(action.get("status", "success")).strip().lower()
            return "success" if status == "success" else "failure"
    return None


def _collect_one(
    task: dict[str, Any],
    *,
    candidate: int,
    output_dir: Path,
    teacher_spec: dict[str, Any],
    api_key: str,
    env_command: str,
    max_steps: int,
    success_reward: float,
) -> dict[str, Any]:
    safe_key = task["task_key"].replace(":", "__").replace("/", "_")
    rollout_id = f"{safe_key}__candidate_{candidate:03d}"
    run_dir = output_dir / "rollouts" / rollout_id
    run_dir.mkdir(parents=True, exist_ok=False)
    teacher = OpenAIChatTeacher(teacher_spec, api_key=api_key)
    steps: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    parse_errors = 0
    termination: str | None = None
    runtime_error: str | None = None
    reward = 0.0
    environment_success = False
    try:
        with JsonlRpcEnvironment(env_command) as environment:
            observation = environment.call(
                "reset", {"task": task, "work_dir": str(run_dir)}
            )
            current = _copy_observation(
                observation,
                run_dir / "steps" / "step_000.png",
                context="collection reset",
            )
            for step_index in range(max_steps):
                try:
                    raw, actions = teacher.act(
                        instruction=task["instruction"],
                        image_path=Path(current["image_path"]),
                        history=history,
                    )
                except ContractError:
                    parse_errors += 1
                    raise
                traces: list[dict[str, Any]] = []
                latest_observation: Any = current
                for action in actions:
                    if action["action"] == "terminate":
                        cursor = (
                            traces[-1]["cursor_after"] if traces else current["cursor"]
                        )
                        traces.append(
                            {
                                "cursor_before": cursor,
                                "cursor_after": cursor,
                                "resolved_target_px": None,
                            }
                        )
                        continue
                    result = environment.call("step_native", {"action": action})
                    if not isinstance(result, dict) or not isinstance(
                        result.get("trace"), dict
                    ):
                        raise ContractError(
                            "step_native must return observation and trace objects"
                        )
                    trace = result["trace"]
                    required = {"cursor_before", "cursor_after", "resolved_target_px"}
                    if not required.issubset(trace):
                        raise ContractError(
                            "step_native trace lacks cursor/target telemetry"
                        )
                    traces.append(trace)
                    latest_observation = result.get("observation")
                next_observation = _copy_observation(
                    latest_observation,
                    run_dir / "steps" / f"step_{step_index + 1:03d}.png",
                    context=f"collection step {step_index}",
                )
                steps.append(
                    {
                        "step_index": step_index,
                        "observation_before": current,
                        "teacher_response": raw,
                        "actions": actions,
                        "execution_traces": traces,
                        "observation_after": next_observation,
                    }
                )
                history.extend(
                    [
                        {
                            "role": "user",
                            "content": task["instruction"]
                            if step_index == 0
                            else "Continue from the current screenshot.",
                        },
                        {"role": "assistant", "content": raw},
                    ]
                )
                current = next_observation
                termination = _termination(actions)
                if termination is not None:
                    break
            reward_result = environment.call("reward")
            if not isinstance(reward_result, dict):
                raise ContractError("reward result must be an object")
            reward = require_finite_score(
                reward_result.get("reward"), context="collection reward"
            )
            environment_success = reward_result.get("success") is True
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        # Preserve failed attempts for deterministic rejection diagnostics.
        runtime_error = f"{type(exc).__name__}: {exc}"
    success = (
        environment_success and reward >= success_reward and termination == "success"
    )
    rollout = {
        "schema_version": SCHEMA_VERSION,
        "rollout_id": rollout_id,
        "task": task,
        "teacher": {
            "model_id": teacher_spec["model_id"],
            "model_revision": teacher_spec["model_revision"],
            "spec_sha256": teacher_spec["spec_sha256"],
            "system_prompt_sha256": teacher_spec["system_prompt_sha256"],
            "action_space": "native_absolute",
            "coordinate_space": teacher_spec["coordinate_space"],
            "coordinate_grid": teacher_spec.get("coordinate_grid"),
        },
        "steps": steps,
        "result": {
            "reward": reward,
            "success": success,
            "environment_success": environment_success,
            "termination": termination,
            "parse_errors": parse_errors,
            "error": runtime_error,
        },
    }
    write_json(run_dir / "rollout.json", rollout)
    return {
        "rollout_id": rollout_id,
        "task_key": task["task_key"],
        "candidate": candidate,
        "path": str((run_dir / "rollout.json").resolve()),
        "sha256": file_sha256(run_dir / "rollout.json"),
        "success": success,
    }


def collect_rollouts(
    task_manifest_dir: Path,
    teacher_spec_path: Path,
    output_dir: Path,
    *,
    env_command: str,
    api_key: str = "none",
    candidates_per_task: int = 1,
    max_steps: int = 30,
    workers: int = 1,
    success_reward: float = 1.0,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    ensure_empty_output(output_dir)
    if (
        min(candidates_per_task, max_steps, workers, shard_count) < 1
        or not 0 <= shard_index < shard_count
    ):
        raise ContractError("invalid collection worker/shard settings")
    tasks = load_task_rows(task_manifest_dir)
    tasks = [
        task
        for task in tasks
        if int(object_sha256(task["task_key"])[:16], 16) % shard_count == shard_index
    ]
    teacher_spec = load_teacher_spec(teacher_spec_path)
    work = [
        (task, candidate) for task in tasks for candidate in range(candidates_per_task)
    ]

    def run(item: tuple[dict[str, Any], int]) -> dict[str, Any]:
        return _collect_one(
            item[0],
            candidate=item[1],
            output_dir=output_dir,
            teacher_spec=teacher_spec,
            api_key=api_key,
            env_command=env_command,
            max_steps=max_steps,
            success_reward=success_reward,
        )

    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="teacher-sft-vm"
    ) as pool:
        index = list(pool.map(run, work))
    index.sort(key=lambda row: row["rollout_id"])
    write_jsonl(output_dir / "index.jsonl", index)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "teacher_sft_absolute_rollouts",
        "construction_scope": "train_only",
        "task_manifest_sha256": file_sha256(task_manifest_dir / "manifest.json"),
        "teacher_spec_sha256": teacher_spec["spec_sha256"],
        "index_sha256": file_sha256(output_dir / "index.jsonl"),
        "candidates_per_task": candidates_per_task,
        "max_steps": max_steps,
        "success_reward": success_reward,
        "shard": {"index": shard_index, "count": shard_count},
        "counts": {
            "tasks": len(tasks),
            "rollouts": len(index),
            "successful": sum(r["success"] for r in index),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
