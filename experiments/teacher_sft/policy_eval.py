"""Train-derived closed-loop eval for a compact-relative SFT checkpoint."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from experiments.teacher_sft import SCHEMA_VERSION
from experiments.teacher_sft.actions import parse_compact_sequence
from experiments.teacher_sft.contracts import (
    ContractError,
    ensure_empty_output,
    file_sha256,
    require_finite_score,
    write_json,
    write_jsonl,
)
from experiments.teacher_sft.rpc import JsonlRpcEnvironment, assert_observation
from experiments.teacher_sft.sft import SYSTEM_PROMPT
from experiments.teacher_sft.task_sources import load_task_rows


def _complete(
    *,
    base_url: str,
    model_id: str,
    api_key: str,
    instruction: str,
    image_path: Path,
    history: list[dict[str, Any]],
) -> str:
    image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model_id,
        "temperature": 0,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    },
                ],
            },
        ],
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read())
        return str(body["choices"][0]["message"]["content"])
    except (
        OSError,
        urllib.error.URLError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
    ) as exc:
        raise ContractError(f"compact policy request failed: {exc}") from exc


def evaluate_policy(
    task_manifest_dir: Path,
    output_dir: Path,
    *,
    base_url: str,
    model_id: str,
    model_revision: str,
    env_command: str,
    api_key: str = "none",
    split: str = "train_validation",
    max_steps: int = 30,
    min_reward: float = 1.0,
) -> dict[str, Any]:
    ensure_empty_output(output_dir)
    if split not in {"train", "train_validation"}:
        raise ContractError("policy eval is restricted to train-derived splits")
    tasks = [
        task for task in load_task_rows(task_manifest_dir) if task["split"] == split
    ]
    if not tasks:
        raise ContractError(f"policy eval selected no tasks from split {split!r}")
    results = []
    for task in tasks:
        termination = None
        error = None
        reward = 0.0
        environment_success = False
        try:
            with JsonlRpcEnvironment(env_command) as environment:
                history: list[dict[str, Any]] = []
                observation = assert_observation(
                    environment.call("reset", {"task": task, "evaluation": True}),
                    context="policy eval reset",
                )
                for _step in range(max_steps):
                    response = _complete(
                        base_url=base_url,
                        model_id=model_id,
                        api_key=api_key,
                        instruction=task["instruction"],
                        image_path=Path(observation["image_path"]),
                        history=history,
                    )
                    sequence = parse_compact_sequence(response)
                    control_positions = [
                        index for index, action in enumerate(sequence) if action.control
                    ]
                    if control_positions and control_positions != [len(sequence) - 1]:
                        raise ContractError(
                            "a control token must occur once, at sequence end"
                        )
                    executable = sequence[:-1] if control_positions else sequence
                    if executable:
                        executable_text = "\n".join(
                            action.render() for action in executable
                        )
                        observation = assert_observation(
                            environment.call(
                                "step_compact", {"sequence": executable_text}
                            ),
                            context="policy eval step",
                        )
                    history.extend(
                        [
                            {
                                "role": "user",
                                "content": task["instruction"]
                                if not history
                                else "Continue from the current screenshot.",
                            },
                            {"role": "assistant", "content": response},
                        ]
                    )
                    if control_positions:
                        termination = (
                            "success"
                            if sequence[-1].control == "TERMINATE"
                            else "failure"
                        )
                        break
                reward_result = environment.call("reward")
                if not isinstance(reward_result, dict):
                    raise ContractError("policy eval reward response must be an object")
                reward = require_finite_score(
                    reward_result.get("reward"), context="policy reward"
                )
                environment_success = reward_result.get("success") is True
        except (ContractError, OSError, RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "task_key": task["task_key"],
                "split": split,
                "reward": reward,
                "success": (
                    environment_success
                    and reward >= min_reward
                    and termination == "success"
                ),
                "termination": termination,
                "error": error,
            }
        )
    write_jsonl(output_dir / "results.jsonl", results)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "teacher_sft_policy_eval",
        "evaluation_scope": "train_derived_only",
        "split": split,
        "model_id": model_id,
        "model_revision": model_revision,
        "task_manifest_sha256": file_sha256(task_manifest_dir / "manifest.json"),
        "results_sha256": file_sha256(output_dir / "results.jsonl"),
        "counts": {
            "tasks": len(results),
            "successful": sum(result["success"] for result in results),
            "errors": sum(bool(result["error"]) for result in results),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
