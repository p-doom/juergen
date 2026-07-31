from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .actions import ARMS, build_recovery_demonstration
from .outcomes import OutcomeLabel
from .spec import RecoveryTask, load_recovery_tasks


SCHEMA_PATH = Path(__file__).with_name("on_policy_rollout.schema.json")
FORBIDDEN_KEYS = {
    "reward",
    "hidden_reward",
    "hidden_state",
    "oracle",
    "oracle_state",
    "expected",
    "near_miss",
    "trainer_evaluation",
}


class RolloutSchemaError(RuntimeError):
    pass


def _walk_forbidden(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        leaked = FORBIDDEN_KEYS.intersection(value)
        if leaked:
            raise RolloutSchemaError(f"trainer-only field leak at {path}: {sorted(leaked)}")
        for key, child in value.items():
            _walk_forbidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden(child, f"{path}[{index}]")


def validate_rollout_record(record: dict[str, Any]) -> None:
    _walk_forbidden(record)
    required = {
        "schema_version",
        "source",
        "task_id",
        "task_sha256",
        "split",
        "arm",
        "perturbation",
        "base_horizon",
        "recovery_horizon",
        "events",
        "trainer_only_values_exported",
    }
    if set(record) != required:
        raise RolloutSchemaError(
            f"rollout top-level fields mismatch: {sorted(set(record) ^ required)}"
        )
    if record["schema_version"] != 1 or record["arm"] not in ARMS:
        raise RolloutSchemaError("rollout version/arm mismatch")
    if record["split"] not in {"train", "development"}:
        raise RolloutSchemaError("sealed/unknown split is not exportable")
    if record["trainer_only_values_exported"] is not False:
        raise RolloutSchemaError("trainer-only export flag must be false")
    if int(record["recovery_horizon"]) != int(record["base_horizon"]) + 2:
        raise RolloutSchemaError("recovery horizon is not base horizon+2")
    events = record["events"]
    if not isinstance(events, list) or not events:
        raise RolloutSchemaError("rollout events missing")
    sequence = [event.get("sequence_index") for event in events if isinstance(event, dict)]
    if sequence != list(range(len(events))):
        raise RolloutSchemaError("rollout event sequence is not contiguous")
    labels = {item.value for item in OutcomeLabel}
    for event in events:
        if not isinstance(event, dict):
            raise RolloutSchemaError("rollout event must be an object")
        expected_fields = {
            "sequence_index",
            "policy_step",
            "origin",
            "action",
            "executor_dispatch_status",
            "outcome_label",
            "screenshot_sha256",
        }
        if set(event) != expected_fields:
            raise RolloutSchemaError("rollout event fields mismatch")
        if event["outcome_label"] not in labels:
            raise RolloutSchemaError("unknown rollout outcome label")
        origin = event["origin"]
        label = event["outcome_label"]
        if origin == "controller_injection":
            if event["policy_step"] is not None:
                raise RolloutSchemaError("controller injection consumed policy horizon")
            if label not in {"injected_perturbation", "executor_failure"}:
                raise RolloutSchemaError("controller injection label mismatch")
        elif origin == "on_policy":
            if not isinstance(event["policy_step"], int) or event["policy_step"] <= 0:
                raise RolloutSchemaError("on-policy event has invalid step")
            if label not in {
                "natural_ineffective_action",
                "executor_failure",
                "effective_recovery_action",
            }:
                raise RolloutSchemaError("on-policy outcome label mismatch")
        elif origin == "scripted_teacher":
            if label != "scripted_recovery_action":
                raise RolloutSchemaError("scripted teacher label mismatch")
        else:
            raise RolloutSchemaError(f"unknown event origin: {origin!r}")


def _event(
    sequence_index: int,
    *,
    origin: str,
    action: dict[str, Any] | str,
    label: str,
    policy_step: int | None,
) -> dict[str, Any]:
    import hashlib

    token = json.dumps(
        [sequence_index, origin, action], ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return {
        "sequence_index": sequence_index,
        "policy_step": policy_step,
        "origin": origin,
        "action": action,
        "executor_dispatch_status": "ok",
        "outcome_label": label,
        "screenshot_sha256": hashlib.sha256(token).hexdigest(),
    }


def scripted_recovery_records(split: str, arm: str) -> list[dict[str, Any]]:
    if split not in {"train", "development"}:
        raise ValueError("scripted recovery export is train/development only")
    rows: list[dict[str, Any]] = []
    for task in load_recovery_tasks(split):
        demo = build_recovery_demonstration(
            task, arm=arm, initial_cursor=(73, 91)
        )
        events: list[dict[str, Any]] = []
        for action in demo.perturbation.actions:
            events.append(
                _event(
                    len(events),
                    origin="controller_injection",
                    action=action,
                    label="injected_perturbation",
                    policy_step=None,
                )
            )
        for step, action in enumerate(demo.policy_actions, start=1):
            events.append(
                _event(
                    len(events),
                    origin="scripted_teacher",
                    action=action,
                    label="scripted_recovery_action",
                    policy_step=step,
                )
            )
        record = {
            "schema_version": 1,
            "source": "scripted_controlled_perturbation_recovery",
            "task_id": task.id,
            "task_sha256": task.task_sha256,
            "split": split,
            "arm": arm,
            "perturbation": task.perturbation,
            "base_horizon": task.base_horizon,
            "recovery_horizon": task.recovery_horizon,
            "events": events,
            "trainer_only_values_exported": False,
        }
        validate_rollout_record(record)
        rows.append(record)
    return rows


def public_on_policy_record(
    task: RecoveryTask,
    *,
    arm: str,
    events: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        "source": "on_policy_controlled_perturbation_rollout",
        "task_id": task.id,
        "task_sha256": task.task_sha256,
        "split": task.split,
        "arm": arm,
        "perturbation": task.perturbation,
        "base_horizon": task.base_horizon,
        "recovery_horizon": task.recovery_horizon,
        "events": list(events),
        "trainer_only_values_exported": False,
    }
    validate_rollout_record(record)
    return record


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                validate_rollout_record(row)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(raw_path, path)
    finally:
        Path(raw_path).unlink(missing_ok=True)
