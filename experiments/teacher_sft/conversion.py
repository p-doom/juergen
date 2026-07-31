"""Accepted absolute teacher rollouts -> verified compact-relative trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.teacher_sft import CONVERSION_VERSION, SCHEMA_VERSION
from experiments.teacher_sft.actions import (
    SymbolicState,
    convert_native_action,
    parse_compact_sequence,
)
from experiments.teacher_sft.contracts import (
    ContractError,
    ensure_empty_output,
    file_sha256,
    iter_jsonl,
    object_sha256,
    read_json,
    verify_declared_hash,
    write_json,
    write_jsonl,
)


def _point(value: Any, context: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(v, int) for v in value)
    ):
        raise ContractError(f"{context} must be two integers")
    return value[0], value[1]


def _verify_observation(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{context} observation is not an object")
    path = Path(str(value.get("image_path", ""))).resolve()
    verify_declared_hash(path, value.get("image_sha256"), context=f"{context} image")
    _point(value.get("cursor"), f"{context}.cursor")
    screen = _point(value.get("screen_size"), f"{context}.screen_size")
    if screen[0] <= 1 or screen[1] <= 1:
        raise ContractError(f"{context} has invalid screen_size")
    return {**value, "image_path": str(path)}


def convert_rollout(rollout: dict[str, Any], rollout_sha256: str) -> dict[str, Any]:
    task = rollout["task"]
    if task.get("source_split") != "train" or task.get("split") not in {
        "train",
        "train_validation",
    }:
        raise ContractError("conversion input is not train-scoped")
    steps = rollout.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ContractError("rollout contains no steps")
    output_steps: list[dict[str, Any]] = []
    state: SymbolicState | None = None
    replay_state: SymbolicState | None = None
    saw_terminate = False
    terminal_control: str | None = None
    prior_after: dict[str, Any] | None = None
    for expected_index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("step_index") != expected_index:
            raise ContractError("rollout steps must be contiguous and zero-based")
        before = _verify_observation(
            step.get("observation_before"), context=f"step {expected_index} before"
        )
        if prior_after is not None and (
            prior_after["image_sha256"] != before["image_sha256"]
            or prior_after["cursor"] != before["cursor"]
            or prior_after["screen_size"] != before["screen_size"]
        ):
            raise ContractError(f"observation chain break at step {expected_index}")
        if state is None:
            state = SymbolicState(
                cursor=_point(before["cursor"], "initial cursor"),
                screen_size=_point(before["screen_size"], "initial screen"),
            )
            replay_state = SymbolicState(
                cursor=state.cursor, screen_size=state.screen_size
            )
        elif state.cursor != _point(before["cursor"], "before cursor"):
            raise ContractError(f"cursor mismatch at step {expected_index}")
        if state.screen_size != _point(before["screen_size"], "screen_size"):
            raise ContractError("screen size changed during rollout")
        actions = step.get("actions")
        traces = step.get("execution_traces")
        if not isinstance(actions, list) or not actions or not isinstance(traces, list):
            raise ContractError(f"step {expected_index} lacks actions/traces")
        if len(actions) != len(traces):
            raise ContractError(
                f"step {expected_index} action/trace cardinality mismatch"
            )
        compact = []
        for action, trace in zip(actions, traces, strict=True):
            converted = convert_native_action(action, trace, state)
            if saw_terminate:
                raise ContractError("action appears after termination")
            if any(item.control in {"TERMINATE", "FAIL"} for item in converted):
                saw_terminate = True
                terminal_control = next(
                    item.control for item in converted if item.control
                )
            compact.extend(converted)
        compact_text = "\n".join(item.render() for item in compact)
        parsed = parse_compact_sequence(compact_text)
        if tuple(item.render() for item in parsed) != tuple(
            item.render() for item in compact
        ):
            raise ContractError("compact parse/format round trip mismatch")
        assert replay_state is not None
        for action in parsed:
            replay_state.apply(action)
        if replay_state.cursor != state.cursor or replay_state.held != state.held:
            raise ContractError(f"symbolic replay diverged at step {expected_index}")
        if (
            replay_state.typed_text != state.typed_text
            or replay_state.scroll_total != state.scroll_total
        ):
            raise ContractError(
                f"symbolic replay side effects diverged at step {expected_index}"
            )
        after = _verify_observation(
            step.get("observation_after"), context=f"step {expected_index} after"
        )
        if state.cursor != _point(after["cursor"], "after cursor"):
            raise ContractError(
                f"converted cursor differs from post-step cursor at {expected_index}"
            )
        if state.screen_size != _point(after["screen_size"], "after screen_size"):
            raise ContractError("screen size changed after action")
        prior_after = after
        output_steps.append(
            {
                "step_index": expected_index,
                "observation": before,
                "compact_action": compact_text,
                "compact_action_sha256": object_sha256(compact_text),
                "cursor_after": list(state.cursor),
                "held_after": sorted(state.held),
            }
        )
    if state is None or replay_state is None or state.held or replay_state.held:
        raise ContractError(
            f"rollout ends with held inputs: {sorted(state.held) if state else []}"
        )
    if not saw_terminate or terminal_control != "TERMINATE":
        raise ContractError("successful rollout has no converted TERMINATE")
    converted = {
        "schema_version": SCHEMA_VERSION,
        "conversion_version": CONVERSION_VERSION,
        "rollout_id": rollout["rollout_id"],
        "source_rollout_sha256": rollout_sha256,
        "task": task,
        "teacher": rollout["teacher"],
        "steps": output_steps,
        "result": rollout["result"],
    }
    converted["converted_row_sha256"] = object_sha256(converted)
    return converted


def convert_accepted(rejection_dir: Path, output_dir: Path) -> dict[str, Any]:
    ensure_empty_output(output_dir)
    rejection_manifest = read_json(rejection_dir / "manifest.json")
    if (
        not isinstance(rejection_manifest, dict)
        or rejection_manifest.get("construction_scope") != "train_only"
    ):
        raise ContractError("rejection artifact is not train-only")
    accepted_path = rejection_dir / "accepted.jsonl"
    if file_sha256(accepted_path) != rejection_manifest.get("accepted_sha256"):
        raise ContractError("accepted.jsonl hash mismatch")
    rows = []
    failures = []
    for ref in iter_jsonl(accepted_path):
        rollout_path = Path(str(ref.get("path", ""))).resolve()
        try:
            verify_declared_hash(
                rollout_path, ref.get("sha256"), context="accepted rollout"
            )
            rollout = read_json(rollout_path)
            row = convert_rollout(rollout, ref["sha256"])
        except ContractError as exc:
            failures.append({"rollout_id": ref.get("rollout_id"), "reason": str(exc)})
            continue
        rows.append(row)
    # Fail closed: an accepted success that cannot be converted invalidates the stage,
    # while still writing a diagnostic quarantine without a completion manifest.
    if failures:
        write_jsonl(output_dir / "conversion_quarantine.jsonl", failures)
        raise ContractError(
            f"{len(failures)} accepted rollout(s) failed deterministic conversion"
        )
    rows.sort(key=lambda row: (row["task"]["task_key"], row["rollout_id"]))
    write_jsonl(output_dir / "converted.jsonl", rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "teacher_sft_converted_relative",
        "construction_scope": "train_only",
        "conversion_version": CONVERSION_VERSION,
        "rejection_manifest_sha256": file_sha256(rejection_dir / "manifest.json"),
        "converted_sha256": file_sha256(output_dir / "converted.jsonl"),
        "n_rollouts": len(rows),
        "n_steps": sum(len(row["steps"]) for row in rows),
        "symbolic_replay_verified": True,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
