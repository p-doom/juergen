"""Fresh-process oracle bridge for semantic same-app curriculum tasks."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ...rung1b_realapps.fixtures import Fixture as RealAppFixture
from ...rung1b_realapps.oracle import evaluate_state as evaluate_realapp_state
from ..fixtures import Fixture as SameAppFixture
from ..fixtures import canonical_json
from ..oracle import (
    evaluate_state as evaluate_sameapp_state,
    initial_state as sameapp_initial_state,
    scripted_state as sameapp_scripted_state,
)
from .manifests import load_manifest
from .schema import SemanticTask


@dataclass(frozen=True)
class OracleResult:
    task_id: str
    fixture_sha256: str
    oracle_status: str
    MOUSE_SOLVED: bool
    semantic_step_index: int | None
    matched_target_ref: str | None
    semantic_state_sha256: str
    reason: str
    oracle_pid: int


def _content_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def as_sameapp_fixture(task: SemanticTask) -> SameAppFixture:
    return SameAppFixture(
        id=task.task_id,
        app=task.app,
        split=task.split,
        parameter_seed=task.parameter_seed,
        semantic_steps=task.semantic_step_count,
        horizon=max(task.budget_contract["primitive_action_caps"].values()),
        instruction=task.instruction,
        params=task.params,
        expected=task.expected,
        near_miss=task.near_miss,
        fixture_sha256=task.fixture_sha256,
    )


def as_vscode_fixture(task: SemanticTask) -> RealAppFixture:
    values: dict[str, Any] = dict(
        id=task.task_id,
        template="vscode_focus_type",
        split=task.split,
        parameter_seed=task.parameter_seed,
        horizon=max(task.budget_contract["primitive_action_caps"].values()),
        instruction=task.instruction,
        params=task.params,
        expected=task.expected,
        near_miss=task.near_miss,
        fixture_sha256=task.fixture_sha256,
    )
    # Harness commit 173a9d5 adds these required fields. Keep this bridge
    # compatible with both its parent and merged constructor.
    fields = getattr(RealAppFixture, "__dataclass_fields__", {})
    if "gate_role" in fields:
        values["gate_role"] = task.gate_role
    if "coverage_label" in fields:
        values["coverage_label"] = task.coverage_label
    return RealAppFixture(**values)


# Compatibility aliases for the first scaffold consumer. New integrations
# should use the public ``as_*`` names above.
_sameapp_fixture = as_sameapp_fixture
_vscode_fixture = as_vscode_fixture


def initial_state(task: SemanticTask) -> dict[str, Any]:
    if task.app != "vscode":
        state = sameapp_initial_state(as_sameapp_fixture(task))
    else:
        content = str(task.params["initial_text"])
        state = {
            "schema_version": 1,
            "fixture_id": task.task_id,
            "fixture_sha256": task.fixture_sha256,
            "application": "vscode",
            "file_name": task.params["file_name"],
            "content_b64": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "content_sha256": _content_sha256(content),
        }
    state["held_inputs"] = []
    return state


def scripted_state(task: SemanticTask, *, near_miss: bool) -> dict[str, Any]:
    if task.app != "vscode":
        state = sameapp_scripted_state(as_sameapp_fixture(task), near_miss=near_miss)
    else:
        expected = task.near_miss if near_miss else task.expected
        content = str(expected["text"])
        state = {
            "schema_version": 1,
            "fixture_id": task.task_id,
            "fixture_sha256": task.fixture_sha256,
            "application": "vscode",
            "file_name": task.params["file_name"],
            "content_b64": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "content_sha256": _content_sha256(content),
        }
    state["held_inputs"] = []
    return state


def reset_signature(task: SemanticTask, state: dict[str, Any]) -> str:
    expected = initial_state(task)
    if state != expected:
        raise ValueError(f"{task.task_id}: reset state is not the exact initial state")
    return hashlib.sha256(canonical_json(state)).hexdigest()


def evaluate_state(task: SemanticTask, state: dict[str, Any]) -> OracleResult:
    state_sha256 = hashlib.sha256(canonical_json(state)).hexdigest()
    if state.get("held_inputs") != []:
        return OracleResult(
            task.task_id,
            task.fixture_sha256,
            "ok",
            False,
            None,
            None,
            state_sha256,
            "final input state is not fully released",
            os.getpid(),
        )
    if state.get("fixture_id") != task.task_id or state.get("fixture_sha256") != task.fixture_sha256:
        return OracleResult(
            task.task_id,
            task.fixture_sha256,
            "error",
            False,
            None,
            None,
            state_sha256,
            "task identity or fixture seal mismatch",
            os.getpid(),
        )
    legacy = (
        evaluate_realapp_state(as_vscode_fixture(task), state)
        if task.app == "vscode"
        else evaluate_sameapp_state(as_sameapp_fixture(task), state)
    )
    return OracleResult(
        task.task_id,
        task.fixture_sha256,
        legacy.oracle_status,
        legacy.MOUSE_SOLVED,
        None,
        None,
        state_sha256,
        legacy.reason,
        os.getpid(),
    )


def evaluate_semantic_state(
    task: SemanticTask,
    state: dict[str, Any],
    *,
    expected_step_index: int,
    expected_target_ref: str,
) -> OracleResult:
    """Verify a live semantic cursor transition from task-owned history."""

    state_sha256 = hashlib.sha256(canonical_json(state)).hexdigest()
    try:
        if state.get("task_id") != task.task_id or state.get("fixture_sha256") != task.fixture_sha256:
            raise ValueError("task identity or fixture seal mismatch")
        if state.get("held_inputs") != []:
            raise ValueError("semantic state has held inputs")
        if not 1 <= expected_step_index <= len(task.gold_cursor_history):
            raise ValueError("semantic step index is outside task history")
        milestone = task.gold_cursor_history[expected_step_index - 1]
        if milestone.step_id != expected_step_index:
            raise ValueError("task semantic history is not index-aligned")
        if milestone.target_ref != expected_target_ref:
            raise ValueError("expected target does not match task-owned history")
        geometry = state.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError("live geometry is missing")
        if milestone.cursor_after_ref == "runtime.initial_cursor":
            resolved = state.get("initial_cursor")
        else:
            prefix = "geometry."
            if not milestone.cursor_after_ref.startswith(prefix):
                raise ValueError("unsupported cursor reference")
            resolved = geometry.get(milestone.cursor_after_ref[len(prefix) :])
        cursor = state.get("cursor")
        if not isinstance(resolved, (list, tuple)) or not isinstance(cursor, (list, tuple)):
            raise ValueError("live cursor resolution is missing")
        solved = list(cursor) == list(resolved)
        reason = (
            "live cursor matches task-owned semantic target"
            if solved
            else "live cursor does not match task-owned semantic target"
        )
        return OracleResult(
            task.task_id,
            task.fixture_sha256,
            "ok",
            solved,
            expected_step_index,
            milestone.target_ref if solved else None,
            state_sha256,
            reason,
            os.getpid(),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return OracleResult(
            task.task_id,
            task.fixture_sha256,
            "error",
            False,
            expected_step_index,
            None,
            state_sha256,
            str(exc),
            os.getpid(),
        )


def evaluate_in_fresh_process(
    task: SemanticTask,
    state: dict[str, Any],
    *,
    expected_step_index: int | None = None,
    expected_target_ref: str | None = None,
    timeout_s: float = 30.0,
) -> OracleResult:
    fd, raw_path = tempfile.mkstemp(prefix="r2_semantic_oracle_", suffix=".json")
    path = Path(raw_path)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, sort_keys=True)
        command = [
                sys.executable,
                "-m",
                "osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.oracle",
                "--split",
                task.split,
                "--task-id",
                task.task_id,
                "--state",
                str(path),
            ]
        if expected_step_index is not None:
            if expected_target_ref is None:
                raise ValueError("semantic oracle requires expected_target_ref")
            command.extend(
                [
                    "--expected-step-index",
                    str(expected_step_index),
                    "--expected-target-ref",
                    expected_target_ref,
                ]
            )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if completed.returncode not in {0, 3}:
            raise RuntimeError(
                f"fresh curriculum oracle failed rc={completed.returncode}: {completed.stderr.strip()}"
            )
        result = OracleResult(**json.loads(completed.stdout))
        if result.oracle_pid == os.getpid():
            raise RuntimeError("fresh-process oracle isolation failed")
        return result
    finally:
        path.unlink(missing_ok=True)


def verify_fixture_contract(
    task: SemanticTask, *, artifact_roots: dict[str, Path]
) -> dict[str, Any]:
    """Verify independently extracted artifacts through isolated oracles."""

    required = {"reset", "reset_repeat", "near", "gold"}
    if set(artifact_roots) != required:
        raise ValueError(
            f"{task.task_id}: artifact roots must be exactly {sorted(required)}"
        )
    module = importlib.import_module(task.verifier["state_extractor_module"])
    extractor = getattr(module, task.verifier["state_extractor_entrypoint"])
    states = {
        name: extractor(task, Path(artifact_roots[name])) for name in sorted(required)
    }
    reset_one = reset_signature(task, states["reset"])
    reset_two = reset_signature(task, states["reset_repeat"])
    reset_oracle = evaluate_in_fresh_process(task, states["reset"])
    near_miss = evaluate_in_fresh_process(task, states["near"])
    gold = evaluate_in_fresh_process(task, states["gold"])
    return {
        "task_id": task.task_id,
        "fixture_sha256": task.fixture_sha256,
        "reset_rejected": (
            reset_oracle.oracle_status == "ok" and reset_oracle.MOUSE_SOLVED is False
        ),
        "near_miss_rejected": (
            near_miss.oracle_status == "ok" and near_miss.MOUSE_SOLVED is False
        ),
        "gold_passed": gold.oracle_status == "ok" and gold.MOUSE_SOLVED is True,
        "reset_reproducible": reset_one == reset_two,
        "fresh_process_final_oracle": gold.oracle_pid != os.getpid(),
        "zero_held_inputs": states["gold"].get("held_inputs") == [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--expected-step-index", type=int)
    parser.add_argument("--expected-target-ref")
    args = parser.parse_args(argv)
    task = load_manifest(args.split).by_id(args.task_id)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    if (args.expected_step_index is None) != (args.expected_target_ref is None):
        parser.error("semantic mode requires both expected-step-index and expected-target-ref")
    result = (
        evaluate_semantic_state(
            task,
            state,
            expected_step_index=args.expected_step_index,
            expected_target_ref=args.expected_target_ref,
        )
        if args.expected_step_index is not None
        else evaluate_state(task, state)
    )
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.oracle_status == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
