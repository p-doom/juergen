"""Fresh-process oracle bridge for semantic same-app curriculum tasks."""

from __future__ import annotations

import argparse
import base64
import hashlib
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
    reason: str
    oracle_pid: int


def _content_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sameapp_fixture(task: SemanticTask) -> SameAppFixture:
    return SameAppFixture(
        id=task.task_id,
        app=task.app,
        split=task.split,
        parameter_seed=task.parameter_seed,
        semantic_steps=task.semantic_step_count,
        horizon=task.max_action_turns,
        instruction=task.instruction,
        params=task.params,
        expected=task.expected,
        near_miss=task.near_miss,
        fixture_sha256=task.fixture_sha256,
    )


def _vscode_fixture(task: SemanticTask) -> RealAppFixture:
    return RealAppFixture(
        id=task.task_id,
        template="vscode_focus_type",
        split=task.split,
        parameter_seed=task.parameter_seed,
        horizon=task.max_action_turns,
        instruction=task.instruction,
        params=task.params,
        expected=task.expected,
        near_miss=task.near_miss,
        fixture_sha256=task.fixture_sha256,
    )


def initial_state(task: SemanticTask) -> dict[str, Any]:
    if task.app != "vscode":
        state = sameapp_initial_state(_sameapp_fixture(task))
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
            "saved": False,
        }
    state["held_inputs"] = []
    return state


def scripted_state(task: SemanticTask, *, near_miss: bool) -> dict[str, Any]:
    if task.app != "vscode":
        state = sameapp_scripted_state(_sameapp_fixture(task), near_miss=near_miss)
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
            "saved": bool(expected.get("saved", True)),
        }
    state["held_inputs"] = []
    return state


def reset_signature(task: SemanticTask, state: dict[str, Any]) -> str:
    expected = initial_state(task)
    if state != expected:
        raise ValueError(f"{task.task_id}: reset state is not the exact initial state")
    return hashlib.sha256(canonical_json(state)).hexdigest()


def evaluate_state(task: SemanticTask, state: dict[str, Any]) -> OracleResult:
    if state.get("held_inputs") != []:
        return OracleResult(
            task.task_id,
            task.fixture_sha256,
            "ok",
            False,
            "final input state is not fully released",
            os.getpid(),
        )
    if state.get("fixture_id") != task.task_id or state.get("fixture_sha256") != task.fixture_sha256:
        return OracleResult(
            task.task_id,
            task.fixture_sha256,
            "error",
            False,
            "task identity or fixture seal mismatch",
            os.getpid(),
        )
    legacy = (
        evaluate_realapp_state(_vscode_fixture(task), state)
        if task.app == "vscode"
        else evaluate_sameapp_state(_sameapp_fixture(task), state)
    )
    return OracleResult(
        task.task_id,
        task.fixture_sha256,
        legacy.oracle_status,
        legacy.MOUSE_SOLVED,
        legacy.reason,
        os.getpid(),
    )


def evaluate_in_fresh_process(
    task: SemanticTask, state: dict[str, Any], *, timeout_s: float = 30.0
) -> OracleResult:
    fd, raw_path = tempfile.mkstemp(prefix="r2_semantic_oracle_", suffix=".json")
    path = Path(raw_path)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, sort_keys=True)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.oracle",
                "--split",
                task.split,
                "--task-id",
                task.task_id,
                "--state",
                str(path),
            ],
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


def verify_fixture_contract(task: SemanticTask) -> dict[str, Any]:
    first_reset = initial_state(task)
    second_reset = initial_state(task)
    reset_one = reset_signature(task, first_reset)
    reset_two = reset_signature(task, second_reset)
    reset_oracle = evaluate_state(task, first_reset)
    near_miss = evaluate_state(task, scripted_state(task, near_miss=True))
    gold = evaluate_in_fresh_process(task, scripted_state(task, near_miss=False))
    return {
        "task_id": task.task_id,
        "fixture_sha256": task.fixture_sha256,
        "reset_rejected": reset_oracle.MOUSE_SOLVED is False,
        "near_miss_rejected": near_miss.MOUSE_SOLVED is False,
        "gold_passed": gold.MOUSE_SOLVED is True,
        "reset_reproducible": reset_one == reset_two,
        "fresh_process_final_oracle": gold.oracle_pid != os.getpid(),
        "zero_held_inputs": scripted_state(task, near_miss=False)["held_inputs"] == [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args(argv)
    task = load_manifest(args.split).by_id(args.task_id)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    result = evaluate_state(task, state)
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.oracle_status == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
