from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .manifest import TaskDefinition, payload_sha256


@dataclass(frozen=True)
class HiddenOracleResult:
    task_id: str
    task_sha256: str
    oracle_status: str
    MOUSE_SOLVED: bool
    reason: str
    state_sha256: str
    oracle_pid: int


def evaluate_hidden_payload(payload: dict[str, Any]) -> HiddenOracleResult:
    task = payload.get("task")
    state = payload.get("state")
    task_id = str(task.get("task_id", "")) if isinstance(task, dict) else ""
    task_sha = str(payload.get("task_sha256", ""))
    status = "ok"
    solved = False
    reason = "hidden state oracle rejected"
    state_sha = payload_sha256(state)
    try:
        if not isinstance(task, dict) or not isinstance(state, dict):
            raise ValueError("task/state payload missing")
        if payload_sha256(task) != task_sha:
            raise ValueError("task payload seal mismatch")
        if state.get("task_id") != task_id or state.get("task_sha256") != task_sha:
            raise ValueError("state provenance mismatch")
        steps = task.get("steps")
        if not isinstance(steps, list) or state.get("completed_steps") != steps:
            reason = "ordered semantic steps incomplete"
        elif state.get("held_buttons") or state.get("held_keys"):
            reason = "input remained held"
        elif "coalesced_type" in steps and state.get("text") != task.get("target_text"):
            reason = "typed text mismatch"
        elif "scroll" in steps and (
            int(state.get("scroll_total", 0)) * int(task.get("scroll_clicks", 0)) <= 0
            or abs(int(state.get("scroll_total", 0)))
            < abs(int(task.get("scroll_clicks", 0)))
        ):
            reason = "scroll direction/distance mismatch"
        elif "click" in steps and state.get("clicked") is not True:
            reason = "confirmation click missing"
        elif "drag" in steps and state.get("drag_complete") is not True:
            reason = "drag destination missing"
        else:
            solved = True
            reason = "all hidden state predicates passed"
    except (TypeError, ValueError) as exc:
        status = "error"
        reason = str(exc)
    return HiddenOracleResult(
        task_id=task_id,
        task_sha256=task_sha,
        oracle_status=status,
        MOUSE_SOLVED=bool(status == "ok" and solved),
        reason=reason,
        state_sha256=state_sha,
        oracle_pid=os.getpid(),
    )


def evaluate_in_fresh_process(
    task: TaskDefinition, state: dict[str, Any], *, timeout_s: float = 30.0
) -> HiddenOracleResult:
    payload = {
        "task": task.unsigned_payload(),
        "task_sha256": task.task_sha256,
        "state": state,
    }
    descriptor, raw_path = tempfile.mkstemp(prefix="r33_hidden_oracle_", suffix=".json")
    path = Path(raw_path)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "osworld_parity.mixed_action_short_vm.hidden_oracle",
                "--payload",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if completed.returncode not in {0, 3}:
            raise RuntimeError(
                f"hidden oracle failed rc={completed.returncode}: "
                f"{completed.stderr.strip()}"
            )
        try:
            result = HiddenOracleResult(**json.loads(completed.stdout))
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(
                f"hidden oracle returned invalid JSON: {completed.stdout!r}"
            ) from exc
        if result.oracle_pid == os.getpid():
            raise RuntimeError("hidden oracle did not run in a fresh process")
        return result
    finally:
        path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    result = evaluate_hidden_payload(payload)
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.oracle_status == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
