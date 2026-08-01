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

from .oracle import evaluate_state
from .stage0_loader import Stage0Record, load_stage0_inventory


@dataclass(frozen=True)
class Stage0OracleResult:
    record_id: str
    manifest_payload_sha256: str
    oracle_status: str
    MOUSE_SOLVED: bool
    reason: str
    component_results: tuple[dict[str, Any], ...]
    oracle_pid: int


def evaluate_composed(
    record: Stage0Record,
    states: list[dict[str, Any]],
) -> Stage0OracleResult:
    inventory = load_stage0_inventory()
    status = "ok"
    reason = "one or more ordered component states do not match"
    rows: list[dict[str, Any]] = []
    if len(states) != len(record.component_tasks):
        status = "error"
        reason = "composed state count does not match ordered source tasks"
    else:
        for order, (task, state) in enumerate(
            zip(record.component_tasks, states, strict=True), start=1
        ):
            result = evaluate_state(task, state)
            rows.append(
                {
                    "order": order,
                    "source_task_id": task.id,
                    "app": task.app,
                    "fixture_sha256": task.fixture_sha256,
                    "oracle_status": result.oracle_status,
                    "solved": result.MOUSE_SOLVED,
                    "reason": result.reason,
                }
            )
            if result.oracle_status != "ok":
                status = "error"
                reason = f"component {order} oracle error: {result.reason}"
                break
    solved = bool(
        status == "ok"
        and len(rows) == len(record.component_tasks)
        and all(row["solved"] for row in rows)
    )
    if solved:
        reason = "all ordered component states match in fresh composed verifier"
    return Stage0OracleResult(
        record_id=record.id,
        manifest_payload_sha256=inventory.manifest_payload_sha256,
        oracle_status=status,
        MOUSE_SOLVED=solved,
        reason=reason,
        component_results=tuple(rows),
        oracle_pid=os.getpid(),
    )


def evaluate_composed_in_fresh_process(
    record: Stage0Record,
    states: list[dict[str, Any]],
    *,
    timeout_s: float = 30.0,
) -> Stage0OracleResult:
    fd, raw_path = tempfile.mkstemp(prefix="cleanroom_stage0_oracle_", suffix=".json")
    path = Path(raw_path)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(states, handle, ensure_ascii=False, sort_keys=True)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                __name__,
                "--record-id",
                record.id,
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
                f"fresh composed oracle failed rc={completed.returncode}: {completed.stderr.strip()}"
            )
        result = Stage0OracleResult(**json.loads(completed.stdout))
        if result.oracle_pid == os.getpid():
            raise RuntimeError("fresh composed oracle reused the caller PID")
        return result
    finally:
        path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fresh composed Stage0 verifier")
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--state", required=True, type=Path)
    args = parser.parse_args(argv)
    record = load_stage0_inventory().by_id(args.record_id)
    states = json.loads(args.state.read_text(encoding="utf-8"))
    if not isinstance(states, list) or not all(isinstance(state, dict) for state in states):
        raise SystemExit("composed state must be a list of objects")
    result = evaluate_composed(record, states)
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.oracle_status == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
