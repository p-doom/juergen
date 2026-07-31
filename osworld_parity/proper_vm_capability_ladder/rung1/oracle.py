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

from .fixtures import Fixture, load_manifest


@dataclass(frozen=True)
class OracleResult:
    fixture_id: str
    fixture_sha256: str
    oracle_status: str
    MOUSE_SOLVED: bool
    reason: str
    oracle_pid: int


def evaluate_state(fixture: Fixture, state: dict[str, Any]) -> OracleResult:
    status = "ok"
    solved = False
    reason = "state oracle rejected"
    try:
        if state.get("fixture_id") != fixture.id:
            raise ValueError("fixture id mismatch")
        if state.get("fixture_sha256") != fixture.fixture_sha256:
            raise ValueError("fixture hash mismatch")
        if state.get("ready") is not True:
            raise ValueError("fixture did not report ready")
        current = state.get("current")
        if not isinstance(current, dict):
            raise ValueError("current state missing")
        if fixture.template == "click":
            solved = current.get("checked") is fixture.expected["checked"]
        elif fixture.template == "focus_type":
            solved = current.get("text") == fixture.expected["text"]
        elif fixture.template == "scroll":
            initial = int(fixture.params["initial_y"])
            observed = int(current.get("scroll_y", initial))
            minimum = int(fixture.expected["min_delta"])
            if fixture.expected["direction"] == "down":
                solved = observed - initial >= minimum
            else:
                solved = initial - observed >= minimum
        elif fixture.template == "drag":
            solved = int(current.get("value", -1)) == int(fixture.expected["value"])
        else:
            raise ValueError(f"unsupported template {fixture.template!r}")
        reason = "expected state observed" if solved else "expected state not observed"
    except (KeyError, TypeError, ValueError) as exc:
        status = "error"
        reason = str(exc)
    return OracleResult(
        fixture_id=fixture.id,
        fixture_sha256=fixture.fixture_sha256,
        oracle_status=status,
        MOUSE_SOLVED=bool(status == "ok" and solved),
        reason=reason,
        oracle_pid=os.getpid(),
    )


def evaluate_in_fresh_process(
    fixture: Fixture, state: dict[str, Any], *, timeout_s: float = 30.0
) -> OracleResult:
    fd, raw_path = tempfile.mkstemp(prefix="r1a_oracle_", suffix=".json")
    path = Path(raw_path)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, sort_keys=True)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "osworld_parity.proper_vm_capability_ladder.rung1.oracle",
                "--fixture-id",
                fixture.id,
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
                f"oracle process failed rc={completed.returncode}: {completed.stderr.strip()}"
            )
        try:
            payload = json.loads(completed.stdout)
            result = OracleResult(**payload)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(f"oracle returned invalid JSON: {completed.stdout!r}") from exc
        if result.oracle_pid == os.getpid():
            raise RuntimeError("oracle did not execute in a fresh process")
        return result
    finally:
        path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args(argv)
    fixture = load_manifest().by_id(args.fixture_id)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    result = evaluate_state(fixture, state)
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.oracle_status == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
