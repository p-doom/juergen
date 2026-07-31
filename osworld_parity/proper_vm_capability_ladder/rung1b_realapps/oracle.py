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

from .fixtures import Fixture, load_manifest


@dataclass(frozen=True)
class OracleResult:
    fixture_id: str
    fixture_sha256: str
    oracle_status: str
    MOUSE_SOLVED: bool
    reason: str
    oracle_pid: int


def _file_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evaluate_state(fixture: Fixture, state: dict[str, Any]) -> OracleResult:
    status = "ok"
    solved = False
    reason = "expected hidden application state not observed"
    try:
        if state.get("schema_version") != 1:
            raise ValueError("state schema mismatch")
        if state.get("fixture_id") != fixture.id:
            raise ValueError("fixture id mismatch")
        if state.get("fixture_sha256") != fixture.fixture_sha256:
            raise ValueError("fixture hash mismatch")
        if fixture.template == "vscode_focus_type":
            if state.get("application") != "vscode":
                raise ValueError("VS Code state provenance missing")
            encoded = state.get("content_b64")
            if not isinstance(encoded, str):
                raise ValueError("saved file content missing")
            try:
                content = base64.b64decode(encoded, validate=True).decode("utf-8")
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError("saved file is not exact UTF-8") from exc
            solved = (
                state.get("file_name") == fixture.params["file_name"]
                and content == fixture.expected["text"]
                and state.get("content_sha256") == _file_sha256(content)
            )
        elif fixture.template == "local_document_scroll":
            if state.get("application") != "chrome":
                raise ValueError("Chrome state provenance missing")
            if state.get("document_kind") != "guest_local_development_document":
                raise ValueError("local document provenance missing")
            initial = int(fixture.params["initial_y"])
            observed = int(state["scroll_y"])
            minimum = int(fixture.expected["min_delta"])
            solved = (
                observed - initial >= minimum
                if fixture.params["direction"] == "down"
                else initial - observed >= minimum
            )
        elif fixture.template == "files_drag":
            if state.get("application") != "files" or state.get("drag_backend") != "filesystem":
                raise ValueError("Files filesystem-state provenance missing")
            expected_sha = _file_sha256(str(fixture.params["content"]))
            solved = (
                state.get("source_exists") is False
                and state.get("destination_sha256") == expected_sha
                and state.get("decoy_sha256") is None
            )
        else:
            raise ValueError(f"unsupported template: {fixture.template}")
        if solved:
            reason = "expected hidden application state observed"
    except (KeyError, TypeError, ValueError) as exc:
        status = "error"
        reason = str(exc)
    return OracleResult(
        fixture.id,
        fixture.fixture_sha256,
        status,
        bool(status == "ok" and solved),
        reason,
        os.getpid(),
    )


def evaluate_in_fresh_process(
    fixture: Fixture, state: dict[str, Any], *, timeout_s: float = 30.0
) -> OracleResult:
    fd, raw_path = tempfile.mkstemp(prefix="r1b_hidden_oracle_", suffix=".json")
    path = Path(raw_path)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, sort_keys=True)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "osworld_parity.proper_vm_capability_ladder.rung1b_realapps.oracle",
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
                f"hidden oracle failed rc={completed.returncode}: {completed.stderr.strip()}"
            )
        try:
            result = OracleResult(**json.loads(completed.stdout))
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError("hidden oracle returned invalid JSON") from exc
        if result.oracle_pid == os.getpid():
            raise RuntimeError("oracle isolation failed: PID was reused")
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

