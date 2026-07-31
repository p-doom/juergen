from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .fixtures import Fixture, canonical_json, load_manifest


@dataclass(frozen=True)
class OracleResult:
    fixture_id: str
    fixture_sha256: str
    oracle_status: str
    MOUSE_SOLVED: bool
    reason: str
    oracle_pid: int


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def initial_state(fixture: Fixture) -> dict[str, Any]:
    state: dict[str, Any] = {
        "schema_version": 1,
        "fixture_id": fixture.id,
        "fixture_sha256": fixture.fixture_sha256,
        "app": fixture.app,
        "saved": False,
    }
    if fixture.app == "writer":
        text = str(fixture.params["initial_text"])
        state.update({"text": text, "content_sha256": content_sha256(text), "bold": False})
    elif fixture.app == "calc":
        state.update({"cell": fixture.params["cell"], "formula": None, "display_value": fixture.params["initial_value"]})
    elif fixture.app == "files":
        state.update(
            {
                "source_exists": True,
                "destination": None,
                "final_name": fixture.params["source_name"],
                "content_sha256": content_sha256(str(fixture.params["content"])),
            }
        )
    elif fixture.app == "chrome":
        state.update(
            {
                "section": "root",
                "scroll_y": int(fixture.params.get("initial_scroll_y", 0)),
                "setting_enabled": False,
            }
        )
    else:
        raise ValueError(f"unsupported fixture app: {fixture.app}")
    return state


def scripted_state(fixture: Fixture, *, near_miss: bool) -> dict[str, Any]:
    state = initial_state(fixture)
    if fixture.app == "writer":
        text = str(fixture.near_miss["text"] if near_miss else fixture.expected["text"])
        state.update(
            {
                "text": text,
                "content_sha256": content_sha256(text),
                "bold": bool(not near_miss and fixture.expected["bold"]),
                "saved": True,
            }
        )
    elif fixture.app == "calc":
        expected = fixture.near_miss if near_miss else fixture.expected
        state.update(
            {
                "formula": expected["formula"],
                "display_value": expected["display_value"],
                "saved": True,
            }
        )
    elif fixture.app == "files":
        expected = fixture.near_miss if near_miss else fixture.expected
        state.update(
            {
                "source_exists": False,
                "destination": expected["destination"],
                "final_name": expected["final_name"],
                "saved": True,
            }
        )
    elif fixture.app == "chrome":
        expected = fixture.near_miss if near_miss else fixture.expected
        initial = int(fixture.params.get("initial_scroll_y", 0))
        if fixture.params.get("scroll_direction", "down") == "up":
            scroll_y = initial - int(fixture.params["minimum_scroll_delta"])
        elif "minimum_scroll_delta" in fixture.params:
            scroll_y = initial + int(fixture.params["minimum_scroll_delta"])
        else:
            scroll_y = int(fixture.params["minimum_scroll_y"])
        state.update(
            {
                "section": expected["section"],
                "scroll_y": scroll_y,
                "setting_enabled": expected["setting_enabled"],
                "saved": True,
            }
        )
    return state


def reset_signature(fixture: Fixture, state: dict[str, Any]) -> str:
    stable = {key: value for key, value in state.items() if key not in {"generation", "probe_pid", "timestamp_ns"}}
    if stable != initial_state(fixture):
        raise ValueError(f"{fixture.id}: reset state is not the exact initial state")
    return hashlib.sha256(canonical_json(stable)).hexdigest()


def evaluate_state(fixture: Fixture, state: dict[str, Any]) -> OracleResult:
    status = "ok"
    solved = False
    reason = "expected hidden application state not observed"
    try:
        if state.get("schema_version") != 1:
            raise ValueError("state schema mismatch")
        if state.get("fixture_id") != fixture.id or state.get("fixture_sha256") != fixture.fixture_sha256:
            raise ValueError("fixture identity mismatch")
        if state.get("app") != fixture.app:
            raise ValueError("application provenance mismatch")
        if fixture.app == "writer":
            text = str(state["text"])
            solved = (
                text == fixture.expected["text"]
                and state.get("content_sha256") == content_sha256(text)
                and state.get("bold") is fixture.expected["bold"]
                and state.get("saved") is True
            )
        elif fixture.app == "calc":
            solved = (
                state.get("cell") == fixture.params["cell"]
                and state.get("formula") == fixture.expected["formula"]
                and str(state.get("display_value")) == fixture.expected["display_value"]
                and state.get("saved") is True
            )
        elif fixture.app == "files":
            solved = (
                state.get("source_exists") is False
                and state.get("destination") == fixture.expected["destination"]
                and state.get("final_name") == fixture.expected["final_name"]
                and state.get("content_sha256") == content_sha256(str(fixture.params["content"]))
            )
        elif fixture.app == "chrome":
            initial = int(fixture.params.get("initial_scroll_y", 0))
            observed = int(state.get("scroll_y", -1))
            if "minimum_scroll_delta" in fixture.params:
                delta = int(fixture.params["minimum_scroll_delta"])
                scroll_ok = (
                    initial - observed >= delta
                    if fixture.params.get("scroll_direction", "down") == "up"
                    else observed - initial >= delta
                )
            else:
                scroll_ok = observed >= int(fixture.params["minimum_scroll_y"])
            solved = (
                state.get("section") == fixture.expected["section"]
                and scroll_ok
                and state.get("setting_enabled") is fixture.expected["setting_enabled"]
            )
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
    fd, raw_path = tempfile.mkstemp(prefix="r2_sameapp_oracle_", suffix=".json")
    path = Path(raw_path)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, sort_keys=True)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "osworld_parity.proper_vm_capability_ladder.rung2_sameapp.oracle",
                "--split",
                fixture.split,
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
            raise RuntimeError(f"hidden oracle failed rc={completed.returncode}: {completed.stderr.strip()}")
        result = OracleResult(**json.loads(completed.stdout))
        if result.oracle_pid == os.getpid():
            raise RuntimeError("fresh-process oracle isolation failed")
        return result
    finally:
        path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args(argv)
    fixture = load_manifest(args.split).by_id(args.fixture_id)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    result = evaluate_state(fixture, state)
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.oracle_status == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
