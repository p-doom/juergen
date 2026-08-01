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

from .schema import Task, canonical_json, load_corpus


@dataclass(frozen=True)
class OracleResult:
    task_id: str
    fixture_sha256: str
    oracle_status: str
    MOUSE_SOLVED: bool
    reason: str
    oracle_pid: int


def _content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def initial_state(task: Any) -> dict[str, Any]:
    if task.app == "vscode":
        text = str(task.params["initial_text"])
        return {
            "schema_version": 1,
            "fixture_id": task.id,
            "fixture_sha256": task.fixture_sha256,
            "application": "vscode",
            "file_name": task.params["file_name"],
            "content_b64": base64.b64encode(text.encode("utf-8")).decode("ascii"),
            "content_sha256": _content_sha256(text),
        }
    state: dict[str, Any] = {
        "schema_version": 1,
        "fixture_id": task.id,
        "fixture_sha256": task.fixture_sha256,
        "app": task.app,
        "saved": False,
    }
    if task.app == "writer":
        text = str(task.params["initial_text"])
        state.update({"text": text, "content_sha256": _content_sha256(text), "bold": False})
    elif task.app == "calc":
        state.update({"cell": task.params["cell"], "formula": None, "display_value": task.params["initial_value"]})
    elif task.app == "files":
        state.update(
            {
                "source_exists": True,
                "destination": None,
                "final_name": task.params["source_name"],
                "content_sha256": _content_sha256(str(task.params["content"])),
            }
        )
    elif task.app == "chrome":
        state.update(
            {
                "section": "root",
                "scroll_y": int(task.params["initial_scroll_y"]),
                "setting_enabled": False,
            }
        )
    else:
        raise ValueError(f"unsupported app: {task.app}")
    return state


def scripted_state(task: Any, *, near_miss: bool) -> dict[str, Any]:
    state = initial_state(task)
    expected = task.near_miss if near_miss else task.expected
    if task.app == "vscode":
        text = str(expected["text"])
        state.update(
            {
                "content_b64": base64.b64encode(text.encode("utf-8")).decode("ascii"),
                "content_sha256": _content_sha256(text),
            }
        )
    elif task.app == "writer":
        text = str(expected["text"])
        state.update(
            {
                "text": text,
                "content_sha256": _content_sha256(text),
                "bold": bool(expected["bold"]),
                "saved": True,
            }
        )
    elif task.app == "calc":
        state.update(
            {
                "formula": expected["formula"],
                "display_value": expected["display_value"],
                "saved": True,
            }
        )
    elif task.app == "files":
        state.update(
            {
                "source_exists": False,
                "destination": expected["destination"],
                "final_name": expected["final_name"],
                "saved": True,
            }
        )
    else:
        delta = int(task.params["minimum_scroll_delta"])
        state.update(
            {
                "section": expected["section"],
                "scroll_y": int(task.params["initial_scroll_y"]) + delta,
                "setting_enabled": expected["setting_enabled"],
                "saved": bool(expected["setting_enabled"]),
            }
        )
    return state


def reset_signature(task: Any, state: dict[str, Any]) -> str:
    stable = {key: value for key, value in state.items() if key not in {"generation", "probe_pid", "timestamp_ns"}}
    if stable != initial_state(task):
        raise ValueError(f"{task.id}: reset state is not exact")
    return hashlib.sha256(canonical_json(stable)).hexdigest()


def evaluate_state(task: Any, state: dict[str, Any]) -> OracleResult:
    status = "ok"
    solved = False
    reason = "expected hidden application state not observed"
    try:
        if state.get("schema_version") != 1:
            raise ValueError("state schema mismatch")
        if state.get("fixture_id") != task.id or state.get("fixture_sha256") != task.fixture_sha256:
            raise ValueError("fixture identity mismatch")
        if task.app == "vscode":
            if state.get("application") != "vscode":
                raise ValueError("VS Code application provenance mismatch")
        elif state.get("app") != task.app:
            raise ValueError("application provenance mismatch")
        if task.app == "vscode":
            encoded = state.get("content_b64")
            if not isinstance(encoded, str):
                raise ValueError("VS Code saved content is missing")
            content = base64.b64decode(encoded, validate=True).decode("utf-8")
            solved = (
                state.get("file_name") == task.params["file_name"]
                and content == task.expected["text"]
                and state.get("content_sha256") == _content_sha256(content)
            )
        elif task.app == "writer":
            text = str(state["text"])
            solved = (
                text == task.expected["text"]
                and state.get("content_sha256") == _content_sha256(text)
                and state.get("bold") is bool(task.expected["bold"])
                and state.get("saved") is True
            )
        elif task.app == "calc":
            solved = (
                state.get("cell") == task.params["cell"]
                and state.get("formula") == task.expected["formula"]
                and str(state.get("display_value")) == task.expected["display_value"]
                and state.get("saved") is True
            )
        elif task.app == "files":
            solved = (
                state.get("source_exists") is False
                and state.get("destination") == task.expected["destination"]
                and state.get("final_name") == task.expected["final_name"]
                and state.get("content_sha256") == _content_sha256(str(task.params["content"]))
            )
        elif task.app == "chrome":
            initial = int(task.params["initial_scroll_y"])
            observed = int(state["scroll_y"])
            scroll_ok = observed - initial >= int(task.params["minimum_scroll_delta"])
            solved = (
                state.get("section") == task.expected["section"]
                and scroll_ok
                and state.get("setting_enabled") is True
            )
        if solved:
            reason = "expected hidden application state observed"
    except (KeyError, TypeError, ValueError) as exc:
        status = "error"
        reason = str(exc)
    return OracleResult(
        task.id,
        task.fixture_sha256,
        status,
        bool(status == "ok" and solved),
        reason,
        os.getpid(),
    )


def evaluate_in_fresh_process(task: Any, state: dict[str, Any], *, timeout_s: float = 30.0) -> OracleResult:
    fd, raw_path = tempfile.mkstemp(prefix="cleanroom_dev_oracle_", suffix=".json")
    path = Path(raw_path)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, sort_keys=True)
        inventory = (
            "plumbing-smoke"
            if task.id.startswith("cln-smoke-")
            else "stage0"
            if task.id.startswith("cln-s0-src-")
            else "corpus"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                __name__,
                "--inventory",
                inventory,
                "--task-id",
                task.id,
                "--state",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if completed.returncode not in {0, 3}:
            raise RuntimeError(f"fresh oracle failed rc={completed.returncode}: {completed.stderr.strip()}")
        result = OracleResult(**json.loads(completed.stdout))
        if result.oracle_pid == os.getpid():
            raise RuntimeError("fresh-process oracle reused the caller PID")
        return result
    finally:
        path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory", choices=("corpus", "plumbing-smoke", "stage0"), default="corpus"
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.inventory == "plumbing-smoke":
        from .smoke_schema import load_smoke

        task = load_smoke().by_id(args.task_id)
    elif args.inventory == "stage0":
        from .stage0_loader import load_stage0_inventory

        task = load_stage0_inventory().source_by_id(args.task_id)
    else:
        task = load_corpus().by_id(args.task_id)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    result = evaluate_state(task, state)
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.oracle_status == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
