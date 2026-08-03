from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from .suite import DevelopmentTask


@dataclass(frozen=True)
class PostconditionResult:
    task_id: str
    oracle_status: str
    success: bool
    reason: str
    evidence: dict[str, Any]
    oracle_pid: int


def _history_has_exact(history: object, command: str) -> bool:
    if not isinstance(history, str):
        return False
    return any(line.strip().lstrip("0123456789 ").strip() == command for line in history.splitlines())


def evaluate(task: DevelopmentTask, state: dict[str, Any]) -> PostconditionResult:
    status = "ok"
    success = False
    reason = "required realized VM state was not observed"
    evidence: dict[str, Any] = {}
    try:
        if state.get("schema_version") != 1 or state.get("task_id") != task.id:
            raise ValueError("state identity/schema mismatch")
        if task.kind == "terminal_command":
            command = str(task.expected["command"])
            marker = str(task.expected["listing_marker"])
            command_executed = _history_has_exact(state.get("history"), command)
            listing_observed = marker in str(state.get("transcript", ""))
            prompt_returned = int(state.get("prompt_count", 0)) >= 2
            evidence = {
                "command_executed": command_executed,
                "listing_marker_observed_in_terminal_output": listing_observed,
                "shell_prompt_returned": prompt_returned,
                "expected_command": command,
                "expected_listing_marker": marker,
            }
            success = command_executed and listing_observed and prompt_returned
        elif task.kind == "terminal_exact_text":
            expected = str(task.expected["text"])
            observed = state.get("captured_text")
            evidence = {
                "capture_file_exists": state.get("capture_file_exists") is True,
                "exact_text_match": observed == expected,
                "expected_text": expected,
                "observed_text": observed,
            }
            success = evidence["capture_file_exists"] and evidence["exact_text_match"]
        elif task.kind == "open_chrome":
            active = str(state.get("active_window", "")).casefold()
            markers = [str(value).casefold() for value in task.expected["active_window_class_any"]]
            active_matches = any(marker in active for marker in markers)
            chrome_process = bool(state.get("chrome_process"))
            evidence = {
                "chrome_process_observed": chrome_process,
                "chrome_is_foreground": active_matches,
                "active_window": state.get("active_window"),
                "window_inventory": state.get("windows"),
            }
            success = chrome_process and active_matches
        elif task.kind == "focus_terminal_and_type":
            command = str(task.expected["command"])
            content = str(task.expected["content"])
            active = str(state.get("active_window", "")).casefold()
            evidence = {
                "terminal_is_foreground": "terminal" in active,
                "command_executed": _history_has_exact(state.get("history"), command),
                "proof_file_exists": state.get("proof_file_exists") is True,
                "proof_file_exact": state.get("proof_file_content") == content,
                "expected_command": command,
                "expected_content": content,
            }
            success = all(
                evidence[key]
                for key in (
                    "terminal_is_foreground",
                    "command_executed",
                    "proof_file_exists",
                    "proof_file_exact",
                )
            )
        else:  # pragma: no cover - loader fixes the set
            raise ValueError(f"unsupported task kind: {task.kind}")
        if success:
            reason = "all required realized VM postconditions observed"
    except (KeyError, TypeError, ValueError) as exc:
        status = "error"
        reason = str(exc)
    return PostconditionResult(task.id, status, bool(status == "ok" and success), reason, evidence, os.getpid())


def as_json(result: PostconditionResult) -> dict[str, Any]:
    return asdict(result)
