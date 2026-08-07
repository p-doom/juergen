"""Postconditions decided from realized guest state.

Every clause is recorded separately in `evidence` so a partial pass is diagnosable
without re-running the VM, and an unreadable state is `status="error"` rather than
`success=False` — collapsing those two is how a broken probe becomes a silent 0/4.
"""

from __future__ import annotations

from typing import Any

from evals.oracles import OracleOutcome

__all__ = ["evaluate_postcondition", "history_has_exact"]


def history_has_exact(history: object, command: str) -> bool:
    """True iff a shell-history line is exactly `command`.

    The leading-digit strip handles numbered `history` output; the equality is
    exact on purpose — a cell that accepts a superstring accepts `ls -la` for `ls`.
    """
    if not isinstance(history, str):
        return False
    return any(
        line.strip().lstrip("0123456789 ").strip() == command
        for line in history.splitlines()
    )


def evaluate_postcondition(
    task_id: str, kind: str, expected: dict[str, Any], state: dict[str, Any]
) -> OracleOutcome:
    status = "ok"
    success = False
    reason = "required realized VM state was not observed"
    evidence: dict[str, Any] = {}
    try:
        if state.get("schema_version") != 1 or state.get("task_id") != task_id:
            raise ValueError("state identity/schema mismatch")
        if kind == "terminal_command":
            command = str(expected["command"])
            marker = str(expected["listing_marker"])
            command_executed = history_has_exact(state.get("history"), command)
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
        elif kind == "terminal_exact_text":
            wanted = str(expected["text"])
            observed = state.get("captured_text")
            evidence = {
                "capture_file_exists": state.get("capture_file_exists") is True,
                "exact_text_match": observed == wanted,
                "expected_text": wanted,
                "observed_text": observed,
            }
            success = bool(evidence["capture_file_exists"] and evidence["exact_text_match"])
        elif kind == "open_chrome":
            active = str(state.get("active_window", "")).casefold()
            markers = [str(value).casefold() for value in expected["active_window_class_any"]]
            active_matches = any(marker in active for marker in markers)
            chrome_process = bool(state.get("chrome_process"))
            evidence = {
                "chrome_process_observed": chrome_process,
                "chrome_is_foreground": active_matches,
                "active_window": state.get("active_window"),
                "window_inventory": state.get("windows"),
            }
            success = chrome_process and active_matches
        elif kind == "focus_terminal_and_type":
            command = str(expected["command"])
            content = str(expected["content"])
            active = str(state.get("active_window", "")).casefold()
            evidence = {
                "terminal_is_foreground": "terminal" in active,
                "command_executed": history_has_exact(state.get("history"), command),
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
        else:  # pragma: no cover - the loader fixes the set
            raise ValueError(f"unsupported task kind: {kind}")
        if success:
            reason = "all required realized VM postconditions observed"
    except (KeyError, TypeError, ValueError) as exc:
        status = "error"
        reason = str(exc)
    return OracleOutcome(
        task_id=task_id,
        status=status,
        success=bool(status == "ok" and success),
        reason=reason,
        evidence=evidence,
    )
