from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .contracts import InfrastructureFailure, VerifierState, sha256_json
from .manifest import Task


ORACLE_RESULT_FIELDS = {
    "task_id",
    "fixture_sha256",
    "oracle_status",
    "MOUSE_SOLVED",
    "semantic_step_index",
    "matched_target_ref",
    "semantic_state_sha256",
    "reason",
    "oracle_pid",
}


class FreshProcessTaskVerifier:
    """Invoke the task-registered oracle in a new Python process."""

    def verify(
        self,
        *,
        task: Task,
        state: dict[str, Any],
        expected_step_index: int | None,
        expected_target_ref: str | None,
        timeout_seconds: float,
    ) -> VerifierState:
        fd, raw_path = tempfile.mkstemp(prefix="paired_eval_state_", suffix=".json")
        state_path = Path(raw_path)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, sort_keys=True)
            command = [
                sys.executable,
                "-m",
                task.verifier_module,
                "--split",
                "development",
                "--task-id",
                task.task_id,
                "--state",
                str(state_path),
            ]
            if expected_step_index is not None:
                command.extend(["--expected-step-index", str(expected_step_index)])
            if expected_target_ref is not None:
                command.extend(["--expected-target-ref", expected_target_ref])
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            if completed.returncode not in {0, 3}:
                raise InfrastructureFailure(
                    "verifier",
                    f"fresh task verifier failed rc={completed.returncode}: "
                    f"{completed.stderr.strip()}",
                )
            try:
                result = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise InfrastructureFailure(
                    "verifier", f"fresh task verifier returned invalid JSON: {exc}"
                ) from exc
            if not isinstance(result, dict) or set(result) != ORACLE_RESULT_FIELDS:
                raise InfrastructureFailure("verifier", "oracle result schema drift")
            if result["task_id"] != task.task_id or result["fixture_sha256"] != task.fixture_sha256:
                raise InfrastructureFailure("verifier", "oracle task identity mismatch")
            if result["oracle_pid"] == os.getpid() or not isinstance(result["oracle_pid"], int):
                raise InfrastructureFailure("verifier", "oracle did not prove process isolation")
            observed_state_sha = sha256_json(state)
            if result["semantic_state_sha256"] != observed_state_sha:
                raise InfrastructureFailure("verifier", "oracle semantic state hash mismatch")
            semantic_step = result["semantic_step_index"]
            if semantic_step is None:
                semantic_step = task.semantic_step_count if result["MOUSE_SOLVED"] else 0
            return VerifierState(
                status=result["oracle_status"],
                task_solved=bool(result["MOUSE_SOLVED"]),
                semantic_step_index=int(semantic_step),
                semantic_state=dict(state),
                matched_target_ref=result["matched_target_ref"],
                reason=str(result["reason"]),
                oracle_pid=int(result["oracle_pid"]),
                verifier_module=task.verifier_module,
            )
        except subprocess.TimeoutExpired as exc:
            raise InfrastructureFailure("verifier", "fresh task verifier timed out") from exc
        finally:
            state_path.unlink(missing_ok=True)
