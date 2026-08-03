from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_KINDS = {
    "terminal_command",
    "terminal_exact_text",
    "open_chrome",
    "focus_terminal_and_type",
}


@dataclass(frozen=True)
class DevelopmentTask:
    id: str
    kind: str
    instruction: str
    expected: dict[str, Any]
    max_steps: int


@dataclass(frozen=True)
class DevelopmentSuite:
    suite_id: str
    role: str
    final_benchmark: bool
    tasks: tuple[DevelopmentTask, ...]
    manifest_sha256: str

    def by_id(self, task_id: str) -> DevelopmentTask:
        matches = [task for task in self.tasks if task.id == task_id]
        if len(matches) != 1:
            raise KeyError(task_id)
        return matches[0]


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_suite(path: Path | None = None) -> DevelopmentSuite:
    path = path or Path(__file__).with_name("suite.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("sign-of-life suite schema mismatch")
    if value.get("role") != "single_fixed_development_gate":
        raise ValueError("suite must remain a single fixed development gate")
    if value.get("final_benchmark") is not False:
        raise ValueError("development suite must not be labelled as a final benchmark")
    raw_tasks = value.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 4:
        raise ValueError("sign-of-life v2 must contain exactly four fixed cells")
    tasks: list[DevelopmentTask] = []
    seen: set[str] = set()
    for raw in raw_tasks:
        if not isinstance(raw, dict):
            raise ValueError("task must be an object")
        task_id = raw.get("id")
        kind = raw.get("kind")
        instruction = raw.get("instruction")
        expected = raw.get("expected")
        max_steps = raw.get("max_steps")
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ValueError("task ids must be unique non-empty strings")
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"unsupported task kind: {kind!r}")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError(f"{task_id}: instruction missing")
        if not isinstance(expected, dict) or not expected:
            raise ValueError(f"{task_id}: expected state missing")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= 12:
            raise ValueError(f"{task_id}: max_steps outside [1, 12]")
        seen.add(task_id)
        tasks.append(DevelopmentTask(task_id, kind, instruction, expected, max_steps))
    if {task.kind for task in tasks} != ALLOWED_KINDS:
        raise ValueError("suite capability coverage drift")
    return DevelopmentSuite(
        suite_id=str(value["suite_id"]),
        role=str(value["role"]),
        final_benchmark=False,
        tasks=tuple(tasks),
        manifest_sha256=hashlib.sha256(canonical_json(value)).hexdigest(),
    )
