"""The sealed four-cell suite manifest.

One fixed development gate — not a benchmark, not a train/dev/test split. The
loader's job is to refuse drift: exactly four cells, exactly the four capability
kinds, unique ids, `max_steps` in [1, 12], `final_benchmark` false. Every check is
kept from the original loader, and `manifest_sha256` is the canonical-JSON hash of
the same bytes, so a run's recorded suite hash stays comparable across the
refactor (`1bf13a84808fd144cf6565c61a303d97e37716d7129f22f9ae46a4dcc3bfbaac`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ALLOWED_KINDS",
    "DevelopmentSuite",
    "DevelopmentTask",
    "NO_SUBMIT_CELLS",
    "SUBMIT_VERBS",
    "SUITE_PATH",
    "canonical_json",
    "load_suite",
]

SUITE_PATH = Path(__file__).with_name("suite.json")

ALLOWED_KINDS = {
    "terminal_command",
    "terminal_exact_text",
    "open_chrome",
    "focus_terminal_and_type",
}

SUBMIT_VERBS = ("press enter", "press return", "hit enter", "and press ⏎")
"""Instruction phrases that make submission part of the cell's own requirement.

Used by `_check_no_submit_consistency` to make the item-14 defect class
structurally unreachable rather than a comment."""

NO_SUBMIT_CELLS: frozenset[str] = frozenset()
"""Cells whose success must not depend on pressing Return, read by failure-mode
indicator D.

**Empty, and that is the resolved state, not an oversight.** No cell in this suite
qualifies. `terminal_exact_text`, the only plausible candidate, requires submission
four ways over:

  * the cell's own instruction ends *"and press Enter"*;
  * its guest fixture (`guest._setup_terminal_exact_text`) completes an
    `IFS= read -r` only once a newline arrives;
  * its oracle requires `capture_file_exists`, and the capture file is written by the
    line *after* `read` returns — so the file's existence IS evidence of a Return;
  * its **oracle control arm** — the arm the calibration defines as 4/4 — emits
    `0 0 0 ; +Return -Return`.

So indicator D was penalising the behaviour four independent parts of the cell
require, including the gold plan. A cell whose own instruction demands submission is
not a no-submit cell.

Two consequences a reader needs. First, because this was the *only* entry, indicator
D never had a valid cell to fire on in this suite: every non-zero D reading on the
sign-of-life gate was this misclassification, and every future one is now
structurally zero here until a genuine no-submit cell is added. Second, D is a
`@vf.metric`, never a reward, and `no_submit` is read in exactly one place
(`indicators.py:231`) — so no published pass count, including the Phase-B-compact
2/4, was ever a function of this flag.

Adding a genuine no-submit cell is supported and checked: the loader refuses a cell
that is both listed here and phrased as a submission (`SUBMIT_VERBS`)."""


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


def _check_no_submit_consistency(tasks: list["DevelopmentTask"]) -> None:
    """Refuse a suite whose instructions and whose indicators disagree.

    This is the general form of the item-14 defect: an indicator must never penalise
    an action the cell's own instruction demands. Two ways to get that wrong, both
    refused here rather than left as a comment:

      * a cell listed in `NO_SUBMIT_CELLS` whose instruction says "press Enter" — the
        exact contradiction that made indicator D fire on the gold control arm;
      * an entry in `NO_SUBMIT_CELLS` naming no cell in the suite, which is how a
        rename silently stops applying the flag.
    """
    by_id = {task.id: task for task in tasks}
    unknown = sorted(NO_SUBMIT_CELLS - set(by_id))
    if unknown:
        raise ValueError(
            f"NO_SUBMIT_CELLS names cells that are not in the suite: {unknown}; "
            "a renamed cell must not silently lose the flag"
        )
    for cell_id in sorted(NO_SUBMIT_CELLS):
        instruction = by_id[cell_id].instruction.casefold()
        found = [verb for verb in SUBMIT_VERBS if verb in instruction]
        if found:
            raise ValueError(
                f"{cell_id}: listed in NO_SUBMIT_CELLS but its own instruction "
                f"requires submission ({found!r}). Indicator D would penalise the "
                "behaviour the cell demands, which is what it means for the "
                "classification rather than the indicator to be wrong."
            )


def load_suite(path: Path | None = None) -> DevelopmentSuite:
    value = json.loads((path or SUITE_PATH).read_text(encoding="utf-8"))
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
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or not 1 <= max_steps <= 12
        ):
            raise ValueError(f"{task_id}: max_steps outside [1, 12]")
        seen.add(task_id)
        tasks.append(DevelopmentTask(task_id, kind, instruction, expected, max_steps))
    if {task.kind for task in tasks} != ALLOWED_KINDS:
        raise ValueError("suite capability coverage drift")
    _check_no_submit_consistency(tasks)
    return DevelopmentSuite(
        suite_id=str(value["suite_id"]),
        role=str(value["role"]),
        final_benchmark=False,
        tasks=tuple(tasks),
        manifest_sha256=hashlib.sha256(canonical_json(value)).hexdigest(),
    )
