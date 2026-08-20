"""The sealed suite manifest: a scored tier and a candidate tier.

One fixed development gate — not a benchmark, not a train/dev/test split. The
loader's job is to refuse drift: every cell declares a tier, ids are unique,
`max_steps` is in [1, 12], `final_benchmark` is false, and each tier's kinds are
exactly its own registered set, so neither an orphan kind nor an orphan cell can
survive a rename.

The two tiers exist because a cell whose oracle has never been measured on real
hardware cannot be allowed into a mean that is quoted as calibrated. The scored
tier is the four cells whose oracle reads 4/4 and whose negative reads 0/4 on a
real VM per grammar; the candidate tier is where a new cell lives until its own
oracle and negative read the same way. Promotion is a one-word data change here,
and it moves the scored digest — deliberately, because the gate is then a
different gate.

`scored_sha256` is the canonical-JSON hash of the scored tasks alone, so adding a
candidate cell cannot silently change the identity a scored run records.
`manifest_sha256` is the hash of the whole manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ALLOWED_KINDS",
    "CANDIDATE_KINDS",
    "DevelopmentSuite",
    "DevelopmentTask",
    "NO_SUBMIT_CELLS",
    "SCORED_KINDS",
    "SUBMIT_VERBS",
    "SUITE_PATH",
    "TIERS",
    "canonical_json",
    "load_suite",
]

SUITE_PATH = Path(__file__).with_name("suite.json")

SCORED_KINDS = {
    "terminal_command",
    "terminal_exact_text",
    "open_chrome",
    "focus_terminal_and_type",
}
"""The calibrated four. Oracle 4/4 and negative 0/4 on a real VM, per grammar."""

CANDIDATE_KINDS = {
    "submit_only",
    "staged_confirm",
    "tk_target_click",
    "tk_no_submit_entry",
}
"""Each isolates one mechanism we have measured failing:

  * `submit_only` — submission as a key transition. The cell's guest reader
    records every character that arrived before the newline, so the literal
    `\\n`-inside-`type()` defect (four characters typed, `command_executed:
    false`) reads as a non-empty prefix and a run that never completed, instead
    of hiding inside a cell that also grades typing accuracy.
  * `staged_confirm` — stopping when the first sub-goal looks done. The
    instruction names the goal, not the steps; the second stage is only visible
    on screen. `TERMINATE` is 4.1x under-weighted and has a measured 2.1%
    false-alarm floor, and this is where that shows up as a failed cell rather
    than as a footnote.
  * `tk_target_click` — a target no single move from the declared cursor start
    can reach, since the observed output support collapses to {0, +-1, +-10,
    +-100}. The premise is asserted at setup from the measured bbox, not assumed.
  * `tk_no_submit_entry` — the reflexive Return. Success requires typing into a
    field and clicking a specific button *without* submitting, and `submitted` is
    sticky in the fixture, so a Return anywhere in the window is unrecoverable.
"""

ALLOWED_KINDS = SCORED_KINDS | CANDIDATE_KINDS

TIERS = ("scored", "candidate")

SUBMIT_VERBS = ("press enter", "press return", "hit enter", "and press ⏎")
"""Instruction phrases that make submission part of the cell's own requirement.

Used by `_check_no_submit_consistency` to refuse a cell that is both listed in
`NO_SUBMIT_CELLS` and phrased as a submission."""

NO_SUBMIT_CELLS: frozenset[str] = frozenset({"panel_no_submit_entry"})
"""Cells whose success must not depend on pressing Return, read by failure-mode
indicator D.

`panel_no_submit_entry` is the first genuine entry, and it is a candidate cell —
so D remains structurally zero on the scored tier and every non-zero D reading
ever published against the scored gate is still the old misclassification.

That misclassification: `terminal_exact_text` was listed here, and it requires
submission four ways over — its instruction ends *"and press Enter"*, its guest
fixture completes an `IFS= read -r` only once a newline arrives, its oracle
requires the capture file that the line *after* `read` writes, and its oracle
control arm emits `0 0 0 ; +Return -Return`. D was penalising the behaviour four
independent parts of the cell require, including the gold plan. D is a
`@vf.metric`, never a reward, with `no_submit` read in exactly one place
(`indicators.py`), so no published pass count — including the Phase-B-compact
2/4 — was ever a function of this flag.

The loader refuses a cell that is both listed here and phrased as a submission,
and refuses an entry naming no cell in the suite."""


@dataclass(frozen=True)
class DevelopmentTask:
    id: str
    kind: str
    tier: str
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
    scored_sha256: str

    def by_id(self, task_id: str) -> DevelopmentTask:
        matches = [task for task in self.tasks if task.id == task_id]
        if len(matches) != 1:
            raise KeyError(task_id)
        return matches[0]

    def for_tier(self, tier: str) -> tuple[DevelopmentTask, ...]:
        if tier not in TIERS:
            raise ValueError(f"unknown suite tier {tier!r}; known: {list(TIERS)}")
        return tuple(task for task in self.tasks if task.tier == tier)


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _check_no_submit_consistency(tasks: list["DevelopmentTask"]) -> None:
    """Refuse a suite whose instructions and whose indicators disagree.

    An indicator must never penalise an action the cell's own instruction demands.
    Two ways to get that wrong, both refused here:

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
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise ValueError("sign-of-life suite schema mismatch")
    if value.get("role") != "single_fixed_development_gate":
        raise ValueError("suite must remain a single fixed development gate")
    if value.get("final_benchmark") is not False:
        raise ValueError("development suite must not be labelled as a final benchmark")
    raw_tasks = value.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("sign-of-life suite carries a list of cells")
    tasks: list[DevelopmentTask] = []
    seen: set[str] = set()
    for raw in raw_tasks:
        if not isinstance(raw, dict):
            raise ValueError("task must be an object")
        task_id = raw.get("id")
        kind = raw.get("kind")
        tier = raw.get("tier")
        instruction = raw.get("instruction")
        expected = raw.get("expected")
        max_steps = raw.get("max_steps")
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ValueError("task ids must be unique non-empty strings")
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"unsupported task kind: {kind!r}")
        if tier not in TIERS:
            raise ValueError(f"{task_id}: tier must be one of {list(TIERS)}, got {tier!r}")
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
        tasks.append(DevelopmentTask(task_id, kind, tier, instruction, expected, max_steps))
    scored = [task for task in tasks if task.tier == "scored"]
    if len(scored) != 4 or {task.kind for task in scored} != SCORED_KINDS:
        raise ValueError("the scored tier is the four calibrated cells, one per kind")
    candidate_kinds = [task.kind for task in tasks if task.tier == "candidate"]
    if sorted(candidate_kinds) != sorted(CANDIDATE_KINDS):
        raise ValueError("candidate tier coverage drift: one cell per candidate kind")
    _check_no_submit_consistency(tasks)
    return DevelopmentSuite(
        suite_id=str(value["suite_id"]),
        role=str(value["role"]),
        final_benchmark=False,
        tasks=tuple(tasks),
        manifest_sha256=hashlib.sha256(canonical_json(value)).hexdigest(),
        scored_sha256=hashlib.sha256(
            canonical_json([raw for raw in raw_tasks if raw.get("tier") == "scored"])
        ).hexdigest(),
    )
