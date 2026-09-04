"""The sealed suite manifest: a scored tier and a candidate tier.

One fixed development gate — not a benchmark, not a train/dev/test split. The
loader's job is to refuse drift: every cell declares a tier, ids are unique,
`max_steps` is in [1, 12], `final_benchmark` is false, and each tier's kinds are
exactly its own registered set, so neither an orphan kind nor an orphan cell can
survive a rename.

The two tiers exist because a cell whose oracle has never been measured on real
hardware cannot be allowed into a mean that is quoted as calibrated, and because
a cell the reference model already passes cannot measure a gain. A cell reaches
the scored tier only with both readings on real VMs: its own oracle passing and
negative failing, per grammar, and a reference model failing it. Promotion is a
one-word data change here, and it moves the scored digest — deliberately, because
the gate is then a different gate.

The tier earned itself on its first use. Two cells were argued into existence as
"the base model fails premature termination", on the strength of the base failing
the analogous scored cell 3/3 — and when the probe finally ran against the cells
themselves, the base passed both 3/3 and failed the two cells that had been filed
as mere regression detectors. The proxy was inverted; the tier is what kept the
wrong pair out of the mean.

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
"""Four cells, all four fully calibrated, none yet admissible to the scored tier.
Calibration is not what blocks them; a valid reference reading is.

Rule 1 is satisfied for every one on real VMs: oracle 3/3 per cell in
`ordered_events_v3` (141317), `native_absolute` (141322) and `deltatype_v2`
(141320); negative 0/4 in all three (141316, 141323, 141321), and for the two
panel cells 0/3 at three trials as well (141407, 141408, 141409). The scripted
arms resolve every coordinate from the fixture's own measurement, so none of that
depends on model output.

  * `submit_only` -- submission as a key transition. The guest reader publishes
    every character that arrived before the newline, so a literal `\\n` inside
    `type()` (3 of 3 Phase-B draws) reads as a two-character prefix on a read
    that never completed. HELD: off-the-shelf Qwen3-VL-4B passes it 3/3 (141319),
    and the cell needs no click, so that reading is clean. A cell the reference
    passes cannot register a gain.
  * `staged_confirm` -- stopping when the first sub-goal looks done; the
    instruction names the goal and only the screen names the second stage. HELD
    on the same clean evidence: base 3/3, no click involved.
  * `tk_target_click` -- a target no single move from the declared cursor start
    reaches, so passing it requires chaining moves rather than one lucky jump.
    The premise is asserted at setup from the measured bbox.
  * `tk_no_submit_entry` -- the reflexive Return: type into a field and click a
    named button *without* submitting, with `submitted` sticky in the fixture.

The last two were promoted on a base reading of 0/3 each and returned here,
because that 0/3 was the arm failing rather than the model. The off-the-shelf arm
emits 0-999-per-axis coordinates while `native_absolute`'s prompt declares
absolute screen pixels, so its clicks land ~460 px left and ~50 px high: in six of
six episodes the de-normalised point falls INSIDE the measured target, 6-31 px
from centre, on targets 172x31 and 230x23. The model grounds these cells
correctly, and under a matched convention it plausibly passes both -- which would
make them base-passing cells like the two above. Unproven, not rejected.

A valid re-probe has to satisfy three things, because pass/fail alone hid that
artefact for six episodes:

  * a matched coordinate convention, established from the run's own prompt and
    the model's own output rather than assumed;
  * targets far from the origin. Near (0,0) the conventions nearly coincide --
    emitted (15,59), de-normalised (29,64), dock icon at (35,60) -- so a cell
    whose target sits near the origin cannot expose a convention error at all,
    and the error grows with distance from it;
  * residuals reported, not pass/fail alone. The distance from the click to the
    measured target centre is what turned "the model cannot ground this" into
    "the arm mis-reads its output"."""

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
so D remains structurally zero on the scored tier, and every non-zero D reading
ever published against the scored gate is the old misclassification rather than a
signal. D becomes a live scored signal only if that cell is promoted.

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
    if len(scored) != len(SCORED_KINDS) or {task.kind for task in scored} != SCORED_KINDS:
        raise ValueError("the scored tier is the calibrated cells, one per kind")
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
