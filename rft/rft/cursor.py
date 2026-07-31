"""Cursor-motion measurement that does not trust ``info.cursor_after``.

**Defect #4.** For the ``computer_use``/``move_rel`` grammar, the per-step
``info.cursor_after`` field written by the rollout harness is stale: it equals
``cursor_before`` in 97.4% of 17,090 recorded steps. Any "did the cursor move?"
metric computed from ``cursor_after - cursor_before`` therefore reads ~0 motion
for a policy that may well be moving the cursor.

The fix is structural, not a patch: cursor motion is a property of a *pair of
consecutive steps*, so measure it **between** steps, using each step's
``cursor_before`` (which is observed fresh at the top of the step) and the next
step's ``cursor_before``. This module provides that, plus a loud detector for
the stale-field condition so the defect can never silently return.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rft.errors import MissingFieldError, SchemaError


@dataclass(frozen=True)
class StepCursor:
    """The cursor position observed at the *start* of one rollout step."""

    step_index: int
    x: float
    y: float


@dataclass(frozen=True)
class CursorMotionReport:
    """Between-step cursor motion for one trajectory.

    ``displacements`` has ``len(steps) - 1`` entries: entry *i* is the motion
    observed between the start of step *i* and the start of step *i+1*, i.e.
    the motion actually effected by step *i*.
    """

    displacements: tuple[tuple[float, float], ...]
    n_steps: int

    @property
    def n_transitions(self) -> int:
        return len(self.displacements)

    @property
    def n_moved(self) -> int:
        return sum(1 for dx, dy in self.displacements if (dx, dy) != (0.0, 0.0))

    @property
    def moved_fraction(self) -> float:
        if not self.displacements:
            raise SchemaError(
                "a single-step trajectory has no between-step transitions; "
                "cursor motion is undefined (do not report 0.0 for it)"
            )
        return self.n_moved / len(self.displacements)


def _read_xy(info: Any, key: str, step_index: int) -> tuple[float, float]:
    if not isinstance(info, dict):
        raise SchemaError(f"step {step_index}: `info` is {type(info).__name__}, expected dict")
    if key not in info:
        raise MissingFieldError(f"steps[{step_index}].info.{key}", available=list(info.keys()))
    value = info[key]
    if isinstance(value, dict):
        if "x" not in value or "y" not in value:
            raise MissingFieldError(f"steps[{step_index}].info.{key}.x/.y",
                                    available=list(value.keys()))
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise SchemaError(
        f"step {step_index}: info.{key} is {value!r}; expected [x, y] or {{'x':..,'y':..}}"
    )


def cursor_positions(steps: Sequence[Any], *, key: str = "cursor_before") -> list[StepCursor]:
    """Extract the per-step observed cursor position.

    ``key`` defaults to ``cursor_before`` on purpose: that is the field observed
    fresh at the top of each step. ``cursor_after`` is the stale one.
    """
    out: list[StepCursor] = []
    for i, step in enumerate(steps):
        info = step.get("info") if isinstance(step, dict) else None
        x, y = _read_xy(info, key, i)
        out.append(StepCursor(step_index=i, x=x, y=y))
    return out


def cursor_motion_between_steps(
    steps: Sequence[Any], *, key: str = "cursor_before"
) -> CursorMotionReport:
    """Measure cursor motion BETWEEN consecutive steps.

    This is the only supported way to measure cursor motion in this package.
    There is deliberately no helper that subtracts ``cursor_after`` from
    ``cursor_before`` within a single step.

    Raises:
        MissingFieldError / SchemaError: on any step whose cursor observation is
            absent or malformed. A trajectory with a broken step is not a
            trajectory with zero motion.
    """
    positions = cursor_positions(steps, key=key)
    displacements = tuple(
        (positions[i + 1].x - positions[i].x, positions[i + 1].y - positions[i].y)
        for i in range(len(positions) - 1)
    )
    return CursorMotionReport(displacements=displacements, n_steps=len(positions))


#: Above this fraction of steps with ``cursor_after == cursor_before``, the
#: field is presumed stale rather than the policy presumed motionless. The
#: observed defect rate was 0.974.
STALE_CURSOR_AFTER_THRESHOLD: float = 0.90


@dataclass(frozen=True)
class StaleCursorAfterReport:
    n_steps: int
    n_equal: int
    threshold: float

    @property
    def equal_fraction(self) -> float:
        if not self.n_steps:
            raise SchemaError("no steps: staleness is undefined, not 0.0")
        return self.n_equal / self.n_steps

    @property
    def is_stale(self) -> bool:
        return self.equal_fraction >= self.threshold

    def describe(self) -> str:
        return (
            f"info.cursor_after == info.cursor_before on {self.n_equal}/{self.n_steps} steps "
            f"({self.equal_fraction:.1%}); threshold {self.threshold:.0%} -> "
            f"{'STALE (defect #4) - measure motion BETWEEN steps' if self.is_stale else 'ok'}"
        )


def detect_stale_cursor_after(steps: Sequence[Any]) -> StaleCursorAfterReport:
    """Report how often ``cursor_after`` equals ``cursor_before``.

    Every eval that touches cursor motion runs this and prints
    :meth:`StaleCursorAfterReport.describe`, so that a harness which starts
    writing a stale field again is caught in the diagnostics rather than in a
    research conclusion six weeks later.
    """
    n_equal = 0
    n_steps = 0
    for i, step in enumerate(steps):
        info = step.get("info") if isinstance(step, dict) else None
        if not isinstance(info, dict) or "cursor_after" not in info:
            # No cursor_after at all is fine (and preferable) - nothing to be
            # stale. Skip rather than count, and do not invent a value.
            continue
        before = _read_xy(info, "cursor_before", i)
        after = _read_xy(info, "cursor_after", i)
        n_steps += 1
        if before == after:
            n_equal += 1
    return StaleCursorAfterReport(
        n_steps=n_steps, n_equal=n_equal, threshold=STALE_CURSOR_AFTER_THRESHOLD
    )
