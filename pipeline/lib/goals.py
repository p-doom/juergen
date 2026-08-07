"""Goals: the uniform annotation contract + projection onto training views.

A goal is a half-open interval of MASTER ticks ``[start_master_idx,
end_master_idx)`` with an instruction — produced by any stage-03b annotation
method, consumed by stage 04. Master indices are the universal coordinate
system: an annotation made at k fps projects onto a training view at any
x fps because both are mappings over the same integer axis. View-local frame
indices are never persisted; ``view_span_to_master`` converts them at the
annotation stage's write time.

Projection rule: membership is tested against the view's actual selected
frames (``master_idx ∈ [start, end)``), never derived by fps arithmetic —
masked slots mean the j-th frame is not at tick j*stride.
``snap_start="before"`` (default) additionally includes the last selected
frame at-or-before the goal start when no member frame sits exactly at it:
that frame is the observation the goal's first action was taken from.
Rejections are counted, never silent.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.lib.common import read_jsonl
from pipeline.lib.views import SegmentView, ViewFrame

SNAP_START_MODES = ("before", "inside")

# Required keys of a goals.jsonl row (the uniform contract across methods).
REQUIRED_GOAL_KEYS = (
    "goal_id",
    "segment_id",
    "recording_id",
    "start_master_idx",
    "end_master_idx",
    "instruction",
    "method",
    "model",
    "prompt_pack_sha",
)
# Optional: instruction_variants, anchor, grounding, plan, plan_flags.


def validate_goal_row(row: dict[str, Any]) -> None:
    """Raise ValueError if ``row`` violates the goals contract."""
    missing = [k for k in REQUIRED_GOAL_KEYS if k not in row]
    if missing:
        raise ValueError(f"goal row missing keys {missing}: {row.get('goal_id')!r}")
    start, end = row["start_master_idx"], row["end_master_idx"]
    if not (isinstance(start, int) and isinstance(end, int)):
        raise ValueError(
            f"goal {row['goal_id']!r}: master indices must be integers "
            f"(got {start!r}, {end!r}) — view-local or float coordinates are a bug"
        )
    if not 0 <= start < end:
        raise ValueError(f"goal {row['goal_id']!r}: bad interval [{start}, {end})")
    if not (isinstance(row["instruction"], str) and row["instruction"].strip()):
        raise ValueError(f"goal {row['goal_id']!r}: empty instruction")
    variants = row.get("instruction_variants")
    if variants is not None and not (
        isinstance(variants, list) and all(isinstance(v, str) for v in variants)
    ):
        raise ValueError(f"goal {row['goal_id']!r}: instruction_variants must be a list of strings")


def load_goals(goals_path: Path) -> list[dict[str, Any]]:
    """Read + validate a goals.jsonl; raises on the first malformed row."""
    rows = read_jsonl(goals_path)
    for row in rows:
        validate_goal_row(row)
    return rows


def goals_by_segment(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row["segment_id"]), []).append(row)
    for seg_rows in out.values():
        seg_rows.sort(key=lambda r: (int(r["start_master_idx"]), int(r["end_master_idx"])))
    return out


def view_span_to_master(view: SegmentView, start_view_idx: int, end_view_idx: int) -> tuple[int, int]:
    """Convert a half-open view-local frame span ``[start, end)`` to a master
    interval, using the frames' actual window boundaries: from the first
    frame's tick to the last frame's window end. This is the only path by which
    an annotation method's view-local output reaches disk."""
    if not 0 <= start_view_idx < end_view_idx <= len(view.frames):
        raise ValueError(
            f"view span [{start_view_idx}, {end_view_idx}) out of range for "
            f"{len(view.frames)} frames of {view.segment_id}"
        )
    return view.frames[start_view_idx].master_idx, view.frames[end_view_idx - 1].win_end


@dataclass
class GoalProjection:
    goal: dict[str, Any]
    frames: list[ViewFrame]  # ordered subset of the view's frames
    snapped_start: bool  # True when snap_start added the prior observation frame


@dataclass
class ProjectionStats:
    n_goals: int = 0
    n_projected: int = 0
    n_empty_projection: int = 0
    n_too_few_frames: int = 0
    n_snapped: int = 0
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def _reject(self, goal: dict[str, Any], reason: str) -> None:
        self.rejected.append({"goal_id": goal.get("goal_id"), "reason": reason})


def project_goals(
    goals: Iterable[dict[str, Any]],
    view: SegmentView,
    *,
    snap_start: str = "before",
    min_frames: int = 1,
) -> tuple[list[GoalProjection], ProjectionStats]:
    """Project goals (master intervals) onto a view's actual selected frames.

    Rejections (never silent, all counted):
      * ``empty_projection`` — no selected frame inside the goal interval;
      * ``too_few_frames``   — fewer than ``min_frames`` after snapping.
    """
    if snap_start not in SNAP_START_MODES:
        raise ValueError(f"snap_start must be one of {SNAP_START_MODES}, got {snap_start!r}")
    stats = ProjectionStats()
    projections: list[GoalProjection] = []
    frames = view.frames  # sorted by master_idx by construction

    for goal in goals:
        stats.n_goals += 1
        if str(goal["segment_id"]) != view.segment_id:
            raise ValueError(
                f"goal {goal.get('goal_id')!r} belongs to segment {goal['segment_id']!r}, "
                f"not {view.segment_id!r}"
            )
        start, end = int(goal["start_master_idx"]), int(goal["end_master_idx"])
        members = [f for f in frames if start <= f.master_idx < end]
        if not members:
            stats.n_empty_projection += 1
            stats._reject(goal, "empty_projection")
            continue
        snapped = False
        if snap_start == "before" and members[0].master_idx > start:
            # The frame the goal's first action was taken from: the last
            # selected frame before the goal start (its window covers it).
            prior = [f for f in frames if f.master_idx < start]
            if prior:
                members = [prior[-1], *members]
                snapped = True
                stats.n_snapped += 1
        if len(members) < min_frames:
            stats.n_too_few_frames += 1
            stats._reject(goal, "too_few_frames")
            continue
        stats.n_projected += 1
        projections.append(GoalProjection(goal=goal, frames=members, snapped_start=snapped))

    return projections, stats


def assert_same_artifact(recorded_id: str, current_id: str, *, what: str) -> None:
    """Refuse joins across different builds of the same logical artifact."""
    if recorded_id != current_id:
        raise ValueError(
            f"{what} mismatch: this input was built against\n  {recorded_id}\n"
            f"but the join target is\n  {current_id}\n"
            "— rebuild the downstream artifact against matching inputs."
        )
