"""Which frames of a segment become a conversation, judged by foreground app.

The recorder tells us which application had focus (``lib/app_context``), and a
crowd-cast day is one long unsegmented session: a recording wanders from a
browser to a terminal to a PDF viewer with nothing marking the transitions. Two
different needs fall out of that, and this module is both:

  * GATE (``--include-app`` / ``--exclude-app`` / ``--app-min-frac``) — keep only
    conversations whose dominant app is one you asked for. A browser-only dataset
    is this, and nothing else.
  * SPLIT (``--split-by-app``) — cut a segment at every app switch, so one
    conversation is one application. Without it a single conversation teaches the
    model to switch apps mid-episode for no stated reason, because the reason was
    the demonstrator's, not the screen's.

Two rules are load-bearing:

``accepts`` never lets an UNRESOLVED app satisfy an include list. UNCAPTURED (the
recorder's privacy blackout) and UNKNOWN (could-not-resolve) both mean "no app",
and admitting them under ``--include-app firefox`` would put unlabelled frames in
a dataset whose whole claim is that they are Firefox.

The seam turn is dropped by default. The frame at an app boundary carries an
action label whose window STRADDLES the switch — its keystrokes belong partly to
the app being left and partly to the one arriving. Kept, it is the one turn in a
split conversation guaranteed to be mislabelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipeline.crowdcast.lib.app_context import (
    UNRESOLVED_APPS,
    frame_app_stats,
    iter_app_spans,
    load_app_track,
    resolve_app_selector,
)

__all__ = [
    "APP_UNKNOWN_MODES",
    "AppFilter",
    "AppSpan",
    "app_stats",
    "label_view_frames",
    "plan_app_spans",
    "split_app_selectors",
]

APP_UNKNOWN_MODES = ("keep", "drop")


@dataclass(frozen=True)
class AppFilter:
    """Resolved ``--include-app`` / ``--exclude-app`` / ``--split-by-app`` policy."""

    include: frozenset[str] = frozenset()
    exclude: frozenset[str] = frozenset()
    min_frac: float = 0.0
    unknown: str = "keep"
    split: bool = False
    min_run_frames: int = 1
    drop_seam_turns: bool = True

    @property
    def active(self) -> bool:
        return bool(
            self.include
            or self.exclude
            or self.split
            or self.min_frac > 0.0
            or self.unknown == "drop"
        )

    def accepts(self, app: str | None, frac: float) -> bool:
        """Does a conversation whose dominant app is ``app`` — holding ``frac`` of
        its labelled frames — survive?

        An unresolved or absent label passes only when no include list is set AND
        unknowns are kept. Never claim a segment is Firefox because nothing said
        otherwise.
        """
        if app is None or app in UNRESOLVED_APPS:
            return not self.include and self.unknown == "keep"
        if self.include and app not in self.include:
            return False
        if app in self.exclude:
            return False
        return frac >= self.min_frac

    @classmethod
    def from_args(cls, args: Any) -> AppFilter:
        return cls(
            include=frozenset(
                resolve_app_selector(s) for s in split_app_selectors(args.include_app)
            ),
            exclude=frozenset(
                resolve_app_selector(s) for s in split_app_selectors(args.exclude_app)
            ),
            min_frac=float(args.app_min_frac),
            unknown=str(args.app_unknown),
            split=bool(args.split_by_app),
            min_run_frames=int(args.app_min_run_frames),
            drop_seam_turns=bool(args.app_drop_seam_turns),
        )


def split_app_selectors(values: list[str] | None) -> list[str]:
    """Flatten ``--include-app`` / ``--exclude-app``: repeatable AND
    comma-separated, because labctl renders every recipe arg as a single
    ``--key=value`` and so cannot repeat a flag."""
    out: list[str] = []
    for value in values or []:
        out.extend(
            part for part in str(value).replace(";", ",").split(",") if part.strip()
        )
    return out


def app_stats(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Frame-weighted app scoring over the frames a conversation contains.

    The same function stage 03 writes into its index row, so a prefilter on that
    row and this gate can never disagree.
    """
    return frame_app_stats(frames)


def label_view_frames(view: Any) -> list[dict[str, Any]]:
    """``{app, app_window_switches}`` per frame of a stage-03 FILTER view.

    The sampler writes these itself; a filter view carries only master ticks, so
    they are derived here from the same source on the same clock — the realigned
    keylog the view names, folded onto the master axis. A view with no keylog
    yields unlabelled frames rather than raising: `accepts` already refuses to
    read "no app" as any particular app.

    Each frame is labelled by the app in focus at its OWN tick, and counts the
    switches inside its action window ``[win_start, win_end)`` — which is what
    marks a seam turn, whose label mixes two applications.
    """
    frames = list(getattr(view, "frames", []) or [])
    if not frames:
        return []
    keylog = getattr(view, "keylog_path", None)
    master_fps = float(getattr(view, "master_fps", 0.0) or 0.0)
    if not keylog or master_fps <= 0:
        return [{"app": None, "app_window_switches": 0} for _ in frames]
    axis_end = max(int(f.win_end) for f in frames)
    track = load_app_track(keylog, n_ticks=axis_end, master_fps=master_fps)
    return [
        {
            "app": track.at(int(f.master_idx)),
            "app_window_switches": len(
                track.switches_in(int(f.win_start), int(f.win_end))
            ),
        }
        for f in frames
    ]


@dataclass(frozen=True)
class AppSpan:
    """One ``[lo, hi)`` frame span of a segment that becomes one conversation."""

    app: str | None
    lo: int
    hi: int
    seam_trimmed: bool = False


def plan_app_spans(
    frames: list[dict[str, Any]],
    cfg: AppFilter,
    *,
    min_frames: int,
) -> list[AppSpan]:
    """Which frame spans of one segment become conversations.

    Gate mode returns at most one span — the whole segment, or nothing. Split mode
    returns one span per maximal same-app run that passes the filter and is long
    enough, after trimming the seam turn.
    """
    if not frames:
        return []
    if not cfg.split:
        stats = app_stats(frames)
        if not cfg.accepts(stats["app"], float(stats["app_frac"])):
            return []
        return [AppSpan(stats["app"], 0, len(frames))]

    out: list[AppSpan] = []
    for app, lo, hi in iter_app_spans(frames):
        end, trimmed = hi, False
        if cfg.drop_seam_turns and end > lo and frames[end - 1].get("app_window_switches"):
            end -= 1  # that turn's label mixes this app with the next one
            trimmed = True
        if end - lo < max(cfg.min_run_frames, min_frames, 1):
            continue
        if not cfg.accepts(app, 1.0):
            continue
        out.append(AppSpan(app, lo, end, trimmed))
    return out
