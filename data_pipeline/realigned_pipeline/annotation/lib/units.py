"""Unit chunking + frame rendering for annotation methods.

An ``AnnotationUnit`` is the work quantum stage 03b dispatches to a labeler
model: one segment's view (frames @k fps within filter survivors), split into
window-units ONLY when the segment exceeds the model's context budget. Cuts
are snapped to a command/prompt SUBMISSION (Return/Enter) or a real time-gap —
never mid typing-burst — so one action (and its goal) is not split across two
windows; non-final windows also see a trailing tail-buffer of context frames
(goals that START in the buffer belong to the next window and are dropped).

Coordinates: methods see the segment's dense view-local frame indices (the
``frame <N>`` labels sent to the model ARE view indices); nothing view-local
is ever persisted — the stage converts spans to master intervals at write time
(lib/goals.view_span_to_master).
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from realigned_pipeline.lib.image_store import read_jpeg_bytes
from realigned_pipeline.lib.views import SegmentView, ViewFrame

# ---------------------------------------------------------------------------
# Frame rendering (ar:// store -> in-memory data URLs; no jpegs on disk)
# ---------------------------------------------------------------------------


def resize_to_height(frame: Any, height: int) -> Any:
    if height <= 0 or frame.shape[0] == height:
        return frame
    scale = height / frame.shape[0]
    width = max(2, round((frame.shape[1] * scale) / 2) * 2)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(frame, (width, height), interpolation=interp)


def frames_to_data_urls(
    refs: list[str],
    target_height: int = 0,
    jpeg_quality: int = 80,
) -> list[str]:
    """In-memory ``data:image/jpeg`` URLs read straight from each ``ar://``
    grain URI. Bytes pass through verbatim unless ``target_height`` is below
    the stored height, in which case every frame is decoded, downscaled and
    re-encoded (the overflow-clip rescue knob). One probe per segment."""
    if not refs:
        return []
    resize = False
    if target_height and target_height > 0:
        probe = cv2.imdecode(np.frombuffer(read_jpeg_bytes(refs[0]), np.uint8), cv2.IMREAD_COLOR)
        if probe is None:
            raise RuntimeError(f"could not decode stored frame: {refs[0]}")
        resize = probe.shape[0] > target_height
    urls: list[str] = []
    for ref in refs:
        raw = read_jpeg_bytes(ref)
        if resize:
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"could not decode stored frame: {ref}")
            ok, enc = cv2.imencode(
                ".jpg", resize_to_height(img, target_height),
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            if not ok:
                raise RuntimeError(f"could not re-encode frame: {ref}")
            raw = enc.tobytes()
        urls.append("data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"))
    return urls


def est_frame_tokens(ref: str) -> int:
    """Vision tokens for one stored frame: ceil(h/28)*ceil(w/28) (verified
    within ~2% of real prompt_tokens). Budgets frames-per-context."""
    img = cv2.imdecode(np.frombuffer(read_jpeg_bytes(ref), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not decode frame for token estimate: {ref}")
    h, w = img.shape[:2]
    return math.ceil(h / 28) * math.ceil(w / 28)


# ---------------------------------------------------------------------------
# Submission-aware window planning (ported from annotation_pipeline)
# ---------------------------------------------------------------------------


def frame_activity(action: str | None) -> str:
    """Classify a frame by its derived action label: "idle" (NO_OP), "type"
    (mid keyboard-burst), or "other". A lone idle frame is NOT a reliable
    boundary — people pause a beat mid-typing — so cutting keys on SUBMISSIONS
    and real time-gaps, using this only to avoid cutting between two "type"
    frames."""
    if not action or action == "NO_OP":
        return "idle"
    tail = action.split(";", 1)[1] if ";" in action else action
    if any(tok in tail for tok in ("Key", "Return", "Backspace", "Space", "Enter",
                                   "Digit", "Num", "Minus", "Slash", "Period", "Comma")):
        return "type"
    return "other"


def _is_submission(action: str | None) -> bool:
    """A Return/Enter keypress commits a typed command / prompt / message —
    the real boundary BETWEEN activities in continuous work."""
    return bool(action) and ("Return" in action or "Enter" in action)


def _best_cut(lo: int, hi: int, ideal: int, actions: list[str | None],
              times: list[float] | None, big_gap_s: float) -> int:
    """Cut index in [lo, hi] (window splits BEFORE this frame) least likely to
    slice an in-progress action, nearest ``ideal`` among equals."""
    best, best_key = ideal, None
    for i in range(lo, hi + 1):
        a_prev, a_cur = frame_activity(actions[i - 1]), frame_activity(actions[i])
        gap = (times[i] - times[i - 1]) if times else 0.0
        if _is_submission(actions[i - 1]) or gap >= big_gap_s:
            cost = 0                              # just submitted, or a real pause
        elif a_prev != "type" and a_cur != "type":
            cost = 1
        elif a_prev != "type" or a_cur != "type":
            cost = 2
        else:
            cost = 3                              # both sides typing — mid-burst
        key = (cost, abs(i - ideal))
        if best_key is None or key < best_key:
            best_key, best = key, i
    return best


def plan_windows(n: int, max_frames: int, overlap: int = 0, *,
                 actions: list[str | None] | None = None,
                 times: list[float] | None = None,
                 slack: int = 0, big_gap_s: float = 6.0) -> list[tuple[int, int]]:
    """Partition [0, n) into (lo, hi) half-open windows, each <= max_frames.
    A split happens ONLY when needed (n > max_frames); otherwise one window.
    With ``actions`` + ``slack``, each interior boundary snaps within ±slack to
    a submission or genuine time-gap, never inside an unsubmitted typing burst."""
    if n <= 0:
        return []
    if n <= max_frames:
        return [(0, n)]
    snap = bool(actions) and slack > 0
    n_win = math.ceil(n / max_frames)             # fewest windows that fit budget
    boundaries: list[int] = []
    prev = 0
    for k in range(1, n_win):
        ideal = round(k * n / n_win)
        # Feasible range keeping the left window and all remaining windows <= max_frames.
        lo = max(prev + 1, n - (n_win - k) * max_frames)
        hi = min(prev + max_frames, n - 1)
        if snap:
            lo = max(lo, ideal - slack)
            hi = min(hi, ideal + slack)
            if lo > hi:                            # snap range infeasible — forced cut
                p = min(max(ideal, prev + 1), prev + max_frames, n - 1)
            else:
                p = _best_cut(lo, hi, min(max(ideal, lo), hi), actions, times, big_gap_s)
        else:
            p = min(max(ideal, prev + 1), prev + max_frames, n - 1)
        boundaries.append(p)
        prev = p
    bounds = [0, *boundaries, n]
    wins: list[tuple[int, int]] = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if lo >= hi:
            continue
        lo = max(0, lo - (overlap if i > 0 else 0))
        hi = min(n, hi + (overlap if i < len(bounds) - 2 else 0))
        wins.append((lo, hi))
    return wins


# ---------------------------------------------------------------------------
# AnnotationUnit
# ---------------------------------------------------------------------------


@dataclass
class AnnotationUnit:
    """One labeler work quantum: an owned span [lo, hi) of a segment view's
    frames plus ``tail_buffer`` trailing context frames. ``actions`` are the
    derived canonical labels for ALL of the segment's view frames (used for
    keystroke-burst snapping and window planning; method-internal only)."""

    unit_id: str
    view: SegmentView
    window_index: int
    n_windows: int
    lo: int  # owned view-index range [lo, hi)
    hi: int
    tail_buffer: int  # context frames past hi actually sent
    actions: list[str]

    @property
    def segment_id(self) -> str:
        return self.view.segment_id

    @property
    def sent_frames(self) -> list[ViewFrame]:
        return self.view.frames[self.lo : self.hi + self.tail_buffer]

    @property
    def sent_view_indices(self) -> list[int]:
        return [f.view_idx for f in self.sent_frames]

    @property
    def owned_hi_view_idx(self) -> int:
        """Last view index this unit OWNS (goals starting past it are the next
        window's)."""
        return self.hi - 1

    def sent_actions(self) -> list[str]:
        return [self.actions[f.view_idx] for f in self.sent_frames]

    def image_refs(self) -> list[str]:
        return [str(f.image) for f in self.sent_frames]


def build_units(
    view: SegmentView,
    actions: list[str],
    *,
    context_limit: int,
    completion_reserve: int,
    safety_margin: int,
    max_frames_per_window: int = 0,
    snap_slack: int = 25,
    tail_buffer: int = 5,
) -> list[AnnotationUnit]:
    """Split one segment view into AnnotationUnits under the context budget.
    Single-window segments keep the bare segment_id as unit_id (the common
    case); splits get ``__wN`` suffixes."""
    n = len(view.frames)
    if n == 0:
        return []
    if max_frames_per_window:
        max_fpw = max_frames_per_window
    else:
        per_frame = est_frame_tokens(str(view.frames[0].image))
        budget = max(1, context_limit - completion_reserve - safety_margin)
        max_fpw = max(1, int(budget / (per_frame * 1.05)))
    times = [f.t_s for f in view.frames]
    windows = plan_windows(n, max_fpw, 0, actions=actions, times=times, slack=snap_slack)
    nw = len(windows)
    units: list[AnnotationUnit] = []
    for wi, (lo, hi) in enumerate(windows):
        tail = min(tail_buffer, n - hi) if (nw > 1 and wi < nw - 1) else 0
        units.append(AnnotationUnit(
            unit_id=view.segment_id if nw <= 1 else f"{view.segment_id}__w{wi}",
            view=view,
            window_index=wi,
            n_windows=nw,
            lo=lo,
            hi=hi,
            tail_buffer=tail,
            actions=actions,
        ))
    return units
