"""Unit chunking + frame rendering for annotation methods.

An ``AnnotationUnit`` is the work quantum stage 03b dispatches to a labeler
model: one segment's view (frames @k fps within filter survivors), split into
window-units only when the segment exceeds the model's context budget. Cuts
are snapped to a command/prompt submission (Return/Enter) or a real time-gap —
never mid typing-burst — so one action (and its goal) is not split across two
windows; non-final windows also see a trailing tail-buffer of context frames
(goals that start in the buffer belong to the next window and are dropped).

Coordinates: methods see the segment's dense view-local frame indices (the
``frame <N>`` labels sent to the model are view indices); nothing view-local
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

from pipeline.lib.action_format import TEXT_KEYS, WindowKeyboard
from pipeline.lib.image_store import read_jpeg_bytes
from pipeline.lib.views import SegmentView, ViewFrame



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


#: A keyboard burst is the text keys plus the ones that edit or commit one without
#: producing a character. The text keys come from ``lib/action_format.TEXT_KEYS``,
#: the same set ``ordered_events_v3`` folds into ``type()``, so the planner and the
#: label emitter cannot disagree about a key: the substring marker list this
#: replaces called ``.``/``;``/``'``/``=``/``[``/``]`` non-typing while the grammar
#: folded them into a burst. Exact names now, because three of those markers
#: (``Digit``, ``Period``, ``Enter``) matched none of the 86 names real keylogs
#: spell. Both directions of delete are in: with ``Backspace`` alone, the same
#: correction read as typing or not by which way the demonstrator deleted, so a
#: window could be cut through a run of forward-deletes retracting the text of the
#: window before it (``ForwardDelete`` is how macOS spells the key).
_BURST_KEYS = TEXT_KEYS | {"Return", "Backspace", "Delete", "ForwardDelete"}


def is_typing(keyboard: WindowKeyboard) -> bool:
    """Whether a frame's window is mid keyboard-burst.

    A lone idle frame is not a reliable boundary — people pause a beat
    mid-typing — so cutting keys on submissions and real time-gaps, and this
    only keeps a cut from landing between two typing frames."""
    return bool(keyboard.texts) or any(name in _BURST_KEYS for name in keyboard.names)


def _is_submission(keyboard: WindowKeyboard) -> bool:
    """A Return/Enter transition commits a typed command / prompt / message —
    the real boundary between activities in continuous work. Press and release
    alike: either half of the pair marks the same commit."""
    return any("Return" in name or "Enter" in name for name in keyboard.names)


def _best_cut(lo: int, hi: int, ideal: int, keyboard: list[WindowKeyboard],
              times: list[float] | None, big_gap_s: float) -> int:
    """Cut index in [lo, hi] (window splits BEFORE this frame) least likely to
    slice an in-progress action, nearest ``ideal`` among equals."""
    best, best_key = ideal, None
    for i in range(lo, hi + 1):
        gap = (times[i] - times[i - 1]) if times else 0.0
        if _is_submission(keyboard[i - 1]) or gap >= big_gap_s:
            cost = 0                              # just submitted, or a real pause
        else:
            # 1 neither side typing, 2 one side, 3 both — mid-burst.
            cost = 1 + int(is_typing(keyboard[i - 1])) + int(is_typing(keyboard[i]))
        key = (cost, abs(i - ideal))
        if best_key is None or key < best_key:
            best_key, best = key, i
    return best


def plan_windows(n: int, max_frames: int, overlap: int = 0, *,
                 keyboard: list[WindowKeyboard],
                 times: list[float] | None = None,
                 slack: int = 0, big_gap_s: float = 6.0) -> list[tuple[int, int]]:
    """Partition [0, n) into (lo, hi) half-open windows, each <= max_frames.
    A split happens only when needed (n > max_frames); otherwise one window.
    With ``slack`` > 0, each interior boundary snaps within ±slack to a
    submission or genuine time-gap, never inside an unsubmitted typing burst."""
    if n <= 0:
        return []
    if n <= max_frames:
        return [(0, n)]
    snap = slack > 0
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
                p = _best_cut(lo, hi, min(max(ideal, lo), hi), keyboard, times, big_gap_s)
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


@dataclass
class AnnotationUnit:
    """One labeler work quantum: an owned span [lo, hi) of a segment view's
    frames plus ``tail_buffer`` trailing context frames. ``keyboard`` is what
    the demonstrator typed in every one of the segment's view frames (used for
    keystroke-burst snapping and window planning; method-internal only)."""

    unit_id: str
    view: SegmentView
    window_index: int
    n_windows: int
    lo: int  # owned view-index range [lo, hi)
    hi: int
    tail_buffer: int  # context frames past hi actually sent
    keyboard: list[WindowKeyboard]

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
        """Last view index this unit owns (goals starting past it are the next
        window's)."""
        return self.hi - 1

    def image_refs(self) -> list[str]:
        return [str(f.image) for f in self.sent_frames]


def build_units(
    view: SegmentView,
    keyboard: list[WindowKeyboard],
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
    windows = plan_windows(n, max_fpw, 0, keyboard=keyboard, times=times, slack=snap_slack)
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
            keyboard=keyboard,
        ))
    return units
