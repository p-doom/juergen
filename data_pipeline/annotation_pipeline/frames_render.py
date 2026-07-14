#!/usr/bin/env python3
"""Stored-frame loading and window selection for the Stage-03 annotator."""

from __future__ import annotations

from typing import Any

import cv2


def resize_to_height(frame: Any, height: int) -> Any:
    if height <= 0 or frame.shape[0] == height:
        return frame
    scale = height / frame.shape[0]
    width = max(2, round((frame.shape[1] * scale) / 2) * 2)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(frame, (width, height), interpolation=interp)


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------


def frames_to_data_urls(
    records: list[dict[str, Any]],
    target_height: int = 0,
    jpeg_quality: int = 80,
) -> list[str]:
    """In-memory ``data:image/jpeg`` URLs for the labeler, read straight from
    each record's stored JPEG (the stage-01 ``ar://`` array_record URI, or a
    stored image reference) — no frames written to disk.

    Bytes are passed through verbatim unless ``target_height`` is set below the
    stored frame height, in which case every frame is decoded, downscaled and
    re-encoded at ``jpeg_quality`` (the overflow-clip rescue knob). All frames in
    a segment share one render, so the stored height is probed once.
    """
    import base64  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from annotation_pipeline.image_store import read_jpeg_bytes  # noqa: PLC0415

    if not records:
        return []

    refs = [r["image_path"] for r in records]
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
            img = resize_to_height(img, target_height)
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if not ok:
                raise RuntimeError(f"could not re-encode frame: {ref}")
            raw = enc.tobytes()
        urls.append("data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"))
    return urls


def est_frame_tokens(ref: str) -> int:
    """Vision tokens for one stored frame: ceil(h/28)*ceil(w/28) (verified within
    ~2% of real prompt_tokens). Used to budget how many frames fit one context."""
    import numpy as np  # noqa: PLC0415

    from annotation_pipeline.image_store import read_jpeg_bytes  # noqa: PLC0415

    img = cv2.imdecode(np.frombuffer(read_jpeg_bytes(ref), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not decode frame for token estimate: {ref}")
    import math  # noqa: PLC0415

    h, w = img.shape[:2]
    return math.ceil(h / 28) * math.ceil(w / 28)


def _best_cut(
    lo: int,
    hi: int,
    ideal: int,
    observations: list[dict[str, Any]],
    times: list[float] | None,
    big_gap_s: float,
) -> int:
    """Pick the cut index in [lo, hi] (window splits BEFORE this frame) that is
    least likely to slice through an in-progress action, preferring one nearest
    `ideal`. Cost: a submission just happened, or a genuine time-gap precedes the
    frame = 0 (ideal boundary); neither side typing = 1; one side typing = 2; both
    sides typing (mid-burst) = 3."""
    best, best_key = ideal, None
    for i in range(lo, hi + 1):
        a_prev = str(observations[i - 1]["activity"])
        a_cur = str(observations[i]["activity"])
        gap = (times[i] - times[i - 1]) if times else 0.0
        if observations[i - 1]["has_submission"] or gap >= big_gap_s:
            cost = 0  # just submitted, or a real pause
        elif a_prev != "type" and a_cur != "type":
            cost = 1
        elif a_prev != "type" or a_cur != "type":
            cost = 2
        else:
            cost = 3  # both sides typing — mid-burst
        key = (cost, abs(i - ideal))
        if best_key is None or key < best_key:
            best_key, best = key, i
    return best


def plan_windows(
    n: int,
    max_frames: int,
    overlap: int = 0,
    *,
    observations: list[dict[str, Any]] | None = None,
    times: list[float] | None = None,
    slack: int = 0,
    big_gap_s: float = 6.0,
) -> list[tuple[int, int]]:
    """Partition [0, n) into windows, each <= max_frames, with `overlap` shared
    frames at each interior boundary (0 = hard cut). Returns (lo, hi) half-open
    spans. A split happens ONLY when needed (n > max_frames); otherwise one window.

    When Stage-02 `observations` (len == n) and `slack` > 0 are given,
    each interior boundary is snapped within ±slack to the nearest SUBMISSION
    (Return/Enter) or genuine time-gap — never inside an unsubmitted typing burst —
    so one action (and its goal) is not split across two windows. Each candidate
    is clamped so every window still fits max_frames."""
    import math  # noqa: PLC0415

    if n <= 0:
        return []
    if n <= max_frames:  # split only if needed
        return [(0, n)]
    snap = bool(observations) and slack > 0
    n_win = math.ceil(n / max_frames)  # fewest windows that fit budget
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
            if lo > hi:  # snap range infeasible — forced cut
                p = min(max(ideal, prev + 1), prev + max_frames, n - 1)
            else:
                assert observations is not None
                p = _best_cut(
                    lo,
                    hi,
                    min(max(ideal, lo), hi),
                    observations,
                    times,
                    big_gap_s,
                )
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


def evenly(pool: list[Any], k: int) -> list[Any]:
    if k <= 0 or not pool:
        return []
    if len(pool) <= k:
        return list(pool)
    step = len(pool) / k
    return [pool[min(len(pool) - 1, int(i * step))] for i in range(k)]


def select_naming_frames(frames: list[dict[str, Any]], max_images: int) -> list[dict[str, Any]]:
    """Always first/last, prefer active frames in between."""
    n = len(frames)
    if n <= max_images:
        return frames
    picks: set[int] = {0, n - 1}
    budget = max_images - len(picks)
    non_noop = [i for i in range(1, n - 1) if not frames[i]["is_noop"]]
    picks |= set(evenly(non_noop, budget))
    remaining = max_images - len(picks)
    if remaining > 0:
        rest = [i for i in range(n) if i not in picks]
        picks |= set(evenly(rest, remaining))
    return [frames[i] for i in sorted(picks)]
