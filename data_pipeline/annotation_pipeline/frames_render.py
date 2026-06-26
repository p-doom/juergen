#!/usr/bin/env python3
"""Frame sampling + rendering for the VLM annotator (stage 02).

Frames are rendered straight from the raw MP4 at the kept records' source frame
indices, clean (no burned-in overlay — timestamps are passed as interleaved text
in the request). Window/interval frame selection is activity-biased.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from annotation_pipeline.common import ensure_dir, read_jsonl


def resize_to_height(frame: Any, height: int) -> Any:
    if height <= 0 or frame.shape[0] == height:
        return frame
    scale = height / frame.shape[0]
    width = max(2, round((frame.shape[1] * scale) / 2) * 2)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(frame, (width, height), interpolation=interp)


def load_vlm_video_sources(manifest_path: Path) -> dict[str, Path]:
    """{segment_id -> raw MP4 path} from a stage-00 manifest."""
    sources: dict[str, Path] = {}
    for row in read_jsonl(manifest_path):
        seg = str(row.get("segment_id", ""))
        vid = row.get("video_path")
        if seg and vid:
            sources[seg] = Path(vid)
    if not sources:
        raise RuntimeError(f"No video sources in manifest: {manifest_path}")
    return sources


def read_record_frame(
    record: dict[str, Any],
    video_by_segment: dict[str, Path],
    captures: dict[str, cv2.VideoCapture],
) -> Any:
    segment_id = str(record.get("segment_id", ""))
    video_path = video_by_segment.get(segment_id)
    if video_path is None:
        raise RuntimeError(f"No raw video source for segment_id={segment_id!r}")
    cap = captures.get(segment_id)
    if cap is None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open raw video: {video_path}")
        captures[segment_id] = cap
    idx = int(record.get("source_frame_idx", -1))
    if idx < 0:
        raise RuntimeError(f"Invalid source_frame_idx on frame record: {record}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"could not read frame {idx} from {video_path}")
    return frame


def render_frames(
    records: list[dict[str, Any]],
    output_dir: Path,
    jpeg_quality: int,
    video_by_segment: dict[str, Path],
    target_height: int,
) -> list[Path]:
    ensure_dir(output_dir)
    for old_frame in output_dir.glob("frame_*.jpg"):
        old_frame.unlink()
    image_paths: list[Path] = []
    captures: dict[str, cv2.VideoCapture] = {}
    try:
        for out_idx, record in enumerate(records):
            frame = resize_to_height(
                read_record_frame(record, video_by_segment, captures), target_height
            )
            image_path = output_dir / f"frame_{out_idx:06d}.jpg"
            cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            image_paths.append(image_path)
    finally:
        for cap in captures.values():
            cap.release()
    return image_paths


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
    plain file path for legacy runs) — no frames written to disk.

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


def frame_activity(action: str | None) -> str:
    """Classify a frame by its stage-01 action label: "idle" (NO_OP), "type" (the
    user is mid keyboard-burst), or "other" (mouse / click / scroll / no events).
    NOTE: a lone "idle"/NO_OP frame is NOT a reliable activity boundary — people
    pause for a frame mid-typing — so segmentation keys on SUBMISSIONS and real
    time-gaps (see _is_submission / plan_windows), using this only to avoid
    cutting between two "type" frames."""
    if not action or action == "NO_OP":
        return "idle"
    # Action strings look like "<dx> <dy> <scroll> ; <key events>" — keyboard
    # events (+KeyA, -Return, +Backspace, +Space, +Digit3, +NumpadEnter, …) only
    # appear after the ";". Their presence means the user is typing this frame.
    tail = action.split(";", 1)[1] if ";" in action else action
    if any(tok in tail for tok in ("Key", "Return", "Backspace", "Space", "Enter", "Digit", "Num", "Minus", "Slash", "Period", "Comma")):
        return "type"
    return "other"


def _is_submission(action: str | None) -> bool:
    """True if this frame commits a typed command / prompt / message — a Return or
    Enter keypress. These are the real boundaries BETWEEN activities in continuous
    work (where idle gaps don't exist): the action just finished, so the next
    frame is a clean place to cut."""
    return bool(action) and ("Return" in action or "Enter" in action)


def _best_cut(lo: int, hi: int, ideal: int, actions: list[str | None],
              times: list[float] | None, big_gap_s: float) -> int:
    """Pick the cut index in [lo, hi] (window splits BEFORE this frame) that is
    least likely to slice through an in-progress action, preferring one nearest
    `ideal`. Cost: a submission just happened, or a genuine time-gap precedes the
    frame = 0 (ideal boundary); neither side typing = 1; one side typing = 2; both
    sides typing (mid-burst) = 3."""
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
    """Partition [0, n) into windows, each <= max_frames, with `overlap` shared
    frames at each interior boundary (0 = hard cut). Returns (lo, hi) half-open
    spans. A split happens ONLY when needed (n > max_frames); otherwise one window.

    When `actions` (per-frame stage-01 labels, len == n) and `slack` > 0 are given,
    each interior boundary is snapped within ±slack to the nearest SUBMISSION
    (Return/Enter) or genuine time-gap — never inside an unsubmitted typing burst —
    so one action (and its goal) is not split across two windows. Each candidate
    is clamped so every window still fits max_frames."""
    import math  # noqa: PLC0415

    if n <= 0:
        return []
    if n <= max_frames:                           # split only if needed
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


def evenly(pool: list[Any], k: int) -> list[Any]:
    if k <= 0 or not pool:
        return []
    if len(pool) <= k:
        return list(pool)
    step = len(pool) / k
    return [pool[min(len(pool) - 1, int(i * step))] for i in range(k)]


def records_in_index_span(
    records: list[dict[str, Any]], start_idx: int, end_idx: int
) -> list[dict[str, Any]]:
    """Kept records whose ``global_frame_idx`` falls in the inclusive span.

    Pass 1 returns activity boundaries as frame indices (stable across sampling/
    NO_OP-filtering, unlike wall-clock time). This maps such a span back onto the
    full kept-frame stream so Pass 2 sees every frame in it, not just the sparse
    subset Pass 1 was shown. Returned in ``global_frame_idx`` order.
    """
    lo, hi = (start_idx, end_idx) if start_idx <= end_idx else (end_idx, start_idx)
    out = [r for r in records if lo <= int(r["global_frame_idx"]) <= hi]
    out.sort(key=lambda r: int(r["global_frame_idx"]))
    return out


def select_naming_frames(frames: list[dict[str, Any]], max_images: int) -> list[dict[str, Any]]:
    """Always first/last, prefer active frames in between."""
    n = len(frames)
    if n <= max_images:
        return frames
    picks: set[int] = {0, n - 1}
    budget = max_images - len(picks)
    non_noop = [i for i in range(1, n - 1) if frames[i]["action"] != "NO_OP"]
    picks |= set(evenly(non_noop, budget))
    remaining = max_images - len(picks)
    if remaining > 0:
        rest = [i for i in range(n) if i not in picks]
        picks |= set(evenly(rest, remaining))
    return [frames[i] for i in sorted(picks)]
