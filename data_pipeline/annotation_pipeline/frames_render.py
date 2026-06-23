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


def evenly(pool: list[Any], k: int) -> list[Any]:
    if k <= 0 or not pool:
        return []
    if len(pool) <= k:
        return list(pool)
    step = len(pool) / k
    return [pool[min(len(pool) - 1, int(i * step))] for i in range(k)]


def segment_windows(t_min: float, t_max: float, window_s: float, overlap_s: float) -> list[tuple[float, float]]:
    stride = max(1.0, window_s - overlap_s)
    windows: list[tuple[float, float]] = []
    start = t_min
    while True:
        end = start + window_s
        windows.append((start, end))
        if end >= t_max:
            break
        start += stride
    return windows


def sample_window_frames(
    in_window: list[dict[str, Any]], start_s: float, end_s: float, max_images: int
) -> list[dict[str, Any]]:
    """Sparse-sample a window: one frame per time slot, preferring active frames."""
    if len(in_window) <= max_images:
        return in_window
    step = (end_s - start_s) / max_images
    picked: list[dict[str, Any]] = []
    used: set[int] = set()
    for i in range(max_images):
        lo = start_s + i * step
        hi = lo + step
        slot = [
            j for j, r in enumerate(in_window)
            if lo <= float(r["global_time_s"]) < hi and j not in used
        ]
        if not slot:
            continue
        active = [j for j in slot if in_window[j]["action"] != "NO_OP"]
        j = active[len(active) // 2] if active else slot[len(slot) // 2]
        picked.append(in_window[j])
        used.add(j)
    return picked


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
