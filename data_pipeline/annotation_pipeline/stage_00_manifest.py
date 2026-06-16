#!/usr/bin/env python3
"""Stage 00: build a manifest of MP4/msgpack pairs for a recording slice."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2

from annotation_pipeline import config
from annotation_pipeline.common import ensure_dir, keylog_summary, write_json, write_jsonl


RECORDING_RE = re.compile(r"^recording_(?P<recording_id>[0-9a-fA-F-]+)_seg(?P<seg>\d+).*\.mp4$")


def keylog_for_video(video_path: Path, recording_id: str, segment_idx: int) -> Path:
    name = f"input_{recording_id}_seg{segment_idx:04d}.msgpack"
    return video_path.parent.parent / "keylogs" / name


def probe_video(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        opened = bool(cap.isOpened()) and fps > 0 and frame_count > 0
        duration_s = frame_count / fps if opened else 0.0
    finally:
        cap.release()
    return {
        "video_ok": opened,
        "video_fps": fps,
        "video_frame_count": frame_count,
        "video_duration_s": round(duration_s, 6),
        "video_width": width,
        "video_height": height,
        "video_size_bytes": video_path.stat().st_size if video_path.exists() else 0,
    }


def discover(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_root = args.raw_root.expanduser().resolve()
    rec_dir = raw_root / "uploads" / args.version / args.user_id / "recordings"
    if not rec_dir.is_dir():
        raise FileNotFoundError(f"recordings directory not found: {rec_dir}")

    rows: list[dict[str, Any]] = []
    for video_path in sorted(rec_dir.glob(f"recording_{args.recording_id}_seg*.mp4")):
        match = RECORDING_RE.match(video_path.name)
        if not match:
            continue
        segment_idx = int(match.group("seg"))
        if segment_idx < args.segment_start or segment_idx > args.segment_end:
            continue
        keylog_path = keylog_for_video(video_path, args.recording_id, segment_idx)
        segment_id = f"{args.recording_id}_seg{segment_idx:04d}"
        row = {
            "version": args.version,
            "user_id": args.user_id,
            "recording_id": args.recording_id,
            "segment_idx": segment_idx,
            "segment_id": segment_id,
            "video_path": str(video_path),
            "keylog_path": str(keylog_path),
        }
        row.update(probe_video(video_path))
        row.update(keylog_summary(keylog_path))
        rows.append(row)

    rows.sort(key=lambda row: int(row["segment_idx"]))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=config.RAW_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default=config.PILOT_VERSION)
    parser.add_argument("--user-id", default=config.PILOT_USER_ID)
    parser.add_argument("--recording-id", default=config.PILOT_RECORDING_ID)
    parser.add_argument("--segment-start", type=int, default=config.PILOT_SEGMENT_START)
    parser.add_argument("--segment-end", type=int, default=config.PILOT_SEGMENT_END)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    rows = discover(args)
    if not rows:
        raise RuntimeError("No manifest rows discovered")

    write_jsonl(output_dir / "manifest.jsonl", rows)
    write_json(
        output_dir / "manifest_summary.json",
        {
            "n_segments": len(rows),
            "n_video_ok": sum(1 for row in rows if row["video_ok"]),
            "n_keylogs": sum(1 for row in rows if row["keylog_exists"]),
            "total_video_duration_s": round(sum(row["video_duration_s"] for row in rows), 3),
            "total_keylog_events": sum(row["n_keylog_events"] for row in rows),
            "recording_id": args.recording_id,
            "segment_start": args.segment_start,
            "segment_end": args.segment_end,
        },
    )
    print(f"Wrote {len(rows)} manifest rows to {output_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
