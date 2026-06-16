#!/usr/bin/env python3
"""Stage 01: extract 2fps frames and align next-window action bins."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from annotation_pipeline import config
from annotation_pipeline.common import (
    ActionBin,
    aggregate_actions,
    ceil_frames,
    ensure_dir,
    format_action,
    merge_action_bins,
    read_jsonl,
    write_json,
    write_jsonl,
)


def jpeg_quality_to_qscale(jpeg_quality: int) -> int:
    """Approximate OpenCV's 0-100 JPEG quality on FFmpeg's 2-31 MJPEG qscale."""
    quality = max(1, min(100, int(jpeg_quality)))
    return max(2, min(31, round(31 - (quality / 100) * 30)))


def resolve_ffmpeg_bin(value: str | None) -> str:
    if not value:
        raise RuntimeError(
            "ffmpeg binary not found. Install ffmpeg or set V3_FFMPEG_BIN/FFMPEG_BIN."
        )
    expanded = str(Path(value).expanduser())
    if Path(expanded).exists():
        return expanded
    path_bin = shutil.which(value)
    if path_bin:
        return path_bin
    raise RuntimeError(f"ffmpeg binary not found: {value}")


def extract_frames_ffmpeg(
    video_path: Path,
    output_dir: Path,
    target_fps: int,
    target_height: int,
    jpeg_quality: int,
    ffmpeg_bin: str,
) -> str:
    ensure_dir(output_dir)
    for old_frame in output_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    scale = f"scale=-2:{target_height}" if target_height > 0 else "null"
    video_filter = f"fps=fps={target_fps}:start_time=0:round=near:eof_action=pass,{scale}"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        video_filter,
        "-q:v",
        str(jpeg_quality_to_qscale(jpeg_quality)),
        "-start_number",
        "0",
        str(output_dir / "frame_%06d.jpg"),
    ]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg frame extraction failed for "
            f"{video_path} with exit code {result.returncode}:\n{result.stderr.strip()}"
        )
    return video_filter


def extract_segment(
    row: dict[str, Any],
    frames_dir: Path,
    target_fps: int,
    target_height: int,
    jpeg_quality: int,
    black_frame_threshold: float,
    max_noop_run: int,
    ffmpeg_bin: str,
    segment_offset_s: float,
    next_global_frame_idx: int,
    initial_noop_run: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    video_path = Path(row["video_path"])
    keylog_path = Path(row["keylog_path"])
    duration_s = float(row["video_duration_s"])
    video_fps = float(row["video_fps"])
    video_frame_count = int(row["video_frame_count"])
    n_bins = ceil_frames(duration_s, target_fps)

    bins, action_stats = aggregate_actions(keylog_path, n_bins, target_fps)
    segment_frame_dir = ensure_dir(frames_dir / row["segment_id"])
    ffmpeg_filter = extract_frames_ffmpeg(
        video_path=video_path,
        output_dir=segment_frame_dir,
        target_fps=target_fps,
        target_height=target_height,
        jpeg_quality=jpeg_quality,
        ffmpeg_bin=ffmpeg_bin,
    )

    records: list[dict[str, Any]] = []
    skipped_black = 0
    failed_reads = 0
    n_extra_frames_deleted = 0
    n_noop_capped = 0
    n_bins_carried = 0
    noop_run = initial_noop_run
    # When a frame is dropped (black/unreadable) its action bin is carried into
    # the next kept frame, so press/release pairs and drags stay coherent.
    carry: ActionBin | None = None

    for local_bin_idx in range(n_bins):
        action_bin = bins[local_bin_idx]
        if carry is not None:
            action_bin = merge_action_bins(carry, action_bin)
            carry = None
            n_bins_carried += 1
        local_time_s = local_bin_idx / target_fps
        source_frame_idx = min(
            max(0, int(round(local_time_s * video_fps))),
            max(0, video_frame_count - 1),
        )
        image_path = segment_frame_dir / f"frame_{local_bin_idx:06d}.jpg"
        frame = cv2.imread(str(image_path))
        if frame is None:
            failed_reads += 1
            carry = action_bin
            continue
        mean_intensity = float(frame.mean())
        if black_frame_threshold > 0 and mean_intensity < black_frame_threshold:
            skipped_black += 1
            image_path.unlink(missing_ok=True)
            carry = action_bin
            continue

        action = format_action(action_bin)
        if action == "NO_OP":
            if max_noop_run >= 0 and noop_run >= max_noop_run:
                n_noop_capped += 1
                image_path.unlink(missing_ok=True)
                continue
            noop_run += 1
        else:
            noop_run = 0

        global_time_s = segment_offset_s + local_time_s
        records.append(
            {
                "recording_id": row["recording_id"],
                "segment_id": row["segment_id"],
                "segment_idx": row["segment_idx"],
                "local_bin_idx": local_bin_idx,
                "global_frame_idx": next_global_frame_idx + len(records),
                "local_time_s": round(local_time_s, 6),
                "global_time_s": round(global_time_s, 6),
                "source_frame_idx": source_frame_idx,
                "image_path": str(image_path.resolve()),
                "action": action,
                "mean_intensity": round(mean_intensity, 3),
            }
        )

    for extra_frame in segment_frame_dir.glob("frame_*.jpg"):
        try:
            frame_idx = int(extra_frame.stem.removeprefix("frame_"))
        except ValueError:
            continue
        if frame_idx >= n_bins:
            extra_frame.unlink()
            n_extra_frames_deleted += 1

    summary = {
        "segment_id": row["segment_id"],
        "segment_idx": row["segment_idx"],
        "duration_s": duration_s,
        "n_bins": n_bins,
        "n_frames_kept": len(records),
        "n_black_skipped": skipped_black,
        "n_failed_reads": failed_reads,
        "n_extra_frames_deleted": n_extra_frames_deleted,
        "n_noop_capped": n_noop_capped,
        "max_noop_run": max_noop_run,
        "n_bins_carried": n_bins_carried,
        "n_tail_events_dropped": len(carry.events) if carry is not None else 0,
        "ending_noop_run": noop_run,
        "extraction_backend": "ffmpeg",
        "ffmpeg_filter": ffmpeg_filter,
        "action_stats": asdict(action_stats),
        "n_non_noop": sum(1 for record in records if record["action"] != "NO_OP"),
    }
    return records, summary, noop_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-fps", type=int, default=config.DEFAULT_TARGET_FPS)
    parser.add_argument("--target-height", type=int, default=config.DEFAULT_TARGET_HEIGHT)
    parser.add_argument("--jpeg-quality", type=int, default=config.DEFAULT_JPEG_QUALITY)
    parser.add_argument("--ffmpeg-bin", default=config.ffmpeg_bin())
    parser.add_argument(
        "--max-noop-run",
        type=int,
        default=config.DEFAULT_STAGE01_MAX_NOOP_RUN,
        help="Keep at most this many consecutive NO_OP frames in frame_records; -1 disables.",
    )
    parser.add_argument(
        "--black-frame-threshold",
        type=float,
        default=config.DEFAULT_BLACK_FRAME_THRESHOLD,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)

    output_dir = ensure_dir(args.output_dir)
    frames_dir = ensure_dir(output_dir / "frames")
    manifest = read_jsonl(args.manifest)
    if not manifest:
        raise RuntimeError(f"Empty manifest: {args.manifest}")

    all_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    segment_offset_s = 0.0
    noop_run = 0
    for row in manifest:
        if not row.get("video_ok"):
            summaries.append(
                {
                    "segment_id": row["segment_id"],
                    "segment_idx": row["segment_idx"],
                    "skip_reason": "video_not_decodable",
                }
            )
            continue
        records, summary, noop_run = extract_segment(
            row=row,
            frames_dir=frames_dir,
            target_fps=args.target_fps,
            target_height=args.target_height,
            jpeg_quality=args.jpeg_quality,
            black_frame_threshold=args.black_frame_threshold,
            max_noop_run=args.max_noop_run,
            ffmpeg_bin=ffmpeg_bin,
            segment_offset_s=segment_offset_s,
            next_global_frame_idx=len(all_records),
            initial_noop_run=noop_run,
        )
        all_records.extend(records)
        summaries.append(summary)
        segment_offset_s += float(row["video_duration_s"])

    write_jsonl(output_dir / "frame_records.jsonl", all_records)
    write_json(output_dir / "segment_summaries.json", summaries)
    write_json(
        output_dir / "frames_actions_summary.json",
        {
            "n_segments": len(manifest),
            "n_frames": len(all_records),
            "n_non_noop": sum(1 for record in all_records if record["action"] != "NO_OP"),
            "target_fps": args.target_fps,
            "target_height": args.target_height,
            "jpeg_quality": args.jpeg_quality,
            "black_frame_threshold": args.black_frame_threshold,
            "max_noop_run": args.max_noop_run,
            "n_noop_capped": sum(row.get("n_noop_capped", 0) for row in summaries),
            "extraction_backend": "ffmpeg",
            "ffmpeg_bin": ffmpeg_bin,
            "total_duration_s": round(segment_offset_s, 3),
        },
    )
    print(f"Wrote {len(all_records)} frame/action records to {output_dir}")


if __name__ == "__main__":
    main()
