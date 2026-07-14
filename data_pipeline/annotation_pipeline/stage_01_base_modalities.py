#!/usr/bin/env python3
"""Stage 01: materialize timestamped frames and an unbinned keylog event timeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from annotation_pipeline import config
from annotation_pipeline.common import (
    ceil_frames,
    ensure_dir,
    normalize_keylog_events,
    read_jsonl,
    write_json,
    write_jsonl,
)
from annotation_pipeline.image_store import make_arrayrecord_image_uri


def jpeg_quality_to_qscale(jpeg_quality: int) -> int:
    quality = max(1, min(100, int(jpeg_quality)))
    return max(2, min(31, round(31 - (quality / 100) * 30)))


def resolve_ffmpeg_bin(value: str | None) -> str:
    if not value:
        raise RuntimeError(
            "ffmpeg binary not found. Install ffmpeg or set "
            "JUERGEN_ANNOTATION_FFMPEG_BIN/FFMPEG_BIN."
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
    base_fps: float,
    target_height: int,
    jpeg_quality: int,
    ffmpeg_bin: str,
) -> str:
    ensure_dir(output_dir)
    for old_frame in output_dir.glob("frame_*.jpg"):
        old_frame.unlink()
    scale = f"scale=-2:{target_height}" if target_height > 0 else "null"
    video_filter = f"fps=fps={base_fps}:start_time=0:round=near:eof_action=pass,{scale}"
    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        str(int(os.environ.get("JUERGEN_ANNOTATION_FFMPEG_THREADS", "4"))),
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
            f"ffmpeg frame extraction failed for {video_path} "
            f"with exit code {result.returncode}:\n{result.stderr.strip()}"
        )
    return video_filter


def pack_frames(records: list[dict[str, Any]], segment_frame_dir: Path) -> dict[str, Any]:
    from array_record.python.array_record_module import ArrayRecordWriter  # noqa: PLC0415

    shard_path = segment_frame_dir / "images.array_record"
    manifest_path = segment_frame_dir / "frame_manifest.jsonl"
    shard_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    if not records:
        for stray in segment_frame_dir.glob("frame_*.jpg"):
            stray.unlink()
        return {"format": "arrayrecord", "num_records": 0, "total_jpeg_bytes": 0}

    total_jpeg_bytes = 0
    writer = ArrayRecordWriter(str(shard_path), "group_size:1")
    try:
        with manifest_path.open("w") as manifest_f:
            for record_index, record in enumerate(records):
                jpeg = Path(record["image_path"]).read_bytes()
                writer.write(jpeg)
                uri = make_arrayrecord_image_uri(shard_path, record_index)
                manifest_f.write(
                    json.dumps(
                        {
                            "frame_idx": record_index,
                            "image": uri,
                            "shard_path": str(shard_path),
                            "record_index": record_index,
                            "local_frame_idx": record["local_frame_idx"],
                            "jpeg_bytes": len(jpeg),
                            "sha256": hashlib.sha256(jpeg).hexdigest(),
                        }
                    )
                    + "\n"
                )
                record["image_path"] = uri
                total_jpeg_bytes += len(jpeg)
    finally:
        writer.close()
    for stray in segment_frame_dir.glob("frame_*.jpg"):
        stray.unlink()
    return {
        "format": "arrayrecord",
        "shard_path": str(shard_path),
        "manifest_path": str(manifest_path),
        "num_records": len(records),
        "total_jpeg_bytes": total_jpeg_bytes,
    }


def extract_segment(
    row: dict[str, Any],
    frames_dir: Path,
    *,
    base_fps: float,
    target_height: int,
    jpeg_quality: int,
    ffmpeg_bin: str,
    segment_offset_s: float,
    next_global_frame_idx: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    video_path = Path(row["video_path"])
    keylog_path = Path(row["keylog_path"])
    duration_s = float(row["video_duration_s"])
    video_fps = float(row["video_fps"])
    video_frame_count = int(row["video_frame_count"])
    n_expected_frames = ceil_frames(duration_s, base_fps)
    segment_frame_dir = ensure_dir(frames_dir / str(row["segment_id"]))
    ffmpeg_filter = extract_frames_ffmpeg(
        video_path,
        segment_frame_dir,
        base_fps,
        target_height,
        jpeg_quality,
        ffmpeg_bin,
    )

    records: list[dict[str, Any]] = []
    for local_frame_idx in range(n_expected_frames):
        local_time_s = local_frame_idx / base_fps
        image_path = segment_frame_dir / f"frame_{local_frame_idx:06d}.jpg"
        if not image_path.exists():
            continue
        source_frame_idx = min(
            max(0, round(local_time_s * video_fps)),
            max(0, video_frame_count - 1),
        )
        records.append(
            {
                "recording_id": row["recording_id"],
                "segment_id": row["segment_id"],
                "segment_idx": row["segment_idx"],
                "local_frame_idx": local_frame_idx,
                "global_frame_idx": next_global_frame_idx + len(records),
                "local_time_s": round(local_time_s, 6),
                "global_time_s": round(segment_offset_s + local_time_s, 6),
                "source_frame_idx": source_frame_idx,
                "image_path": str(image_path.resolve()),
            }
        )

    events, action_stats = normalize_keylog_events(
        keylog_path,
        recording_id=str(row["recording_id"]),
        segment_id=str(row["segment_id"]),
        segment_idx=int(row["segment_idx"]),
        segment_offset_s=segment_offset_s,
    )
    image_store = pack_frames(records, segment_frame_dir)
    summary = {
        "segment_id": row["segment_id"],
        "segment_idx": row["segment_idx"],
        "duration_s": duration_s,
        "n_expected_frames": n_expected_frames,
        "n_frames": len(records),
        "n_missing_frames": n_expected_frames - len(records),
        "n_events": len(events),
        "extraction_backend": "ffmpeg",
        "ffmpeg_filter": ffmpeg_filter,
        "action_stats": asdict(action_stats),
        "image_store": image_store,
    }
    return records, events, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-fps", type=float, default=config.DEFAULT_BASE_FPS)
    parser.add_argument("--target-height", type=int, default=config.DEFAULT_TARGET_HEIGHT)
    parser.add_argument("--jpeg-quality", type=int, default=config.DEFAULT_JPEG_QUALITY)
    parser.add_argument("--ffmpeg-bin", default=config.ffmpeg_bin())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.base_fps <= 0:
        raise ValueError("base_fps must be positive")
    ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)
    output_dir = ensure_dir(args.output_dir)
    frames_dir = ensure_dir(output_dir / "frames")
    manifest = read_jsonl(args.manifest)
    if not manifest:
        raise RuntimeError(f"Empty manifest: {args.manifest}")

    all_frames: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    segment_offset_s = 0.0
    for row in manifest:
        duration_s = float(row["video_duration_s"])
        if not row.get("video_ok"):
            summaries.append(
                {
                    "segment_id": row["segment_id"],
                    "segment_idx": row["segment_idx"],
                    "skip_reason": "video_not_decodable",
                }
            )
            segment_offset_s += duration_s
            continue
        frames, events, summary = extract_segment(
            row,
            frames_dir,
            base_fps=args.base_fps,
            target_height=args.target_height,
            jpeg_quality=args.jpeg_quality,
            ffmpeg_bin=ffmpeg_bin,
            segment_offset_s=segment_offset_s,
            next_global_frame_idx=len(all_frames),
        )
        all_frames.extend(frames)
        all_events.extend(events)
        summaries.append(summary)
        segment_offset_s += duration_s

    write_jsonl(output_dir / "frames.jsonl", all_frames)
    write_jsonl(output_dir / "events.jsonl", all_events)
    write_json(output_dir / "segment_summaries.json", summaries)
    write_json(
        output_dir / "manifest.json",
        {
            "stage": "base_modalities",
            "schema_version": 1,
            "n_segments": len(manifest),
            "n_frames": len(all_frames),
            "n_events": len(all_events),
            "base_fps": args.base_fps,
            "target_height": args.target_height,
            "jpeg_quality": args.jpeg_quality,
            "total_duration_s": round(segment_offset_s, 3),
            "files": {
                "frames": "frames.jsonl",
                "events": "events.jsonl",
                "segment_summaries": "segment_summaries.json",
            },
        },
    )
    print(f"Wrote {len(all_frames)} frames and {len(all_events)} events to {output_dir}")


if __name__ == "__main__":
    main()
