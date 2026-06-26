#!/usr/bin/env python3
"""Stage 01: extract 2fps frames and align next-window action bins."""

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
from annotation_pipeline.image_store import make_arrayrecord_image_uri


def jpeg_quality_to_qscale(jpeg_quality: int) -> int:
    """Approximate OpenCV's 0-100 JPEG quality on FFmpeg's 2-31 MJPEG qscale."""
    quality = max(1, min(100, int(jpeg_quality)))
    return max(2, min(31, round(31 - (quality / 100) * 30)))


def resolve_ffmpeg_bin(value: str | None) -> str:
    if not value:
        raise RuntimeError(
            "ffmpeg binary not found. Install ffmpeg or set JUERGEN_ANNOTATION_FFMPEG_BIN/FFMPEG_BIN."
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
    target_fps: float,
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
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        # Cap threads: on shared login nodes ffmpeg's default (all cores) gets
        # SIGKILL'd by the CPU-usage policy killer. A small cap is plenty for
        # sparse (1-2 fps) frame extraction and stays under the radar.
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
            "ffmpeg frame extraction failed for "
            f"{video_path} with exit code {result.returncode}:\n{result.stderr.strip()}"
        )
    return video_filter


def pack_segment_to_arrayrecord(
    records: list[dict[str, Any]],
    segment_frame_dir: Path,
) -> dict[str, Any]:
    """Pack kept-frame JPEGs into one ``images.array_record`` (grain store).

    Records are packed in ``frame_records`` order; record ``i`` in the shard is
    ``records[i]``. Each record's ``image_path`` is rewritten in place to an
    ``ar://`` URI. Loose ``frame_*.jpg`` are deleted afterwards, so the segment
    dir keeps only the shard and its ``frame_manifest.jsonl`` sidecar.
    """
    from array_record.python.array_record_module import ArrayRecordWriter  # noqa: PLC0415

    shard_path = segment_frame_dir / "images.array_record"
    manifest_path = segment_frame_dir / "frame_manifest.jsonl"

    if not records:
        # Nothing kept (all black/NO_OP-capped): drop stray frames, no shard.
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
                            "local_bin_idx": record["local_bin_idx"],
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
    target_fps: float,
    target_height: int,
    jpeg_quality: int,
    ffmpeg_bin: str,
    segment_offset_s: float,
    next_global_frame_idx: int,
    noop_keep_head: int,
    noop_keep_tail: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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

    n_bins_carried = 0
    # Pass 1: every bin that has an extracted frame, with its final action
    # (carrying dropped-frame action bins forward so press/release stay coherent).
    candidates: list[dict[str, Any]] = []
    missing_frames = 0
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
        if not image_path.exists():
            missing_frames += 1
            carry = action_bin
            continue
        candidates.append({
            "local_bin_idx": local_bin_idx,
            "local_time_s": local_time_s,
            "source_frame_idx": source_frame_idx,
            "image_path": image_path,
            "action": format_action(action_bin),
        })

    # Pass 2: keep every active frame; within each maximal run of consecutive
    # NO_OP frames keep the first noop_keep_head and the last noop_keep_tail and
    # drop the middle, so the start AND end of a wait (e.g. an agent finishing)
    # stay visible.
    keep = [True] * len(candidates)
    n_noop_dropped = 0
    i = 0
    while i < len(candidates):
        if candidates[i]["action"] != "NO_OP":
            i += 1
            continue
        j = i
        while j < len(candidates) and candidates[j]["action"] == "NO_OP":
            j += 1
        if (j - i) > noop_keep_head + noop_keep_tail:
            for k in range(i + noop_keep_head, j - noop_keep_tail):
                keep[k] = False
                n_noop_dropped += 1
        i = j

    records: list[dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        if not keep[idx]:
            cand["image_path"].unlink(missing_ok=True)
            continue
        global_time_s = segment_offset_s + cand["local_time_s"]
        records.append(
            {
                "recording_id": row["recording_id"],
                "segment_id": row["segment_id"],
                "segment_idx": row["segment_idx"],
                "local_bin_idx": cand["local_bin_idx"],
                "global_frame_idx": next_global_frame_idx + len(records),
                "local_time_s": round(cand["local_time_s"], 6),
                "global_time_s": round(global_time_s, 6),
                "source_frame_idx": cand["source_frame_idx"],
                "image_path": str(cand["image_path"].resolve()),
                "action": cand["action"],
            }
        )

    # Pack the kept JPEGs into one images.array_record (grain), rewrite each
    # record's image_path to its ar:// URI, and delete the loose frames.
    image_store = pack_segment_to_arrayrecord(records, segment_frame_dir)

    summary = {
        "segment_id": row["segment_id"],
        "segment_idx": row["segment_idx"],
        "duration_s": duration_s,
        "n_bins": n_bins,
        "n_frames_kept": len(records),
        "n_missing_frames": missing_frames,
        "n_noop_dropped": n_noop_dropped,
        "noop_keep_head": noop_keep_head,
        "noop_keep_tail": noop_keep_tail,
        "n_bins_carried": n_bins_carried,
        "n_tail_events_dropped": len(carry.events) if carry is not None else 0,
        "extraction_backend": "ffmpeg",
        "ffmpeg_filter": ffmpeg_filter,
        "action_stats": asdict(action_stats),
        "n_non_noop": sum(1 for record in records if record["action"] != "NO_OP"),
        "image_store": image_store,
    }
    return records, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=config.DEFAULT_TARGET_FPS)
    parser.add_argument("--target-height", type=int, default=config.DEFAULT_TARGET_HEIGHT)
    parser.add_argument("--jpeg-quality", type=int, default=config.DEFAULT_JPEG_QUALITY)
    parser.add_argument("--ffmpeg-bin", default=config.ffmpeg_bin())
    parser.add_argument(
        "--noop-keep-head",
        type=int,
        default=config.DEFAULT_NOOP_KEEP_HEAD,
        help="Within each NO_OP run, keep the first this many frames.",
    )
    parser.add_argument(
        "--noop-keep-tail",
        type=int,
        default=config.DEFAULT_NOOP_KEEP_TAIL,
        help="Within each NO_OP run, keep the last this many frames (so a wait's end / agent finish is seen).",
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
        records, summary = extract_segment(
            row=row,
            frames_dir=frames_dir,
            target_fps=args.target_fps,
            target_height=args.target_height,
            jpeg_quality=args.jpeg_quality,
            ffmpeg_bin=ffmpeg_bin,
            segment_offset_s=segment_offset_s,
            next_global_frame_idx=len(all_records),
            noop_keep_head=args.noop_keep_head,
            noop_keep_tail=args.noop_keep_tail,
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
            "noop_keep_head": args.noop_keep_head,
            "noop_keep_tail": args.noop_keep_tail,
            "n_noop_dropped": sum(row.get("n_noop_dropped", 0) for row in summaries),
            "extraction_backend": "ffmpeg",
            "ffmpeg_bin": ffmpeg_bin,
            "total_duration_s": round(segment_offset_s, 3),
        },
    )
    print(f"Wrote {len(all_records)} frame/action records to {output_dir}")


if __name__ == "__main__":
    main()
