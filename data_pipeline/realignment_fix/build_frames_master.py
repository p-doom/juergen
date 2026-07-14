#!/usr/bin/env python3
"""Stage 01a (frames-master): decode every segment's mp4 ONCE into a JPEG
ArrayRecord frame store, so downstream fps experiments never re-decode.

Today ``stage_01_frames_actions`` couples the (expensive) ffmpeg decode to the
(cheap) action binning + NO_OP thinning: changing ``--target-fps`` re-decodes
the whole corpus. This stage splits off the decode. It reads the discover
``clips_manifest.jsonl`` and, per segment, extracts frames at a fixed
MASTER fps into one ``images.array_record`` (grain store), referenced as
``ar:///abs/path/images.array_record#idx`` -- the same URI scheme
``image_store`` / stage 01 / stage 02 already consume.

A downstream sampler (01b) then picks the nearest master record per target-fps
bin and bins the keylog actions, emitting a metadata-only dataset (no new JPEG
bytes). Because of that, the MASTER fps is the sampling CEILING: you can sample
DOWN to any fps <= master, never up. Pick it as the highest fps you will ever
want (storage scales linearly with it). No action binning / NO_OP thinning
happens here -- those are fps-dependent and belong to 01b.

Frames are stored CFR at ``master_fps`` (ffmpeg's ``fps=`` filter resamples any
VFR source), so master record ``i`` is at ``source_time_s = i / master_fps``.
``source_frame_idx`` (nearest frame in the ORIGINAL video) is recorded for
provenance only; sampling keys on ``source_time_s``.

Source mp4s are read-only; everything is written under --output-dir.

Outputs (under --output-dir):
  frames/<segment_id>/images.array_record   one grain shard per segment.
  frames/<segment_id>/frame_manifest.jsonl  per-record ar:// URI + source_time_s
                                            + source_frame_idx + sha256.
  segment_index.jsonl                       one row per segment: shard path,
                                            num_records, master_fps, video-relative
                                            timing + video provenance. Alignment-
                                            agnostic and keylog-free: the decode
                                            ignores the keylog, so 01b joins the
                                            realigned manifest by segment_id to
                                            bin actions.
  frames_master_summary.json                aggregate stats.
  manifest.json                             artifact marker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm  # progress bar; arrives transitively via transformers/datasets

# Make the sibling ``annotation_pipeline`` package importable when this script
# is run directly from its own folder (mirrors run_dataset's PYTHONPATH setup).
DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from annotation_pipeline import config  # noqa: E402
from annotation_pipeline.common import ensure_dir, read_jsonl, write_json  # noqa: E402
from annotation_pipeline.image_store import make_arrayrecord_image_uri  # noqa: E402
from annotation_pipeline.stage_01_frames_actions import (  # noqa: E402
    extract_frames_ffmpeg,
    resolve_ffmpeg_bin,
)

DEFAULT_MASTER_FPS = 4.0


def pack_master_arrayrecord(
    frame_paths: list[Path],
    segment_frame_dir: Path,
    master_fps: float,
    video_fps: float,
    video_frame_count: int,
) -> dict[str, Any]:
    """Pack the decoded JPEGs into one ``images.array_record`` (grain store).

    Records are written in ``frame_paths`` order; record ``i`` in the shard is
    ``frame_paths[i]`` and sits at ``source_time_s = i / master_fps`` in the
    resampled (CFR) stream. Writes ``frame_manifest.jsonl`` and deletes the
    loose ``frame_*.jpg`` afterwards, so the segment dir keeps only the shard +
    its sidecar.
    """
    from array_record.python.array_record_module import ArrayRecordWriter  # noqa: PLC0415

    shard_path = segment_frame_dir / "images.array_record"
    manifest_path = segment_frame_dir / "frame_manifest.jsonl"
    # Drop any stale artifacts from a partial prior run before rewriting.
    shard_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)

    if not frame_paths:
        return {"num_records": 0, "total_jpeg_bytes": 0}

    max_src_idx = max(0, int(video_frame_count) - 1) if video_frame_count else 0
    total_jpeg_bytes = 0
    writer = ArrayRecordWriter(str(shard_path), "group_size:1")
    try:
        with manifest_path.open("w") as manifest_f:
            for record_index, frame_path in enumerate(frame_paths):
                jpeg = frame_path.read_bytes()
                writer.write(jpeg)
                source_time_s = record_index / master_fps
                source_frame_idx = (
                    min(max(0, round(source_time_s * video_fps)), max_src_idx)
                    if video_fps > 0
                    else 0
                )
                manifest_f.write(
                    json.dumps(
                        {
                            "record_index": record_index,
                            "image": make_arrayrecord_image_uri(shard_path, record_index),
                            "shard_path": str(shard_path),
                            "source_time_s": round(source_time_s, 6),
                            "source_frame_idx": source_frame_idx,
                            "jpeg_bytes": len(jpeg),
                            "sha256": hashlib.sha256(jpeg).hexdigest(),
                        }
                    )
                    + "\n"
                )
                total_jpeg_bytes += len(jpeg)
    finally:
        writer.close()

    for stray in segment_frame_dir.glob("frame_*.jpg"):
        stray.unlink()

    return {
        "shard_path": str(shard_path),
        "manifest_path": str(manifest_path),
        "num_records": len(frame_paths),
        "total_jpeg_bytes": total_jpeg_bytes,
    }


def build_segment_master(task: dict[str, Any]) -> dict[str, Any]:
    """Worker: decode one segment's mp4 into its master frame store and return
    an index row. Picklable; no shared state. Failures are captured, not raised,
    so one bad segment never aborts the pool."""
    row = task["row"]
    seg = str(row["segment_id"])
    segment_frame_dir = ensure_dir(Path(task["frames_dir"]) / seg)
    manifest_path = segment_frame_dir / "frame_manifest.jsonl"

    base_row = {
        "segment_id": seg,
        "recording_id": row.get("recording_id"),
        "segment_idx": row.get("segment_idx"),
        "master_fps": task["master_fps"],
        "target_height": task["target_height"],
        "jpeg_quality": task["jpeg_quality"],
        "video_duration_s": row.get("video_duration_s"),
        "video_fps": row.get("video_fps"),
    }

    if not row.get("video_ok"):
        return {**base_row, "status": "skipped_video_not_ok", "num_records": 0}
    video_path = row.get("video_path")
    if not video_path or not Path(video_path).exists():
        return {**base_row, "status": "skipped_no_video", "num_records": 0}

    # Resume: a finished segment already has its per-frame manifest (written
    # last, after the shard is closed). --force reprocesses regardless.
    if manifest_path.exists() and not task["force"]:
        existing = read_jsonl(manifest_path)
        return {
            **base_row,
            "status": "cached",
            "num_records": len(existing),
            "shard_path": str(segment_frame_dir / "images.array_record"),
            "total_jpeg_bytes": sum(int(r.get("jpeg_bytes", 0)) for r in existing),
        }

    try:
        extract_frames_ffmpeg(
            video_path=Path(video_path),
            output_dir=segment_frame_dir,
            target_fps=task["master_fps"],
            target_height=task["target_height"],
            jpeg_quality=task["jpeg_quality"],
            ffmpeg_bin=task["ffmpeg_bin"],
        )
    except Exception as exc:  # noqa: BLE001 - one segment failing must not kill the run
        return {**base_row, "status": "failed", "num_records": 0, "error": f"{type(exc).__name__}: {exc}"}

    frame_paths = sorted(segment_frame_dir.glob("frame_*.jpg"))
    packed = pack_master_arrayrecord(
        frame_paths,
        segment_frame_dir,
        master_fps=task["master_fps"],
        video_fps=float(row.get("video_fps") or 0.0),
        video_frame_count=int(row.get("video_frame_count") or 0),
    )
    return {
        **base_row,
        "status": "ok" if packed["num_records"] else "empty",
        "num_records": packed["num_records"],
        "shard_path": packed.get("shard_path"),
        "frame_manifest": packed.get("manifest_path"),
        "total_jpeg_bytes": packed["total_jpeg_bytes"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--clips-manifest",
        type=Path,
        required=True,
        help="discover clips_manifest.jsonl (or any manifest with segment_id/"
        "video_path/video_ok/video_fps/video_duration_s rows). Keylog-free: the "
        "decode is alignment-agnostic, so realignment is not needed here.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--master-fps",
        type=float,
        default=DEFAULT_MASTER_FPS,
        help="Uniform frame rate to decode+store at. This is the CEILING for "
        "downstream sampling (you can only sample DOWN from it). Storage scales "
        f"linearly with it. Default {DEFAULT_MASTER_FPS}.",
    )
    p.add_argument("--target-height", type=int, default=config.DEFAULT_TARGET_HEIGHT)
    p.add_argument("--jpeg-quality", type=int, default=config.DEFAULT_JPEG_QUALITY)
    p.add_argument("--ffmpeg-bin", default=config.ffmpeg_bin())
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel segment decodes (0 = cpu_count()). Each spawns one ffmpeg "
        "(itself capped via JUERGEN_ANNOTATION_FFMPEG_THREADS, default 4). Run on "
        "a CPU allocation, not the shared login node.",
    )
    p.add_argument("--limit", type=int, default=None, help="Process only the first N segments (debug).")
    p.add_argument("--force", action="store_true", help="Re-decode segments that already have a frame_manifest.jsonl.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.master_fps <= 0:
        raise SystemExit("--master-fps must be > 0")
    ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)

    out_dir = ensure_dir(args.output_dir)
    frames_dir = ensure_dir(out_dir / "frames")

    rows = read_jsonl(args.clips_manifest)
    if not rows:
        raise RuntimeError(f"Empty clip manifest: {args.clips_manifest}")
    if args.limit is not None:
        rows = rows[: args.limit]

    tasks = [
        {
            "row": row,
            "frames_dir": str(frames_dir),
            "master_fps": args.master_fps,
            "target_height": args.target_height,
            "jpeg_quality": args.jpeg_quality,
            "ffmpeg_bin": ffmpeg_bin,
            "force": args.force,
        }
        for row in rows
    ]

    n_workers = args.num_workers or mp.cpu_count()
    n_workers = max(1, min(n_workers, len(tasks)))
    print(
        f"[frames_master] {len(tasks)} segments | master_fps={args.master_fps} "
        f"height={args.target_height} q={args.jpeg_quality} | workers={n_workers}",
        flush=True,
    )

    counts: Counter = Counter()
    index_rows: list[dict[str, Any]] = []
    n_records_total = 0
    total_jpeg_bytes = 0
    with mp.Pool(n_workers) as pool:
        # Results arrive out of order (imap_unordered); the bar just counts
        # completions, giving live elapsed / rate / ETA over the whole dataset.
        # mininterval throttles redraws so a slurm log isn't spammed.
        bar = tqdm(
            pool.imap_unordered(build_segment_master, tasks, chunksize=4),
            total=len(tasks), unit="seg", desc="[frames_master] decode",
            smoothing=0.05, mininterval=2.0, dynamic_ncols=True,
        )
        for res in bar:
            counts[res["status"]] += 1
            n_records_total += int(res.get("num_records") or 0)
            total_jpeg_bytes += int(res.get("total_jpeg_bytes") or 0)
            index_rows.append(res)
            if res["status"] == "failed":
                bar.write(f"  FAIL {res['segment_id']}: {res.get('error')}")
            bar.set_postfix(
                ok=counts.get("ok", 0), cached=counts.get("cached", 0),
                fail=counts.get("failed", 0), gb=round(total_jpeg_bytes / 1e9, 2),
                refresh=False,
            )
        bar.close()

    index_rows.sort(key=lambda r: str(r["segment_id"]))
    with (out_dir / "segment_index.jsonl").open("w") as f:
        for r in index_rows:
            f.write(json.dumps(r) + "\n")

    summary = {
        "master_fps": args.master_fps,
        "target_height": args.target_height,
        "jpeg_quality": args.jpeg_quality,
        "n_segments": len(tasks),
        "status_counts": dict(counts),
        "n_records_total": n_records_total,
        "total_jpeg_bytes": total_jpeg_bytes,
        "ffmpeg_bin": ffmpeg_bin,
        "source_clips_manifest": str(args.clips_manifest),
    }
    write_json(out_dir / "frames_master_summary.json", summary)
    write_json(
        out_dir / "manifest.json",
        {
            "artifact_type": "juergen_annotation_frames_master",
            "schema_version": 1,
            "segment_index": "segment_index.jsonl",
            **summary,
        },
    )
    print(
        f"[frames_master] done: {dict(counts)} | {n_records_total} records, "
        f"{total_jpeg_bytes / 1e9:.2f} GB -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
