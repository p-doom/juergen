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

# Make the ``pipeline`` package importable when this script
# is run directly from its own folder (mirrors run_dataset's PYTHONPATH setup).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.lib import config  # noqa: E402
from pipeline.lib.common import ensure_dir, read_jsonl, write_json  # noqa: E402
from pipeline.lib.image_store import make_arrayrecord_image_uri  # noqa: E402
from pipeline.lib.frames_actions import (  # noqa: E402
    extract_frames_ffmpeg,
    resolve_ffmpeg_bin,
)

DEFAULT_MASTER_FPS = 4.0


def _luma_metrics(jpeg: bytes) -> "tuple[float | None, float | None]":
    """Black-frame detection for one JPEG, computed in a single grayscale
    histogram pass: ``(mean_luma, frac_dark)`` where mean_luma is 0-255 and
    frac_dark is the fraction of pixels below ``config.BLACK_DARK_CUTOFF``.

    These are raw METRICS, not a boolean -- the sampler (01b) applies the
    threshold, so it stays tunable without re-decoding. Returns ``(None, None)``
    if the frame can't be decoded, so one bad frame never aborts the build and
    the sampler simply won't be able to drop it (absence of evidence != black)."""
    try:
        import io  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        with Image.open(io.BytesIO(jpeg)) as im:
            hist = im.convert("L").histogram()  # 256 luma bins
    except Exception:  # noqa: BLE001 — detection is best-effort, never fatal
        return None, None
    total = sum(hist) or 1
    mean_luma = sum(i * c for i, c in enumerate(hist)) / total
    frac_dark = sum(hist[: config.BLACK_DARK_CUTOFF]) / total
    return round(mean_luma, 3), round(frac_dark, 5)


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
                # Black-frame FLAG only: metrics are recorded here (the one place
                # that holds pixels); the sampler applies the drop threshold.
                mean_luma, frac_dark = _luma_metrics(jpeg)
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
                            "mean_luma": mean_luma,
                            "frac_dark": frac_dark,
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


ARTIFACT_MARKER = {
    "artifact_type": "juergen_annotation_frames_master",
    "schema_version": 1,
    "segment_index": "segment_index.jsonl",
}


def write_index_jsonl(path: Path, index_rows: list[dict[str, Any]]) -> None:
    """Write the per-segment index, one JSON row per line, sorted by segment_id."""
    with path.open("w") as f:
        for r in sorted(index_rows, key=lambda r: str(r["segment_id"])):
            f.write(json.dumps(r) + "\n")


def aggregate_summary(
    index_rows: list[dict[str, Any]],
    *,
    master_fps: float,
    target_height: int,
    jpeg_quality: int,
    ffmpeg_bin: str | None,
    source_clips_manifest: str | None,
) -> dict[str, Any]:
    """Roll per-segment index rows up into the summary dict. Shared by the
    single-shard decode path and the merge path so both emit the same shape."""
    counts: Counter = Counter(str(r.get("status", "unknown")) for r in index_rows)
    return {
        "master_fps": master_fps,
        "target_height": target_height,
        "jpeg_quality": jpeg_quality,
        "n_segments": len(index_rows),
        "status_counts": dict(counts),
        "n_records_total": sum(int(r.get("num_records") or 0) for r in index_rows),
        "total_jpeg_bytes": sum(int(r.get("total_jpeg_bytes") or 0) for r in index_rows),
        "ffmpeg_bin": ffmpeg_bin,
        "source_clips_manifest": source_clips_manifest,
    }


def write_summary_and_manifest(out_dir: Path, summary: dict[str, Any]) -> None:
    """Write the aggregate summary + the manifest.json artifact marker."""
    write_json(out_dir / "frames_master_summary.json", summary)
    write_json(out_dir / "manifest.json", {**ARTIFACT_MARKER, **summary})


def run_merge(args: argparse.Namespace) -> None:
    """Fold the per-shard segment_index.shard*_of_<N>.jsonl files (written by the
    array-job shard tasks into a SHARED --output-dir) into the canonical
    segment_index.jsonl, then write the summary + manifest.json marker. Scoped to
    ``_of_<num_shards>`` so stale files from a run with a different shard count are
    ignored, and deduped by segment_id so any accidental overlap can't double-count.
    """
    out_dir = args.output_dir
    n = args.num_shards
    if n < 2:
        raise SystemExit("--merge requires --num-shards > 1")
    if not out_dir.is_dir():
        raise SystemExit(f"--merge: --output-dir does not exist: {out_dir}")

    suffix = f"_of_{n:04d}"
    shard_index_files = sorted(out_dir.glob(f"segment_index.shard*{suffix}.jsonl"))
    if not shard_index_files:
        raise SystemExit(
            f"[merge] no segment_index.shard*{suffix}.jsonl under {out_dir} "
            f"(did the shard tasks run with --num-shards {n}?)"
        )
    present = sorted(int(p.name.split(".shard")[1].split("_of_")[0]) for p in shard_index_files)
    missing = sorted(set(range(n)) - set(present))
    if missing:
        print(f"[merge] WARNING: no shard index for shards {missing}", flush=True)

    by_seg: dict[str, dict[str, Any]] = {}
    for sf in shard_index_files:
        for row in read_jsonl(sf):
            by_seg[str(row["segment_id"])] = row
    index_rows = list(by_seg.values())

    # Decode scalars (ffmpeg_bin / source manifest) live in the shard summaries;
    # fall back to the first index row for the numerics if a summary is missing.
    shard_summaries = sorted(out_dir.glob(f"frames_master_summary.shard*{suffix}.json"))
    scalars = json.loads(shard_summaries[0].read_text()) if shard_summaries else {}
    first = index_rows[0] if index_rows else {}
    summary = aggregate_summary(
        index_rows,
        master_fps=scalars.get("master_fps", first.get("master_fps")),
        target_height=scalars.get("target_height", first.get("target_height")),
        jpeg_quality=scalars.get("jpeg_quality", first.get("jpeg_quality")),
        ffmpeg_bin=scalars.get("ffmpeg_bin"),
        source_clips_manifest=scalars.get("source_clips_manifest"),
    )
    summary["num_shards"] = n
    summary["merged_shards"] = present

    write_index_jsonl(out_dir / "segment_index.jsonl", index_rows)
    write_summary_and_manifest(out_dir, summary)
    print(
        f"[merge] {len(shard_index_files)} shards -> {len(index_rows)} segments, "
        f"{summary['n_records_total']} records, {summary['total_jpeg_bytes'] / 1e9:.2f} GB "
        f"| status={summary['status_counts']} -> {out_dir}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--clips-manifest",
        type=Path,
        default=None,
        help="discover clips_manifest.jsonl (or any manifest with segment_id/"
        "video_path/video_ok/video_fps/video_duration_s rows). Keylog-free: the "
        "decode is alignment-agnostic, so realignment is not needed here. Required "
        "for decoding; ignored in --merge mode.",
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
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the manifest into N disjoint round-robin shards for parallel "
        "jobs; this shard processes rows[shard_index::num_shards]. All shards MUST "
        "share ONE --output-dir: per-segment frame dirs are keyed by segment_id and "
        "so never collide. With N>1 each shard writes "
        "segment_index.shard<I>_of_<N>.jsonl (NOT the top-level manifest.json); run "
        "--merge afterwards to fold them into the canonical segment_index.jsonl + "
        "manifest.json marker.",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="This job's shard, in [0, num_shards). Injected from "
        "$SLURM_ARRAY_TASK_ID by the labctl [sweep].",
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help="Merge mode: fold every segment_index.shard*_of_<num_shards>.jsonl "
        "under --output-dir into the canonical segment_index.jsonl and write the "
        "frames_master_summary.json + manifest.json marker. Decodes nothing; only "
        "--output-dir and --num-shards are read.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge:
        run_merge(args)
        return
    if args.master_fps <= 0:
        raise SystemExit("--master-fps must be > 0")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index must be in [0, {args.num_shards}); got {args.shard_index}")
    if args.clips_manifest is None:
        raise SystemExit("--clips-manifest is required for decoding (only --merge omits it)")
    ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)

    out_dir = ensure_dir(args.output_dir)
    frames_dir = ensure_dir(out_dir / "frames")

    rows = read_jsonl(args.clips_manifest)
    if not rows:
        raise RuntimeError(f"Empty clip manifest: {args.clips_manifest}")
    if args.limit is not None:
        rows = rows[: args.limit]
    sharded = args.num_shards > 1
    if sharded:
        # Round-robin stride: disjoint AND exhaustive across shard 0..N-1, and
        # load-balances better than contiguous blocks when long segments cluster
        # together in the manifest. read_jsonl preserves file order, so the split
        # is deterministic given the same clips_manifest.
        rows = rows[args.shard_index :: args.num_shards]

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
    shard_note = f" | shard {args.shard_index}/{args.num_shards}" if sharded else ""
    print(
        f"[frames_master] {len(tasks)} segments{shard_note} | master_fps={args.master_fps} "
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

    summary = aggregate_summary(
        index_rows,
        master_fps=args.master_fps,
        target_height=args.target_height,
        jpeg_quality=args.jpeg_quality,
        ffmpeg_bin=ffmpeg_bin,
        source_clips_manifest=str(args.clips_manifest),
    )
    if sharded:
        # Shard task: write a uniquely-named index + summary (no collision between
        # the N tasks sharing this dir) and DON'T touch manifest.json -- the merge
        # step writes the marker once, after all shards succeed.
        tag = f"shard{args.shard_index:04d}_of_{args.num_shards:04d}"
        write_index_jsonl(out_dir / f"segment_index.{tag}.jsonl", index_rows)
        summary["shard_index"] = args.shard_index
        summary["num_shards"] = args.num_shards
        write_json(out_dir / f"frames_master_summary.{tag}.json", summary)
        print(
            f"[frames_master] {tag} done: {dict(counts)} | {n_records_total} records, "
            f"{total_jpeg_bytes / 1e9:.2f} GB -> {out_dir} "
            f"(run --merge --num-shards {args.num_shards} to finalize)",
            flush=True,
        )
    else:
        write_index_jsonl(out_dir / "segment_index.jsonl", index_rows)
        write_summary_and_manifest(out_dir, summary)
        print(
            f"[frames_master] done: {dict(counts)} | {n_records_total} records, "
            f"{total_jpeg_bytes / 1e9:.2f} GB -> {out_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
