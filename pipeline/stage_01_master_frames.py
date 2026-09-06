"""Stage 01a (frames-master): decode every segment's mp4 once into a JPEG
ArrayRecord frame store, so downstream fps experiments never re-decode.

Reads the discover ``clips_manifest.jsonl`` and, per segment, extracts frames
at a fixed master fps into one ``images.array_record`` (grain store),
referenced as ``ar:///abs/path/images.array_record#idx`` -- the same URI scheme
``image_store`` / stage 01 / stage 02 already consume.

A downstream sampler (01b) then picks the nearest master record per target-fps
bin and bins the keylog actions, emitting a metadata-only dataset (no new JPEG
bytes). The master fps is therefore the sampling ceiling: you can sample down
to any fps <= master, never up. Pick it as the highest fps you will ever want
(storage scales linearly with it). No action binning / NO_OP thinning happens
here -- those are fps-dependent and belong to 01b.

Frames are stored CFR at ``master_fps`` (ffmpeg's ``fps=`` filter resamples any
VFR source), so master record ``i`` is at ``source_time_s = i / master_fps``.
``source_frame_idx`` (nearest frame in the original video) is recorded for
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

from PIL import Image
from tqdm import tqdm

# Make the ``pipeline`` package importable when this script
# is run directly from its own folder (mirrors run_dataset's PYTHONPATH setup).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from image_domain import encode_jpeg_q92
from pipeline.lib import config
from pipeline.lib.common import ensure_dir, read_jsonl, write_json
from pipeline.lib.frames_actions import (
    extract_frames_ffmpeg,
    resolve_ffmpeg_bin,
)
from pipeline.lib.image_store import make_arrayrecord_image_uri
from pipeline.lib.manifest import file_sha256_short
from pipeline.lib.master_frames import (
    resolve_master_artifact,
    validate_master_segment,
    validate_master_segment_receipt,
)
from pipeline.lib.source_clips import resolve_source_clips

DEFAULT_MASTER_FPS = 4.0


def _luma_metrics(jpeg: bytes) -> tuple[float, float]:
    """Black-frame detection for one JPEG, computed in a single grayscale
    histogram pass: ``(mean_luma, frac_dark)`` where mean_luma is 0-255 and
    frac_dark is the fraction of pixels below ``config.BLACK_DARK_CUTOFF``.

    These are raw metrics, not a boolean; the filter applies the threshold."""
    import io

    from PIL import Image

    with Image.open(io.BytesIO(jpeg)) as im:
        im.load()
        hist = im.convert("L").histogram()
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
    from array_record.python.array_record_module import (
        ArrayRecordWriter,
    )

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
                with Image.open(frame_path) as frame:
                    frame.load()
                    jpeg = encode_jpeg_q92(frame)
                writer.write(jpeg)
                source_time_s = record_index / master_fps
                source_frame_idx = (
                    min(max(0, round(source_time_s * video_fps)), max_src_idx)
                    if video_fps > 0
                    else 0
                )
                # Metrics only; the sampler applies the drop threshold.
                mean_luma, frac_dark = _luma_metrics(jpeg)
                manifest_f.write(
                    json.dumps(
                        {
                            "record_index": record_index,
                            "image": make_arrayrecord_image_uri(
                                shard_path, record_index
                            ),
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
        "shard_sha256": file_sha256_short(shard_path, n=64),
        "frame_manifest_sha256": file_sha256_short(manifest_path, n=64),
        "num_records": len(frame_paths),
        "total_jpeg_bytes": total_jpeg_bytes,
    }


def _cached_segment(
    segment_frame_dir: Path, expected_inputs: dict[str, Any]
) -> dict[str, Any] | None:
    marker_path = segment_frame_dir / "segment_manifest.json"
    if not marker_path.is_file():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        not isinstance(marker, dict)
        or set(marker) != {"schema_version", "inputs", "outputs"}
        or marker.get("schema_version") != 1
    ):
        raise ValueError(f"invalid cached segment marker: {marker_path}")
    if marker.get("inputs") != expected_inputs:
        return None
    outputs = marker.get("outputs")
    required = {
        "frame_manifest_sha256",
        "num_records",
        "shard_sha256",
        "total_jpeg_bytes",
    }
    if not isinstance(outputs, dict) or set(outputs) != required:
        raise ValueError(f"invalid cached segment marker: {marker_path}")
    shard = segment_frame_dir / "images.array_record"
    frame_manifest = segment_frame_dir / "frame_manifest.jsonl"
    if not shard.is_file() or not frame_manifest.is_file():
        raise FileNotFoundError(
            f"cached segment payload is incomplete: {segment_frame_dir}"
        )
    if file_sha256_short(shard, n=64) != outputs["shard_sha256"]:
        raise ValueError(f"cached segment shard digest mismatch: {shard}")
    if file_sha256_short(frame_manifest, n=64) != outputs["frame_manifest_sha256"]:
        raise ValueError(f"cached frame manifest digest mismatch: {frame_manifest}")
    rows = read_jsonl(frame_manifest)
    if not rows or len(rows) != outputs["num_records"]:
        raise ValueError(f"cached frame count mismatch: {frame_manifest}")
    if sum(int(row["jpeg_bytes"]) for row in rows) != outputs["total_jpeg_bytes"]:
        raise ValueError(f"cached JPEG byte count mismatch: {frame_manifest}")
    from array_record.python.array_record_module import ArrayRecordReader

    reader = ArrayRecordReader(str(shard))
    try:
        if reader.num_records() != len(rows):
            raise ValueError(f"cached ArrayRecord count mismatch: {shard}")
    finally:
        reader.close()
    return {
        "shard_path": str(shard),
        "frame_manifest": str(frame_manifest),
        **outputs,
    }


def _write_atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def build_segment_master(task: dict[str, Any]) -> dict[str, Any]:
    """Decode one segment and return its canonical index row."""
    row = task["row"]
    seg = str(row["segment_id"])
    segment_frame_dir = ensure_dir(Path(task["frames_dir"]) / seg)
    base_row = {
        "segment_id": seg,
        "recording_id": row.get("recording_id"),
        "segment_idx": row.get("segment_idx"),
        "master_fps": task["master_fps"],
        "target_height": task["target_height"],
        "jpeg_quality": task["jpeg_quality"],
        "video_duration_s": row.get("video_duration_s"),
        "video_fps": row.get("video_fps"),
        "video_sha256": row["video_sha256"],
    }

    video_path = row.get("video_path")
    if row.get("video_ok") is not True or not isinstance(video_path, str):
        raise ValueError(f"source segment {seg} has no canonical video")
    video = Path(video_path)
    if not video.is_file() or video.stat().st_size == 0:
        raise FileNotFoundError(f"source video is missing or empty: {video}")
    observed_video_sha = file_sha256_short(video, n=64)
    if observed_video_sha != row.get("video_sha256"):
        raise ValueError(f"source video digest mismatch: {video}")

    cache_inputs = {
        "jpeg_quality": task["jpeg_quality"],
        "master_fps": task["master_fps"],
        "target_height": task["target_height"],
        "video_sha256": row["video_sha256"],
    }
    if cached := _cached_segment(segment_frame_dir, cache_inputs):
        result = {**base_row, "status": "ok", **cached}
        validate_master_segment(
            result,
            root=Path(task["frames_dir"]).parent,
            source_row=row,
        )
        return result
    (segment_frame_dir / "segment_manifest.json").unlink(missing_ok=True)

    extract_frames_ffmpeg(
        video_path=video,
        output_dir=segment_frame_dir,
        target_fps=task["master_fps"],
        target_height=task["target_height"],
        jpeg_quality=task["jpeg_quality"],
        ffmpeg_bin=task["ffmpeg_bin"],
    )

    frame_paths = sorted(segment_frame_dir.glob("frame_*.jpg"))
    packed = pack_master_arrayrecord(
        frame_paths,
        segment_frame_dir,
        master_fps=task["master_fps"],
        video_fps=float(row.get("video_fps") or 0.0),
        video_frame_count=int(row.get("video_frame_count") or 0),
    )
    if not packed["num_records"]:
        raise ValueError(f"source video decoded no frames: {video}")
    outputs = {
        key: packed[key]
        for key in (
            "frame_manifest_sha256",
            "num_records",
            "shard_sha256",
            "total_jpeg_bytes",
        )
    }
    _write_atomic_json(
        segment_frame_dir / "segment_manifest.json",
        {"schema_version": 1, "inputs": cache_inputs, "outputs": outputs},
    )
    result = {
        **base_row,
        "status": "ok",
        "shard_path": packed["shard_path"],
        "frame_manifest": packed["manifest_path"],
        **outputs,
    }
    validate_master_segment(
        result,
        root=Path(task["frames_dir"]).parent,
        source_row=row,
    )
    return result


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
    source: dict[str, str],
) -> dict[str, Any]:
    """Roll per-segment index rows up into the summary dict. Shared by the
    single-shard decode path and the merge path so both emit the same shape."""
    counts: Counter = Counter(str(r.get("status", "unknown")) for r in index_rows)
    return {
        "master_fps": master_fps,
        "target_height": target_height,
        "jpeg_quality": jpeg_quality,
        "n_segments": len(index_rows),
        "status_counts": dict(sorted(counts.items())),
        "n_records_total": sum(int(r.get("num_records") or 0) for r in index_rows),
        "total_jpeg_bytes": sum(
            int(r.get("total_jpeg_bytes") or 0) for r in index_rows
        ),
        "ffmpeg_bin": ffmpeg_bin,
        "source_clips_manifest": source["path"],
        "source_clips_sha256": source["sha256"],
        "source_clips_id": source["artifact_id"],
    }


def write_summary_and_manifest(out_dir: Path, summary: dict[str, Any]) -> None:
    """Write the aggregate summary + the manifest.json artifact marker."""
    write_json(out_dir / "frames_master_summary.json", summary)
    _write_atomic_json(
        out_dir / "manifest.json",
        {
            **ARTIFACT_MARKER,
            **summary,
            "segment_index_sha256": file_sha256_short(
                out_dir / "segment_index.jsonl", n=64
            ),
        },
    )


def run_merge(args: argparse.Namespace) -> None:
    """Publish one master-frame artifact from a complete shard set."""
    out_dir = args.output_dir
    n = args.num_shards
    if n < 2:
        raise SystemExit("--merge requires --num-shards > 1")
    if not out_dir.is_dir():
        raise SystemExit(f"--merge: --output-dir does not exist: {out_dir}")
    (out_dir / "manifest.json").unlink(missing_ok=True)

    suffix = f"_of_{n:04d}"
    shard_index_files = [
        out_dir / f"segment_index.shard{index:04d}{suffix}.jsonl" for index in range(n)
    ]
    observed_indexes = set(out_dir.glob(f"segment_index.shard*{suffix}.jsonl"))
    if observed_indexes != set(shard_index_files):
        raise RuntimeError("master-frame shard index set is incomplete or noncanonical")
    present = list(range(n))

    by_seg: dict[str, dict[str, Any]] = {}
    for sf in shard_index_files:
        for row in read_jsonl(sf):
            segment_id = str(row["segment_id"])
            if segment_id in by_seg:
                raise ValueError(
                    f"duplicate segment across shard indexes: {segment_id}"
                )
            by_seg[segment_id] = row
    index_rows = list(by_seg.values())
    if not index_rows or any(row.get("status") != "ok" for row in index_rows):
        raise RuntimeError(
            "cannot publish an empty or incomplete master frame artifact"
        )

    shard_summaries = [
        out_dir / f"frames_master_summary.shard{index:04d}{suffix}.json"
        for index in range(n)
    ]
    observed_summaries = set(out_dir.glob(f"frames_master_summary.shard*{suffix}.json"))
    if observed_summaries != set(shard_summaries):
        raise RuntimeError(
            "master-frame shard summary set is incomplete or noncanonical"
        )
    summary_by_index = {
        int(path.name.split(".shard")[1].split("_of_")[0]): json.loads(path.read_text())
        for path in shard_summaries
    }
    if set(summary_by_index) != set(range(n)):
        raise RuntimeError(
            f"missing or duplicate shard summaries: "
            f"{sorted(set(range(n)) - set(summary_by_index))}"
        )
    scalar_fields = (
        "ffmpeg_bin",
        "jpeg_quality",
        "master_fps",
        "num_shards",
        "source_clips_id",
        "source_clips_manifest",
        "source_clips_sha256",
        "target_height",
    )
    scalars = summary_by_index[0]
    source_rows, source = resolve_source_clips(Path(scalars["source_clips_manifest"]))
    if any(
        scalars[field] != source[key]
        for field, key in (
            ("source_clips_manifest", "path"),
            ("source_clips_sha256", "sha256"),
            ("source_clips_id", "artifact_id"),
        )
    ):
        raise ValueError("master-frame shard source identity is stale")
    source_by_segment = {str(row["segment_id"]): row for row in source_rows}
    for index, shard_summary in summary_by_index.items():
        shard_rows = read_jsonl(shard_index_files[index])
        if shard_summary.get("shard_index") != index or any(
            shard_summary.get(field) != scalars.get(field) for field in scalar_fields
        ):
            raise ValueError(f"master-frame shard summary mismatch: {index}")
        expected_counts = {
            "n_segments": len(shard_rows),
            "n_records_total": sum(row["num_records"] for row in shard_rows),
            "total_jpeg_bytes": sum(row["total_jpeg_bytes"] for row in shard_rows),
            "status_counts": {"ok": len(shard_rows)},
        }
        if any(
            shard_summary.get(field) != value
            for field, value in expected_counts.items()
        ):
            raise ValueError(f"master-frame shard summary counts mismatch: {index}")
        expected_segments = [str(row["segment_id"]) for row in source_rows[index::n]]
        if [str(row.get("segment_id")) for row in shard_rows] != expected_segments:
            raise ValueError(f"master-frame shard source coverage mismatch: {index}")
    if set(by_seg) != set(source_by_segment):
        raise ValueError("master-frame shards do not cover the exact Stage00 inventory")
    for row in index_rows:
        if any(
            row.get(field) != scalars[field]
            for field in ("jpeg_quality", "master_fps", "target_height")
        ):
            raise ValueError(
                f"master-frame row parameters mismatch: {row['segment_id']}"
            )
        validate_master_segment_receipt(
            row,
            root=out_dir,
            source_row=source_by_segment[str(row["segment_id"])],
        )
    summary = aggregate_summary(
        index_rows,
        master_fps=scalars["master_fps"],
        target_height=scalars["target_height"],
        jpeg_quality=scalars["jpeg_quality"],
        ffmpeg_bin=scalars["ffmpeg_bin"],
        source=source,
    )
    summary["num_shards"] = n
    summary["merged_shards"] = present

    for path in (*shard_index_files, *shard_summaries):
        path.unlink()
    write_index_jsonl(out_dir / "segment_index.jsonl", index_rows)
    write_summary_and_manifest(out_dir, summary)
    try:
        resolve_master_artifact(out_dir)
    except Exception:
        (out_dir / "manifest.json").unlink(missing_ok=True)
        raise
    print(
        f"[merge] {len(shard_index_files)} shards -> {len(index_rows)} segments, "
        f"{summary['n_records_total']} records, {summary['total_jpeg_bytes'] / 1e9:.2f} GB "
        f"| status={summary['status_counts']} -> {out_dir}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--clips-manifest",
        type=Path,
        default=None,
        help="Stage00 clips_manifest.jsonl. Required for decoding; ignored in "
        "--merge mode.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--master-fps",
        type=float,
        default=DEFAULT_MASTER_FPS,
        help="Uniform frame rate to decode+store at. This is the ceiling for "
        "downstream sampling (you can only sample down from it). Storage scales "
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
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the manifest into N disjoint round-robin shards for parallel "
        "jobs; this shard processes rows[shard_index::num_shards]. All shards must "
        "share one --output-dir: per-segment frame dirs are keyed by segment_id and "
        "so never collide. With N>1 each shard writes "
        "segment_index.shard<I>_of_<N>.jsonl, not the top-level manifest.json; run "
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
    out_dir = ensure_dir(args.output_dir.resolve())
    (out_dir / "manifest.json").unlink(missing_ok=True)
    if args.merge:
        run_merge(args)
        return
    if args.master_fps <= 0:
        raise SystemExit("--master-fps must be > 0")
    if args.target_height <= 0:
        raise SystemExit("--target-height must be > 0")
    if args.jpeg_quality != config.DEFAULT_JPEG_QUALITY:
        raise SystemExit(f"--jpeg-quality must be {config.DEFAULT_JPEG_QUALITY}")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(
            f"--shard-index must be in [0, {args.num_shards}); got {args.shard_index}"
        )
    if args.clips_manifest is None:
        raise SystemExit(
            "--clips-manifest is required for decoding (only --merge omits it)"
        )
    ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg_bin)

    frames_dir = ensure_dir(out_dir / "frames")

    rows, source = resolve_source_clips(args.clips_manifest)
    sharded = args.num_shards > 1
    if sharded:
        # Round-robin stride: disjoint and exhaustive across shard 0..N-1, and
        # load-balances better than contiguous blocks when long segments cluster
        # together in the manifest. read_jsonl preserves file order, so the split
        # is deterministic given the same clips_manifest.
        rows = rows[args.shard_index :: args.num_shards]
    if not rows:
        raise RuntimeError("selected master-frame shard has no segments")

    tasks = [
        {
            "row": row,
            "frames_dir": str(frames_dir),
            "master_fps": args.master_fps,
            "target_height": args.target_height,
            "jpeg_quality": args.jpeg_quality,
            "ffmpeg_bin": ffmpeg_bin,
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
        # Results arrive out of order (imap_unordered); the bar counts
        # completions. mininterval throttles redraws so a slurm log isn't spammed.
        bar = tqdm(
            pool.imap_unordered(build_segment_master, tasks, chunksize=4),
            total=len(tasks),
            unit="seg",
            desc="[frames_master] decode",
            smoothing=0.05,
            mininterval=2.0,
            dynamic_ncols=True,
        )
        for res in bar:
            counts[res["status"]] += 1
            n_records_total += int(res.get("num_records") or 0)
            total_jpeg_bytes += int(res.get("total_jpeg_bytes") or 0)
            index_rows.append(res)
            bar.set_postfix(
                ok=counts.get("ok", 0),
                fail=counts.get("failed", 0),
                gb=round(total_jpeg_bytes / 1e9, 2),
                refresh=False,
            )
        bar.close()

    if any(row.get("status") != "ok" for row in index_rows):
        raise RuntimeError("master-frame decode did not complete every segment")

    _, current_source = resolve_source_clips(args.clips_manifest)
    if current_source != source:
        raise RuntimeError("Stage00 source identity changed during Stage01")
    summary = aggregate_summary(
        index_rows,
        master_fps=args.master_fps,
        target_height=args.target_height,
        jpeg_quality=args.jpeg_quality,
        ffmpeg_bin=ffmpeg_bin,
        source=source,
    )
    if sharded:
        # Shard task: write a uniquely-named index + summary (no collision between
        # the N tasks sharing this dir) and leave manifest.json alone -- the merge
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
        source_by_segment = {str(row["segment_id"]): row for row in rows}
        if {str(row["segment_id"]) for row in index_rows} != set(source_by_segment):
            raise ValueError("Stage01 index does not cover the exact Stage00 inventory")
        write_index_jsonl(out_dir / "segment_index.jsonl", index_rows)
        write_summary_and_manifest(out_dir, summary)
        try:
            resolve_master_artifact(out_dir)
        except Exception:
            (out_dir / "manifest.json").unlink(missing_ok=True)
            raise
        print(
            f"[frames_master] done: {dict(counts)} | {n_records_total} records, "
            f"{total_jpeg_bytes / 1e9:.2f} GB -> {out_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
