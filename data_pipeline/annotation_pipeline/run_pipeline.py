#!/usr/bin/env python3
"""Run the Crowd-Cast trajectory -> SFT pipeline over registered clips.

Single validated configuration: Qwen3.6-27B (BF16, sglang) annotates 90s
windows with thinking off; a verification pass (stage 02 pass C) is the quality
gate; canonical SFT export is stage 04. Labctl + omegalax own training token
counts and bucket materialization.

Stages:
  00 manifest          MP4 + keylog pairs for a clip's segment slice
  01 frames+actions    2fps 720p frames + per-frame action strings (cached)
  02 segment+name+verify   pass A boundaries -> pass B instructions -> pass C
                           grounded verification (writes a `verified` flag)
  03 assemble          verified trajectories -> run-level neutral samples
  04 canonical         run-level canonical SFT artifact
  05 buckets           optional local token/bucket distribution inspector

Every stage materializes a dataset-level artifact under
outputs/runs/<run-name>/stage_*. Resume state is tracked in each stage's
progress.jsonl ledger.

Requires a running sglang server (slurm/run_pipeline.sbatch starts one and sets
JUERGEN_ANNOTATION_VLM_BASE_URL). Example:
    python -m annotation_pipeline.run_pipeline --run-name smoke --clips bbbf_s0000-0003
    python -m annotation_pipeline.run_pipeline --run-name smoke --clips all
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

from annotation_pipeline import config


PIPELINE_DIR = Path(__file__).resolve().parent
# Frame cache + runs live under the dataset's processed/ dir, not the code tree.
OUTPUTS_DIR = config.PROCESSED_ROOT
CLIPS_PATH = PIPELINE_DIR / "clips.json"


def load_clips(path: Path = CLIPS_PATH) -> dict[str, dict[str, Any]]:
    return json.loads(Path(path).read_text())["clips"]


def read_json_file(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def read_jsonl_file(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl_file(path))


def iter_jsonl_file(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as fh:
        for line_num, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSONL at {path}:{line_num}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Bad JSONL at {path}:{line_num}: expected object")
            yield row


def write_json_file(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl_file(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def replace_jsonl_rows_for_clip(path: Path, clip_id: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        existing = [row for row in read_jsonl_file(path) if str(row.get("clip_id")) != clip_id]
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with tmp_path.open("w") as out:
            for row in existing:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            for row in rows:
                out.write(json.dumps(with_clip(row, clip_id), ensure_ascii=False) + "\n")
        tmp_path.replace(path)
        fcntl.flock(lock_fh, fcntl.LOCK_UN)


def with_clip(row: dict[str, Any], clip_id: str, source_path: Path | None = None) -> dict[str, Any]:
    result = dict(row)
    result.setdefault("clip_id", clip_id)
    if source_path is not None:
        result.setdefault("source_path", str(source_path))
    return result


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def progress_rows(stage_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl_file(stage_dir / "progress.jsonl")


def progress_by_clip(stage_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in progress_rows(stage_dir):
        clip_id = str(row.get("clip_id") or "")
        if clip_id:
            rows[clip_id] = row
    return rows


def progress_is_terminal(stage_dir: Path, clip_id: str) -> bool:
    status = str(progress_by_clip(stage_dir).get(clip_id, {}).get("status") or "")
    return status in {"done", "skipped"}


def write_progress(stage_dir: Path, clip_id: str, status: str, **extra: Any) -> None:
    row = {
        "clip_id": clip_id,
        "status": status,
        "updated_utc": utc_now(),
        **extra,
    }
    replace_jsonl_rows_for_clip(stage_dir / "progress.jsonl", clip_id, [row])


def rewrite_summary_from_progress(stage_dir: Path, filename: str, summary: dict[str, Any]) -> None:
    statuses = Counter(str(row.get("status") or "unknown") for row in progress_rows(stage_dir))
    summary["progress"] = dict(sorted(statuses.items()))
    write_json_file(stage_dir / filename, summary)


def aggregate_frame_cache_outputs(
    run_dir: Path,
    clips: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize stage 00/01 as run-level JSONL tables from the frame cache."""
    stage00_out = run_dir / "stage_00_manifest"
    stage01_out = run_dir / "stage_01_frames_actions"
    stage00_out.mkdir(parents=True, exist_ok=True)
    stage01_out.mkdir(parents=True, exist_ok=True)

    n_manifest_rows = 0
    n_frame_records = 0
    n_segment_summaries = 0
    manifest_clip_ids: set[str] = set()
    frame_clip_ids: set[str] = set()
    manifest_summaries: list[dict[str, Any]] = []
    frame_summaries: list[dict[str, Any]] = []
    stage00_progress: list[dict[str, Any]] = []
    stage01_progress: list[dict[str, Any]] = []

    with (
        (stage00_out / "manifest.jsonl").open("w") as manifest_fh,
        (stage01_out / "frame_records.jsonl").open("w") as frame_fh,
        (stage01_out / "segment_summaries.jsonl").open("w") as segment_fh,
    ):
        for clip_id, clip in sorted(clips.items()):
            cache = frames_cache_dir(
                clip,
                args.target_fps,
                args.target_height,
                args.stage01_max_noop_run,
                root=args.cache_root,
            )
            stage00 = cache / "stage_00_manifest"
            stage01 = cache / "stage_01_frames_actions"

            clip_manifest_rows = 0
            for row in iter_jsonl_file(stage00 / "manifest.jsonl"):
                manifest_clip_ids.add(clip_id)
                manifest_fh.write(json.dumps(with_clip(row, clip_id), ensure_ascii=False) + "\n")
                n_manifest_rows += 1
                clip_manifest_rows += 1

            manifest_summary_path = stage00 / "manifest_summary.json"
            manifest_summary = read_json_file(manifest_summary_path)
            if isinstance(manifest_summary, dict):
                manifest_clip_ids.add(clip_id)
                manifest_summaries.append(with_clip(manifest_summary, clip_id, manifest_summary_path))

            clip_frame_rows = 0
            for row in iter_jsonl_file(stage01 / "frame_records.jsonl"):
                frame_clip_ids.add(clip_id)
                frame_fh.write(json.dumps(with_clip(row, clip_id), ensure_ascii=False) + "\n")
                n_frame_records += 1
                clip_frame_rows += 1

            segment_summaries = read_json_file(stage01 / "segment_summaries.json", [])
            clip_segment_rows = 0
            if isinstance(segment_summaries, list):
                for row in segment_summaries:
                    if not isinstance(row, dict):
                        continue
                    frame_clip_ids.add(clip_id)
                    segment_fh.write(json.dumps(with_clip(row, clip_id), ensure_ascii=False) + "\n")
                    n_segment_summaries += 1
                    clip_segment_rows += 1

            frame_summary_path = stage01 / "frames_actions_summary.json"
            frame_summary = read_json_file(frame_summary_path)
            if isinstance(frame_summary, dict):
                frame_clip_ids.add(clip_id)
                frame_summaries.append(with_clip(frame_summary, clip_id, frame_summary_path))
            if clip_manifest_rows or isinstance(manifest_summary, dict):
                stage00_progress.append(
                    {
                        "clip_id": clip_id,
                        "status": "done",
                        "updated_utc": utc_now(),
                        "n_manifest_rows": clip_manifest_rows,
                    }
                )
            if clip_frame_rows or clip_segment_rows or isinstance(frame_summary, dict):
                stage01_progress.append(
                    {
                        "clip_id": clip_id,
                        "status": "done",
                        "updated_utc": utc_now(),
                        "n_frame_records": clip_frame_rows,
                        "n_segment_summaries": clip_segment_rows,
                    }
                )

    manifest_summaries.sort(key=lambda row: str(row.get("clip_id", "")))
    frame_summaries.sort(key=lambda row: str(row.get("clip_id", "")))
    write_jsonl_file(stage00_out / "clip_summaries.jsonl", manifest_summaries)
    write_jsonl_file(stage01_out / "clip_summaries.jsonl", frame_summaries)
    write_jsonl_file(stage00_out / "progress.jsonl", stage00_progress)
    write_jsonl_file(stage01_out / "progress.jsonl", stage01_progress)

    stage00_summary = {
        "artifact_type": "stage_00_manifest_dataset",
        "source": "aggregated_frame_cache_outputs",
        "n_clips": len(manifest_clip_ids),
        "n_manifest_rows": n_manifest_rows,
        "files": {
            "manifest": "manifest.jsonl",
            "clip_summaries": "clip_summaries.jsonl",
            "progress": "progress.jsonl",
        },
        "progress": dict(sorted(Counter(row["status"] for row in stage00_progress).items())),
    }
    stage01_summary = {
        "artifact_type": "stage_01_frames_actions_dataset",
        "source": "aggregated_frame_cache_outputs",
        "n_clips": len(frame_clip_ids),
        "n_frame_records": n_frame_records,
        "n_segment_summaries": n_segment_summaries,
        "target_fps": args.target_fps,
        "target_height": args.target_height,
        "max_noop_run": args.stage01_max_noop_run,
        "files": {
            "frame_records": "frame_records.jsonl",
            "segment_summaries": "segment_summaries.jsonl",
            "clip_summaries": "clip_summaries.jsonl",
            "progress": "progress.jsonl",
        },
        "progress": dict(sorted(Counter(row["status"] for row in stage01_progress).items())),
    }
    write_json_file(stage00_out / "manifest_summary.json", stage00_summary)
    write_json_file(stage01_out / "frames_actions_summary.json", stage01_summary)
    print(f"Aggregated stage 00: {stage00_summary['n_manifest_rows']} rows -> {stage00_out}")
    print(f"Aggregated stage 01: {stage01_summary['n_frame_records']} frames -> {stage01_out}")
    return stage00_summary, stage01_summary


def commit_frame_cache_clip(run_dir: Path, clip_id: str, cache: Path) -> None:
    stage00 = run_dir / "stage_00_manifest"
    stage01 = run_dir / "stage_01_frames_actions"
    cache00 = cache / "stage_00_manifest"
    cache01 = cache / "stage_01_frames_actions"

    manifest_rows = read_jsonl_file(cache00 / "manifest.jsonl")
    replace_jsonl_rows_for_clip(stage00 / "manifest.jsonl", clip_id, manifest_rows)
    manifest_summary = read_json_file(cache00 / "manifest_summary.json", {})
    replace_jsonl_rows_for_clip(
        stage00 / "clip_summaries.jsonl",
        clip_id,
        [manifest_summary] if isinstance(manifest_summary, dict) else [],
    )
    write_progress(stage00, clip_id, "done", n_manifest_rows=len(manifest_rows))

    frame_rows = read_jsonl_file(cache01 / "frame_records.jsonl")
    segment_summaries = read_json_file(cache01 / "segment_summaries.json", [])
    segment_rows = [row for row in segment_summaries if isinstance(row, dict)]
    frame_summary = read_json_file(cache01 / "frames_actions_summary.json", {})
    replace_jsonl_rows_for_clip(stage01 / "frame_records.jsonl", clip_id, frame_rows)
    replace_jsonl_rows_for_clip(stage01 / "segment_summaries.jsonl", clip_id, segment_rows)
    replace_jsonl_rows_for_clip(
        stage01 / "clip_summaries.jsonl",
        clip_id,
        [frame_summary] if isinstance(frame_summary, dict) else [],
    )
    write_progress(
        stage01,
        clip_id,
        "done",
        n_frame_records=len(frame_rows),
        n_segment_summaries=len(segment_rows),
    )

    rewrite_frame_stage_summaries(run_dir)


def rewrite_frame_stage_summaries(run_dir: Path) -> None:
    stage00 = run_dir / "stage_00_manifest"
    stage01 = run_dir / "stage_01_frames_actions"
    stage00_progress = progress_rows(stage00)
    stage01_progress = progress_rows(stage01)
    rewrite_summary_from_progress(
        stage00,
        "manifest_summary.json",
        {
            "artifact_type": "stage_00_manifest_dataset",
            "n_clips": sum(1 for row in stage00_progress if row.get("status") == "done"),
            "n_manifest_rows": sum(int(row.get("n_manifest_rows") or 0) for row in stage00_progress),
            "files": {
                "manifest": "manifest.jsonl",
                "clip_summaries": "clip_summaries.jsonl",
                "progress": "progress.jsonl",
            },
        },
    )
    rewrite_summary_from_progress(
        stage01,
        "frames_actions_summary.json",
        {
            "artifact_type": "stage_01_frames_actions_dataset",
            "n_clips": sum(1 for row in stage01_progress if row.get("status") == "done"),
            "n_frame_records": sum(int(row.get("n_frame_records") or 0) for row in stage01_progress),
            "n_segment_summaries": sum(
                int(row.get("n_segment_summaries") or 0) for row in stage01_progress
            ),
            "files": {
                "frame_records": "frame_records.jsonl",
                "segment_summaries": "segment_summaries.jsonl",
                "clip_summaries": "clip_summaries.jsonl",
                "progress": "progress.jsonl",
            },
        },
    )


def commit_stage02_clip(run_dir: Path, clip_id: str, temp_stage02: Path) -> None:
    stage02 = run_dir / "stage_02_segment"
    raw = read_json_file(temp_stage02 / "trajectories_raw.json", {})
    raw_rows = [raw] if isinstance(raw, dict) else []
    candidates = read_jsonl_file(temp_stage02 / "pass_a_candidates.jsonl")
    merged = read_jsonl_file(temp_stage02 / "pass_a_merged_segments.jsonl")
    rejected = read_json_file(temp_stage02 / "naming_rejected.json", [])
    rejected_rows = [row if isinstance(row, dict) else {"value": row} for row in rejected]
    summary = read_json_file(temp_stage02 / "stage02_summary.json", {})
    summary_rows = [summary] if isinstance(summary, dict) else []

    replace_jsonl_rows_for_clip(stage02 / "trajectories_raw.jsonl", clip_id, raw_rows)
    replace_jsonl_rows_for_clip(stage02 / "pass_a_candidates.jsonl", clip_id, candidates)
    replace_jsonl_rows_for_clip(stage02 / "pass_a_merged_segments.jsonl", clip_id, merged)
    replace_jsonl_rows_for_clip(stage02 / "naming_rejected.jsonl", clip_id, rejected_rows)
    replace_jsonl_rows_for_clip(stage02 / "clip_summaries.jsonl", clip_id, summary_rows)
    write_progress(
        stage02,
        clip_id,
        "done",
        n_candidates=len(candidates),
        n_merged_segments=len(merged),
        n_trajectories=len(raw.get("trajectories", []) if isinstance(raw, dict) else []),
        n_naming_rejected=len(rejected_rows),
    )
    rewrite_stage02_summary(stage02)


def rewrite_stage02_summary(stage02: Path) -> None:
    raw_rows = read_jsonl_file(stage02 / "trajectories_raw.jsonl")
    rewrite_summary_from_progress(
        stage02,
        "stage02_summary.json",
        {
            "artifact_type": "stage_02_segment_dataset",
            "n_clips": sum(1 for row in progress_rows(stage02) if row.get("status") == "done"),
            "n_candidates": sum(int(row.get("n_candidates") or 0) for row in progress_rows(stage02)),
            "n_merged_segments": sum(
                int(row.get("n_merged_segments") or 0) for row in progress_rows(stage02)
            ),
            "n_trajectories": sum(len(row.get("trajectories", []) or []) for row in raw_rows),
            "n_verified": sum(
                1
                for row in raw_rows
                for trajectory in (row.get("trajectories", []) or [])
                if isinstance(trajectory, dict) and trajectory.get("verified")
            ),
            "files": {
                "trajectories_raw": "trajectories_raw.jsonl",
                "pass_a_candidates": "pass_a_candidates.jsonl",
                "pass_a_merged_segments": "pass_a_merged_segments.jsonl",
                "naming_rejected": "naming_rejected.jsonl",
                "clip_summaries": "clip_summaries.jsonl",
                "progress": "progress.jsonl",
            },
        },
    )


def stage02_trajectory_for_clip(stage02: Path, clip_id: str) -> dict[str, Any] | None:
    for row in read_jsonl_file(stage02 / "trajectories_raw.jsonl"):
        if str(row.get("clip_id")) == clip_id:
            value = dict(row)
            value.pop("clip_id", None)
            return value
    return None


def commit_stage03_clip(run_dir: Path, clip_id: str, temp_stage03: Path) -> None:
    stage03 = run_dir / "stage_03_assemble"
    samples = read_jsonl_file(temp_stage03 / "trajectories.jsonl")
    rejected = read_jsonl_file(temp_stage03 / "rejected_trajectories.jsonl")
    summary = read_json_file(temp_stage03 / "assemble_summary.json", {})
    summary_rows = [summary] if isinstance(summary, dict) else []

    replace_jsonl_rows_for_clip(stage03 / "trajectories.jsonl", clip_id, samples)
    replace_jsonl_rows_for_clip(stage03 / "rejected_trajectories.jsonl", clip_id, rejected)
    replace_jsonl_rows_for_clip(stage03 / "clip_summaries.jsonl", clip_id, summary_rows)
    write_progress(stage03, clip_id, "done", n_samples=len(samples), n_rejected=len(rejected))
    rewrite_stage03_summary(stage03)


def rewrite_stage03_summary(stage03: Path) -> None:
    rejected = read_jsonl_file(stage03 / "rejected_trajectories.jsonl")
    reject_reasons = Counter(str(row.get("reason", "unknown")) for row in rejected)
    rewrite_summary_from_progress(
        stage03,
        "assemble_summary.json",
        {
            "artifact_type": "stage_03_assemble_dataset",
            "n_clips": sum(1 for row in progress_rows(stage03) if row.get("status") == "done"),
            "n_samples": sum(int(row.get("n_samples") or 0) for row in progress_rows(stage03)),
            "n_rejected": len(rejected),
            "reject_reasons": dict(sorted(reject_reasons.items())),
            "files": {
                "trajectories": "trajectories.jsonl",
                "rejected_trajectories": "rejected_trajectories.jsonl",
                "clip_summaries": "clip_summaries.jsonl",
                "progress": "progress.jsonl",
            },
        },
    )


def phase_done(args: argparse.Namespace, clip: dict[str, Any], clip_id: str, run_dir: Path) -> bool:
    """Whether the requested --stages output already exists (for --resume)."""
    if args.stages == "frames":
        return progress_is_terminal(run_dir / "stage_01_frames_actions", clip_id)
    if args.stages == "annotate":
        return progress_is_terminal(run_dir / "stage_02_segment", clip_id)
    return progress_is_terminal(run_dir / "stage_03_assemble", clip_id)


def run_step(script: str, args: list[str], python: str | None = None) -> None:
    module = f"annotation_pipeline.{Path(script).stem}"
    cmd = [python or sys.executable, "-m", module, *args]
    print("\n$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed with exit code {result.returncode}")


def noop_cache_label(max_noop_run: int) -> str:
    return "noopall" if max_noop_run < 0 else f"noop{max_noop_run}"


def frames_cache_dir(
    clip: dict[str, Any],
    fps: int,
    height: int,
    max_noop: int,
    *,
    root: Path | None = None,
) -> Path:
    rec8 = clip["recording_id"][:8]
    cache_root = root or (OUTPUTS_DIR / "cache" / "frames")
    return (
        cache_root
        / (f"{rec8}_s{clip['segment_start']:04d}-{clip['segment_end']:04d}"
           f"_{fps}fps_{height}p_{noop_cache_label(max_noop)}")
    )


def frames_cache_is_complete(cache: Path, fps: int, height: int, max_noop: int) -> bool:
    summary_path = cache / "stage_01_frames_actions" / "frames_actions_summary.json"
    records = cache / "stage_01_frames_actions" / "frame_records.jsonl"
    if not ((cache / ".complete").exists() and summary_path.exists() and records.exists()):
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return False
    return (
        summary.get("extraction_backend") == "ffmpeg"
        and int(summary.get("target_fps", -1)) == fps
        and int(summary.get("target_height", -1)) == height
        and int(summary.get("max_noop_run", -999)) == max_noop
    )


def ensure_frames_cache(clip: dict[str, Any], args: argparse.Namespace) -> Path:
    """Run stage 00+01 once per (clip, fps, height); reuse afterwards."""
    cache = frames_cache_dir(
        clip,
        args.target_fps,
        args.target_height,
        args.stage01_max_noop_run,
        root=args.cache_root,
    )
    if frames_cache_is_complete(cache, args.target_fps, args.target_height, args.stage01_max_noop_run):
        print(f"frames cache hit: {cache}")
        return cache
    run_step("stage_00_manifest.py", [
        "--raw-root", str(args.raw_root),
        "--output-dir", str(cache / "stage_00_manifest"),
        "--version", clip["version"],
        "--user-id", clip["user_id"],
        "--recording-id", clip["recording_id"],
        "--segment-start", str(clip["segment_start"]),
        "--segment-end", str(clip["segment_end"]),
    ])
    run_step("stage_01_frames_actions.py", [
        "--manifest", str(cache / "stage_00_manifest" / "manifest.jsonl"),
        "--output-dir", str(cache / "stage_01_frames_actions"),
        "--target-fps", str(args.target_fps),
        "--target-height", str(args.target_height),
        "--max-noop-run", str(args.stage01_max_noop_run),
        "--ffmpeg-bin", str(args.ffmpeg_bin or ""),
    ])
    (cache / ".complete").write_text(dt.datetime.now(dt.timezone.utc).isoformat() + "\n")
    return cache


def run_clip(args: argparse.Namespace, clip_id: str, clip: dict[str, Any], run_dir: Path) -> None:
    # frames/all extract; annotate/assemble require a pre-built cache and skip
    # clips not yet extracted, so the GPU phase never falls back to ffmpeg.
    if args.stages in ("annotate", "assemble"):
        cache = frames_cache_dir(
            clip,
            args.target_fps,
            args.target_height,
            args.stage01_max_noop_run,
            root=args.cache_root,
        )
        if not frames_cache_is_complete(cache, args.target_fps, args.target_height, args.stage01_max_noop_run):
            print(f"  skip {clip_id}: frames not extracted yet", flush=True)
            return
    else:
        cache = ensure_frames_cache(clip, args)
    if args.stages == "all" and not progress_is_terminal(run_dir / "stage_01_frames_actions", clip_id):
        commit_frame_cache_clip(run_dir, clip_id, cache)
    if args.stages == "frames":
        commit_frame_cache_clip(run_dir, clip_id, cache)
        return
    manifest = cache / "stage_00_manifest" / "manifest.jsonl"
    frame_records = cache / "stage_01_frames_actions" / "frame_records.jsonl"
    # A clip can extract to zero kept frames (video all black/idle while the
    # keylog had activity, so the idle pre-gate kept it). Nothing to annotate.
    if not frame_records.exists() or frame_records.stat().st_size == 0:
        print(f"  skip {clip_id}: no usable frames (all black/idle)", flush=True)
        if args.stages in ("all", "annotate"):
            write_progress(run_dir / "stage_02_segment", clip_id, "skipped", reason="no_usable_frames")
            rewrite_stage02_summary(run_dir / "stage_02_segment")
        if args.stages in ("all", "assemble"):
            write_progress(run_dir / "stage_03_assemble", clip_id, "skipped", reason="no_usable_frames")
            rewrite_stage03_summary(run_dir / "stage_03_assemble")
        return
    annotation_root = args.annotation_run_dir or run_dir
    stage02 = annotation_root / "stage_02_segment"
    stage03 = run_dir / "stage_03_assemble"
    tmp_root = run_dir / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    if args.stages in ("all", "annotate"):
        if progress_is_terminal(run_dir / "stage_02_segment", clip_id):
            print(f"  skip {clip_id}: stage 02 progress complete", flush=True)
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix=f"{clip_id}_stage02_", dir=tmp_root))
            try:
                run_step("stage_02_vlm_trajectories.py", [
                    "--frame-records", str(frame_records),
                    "--manifest", str(manifest),
                    "--output-dir", str(temp_dir),
                    "--model", args.vlm_model,
                    "--segment-window-s", str(args.segment_window_s),
                    "--segment-overlap-s", str(args.segment_overlap_s),
                    "--vlm-frame-height", str(args.vlm_frame_height),
                    "--max-concurrency", str(args.max_concurrency),
                ])
                commit_stage02_clip(run_dir, clip_id, temp_dir)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
    trajectory = stage02_trajectory_for_clip(stage02, clip_id)
    if args.stages == "assemble" and trajectory is None:
        print(f"  skip {clip_id}: not annotated yet", flush=True)
        return
    if args.stages in ("all", "assemble"):
        if progress_is_terminal(stage03, clip_id):
            print(f"  skip {clip_id}: stage 03 progress complete", flush=True)
            return
        if trajectory is None:
            write_progress(stage03, clip_id, "skipped", reason="not_annotated")
            rewrite_stage03_summary(stage03)
            return
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{clip_id}_stage03_", dir=tmp_root))
        try:
            trajectory_path = temp_dir / "trajectories_raw.json"
            write_json_file(trajectory_path, trajectory)
            run_step("stage_03_assemble_trajectories.py", [
                "--frame-records", str(frame_records),
                "--trajectories", str(trajectory_path),
                "--output-dir", str(temp_dir),
            ])
            commit_stage03_clip(run_dir, clip_id, temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_stage04_canonical(args: argparse.Namespace, run_dir: Path) -> Path:
    output_dir = args.canonical_output_dir or (run_dir / "stage_04_canonical_sft")
    cmd = [
        "--run-dir", str(run_dir),
        "--output-dir", str(output_dir),
        "--split-group", args.canonical_split_group,
        "--val-frac", str(args.canonical_val_frac),
        "--seed", str(args.canonical_seed),
        "--image-path-mode", args.canonical_image_path_mode,
        "--terminal-mode", args.canonical_terminal_mode,
        "--overwrite",
    ]
    if args.canonical_system_prompt_text:
        cmd.extend(["--system-prompt-text", args.canonical_system_prompt_text])
    if args.canonical_system_prompt_file:
        cmd.extend(["--system-prompt-file", str(args.canonical_system_prompt_file)])
    if args.canonical_terminal_token:
        cmd.extend(["--terminal-token", args.canonical_terminal_token])
    run_step("stage_04_build_canonical_sft.py", cmd)
    return output_dir


def run_stage05_length_buckets(args: argparse.Namespace, canonical_dir: Path, run_dir: Path) -> None:
    # Local iteration only. The labctl/omegalax path owns real token counts and
    # training buckets.
    run_step("stage_05_length_buckets.py", [
        "--samples", str(canonical_dir / "chat.jsonl"),
        "--output-dir", str(run_dir / "stage_05_length_buckets"),
        "--tokenizer", args.trainee_model,
        "--processor", args.trainee_model,
    ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=None, help="outputs/runs/<run-name>/ (default: UTC timestamp).")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Explicit run artifact root. Defaults to processed/runs/<run-name>.")
    p.add_argument("--cache-root", type=Path, default=None,
                   help="Explicit stage-00/01 frame cache root. Defaults to processed/cache/frames.")
    p.add_argument("--annotation-run-dir", type=Path, default=None,
                   help="Read stage-02 annotations from this run root when --stages assemble.")
    p.add_argument("--clips", nargs="+", default=["all"], help="Clip ids from the clips file, or 'all'.")
    p.add_argument("--clips-file", type=Path, default=CLIPS_PATH,
                   help="JSON of clips to run (default clips.json; use clips_dataset.json for the full set).")
    p.add_argument("--shard", default=None,
                   help="Process only shard i of N as 'i/N' (clips split deterministically); for array jobs.")
    p.add_argument("--resume", action="store_true",
                   help="Reuse an existing run dir and skip clips whose --stages output already exists.")
    p.add_argument("--stages", choices=["all", "frames", "annotate", "assemble"], default="all",
                   help="Which stages to run: frames=00+01 (CPU), annotate=02 (GPU), "
                        "assemble=03+04 (CPU), all=everything. Lets CPU phases run as 0-GPU "
                        "jobs so the GPU phase never idles on ffmpeg/tokenization.")
    p.add_argument("--max-concurrency", type=int, default=int(os.environ.get("JUERGEN_ANNOTATION_MAX_CONCURRENCY", "8")),
                   help="In-flight VLM requests per pass (passed to stage 02).")
    p.add_argument("--raw-root", type=Path, default=config.RAW_DATA_ROOT)
    p.add_argument("--target-fps", type=int, default=config.DEFAULT_TARGET_FPS)
    p.add_argument("--target-height", type=int, default=config.DEFAULT_TARGET_HEIGHT)
    p.add_argument("--vlm-frame-height", type=int, default=config.DEFAULT_VLM_FRAME_HEIGHT)
    p.add_argument("--ffmpeg-bin", default=config.ffmpeg_bin())
    p.add_argument("--stage01-max-noop-run", type=int, default=config.DEFAULT_STAGE01_MAX_NOOP_RUN)
    p.add_argument("--vlm-model", default=config.vlm_model())
    p.add_argument("--trainee-model", default=config.DEFAULT_TRAINEE_MODEL)
    p.add_argument("--segment-window-s", type=float, default=config.DEFAULT_SEGMENT_WINDOW_S)
    p.add_argument("--segment-overlap-s", type=float, default=config.DEFAULT_SEGMENT_OVERLAP_S)
    p.add_argument("--canonical-output-dir", type=Path, default=None)
    p.add_argument("--canonical-split-group", choices=["recording_id", "clip_id"], default="recording_id")
    p.add_argument("--canonical-val-frac", type=float, default=0.1)
    p.add_argument("--canonical-seed", type=int, default=0)
    p.add_argument("--canonical-image-path-mode", choices=["absolute", "preserve"], default="absolute")
    p.add_argument("--canonical-system-prompt-text", default=None)
    p.add_argument("--canonical-system-prompt-file", type=Path, default=None)
    p.add_argument("--canonical-terminal-token", default=None)
    p.add_argument("--canonical-terminal-mode", choices=["none", "replace_final_assistant", "append_assistant"], default="none")
    p.add_argument(
        "--stage05-length-buckets",
        action="store_true",
        help="After stage 04, run the old local length-bucket distribution inspector.",
    )
    p.add_argument(
        "--skip-canonical",
        action="store_true",
        help="For --stages assemble, stop after stage 03 so labctl can run stage 04 separately.",
    )
    return p.parse_args()


def select_shard(clip_ids: list[str], shard: str | None) -> list[str]:
    """'i/N' -> the deterministic i-th of N contiguous shards of sorted clip ids."""
    if not shard:
        return clip_ids
    i, n = (int(x) for x in shard.split("/"))
    ordered = sorted(clip_ids)
    return [c for idx, c in enumerate(ordered) if idx % n == i]


def main() -> None:
    args = parse_args()
    clips = load_clips(args.clips_file)
    if args.clips == ["all"]:
        selected = dict(clips)
    else:
        unknown = [c for c in args.clips if c not in clips]
        if unknown:
            raise SystemExit(f"Unknown clip ids {unknown}; known: {sorted(clips)}")
        selected = {c: clips[c] for c in args.clips}
    requested = dict(selected)
    if args.shard:
        shard_ids = set(select_shard(list(selected), args.shard))
        selected = {c: v for c, v in selected.items() if c in shard_ids}
        print(f"shard {args.shard}: {len(selected)} clips")

    run_name = args.run_name or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else OUTPUTS_DIR / "runs" / run_name
    if (run_dir / "run_config.json").exists() and not args.resume:
        raise SystemExit(f"run {run_name!r} already exists at {run_dir}; use --resume or a new --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps({
        "run_name": run_name,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": sys.argv[1:],
        "clips": sorted(requested),
        "selected_clips": sorted(selected),
        "shard": args.shard,
        "clips_file": str(args.clips_file),
        "run_dir": str(run_dir),
        "cache_root": str(args.cache_root) if args.cache_root else str(OUTPUTS_DIR / "cache" / "frames"),
        "annotation_run_dir": str(args.annotation_run_dir) if args.annotation_run_dir else None,
        "vlm_model": args.vlm_model,
        "vlm_base_url": config.vlm_base_url(),
        "trainee_model": args.trainee_model,
        "segment_window_s": args.segment_window_s,
        "segment_overlap_s": args.segment_overlap_s,
        "enable_thinking": config.DEFAULT_ENABLE_THINKING,
        "target_fps": args.target_fps,
        "target_height": args.target_height,
        "vlm_frame_height": args.vlm_frame_height,
    }, indent=2) + "\n")

    failures = 0
    for clip_id, clip in selected.items():
        if args.resume and phase_done(args, clip, clip_id, run_dir):
            print(f"\n=== clip {clip_id} (skip: {args.stages} done) ===", flush=True)
            continue
        print(f"\n=== clip {clip_id} [{args.stages}] ===", flush=True)
        try:
            run_clip(args, clip_id, clip, run_dir)
        except Exception as exc:  # noqa: BLE001 - isolate a bad clip, keep the run going
            failures += 1
            print(f"!!! clip {clip_id} failed: {type(exc).__name__}: {exc}", flush=True)
            with (run_dir / "failed_clips.jsonl").open("a") as fh:
                fh.write(json.dumps({"clip_id": clip_id, "error": f"{type(exc).__name__}: {exc}"}) + "\n")

    if args.stages in ("all", "assemble") and not args.skip_canonical:
        try:
            canonical_dir = run_stage04_canonical(args, run_dir)
            if args.stage05_length_buckets:
                run_stage05_length_buckets(args, canonical_dir, run_dir)
        except Exception as exc:  # noqa: BLE001 - report run-level export failure clearly.
            failures += 1
            print(f"!!! run-level stage failed: {type(exc).__name__}: {exc}", flush=True)
    print(f"\nDone. Run outputs under {run_dir}" + (f" ({failures} clip(s) failed)" if failures else ""))


if __name__ == "__main__":
    main()
