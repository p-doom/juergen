#!/usr/bin/env python3
"""Run the v3 Crowd-Cast trajectory -> SFT pipeline over registered clips.

Single validated configuration: Qwen3.6-27B (BF16, sglang) annotates 90s
windows with thinking off; a verification pass (stage 02 pass C) is the quality
gate; samples are tokenized with the trainee model's real processor.

Stages:
  00 manifest          MP4 + keylog pairs for a clip's segment slice
  01 frames+actions    2fps 720p frames + per-frame action strings (cached)
  02 segment+name+verify   pass A boundaries -> pass B instructions -> pass C
                           grounded verification (writes a `verified` flag)
  03 assemble          verified trajectories -> SFT chat messages
  04 buckets           exact Qwen3-VL-2B token counts -> length-bucketed JSONL

Stages 00+01 are cached per (clip, fps, height) under outputs/cache/frames/;
each run's annotations/samples live under outputs/runs/<run-name>/.

Requires a running sglang server (slurm/run_pipeline.sbatch starts one and sets
V3_VLM_BASE_URL). Example:
    python run_pipeline.py --run-name v1 --clips bbbf_s0000-0003
    python run_pipeline.py --run-name v1 --clips all
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import config


PIPELINE_DIR = Path(__file__).resolve().parent
# Frame cache + runs live under the dataset's processed/ dir, not the code tree.
OUTPUTS_DIR = config.PROCESSED_ROOT
CLIPS_PATH = PIPELINE_DIR / "clips.json"


def load_clips(path: Path = CLIPS_PATH) -> dict[str, dict[str, Any]]:
    return json.loads(Path(path).read_text())["clips"]


def phase_done(args: argparse.Namespace, clip: dict[str, Any], clip_id: str, run_dir: Path) -> bool:
    """Whether the requested --stages output already exists (for --resume)."""
    cd = run_dir / clip_id
    if args.stages == "frames":
        cache = frames_cache_dir(clip, args.target_fps, args.target_height, args.stage01_max_noop_run)
        return frames_cache_is_complete(cache, args.target_fps, args.target_height, args.stage01_max_noop_run)
    if args.stages == "annotate":
        return (cd / "stage_02_segment" / "trajectories_raw.json").exists()
    # "assemble" and "all" both finish at the stage-04 bucket summary.
    return (cd / "stage_04_sft_samples" / "bucket_summary.json").exists()


def run_step(script: str, args: list[str], python: str | None = None) -> None:
    cmd = [python or sys.executable, str(PIPELINE_DIR / script), *args]
    print("\n$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed with exit code {result.returncode}")


def noop_cache_label(max_noop_run: int) -> str:
    return "noopall" if max_noop_run < 0 else f"noop{max_noop_run}"


def frames_cache_dir(clip: dict[str, Any], fps: int, height: int, max_noop: int) -> Path:
    rec8 = clip["recording_id"][:8]
    return (
        OUTPUTS_DIR / "cache" / "frames"
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
    cache = frames_cache_dir(clip, args.target_fps, args.target_height, args.stage01_max_noop_run)
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
        cache = frames_cache_dir(clip, args.target_fps, args.target_height, args.stage01_max_noop_run)
        if not frames_cache_is_complete(cache, args.target_fps, args.target_height, args.stage01_max_noop_run):
            print(f"  skip {clip_id}: frames not extracted yet", flush=True)
            return
    else:
        cache = ensure_frames_cache(clip, args)
    if args.stages == "frames":
        return
    manifest = cache / "stage_00_manifest" / "manifest.jsonl"
    frame_records = cache / "stage_01_frames_actions" / "frame_records.jsonl"
    # A clip can extract to zero kept frames (video all black/idle while the
    # keylog had activity, so the idle pre-gate kept it). Nothing to annotate.
    if not frame_records.exists() or frame_records.stat().st_size == 0:
        print(f"  skip {clip_id}: no usable frames (all black/idle)", flush=True)
        return
    clip_dir = run_dir / clip_id
    stage02 = clip_dir / "stage_02_segment"
    stage03 = clip_dir / "stage_03_assemble"
    stage04 = clip_dir / "stage_04_sft_samples"

    if args.stages in ("all", "annotate"):
        run_step("stage_02_vlm_trajectories.py", [
            "--frame-records", str(frame_records),
            "--manifest", str(manifest),
            "--output-dir", str(stage02),
            "--model", args.vlm_model,
            "--segment-window-s", str(args.segment_window_s),
            "--segment-overlap-s", str(args.segment_overlap_s),
            "--vlm-frame-height", str(args.vlm_frame_height),
            "--max-concurrency", str(args.max_concurrency),
        ])
    if args.stages == "assemble" and not (stage02 / "trajectories_raw.json").exists():
        print(f"  skip {clip_id}: not annotated yet", flush=True)
        return
    if args.stages in ("all", "assemble"):
        run_step("stage_03_assemble_trajectories.py", [
            "--frame-records", str(frame_records),
            "--trajectories", str(stage02 / "trajectories_raw.json"),
            "--output-dir", str(stage03),
        ])
        # Exact token buckets via the vendored qwen3_encoding + trainee processor,
        # in this same venv (transformers 5.2, torch-free). No omegalax.
        run_step("stage_04_length_buckets.py", [
            "--samples", str(stage03 / "trajectories.jsonl"),
            "--output-dir", str(stage04),
            "--tokenizer", args.trainee_model,
            "--processor", args.trainee_model,
        ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=None, help="outputs/runs/<run-name>/ (default: UTC timestamp).")
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
    p.add_argument("--max-concurrency", type=int, default=int(os.environ.get("V3_MAX_CONCURRENCY", "8")),
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
    if args.shard:
        shard_ids = set(select_shard(list(selected), args.shard))
        selected = {c: v for c, v in selected.items() if c in shard_ids}
        print(f"shard {args.shard}: {len(selected)} clips")

    run_name = args.run_name or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / "runs" / run_name
    if (run_dir / "run_config.json").exists() and not args.resume:
        raise SystemExit(f"run {run_name!r} already exists at {run_dir}; use --resume or a new --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps({
        "run_name": run_name,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": sys.argv[1:],
        "clips": sorted(selected),
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
    print(f"\nDone. Run outputs under {run_dir}" + (f" ({failures} clip(s) failed)" if failures else ""))


if __name__ == "__main__":
    main()
