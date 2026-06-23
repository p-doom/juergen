#!/usr/bin/env python3
"""Iterate the redesigned annotation pipeline on the curated clip set.

One command: for each clip in ``iteration_clips.json`` it synthesizes a
single-segment stage-00 manifest from the dataset manifest, runs stage 01
(frame/action records) and the new stage 02 (hindsight annotation), then builds
the HTML review report and runs the LLM judge.

Everything is cached/resumable: a clip whose ``frame_records.jsonl`` /
``trajectories_raw.json`` already exist is skipped (``--force`` re-runs stage 02,
``--force-frames`` re-runs stage 01). Stage 02 also caches every labeler
response, so re-running after a prompt edit only re-spends the changed calls.

    python -m annotation_pipeline.run_iteration --run-name v1
    python -m annotation_pipeline.run_iteration --run-name v1 --clips codex_6878cdfa_s0 --force
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from annotation_pipeline.common import ensure_dir, write_jsonl

PIPELINE_DIR = Path(__file__).resolve().parent
DATA_PIPELINE_DIR = PIPELINE_DIR.parent


def load_clips(clips_path: Path) -> tuple[dict[str, Any], Path]:
    spec = json.loads(clips_path.read_text())
    dataset_root = Path(spec["dataset_root"])
    manifest_path = dataset_root / spec.get("manifest", "manifest.selected_segments.jsonl")
    return spec, manifest_path


def index_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx[row["segment_id"]] = row
    return idx


def run_module(module: str, args: list[str]) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(DATA_PIPELINE_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, "-m", f"annotation_pipeline.{module}", *args]
    print("  $", " ".join(str(c) for c in cmd[2:]))
    subprocess.run(cmd, check=True, env=env, cwd=str(DATA_PIPELINE_DIR))


def process_clip(clip_key: str, clip: dict[str, Any], row: dict[str, Any], run_dir: Path,
                 args: argparse.Namespace) -> dict[str, Any]:
    clip_dir = ensure_dir(run_dir / "clips" / clip_key)
    stage00 = ensure_dir(clip_dir / "stage_00")
    stage01 = clip_dir / "stage_01"
    stage02 = clip_dir / "stage_02"

    manifest_path = stage00 / "manifest.jsonl"
    write_jsonl(manifest_path, [row])

    frame_records = stage01 / "frame_records.jsonl"
    if args.force_frames or not frame_records.exists():
        run_module("stage_01_frames_actions", [
            "--manifest", str(manifest_path), "--output-dir", str(stage01),
        ])
    else:
        print("  [skip stage 01: frame_records.jsonl exists]")

    traj = stage02 / "trajectories_raw.json"
    if args.force or args.refresh or not traj.exists():
        stage02_args = [
            "--frame-records", str(frame_records),
            "--keylog", str(row["keylog_path"]),
            "--manifest", str(manifest_path),
            "--output-dir", str(stage02),
        ]
        if args.reasoning_effort:
            stage02_args += ["--reasoning-effort", args.reasoning_effort]
        if args.refresh:
            stage02_args += ["--refresh", args.refresh]
        run_module("stage_02_annotate", stage02_args)
    else:
        print("  [skip stage 02: trajectories_raw.json exists]")

    summary = json.loads((stage02 / "stage02_summary.json").read_text())
    return {"clip_key": clip_key, "segment_id": row["segment_id"], **clip, **summary}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default="v1")
    p.add_argument("--clips-file", type=Path, default=PIPELINE_DIR / "iteration_clips.json")
    p.add_argument("--clips", nargs="*", default=None, help="subset of clip keys (default: all)")
    p.add_argument("--out-root", type=Path, default=PIPELINE_DIR / "iteration_runs")
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--refresh", default=None,
                   help="Invalidate+re-run only these stage-02 steps on every clip "
                        "(e.g. 'verify'); reuses cached perceive/segment/label.")
    p.add_argument("--force", action="store_true", help="re-run stage 02 even if cached")
    p.add_argument("--force-frames", action="store_true", help="re-run stage 01 frame extraction")
    p.add_argument("--no-review", action="store_true")
    p.add_argument("--no-judge", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec, manifest_path = load_clips(args.clips_file)
    idx = index_manifest(manifest_path)
    clips = spec["clips"]
    keys = args.clips or list(clips.keys())

    run_dir = ensure_dir(args.out_root / args.run_name)
    rows = []
    for key in keys:
        clip = clips[key]
        seg = clip["segment_id"]
        row = idx.get(seg)
        if row is None:
            print(f"[{key}] segment {seg} not in dataset manifest; skipping")
            continue
        print(f"\n=== {key}  ({clip.get('app','?')}, {clip.get('duration_s','?')}s) ===")
        try:
            rows.append(process_clip(key, clip, row, run_dir, args))
        except subprocess.CalledProcessError as exc:
            print(f"[{key}] FAILED: {exc}")
            rows.append({"clip_key": key, "segment_id": seg, "error": str(exc)})

    (run_dir / "run_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nWrote run summary: {run_dir/'run_summary.json'}")

    if not args.no_review:
        run_module("review_report", ["--run-dir", str(run_dir)])
    if not args.no_judge:
        judge_args = ["--run-dir", str(run_dir)]
        if args.refresh or args.force:
            judge_args.append("--no-cache")  # sample set changed; re-judge fresh
        run_module("judge", judge_args)
    print(f"\nDone. Open {run_dir/'review.html'}")


if __name__ == "__main__":
    main()
