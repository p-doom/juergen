#!/usr/bin/env python3
"""Build an SFT dataset from a dataset_runs/<run> annotation run.

Walks <run>/<model>/clips/<uid>, groups window-units back into their PARENT
segment, and for each parent feeds the parent's FULL frame_records (from
<run>/_frames/clips/<parent>/stage_01) plus the concatenated goal trajectories
(from each unit's stage_02/trajectories_raw.json) through stage 03's
``assemble_samples`` -> SFT samples, then stage 04 (canonical artifact). Stage 05
(length buckets) is optional (--buckets) since it needs the trainee tokenizer.

Each sample carries full provenance so it traces back clip -> user -> recording:
recording_id, segment_id (== clip_id), parent_segment_id, user_id, version,
video_path, plus start/end_frame_idx and the source goal (anchor/grounding).

  PYTHONPATH=. python3 -m annotation_pipeline.build_sft \
      --run-dir annotation_pipeline/dataset_runs/qc30v2 \
      --out annotation_pipeline/dataset_runs/qc30v2/sft
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from annotation_pipeline.common import ensure_dir, read_jsonl, write_json, write_jsonl
from annotation_pipeline.stage_03_assemble_trajectories import assemble_samples
from annotation_pipeline.stage_04_build_canonical_sft import build_canonical_sft

PIPELINE_DIR = Path(__file__).resolve().parent
DATA_PIPELINE_DIR = PIPELINE_DIR.parent

# Keys copied from the stage-00 manifest row onto every sample for traceback.
PROVENANCE_KEYS = ("user_id", "version", "recording_id", "video_path")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="A dataset_runs/<run> dir (contains <model>/clips and _frames).")
    p.add_argument("--frames-root", type=Path, default=None,
                   help="Shared stage-01 _frames/ dir. Default <run-dir>/_frames. Point at a "
                        "separate frames-phase output when stage 01 ran as its own labctl stage.")
    p.add_argument("--out", type=Path, required=True, help="Output dir for the SFT artifact.")
    p.add_argument("--min-frames", type=int, default=1)
    p.add_argument("--include-variants", action="store_true",
                   help="Emit one sample per instruction paraphrase (3x) instead of main only.")
    p.add_argument("--split-group", default="recording_id", choices=("recording_id", "clip_id"))
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--buckets", action="store_true",
                   help="Also run stage 05 length-bucketing (needs trainee tokenizer; compute node).")
    p.add_argument("--assemble-only", action="store_true",
                   help="Stop after writing stage_03_assemble/ (the aggregated trajectories.jsonl); "
                        "skip stage 04 canonicalization. Lets a downstream step run stage 04 "
                        "independently (e.g. with its own system prompt / terminal policy).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    frames_root = args.frames_root.resolve() if args.frames_root else run_dir / "_frames"
    out = ensure_dir(args.out)

    # Gather every annotated unit across all model dirs, grouped by parent segment.
    # Model dirs are those with a clips/ subdir; skip _frames and any output dirs.
    model_dirs = [m for m in run_dir.iterdir()
                  if m.is_dir() and m.name != "_frames" and (m / "clips").is_dir()] if run_dir.is_dir() else []
    units = [c for m in model_dirs for c in (m / "clips").iterdir() if c.is_dir()]
    by_parent: dict[str, list[tuple[int, list[dict[str, Any]]]]] = defaultdict(list)
    for u in units:
        res = u / "stage_02" / "stage02_result.json"
        traj = u / "stage_02" / "trajectories_raw.json"
        if not (res.exists() and traj.exists()):
            continue
        sres = json.loads(res.read_text())
        parent = str(sres.get("parent_segment_id") or sres.get("segment_id") or u.name)
        wi = int(sres.get("window_index", 0))
        trajs = json.loads(traj.read_text()).get("trajectories", [])
        by_parent[parent].append((wi, trajs))

    all_samples: list[dict[str, Any]] = []
    all_rejected: list[dict[str, Any]] = []
    n_parents = 0
    for parent, parts in sorted(by_parent.items()):
        fr_path = frames_root / "clips" / parent / "stage_01" / "frame_records.jsonl"
        frame_records = read_jsonl(fr_path) if fr_path.exists() else []
        if not frame_records:
            print(f"  skip {parent}: no frame_records", file=sys.stderr)
            continue
        # Concatenate the parent's goals in window order (windows are disjoint by
        # construction: w0 ends at its owned cut, w1 starts after).
        trajectories = [t for _wi, trajs in sorted(parts) for t in trajs]
        samples, rejected = assemble_samples(
            frame_records, trajectories,
            min_frames=args.min_frames, include_variants=args.include_variants)

        man = frames_root / "clips" / parent / "stage_00" / "manifest.jsonl"
        row = (read_jsonl(man) or [{}])[0] if man.exists() else {}
        prov = {k: row.get(k) for k in PROVENANCE_KEYS}
        prov["parent_segment_id"] = parent
        for s in samples:
            for k, v in prov.items():
                s.setdefault(k, v)
        all_samples.extend(samples)
        all_rejected.extend(rejected)
        n_parents += 1

    # Aggregated stage-03 output where stage 04 expects it.
    s3dir = ensure_dir(out / "stage_03_assemble")
    write_jsonl(s3dir / "trajectories.jsonl", all_samples)
    write_jsonl(s3dir / "rejected_trajectories.jsonl", all_rejected)
    write_json(s3dir / "assemble_summary.json", {
        "source_run_dir": str(run_dir), "n_parents": n_parents,
        "n_samples": len(all_samples), "n_rejected": len(all_rejected),
        "reject_reasons": {r: sum(1 for x in all_rejected if x["reason"] == r)
                           for r in sorted({x["reason"] for x in all_rejected})},
    })
    print(f"[build_sft] {n_parents} parent segments -> {len(all_samples)} samples "
          f"({len(all_rejected)} rejected). stage_03 -> {s3dir}", flush=True)

    if args.assemble_only:
        print("[build_sft] --assemble-only: stopping after stage_03_assemble.", flush=True)
        return

    # Stage 04: canonical, portable SFT artifact (ar:// frame URIs pass through).
    canonical = out / "canonical"
    m = build_canonical_sft(
        run_dir=out, output_dir=canonical, split_group=args.split_group,
        val_frac=args.val_frac, seed=args.seed, image_path_mode="absolute", overwrite=True)
    print(f"[build_sft] stage_04 -> {canonical}: {m['n_samples']} samples "
          f"{m['counts_by_split']} ({m['n_rejected']} rejected)", flush=True)

    # Per-split chat.jsonl so <out> is a drop-in source_path for omegalax's
    # stage_c (it reads <source>/<split>/chat.jsonl and compiles each split).
    rows = read_jsonl(canonical / "chat.jsonl")
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_split[str(r.get("split") or "train")].append(r)
    for split, srows in sorted(by_split.items()):
        write_jsonl(ensure_dir(out / split) / "chat.jsonl", srows)
    print(f"[build_sft] per-split chat.jsonl -> {out}/<split>/chat.jsonl "
          f"{ {s: len(v) for s, v in sorted(by_split.items())} }", flush=True)

    # Stage 05 (optional): exact length buckets via the trainee tokenizer.
    if args.buckets:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(DATA_PIPELINE_DIR) + (os.pathsep + env.get("PYTHONPATH", ""))
        subprocess.run([sys.executable, "-m", "annotation_pipeline.stage_05_length_buckets",
                        "--samples", str(canonical / "chat.jsonl"),
                        "--output-dir", str(out / "buckets")],
                       check=True, cwd=str(DATA_PIPELINE_DIR), env=env)
        print(f"[build_sft] stage_05 -> {out / 'buckets'}", flush=True)


if __name__ == "__main__":
    main()
