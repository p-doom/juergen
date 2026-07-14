#!/usr/bin/env python3
"""Re-run Stage 03 extraction and Stage 04 refinement for an annotation run.

The Stage-02 observation view and cached Stage-03 describe response are reused;
Stages 00-02 are never regenerated.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
DATA_PIPELINE_DIR = PIPELINE_DIR.parent


def _run_module(module: str, args: list[str], *, model: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(DATA_PIPELINE_DIR) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    if model:
        env["LABELER_MODEL"] = model
    return subprocess.run(
        [sys.executable, "-m", f"annotation_pipeline.{module}", *args],
        env=env,
        cwd=str(DATA_PIPELINE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )


def reextract_unit(unit_dir: Path, vlm_height: int) -> tuple[str, str]:
    unit_id = unit_dir.name
    observations = unit_dir / "stage_02_annotation_view" / "observations.jsonl"
    annotation_dir = unit_dir / "stage_03_annotation"
    annotation_path = annotation_dir / "annotation.json"
    boundary_dir = unit_dir / "stage_04_boundaries"
    boundary_manifest_path = boundary_dir / "manifest.json"
    required = (observations, annotation_path, boundary_manifest_path)
    if any(not path.exists() for path in required):
        return unit_id, "skip (incomplete Stage 02-04 unit)"

    annotation = json.loads(annotation_path.read_text())
    boundary_manifest = json.loads(boundary_manifest_path.read_text())
    model = str(annotation["model"])
    parent = str(annotation["parent_segment_id"])
    stage03_args = [
        "--observations",
        str(observations),
        "--output-dir",
        str(annotation_dir),
        "--vlm-frame-height",
        str(vlm_height),
        "--parent-segment-id",
        parent,
        "--window-index",
        str(annotation["window_index"]),
        "--n-windows",
        str(annotation["n_windows"]),
        "--tail-buffer",
        str(annotation["tail_buffer"]),
        "--model",
        model,
        "--refresh",
        "extract_from_prose",
    ]
    result = _run_module("stage_03_annotate", stage03_args, model=model)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace")[-300:]
        return unit_id, f"FAIL Stage 03 rc={result.returncode}: {detail}"

    result = _run_module(
        "stage_04_refine_boundaries",
        [
            "--annotation-dir",
            str(annotation_dir),
            "--observations",
            str(observations),
            "--output-dir",
            str(boundary_dir),
            "--policy",
            str(boundary_manifest["policy"]),
        ],
        model=None,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace")[-300:]
        return unit_id, f"FAIL Stage 04 rc={result.returncode}: {detail}"

    summary = json.loads((annotation_dir / "manifest.json").read_text())
    return unit_id, f"ok [{model}] goals={summary['n_goals_prose']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--vlm-frame-height", type=int, default=720)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    model_dirs = [
        path
        for path in run_dir.iterdir()
        if path.is_dir() and path.name != "_modalities" and (path / "clips").is_dir()
    ]
    units = sorted(
        unit
        for model_dir in model_dirs
        for unit in (model_dir / "clips").iterdir()
        if unit.is_dir()
    )
    print(f"[reextract] {len(units)} units under {run_dir}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {
            executor.submit(reextract_unit, unit, args.vlm_frame_height): unit for unit in units
        }
        for done, future in enumerate(as_completed(futures), start=1):
            unit_id, message = future.result()
            print(f"  [{done}/{len(units)}] {unit_id}: {message}", flush=True)


if __name__ == "__main__":
    main()
