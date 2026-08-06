#!/usr/bin/env python3
"""Re-run ONLY the extract pass over an existing dataset_run, in place.

SUPERSEDED, NOT PORTED — same reason as ``build_sft.py``: it subprocess-invokes
``annotation_pipeline.stage_02_annotate`` over the legacy
``dataset_runs/<run>/<model>/clips/<uid>`` tree, so it cannot outlive the engine
it drives. The capability itself already exists in the current generation:
``pipeline/annotation/methods/describe_extract/annotator.py`` caches the two
passes independently (``<calls>/<model>/<unit>/describe_prose.txt`` and
``extract_from_prose.txt``), so an extract-prompt iteration is "delete the
``extract_from_prose.txt`` cache files, re-run ``stage_annotate.py --force``" —
describe is reused, no tokens re-spent. The one affordance not carried over is
the ``--refresh <call-name>`` flag that did that invalidation for you.

Used to validate a change to the EXTRACT prompt without re-paying for describe:
for every clip dir already under <run>/<model>/clips/<uid>, invalidate just the
extract cache (`--refresh extract_from_prose`) and re-run stage_02 with
`--variants prose`. The cached describe_prose response is reused (its prompt is
unchanged), each clip stays on its original model, and window provenance is
preserved from the clip's window.json. Frames come from the clip's own
stage_01/frame_records.jsonl, so no ffmpeg/stage-01 work happens.

  PYTHONPATH=. python3 -m annotation_pipeline.reextract_run \
      --run-dir annotation_pipeline/dataset_runs/qc30 --concurrency 8
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


def reextract_clip(clip: Path, frames_root: Path, vlm_height: int) -> tuple[str, str]:
    uid = clip.name
    model = clip.parent.parent.name  # <run>/<model>/clips/<uid>
    fr = clip / "stage_01" / "frame_records.jsonl"
    stage02 = clip / "stage_02"
    res = stage02 / "stage02_result.json"
    if not fr.exists() or not res.exists():
        return uid, "skip (no frames/result)"
    d = json.loads(res.read_text())
    parent = d.get("parent_segment_id") or d.get("segment_id") or uid
    wi, nw = int(d.get("window_index", 0)), int(d.get("n_windows", 1))
    manifest = frames_root / "clips" / parent / "stage_00" / "manifest.jsonl"
    wjson = clip / "window.json"
    tail_buf = int(json.loads(wjson.read_text()).get("tail_buffer", 0)) if wjson.exists() else 0

    args = ["--frame-records", str(fr), "--manifest", str(manifest),
            "--output-dir", str(stage02),
            "--vlm-frame-height", str(vlm_height),
            "--parent-segment-id", parent, "--window-index", str(wi), "--n-windows", str(nw),
            "--tail-buffer", str(tail_buf), "--model", model]
    # REEX_REFRESH="" re-applies only the deterministic post-processing
    # (clean_goals + snap_goal_starts) on the CACHED model response — no tokens.
    refresh = os.environ.get("REEX_REFRESH", "extract_from_prose")
    if refresh:
        args += ["--refresh", refresh]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(DATA_PIPELINE_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["LABELER_MODEL"] = model
    cmd = [sys.executable, "-m", "annotation_pipeline.stage_02_annotate", *args]
    p = subprocess.run(cmd, env=env, cwd=str(DATA_PIPELINE_DIR),
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if p.returncode != 0:
        return uid, f"FAIL rc={p.returncode}: {p.stderr.decode('utf-8','replace')[-300:]}"
    summ = json.loads((stage02 / "stage02_summary.json").read_text())
    return uid, f"ok [{model}] goals={summ.get('n_goals_prose')}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--vlm-frame-height", type=int, default=720)
    a = ap.parse_args()
    run_dir = a.run_dir.resolve()
    frames_root = run_dir / "_frames"
    # Model dirs are those with a clips/ subdir; skip _frames and any output dirs.
    clips = sorted(c for m in run_dir.iterdir()
                   if m.is_dir() and m.name != "_frames" and (m / "clips").is_dir()
                   for c in (m / "clips").iterdir() if c.is_dir()) if run_dir.is_dir() else []
    print(f"[reextract] {len(clips)} clips under {run_dir} | concurrency={a.concurrency}", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        futs = {ex.submit(reextract_clip, c, frames_root, a.vlm_frame_height): c for c in clips}
        for f in as_completed(futs):
            uid, msg = f.result()
            done += 1
            print(f"  [{done}/{len(clips)}] {uid}: {msg}", flush=True)
    print("[reextract] done", flush=True)


if __name__ == "__main__":
    main()
