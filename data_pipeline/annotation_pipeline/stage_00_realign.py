#!/usr/bin/env python3
"""Stage 00b (realign): recover the keylog->video time-map and emit corrected keylogs.

Slots into the annotation pipeline between `discover` (build_manifest -> clip
manifest) and `frames` (stage_01). Reads the discover clip manifest, groups
segments by recording, threads idle pauses across segment boundaries, and:

  * writes corrected keylogs (event timestamps re-stamped from the OBS global
    clock to recorded-video PTS) for every segment that has a non-trivial map;
  * writes a REALIGNED clip manifest -- the discover rows verbatim, plus the
    alignment status, with ``keylog_path`` repointed at the corrected keylog for
    corrected segments (``aligned`` segments keep their raw keylog).

Stage 01 then reads this manifest unchanged: bucketing the corrected keylog by
raw timestamp == bucketing the raw keylog by corrected video time, so actions
land on the right frames with no change to the frames/annotate/assemble/canonical
code. The realignment math lives in ``realign_lib`` (spec-faithful: per-segment
naive vs cross-segment global, overhang-refined leading collapse, 5-status
taxonomy). Cross-segment threading needs *every* segment of a recording, so
siblings are enumerated from the source uploads tree, not just manifest rows.

Source keylogs/mp4s are read-only; corrected keylogs go to the output artifact.

Outputs (under --output-dir):
  clips_manifest.jsonl         realigned manifest (frames stage consumes this).
  corrected_keylogs/<sid>.msgpack   video-PTS keylogs for corrected segments.
  alignment.jsonl              per-segment map + spec-v2 status certificate.
  realign_summary.json         status counts + params.
  manifest.json                artifact marker.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import msgpack

from annotation_pipeline import realign_lib as R
from annotation_pipeline.common import ensure_dir, read_jsonl, write_json


def build_keylog_index(uploads_roots: set[Path]) -> dict[str, list[Path]]:
    """Index every source keylog by recording_id (uuid), across ALL version/user
    dirs (a recording's segments can be split across recorder-version dirs).
    Globs each uploads root once."""
    index: dict[str, list[Path]] = defaultdict(list)
    for root in uploads_roots:
        for kp in root.glob("*/*/keylogs/input_*_seg*.msgpack"):
            name = kp.name[len("input_"):]
            uuid = name.rsplit("_seg", 1)[0]
            index[uuid].append(kp)
    return index


def sibling_segments(recording_id: str, keylogs: list[Path]) -> list[dict[str, Any]]:
    """Build seg dicts for ALL segments of a recording from its indexed keylogs,
    deduped by segment index (preferring the copy with a decodable video)."""
    by_idx: dict[int, dict[str, Any]] = {}
    for kp in keylogs:
        seg_str = kp.stem.split("_seg")[-1]
        try:
            seg_idx = int(seg_str)
        except ValueError:
            continue
        rec_dir = kp.parent.parent / "recordings"
        # videos may be zero-padded (seg0007) or not (seg7); accept either.
        vp = rec_dir / f"recording_{recording_id}_seg{seg_idx:04d}.mp4"
        if not vp.exists():
            vp = rec_dir / f"recording_{recording_id}_seg{seg_idx}.mp4"
        cand = {
            "segment_id": f"{recording_id}_seg{seg_idx:04d}",
            "segment_idx": seg_idx,
            "keylog_path": str(kp),
            "video_path": str(vp) if vp.exists() else None,
        }
        prev = by_idx.get(seg_idx)
        if prev is None or (prev["video_path"] is None and cand["video_path"]):
            by_idx[seg_idx] = cand
    return [by_idx[i] for i in sorted(by_idx)]


def write_corrected_keylog(src_keylog: Path, splices: list[dict], out_path: Path) -> None:
    """Re-stamp every keylog event ts (us) keylog-clock -> video-PTS. Lossless
    structure; only timestamps change."""
    corrected = []
    for entry in R.load_keylog(str(src_keylog)):
        if not isinstance(entry, list) or len(entry) < 2:
            corrected.append(entry)
            continue
        try:
            kt = int(entry[0]) / 1e6
        except (TypeError, ValueError):
            corrected.append(entry)
            continue
        corrected.append([int(round(R.keylog_to_video(kt, splices) * 1e6)), entry[1]])
    ensure_dir(out_path.parent)
    out_path.write_bytes(msgpack.packb(corrected, use_bin_type=True))


def realign_one_recording(task: dict) -> dict:
    """Worker: realign one recording. Writes corrected keylogs for its
    dataset-kept segments that have splices; returns per-segment alignment rows
    (with the corrected keylog path, if any). Picklable; no shared state."""
    rec_id = task["recording_id"]
    segs = sibling_segments(rec_id, [Path(p) for p in task["keylogs"]])
    kept = task["kept"]  # {segment_id: {video_dur_s, src_keylog_path}}
    for s in segs:
        vd = (kept.get(s["segment_id"]) or {}).get("video_dur_s")
        if vd is not None:
            s["video_dur_s"] = vd
    results = R.realign_recording(segs, task["idle_timeout"], task["closure_tol"])
    out_dir = Path(task["out_dir"])

    rows: list[dict] = []
    for sid, res in results.items():
        if sid not in kept:
            continue  # threading used all siblings; emit only dataset-kept
        corrected_path = None
        if res["splices"]:
            corrected_path = out_dir / "corrected_keylogs" / f"{sid}.msgpack"
            write_corrected_keylog(Path(kept[sid]["src_keylog"]), res["splices"], corrected_path)
        rows.append({
            "segment_id": sid, "recording_id": rec_id, "segment_idx": res["segment_idx"],
            "status": res["status"], "closed": res["closed"], "model": res["model"],
            "leading_method": res["leading_method"], "n_pauses": res["n_pauses"],
            "total_collapse_s": round(res["total_collapse_s"], 6),
            "overhang_s": round(res["overhang_s"], 6),
            "residual_s": round(res["residual_s"], 6),
            "corr_end_s": round(res["corr_end_s"], 6),
            "keylog_span_s": round(res["keylog_span_s"], 6),
            "video_dur_s": res["video_dur_s"],
            "corrected_keylog_path": str(corrected_path) if corrected_path else None,
            "splices": [{"kp": round(s["kp"], 6), "vp": round(s["vp"], 6),
                         "collapse": round(s["collapse"], 6), "leading": s["leading"]}
                        for s in res["splices"]],
        })
    return {"rows": rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clips-manifest", type=Path, required=True,
                   help="discover output clips_manifest.jsonl.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--idle-timeout", type=float, default=R.IDLE_TIMEOUT)
    p.add_argument("--closure-tol", type=float, default=R.CLOSURE_TOL)
    p.add_argument("--num-workers", type=int, default=0, help="0 = cpu_count().")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    clip_rows = read_jsonl(args.clips_manifest)
    if not clip_rows:
        raise RuntimeError(f"Empty clip manifest: {args.clips_manifest}")

    by_rec: dict[str, list[dict]] = defaultdict(list)
    for r in clip_rows:
        by_rec[r["recording_id"]].append(r)

    uploads_roots = {Path(r["keylog_path"]).parents[3] for r in clip_rows}
    print(f"indexing source keylogs under {len(uploads_roots)} uploads root(s)...", flush=True)
    keylog_index = build_keylog_index(uploads_roots)

    tasks = [{
        "recording_id": rec_id,
        # union the global index with this recording's own keylogs so a manifest
        # segment can never be dropped from the threading.
        "keylogs": sorted({str(p) for p in keylog_index.get(rec_id, [])}
                          | {r["keylog_path"] for r in kept}),
        "kept": {r["segment_id"]: {"video_dur_s": r.get("video_duration_s"),
                                   "src_keylog": r["keylog_path"]} for r in kept},
        "idle_timeout": args.idle_timeout, "closure_tol": args.closure_tol,
        "out_dir": str(out_dir),
    } for rec_id, kept in by_rec.items()]

    n_workers = args.num_workers or mp.cpu_count()
    n_workers = min(n_workers, len(tasks))
    counts: Counter = Counter()
    align_by_sid: dict[str, dict] = {}
    with mp.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(realign_one_recording, tasks, chunksize=8), 1):
            for row in r["rows"]:
                counts[row["status"]] += 1
                align_by_sid[row["segment_id"]] = row
            if i % 1000 == 0:
                print(f"  {i}/{len(tasks)} recordings", flush=True)

    # alignment.jsonl: the per-segment certificate.
    with (out_dir / "alignment.jsonl").open("w") as f:
        for sid in sorted(align_by_sid):
            f.write(json.dumps(align_by_sid[sid]) + "\n")

    # realigned clips_manifest.jsonl: discover rows + alignment status, keylog_path
    # repointed at the corrected keylog for corrected segments (frames consumes this).
    n_repointed = 0
    with (out_dir / "clips_manifest.jsonl").open("w") as f:
        for r in clip_rows:
            a = align_by_sid.get(r["segment_id"])
            row = dict(r)
            if a:
                row["alignment_status"] = a["status"]
                row["alignment_closed"] = a["closed"]
                row["alignment_total_collapse_s"] = a["total_collapse_s"]
                row["alignment_residual_s"] = a["residual_s"]
                row["raw_keylog_path"] = r["keylog_path"]
                if a["corrected_keylog_path"]:
                    row["keylog_path"] = a["corrected_keylog_path"]
                    n_repointed += 1
            f.write(json.dumps(row) + "\n")

    n = len(align_by_sid)
    summary = {
        "n_segments": n, "n_recordings": len(by_rec),
        "idle_timeout_s": args.idle_timeout, "closure_tol_s": args.closure_tol,
        "status_counts": dict(counts),
        "n_closed": sum(counts[s] for s in R.CLOSED_STATUSES),
        "n_corrected": n - counts.get("aligned", 0),
        "n_keylogs_repointed": n_repointed,
        "source_clips_manifest": str(args.clips_manifest),
    }
    write_json(out_dir / "realign_summary.json", summary)
    write_json(out_dir / "manifest.json", {
        "artifact_type": "juergen_annotation_clip_manifest_realigned",
        "schema_version": 1, "clips_file": "clips_manifest.jsonl", **summary,
    })
    print(f"Wrote {n} segments; repointed {n_repointed} keylogs -> {out_dir}")
    print(f"status: {dict(counts)}")


if __name__ == "__main__":
    main()
