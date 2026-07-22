#!/usr/bin/env python3
"""Pre/post alignment audit: new filter+selector+formatter vs a FROZEN legacy
stage-03 sampling artifact (pre-rewrite code output) on the same inputs.

For every segment present in both, this compares
  * the SELECTED FRAMES (by master record index — the pixels shown), and
  * the ACTION STRING paired with each frame (the label the trainee learns),
and classifies every difference. The gate PASSES iff every difference falls
into one of the two DESIGNED semantic changes:

  new_only_noop      frames only the new view keeps, all labeled NO_OP —
                     raw-activity idle judgment keeps sub-round drift seconds
                     that legacy's rounded-bin NO_OP judgment dropped.
  expected_tail      the segment's LAST frame differs AND the keylog runs past
                     the video's end — legacy binned post-coverage events into
                     the final action; the new policy discards them as
                     no_coverage (nothing was visible). Deltas differ by
                     exactly those discarded events.
  legacy_only_post_coverage
                     a frame only legacy kept, whose bin has ZERO in-coverage
                     activity — it owed its existence entirely to keylog
                     events after the video's last frame (same root cause as
                     expected_tail; legacy paired them with a stale frame).
  new_only_clamped   a new-only frame whose non-NO_OP label consists SOLELY of
                     clamp-emitted key events (zero deltas): a press/release
                     pulled back from a dead zone or post-coverage span onto
                     the last visible frame. Legacy dropped such events
                     silently, leaving the key DANGLING for the rest of the
                     conversation; the new policy closes it by construction.

ANY other difference — a shared frame with a different action string, a
legacy frame the new view lost for any in-coverage reason, a non-NO_OP
new-only frame — is a real regression and fails the gate. Zero mismatches on
shared frames == the frame<->action pairing (i.e. the realignment as
consumed) is bit-for-bit what the pre-change pipeline produced.

To reproduce legacy fps-1 noop_mode=none selection exactly, build the filter
with the legacy-equivalent idle knobs first::

    uv run python realigned_pipeline/stage_03_filter.py \
        --frames-master-dir <master> --clips-manifest <realigned manifest> \
        --output-dir <tmp filter> \
        --idle-min-duration-s 0 --idle-keep-head-s 0 --idle-keep-tail-s 0 \
        --idle-judgment-bin-s 1.0   # = one legacy bin at --fps 1

    uv run python realigned_pipeline/verify_against_legacy.py \
        --legacy-sample-dir <frozen stage-03 artifact> \
        --filter-dir <tmp filter> --fps 1
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.lib.action_format import get_formatter  # noqa: E402
from realigned_pipeline.lib.common import normalize_dashed_argv, read_jsonl  # noqa: E402
from realigned_pipeline.lib.events import apply_label_policy, load_events  # noqa: E402
from realigned_pipeline.lib.views import FilterArtifact, build_segment_view  # noqa: E402

LEGACY_USABLE = {"ok", "cached"}


def _is_raw_activity(e) -> bool:
    if e.kind == "move":
        return e.dx != 0.0 or e.dy != 0.0
    if e.kind == "scroll":
        return e.scroll != 0.0
    return True  # press / release


def audit_segment(task: dict) -> dict:
    seg = task["segment_id"]
    out = {"segment_id": seg, "status": "ok", "n_shared": 0, "n_mismatch": 0,
           "n_legacy_only": 0, "n_legacy_only_post_coverage": 0,
           "n_new_only_noop": 0, "n_new_only_active": 0, "n_new_only_clamped": 0,
           "n_expected_tail": 0, "examples": []}
    try:
        legacy = read_jsonl(Path(task["legacy_frame_records"]))
        legacy_by_master = {int(r["master_record_index"]): str(r["action"]) for r in legacy}

        filter_seg = json.loads(Path(task["filter_seg_path"]).read_text())
        view = build_segment_view(filter_seg, fps=task["fps"])
        keylog = filter_seg.get("keylog_path")
        events, _ = load_events(Path(keylog)) if keylog else ([], None)
        if view.frames:
            result = get_formatter("canonical").format_segment(
                events, view.windows(), view.dead_zones, master_fps=view.master_fps)
            new_by_master = {f.master_idx: result.labels[f.view_idx] for f in view.frames}
        else:
            new_by_master = {}

        shared = set(legacy_by_master) & set(new_by_master)
        legacy_only = set(legacy_by_master) - set(new_by_master)
        new_only = set(new_by_master) - set(legacy_by_master)
        mismatches = {m for m in shared if legacy_by_master[m] != new_by_master[m]}

        master_fps = view.master_fps
        n_records = view.n_records
        has_post_coverage = any(int(e.t_s * master_fps) >= n_records for e in events)

        # Designed tail difference: keylog events past master coverage were
        # binned into legacy's final action; the new policy discards them.
        if view.frames:
            last_master = view.frames[-1].master_idx
            if last_master in mismatches and has_post_coverage:
                mismatches.discard(last_master)
                out["n_expected_tail"] = 1

        # Designed loss: a legacy frame whose bin [m, m+stride) has ZERO
        # in-coverage activity was kept only because post-coverage events fell
        # into its (duration-derived, coverage-overhanging) legacy bin.
        stride = round(master_fps / task["fps"])
        if legacy_only and has_post_coverage:
            active_ticks = {int(e.t_s * master_fps) for e in events
                            if _is_raw_activity(e) and 0 <= int(e.t_s * master_fps) < n_records}
            post_only = {m for m in legacy_only
                         if not any(t in active_ticks for t in range(m, min(m + stride, n_records)))}
            legacy_only -= post_only
            out["n_legacy_only_post_coverage"] = len(post_only)

        out["n_shared"] = len(shared)
        out["n_mismatch"] = len(mismatches)
        out["n_legacy_only"] = len(legacy_only)
        out["n_new_only_noop"] = sum(1 for m in new_only if new_by_master[m] == "NO_OP")
        new_only_active = {m for m in new_only if new_by_master[m] != "NO_OP"}
        # Designed: a new-only frame whose key events are ALL clamp-emissions
        # (a release/press pulled back from a dead zone or post-coverage span)
        # and whose deltas are zero. Legacy dropped those events silently and
        # left the key dangling; the new policy closes it on a visible frame.
        if new_only_active and view.frames:
            vi_by_master = {f.master_idx: f.view_idx for f in view.frames}
            labeled, _ = apply_label_policy(
                events, view.windows(), view.dead_zones, master_fps=master_fps)
            for m in sorted(new_only_active):
                owned = [le for le in labeled if le.window == vi_by_master[m]]
                keys = [le for le in owned if le.event.kind in ("press", "release")]
                if (keys and all(le.clamped for le in keys)
                        and new_by_master[m].startswith("0 0 0 ;")):
                    new_only_active.discard(m)
                    out["n_new_only_clamped"] += 1
        out["n_new_only_active"] = len(new_only_active)
        for m in sorted(mismatches)[:3]:
            out["examples"].append(
                {"kind": "mismatch", "master_idx": m,
                 "legacy": legacy_by_master[m], "new": new_by_master[m]})
        for m in sorted(legacy_only)[:3]:
            out["examples"].append(
                {"kind": "legacy_only", "master_idx": m, "legacy": legacy_by_master[m]})
        for m in sorted(new_only_active)[:3]:
            out["examples"].append(
                {"kind": "new_only_active", "master_idx": m, "new": new_by_master[m]})
        if mismatches or legacy_only or out["n_new_only_active"]:
            out["status"] = "DIFF"
        return out
    except Exception as exc:  # noqa: BLE001 - per-segment fault isolation
        return {**out, "status": "failed", "error": f"{exc}"}


def parse_args() -> argparse.Namespace:
    normalize_dashed_argv()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--legacy-sample-dir", type=Path, required=True,
                   help="A FROZEN pre-rewrite stage-03 artifact (sample_index.jsonl + "
                        "clips/<seg>/stage_01/frame_records.jsonl).")
    p.add_argument("--filter-dir", type=Path, required=True,
                   help="A new stage-03 filter artifact built with legacy-equivalent "
                        "idle knobs (see module docstring).")
    p.add_argument("--fps", type=float, required=True,
                   help="The legacy artifact's target fps (its sample_summary.json target_fps).")
    p.add_argument("--report", type=Path, default=None,
                   help="Write the per-segment audit rows here (jsonl).")
    p.add_argument("--num-workers", type=int, default=0, help="0 = cpu_count().")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    art = FilterArtifact(args.filter_dir)
    art.stride_for(args.fps)

    legacy_index = read_jsonl(args.legacy_sample_dir / "sample_index.jsonl")
    legacy_by_seg = {str(r["segment_id"]): r for r in legacy_index
                     if r.get("status") in LEGACY_USABLE}

    tasks = []
    n_no_legacy = 0
    rows = art.usable_rows()
    if args.limit is not None:
        rows = rows[: args.limit]
    for row in rows:
        seg = str(row["segment_id"])
        lrow = legacy_by_seg.get(seg)
        if lrow is None:
            n_no_legacy += 1
            continue
        tasks.append({
            "segment_id": seg,
            "legacy_frame_records": lrow["frame_records"],
            "filter_seg_path": str(art.segment_path(seg)),
            "fps": args.fps,
        })
    if not tasks:
        raise SystemExit("no overlapping segments between the legacy artifact and the filter")

    n_workers = max(1, min(args.num_workers or mp.cpu_count(), len(tasks)))
    print(f"[verify] {len(tasks)} segments in both ({n_no_legacy} filter-only skipped) "
          f"| fps={args.fps} workers={n_workers}", flush=True)

    totals: Counter = Counter()
    statuses: Counter = Counter()
    diff_examples: list[dict] = []
    report_rows: list[dict] = []
    with mp.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(audit_segment, tasks, chunksize=8), 1):
            statuses[res["status"]] += 1
            for k in ("n_shared", "n_mismatch", "n_legacy_only", "n_legacy_only_post_coverage",
                      "n_new_only_noop", "n_new_only_active", "n_new_only_clamped",
                      "n_expected_tail"):
                totals[k] += int(res.get(k) or 0)
            if res["status"] in ("DIFF", "failed"):
                diff_examples.append(res)
                print(f"  {res['status']} {res['segment_id']}: "
                      f"mismatch={res.get('n_mismatch')} legacy_only={res.get('n_legacy_only')} "
                      f"new_only_active={res.get('n_new_only_active')} "
                      f"{res.get('error') or res.get('examples')}", flush=True)
            report_rows.append(res)
            if i % 2000 == 0:
                print(f"  {i}/{len(tasks)} | {dict(statuses)}", flush=True)

    if args.report:
        with args.report.open("w") as f:
            for r in sorted(report_rows, key=lambda r: r["segment_id"]):
                f.write(json.dumps(r) + "\n")

    n_regressions = (totals["n_mismatch"] + totals["n_legacy_only"]
                     + totals["n_new_only_active"] + statuses["failed"])
    print(f"\n[verify] segments: {dict(statuses)}")
    print(f"[verify] shared frames: {totals['n_shared']} | byte-identical: "
          f"{totals['n_shared'] - totals['n_mismatch']} | MISMATCH: {totals['n_mismatch']}")
    print(f"[verify] designed diffs: {totals['n_new_only_noop']} new-only NO_OP frames, "
          f"{totals['n_expected_tail']} post-coverage tail actions, "
          f"{totals['n_legacy_only_post_coverage']} post-coverage-only legacy frames, "
          f"{totals['n_new_only_clamped']} clamp-closure frames (legacy dangling keys)")
    print(f"[verify] regressions: {totals['n_mismatch']} mismatched + "
          f"{totals['n_legacy_only']} lost + {totals['n_new_only_active']} new-active "
          f"+ {statuses['failed']} failed")
    print(f"[verify] GATE: {'PASS' if n_regressions == 0 else 'FAIL'}")
    if n_regressions:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
