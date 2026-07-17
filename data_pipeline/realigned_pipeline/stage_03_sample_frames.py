#!/usr/bin/env python3
"""Stage 01b (sampler): turn a frames-master store into a target-fps
``frame_records.jsonl`` WITHOUT re-decoding any mp4.

Stage 01a (``build_frames_master``) decoded every segment's mp4 ONCE into a JPEG
``images.array_record`` at a fixed MASTER fps. That store is deliberately
**alignment-agnostic and keylog-free**: the decode never reads the keylog, so
realignment (the keylog↔video time fix) is NOT upstream of it. Realignment lands
HERE. This stage is the cheap, metadata-only half of the old
``stage_01_frames_actions``: given a ``--target-fps`` (<= master) it

  * joins each master segment to the REALIGNED ``clips_manifest`` by
    ``segment_id`` to get that segment's realigned/corrected keylog + duration,
  * bins that keylog into per-target-fps ``ActionBin``s (``common.aggregate_actions``),
  * picks, for each target bin, the master record NEAREST in time
    (``round(bin_time * master_fps)``),
  * drops (near-)black frames flagged by stage 01a (``--drop-black-frames``),
    carrying their actions into the next kept frame,
  * thins NO_OP frames per ``--noop-mode`` (``none`` drop all / ``ends`` keep each
    idle run's first+last / ``all`` keep every one; legacy ``--noop-keep-head/tail``
    still honored when ``--noop-mode`` is unset), and
  * emits ``frame_records.jsonl`` whose ``image_path`` points at the SAME master
    shard (``ar:///…/images.array_record#idx``) -- no new JPEG bytes are written.

Two inputs, joined by ``segment_id``:
  --frames-master-dir  a 01a output (segment_index.jsonl + frames/<seg>/…): FRAMES.
  --clips-manifest     the stage 00 realigned clips_manifest.jsonl: the dataset
                       definition + the realigned keylog (its ``keylog_path`` is
                       repointed to the CORRECTED keylog for misaligned segments)
                       + ``alignment_status`` + ``video_duration_s``.

Because 01a is alignment-agnostic, you can re-run realignment (stage 00) and
re-sample here WITHOUT re-decoding. The MASTER fps is the ceiling:
``--target-fps`` must be <= the store's master_fps. At ``target-fps ==
master-fps`` (with the same keylog) the output reproduces stage 01's frames
exactly (bin i -> master record i, same ``source_frame_idx``, same missing set).

Layout is a DROP-IN for ``run_dataset.py --phase annotate``: one
``clips/<segment_id>/stage_01/frame_records.jsonl`` (0-based ``global_frame_idx``,
one segment per file) plus a ``clips/<segment_id>/stage_00/manifest.jsonl``
provenance row, so:

    python -m annotation_pipeline.run_dataset --phase annotate \
        --frames-root <this --output-dir> --manifest <clips_manifest> ...

reads these frames and never touches ffmpeg. (stage 02 requires --manifest as a
flag but does not read it; the actions it needs live in the frame records.)

Outputs (under --output-dir):
  clips/<segment_id>/stage_01/frame_records.jsonl    per-frame image (ar://) + action.
  clips/<segment_id>/stage_01/segment_summaries.json + frames_actions_summary.json
  clips/<segment_id>/stage_00/manifest.jsonl         single-row provenance.
  sample_index.jsonl                                 one row per segment (self-
                                                     sufficient index for stage 02).
  sample_summary.json                                aggregate stats.
  manifest.json                                      artifact marker.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Make the ``realigned_pipeline`` package importable when run directly
# from this folder (mirrors build_frames_master.py).
DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.lib import config  # noqa: E402
from realigned_pipeline.lib.common import (  # noqa: E402
    ActionBin,
    ActionStats,
    aggregate_actions,
    ceil_frames,
    ensure_dir,
    format_action,
    merge_action_bins,
    read_jsonl,
    write_json,
    write_jsonl,
)

# Master statuses (from 01a segment_index.jsonl) that carry a usable frame store.
USABLE_MASTER_STATUSES = {"ok", "cached"}
_FPS_EPS = 1e-9


def _master_frame_manifest(master_row: dict[str, Any], frames_dir: Path, seg: str) -> Path:
    """Locate a segment's master ``frame_manifest.jsonl``: prefer the shard path
    the index recorded; fall back to the canonical ``frames/<seg>/`` layout."""
    shard = master_row.get("shard_path")
    if shard:
        return Path(shard).parent / "frame_manifest.jsonl"
    return frames_dir / seg / "frame_manifest.jsonl"


def _str2bool(s: "str | bool") -> bool:
    """Parse a boolean CLI value. Accepts the labctl ``--flag=value`` form (labctl
    renders every arg as ``--key=value``, so a valueless flag can't be expressed);
    truthy = 1/true/yes/on, everything else False."""
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in ("1", "true", "yes", "on")


def _bin_actions(keylog_path: Path | None, n_bins: int, target_fps: float) -> tuple[list[ActionBin], ActionStats]:
    """Per-target-fps action bins from the (realigned) keylog. A missing/absent
    keylog yields empty bins (``aggregate_actions`` treats a nonexistent file as
    no events)."""
    if keylog_path is None:
        return [ActionBin() for _ in range(n_bins)], ActionStats()
    return aggregate_actions(keylog_path, n_bins, target_fps)


def _is_black(mrec: dict[str, Any], luma_max: float, dark_frac_min: float) -> bool:
    """True if this master record's frame is (near-)black per the stage-01a luma
    metrics. Records without metrics (older masters, decode failures) are NEVER
    dropped -- absence of evidence isn't blackness."""
    ml, fd = mrec.get("mean_luma"), mrec.get("frac_dark")
    return (ml is not None and ml <= luma_max) or (fd is not None and fd >= dark_frac_min)


def _resolve_noop_keep(mode: str | None, head: int, tail: int) -> tuple[int, int, bool]:
    """Effective ``(head, tail, keep_all)`` for NO_OP thinning. ``--noop-mode`` is
    the high-level knob and overrides the legacy head/tail when set:
      ``none`` -> (0, 0, False)  drop every NO_OP frame
      ``ends`` -> (1, 1, False)  keep the first + last frame of each idle run
      ``all``  -> (0, 0, True)   keep every NO_OP frame (skip thinning)
    Unset -> the explicit head/tail (backwards-compatible with existing recipes)."""
    if mode == "all":
        return 0, 0, True
    if mode == "none":
        return 0, 0, False
    if mode == "ends":
        return 1, 1, False
    return head, tail, False


def _is_cached(
    s01_dir: Path,
    target_fps: float,
    head: int,
    tail: int,
    keep_all_noops: bool,
    drop_black: bool,
    black_luma_max: float,
    black_dark_frac_min: float,
) -> list[dict[str, Any]] | None:
    """Return existing records iff a prior run wrote them at the SAME params (so
    re-sampling a different fps / NO_OP mode / black threshold into the same dir
    never silently mixes)."""
    fr = s01_dir / "frame_records.jsonl"
    summ = s01_dir / "frames_actions_summary.json"
    if not (fr.exists() and summ.exists()):
        return None
    try:
        meta = json.loads(summ.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (
        abs(float(meta.get("target_fps", -1)) - target_fps) > _FPS_EPS
        or int(meta.get("noop_keep_head", -1)) != head
        or int(meta.get("noop_keep_tail", -1)) != tail
        or bool(meta.get("noop_keep_all", False)) != keep_all_noops
        or bool(meta.get("drop_black_frames", False)) != drop_black
        or abs(float(meta.get("black_luma_max", -1)) - black_luma_max) > _FPS_EPS
        or abs(float(meta.get("black_dark_frac_min", -1)) - black_dark_frac_min) > _FPS_EPS
    ):
        return None
    return read_jsonl(fr)


def sample_segment(task: dict[str, Any]) -> dict[str, Any]:
    """Worker: sample one segment at target fps -- FRAMES from the master store,
    ACTIONS from the realigned keylog, joined by segment_id -- and write its
    per-clip stage_00/stage_01 artifacts. Picklable; no shared state; returns a
    compact index row. Failures are captured, never raised, so one bad segment
    cannot abort the pool."""
    mrow = task["manifest_row"]        # realigned clips_manifest row (keylog + alignment + duration)
    master_row = task["master_row"]    # master segment_index row (frames) or None
    seg = str(mrow["segment_id"])
    target_fps = float(task["target_fps"])
    head, tail, keep_all_noops = _resolve_noop_keep(
        task.get("noop_mode"), int(task["noop_keep_head"]), int(task["noop_keep_tail"])
    )
    drop_black = bool(task.get("drop_black_frames", True))
    black_luma_max = float(task.get("black_luma_max", config.DEFAULT_BLACK_LUMA_MAX))
    black_dark_frac_min = float(task.get("black_dark_frac_min", config.DEFAULT_BLACK_DARK_FRAC_MIN))

    clip_dir = Path(task["clips_dir"]) / seg
    s01_dir = clip_dir / "stage_01"
    s00_dir = clip_dir / "stage_00"
    frame_records_path = s01_dir / "frame_records.jsonl"

    base = {
        "segment_id": seg,
        "recording_id": mrow.get("recording_id"),
        "segment_idx": mrow.get("segment_idx"),
        "target_fps": target_fps,
        "master_fps": task["master_fps"],
        "keylog_path": mrow.get("keylog_path"),
        "alignment_status": mrow.get("alignment_status"),
        "frame_records": str(frame_records_path),
    }

    # Frames must exist in the master store for this dataset segment.
    if master_row is None:
        return {**base, "status": "no_master_frames", "n_frames": 0, "n_non_noop": 0}
    master_status = master_row.get("status")
    if master_status not in USABLE_MASTER_STATUSES:
        return {**base, "status": f"master_{master_status}", "n_frames": 0, "n_non_noop": 0}

    master_fps = float(master_row.get("master_fps") or task["master_fps"])
    if target_fps > master_fps + _FPS_EPS:
        return {**base, "status": "failed", "n_frames": 0, "n_non_noop": 0,
                "error": f"target_fps {target_fps} exceeds master_fps {master_fps} (can only sample DOWN)"}

    # Resume: reuse only if the existing records were sampled at the same params.
    if not task["force"]:
        cached = _is_cached(s01_dir, target_fps, head, tail, keep_all_noops,
                            drop_black, black_luma_max, black_dark_frac_min)
        if cached is not None:
            return {**base, "status": "cached", "n_frames": len(cached),
                    "n_non_noop": sum(1 for r in cached if r.get("action") != "NO_OP")}

    master_manifest = read_jsonl(_master_frame_manifest(master_row, Path(task["frames_dir"]), seg))
    num_records = len(master_manifest)
    if num_records == 0:
        return {**base, "status": "empty_master", "n_frames": 0, "n_non_noop": 0}

    duration_s = float(mrow.get("video_duration_s") or 0.0)
    n_bins = ceil_frames(duration_s, target_fps)
    keylog = Path(mrow["keylog_path"]) if mrow.get("keylog_path") else None
    bins, action_stats = _bin_actions(keylog, n_bins, target_fps)

    # Pass 1: for each target bin, the nearest master record in time. A bin whose
    # time lands past the master's coverage has no frame -> carry its actions into
    # the next kept bin so a press/release pair is never split (== stage 01).
    candidates: list[dict[str, Any]] = []
    missing_frames = 0
    n_black_dropped = 0
    n_bins_carried = 0
    carry: ActionBin | None = None
    for local_bin_idx in range(n_bins):
        action_bin = bins[local_bin_idx]
        if carry is not None:
            action_bin = merge_action_bins(carry, action_bin)
            carry = None
            n_bins_carried += 1
        local_time_s = local_bin_idx / target_fps
        rec_idx = round(local_time_s * master_fps)
        if rec_idx < 0 or rec_idx >= num_records:
            missing_frames += 1
            carry = action_bin
            continue
        mrec = master_manifest[rec_idx]
        # Black-frame filter: a (near-)black master frame has no usable image, so
        # drop it exactly like an out-of-coverage bin -- carry its actions into the
        # next kept frame (never split a press/release), and let Pass 2 thin NO_OPs
        # over the ALREADY black-filtered sequence.
        if drop_black and _is_black(mrec, black_luma_max, black_dark_frac_min):
            n_black_dropped += 1
            carry = action_bin
            continue
        candidates.append({
            "local_bin_idx": local_bin_idx,
            "local_time_s": local_time_s,
            "source_frame_idx": mrec.get("source_frame_idx"),
            "master_record_index": rec_idx,
            "image": mrec["image"],
            "action": format_action(action_bin),
        })

    # Pass 2: keep every active frame; within each maximal run of consecutive
    # NO_OP frames keep the first HEAD and last TAIL, drop the middle (== stage 01).
    # ``keep_all_noops`` (--noop-mode all) skips thinning entirely; head==tail==0
    # (--noop-mode none) drops whole runs, keeping no NO_OP frames.
    keep = [True] * len(candidates)
    n_noop_dropped = 0
    if not keep_all_noops:
        i = 0
        while i < len(candidates):
            if candidates[i]["action"] != "NO_OP":
                i += 1
                continue
            j = i
            while j < len(candidates) and candidates[j]["action"] == "NO_OP":
                j += 1
            if (j - i) > head + tail:
                for k in range(i + head, j - tail):
                    keep[k] = False
                    n_noop_dropped += 1
            i = j

    records: list[dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        if not keep[idx]:
            continue
        records.append({
            "recording_id": mrow.get("recording_id"),
            "segment_id": seg,
            "segment_idx": mrow.get("segment_idx"),
            "local_bin_idx": cand["local_bin_idx"],
            # One segment per file -> global index is the position within the clip,
            # exactly as run_dataset's per-segment stage 01 produces it.
            "global_frame_idx": len(records),
            "local_time_s": round(cand["local_time_s"], 6),
            "global_time_s": round(cand["local_time_s"], 6),
            "source_frame_idx": cand["source_frame_idx"],
            "master_record_index": cand["master_record_index"],
            "image_path": cand["image"],
            "action": cand["action"],
        })

    seg_summary = {
        "segment_id": seg,
        "segment_idx": mrow.get("segment_idx"),
        "target_fps": target_fps,
        "master_fps": master_fps,
        "duration_s": duration_s,
        "n_bins": n_bins,
        "n_master_records": num_records,
        "n_frames_kept": len(records),
        "n_missing_frames": missing_frames,
        "n_black_dropped": n_black_dropped,
        "n_noop_dropped": n_noop_dropped,
        "noop_mode": task.get("noop_mode"),
        "noop_keep_head": head,
        "noop_keep_tail": tail,
        "noop_keep_all": keep_all_noops,
        "n_bins_carried": n_bins_carried,
        "n_tail_events_dropped": len(carry.events) if carry is not None else 0,
        "sampling_backend": "frames_master",
        "master_shard": master_row.get("shard_path"),
        "keylog_path": mrow.get("keylog_path"),
        "alignment_status": mrow.get("alignment_status"),
        "action_stats": asdict(action_stats),
        "n_non_noop": sum(1 for r in records if r["action"] != "NO_OP"),
    }

    write_jsonl(frame_records_path, records)
    write_json(s01_dir / "segment_summaries.json", [seg_summary])
    # Aggregate summary mirrors stage 01's frames_actions_summary.json schema (n=1
    # segment) so visualize_run and the cache check read it unchanged.
    write_json(s01_dir / "frames_actions_summary.json", {
        "n_segments": 1,
        "n_frames": len(records),
        "n_non_noop": seg_summary["n_non_noop"],
        "target_fps": target_fps,
        "master_fps": master_fps,
        "noop_keep_head": head,
        "noop_keep_tail": tail,
        "noop_keep_all": keep_all_noops,
        "n_noop_dropped": n_noop_dropped,
        "n_black_dropped": n_black_dropped,
        "drop_black_frames": drop_black,
        "black_luma_max": black_luma_max,
        "black_dark_frac_min": black_dark_frac_min,
        "sampling_backend": "frames_master",
        "total_duration_s": round(duration_s, 3),
    })
    # Provenance row for run_dataset --phase annotate (stage 02 wants the flag, not
    # its contents). The realigned manifest row + the master shard + fps sampled.
    write_jsonl(s00_dir / "manifest.jsonl", [{
        **mrow, "sampled_target_fps": target_fps, "master_shard": master_row.get("shard_path"),
    }])

    return {**base, "status": "ok" if records else "empty",
            "n_frames": len(records), "n_non_noop": seg_summary["n_non_noop"],
            "n_missing_frames": missing_frames, "n_noop_dropped": n_noop_dropped,
            "n_black_dropped": n_black_dropped}


def _load_master_fps(master_dir: Path, index_rows: list[dict[str, Any]], fallback: float) -> float:
    """The store's master fps = the sampling ceiling. Prefer the artifact summary;
    fall back to the per-segment index rows (one 01a run -> uniform)."""
    summary_path = master_dir / "frames_master_summary.json"
    if summary_path.is_file():
        try:
            return float(json.loads(summary_path.read_text())["master_fps"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    seen = {float(r["master_fps"]) for r in index_rows if r.get("master_fps") is not None}
    return max(seen) if seen else fallback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--frames-master-dir",
        type=Path,
        required=True,
        help="A stage 01a (build_frames_master) --output-dir: must contain "
        "segment_index.jsonl and frames/<seg>/images.array_record.",
    )
    p.add_argument(
        "--clips-manifest",
        type=Path,
        required=True,
        help="stage 00 realigned clips_manifest.jsonl: the dataset definition and "
        "the source of the realigned keylog (its keylog_path is repointed to the "
        "corrected keylog) + alignment_status + video_duration_s. Joined to the "
        "master store by segment_id.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--target-fps",
        type=float,
        default=config.DEFAULT_TARGET_FPS,
        help=f"Sampling rate. Must be <= the store's master_fps. Default {config.DEFAULT_TARGET_FPS}.",
    )
    p.add_argument(
        "--noop-mode",
        choices=config.NOOP_MODES,
        default=None,
        help="How many NO_OP frames to keep (overrides --noop-keep-head/tail): "
        "'none' drops every idle frame, 'ends' keeps the first+last of each idle "
        "run, 'all' keeps them all. Unset -> use --noop-keep-head/tail.",
    )
    p.add_argument("--noop-keep-head", type=int, default=config.DEFAULT_NOOP_KEEP_HEAD,
                   help="Legacy: within each NO_OP run, keep the first this many frames "
                   "(ignored when --noop-mode is set).")
    p.add_argument("--noop-keep-tail", type=int, default=config.DEFAULT_NOOP_KEEP_TAIL,
                   help="Legacy: within each NO_OP run, keep the last this many frames "
                   "(ignored when --noop-mode is set).")
    p.add_argument("--drop-black-frames", nargs="?", const=True, type=_str2bool,
                   default=config.DEFAULT_DROP_BLACK_FRAMES, metavar="BOOL",
                   help="Drop (near-)black master frames flagged by stage 01a; their "
                   "actions carry into the next kept frame. Bare --drop-black-frames = on; "
                   "pass --drop-black-frames=false to keep black frames (works via labctl's "
                   "--key=value arg form).")
    p.add_argument("--black-luma-max", type=float, default=config.DEFAULT_BLACK_LUMA_MAX,
                   help="Drop a frame if its mean luma (0-255) is <= this.")
    p.add_argument("--black-dark-frac-min", type=float, default=config.DEFAULT_BLACK_DARK_FRAC_MIN,
                   help="...or if this fraction of its pixels are near-black.")
    p.add_argument("--num-workers", type=int, default=0, help="0 = cpu_count().")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N segments (debug).")
    p.add_argument("--force", action="store_true", help="Re-sample segments even if same-fps records already exist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_fps <= 0:
        raise SystemExit("--target-fps must be > 0")

    master_dir = args.frames_master_dir
    index_path = master_dir / "segment_index.jsonl"
    if not index_path.is_file():
        raise SystemExit(f"no segment_index.jsonl under {master_dir} (is it a frames-master artifact?)")
    index_rows = read_jsonl(index_path)
    if not index_rows:
        raise RuntimeError(f"Empty segment index: {index_path}")
    master_by_seg = {str(r["segment_id"]): r for r in index_rows}

    master_fps = _load_master_fps(master_dir, index_rows, args.target_fps)
    if args.target_fps > master_fps + _FPS_EPS:
        raise SystemExit(
            f"--target-fps {args.target_fps} exceeds the store's master_fps {master_fps}: "
            "you can only sample DOWN. Rebuild the frames-master at a higher --master-fps."
        )

    manifest_rows = read_jsonl(args.clips_manifest)
    if not manifest_rows:
        raise RuntimeError(f"Empty clips manifest: {args.clips_manifest}")
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]

    out_dir = ensure_dir(args.output_dir)
    clips_dir = ensure_dir(out_dir / "clips")
    frames_dir = master_dir / "frames"

    tasks = [
        {
            "manifest_row": row,
            "master_row": master_by_seg.get(str(row["segment_id"])),
            "clips_dir": str(clips_dir),
            "frames_dir": str(frames_dir),
            "target_fps": args.target_fps,
            "master_fps": master_fps,
            "noop_mode": args.noop_mode,
            "noop_keep_head": args.noop_keep_head,
            "noop_keep_tail": args.noop_keep_tail,
            "drop_black_frames": args.drop_black_frames,
            "black_luma_max": args.black_luma_max,
            "black_dark_frac_min": args.black_dark_frac_min,
            "force": args.force,
        }
        for row in manifest_rows
    ]

    n_workers = args.num_workers or mp.cpu_count()
    n_workers = max(1, min(n_workers, len(tasks)))
    noop_desc = args.noop_mode or f"head/tail={args.noop_keep_head}/{args.noop_keep_tail}"
    black_desc = (
        f"drop_black(luma<={args.black_luma_max},dark>={args.black_dark_frac_min})"
        if args.drop_black_frames else "keep_black"
    )
    print(
        f"[sample] {len(tasks)} segments | target_fps={args.target_fps} "
        f"(master_fps={master_fps}) noop={noop_desc} {black_desc} "
        f"| workers={n_workers}",
        flush=True,
    )

    counts: Counter = Counter()
    index_out: list[dict[str, Any]] = []
    n_frames_total = 0
    n_non_noop_total = 0
    n_black_dropped_total = 0
    with mp.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(sample_segment, tasks, chunksize=8), 1):
            counts[res["status"]] += 1
            n_frames_total += int(res.get("n_frames") or 0)
            n_non_noop_total += int(res.get("n_non_noop") or 0)
            n_black_dropped_total += int(res.get("n_black_dropped") or 0)
            index_out.append(res)
            if res["status"] == "failed":
                print(f"  FAIL {res['segment_id']}: {res.get('error')}", flush=True)
            if i % 1000 == 0:
                print(f"  {i}/{len(tasks)} segments | {dict(counts)}", flush=True)

    index_out.sort(key=lambda r: str(r["segment_id"]))
    with (out_dir / "sample_index.jsonl").open("w") as f:
        for r in index_out:
            f.write(json.dumps(r) + "\n")

    summary = {
        "target_fps": args.target_fps,
        "master_fps": master_fps,
        "noop_mode": args.noop_mode,
        "noop_keep_head": args.noop_keep_head,
        "noop_keep_tail": args.noop_keep_tail,
        "drop_black_frames": args.drop_black_frames,
        "black_luma_max": args.black_luma_max,
        "black_dark_frac_min": args.black_dark_frac_min,
        "n_segments": len(tasks),
        "status_counts": dict(counts),
        "n_frames_total": n_frames_total,
        "n_non_noop_total": n_non_noop_total,
        "n_black_dropped_total": n_black_dropped_total,
        "frames_master_dir": str(master_dir),
        "source_clips_manifest": str(args.clips_manifest),
    }
    write_json(out_dir / "sample_summary.json", summary)
    write_json(
        out_dir / "manifest.json",
        {
            "artifact_type": "juergen_annotation_frames_sampled",
            "schema_version": 1,
            "sample_index": "sample_index.jsonl",
            "clips_layout": "clips/<segment_id>/stage_01/frame_records.jsonl",
            **summary,
        },
    )
    print(
        f"[sample] done: {dict(counts)} | {n_frames_total} frames "
        f"({n_non_noop_total} non-NO_OP, {n_black_dropped_total} black dropped) -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
