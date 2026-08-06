#!/usr/bin/env python3
"""Stage 03 (filter): a pure keep/drop mask over the master frame axis.

This stage SAMPLES NOTHING. It judges, per master tick (tick i == master
record i == 1/master_fps s), whether that tick's frame is usable for any
downstream consumer, and writes a compact survivor mask:

  * black:  per master record, from the stage-01 luma metrics
    (``mean_luma``/``frac_dark`` vs thresholds) — already master-resolution.
  * idle:   the realigned keylog is judged per ``--idle-judgment-bin-s`` bin
    (default: a 2 s bin with the rounded NO_OP predicate, see
    ``--idle-activity``); the INTERIOR of any inactive run longer than
    ``--idle-min-duration-s`` is dropped, keeping ``--idle-keep-head-s``/
    ``--idle-keep-tail-s`` at each end. All knobs are in SECONDS, so a 4 fps
    and a 15 fps master behave identically.

Because the output is a mask at master resolution (not a sampled dataset),
nothing here caps downstream fps: stage 03b annotates at k fps and stage 04
trains at x fps, each via the shared selector (``lib/views.build_view``),
bounded only by the master fps. Re-running with different thresholds is
metadata-only — no decode, no JPEG bytes.

Two inputs, joined by ``segment_id``:
  --frames-master-dir  stage 01 output: segment_index.jsonl + frames/<seg>/…
  --clips-manifest     stage 00/02 realigned clips_manifest.jsonl (keylog_path
                       repointed to the corrected keylog, alignment_status).

Outputs (under --output-dir):
  filter/<segment_id>.json   kept_ranges [[start,end),…] + dropped intervals
                             ({start,end,reason: black|idle_interior}) + join
                             metadata (shard_path, keylog_path, n_records).
  filter_index.jsonl         one row per segment (status + counts).
  filter_summary.json        aggregate stats.
  manifest.json              artifact marker; carries ``master_store_id`` so
                             consumers refuse joins against a rebuilt master.
  qc_view/<segment_id>.jsonl (only with --qc-view-fps) a diagnostic sampled
                             view with derived canonical action strings, for
                             realignment eyeballing — nothing downstream
                             reads it.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

# Make the ``pipeline`` package importable when run directly
# from this folder (mirrors the other stages).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.lib import config  # noqa: E402
from pipeline.lib.action_format import get_formatter  # noqa: E402
from pipeline.lib.common import (  # noqa: E402
    aggregate_actions,
    ensure_dir,
    format_action,
    normalize_dashed_argv,
    read_jsonl,
    write_json,
    write_jsonl,
)
from pipeline.lib.events import load_events  # noqa: E402
from pipeline.lib.manifest import make_artifact_id  # noqa: E402
from pipeline.lib.views import build_segment_view, resolve_stride  # noqa: E402

# Master statuses (from stage-01 segment_index.jsonl) that carry a usable store.
USABLE_MASTER_STATUSES = {"ok", "cached"}

REASON_KEPT = 0
REASON_BLACK = 1
REASON_IDLE = 2
_REASON_NAMES = {REASON_BLACK: "black", REASON_IDLE: "idle_interior"}


def _master_frame_manifest(master_row: dict[str, Any], frames_dir: Path, seg: str) -> Path:
    """Locate a segment's master ``frame_manifest.jsonl``: prefer the shard path
    the index recorded; fall back to the canonical ``frames/<seg>/`` layout."""
    shard = master_row.get("shard_path")
    if shard:
        return Path(shard).parent / "frame_manifest.jsonl"
    return frames_dir / seg / "frame_manifest.jsonl"


def _is_black(mrec: dict[str, Any], luma_max: float, dark_frac_min: float) -> bool:
    """True if this master record's frame is (near-)black per the stage-01 luma
    metrics. Records without metrics (older masters, decode failures) are NEVER
    dropped -- absence of evidence isn't blackness."""
    ml, fd = mrec.get("mean_luma"), mrec.get("frac_dark")
    return (ml is not None and ml <= luma_max) or (fd is not None and fd >= dark_frac_min)


def _str2bool(s: str | bool) -> bool:
    """Parse a boolean CLI value (labctl renders every arg as ``--key=value``)."""
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in ("1", "true", "yes", "on")


def _activity_mask(keylog_path: Path | None, n_records: int, master_fps: float) -> list[bool]:
    """Per master tick: did the demonstrator do anything in it? Judged on RAW
    events (a slow sub-pixel mouse drift is still activity), so idleness is not
    an artifact of per-tick rounding."""
    active = [False] * n_records
    if keylog_path is None or not keylog_path.exists():
        return active
    events, _ = load_events(keylog_path)
    for e in events:
        tick = int(e.t_s * master_fps)
        if tick < 0 or tick >= n_records:
            continue
        if e.kind == "move":
            if e.dx != 0.0 or e.dy != 0.0:
                active[tick] = True
        elif e.kind == "scroll":
            if e.scroll != 0.0:
                active[tick] = True
        else:  # press / release
            active[tick] = True
    return active


def _rounded_activity_mask(
    keylog_path: Path | None, n_records: int, master_fps: float, bin_ticks: int
) -> list[bool]:
    """Legacy-predicate activity at judgment-bin granularity: a bin is active
    iff its FORMATTED action is non-NO_OP — i.e. summed deltas round to
    nonzero, or it carries deduped key events. Runs the real binning code
    (``aggregate_actions`` + ``format_action``) rather than reimplementing the
    predicate, so delta rounding, scroll fallback and held-set dedup of
    autorepeats/dangling releases all behave identically to the emitted label.
    With ``--idle-min-duration-s 0`` and
    zero head/tail this keeps every frame at the judgment fps (0 NO_OP
    frames)."""
    active = [False] * n_records
    if keylog_path is None or not keylog_path.exists():
        return active
    judgment_fps = master_fps / bin_ticks
    n_bins = (n_records + bin_ticks - 1) // bin_ticks
    bins, _ = aggregate_actions(keylog_path, n_bins, judgment_fps)
    for b, action_bin in enumerate(bins):
        if format_action(action_bin) != "NO_OP":
            for t in range(b * bin_ticks, min((b + 1) * bin_ticks, n_records)):
                active[t] = True
    return active


def _coarsen_activity(active: list[bool], bin_ticks: int) -> list[bool]:
    """Smear activity to judgment-bin granularity: every tick of a bin counts
    as active if ANY tick in it was. Coarser bins make the idle judgment less
    twitchy (a slot tick inside a micro-pause of an otherwise busy second is
    not idle); ``bin_ticks == 1`` is a no-op."""
    if bin_ticks <= 1:
        return active
    n = len(active)
    out = [False] * n
    for b0 in range(0, n, bin_ticks):
        b1 = min(b0 + bin_ticks, n)
        if any(active[b0:b1]):
            for t in range(b0, b1):
                out[t] = True
    return out


def _idle_interiors(
    active: list[bool],
    master_fps: float,
    min_duration_s: float,
    keep_head_s: float,
    keep_tail_s: float,
) -> list[tuple[int, int]]:
    """Interiors of maximal inactive runs longer than ``min_duration_s``:
    ``[run_start + head, run_end - tail)`` in ticks, empty spans skipped."""
    min_ticks = round(min_duration_s * master_fps)
    head = round(keep_head_s * master_fps)
    tail = round(keep_tail_s * master_fps)
    spans: list[tuple[int, int]] = []
    n = len(active)
    i = 0
    while i < n:
        if active[i]:
            i += 1
            continue
        j = i
        while j < n and not active[j]:
            j += 1
        if (j - i) > min_ticks and (i + head) < (j - tail):
            spans.append((i + head, j - tail))
        i = j
    return spans


def _compress_reasons(reasons: list[int]) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Per-tick reason codes -> (kept_ranges, dropped intervals with reason)."""
    kept: list[list[int]] = []
    dropped: list[dict[str, Any]] = []
    i, n = 0, len(reasons)
    while i < n:
        j = i
        while j < n and reasons[j] == reasons[i]:
            j += 1
        if reasons[i] == REASON_KEPT:
            kept.append([i, j])
        else:
            dropped.append({"start": i, "end": j, "reason": _REASON_NAMES[reasons[i]]})
        i = j
    return kept, dropped


def filter_params(
    *,
    drop_black_frames: Any,
    black_luma_max: Any,
    black_dark_frac_min: Any,
    idle_min_duration_s: Any,
    idle_keep_head_s: Any,
    idle_keep_tail_s: Any,
    idle_judgment_bin_s: Any,
    idle_activity: Any,
) -> dict[str, Any]:
    """The ONE definition of a filter's judgment parameters.

    These knobs decide which frames survive, so exactly one function normalises
    and validates them. Every caller passes all eight: there is no default here,
    because the CLI is the one place defaults are applied (from ``lib.config``),
    and a second default would let the same run judge differently depending on
    which entry point built it.

    ``idle_judgment_bin_s`` falsy means "no binning" and normalises to ``None``,
    so a run that passes ``0`` and a run that passes nothing record the same
    value in the per-segment artifact AND in the run manifest.

    ``rounded`` needs a bin to evaluate the NO_OP predicate over; that pairing is
    refused here rather than reaching a worker and failing on ``None * fps``.

    ``qc_view_fps`` is deliberately NOT part of this set: it only affects the
    diagnostic view, and including it would invalidate the resume cache of every
    artifact already on disk.
    """
    activity = str(idle_activity)
    if activity not in config.IDLE_ACTIVITIES:
        raise ValueError(
            f"idle_activity must be one of {list(config.IDLE_ACTIVITIES)}, got {activity!r}"
        )
    bin_s = float(idle_judgment_bin_s) if idle_judgment_bin_s else None
    if activity == "rounded" and bin_s is None:
        raise ValueError(
            "idle_activity 'rounded' needs a nonzero idle_judgment_bin_s: it is the "
            "granularity the NO_OP predicate is evaluated at"
        )
    return {
        "drop_black_frames": bool(drop_black_frames),
        "black_luma_max": float(black_luma_max),
        "black_dark_frac_min": float(black_dark_frac_min),
        "idle_min_duration_s": float(idle_min_duration_s),
        "idle_keep_head_s": float(idle_keep_head_s),
        "idle_keep_tail_s": float(idle_keep_tail_s),
        "idle_judgment_bin_s": bin_s,
        "idle_activity": activity,
    }


#: The task-dict keys ``filter_params`` consumes, so one list drives both the
#: worker's read and the CLI's build.
FILTER_PARAM_KEYS = (
    "drop_black_frames",
    "black_luma_max",
    "black_dark_frac_min",
    "idle_min_duration_s",
    "idle_keep_head_s",
    "idle_keep_tail_s",
    "idle_judgment_bin_s",
    "idle_activity",
)


def _write_qc_view(qc_dir: Path, filter_seg: dict[str, Any], qc_fps: float) -> int:
    """Diagnostic sampled view with derived canonical actions (realignment
    eyeballing only; nothing downstream reads it)."""
    view = build_segment_view(filter_seg, fps=qc_fps)
    if not view.frames:
        return 0  # fully-masked segment: nothing to eyeball
    keylog = filter_seg.get("keylog_path")
    events, _ = load_events(Path(keylog)) if keylog else ([], None)
    result = get_formatter("canonical").format_segment(
        events, view.windows(), view.dead_zones, master_fps=view.master_fps
    )
    # Rows use the frame_records.jsonl field names so visualize_frame_records
    # opens a filter artifact's qc_view/ directly (frames + HUD + timeline).
    rows = [
        {
            "segment_id": view.segment_id,
            "recording_id": view.recording_id,
            "view_idx": f.view_idx,
            "local_bin_idx": f.slot,
            "master_record_index": f.master_idx,
            "local_time_s": round(f.t_s, 6),
            "image_path": f.image,
            "action": result.labels[f.view_idx],
        }
        for f in view.frames
    ]
    write_jsonl(qc_dir / f"{filter_seg['segment_id']}.jsonl", rows)
    return len(rows)


def filter_segment(task: dict[str, Any]) -> dict[str, Any]:
    """Worker: judge one segment's mask and write ``filter/<seg>.json``.
    Picklable; no shared state; failures are captured, never raised, so one
    bad segment cannot abort the pool."""
    mrow = task["manifest_row"]
    master_row = task["master_row"]
    seg = str(mrow["segment_id"])
    params = filter_params(**{k: task[k] for k in FILTER_PARAM_KEYS})
    out_path = Path(task["filter_dir"]) / f"{seg}.json"

    base = {
        "segment_id": seg,
        "recording_id": mrow.get("recording_id"),
        "segment_idx": mrow.get("segment_idx"),
        "alignment_status": mrow.get("alignment_status"),
        "filter_path": str(out_path),
    }
    try:
        if master_row is None:
            return {**base, "status": "no_master_frames"}
        if master_row.get("status") not in USABLE_MASTER_STATUSES:
            return {**base, "status": f"master_{master_row.get('status')}"}
        master_fps = float(master_row.get("master_fps") or task["master_fps"])

        # Resume: reuse only if a prior run judged with the SAME params.
        if not task["force"] and out_path.exists():
            try:
                prev = json.loads(out_path.read_text())
            except (OSError, json.JSONDecodeError):
                prev = None
            if prev is not None and prev.get("params") == params:
                if task.get("qc_view_fps"):
                    _write_qc_view(Path(task["qc_dir"]), prev, float(task["qc_view_fps"]))
                return {
                    **base,
                    "status": "cached",
                    "n_records": prev.get("n_master_records"),
                    **{k: prev.get(k) for k in ("n_kept", "n_black", "n_idle_interior")},
                }

        master_manifest = read_jsonl(
            _master_frame_manifest(master_row, Path(task["frames_dir"]), seg)
        )
        n_records = len(master_manifest)
        if n_records == 0:
            return {**base, "status": "empty_master"}

        reasons = [REASON_KEPT] * n_records
        keylog = Path(mrow["keylog_path"]) if mrow.get("keylog_path") else None
        if params["idle_activity"] == "rounded":
            active = _rounded_activity_mask(
                keylog, n_records, master_fps,
                round(params["idle_judgment_bin_s"] * master_fps),
            )
        else:
            active = _activity_mask(keylog, n_records, master_fps)
            if params["idle_judgment_bin_s"]:
                active = _coarsen_activity(
                    active, round(params["idle_judgment_bin_s"] * master_fps)
                )
        for start, end in _idle_interiors(
            active,
            master_fps,
            params["idle_min_duration_s"],
            params["idle_keep_head_s"],
            params["idle_keep_tail_s"],
        ):
            for t in range(start, end):
                reasons[t] = REASON_IDLE
        # Black wins over idle: a black tick is a dead zone downstream (its
        # pixels are unusable), an idle tick is merely uninteresting.
        if params["drop_black_frames"]:
            for t, mrec in enumerate(master_manifest):
                if _is_black(mrec, params["black_luma_max"], params["black_dark_frac_min"]):
                    reasons[t] = REASON_BLACK

        kept_ranges, dropped = _compress_reasons(reasons)
        n_black = sum(s["end"] - s["start"] for s in dropped if s["reason"] == "black")
        n_idle = sum(s["end"] - s["start"] for s in dropped if s["reason"] == "idle_interior")
        filter_seg_doc = {
            "segment_id": seg,
            "recording_id": mrow.get("recording_id"),
            "segment_idx": mrow.get("segment_idx"),
            "master_fps": master_fps,
            "n_master_records": n_records,
            "video_duration_s": mrow.get("video_duration_s"),
            "shard_path": master_row.get("shard_path"),
            "keylog_path": mrow.get("keylog_path"),
            "alignment_status": mrow.get("alignment_status"),
            "params": params,
            "kept_ranges": kept_ranges,
            "dropped": dropped,
            "n_kept": n_records - n_black - n_idle,
            "n_black": n_black,
            "n_idle_interior": n_idle,
        }
        write_json(out_path, filter_seg_doc)
        if task.get("qc_view_fps"):
            _write_qc_view(Path(task["qc_dir"]), filter_seg_doc, float(task["qc_view_fps"]))
        return {
            **base,
            "status": "ok",
            "n_records": n_records,
            "n_kept": filter_seg_doc["n_kept"],
            "n_black": n_black,
            "n_idle_interior": n_idle,
        }
    except Exception as exc:
        return {**base, "status": "failed", "error": f"{exc}", "traceback": traceback.format_exc()}


def _load_master_fps(master_dir: Path, index_rows: list[dict[str, Any]]) -> float:
    """The store's master fps == the tick rate of the axis being masked."""
    summary_path = master_dir / "frames_master_summary.json"
    if summary_path.is_file():
        try:
            return float(json.loads(summary_path.read_text())["master_fps"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    seen = {float(r["master_fps"]) for r in index_rows if r.get("master_fps") is not None}
    if not seen:
        raise RuntimeError(f"cannot determine master_fps of {master_dir}")
    return max(seen)


def parse_args() -> argparse.Namespace:
    normalize_dashed_argv()  # accept pmanager's --foo_bar=value arg form
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-master-dir", type=Path, required=True,
                   help="Stage 01 output: segment_index.jsonl + frames/<seg>/images.array_record.")
    p.add_argument("--clips-manifest", type=Path, required=True,
                   help="Realigned clips_manifest.jsonl (stage 00/02): dataset definition + "
                        "corrected keylog_path + alignment_status, joined by segment_id.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--drop-black-frames", nargs="?", const=True, type=_str2bool,
                   default=config.DEFAULT_DROP_BLACK_FRAMES, metavar="BOOL",
                   help="Mask (near-)black master ticks (luma metrics from stage 01).")
    p.add_argument("--black-luma-max", type=float, default=config.DEFAULT_BLACK_LUMA_MAX,
                   help="Black if mean luma (0-255) is <= this.")
    p.add_argument("--black-dark-frac-min", type=float, default=config.DEFAULT_BLACK_DARK_FRAC_MIN,
                   help="...or if this fraction of pixels are near-black.")
    p.add_argument("--idle-min-duration-s", type=float, default=config.DEFAULT_IDLE_MIN_DURATION_S,
                   help="Thin the interior of inactive runs LONGER than this many seconds.")
    p.add_argument("--idle-keep-head-s", type=float, default=config.DEFAULT_IDLE_KEEP_HEAD_S,
                   help="Seconds kept at the start of each thinned idle run.")
    p.add_argument("--idle-keep-tail-s", type=float, default=config.DEFAULT_IDLE_KEEP_TAIL_S,
                   help="Seconds kept at the end of each thinned idle run.")
    p.add_argument("--idle-judgment-bin-s", type=float, default=config.DEFAULT_IDLE_JUDGMENT_BIN_S,
                   help="Judge idleness at this granularity (seconds). Default 2 s = the "
                        "default judgment bin. Pass 0 for per-master-"
                        "tick judgment (raw mode only).")
    p.add_argument("--idle-activity", choices=config.IDLE_ACTIVITIES,
                   default=config.DEFAULT_IDLE_ACTIVITY,
                   help="What counts as activity. 'rounded' (default) = the NO_OP "
                        "predicate per judgment bin — the bin's FORMATTED action is non-NO_OP "
                        "(deltas round to nonzero, or deduped key events survive); with the "
                        "min-duration/head/tail 0 it keeps every frame at the judgment "
                        "fps. 'raw' = any nonzero-delta or key event "
                        "(sub-pixel drift is activity; fps-agnostic).")
    p.add_argument("--qc-view-fps", type=float, default=None,
                   help="Also emit qc_view/<seg>.jsonl: a sampled view at this fps with derived "
                        "canonical action strings, for realignment eyeballing (must divide the "
                        "master fps; nothing downstream reads it).")
    p.add_argument("--num-workers", type=int, default=0, help="0 = cpu_count().")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N segments (debug).")
    p.add_argument("--force", action="store_true", help="Re-judge segments even if same-params output exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        params = filter_params(**{k: getattr(args, k) for k in FILTER_PARAM_KEYS})
    except ValueError as exc:
        raise SystemExit(f"bad filter parameters: {exc}") from exc

    master_dir = args.frames_master_dir
    index_path = master_dir / "segment_index.jsonl"
    if not index_path.is_file():
        raise SystemExit(f"no segment_index.jsonl under {master_dir} (is it a frames-master artifact?)")
    if not (master_dir / "manifest.json").is_file():
        raise SystemExit(f"no manifest.json under {master_dir}: cannot fingerprint the master store")
    index_rows = read_jsonl(index_path)
    if not index_rows:
        raise RuntimeError(f"Empty segment index: {index_path}")
    master_by_seg = {str(r["segment_id"]): r for r in index_rows}
    master_fps = _load_master_fps(master_dir, index_rows)

    if args.qc_view_fps is not None:
        resolve_stride(master_fps, args.qc_view_fps)  # fail fast on bad ratios

    manifest_rows = read_jsonl(args.clips_manifest)
    if not manifest_rows:
        raise RuntimeError(f"Empty clips manifest: {args.clips_manifest}")
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]

    out_dir = ensure_dir(args.output_dir)
    filter_dir = ensure_dir(out_dir / "filter")
    qc_dir = ensure_dir(out_dir / "qc_view") if args.qc_view_fps is not None else None

    tasks = [
        {
            "manifest_row": row,
            "master_row": master_by_seg.get(str(row["segment_id"])),
            "filter_dir": str(filter_dir),
            "frames_dir": str(master_dir / "frames"),
            "master_fps": master_fps,
            "drop_black_frames": args.drop_black_frames,
            "black_luma_max": args.black_luma_max,
            "black_dark_frac_min": args.black_dark_frac_min,
            "idle_min_duration_s": args.idle_min_duration_s,
            "idle_keep_head_s": args.idle_keep_head_s,
            "idle_keep_tail_s": args.idle_keep_tail_s,
            "idle_judgment_bin_s": args.idle_judgment_bin_s,
            "idle_activity": args.idle_activity,
            "qc_view_fps": args.qc_view_fps,
            "qc_dir": str(qc_dir) if qc_dir else None,
            "force": args.force,
        }
        for row in manifest_rows
    ]

    n_workers = max(1, min(args.num_workers or mp.cpu_count(), len(tasks)))
    print(
        f"[filter] {len(tasks)} segments | master_fps={master_fps} "
        f"black={'on' if args.drop_black_frames else 'off'}"
        f"(luma<={args.black_luma_max},dark>={args.black_dark_frac_min}) "
        f"idle(min>{args.idle_min_duration_s}s,head={args.idle_keep_head_s}s,"
        f"tail={args.idle_keep_tail_s}s) | workers={n_workers}",
        flush=True,
    )

    counts: Counter = Counter()
    index_out: list[dict[str, Any]] = []
    totals = Counter()
    with mp.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(filter_segment, tasks, chunksize=8), 1):
            counts[res["status"]] += 1
            for key in ("n_records", "n_kept", "n_black", "n_idle_interior"):
                totals[key] += int(res.get(key) or 0)
            index_out.append(res)
            if res["status"] == "failed":
                print(f"  FAIL {res['segment_id']}: {res.get('error')}", flush=True)
            if i % 1000 == 0:
                print(f"  {i}/{len(tasks)} segments | {dict(counts)}", flush=True)

    index_out.sort(key=lambda r: str(r["segment_id"]))
    write_jsonl(out_dir / "filter_index.jsonl", index_out)

    recorded_params = {**params, "qc_view_fps": args.qc_view_fps}
    summary = {
        "master_fps": master_fps,
        "n_segments": len(tasks),
        "status_counts": dict(counts),
        "n_records_total": totals["n_records"],
        "n_kept_total": totals["n_kept"],
        "n_black_total": totals["n_black"],
        "n_idle_interior_total": totals["n_idle_interior"],
        "frames_master_dir": str(master_dir),
        "source_clips_manifest": str(args.clips_manifest),
        **recorded_params,
    }
    write_json(out_dir / "filter_summary.json", summary)
    write_json(out_dir / "manifest.json", {
        "artifact_type": "realigned_filter_mask",
        "schema_version": 1,
        "master_fps": master_fps,
        "master_store_id": make_artifact_id(master_dir),
        "filter_index": "filter_index.jsonl",
        "filter_layout": "filter/<segment_id>.json",
        "params": recorded_params,
        **summary,
    })
    print(
        f"[filter] done: {dict(counts)} | kept {totals['n_kept']}/{totals['n_records']} ticks "
        f"({totals['n_black']} black, {totals['n_idle_interior']} idle-interior) -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
