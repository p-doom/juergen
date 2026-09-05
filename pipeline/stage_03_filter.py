"""Build the canonical Crowd-Cast frame keep/drop mask."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.lib import config
from pipeline.lib.action_format import format_segment
from pipeline.lib.common import (
    EVENT_EXCLUSION_REASONS,
    KeylogError,
    ensure_dir,
    read_jsonl,
    write_json,
    write_json_atomic,
    write_jsonl,
)
from pipeline.lib.events import Window, load_events
from pipeline.lib.manifest import file_sha256_short, make_artifact_id
from pipeline.lib.master_frames import resolve_master_artifact
from pipeline.lib.realign import (
    CLOSED_STATUSES,
    CLOSURE_TOL,
    EXCLUSION_REASONS,
    IDLE_TIMEOUT,
)

_STAGE02_SUMMARY_FIELDS = {
    "closure_tol_s",
    "exclusion_counts",
    "idle_timeout_s",
    "n_accepted_segments",
    "n_corrected",
    "n_excluded_segments",
    "n_keylogs_repointed",
    "n_recordings",
    "n_source_segments",
    "source_clips_id",
    "source_clips_manifest",
    "source_clips_sha256",
    "status_counts",
}
_STAGE02_MANIFEST_FIELDS = _STAGE02_SUMMARY_FIELDS | {
    "alignment_file",
    "alignment_sha256",
    "artifact_type",
    "clips_file",
    "clips_sha256",
    "schema_version",
}
_ACCEPTED_ALIGNMENT_FIELDS = {
    "closed",
    "corrected_keylog_path",
    "corrected_keylog_sha256",
    "corr_end_s",
    "disposition",
    "exclusion_reason",
    "keylog_span_s",
    "leading_method",
    "model",
    "n_pauses",
    "overhang_s",
    "recording_id",
    "residual_s",
    "segment_id",
    "segment_idx",
    "splices",
    "status",
    "total_collapse_s",
    "video_dur_s",
}
_EXCLUDED_ALIGNMENT_FIELDS = {
    "candidates",
    "closed",
    "disposition",
    "exclusion_reason",
    "recording_id",
    "segment_id",
    "segment_idx",
}

REASON_KEPT = 0
REASON_BLACK = 1
REASON_IDLE = 2
_REASON_NAMES = {REASON_BLACK: "black", REASON_IDLE: "idle_interior"}

FILTER_PARAMS = {
    "drop_black_frames": True,
    "black_luma_max": config.DEFAULT_BLACK_LUMA_MAX,
    "black_dark_frac_min": config.DEFAULT_BLACK_DARK_FRAC_MIN,
    "idle_min_duration_s": config.DEFAULT_IDLE_MIN_DURATION_S,
    "idle_keep_head_s": config.DEFAULT_IDLE_KEEP_HEAD_S,
    "idle_keep_tail_s": config.DEFAULT_IDLE_KEEP_TAIL_S,
    "idle_judgment_bin_s": config.DEFAULT_IDLE_JUDGMENT_BIN_S,
    "idle_activity": "canonical_deltatype_v2",
}


def _master_frame_manifest(master_row: dict[str, Any]) -> Path:
    return Path(master_row["shard_path"]).parent / "frame_manifest.jsonl"


def _is_black(record: dict[str, Any]) -> bool:
    mean_luma = record.get("mean_luma")
    dark_fraction = record.get("frac_dark")
    if not isinstance(mean_luma, (int, float)) or not isinstance(
        dark_fraction, (int, float)
    ):
        raise TypeError("master frame has no luma metrics")
    return (
        mean_luma <= config.DEFAULT_BLACK_LUMA_MAX
        or dark_fraction >= config.DEFAULT_BLACK_DARK_FRAC_MIN
    )


def _rounded_activity_mask(
    keylog_path: Path,
    n_records: int,
    master_fps: float,
    bin_ticks: int,
) -> list[bool]:
    if not keylog_path.is_file():
        raise FileNotFoundError(f"Crowd-Cast keylog is missing: {keylog_path}")
    if bin_ticks <= 0:
        raise ValueError("idle judgment bin must contain at least one master tick")
    windows = [
        Window(start, start, min(start + bin_ticks, n_records))
        for start in range(0, n_records, bin_ticks)
    ]
    labels = format_segment(
        load_events(keylog_path), windows, (), master_fps=master_fps
    ).labels
    active = [False] * n_records
    for index, label in enumerate(labels):
        if label != "NO_OP":
            start = index * bin_ticks
            active[start : min(start + bin_ticks, n_records)] = [True] * min(
                bin_ticks, n_records - start
            )
    return active


def _idle_interiors(
    active: list[bool],
    master_fps: float,
    min_duration_s: float,
    keep_head_s: float,
    keep_tail_s: float,
) -> list[tuple[int, int]]:
    min_ticks = round(min_duration_s * master_fps)
    head = round(keep_head_s * master_fps)
    tail = round(keep_tail_s * master_fps)
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(active):
        if active[index]:
            index += 1
            continue
        end = index
        while end < len(active) and not active[end]:
            end += 1
        if end - index > min_ticks and index + head < end - tail:
            spans.append((index + head, end - tail))
        index = end
    return spans


def _compress_reasons(
    reasons: list[int],
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    kept: list[list[int]] = []
    dropped: list[dict[str, Any]] = []
    index = 0
    while index < len(reasons):
        end = index
        while end < len(reasons) and reasons[end] == reasons[index]:
            end += 1
        if reasons[index] == REASON_KEPT:
            kept.append([index, end])
        else:
            dropped.append(
                {"start": index, "end": end, "reason": _REASON_NAMES[reasons[index]]}
            )
        index = end
    return kept, dropped


def filter_segment(task: dict[str, Any]) -> dict[str, Any]:
    manifest_row = task["manifest_row"]
    master_row = task["master_row"]
    segment_id = str(manifest_row["segment_id"])
    if master_row["status"] != "ok":
        raise ValueError(
            f"master segment {segment_id} is not complete: {master_row['status']!r}"
        )
    if (
        manifest_row.get("alignment_closed") is not True
        or manifest_row.get("alignment_status") not in CLOSED_STATUSES
    ):
        raise ValueError(f"segment {segment_id} has no closed alignment")
    master_fps = float(master_row["master_fps"])
    master_manifest = read_jsonl(_master_frame_manifest(master_row))
    if not master_manifest:
        raise ValueError(f"master segment has no frames: {segment_id}")
    keylog = Path(manifest_row["keylog_path"])
    if not keylog.is_file():
        raise FileNotFoundError(f"Crowd-Cast keylog is missing: {keylog}")
    if file_sha256_short(keylog, n=64) != manifest_row.get("keylog_sha256"):
        raise ValueError(f"keylog digest mismatch: {keylog}")
    try:
        active = _rounded_activity_mask(
            keylog,
            len(master_manifest),
            master_fps,
            round(config.DEFAULT_IDLE_JUDGMENT_BIN_S * master_fps),
        )
    except KeylogError as exc:
        if exc.reason not in EVENT_EXCLUSION_REASONS:
            raise
        return {
            "segment_id": segment_id,
            "recording_id": manifest_row["recording_id"],
            "segment_idx": manifest_row["segment_idx"],
            "alignment_status": manifest_row["alignment_status"],
            "keylog_path": str(keylog),
            "keylog_sha256": manifest_row["keylog_sha256"],
            "filter_path": None,
            "filter_sha256": None,
            "status": "excluded_invalid_keylog",
            "exclusion_reason": exc.reason,
            "n_records": len(master_manifest),
            "n_kept": 0,
            "n_black": 0,
            "n_idle_interior": 0,
        }
    reasons = [REASON_KEPT] * len(master_manifest)
    for start, end in _idle_interiors(
        active,
        master_fps,
        config.DEFAULT_IDLE_MIN_DURATION_S,
        config.DEFAULT_IDLE_KEEP_HEAD_S,
        config.DEFAULT_IDLE_KEEP_TAIL_S,
    ):
        reasons[start:end] = [REASON_IDLE] * (end - start)
    for index, record in enumerate(master_manifest):
        if _is_black(record):
            reasons[index] = REASON_BLACK

    kept_ranges, dropped = _compress_reasons(reasons)
    n_black = sum(
        item["end"] - item["start"] for item in dropped if item["reason"] == "black"
    )
    n_idle = sum(
        item["end"] - item["start"]
        for item in dropped
        if item["reason"] == "idle_interior"
    )
    n_kept = len(master_manifest) - n_black - n_idle
    if n_kept <= 0:
        raise ValueError(f"Crowd-Cast filter retained no frames: {segment_id}")
    output = Path(task["filter_dir"]) / f"{segment_id}.json"
    write_json(
        output,
        {
            "segment_id": segment_id,
            "recording_id": manifest_row["recording_id"],
            "segment_idx": manifest_row["segment_idx"],
            "master_fps": master_fps,
            "n_master_records": len(master_manifest),
            "video_duration_s": manifest_row["video_duration_s"],
            "shard_path": master_row["shard_path"],
            "shard_sha256": master_row["shard_sha256"],
            "frame_manifest_sha256": master_row["frame_manifest_sha256"],
            "keylog_path": str(keylog),
            "keylog_sha256": manifest_row["keylog_sha256"],
            "alignment_status": manifest_row["alignment_status"],
            "params": FILTER_PARAMS,
            "kept_ranges": kept_ranges,
            "dropped": dropped,
            "n_kept": n_kept,
            "n_black": n_black,
            "n_idle_interior": n_idle,
        },
    )
    return {
        "segment_id": segment_id,
        "recording_id": manifest_row["recording_id"],
        "segment_idx": manifest_row["segment_idx"],
        "alignment_status": manifest_row["alignment_status"],
        "filter_path": str(output),
        "filter_sha256": file_sha256_short(output, n=64),
        "status": "ok",
        "n_records": len(master_manifest),
        "n_kept": n_kept,
        "n_black": n_black,
        "n_idle_interior": n_idle,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_master_dir", type=Path, required=True)
    parser.add_argument("--clips_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count())
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output = ensure_dir(args.output_dir)
    (output / "manifest.json").unlink(missing_ok=True)
    if args.num_workers <= 0:
        raise SystemExit("--num_workers must be positive")
    master, index_rows = resolve_master_artifact(args.frames_master_dir)

    clips_artifact_manifest = args.clips_manifest.parent / "manifest.json"
    clips_artifact = json.loads(clips_artifact_manifest.read_text(encoding="utf-8"))
    required_clips = {
        "artifact_type": "juergen_annotation_clip_manifest_realigned",
        "schema_version": 2,
        "clips_file": "clips_manifest.jsonl",
        "alignment_file": "alignment.jsonl",
        "idle_timeout_s": IDLE_TIMEOUT,
        "closure_tol_s": CLOSURE_TOL,
    }
    if (
        set(clips_artifact) != _STAGE02_MANIFEST_FIELDS
        or {key: clips_artifact.get(key) for key in required_clips} != required_clips
    ):
        raise ValueError(
            f"Crowd-Cast clips contract mismatch: {clips_artifact_manifest}"
        )
    summary_path = args.clips_manifest.parent / "realign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary != {key: clips_artifact[key] for key in _STAGE02_SUMMARY_FIELDS}:
        raise ValueError(f"Crowd-Cast realignment summary mismatch: {summary_path}")
    if (
        args.clips_manifest.resolve()
        != (args.clips_manifest.parent / clips_artifact["clips_file"]).resolve()
    ):
        raise ValueError("--clips_manifest must be the canonical Stage02 clips file")
    if file_sha256_short(args.clips_manifest, n=64) != clips_artifact.get(
        "clips_sha256"
    ):
        raise ValueError(f"Crowd-Cast clips digest mismatch: {args.clips_manifest}")
    alignment_path = args.clips_manifest.parent / clips_artifact["alignment_file"]
    if file_sha256_short(alignment_path, n=64) != clips_artifact.get(
        "alignment_sha256"
    ):
        raise ValueError(f"Crowd-Cast alignment digest mismatch: {alignment_path}")
    if master.get("source_clips_sha256") != clips_artifact.get(
        "source_clips_sha256"
    ) or master.get("source_clips_id") != clips_artifact.get("source_clips_id"):
        raise ValueError("Crowd-Cast Stage01 and Stage02 source inventories differ")
    manifest_rows = read_jsonl(args.clips_manifest)
    alignment_rows = read_jsonl(alignment_path)
    if not index_rows or not manifest_rows:
        raise ValueError("Crowd-Cast master and clips artifacts must be non-empty")
    master_by_segment = {str(row["segment_id"]): row for row in index_rows}
    manifest_by_segment = {str(row["segment_id"]): row for row in manifest_rows}
    alignment_by_segment = {str(row["segment_id"]): row for row in alignment_rows}
    if len(master_by_segment) != len(index_rows):
        raise ValueError("Crowd-Cast master contains duplicate segments")
    if len(manifest_by_segment) != len(manifest_rows):
        raise ValueError("Crowd-Cast clips contain duplicate segments")
    if len(alignment_by_segment) != len(alignment_rows):
        raise ValueError("Crowd-Cast alignment contains duplicate segments")
    accepted_alignment = {
        segment_id
        for segment_id, row in alignment_by_segment.items()
        if row.get("disposition") == "accepted"
    }
    excluded_alignment = set(alignment_by_segment) - accepted_alignment
    if (
        set(alignment_by_segment) != set(master_by_segment)
        or set(manifest_by_segment) != accepted_alignment
    ):
        raise ValueError(
            "Crowd-Cast Stage02 must partition the Stage01 master segment set"
        )
    for segment_id in accepted_alignment:
        alignment = alignment_by_segment[segment_id]
        clip = manifest_by_segment[segment_id]
        master_row = master_by_segment[segment_id]
        if (
            set(alignment) != _ACCEPTED_ALIGNMENT_FIELDS
            or alignment.get("disposition") != "accepted"
            or alignment.get("closed") is not True
            or alignment.get("status") not in CLOSED_STATUSES
            or alignment.get("exclusion_reason") is not None
            or clip.get("alignment_closed") is not True
            or clip.get("alignment_status") != alignment["status"]
            or alignment.get("recording_id") != master_row.get("recording_id")
            or alignment.get("segment_idx") != master_row.get("segment_idx")
            or clip.get("recording_id") != master_row.get("recording_id")
            or clip.get("segment_idx") != master_row.get("segment_idx")
        ):
            raise ValueError(f"invalid accepted Stage02 segment: {segment_id}")
    for segment_id in excluded_alignment:
        alignment = alignment_by_segment[segment_id]
        master_row = master_by_segment[segment_id]
        if (
            set(alignment) != _EXCLUDED_ALIGNMENT_FIELDS
            or alignment.get("disposition") != "excluded"
            or alignment.get("closed") is not False
            or alignment.get("exclusion_reason") not in EXCLUSION_REASONS
            or alignment.get("recording_id") != master_row.get("recording_id")
            or alignment.get("segment_idx") != master_row.get("segment_idx")
        ):
            raise ValueError(f"invalid excluded Stage02 segment: {segment_id}")
    observed_statuses = Counter(
        alignment_by_segment[segment_id]["status"] for segment_id in accepted_alignment
    )
    observed_exclusions = Counter(
        alignment_by_segment[segment_id]["exclusion_reason"]
        for segment_id in excluded_alignment
    )
    if (
        clips_artifact.get("n_source_segments") != len(master_by_segment)
        or clips_artifact.get("n_accepted_segments") != len(accepted_alignment)
        or clips_artifact.get("n_excluded_segments") != len(excluded_alignment)
        or clips_artifact.get("status_counts")
        != dict(sorted(observed_statuses.items()))
        or clips_artifact.get("exclusion_counts")
        != dict(sorted(observed_exclusions.items()))
    ):
        raise ValueError("Crowd-Cast Stage02 alignment counts do not close")
    filter_dir = ensure_dir(output / "filter")
    tasks = [
        {
            "manifest_row": manifest_by_segment[segment_id],
            "master_row": master_by_segment[segment_id],
            "filter_dir": str(filter_dir),
        }
        for segment_id in sorted(manifest_by_segment)
    ]
    workers = min(args.num_workers, len(tasks))
    if workers == 1:
        results = list(map(filter_segment, tasks))
    else:
        with mp.Pool(workers) as pool:
            results = list(pool.imap_unordered(filter_segment, tasks, chunksize=8))
    results.sort(key=lambda row: row["segment_id"])
    filter_index_path = output / "filter_index.jsonl"
    write_jsonl(filter_index_path, results)
    totals = Counter()
    for result in results:
        for key in ("n_records", "n_kept", "n_black", "n_idle_interior"):
            totals[key] += int(result[key])
    status_counts = Counter(result["status"] for result in results)
    exclusion_counts = Counter(
        result["exclusion_reason"]
        for result in results
        if result["status"] == "excluded_invalid_keylog"
    )
    summary = {
        "master_fps": float(master["master_fps"]),
        "n_segments": len(results),
        "n_master_segments": len(master_by_segment),
        "n_input_segments": len(manifest_by_segment),
        "n_alignment_excluded_segments": len(excluded_alignment),
        "n_accepted_segments": status_counts["ok"],
        "n_excluded_segments": status_counts["excluded_invalid_keylog"],
        "status_counts": dict(sorted(status_counts.items())),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "n_records_total": totals["n_records"],
        "n_kept_total": totals["n_kept"],
        "n_black_total": totals["n_black"],
        "n_idle_interior_total": totals["n_idle_interior"],
        "frames_master_dir": str(args.frames_master_dir.resolve()),
        "source_clips_manifest": str(args.clips_manifest.resolve()),
        **FILTER_PARAMS,
    }
    write_json(output / "filter_summary.json", summary)
    write_json_atomic(
        output / "manifest.json",
        {
            "artifact_type": "realigned_filter_mask",
            "schema_version": 1,
            "master_fps": summary["master_fps"],
            "master_store_id": make_artifact_id(args.frames_master_dir),
            "source_clips_id": make_artifact_id(args.clips_manifest.parent),
            "filter_index": "filter_index.jsonl",
            "filter_index_sha256": file_sha256_short(filter_index_path, n=64),
            "filter_layout": "filter/<segment_id>.json",
            "params": FILTER_PARAMS,
            **summary,
        },
    )


if __name__ == "__main__":
    main()
