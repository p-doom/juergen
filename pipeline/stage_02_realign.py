"""Realign the complete attested Crowd-Cast source inventory."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp

# Make the ``pipeline`` package importable when this stage is run
# directly as a script (mirrors the other stages' PYTHONPATH setup).
import sys
from collections import Counter, defaultdict
from pathlib import Path

import msgpack

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.lib import realign as R
from pipeline.lib.common import ensure_dir, write_json, write_json_atomic
from pipeline.lib.manifest import file_sha256_short
from pipeline.lib.source_clips import resolve_source_clips


def write_corrected_keylog(
    src_keylog: Path, splices: list[dict], out_path: Path
) -> None:
    """Re-stamp every keylog event ts (us) keylog-clock -> video-PTS. Lossless
    structure; only timestamps change."""
    corrected = []
    for entry in R.load_keylog(str(src_keylog)):
        kt = entry[0] / 1e6
        corrected.append([round(R.keylog_to_video(kt, splices) * 1e6), entry[1]])
    ensure_dir(out_path.parent)
    out_path.write_bytes(msgpack.packb(corrected, use_bin_type=True))


def realign_one_recording(task: dict) -> dict:
    """Worker: realign one recording. Writes corrected keylogs for its
    dataset-kept segments that have splices; returns per-segment alignment rows
    (with the corrected keylog path, if any). Picklable; no shared state."""
    rec_id = task["recording_id"]
    segs = task["segments"]
    kept = {
        segment["segment_id"]: {
            "src_keylog": segment["keylog_path"],
        }
        for segment in segs
    }
    results = R.realign_recording(segs)
    out_dir = Path(task["out_dir"])

    rows: list[dict] = []
    for sid, res in results.items():
        if res["closed"] is not True:
            rows.append(
                {
                    "segment_id": sid,
                    "recording_id": rec_id,
                    "segment_idx": res["segment_idx"],
                    "disposition": "excluded",
                    "closed": False,
                    "exclusion_reason": res["exclusion_reason"],
                    "candidates": res["candidates"],
                }
            )
            continue
        corrected_path = None
        if res["splices"]:
            corrected_path = out_dir / "corrected_keylogs" / f"{sid}.msgpack"
            write_corrected_keylog(
                Path(kept[sid]["src_keylog"]), res["splices"], corrected_path
            )
        rows.append(
            {
                "segment_id": sid,
                "recording_id": rec_id,
                "segment_idx": res["segment_idx"],
                "disposition": "accepted",
                "status": res["status"],
                "closed": res["closed"],
                "exclusion_reason": None,
                "model": res["model"],
                "leading_method": res["leading_method"],
                "n_pauses": res["n_pauses"],
                "total_collapse_s": round(res["total_collapse_s"], 6),
                "overhang_s": round(res["overhang_s"], 6),
                "residual_s": round(res["residual_s"], 6),
                "corr_end_s": round(res["corr_end_s"], 6),
                "keylog_span_s": round(res["keylog_span_s"], 6),
                "video_dur_s": res["video_dur_s"],
                "corrected_keylog_path": str(corrected_path)
                if corrected_path
                else None,
                "corrected_keylog_sha256": (
                    file_sha256_short(corrected_path, n=64) if corrected_path else None
                ),
                "splices": [
                    {
                        "kp": round(s["kp"], 6),
                        "vp": round(s["vp"], 6),
                        "collapse": round(s["collapse"], 6),
                        "leading": s["leading"],
                    }
                    for s in res["splices"]
                ],
            }
        )
    return {"rows": rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--clips-manifest",
        type=Path,
        required=True,
        help="discover output clips_manifest.jsonl.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-workers", type=int, default=0, help="0 = cpu_count().")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    (out_dir / "manifest.json").unlink(missing_ok=True)

    clip_rows, source = resolve_source_clips(args.clips_manifest)
    segment_ids = [str(row["segment_id"]) for row in clip_rows]
    for row in clip_rows:
        video = Path(row["video_path"])
        keylog = Path(row["keylog_path"])
        if file_sha256_short(video, n=64) != row.get("video_sha256"):
            raise ValueError(f"video digest mismatch: {video}")
        if file_sha256_short(keylog, n=64) != row.get("keylog_sha256"):
            raise ValueError(f"keylog digest mismatch: {keylog}")

    by_rec: dict[str, list[dict]] = defaultdict(list)
    for r in clip_rows:
        by_rec[r["recording_id"]].append(r)

    tasks = [
        {
            "recording_id": rec_id,
            "segments": [
                {
                    "segment_id": row["segment_id"],
                    "segment_idx": row["segment_idx"],
                    "keylog_path": row["keylog_path"],
                    "video_path": row["video_path"],
                    "video_dur_s": row["video_duration_s"],
                }
                for row in sorted(kept, key=lambda item: item["segment_idx"])
            ],
            "out_dir": str(out_dir),
        }
        for rec_id, kept in by_rec.items()
    ]

    n_workers = args.num_workers or mp.cpu_count()
    n_workers = min(n_workers, len(tasks))
    status_counts: Counter = Counter()
    exclusion_counts: Counter = Counter()
    align_by_sid: dict[str, dict] = {}
    with mp.Pool(n_workers) as pool:
        for i, r in enumerate(
            pool.imap_unordered(realign_one_recording, tasks, chunksize=8), 1
        ):
            for row in r["rows"]:
                if row["segment_id"] in align_by_sid:
                    raise ValueError(f"duplicate alignment row: {row['segment_id']}")
                if row["disposition"] == "accepted":
                    status_counts[row["status"]] += 1
                elif row["disposition"] == "excluded":
                    exclusion_counts[row["exclusion_reason"]] += 1
                else:
                    raise ValueError(f"invalid alignment disposition: {row!r}")
                align_by_sid[row["segment_id"]] = row
            if i % 1000 == 0:
                print(f"  {i}/{len(tasks)} recordings", flush=True)

    if set(align_by_sid) != set(segment_ids):
        raise ValueError("alignment certificate set does not match clip manifest")
    for row in align_by_sid.values():
        if row["disposition"] == "accepted":
            if (
                row["closed"] is not True
                or row["status"] not in R.CLOSED_STATUSES
                or row["exclusion_reason"] is not None
            ):
                raise ValueError(f"invalid accepted alignment row: {row!r}")
        elif (
            row["closed"] is not False
            or row["exclusion_reason"] not in R.EXCLUSION_REASONS
        ):
            raise ValueError(f"invalid excluded alignment row: {row!r}")

    alignment_path = out_dir / "alignment.jsonl"
    with alignment_path.open("w") as f:
        for sid in sorted(align_by_sid):
            f.write(json.dumps(align_by_sid[sid]) + "\n")

    # realigned clips_manifest.jsonl: discover rows + alignment status, keylog_path
    # repointed at the corrected keylog for corrected segments (frames consumes this).
    n_repointed = 0
    clips_path = out_dir / "clips_manifest.jsonl"
    with clips_path.open("w") as f:
        for r in clip_rows:
            a = align_by_sid.get(r["segment_id"])
            if a["disposition"] == "excluded":
                continue
            row = dict(r)
            row["alignment_status"] = a["status"]
            row["alignment_closed"] = a["closed"]
            row["alignment_total_collapse_s"] = a["total_collapse_s"]
            row["alignment_residual_s"] = a["residual_s"]
            row["raw_keylog_path"] = r["keylog_path"]
            row["raw_keylog_sha256"] = r["keylog_sha256"]
            if a["corrected_keylog_path"]:
                row["keylog_path"] = a["corrected_keylog_path"]
                row["keylog_sha256"] = a["corrected_keylog_sha256"]
                n_repointed += 1
            f.write(json.dumps(row) + "\n")

    n_source = len(align_by_sid)
    n_accepted = sum(row["disposition"] == "accepted" for row in align_by_sid.values())
    n_excluded = n_source - n_accepted
    if n_accepted == 0:
        raise ValueError("realignment excluded every Crowd-Cast source segment")
    summary = {
        "n_source_segments": n_source,
        "n_accepted_segments": n_accepted,
        "n_excluded_segments": n_excluded,
        "n_recordings": len(by_rec),
        "idle_timeout_s": R.IDLE_TIMEOUT,
        "closure_tol_s": R.CLOSURE_TOL,
        "status_counts": dict(sorted(status_counts.items())),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "n_corrected": n_accepted - status_counts.get("aligned", 0),
        "n_keylogs_repointed": n_repointed,
        "source_clips_manifest": source["path"],
        "source_clips_sha256": source["sha256"],
        "source_clips_id": source["artifact_id"],
    }
    write_json(out_dir / "realign_summary.json", summary)
    write_json_atomic(
        out_dir / "manifest.json",
        {
            "artifact_type": "juergen_annotation_clip_manifest_realigned",
            "schema_version": 2,
            "clips_file": "clips_manifest.jsonl",
            "clips_sha256": file_sha256_short(clips_path, n=64),
            "alignment_file": "alignment.jsonl",
            "alignment_sha256": file_sha256_short(alignment_path, n=64),
            **summary,
        },
    )
    print(
        f"Accepted {n_accepted}/{n_source} segments; "
        f"repointed {n_repointed} keylogs -> {out_dir}"
    )
    print(f"status: {dict(status_counts)}; excluded: {dict(exclusion_counts)}")


if __name__ == "__main__":
    main()
