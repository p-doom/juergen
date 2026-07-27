"""VideoCUA ingest (stage A) -- turn an unzipped ServiceNow/VideoCUA app tree
into the two inputs the rest of this side-pipeline needs:

  1. ``clips_manifest.jsonl`` -- one row per task, in the schema
     ``realigned_pipeline/stage_01_master_frames.py`` consumes (segment_id /
     video_path / video_ok / video_fps / video_duration_s / video_frame_count).
     Feeding this to stage_01 at ``--master-fps 15`` builds the same 15fps JPEG
     ArrayRecord master store we use for ccast -- decode once, subsample later.

  2. ``tasks.jsonl`` -- one self-contained row per task carrying the goal
     (``task_instruction``) and the normalized action timeline, so the thinking
     annotator (stage B) never has to re-read the raw VideoCUA tree.

VideoCUA layout (per task folder ``<task_id>/``):
    action_log.json           {task_id, task_instruction, platform, action_log[]}
    video/video.mp4           1920x1080 @ 30fps
    video/video_metadata.json {fps, total_frames, duration_seconds, width, height,
                               can_be_loaded, can_be_played, error}

This stage does NOT decode video and does NOT touch the crowd-cast annotation
methods -- it is pure metadata projection.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.lib.common import ensure_dir, write_json  # noqa: E402


def _slug(text: str) -> str:
    """A filesystem/id-safe slug for a platform name (e.g. 'OnlyOffice Forms')."""
    return re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_") or "app"


def _normalize_action(a: dict[str, Any]) -> dict[str, Any]:
    """Flatten one VideoCUA action to {t_s, type, params, groundcua_id}."""
    params = a.get("action_params") or {}
    return {
        "t_s": float(a.get("timestamp") or 0.0),
        "type": str(a.get("action_type") or "").upper(),
        "params": params,
        "groundcua_id": a.get("groundcua_id"),
    }


def _video_ok(meta: dict[str, Any]) -> bool:
    """A task is decodable iff the metadata says so and carries a sane duration
    and fps. Guards the observed corrupt-sentinel duration (INT64_MIN)."""
    try:
        dur = float(meta.get("duration_seconds") or 0.0)
        fps = float(meta.get("fps") or 0.0)
    except (TypeError, ValueError):
        return False
    return bool(
        meta.get("can_be_loaded")
        and meta.get("can_be_played")
        and meta.get("error") in (None, "")
        and 0.0 < dur < 24 * 3600
        and 0.0 < fps <= 240
    )


def iter_task_dirs(raw_dir: Path) -> list[Path]:
    """Every task folder under ``raw_dir`` (a folder containing action_log.json
    with a sibling video/video.mp4). Works whether raw_dir is one app dir or a
    parent of several unzipped app dirs."""
    out = []
    for al in sorted(raw_dir.rglob("action_log.json")):
        if (al.parent / "video" / "video.mp4").is_file():
            out.append(al.parent)
    return out


def build_row(task_dir: Path) -> dict[str, Any] | None:
    al_path = task_dir / "action_log.json"
    meta_path = task_dir / "video" / "video_metadata.json"
    video_path = task_dir / "video" / "video.mp4"
    try:
        al = json.loads(al_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    meta = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}

    platform = str(al.get("platform") or task_dir.parent.name)
    task_id = al.get("task_id")
    if task_id is None:
        task_id = task_dir.name
    segment_id = f"{_slug(platform)}__{task_id}"

    actions = [_normalize_action(a) for a in (al.get("action_log") or [])]
    actions.sort(key=lambda a: a["t_s"])

    return {
        "segment_id": segment_id,
        "recording_id": segment_id,
        "segment_idx": 0,
        "video_path": str(video_path.resolve()),
        "video_ok": _video_ok(meta),
        "video_fps": float(meta.get("fps") or 0.0),
        "video_duration_s": float(meta.get("duration_seconds") or 0.0),
        "video_frame_count": int(meta.get("total_frames") or 0),
        "video_width": int(meta.get("width") or 0),
        "video_height": int(meta.get("height") or 0),
        "platform": platform,
        "task_id": task_id,
        "task_instruction": str(al.get("task_instruction") or ""),
        "n_actions": len(actions),
        "_actions": actions,  # split out into tasks.jsonl below
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, required=True,
                   help="Unzipped VideoCUA app dir (task folders), or a parent of several.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--platforms", nargs="*", default=None,
                   help="Optional platform filter (match on the 'platform' field, case-insensitive).")
    p.add_argument("--limit", type=int, default=None, help="First N tasks only (debug).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    task_dirs = iter_task_dirs(args.raw_dir)
    if args.limit is not None:
        task_dirs = task_dirs[: args.limit]

    want = {p.lower() for p in args.platforms} if args.platforms else None
    clip_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    n_skipped = 0
    for td in task_dirs:
        row = build_row(td)
        if row is None:
            n_skipped += 1
            continue
        if want is not None and row["platform"].lower() not in want:
            continue
        actions = row.pop("_actions")
        if not row["video_ok"]:
            n_skipped += 1
            # still emit the clip row so stage_01 can log the skip; but flag it
        clip_rows.append(row)
        task_rows.append({
            "segment_id": row["segment_id"],
            "task_id": row["task_id"],
            "platform": row["platform"],
            "task_instruction": row["task_instruction"],
            "video_path": row["video_path"],
            "video_fps": row["video_fps"],
            "video_duration_s": row["video_duration_s"],
            "actions": actions,
        })

    clips_path = out_dir / "clips_manifest.jsonl"
    tasks_path = out_dir / "tasks.jsonl"
    with clips_path.open("w") as f:
        for r in clip_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with tasks_path.open("w") as f:
        for r in task_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ok = sum(1 for r in clip_rows if r["video_ok"])
    platforms = sorted({r["platform"] for r in clip_rows})
    write_json(out_dir / "manifest.json", {
        "artifact_type": "videocua_clips_manifest",
        "schema_version": 1,
        "clips_manifest": "clips_manifest.jsonl",
        "tasks": "tasks.jsonl",
        "raw_dir": str(args.raw_dir.resolve()),
        "platforms": platforms,
        "n_tasks": len(clip_rows),
        "n_video_ok": n_ok,
        "n_skipped": n_skipped,
    })
    print(f"[vcua_manifest] {len(clip_rows)} tasks ({n_ok} decodable, {n_skipped} skipped) "
          f"across {len(platforms)} platform(s) -> {clips_path}", flush=True)


if __name__ == "__main__":
    main()
