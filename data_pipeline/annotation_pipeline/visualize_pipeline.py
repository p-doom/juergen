#!/usr/bin/env python3
"""Serve an ad hoc dashboard for inspecting v3 pipeline artifacts."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import socketserver
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from annotation_pipeline import config
from annotation_pipeline.common import read_jsonl
from annotation_pipeline.run_pipeline import frames_cache_dir, load_clips


PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = config.PROCESSED_ROOT


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def read_jsonl_limited(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    return rows if limit is None else rows[:limit]


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_seconds(value: Any) -> str:
    seconds = safe_float(value)
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    if minutes:
        return f"{minutes}m {rest:.1f}s"
    return f"{rest:.1f}s"


def existing_stage04_dir(clip_dir: Path) -> Path:
    stage04 = clip_dir / "stage_04_sft_samples"
    if stage04.exists():
        return stage04
    return clip_dir


def existing_stage02_dir(clip_dir: Path) -> Path:
    """Stage 02 is `stage_02_segment` now; fall back to the pre-refactor name."""
    for name in ("stage_02_segment", "stage_02_vlm_trajectories"):
        candidate = clip_dir / name
        if candidate.exists():
            return candidate
    return clip_dir / "stage_02_segment"


def infer_cache_dir(run_config: dict[str, Any], clip_id: str) -> Path | None:
    clips = load_clips()
    clip = clips.get(clip_id)
    if clip is None:
        return None
    target_fps = safe_int(run_config.get("target_fps"), config.DEFAULT_TARGET_FPS)
    target_height = safe_int(run_config.get("target_height"), config.DEFAULT_TARGET_HEIGHT)
    max_noop_run = safe_int(
        run_config.get("stage01_max_noop_run"),
        config.DEFAULT_STAGE01_MAX_NOOP_RUN,
    )
    preferred = frames_cache_dir(clip, target_fps, target_height, max_noop_run)
    if preferred.exists():
        return preferred

    rec8 = clip["recording_id"][:8]
    legacy_pattern = (
        f"{rec8}_s{clip['segment_start']:04d}-{clip['segment_end']:04d}"
        f"_{target_fps}fps_{target_height}p*"
    )
    matches = sorted((OUTPUTS_DIR / "cache" / "frames").glob(legacy_pattern))
    return matches[-1] if matches else preferred


def sample_records(records: list[dict[str, Any]], count: int = 12) -> list[dict[str, Any]]:
    if len(records) <= count:
        return records
    if count <= 1:
        return [records[0]]
    step = (len(records) - 1) / (count - 1)
    return [records[round(i * step)] for i in range(count)]


def timeline_bins(records: list[dict[str, Any]], bin_count: int = 96) -> list[dict[str, Any]]:
    if not records:
        return []
    start = safe_float(records[0].get("global_time_s"))
    end = safe_float(records[-1].get("global_time_s"))
    span = max(1e-6, end - start)
    bins = [
        {
            "start_s": start + span * i / bin_count,
            "end_s": start + span * (i + 1) / bin_count,
            "frames": 0,
            "active": 0,
        }
        for i in range(bin_count)
    ]
    for record in records:
        idx = min(bin_count - 1, max(0, int((safe_float(record.get("global_time_s")) - start) / span * bin_count)))
        bins[idx]["frames"] += 1
        if record.get("action") != "NO_OP":
            bins[idx]["active"] += 1
    return bins


def intervals_overlap(a_start: Any, a_end: Any, b_start: Any, b_end: Any) -> bool:
    start_a = safe_float(a_start)
    end_a = safe_float(a_end)
    start_b = safe_float(b_start)
    end_b = safe_float(b_end)
    return start_a <= end_b and end_a >= start_b


def normalized_path_key(raw_path: Any) -> str:
    if not raw_path:
        return ""
    try:
        return str(Path(str(raw_path)).expanduser().resolve())
    except (OSError, RuntimeError):
        return str(raw_path)


def iter_sample_image_paths(sample: dict[str, Any]):
    for message in sample.get("messages", []):
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image" and block.get("image"):
                yield str(block["image"])


def compact_sample(sample: dict[str, Any]) -> dict[str, Any]:
    first_image = None
    image_count = 0
    for image_path in iter_sample_image_paths(sample):
        image_count += 1
        if first_image is None:
            first_image = image_path
    return {
        "sample_id": sample.get("sample_id"),
        "instruction": sample.get("instruction"),
        "start_time_s": sample.get("start_time_s"),
        "end_time_s": sample.get("end_time_s"),
        "duration_s": sample.get("duration_s"),
        "n_frames": sample.get("n_frames"),
        "n_non_noop": sample.get("n_non_noop"),
        "image_count": image_count,
        "first_image": first_image,
        "source_trajectory": sample.get("source_trajectory", {}),
    }


def compact_trajectory(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_time_s": row.get("start_time_s"),
        "end_time_s": row.get("end_time_s"),
        "instruction": row.get("instruction"),
        "label": row.get("label") or row.get("segment_label"),
        "confidence": row.get("confidence"),
        "completed": row.get("completed"),
        "reason": row.get("reason"),
        "boundary": row.get("boundary"),
    }


def build_segment_lineage(
    clip_id: str,
    stage00_dir: Path,
    stage01_dir: Path,
    stage02_dir: Path,
    stage03_dir: Path,
    stage04_dir: Path,
    stage02_raw: dict[str, Any],
    stage02_candidates: list[dict[str, Any]],
    stage02_merged: list[dict[str, Any]],
    stage03_samples: list[dict[str, Any]],
    stage04_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest = read_jsonl_limited(stage00_dir / "manifest.jsonl")
    frame_records = read_jsonl_limited(stage01_dir / "frame_records.jsonl")
    segment_summaries = read_json(stage01_dir / "segment_summaries.json", [])
    summary_by_segment = {
        safe_int(row.get("segment_idx"), -1): row
        for row in segment_summaries
        if isinstance(row, dict)
    }
    stage04_by_sample_id = {row.get("sample_id"): row for row in stage04_manifest}
    trajectories = stage02_raw.get("trajectories", []) if isinstance(stage02_raw, dict) else []

    segments: list[dict[str, Any]] = []
    offset_s = 0.0
    for row in manifest:
        duration_s = safe_float(row.get("video_duration_s"))
        start_s = offset_s
        end_s = offset_s + duration_s
        segment_idx = safe_int(row.get("segment_idx"), -1)
        segment_id = str(row.get("segment_id") or f"{row.get('recording_id')}_seg{segment_idx:04d}")
        seg_frames = [record for record in frame_records if record.get("segment_id") == segment_id]
        seg_samples = [
            compact_sample(sample)
            for sample in stage03_samples
            if intervals_overlap(sample.get("start_time_s"), sample.get("end_time_s"), start_s, end_s)
        ]
        stage04_rows = [
            stage04_by_sample_id.get(sample.get("sample_id"), {"sample_id": sample.get("sample_id")})
            for sample in seg_samples
        ]
        segments.append(
            {
                "clip_id": clip_id,
                "user_id": row.get("user_id"),
                "recording_id": row.get("recording_id"),
                "segment_id": segment_id,
                "segment_idx": segment_idx,
                "start_time_s": round(start_s, 6),
                "end_time_s": round(end_s, 6),
                "duration_s": round(duration_s, 6),
                "stage_outputs": {
                    "stage_00_manifest": {
                        "dir": str(stage00_dir),
                        "row": row,
                    },
                    "stage_01_frames_actions": {
                        "dir": str(stage01_dir),
                        "summary": summary_by_segment.get(segment_idx, {}),
                        "n_frame_records": len(seg_frames),
                        "n_non_noop": sum(1 for record in seg_frames if record.get("action") != "NO_OP"),
                        "frame_preview": sample_records(seg_frames, 10),
                        "timeline": timeline_bins(seg_frames, 56),
                    },
                    "stage_02_vlm_trajectories": {
                        "dir": str(stage02_dir),
                        "trajectories": [
                            compact_trajectory(row)
                            for row in trajectories
                            if intervals_overlap(row.get("start_time_s"), row.get("end_time_s"), start_s, end_s)
                        ],
                        "candidates": [
                            compact_trajectory(row)
                            for row in stage02_candidates
                            if intervals_overlap(row.get("start_time_s"), row.get("end_time_s"), start_s, end_s)
                        ],
                        "merged": [
                            compact_trajectory(row)
                            for row in stage02_merged
                            if intervals_overlap(row.get("start_time_s"), row.get("end_time_s"), start_s, end_s)
                        ],
                    },
                    "stage_03_assemble": {
                        "dir": str(stage03_dir),
                        "samples": seg_samples,
                    },
                    "stage_04_sft_samples": {
                        "dir": str(stage04_dir),
                        "rows": stage04_rows,
                    },
                },
            }
        )
        offset_s = end_s
    return segments


def build_lineage_tree(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    users: dict[str, dict[str, Any]] = {}
    for segment in segments:
        user_id = str(segment.get("user_id") or "unknown_user")
        recording_id = str(segment.get("recording_id") or "unknown_recording")
        user = users.setdefault(user_id, {"user_id": user_id, "recordings": {}})
        recording = user["recordings"].setdefault(
            recording_id,
            {"recording_id": recording_id, "segments": []},
        )
        recording["segments"].append(segment)

    tree: list[dict[str, Any]] = []
    for user in users.values():
        recordings = []
        for recording in user["recordings"].values():
            recording["segments"].sort(key=lambda row: (row.get("clip_id", ""), row.get("segment_idx", 0)))
            recordings.append(recording)
        recordings.sort(key=lambda row: row["recording_id"])
        tree.append({"user_id": user["user_id"], "recordings": recordings})
    tree.sort(key=lambda row: row["user_id"])
    return tree


def build_run_summary(run_name: str) -> dict[str, Any]:
    run_dir = OUTPUTS_DIR / "runs" / run_name
    run_config = read_json(run_dir / "run_config.json", {})
    clips: list[dict[str, Any]] = []
    lineage_segments: list[dict[str, Any]] = []

    for clip_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        clip_id = clip_dir.name
        cache_dir = infer_cache_dir(run_config, clip_id)
        stage00 = cache_dir / "stage_00_manifest" if cache_dir else clip_dir / "stage_00_manifest"
        stage01 = cache_dir / "stage_01_frames_actions" if cache_dir else clip_dir / "stage_01_frames_actions"
        stage02 = existing_stage02_dir(clip_dir)
        stage03 = clip_dir / "stage_03_assemble"
        stage04 = existing_stage04_dir(clip_dir)

        manifest = read_jsonl_limited(stage00 / "manifest.jsonl")
        frame_records = read_jsonl_limited(stage01 / "frame_records.jsonl")
        assembled_all = read_jsonl_limited(stage03 / "trajectories.jsonl")
        rejected = read_jsonl_limited(stage03 / "rejected_trajectories.jsonl")
        trajectory_manifest = read_jsonl_limited(stage04 / "trajectory_manifest.jsonl")
        chat_rows = read_jsonl_limited(stage04 / "chat.jsonl", 6)

        stage02_raw = read_json(stage02 / "trajectories_raw.json", {})
        stage02_summary = read_json(stage02 / "stage02_summary.json", {})
        if not stage02_summary and stage02_raw:
            stage02_summary = {
                "annotation_source": stage02_raw.get("annotation_source"),
                "n_trajectories": len(stage02_raw.get("trajectories", []) or []),
                "dry_run": bool(stage02_raw.get("dry_run")),
            }
        stage02_candidates = read_jsonl_limited(stage02 / "pass_a_candidates.jsonl", 100)
        stage02_merged = read_jsonl_limited(stage02 / "pass_a_merged_segments.jsonl", 100)
        segment_lineage = build_segment_lineage(
            clip_id=clip_id,
            stage00_dir=stage00,
            stage01_dir=stage01,
            stage02_dir=stage02,
            stage03_dir=stage03,
            stage04_dir=stage04,
            stage02_raw=stage02_raw,
            stage02_candidates=stage02_candidates,
            stage02_merged=stage02_merged,
            stage03_samples=assembled_all,
            stage04_manifest=trajectory_manifest,
        )
        lineage_segments.extend(segment_lineage)

        stage_data = {
            "stage_00_manifest": {
                "dir": str(stage00),
                "summary": read_json(stage00 / "manifest_summary.json", {}),
                "manifest": manifest,
            },
            "stage_01_frames_actions": {
                "dir": str(stage01),
                "summary": read_json(stage01 / "frames_actions_summary.json", {}),
                "segments": read_json(stage01 / "segment_summaries.json", []),
                "frame_preview": sample_records(frame_records, 14),
                "timeline": timeline_bins(frame_records),
                "n_frame_records": len(frame_records),
            },
            "stage_02_vlm_trajectories": {
                "dir": str(stage02),
                "summary": stage02_summary,
                "raw": stage02_raw,
                "candidates": stage02_candidates,
                "merged": stage02_merged,
                "naming_rejected": read_json(stage02 / "naming_rejected.json", []),
            },
            "stage_03_assemble": {
                "dir": str(stage03),
                "summary": read_json(stage03 / "assemble_summary.json", {}),
                "samples": assembled_all[:25],
                "rejected": rejected[:100],
            },
            "stage_04_sft_samples": {
                "dir": str(stage04),
                "summary": read_json(stage04 / "bucket_summary.json", {}),
                "manifest": trajectory_manifest[:100],
                "chat_preview": chat_rows,
                "bucket_files": sorted(path.name for path in stage04.glob("chat_*.jsonl")),
            },
        }

        clips.append(
            {
                "clip_id": clip_id,
                "clip_dir": str(clip_dir),
                "cache_dir": str(cache_dir) if cache_dir else None,
                "stages": stage_data,
                "lineage_segments": segment_lineage,
            }
        )

    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "run_config": run_config,
        "clips": clips,
        "lineage": build_lineage_tree(lineage_segments),
    }


def build_sample_frame_player(run_name: str, sample_id: str) -> dict[str, Any] | None:
    run_dir = OUTPUTS_DIR / "runs" / run_name
    run_config = read_json(run_dir / "run_config.json", {})
    if not run_dir.exists():
        return None

    for clip_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        clip_id = clip_dir.name
        samples_path = clip_dir / "stage_03_assemble" / "trajectories.jsonl"
        for sample in read_jsonl_limited(samples_path):
            if str(sample.get("sample_id")) != sample_id:
                continue

            cache_dir = infer_cache_dir(run_config, clip_id)
            stage01 = cache_dir / "stage_01_frames_actions" if cache_dir else clip_dir / "stage_01_frames_actions"
            frame_records = read_jsonl_limited(stage01 / "frame_records.jsonl")
            record_by_image = {
                normalized_path_key(record.get("image_path")): record
                for record in frame_records
                if record.get("image_path")
            }
            frames = []
            for idx, image_path in enumerate(iter_sample_image_paths(sample)):
                record = record_by_image.get(normalized_path_key(image_path), {})
                frames.append(
                    {
                        "index": idx,
                        "image_path": image_path,
                        "recording_id": record.get("recording_id"),
                        "segment_id": record.get("segment_id"),
                        "segment_idx": record.get("segment_idx"),
                        "local_bin_idx": record.get("local_bin_idx"),
                        "global_frame_idx": record.get("global_frame_idx"),
                        "local_time_s": record.get("local_time_s"),
                        "global_time_s": record.get("global_time_s"),
                        "source_frame_idx": record.get("source_frame_idx"),
                        "action": record.get("action"),
                    }
                )

            return {
                "run_name": run_name,
                "clip_id": clip_id,
                "sample_id": sample.get("sample_id"),
                "instruction": sample.get("instruction"),
                "start_time_s": sample.get("start_time_s"),
                "end_time_s": sample.get("end_time_s"),
                "duration_s": sample.get("duration_s"),
                "n_frames": sample.get("n_frames"),
                "n_non_noop": sample.get("n_non_noop"),
                "source_trajectory": sample.get("source_trajectory", {}),
                "frames": frames,
            }
    return None


def list_runs() -> list[dict[str, Any]]:
    runs_dir = OUTPUTS_DIR / "runs"
    if not runs_dir.exists():
        return []
    rows = []
    for run_dir in sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True):
        config_path = run_dir / "run_config.json"
        run_config = read_json(config_path, {})
        rows.append(
            {
                "name": run_dir.name,
                "path": str(run_dir),
                "created_utc": run_config.get("created_utc"),
                "clips": run_config.get("clips", []),
                "dry_run_vlm": run_config.get("dry_run_vlm"),
            }
        )
    return rows


def safe_image_path(raw_path: str) -> Path | None:
    if not raw_path:
        return None
    try:
        path = Path(raw_path).expanduser().resolve()
    except RuntimeError:
        return None
    roots = [
        PIPELINE_DIR.resolve(),
        Path("/tmp").resolve(),
        config.RAW_DATA_ROOT.resolve(),
    ]
    if any(path == root or root in path.parents for root in roots):
        return path
    return None


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>v3 Pipeline Inspector</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #151922;
      --muted: #697386;
      --line: #d9dee8;
      --accent: #126f84;
      --accent-2: #815e16;
      --bad: #b42318;
      --ok: #18794e;
      --code: #edf1f5;
      --shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(246,247,249,0.94);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--line);
    }
    .topbar {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 12px;
      align-items: center;
      max-width: 1520px;
      margin: 0 auto;
      padding: 14px 18px;
    }
    h1 { margin: 0; font-size: 18px; font-weight: 720; letter-spacing: 0; }
    select, button {
      height: 34px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 6px;
      padding: 0 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    main {
      max-width: 1520px;
      margin: 0 auto;
      padding: 18px;
    }
    .layout {
      display: grid;
      grid-template-columns: 250px minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }
    nav {
      position: sticky;
      top: 74px;
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow);
      border-radius: 8px;
      padding: 10px;
    }
    .clip-btn {
      width: 100%;
      display: block;
      text-align: left;
      margin: 6px 0;
      border-color: transparent;
      background: transparent;
    }
    .clip-btn.active {
      border-color: rgba(18,111,132,0.35);
      background: rgba(18,111,132,0.08);
      color: var(--accent);
      font-weight: 650;
    }
    details {
      border-top: 1px solid var(--line);
      padding-top: 8px;
      margin-top: 8px;
    }
    details:first-of-type {
      border-top: 0;
      margin-top: 0;
      padding-top: 0;
    }
    summary {
      cursor: pointer;
      font-weight: 680;
      overflow-wrap: anywhere;
    }
    .tree-recording {
      margin: 8px 0 0 10px;
    }
    .segment-btn {
      width: calc(100% - 12px);
      margin: 5px 0 0 12px;
      display: block;
      text-align: left;
      border-color: transparent;
      background: transparent;
      height: auto;
      min-height: 32px;
      padding: 6px 8px;
    }
    .segment-btn.active {
      border-color: rgba(18,111,132,0.35);
      background: rgba(18,111,132,0.08);
      color: var(--accent);
      font-weight: 650;
    }
    .run-meta {
      color: var(--muted);
      font-size: 12px;
      border-top: 1px solid var(--line);
      margin-top: 12px;
      padding-top: 12px;
      overflow-wrap: anywhere;
    }
    .content { min-width: 0; }
    .lineage-canvas {
      position: relative;
      min-height: 240px;
      border-radius: 8px;
      border: 1px solid #252a32;
      margin-bottom: 14px;
      overflow: hidden;
      background-color: #080a0d;
      background-image: radial-gradient(#7f8894 1px, transparent 1px);
      background-size: 48px 48px;
      box-shadow: var(--shadow);
      color: #eceff3;
    }
    .lineage-flow {
      display: grid;
      grid-template-columns: minmax(160px, 1fr) 44px minmax(180px, 1fr) 44px minmax(180px, 1fr) 44px minmax(180px, 1fr);
      gap: 0;
      align-items: center;
      padding: 52px 46px;
      min-height: 240px;
    }
    .flow-node {
      background: #272522;
      border: 2px solid transparent;
      min-height: 78px;
      padding: 14px 16px;
      display: grid;
      align-content: center;
      box-shadow: 0 0 0 1px rgba(255,255,255,0.03);
    }
    .flow-node.selected {
      border-color: #d9432b;
      box-shadow: inset 4px 0 0 #d9432b;
    }
    .flow-eyebrow {
      color: #e04a32;
      font-size: 12px;
      letter-spacing: 1.4px;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .flow-title {
      font-size: 18px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .flow-sub {
      color: #b2b8c2;
      margin-top: 6px;
      font-size: 12px;
    }
    .flow-edge {
      height: 2px;
      background: #45413a;
      position: relative;
    }
    .flow-edge::after {
      content: "";
      position: absolute;
      right: -4px;
      top: -4px;
      width: 10px;
      height: 10px;
      border-top: 2px solid #45413a;
      border-right: 2px solid #45413a;
      transform: rotate(45deg);
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric, section {
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow);
      border-radius: 8px;
    }
    .metric { padding: 12px; min-height: 70px; }
    .label { color: var(--muted); font-size: 12px; }
    .value { font-size: 22px; font-weight: 760; margin-top: 4px; }
    section { margin-bottom: 14px; overflow: hidden; }
    .stage-head {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }
    .stage-num {
      width: 32px;
      height: 32px;
      display: grid;
      place-items: center;
      border-radius: 50%;
      background: var(--ink);
      color: white;
      font-weight: 760;
    }
    h2 { margin: 0; font-size: 15px; }
    .path { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; overflow-wrap: anywhere; }
    .stage-body { padding: 14px; }
    .grid-2 {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
    }
    .grid-3 {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 7px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 650; background: #fbfcfd; }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      background: var(--code);
      padding: 1px 4px;
      border-radius: 4px;
    }
    .timeline {
      display: flex;
      height: 42px;
      align-items: stretch;
      gap: 1px;
      border: 1px solid var(--line);
      background: var(--line);
      border-radius: 6px;
      overflow: hidden;
    }
    .bar {
      flex: 1;
      background: #e9edf3;
      position: relative;
    }
    .bar::after {
      content: "";
      position: absolute;
      left: 0; right: 0; bottom: 0;
      height: var(--h);
      background: var(--accent);
    }
    .frames {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .frame {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fbfcfd;
    }
    .frame img {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 10;
      object-fit: cover;
      background: #d9dee8;
    }
    .frame p {
      margin: 0;
      padding: 7px;
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .pill {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      background: #fbfcfd;
      color: var(--muted);
    }
    .pill.ok { color: var(--ok); border-color: rgba(24,121,78,0.3); background: rgba(24,121,78,0.06); }
    .pill.bad { color: var(--bad); border-color: rgba(180,35,24,0.3); background: rgba(180,35,24,0.06); }
    .sample-list {
      display: grid;
      gap: 8px;
    }
    .sample-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfd;
    }
    .sample-summary {
      display: grid;
      grid-template-columns: 112px minmax(0, 1fr);
      gap: 10px;
      align-items: start;
    }
    .sample-thumb {
      width: 112px;
      aspect-ratio: 16 / 10;
      border-radius: 6px;
      border: 1px solid var(--line);
      object-fit: cover;
      background: #d9dee8;
    }
    .sample-thumb.empty-thumb {
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 12px;
    }
    .sample-title { font-weight: 680; margin-bottom: 4px; }
    .sample-actions {
      margin-top: 8px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .player-shell { margin-top: 10px; }
    .player {
      border: 1px solid #22231f;
      border-radius: 8px;
      overflow: hidden;
      background: #151713;
      color: #e8e4dc;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .player-screen {
      position: relative;
      display: grid;
      place-items: center;
      min-height: 360px;
      background: #050608;
    }
    .player-screen img {
      display: block;
      width: 100%;
      max-height: 68vh;
      object-fit: contain;
      background: #050608;
    }
    .player-frame-count {
      position: absolute;
      top: 8px;
      right: 10px;
      padding: 2px 7px;
      color: #f2eee7;
      background: rgba(0, 0, 0, 0.42);
      border-radius: 4px;
      font-size: 12px;
    }
    .player-controls {
      display: grid;
      grid-template-columns: auto auto minmax(130px, 1fr) auto auto;
      gap: 10px;
      align-items: center;
      padding: 12px 14px;
      background: #151713;
      border-top: 1px solid #26281f;
    }
    .player-controls button {
      width: 32px;
      height: 32px;
      padding: 0;
      background: transparent;
      color: #e8e4dc;
      border-color: transparent;
      font-size: 18px;
      line-height: 1;
    }
    .player-controls button:hover { background: #24261f; }
    .player-range { width: 100%; }
    .player-meta {
      padding: 0 14px 10px;
      color: #c5beb2;
      font-size: 12px;
    }
    .player-action {
      color: #f1ece5;
      font-size: 13px;
      padding: 0 14px 10px;
      overflow-wrap: anywhere;
    }
    .action-table-wrap {
      max-height: 260px;
      overflow: auto;
      border-top: 1px solid #26281f;
    }
    .action-table {
      width: 100%;
      border-collapse: collapse;
      color: #bfb8ad;
      font-size: 12px;
    }
    .action-table th,
    .action-table td {
      padding: 7px 14px;
      border-bottom: 1px solid #24261f;
      background: #151713;
    }
    .action-table th {
      color: #8f897f;
      font-weight: 650;
      text-align: left;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    .action-table td:first-child,
    .action-table th:first-child {
      width: 64px;
      text-align: right;
      color: #d1cabf;
    }
    .action-table tr.active td {
      background: #45251d;
      color: #f4eee8;
    }
    .instruction-box {
      margin-top: 8px;
      padding: 8px;
      border-left: 3px solid var(--accent);
      background: rgba(18,111,132,0.06);
    }
    .subhead {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin: 0 0 8px;
    }
    .muted { color: var(--muted); }
    .empty {
      color: var(--muted);
      padding: 14px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #fbfcfd;
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      nav { position: static; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid-2, .grid-3 { grid-template-columns: 1fr; }
      .lineage-flow {
        grid-template-columns: 1fr;
        gap: 12px;
        padding: 22px;
      }
      .flow-edge {
        width: 2px;
        height: 22px;
        justify-self: center;
      }
      .flow-edge::after {
        right: -4px;
        top: 13px;
        transform: rotate(135deg);
      }
      .sample-summary { grid-template-columns: 1fr; }
      .sample-thumb { width: 100%; }
      .player-screen { min-height: 220px; }
      .player-controls { grid-template-columns: auto auto minmax(80px, 1fr) auto auto; }
      .topbar { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <h1>v3 Pipeline Inspector</h1>
      <select id="runSelect"></select>
      <button id="refreshBtn">Refresh</button>
    </div>
  </header>
  <main>
    <div class="layout">
      <nav>
        <div class="label">Lineage</div>
        <div id="clipNav"></div>
        <div class="run-meta" id="runMeta"></div>
      </nav>
      <div class="content">
        <div id="content"></div>
      </div>
    </div>
  </main>
  <script>
    const state = { runs: [], run: null, selectedSegmentKey: null, playerTimers: new Set(), activePlayer: null };
    const $ = (id) => document.getElementById(id);
    const esc = (v) => String(v ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const fmt = (v) => v === null || v === undefined || v === "" ? "—" : esc(v);
    const num = (v) => Number.isFinite(Number(v)) ? Number(v).toLocaleString() : "—";
    const pct = (a,b) => b ? `${Math.round((a/b)*100)}%` : "—";
    const imgUrl = (path) => `/image?path=${encodeURIComponent(path || "")}`;

    async function api(path) {
      const res = await fetch(path);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    async function init() {
      state.runs = await api("/api/runs");
      $("runSelect").innerHTML = state.runs.map(r => `<option value="${esc(r.name)}">${esc(r.name)}</option>`).join("");
      if (!state.runs.length) {
        $("content").innerHTML = `<div class="empty">No runs found under outputs/runs.</div>`;
        return;
      }
      $("runSelect").addEventListener("change", () => loadRun($("runSelect").value));
      $("refreshBtn").addEventListener("click", () => loadRun($("runSelect").value));
      document.addEventListener("keydown", handlePlayerKeys);
      await loadRun(state.runs[0].name);
    }

    async function loadRun(name) {
      state.run = await api(`/api/run/${encodeURIComponent(name)}`);
      const first = allSegments()[0];
      state.selectedSegmentKey = first ? segmentKey(first) : null;
      render();
    }

    function segmentKey(segment) {
      return `${segment.clip_id}:${segment.segment_id}`;
    }

    function allSegments() {
      const out = [];
      for (const user of state.run?.lineage || []) {
        for (const recording of user.recordings || []) {
          for (const segment of recording.segments || []) out.push(segment);
        }
      }
      return out;
    }

    function selectedSegment() {
      return allSegments().find(seg => segmentKey(seg) === state.selectedSegmentKey) || allSegments()[0] || null;
    }

    function shortId(value, n = 8) {
      const text = String(value || "");
      return text.length > n ? `${text.slice(0, n)}…` : text;
    }

    function renderClipNav() {
      const lineage = state.run?.lineage || [];
      $("clipNav").innerHTML = lineage.map(user => `
        <details open>
          <summary>user ${esc(shortId(user.user_id, 10))}</summary>
          ${(user.recordings || []).map(recording => `
            <details class="tree-recording" open>
              <summary>rec ${esc(shortId(recording.recording_id, 8))}</summary>
              ${(recording.segments || []).map(segment => `
                <button class="segment-btn ${segmentKey(segment) === state.selectedSegmentKey ? "active" : ""}" data-key="${esc(segmentKey(segment))}">
                  seg ${String(segment.segment_idx).padStart(4, "0")}
                  <div class="label">${esc(segment.clip_id)} · ${num(segment.stage_outputs.stage_03_assemble.samples.length)} samples</div>
                </button>
              `).join("")}
            </details>
          `).join("")}
        </details>
      `).join("");
      document.querySelectorAll(".segment-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          state.selectedSegmentKey = btn.dataset.key;
          render();
        });
      });
    }

    function render() {
      if (!state.run) return;
      stopPlayerTimers();
      state.activePlayer = null;
      renderClipNav();
      const cfg = state.run.run_config || {};
      $("runMeta").innerHTML = `
        <div><b>${esc(state.run.run_name)}</b></div>
        <div>${fmt(cfg.created_utc)}</div>
        <div>dry run: ${fmt(cfg.dry_run_vlm)}</div>
        <div>train fps/height: ${fmt(cfg.target_fps)} / ${fmt(cfg.target_height)}p</div>
        <div>vlm height: ${fmt(cfg.vlm_frame_height)}p</div>
        <div>noop cap: ${fmt(cfg.stage01_max_noop_run)}</div>
        <div>${esc(state.run.run_dir)}</div>
      `;
      const segment = selectedSegment();
      if (!segment) {
        $("content").innerHTML = `<div class="empty">No segment lineage in this run.</div>`;
        return;
      }
      $("content").innerHTML = renderSegment(segment);
      bindPlayers();
    }

    function metric(label, value, detail = "") {
      return `<div class="metric"><div class="label">${esc(label)}</div><div class="value">${fmt(value)}</div><div class="label">${esc(detail)}</div></div>`;
    }

    function stage(numLabel, title, path, body) {
      return `<section>
        <div class="stage-head">
          <div class="stage-num">${esc(numLabel)}</div>
          <div><h2>${esc(title)}</h2><div class="path">${esc(path)}</div></div>
          <div></div>
        </div>
        <div class="stage-body">${body}</div>
      </section>`;
    }

    function smallStats(obj) {
      if (!obj || !Object.keys(obj).length) return `<div class="empty">No summary file found.</div>`;
      return `<div class="grid-3">` + Object.entries(obj).slice(0, 18).map(([k,v]) =>
        `<div><div class="label">${esc(k)}</div><div><code>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</code></div></div>`
      ).join("") + `</div>`;
    }

    function table(rows, cols) {
      if (!rows || !rows.length) return `<div class="empty">No rows.</div>`;
      return `<table><thead><tr>${cols.map(c => `<th>${esc(c.label)}</th>`).join("")}</tr></thead><tbody>` +
        rows.map(row => `<tr>${cols.map(c => `<td>${c.render ? c.render(row) : fmt(row[c.key])}</td>`).join("")}</tr>`).join("") +
        `</tbody></table>`;
    }

    function renderTimeline(stage01) {
      const bins = stage01.timeline || [];
      const maxActive = Math.max(1, ...bins.map(b => b.active || 0));
      return `<div class="timeline" title="Active frames per time bin">` +
        bins.map(b => `<div class="bar" style="--h:${Math.max(2, Math.round((b.active || 0) / maxActive * 100))}%"></div>`).join("") +
        `</div>`;
    }

    function renderFrames(stage01) {
      const frames = stage01.frame_preview || [];
      if (!frames.length) return `<div class="empty">No frame preview.</div>`;
      return `<div class="frames">` + frames.map(f => `
        <div class="frame">
          <img src="${imgUrl(f.image_path)}" loading="lazy" />
          <p><b>t=${fmt(f.global_time_s)}s</b><br>${esc(f.action)}<br><span class="muted">${esc(f.segment_id || "")}</span></p>
        </div>
      `).join("") + `</div>`;
    }

    function secs(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return "—";
      if (n >= 60) {
        const minutes = Math.floor(n / 60);
        const rest = n - minutes * 60;
        return `${minutes}m ${rest.toFixed(1)}s`;
      }
      return `${n.toFixed(1)}s`;
    }

    function scaledResolutionForHeight(row, targetHeight) {
      const inputWidth = Number(row?.video_width);
      const inputHeight = Number(row?.video_height);
      if (Number.isFinite(targetHeight) && Number.isFinite(inputWidth) && Number.isFinite(inputHeight) && inputHeight > 0) {
        const scaledWidth = Math.max(2, Math.round((inputWidth * targetHeight / inputHeight) / 2) * 2);
        return `${scaledWidth}x${targetHeight}`;
      }
      return Number.isFinite(targetHeight) ? `${targetHeight}p` : "—";
    }

    function scaledResolution(row) {
      return scaledResolutionForHeight(row, Number(state.run?.run_config?.target_height));
    }

    function renderLineageGraph(segment) {
      const stages = segment.stage_outputs || {};
      const s01 = stages.stage_01_frames_actions || {};
      const s02 = stages.stage_02_vlm_trajectories || {};
      const s03 = stages.stage_03_assemble || {};
      const s04 = stages.stage_04_sft_samples || {};
      const segName = `seg ${String(segment.segment_idx ?? "?").padStart(4, "0")}`;
      return `<div class="lineage-canvas">
        <div class="lineage-flow">
          <div class="flow-node">
            <div class="flow-eyebrow">User</div>
            <div class="flow-title">${esc(shortId(segment.user_id, 14))}</div>
            <div class="flow-sub">source account</div>
          </div>
          <div class="flow-edge"></div>
          <div class="flow-node">
            <div class="flow-eyebrow">Recording</div>
            <div class="flow-title">${esc(shortId(segment.recording_id, 12))}</div>
            <div class="flow-sub">${esc(segment.clip_id)}</div>
          </div>
          <div class="flow-edge"></div>
          <div class="flow-node selected">
            <div class="flow-eyebrow">Segment</div>
            <div class="flow-title">${esc(segName)}</div>
            <div class="flow-sub">${secs(segment.start_time_s)} to ${secs(segment.end_time_s)} · ${num(s01.n_frame_records)} frames</div>
          </div>
          <div class="flow-edge"></div>
          <div class="flow-node">
            <div class="flow-eyebrow">Dataset</div>
            <div class="flow-title">${num((s03.samples || []).length)} samples</div>
            <div class="flow-sub">${num((s02.trajectories || []).length)} VLM trajectories · ${num((s04.rows || []).length)} rows</div>
          </div>
        </div>
      </div>`;
    }

    function stopPlayerTimers() {
      for (const timer of state.playerTimers) clearInterval(timer);
      state.playerTimers.clear();
    }

    function closeOpenPlayers(keepItem = null) {
      stopPlayerTimers();
      state.activePlayer = null;
      document.querySelectorAll(".sample-item").forEach(item => {
        if (keepItem && item === keepItem) return;
        const shell = item.querySelector(".player-shell");
        const btn = item.querySelector(".player-load");
        if (shell) shell.innerHTML = "";
        if (btn) {
          btn.disabled = false;
          btn.textContent = "Open frame player";
        }
      });
    }

    function handlePlayerKeys(event) {
      if (!state.activePlayer) return;
      const tag = event.target?.tagName;
      const typing = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || event.target?.isContentEditable;
      if (typing && !event.target?.classList?.contains("player-range")) return;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        state.activePlayer.step(-1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        state.activePlayer.step(1);
      } else if (event.key === " ") {
        event.preventDefault();
        state.activePlayer.toggle();
      }
    }

    function bindPlayers() {
      document.querySelectorAll(".player-load").forEach(btn => {
        btn.addEventListener("click", async () => {
          const item = btn.closest(".sample-item");
          const shell = item?.querySelector(".player-shell");
          if (!shell) return;
          if (shell.querySelector(".player")) {
            closeOpenPlayers();
            return;
          }
          const original = btn.textContent;
          btn.disabled = true;
          btn.textContent = "Loading...";
          closeOpenPlayers(item);
          try {
            const payload = await api(`/api/sample_frames?run=${encodeURIComponent(state.run.run_name)}&sample_id=${encodeURIComponent(btn.dataset.sampleId)}`);
            mountSamplePlayer(shell, payload, btn);
            btn.textContent = "Close player";
          } catch (err) {
            shell.innerHTML = `<div class="empty">${esc(err.message || err)}</div>`;
            btn.textContent = original;
          } finally {
            btn.disabled = false;
          }
        });
      });
    }

    function mountSamplePlayer(shell, payload, triggerBtn) {
      const frames = payload.frames || [];
      if (!frames.length) {
        shell.innerHTML = `<div class="empty">No frame sequence found for this sample.</div>`;
        return;
      }
      const actionRows = frames.map((frame, i) => `
        <tr data-frame-index="${i}">
          <td>${num(i)}</td>
          <td>${esc(frame.action || "UNKNOWN")}</td>
        </tr>
      `).join("");
      shell.innerHTML = `
        <div class="player" tabindex="0">
          <div class="player-screen">
            <img class="player-img" alt="" />
            <div class="player-frame-count"></div>
          </div>
          <div class="player-controls">
            <button class="player-prev" type="button" title="Previous frame (Left arrow)">&#9664;</button>
            <button class="player-play" type="button" title="Play / pause (Space)">&#9654;</button>
            <input class="player-range" type="range" min="0" max="${frames.length - 1}" value="0" />
            <button class="player-next" type="button" title="Next frame (Right arrow)">&#9654;</button>
            <button class="player-close" type="button" title="Close player">&times;</button>
          </div>
          <div class="player-action"></div>
          <div class="player-meta"></div>
          <div class="action-table-wrap">
            <table class="action-table">
              <thead><tr><th>#</th><th>action</th></tr></thead>
              <tbody>${actionRows}</tbody>
            </table>
          </div>
        </div>`;
      const player = shell.querySelector(".player");
      const img = shell.querySelector(".player-img");
      const frameCount = shell.querySelector(".player-frame-count");
      const currentAction = shell.querySelector(".player-action");
      const range = shell.querySelector(".player-range");
      const meta = shell.querySelector(".player-meta");
      const play = shell.querySelector(".player-play");
      const prev = shell.querySelector(".player-prev");
      const next = shell.querySelector(".player-next");
      const close = shell.querySelector(".player-close");
      const tableBody = shell.querySelector(".action-table tbody");
      const fps = Number(state.run?.run_config?.target_fps) || 2;
      const delayMs = Math.max(80, Math.round(1000 / fps));
      let idx = 0;
      let timer = null;

      function pause() {
        if (timer) {
          clearInterval(timer);
          state.playerTimers.delete(timer);
          timer = null;
        }
        play.innerHTML = "&#9654;";
      }

      function setIndex(nextIdx) {
        idx = Math.max(0, Math.min(frames.length - 1, nextIdx));
        const frame = frames[idx] || {};
        const previousFrame = idx > 0 ? frames[idx - 1] : null;
        const localGap = previousFrame ? Number(frame.local_time_s) - Number(previousFrame.local_time_s) : 0;
        const expectedGap = 1 / fps;
        const skippedGap = Number.isFinite(localGap) && localGap > expectedGap * 1.5;
        img.src = imgUrl(frame.image_path);
        img.alt = `${payload.sample_id} frame ${idx + 1}`;
        range.value = idx;
        frameCount.textContent = `frame ${idx + 1} / ${frames.length}`;
        currentAction.innerHTML = `<code>${esc(frame.action || "UNKNOWN")}</code>`;
        meta.innerHTML = `
          <b>${esc(payload.sample_id)}</b> · #${num(idx)} · bin ${fmt(frame.local_bin_idx)}
          · global ${fmt(frame.global_time_s)}s · local ${fmt(frame.local_time_s)}s
          ${skippedGap ? `· gap from previous kept frame: ${secs(localGap)} (${secs(localGap - expectedGap)} skipped by NO_OP cap)` : ""}
        `;
        const previous = tableBody.querySelector("tr.active");
        if (previous) previous.classList.remove("active");
        const active = tableBody.querySelector(`tr[data-frame-index="${idx}"]`);
        if (active) {
          active.classList.add("active");
          active.scrollIntoView({ block: "nearest" });
        }
      }

      function step(delta) {
        pause();
        setIndex(idx + delta);
      }

      function toggle() {
        if (timer) {
          pause();
          return;
        }
        if (idx >= frames.length - 1) setIndex(0);
        play.innerHTML = "&#10074;&#10074;";
        timer = setInterval(() => {
          if (idx >= frames.length - 1) {
            pause();
            return;
          }
          setIndex(idx + 1);
        }, delayMs);
        state.playerTimers.add(timer);
      }

      function closePlayer() {
        pause();
        shell.innerHTML = "";
        state.activePlayer = null;
        if (triggerBtn) triggerBtn.textContent = "Open frame player";
      }

      prev.addEventListener("click", () => {
        step(-1);
      });
      next.addEventListener("click", () => {
        step(1);
      });
      close.addEventListener("click", () => {
        closePlayer();
      });
      range.addEventListener("input", () => {
        pause();
        setIndex(Number(range.value));
      });
      play.addEventListener("click", () => {
        toggle();
      });
      tableBody.addEventListener("click", (event) => {
        const row = event.target.closest("tr[data-frame-index]");
        if (!row) return;
        pause();
        setIndex(Number(row.dataset.frameIndex));
      });
      state.activePlayer = { step, toggle, close: closePlayer };
      setIndex(0);
      player.focus({ preventScroll: true });
    }

    function renderSamples(stage03, stage04) {
      const samples = stage03.samples || [];
      if (!samples.length) return `<div class="empty">No assembled samples.</div>`;
      const rows = new Map(((stage04 && (stage04.rows || stage04.manifest)) || []).map(row => [row.sample_id, row]));
      return `<div class="sample-list">` + samples.slice(0, 12).map(s => `
        <div class="sample-item">
          <div class="sample-summary">
            ${s.first_image ? `<img class="sample-thumb" src="${imgUrl(s.first_image)}" loading="lazy" />` : `<div class="sample-thumb empty-thumb">no image</div>`}
            <div>
              <div class="sample-title">${esc(s.sample_id)}</div>
              <div class="instruction-box">${esc(s.instruction)}</div>
              <div class="muted">${num(s.n_frames)} frames · ${num(s.n_non_noop)} active · ${secs(s.duration_s)}</div>
              <div class="muted">tokens ${fmt(rows.get(s.sample_id)?.token_count)} · bucket ${fmt(rows.get(s.sample_id)?.bucket)}</div>
              <div class="sample-actions">
                <button class="player-load" type="button" data-sample-id="${esc(s.sample_id)}">Open frame player</button>
                <span class="pill">${num(s.image_count || s.n_frames)} image blocks</span>
              </div>
            </div>
          </div>
          <div class="player-shell"></div>
        </div>
      `).join("") + `</div>`;
    }

    function renderTrajectoryTable(rows) {
      return table(rows || [], [
        {key:"start_time_s", label:"Start", render: r => secs(r.start_time_s)},
        {key:"end_time_s", label:"End", render: r => secs(r.end_time_s)},
        {key:"confidence", label:"Conf"},
        {key:"completed", label:"Done"},
        {key:"instruction", label:"Instruction"}
      ]);
    }

    function renderSegment(segment) {
      const stages = segment.stage_outputs || {};
      const s00 = stages.stage_00_manifest || {};
      const s01 = stages.stage_01_frames_actions || {};
      const s02 = stages.stage_02_vlm_trajectories || {};
      const s03 = stages.stage_03_assemble || {};
      const s04 = stages.stage_04_sft_samples || {};
      const row = s00.row || {};
      const cfg = state.run?.run_config || {};
      const summary01 = s01.summary || {};
      const summary02 = s02.summary || {};
      const inputResolution = row.video_width && row.video_height ? `${row.video_width}x${row.video_height}` : "—";
      const outputResolution = scaledResolution(row);
      const vlmHeight = Number(summary02.vlm_frame_height ?? cfg.vlm_frame_height);
      const vlmResolution = scaledResolutionForHeight(row, vlmHeight);
      const sampleCount = (s03.samples || []).length;
      const finalRows = s04.rows || [];
      const stage01Stats = {
        sampled_fps: cfg.target_fps,
        sampled_resolution: outputResolution,
        frame_records_for_segment: s01.n_frame_records,
        non_noop_records_for_segment: s01.n_non_noop,
        noop_records_for_segment: Math.max(0, Number(s01.n_frame_records || 0) - Number(s01.n_non_noop || 0)),
        ...summary01,
      };
      const manifestStats = {
        raw_fps: row.video_fps,
        raw_resolution: inputResolution,
        raw_frames: row.video_frame_count,
        raw_duration_s: row.video_duration_s,
        video_size_bytes: row.video_size_bytes,
        keylog_events: row.n_keylog_events,
        keylog_duration_s: row.keylog_duration_s,
        event_counts: row.event_counts,
      };
      const stage02Stats = {
        vlm_resolution: vlmResolution,
        ...summary02,
      };
      return `
        ${renderLineageGraph(segment)}
        <div class="summary">
          ${metric("Raw Video", inputResolution, `${fmt(row.video_fps)} fps · ${secs(row.video_duration_s)}`)}
          ${metric("Sampled Frames", num(s01.n_frame_records), `${fmt(cfg.target_fps)} fps · ${outputResolution}`)}
          ${metric("NO_OP Capped", num(summary01.n_noop_capped), `keep first ${fmt(summary01.max_noop_run ?? cfg.stage01_max_noop_run)}`)}
          ${metric("VLM Output", num((s02.trajectories || []).length), `${vlmResolution} inputs · ${num((s02.candidates || []).length)} candidates · ${num((s02.merged || []).length)} merged`)}
          ${metric("SFT Samples", num(sampleCount), `${num(finalRows.length)} final rows`)}
        </div>
        ${stage("00", "Raw segment manifest", s00.dir, `
          <div class="grid-2">
            <div>
              <div class="subhead">Raw video + keylog</div>
              ${smallStats(manifestStats)}
            </div>
            <div>
              <div class="subhead">Manifest row</div>
              ${table([row], [
                {key:"segment_idx", label:"Segment"},
                {key:"video_fps", label:"FPS"},
                {key:"video_width", label:"Width"},
                {key:"video_height", label:"Height"},
                {key:"video_duration_s", label:"Duration"},
                {key:"n_keylog_events", label:"Keylog events"}
              ])}
            </div>
          </div>
        `)}
        ${stage("01", "Sampled frames + aligned actions", s01.dir, `
          <div class="grid-2">
            <div>
              <div class="subhead">Segment statistics</div>
              ${smallStats(stage01Stats)}
            </div>
            <div>
              <div class="subhead">Activity after NO_OP cap</div>
              ${renderTimeline(s01)}
              <div class="label" style="margin-top:8px">${num(s01.n_non_noop)} active frames from ${num(s01.n_frame_records)} kept frame records.</div>
            </div>
          </div>
          ${renderFrames(s01)}
        `)}
        ${stage("02", "VLM trajectories for this segment", s02.dir || "", `
          <div style="margin-bottom:10px">
            <span class="pill">${num((s02.trajectories || []).length)} trajectories</span>
            <span class="pill">${num((s02.candidates || []).length)} candidates</span>
            <span class="pill">${num((s02.merged || []).length)} merged</span>
          </div>
          <div style="margin-bottom:14px">
            <div class="subhead">Annotation render</div>
            ${smallStats(stage02Stats)}
          </div>
          <div class="grid-2">
            <div>
              <div class="subhead">Trajectories sent forward</div>
              ${renderTrajectoryTable(s02.trajectories)}
            </div>
            <div>
              <div class="subhead">Candidates / merged boundaries</div>
              ${renderTrajectoryTable((s02.merged || []).length ? s02.merged : s02.candidates)}
            </div>
          </div>
        `)}
        ${stage("03", "Assembled SFT samples from this segment", s03.dir || "", `
          ${renderSamples(s03, s04)}
        `)}
        ${stage("04", "Final chat rows and buckets", s04.dir || "", `
          ${table(finalRows, [
            {key:"sample_id", label:"Sample"},
            {key:"bucket", label:"Bucket"},
            {key:"token_count", label:"Tokens"},
            {key:"n_frames", label:"Frames"},
            {key:"duration_s", label:"Duration", render: r => secs(r.duration_s)},
            {key:"instruction", label:"Instruction"}
          ])}
        `)}
      `;
    }

    function renderClip(clip) {
      const stages = clip.stages;
      const s00 = stages.stage_00_manifest;
      const s01 = stages.stage_01_frames_actions;
      const s02 = stages.stage_02_vlm_trajectories;
      const s03 = stages.stage_03_assemble;
      const s04 = stages.stage_04_sft_samples;
      const raw = s00.summary || {};
      const fr = s01.summary || {};
      const a = s03.summary || {};
      const b = s04.summary || {};
      const annotation = (s02.raw && s02.raw.annotation_source) || s02.summary.annotation_source || "missing";
      const dry = (s02.raw && s02.raw.dry_run) || false;
      const body = `
        <div class="summary">
          ${metric("Raw Duration", raw.total_video_duration_s ? `${raw.total_video_duration_s}s` : "—", `${num(raw.n_segments)} raw segments`)}
          ${metric("Stage 01 Frames", num(fr.n_frames), `${num(fr.n_non_noop)} active`)}
          ${metric("NO_OP Dropped", num(fr.n_noop_capped), `cap ${fmt(fr.max_noop_run)}`)}
          ${metric("Trajectories", num(a.n_samples), `${num(a.n_rejected)} rejected`)}
          ${metric("SFT Rows", num(b.n_emitted), b.token_count_mode ? `tokens: ${b.token_count_mode}` : "")}
        </div>
        ${stage("00", "Manifest: raw MP4 + keylog discovery", s00.dir, `
          <div class="grid-2">
            ${smallStats(s00.summary)}
            ${table(s00.manifest || [], [
              {key:"segment_idx", label:"Segment"},
              {key:"video_fps", label:"FPS"},
              {key:"video_width", label:"Width"},
              {key:"video_height", label:"Height"},
              {key:"video_duration_s", label:"Duration"},
              {key:"n_keylog_events", label:"Keylog events"}
            ])}
          </div>
        `)}
        ${stage("01", "Frames + keylog actions", s01.dir, `
          <div class="grid-2">
            ${smallStats(s01.summary)}
            <div>
              <div class="label">Activity timeline</div>
              ${renderTimeline(s01)}
              <div class="label" style="margin-top:8px">Tall bars indicate more non-NOOP frames in that time slice.</div>
            </div>
          </div>
          ${renderFrames(s01)}
        `)}
        ${stage("02", "VLM task boundaries + instructions", s02.dir, `
          <div style="margin-bottom:10px">
            <span class="pill ${dry ? "bad" : "ok"}">${esc(annotation)}</span>
            <span class="pill">${num((s02.candidates || []).length)} candidates</span>
            <span class="pill">${num((s02.merged || []).length)} merged</span>
          </div>
          <div class="grid-2">
            ${smallStats(s02.summary)}
            ${table((s02.raw && s02.raw.trajectories || []).slice(0, 12), [
              {key:"start_time_s", label:"Start"},
              {key:"end_time_s", label:"End"},
              {key:"confidence", label:"Conf"},
              {key:"instruction", label:"Instruction"}
            ])}
          </div>
        `)}
        ${stage("03", "SFT trajectory assembly", s03.dir, `
          <div class="grid-2">
            ${smallStats(s03.summary)}
            ${renderSamples(s03, s04)}
          </div>
        `)}
        ${stage("04", "Chat JSONL + length buckets", s04.dir, `
          <div class="grid-2">
            ${smallStats(s04.summary)}
            <div>
              <div class="label">Bucket files</div>
              <p>${(s04.bucket_files || []).map(f => `<code>${esc(f)}</code>`).join(" ") || "—"}</p>
              ${table(s04.manifest || [], [
                {key:"sample_id", label:"Sample"},
                {key:"bucket", label:"Bucket"},
                {key:"token_count", label:"Tokens"},
                {key:"n_frames", label:"Frames"},
                {key:"duration_s", label:"Duration"}
              ])}
            </div>
          </div>
        `)}
      `;
      return body;
    }

    init().catch(err => {
      $("content").innerHTML = `<div class="empty">${esc(err.stack || err.message)}</div>`;
    });
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "V3PipelineDashboard/0.1"

    def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_text(self, value: str, status: HTTPStatus = HTTPStatus.OK, content_type: str = "text/html; charset=utf-8") -> None:
        payload = value.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if path == "/":
                self.send_text(INDEX_HTML)
            elif path == "/api/runs":
                self.send_json(list_runs())
            elif path.startswith("/api/run/"):
                run_name = urllib.parse.unquote(path.removeprefix("/api/run/"))
                if not re.match(r"^[A-Za-z0-9_.-]+$", run_name):
                    self.send_json({"error": "bad run name"}, HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(build_run_summary(run_name))
            elif path == "/api/sample_frames":
                run_name = query.get("run", [""])[0]
                sample_id = query.get("sample_id", [""])[0]
                if not re.match(r"^[A-Za-z0-9_.-]+$", run_name):
                    self.send_json({"error": "bad run name"}, HTTPStatus.BAD_REQUEST)
                    return
                if not sample_id:
                    self.send_json({"error": "missing sample_id"}, HTTPStatus.BAD_REQUEST)
                    return
                payload = build_sample_frame_player(run_name, sample_id)
                if payload is None:
                    self.send_json({"error": "sample not found"}, HTTPStatus.NOT_FOUND)
                    return
                self.send_json(payload)
            elif path == "/image":
                raw_path = query.get("path", [""])[0]
                image_path = safe_image_path(raw_path)
                if image_path is None or not image_path.exists():
                    self.send_response(HTTPStatus.NOT_FOUND)
                    self.end_headers()
                    return
                content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                data = image_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=60")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - dashboard should report errors in-browser
            self.send_json({"error": type(exc).__name__, "message": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")


class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with ReusableThreadingTCPServer((args.host, args.port), DashboardHandler) as httpd:
        httpd.daemon_threads = True
        url = f"http://{args.host}:{args.port}/"
        print(f"Serving v3 pipeline dashboard at {url}")
        if args.open:
            webbrowser.open(url)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
