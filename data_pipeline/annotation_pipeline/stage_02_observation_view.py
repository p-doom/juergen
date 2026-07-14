#!/usr/bin/env python3
"""Stage 02: build an observation view by joining frames to timestamped events."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from annotation_pipeline import config
from annotation_pipeline.common import (
    ACTION_EVENT_KINDS,
    action_bin_to_dict,
    aggregate_event_records,
    ensure_dir,
    event_activity,
    events_have_submission,
    is_noop_action_bin,
    read_jsonl,
    write_json,
    write_jsonl,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _selected_frames(
    frames: list[dict[str, Any]], *, base_fps: float, observation_fps: float
) -> list[dict[str, Any]]:
    if observation_fps <= 0 or observation_fps > base_fps:
        raise ValueError("observation_fps must be > 0 and <= base_fps")
    stride = round(base_fps / observation_fps)
    if stride <= 0 or abs(base_fps / stride - observation_fps) > 1e-9:
        raise ValueError(
            f"observation_fps={observation_fps} must divide base_fps={base_fps} exactly"
        )
    return [frame for frame in frames if int(frame["local_frame_idx"]) % stride == 0]


def _thin_idle_runs(
    observations: list[dict[str, Any]], *, keep_head: int, keep_tail: int
) -> tuple[list[dict[str, Any]], int]:
    if keep_head < 0 or keep_tail < 0:
        raise ValueError("idle keep counts must be non-negative")
    keep = [True] * len(observations)
    dropped = 0
    i = 0
    while i < len(observations):
        if not observations[i]["is_noop"]:
            i += 1
            continue
        j = i
        while j < len(observations) and observations[j]["is_noop"]:
            j += 1
        if j - i > keep_head + keep_tail:
            for index in range(i + keep_head, j - keep_tail):
                keep[index] = False
                dropped += 1
        i = j
    return [item for index, item in enumerate(observations) if keep[index]], dropped


def build_observation_view(
    *,
    frames: list[dict[str, Any]],
    events: list[dict[str, Any]],
    segment_summaries: list[dict[str, Any]],
    base_fps: float,
    observation_fps: float,
    idle_keep_head: int,
    idle_keep_tail: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frames_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    events_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duration_by_segment = {
        str(item["segment_id"]): float(item["duration_s"])
        for item in segment_summaries
        if "duration_s" in item
    }
    for frame in frames:
        frames_by_segment[str(frame["segment_id"])].append(frame)
    for event in events:
        events_by_segment[str(event["segment_id"])].append(event)

    output: list[dict[str, Any]] = []
    per_segment: list[dict[str, Any]] = []
    for segment_id, segment_frames in sorted(
        frames_by_segment.items(), key=lambda item: int(item[1][0]["segment_idx"])
    ):
        segment_frames.sort(key=lambda item: float(item["local_time_s"]))
        selected = _selected_frames(
            segment_frames, base_fps=base_fps, observation_fps=observation_fps
        )
        segment_events = sorted(
            events_by_segment.get(segment_id, []),
            key=lambda item: (int(item["timestamp_us"]), int(item["source_event_idx"])),
        )
        duration_s = duration_by_segment[segment_id]
        observations: list[dict[str, Any]] = []
        event_index = 0
        held: set[str] = set()
        typing_burst_id = -1
        previous_activity = "idle"
        for index, frame in enumerate(selected):
            start_s = float(frame["local_time_s"])
            end_s = (
                float(selected[index + 1]["local_time_s"])
                if index + 1 < len(selected)
                else duration_s
            )
            while (
                event_index < len(segment_events)
                and float(segment_events[event_index]["local_time_s"]) < start_s
            ):
                event_index += 1
            end_index = event_index
            while (
                end_index < len(segment_events)
                and float(segment_events[end_index]["local_time_s"]) < end_s
            ):
                end_index += 1
            interval_events = segment_events[event_index:end_index]
            event_index = end_index
            executable = [e for e in interval_events if e["kind"] in ACTION_EVENT_KINDS]
            activity = event_activity(executable)
            if activity == "type" and previous_activity != "type":
                typing_burst_id += 1
            action_bin = aggregate_event_records(executable, held=held)
            observation = dict(frame)
            observation.update(
                {
                    "interval_start_s": round(start_s, 6),
                    "interval_end_s": round(end_s, 6),
                    "interval_start_global_s": round(float(frame["global_time_s"]), 6),
                    "interval_end_global_s": round(
                        float(frame["global_time_s"]) + (end_s - start_s), 6
                    ),
                    "events": executable,
                    "action_bin": action_bin_to_dict(action_bin),
                    "activity": activity,
                    "typing_burst_id": typing_burst_id if activity == "type" else None,
                    "has_submission": events_have_submission(executable),
                    "is_noop": is_noop_action_bin(action_bin),
                }
            )
            observations.append(observation)
            previous_activity = activity

        kept, n_idle_dropped = _thin_idle_runs(
            observations, keep_head=idle_keep_head, keep_tail=idle_keep_tail
        )
        output.extend(kept)
        per_segment.append(
            {
                "segment_id": segment_id,
                "n_base_frames": len(segment_frames),
                "n_view_frames_before_thinning": len(observations),
                "n_observations": len(kept),
                "n_idle_dropped": n_idle_dropped,
                "n_non_noop": sum(not item["is_noop"] for item in kept),
            }
        )

    for observation_idx, observation in enumerate(output):
        observation["observation_idx"] = observation_idx
    return output, {"per_segment": per_segment}


def materialize_observation_view(
    *,
    base_dir: Path,
    output_dir: Path,
    view_name: str,
    observation_fps: float,
    idle_keep_head: int,
    idle_keep_tail: int,
) -> dict[str, Any]:
    if not view_name.strip():
        raise ValueError("view_name must not be empty")
    base_manifest = _read_json(base_dir / "manifest.json")
    if base_manifest.get("stage") != "base_modalities":
        raise ValueError(f"Not a base-modalities artifact: {base_dir}")
    base_fps = float(base_manifest["base_fps"])
    frames = read_jsonl(base_dir / "frames.jsonl")
    events = read_jsonl(base_dir / "events.jsonl")
    summaries = json.loads((base_dir / "segment_summaries.json").read_text())
    observations, stats = build_observation_view(
        frames=frames,
        events=events,
        segment_summaries=summaries,
        base_fps=base_fps,
        observation_fps=observation_fps,
        idle_keep_head=idle_keep_head,
        idle_keep_tail=idle_keep_tail,
    )
    output_dir = ensure_dir(output_dir)
    write_jsonl(output_dir / "observations.jsonl", observations)
    manifest = {
        "stage": "observation_view",
        "schema_version": 1,
        "view_name": view_name,
        "source_base_dir": str(base_dir.resolve()),
        "base_fps": base_fps,
        "observation_fps": observation_fps,
        "idle_keep_head": idle_keep_head,
        "idle_keep_tail": idle_keep_tail,
        "n_observations": len(observations),
        **stats,
        "files": {"observations": "observations.jsonl"},
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-name", required=True)
    parser.add_argument("--observation-fps", type=float, required=True)
    parser.add_argument("--idle-keep-head", type=int, default=config.DEFAULT_IDLE_KEEP_HEAD)
    parser.add_argument("--idle-keep-tail", type=int, default=config.DEFAULT_IDLE_KEEP_TAIL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize_observation_view(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        view_name=args.view_name,
        observation_fps=args.observation_fps,
        idle_keep_head=args.idle_keep_head,
        idle_keep_tail=args.idle_keep_tail,
    )
    print(f"Wrote {manifest['n_observations']} observations to {args.output_dir}")


if __name__ == "__main__":
    main()
