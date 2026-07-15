#!/usr/bin/env python3
"""Stage 05: slice finalized goals into format-neutral structured trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from annotation_pipeline.common import ensure_dir, read_jsonl, write_json, write_jsonl


def _selected_observations(
    observations: list[dict[str, Any]], start_idx: int, end_idx: int
) -> list[dict[str, Any]]:
    return [item for item in observations if start_idx <= int(item["global_frame_idx"]) <= end_idx]


def assemble_trajectories(
    observations: list[dict[str, Any]],
    goals: list[dict[str, Any]],
    *,
    min_observations: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not observations:
        raise ValueError("observations must not be empty")
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1")
    segment_ids = {str(item["segment_id"]) for item in observations}
    recording_ids = {str(item["recording_id"]) for item in observations}
    if len(segment_ids) != 1 or len(recording_ids) != 1:
        raise ValueError("observations must belong to exactly one segment and recording")
    ordered_observations = sorted(observations, key=lambda item: int(item["global_frame_idx"]))
    recording_id = recording_ids.pop()
    clip_id = segment_ids.pop()
    trajectories: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    ordered_goals = sorted(
        goals,
        key=lambda item: (int(item["start_frame_idx"]), int(item["end_frame_idx"])),
    )
    for goal in ordered_goals:
        goal_idx = int(goal["goal_idx"])
        start_idx = int(goal["start_frame_idx"])
        end_idx = int(goal["end_frame_idx"])
        if end_idx < start_idx:
            rejected.append({"goal_idx": goal_idx, "reason": "bad_frame_bounds", "goal": goal})
            continue
        selected = _selected_observations(ordered_observations, start_idx, end_idx)
        if len(selected) < min_observations:
            rejected.append({"goal_idx": goal_idx, "reason": "too_few_observations", "goal": goal})
            continue
        first = selected[0]
        last = selected[-1]
        steps = [
            {
                "global_frame_idx": int(item["global_frame_idx"]),
                "local_frame_idx": int(item["local_frame_idx"]),
                "local_time_s": float(item["local_time_s"]),
                "global_time_s": float(item["global_time_s"]),
                "interval_start_s": float(item["interval_start_s"]),
                "interval_end_s": float(item["interval_end_s"]),
                "source_frame_idx": int(item["source_frame_idx"]),
                "image_path": str(item["image_path"]),
                "events": list(item["events"]),
                "action_bin": dict(item["action_bin"]),
                "is_noop": bool(item["is_noop"]),
            }
            for item in selected
        ]
        trajectories.append(
            {
                "trajectory_id": f"goal{goal_idx:04d}",
                "recording_id": recording_id,
                "clip_id": clip_id,
                "instruction": str(goal["instruction"]),
                "instruction_variants": list(goal["instruction_variants"]),
                "anchor": str(goal["anchor"]),
                "grounding": str(goal["grounding"]),
                "boundary_policy": str(goal["boundary_policy"]),
                "start_frame_idx": int(first["global_frame_idx"]),
                "end_frame_idx": int(last["global_frame_idx"]),
                "start_time_s": float(first["global_time_s"]),
                "end_time_s": float(last["global_time_s"]),
                "source_frame_start": int(first["source_frame_idx"]),
                "source_frame_end": int(last["source_frame_idx"]),
                "n_observations": len(steps),
                "n_non_noop": sum(not step["is_noop"] for step in steps),
                "source_goal": goal,
                "steps": steps,
            }
        )
    return trajectories, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-observations", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations = read_jsonl(args.observations)
    goals = read_jsonl(args.goals)
    trajectories, rejected = assemble_trajectories(
        observations, goals, min_observations=args.min_observations
    )
    output_dir = ensure_dir(args.output_dir)
    write_jsonl(output_dir / "trajectories.jsonl", trajectories)
    write_jsonl(output_dir / "rejected.jsonl", rejected)
    write_json(
        output_dir / "manifest.json",
        {
            "stage": "structured_trajectory_assembly",
            "schema_version": 1,
            "source_observations": str(args.observations.resolve()),
            "source_goals": str(args.goals.resolve()),
            "n_goals": len(goals),
            "n_trajectories": len(trajectories),
            "n_rejected": len(rejected),
            "reject_reasons": {
                reason: sum(1 for item in rejected if item["reason"] == reason)
                for reason in sorted({item["reason"] for item in rejected})
            },
            "files": {"trajectories": "trajectories.jsonl", "rejected": "rejected.jsonl"},
        },
    )
    print(f"Wrote {len(trajectories)} structured trajectories to {output_dir}")


if __name__ == "__main__":
    main()
