#!/usr/bin/env python3
"""Stage 04: finalize visual goal bounds with an explicit boundary policy."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

from annotation_pipeline.common import ensure_dir, read_jsonl, write_json, write_jsonl

BOUNDARY_POLICIES = ("vision_only", "keylog_refined")


def _snap_typed_goal_starts(
    goals: list[dict[str, Any]], observations: list[dict[str, Any]]
) -> None:
    positions = {int(item["global_frame_idx"]): index for index, item in enumerate(observations)}
    for goal in goals:
        start = int(goal["start_frame_idx"])
        if start not in positions:
            continue
        position = positions[start]
        current_is_type = observations[position]["activity"] == "type"
        previous_is_type = position > 0 and observations[position - 1]["activity"] == "type"
        if not current_is_type and not previous_is_type:
            continue
        while (
            position > 0
            and observations[position - 1]["activity"] == "type"
            and not observations[position - 1]["has_submission"]
        ):
            position -= 1
        goal["start_frame_idx"] = int(observations[position]["global_frame_idx"])
        if int(goal["end_frame_idx"]) < int(goal["start_frame_idx"]):
            goal["end_frame_idx"] = goal["start_frame_idx"]


def _attach_time_bounds(goals: list[dict[str, Any]], observations: list[dict[str, Any]]) -> None:
    by_frame = {int(item["global_frame_idx"]): item for item in observations}
    for goal in goals:
        start_idx = int(goal["start_frame_idx"])
        end_idx = int(goal["end_frame_idx"])
        if start_idx not in by_frame or end_idx not in by_frame:
            raise ValueError(f"Goal boundary is outside the annotation view: {start_idx}-{end_idx}")
        goal["start_time_s"] = float(by_frame[start_idx]["interval_start_global_s"])
        goal["end_time_s"] = float(by_frame[end_idx]["interval_end_global_s"])


def _enforce_nonoverlap(goals: list[dict[str, Any]], observations: list[dict[str, Any]]) -> None:
    frame_indices = sorted(int(item["global_frame_idx"]) for item in observations)
    goals.sort(key=lambda item: (int(item["start_frame_idx"]), int(item["end_frame_idx"])))
    for earlier, later in pairwise(goals):
        if int(earlier["end_frame_idx"]) >= int(later["start_frame_idx"]):
            candidates = [
                frame_idx
                for frame_idx in frame_indices
                if int(earlier["start_frame_idx"]) <= frame_idx < int(later["start_frame_idx"])
            ]
            if not candidates:
                raise ValueError(
                    "Cannot make goals non-overlapping because they share a start observation"
                )
            earlier["end_frame_idx"] = candidates[-1]


def refine_boundaries(
    proposals: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    *,
    policy: str,
) -> list[dict[str, Any]]:
    if policy not in BOUNDARY_POLICIES:
        raise ValueError(f"Unknown boundary policy: {policy}")
    goals: list[dict[str, Any]] = []
    for proposal in proposals:
        if proposal.get("start_frame") is None or proposal.get("end_frame") is None:
            continue
        goals.append(
            {
                "instruction": str(proposal["instruction"]),
                "instruction_variants": list(proposal["instruction_variants"]),
                "anchor": str(proposal["anchor"]),
                "grounding": str(proposal["grounding"]),
                "start_frame_idx": int(proposal["start_frame"]),
                "end_frame_idx": int(proposal["end_frame"]),
            }
        )
    if policy == "keylog_refined":
        _snap_typed_goal_starts(goals, observations)
    _enforce_nonoverlap(goals, observations)
    _attach_time_bounds(goals, observations)
    for goal_idx, goal in enumerate(goals):
        goal["goal_idx"] = goal_idx
        goal["boundary_policy"] = policy
    return goals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=BOUNDARY_POLICIES, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation = json.loads((args.annotation_dir / "annotation.json").read_text())
    if annotation.get("stage") != "visual_annotation":
        raise ValueError(f"Not a Stage-03 visual annotation: {args.annotation_dir}")
    proposals = read_jsonl(args.annotation_dir / "goal_proposals.jsonl")
    observations = read_jsonl(args.observations)
    goals = refine_boundaries(proposals, observations, policy=args.policy)
    output_dir = ensure_dir(args.output_dir)
    write_jsonl(output_dir / "goals.jsonl", goals)
    write_json(
        output_dir / "manifest.json",
        {
            "stage": "boundary_refinement",
            "schema_version": 1,
            "source_annotation_dir": str(args.annotation_dir.resolve()),
            "source_observations": str(args.observations.resolve()),
            "recording_id": annotation["recording_id"],
            "segment_id": annotation["segment_id"],
            "parent_segment_id": annotation["parent_segment_id"],
            "window_index": annotation["window_index"],
            "n_windows": annotation["n_windows"],
            "source_frame_range": annotation["source_frame_range"],
            "annotation_source": annotation["annotation_source"],
            "policy": args.policy,
            "n_proposals": len(proposals),
            "n_goals": len(goals),
            "files": {"goals": "goals.jsonl"},
        },
    )
    print(f"Refined {len(goals)} goals with policy={args.policy} -> {output_dir}")


if __name__ == "__main__":
    main()
