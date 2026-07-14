#!/usr/bin/env python3
"""Run Stage 05 structured assembly and Stage 06 SFT projection over an annotation run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from annotation_pipeline import config
from annotation_pipeline.common import ensure_dir, read_jsonl, write_json, write_jsonl
from annotation_pipeline.stage_02_observation_view import materialize_observation_view
from annotation_pipeline.stage_05_assemble_trajectories import assemble_trajectories
from annotation_pipeline.stage_06_project_sft import ensure_empty_dir, project_sft

PROVENANCE_KEYS = ("user_id", "version", "recording_id", "video_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--modalities-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-observations", type=int, default=1)
    parser.add_argument("--training-fps", type=float, default=config.DEFAULT_TRAINING_FPS)
    parser.add_argument(
        "--training-idle-keep-head", type=int, default=config.DEFAULT_IDLE_KEEP_HEAD
    )
    parser.add_argument(
        "--training-idle-keep-tail", type=int, default=config.DEFAULT_IDLE_KEEP_TAIL
    )
    parser.add_argument("--include-variants", action="store_true")
    parser.add_argument(
        "--split-group", choices=("recording_id", "clip_id"), default="recording_id"
    )
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--system-prompt-text")
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--terminal-token")
    parser.add_argument(
        "--terminal-mode",
        choices=("none", "replace_final_assistant", "append_to_final_assistant"),
        default="none",
    )
    parser.add_argument("--structured-only", action="store_true")
    return parser.parse_args()


def _annotation_units(run_dir: Path) -> list[Path]:
    model_dirs = [
        path
        for path in run_dir.iterdir()
        if path.is_dir() and path.name != "_modalities" and (path / "clips").is_dir()
    ]
    return [unit for model in model_dirs for unit in (model / "clips").iterdir() if unit.is_dir()]


def merge_window_goals(
    parts: list[tuple[int, int, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Order window-local goals and assign parent-wide stable goal indices."""

    expected_counts = {n_windows for _window_idx, n_windows, _goals in parts}
    if len(expected_counts) != 1:
        raise ValueError("Annotation windows disagree about n_windows")
    n_windows = expected_counts.pop()
    window_indices = [window_idx for window_idx, _n_windows, _goals in parts]
    if len(window_indices) != len(set(window_indices)) or set(window_indices) != set(
        range(n_windows)
    ):
        raise ValueError(
            f"Missing annotation windows: expected 0..{n_windows - 1}, got {sorted(window_indices)}"
        )

    merged: list[dict[str, Any]] = []
    for window_idx, _n_windows, window_goals in sorted(parts):
        for window_goal_idx, goal in enumerate(window_goals):
            item = dict(goal)
            item["source_window_idx"] = window_idx
            item["source_window_goal_idx"] = int(goal.get("goal_idx", window_goal_idx))
            item["goal_idx"] = len(merged)
            merged.append(item)
    return merged


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    modalities_root = (
        args.modalities_root.resolve() if args.modalities_root else run_dir / "_modalities"
    )
    output_root = ensure_dir(args.out)
    if args.structured_only:
        stale_sft_dir = ensure_empty_dir(output_root / "stage_06_sft", overwrite=True)
        stale_sft_dir.rmdir()

    by_parent: dict[str, list[tuple[int, int, list[dict[str, Any]]]]] = defaultdict(list)
    for unit in _annotation_units(run_dir):
        boundary_dir = unit / "stage_04_boundaries"
        manifest_path = boundary_dir / "manifest.json"
        goals_path = boundary_dir / "goals.jsonl"
        if not manifest_path.exists() or not goals_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        parent = str(manifest["parent_segment_id"])
        by_parent[parent].append(
            (
                int(manifest["window_index"]),
                int(manifest["n_windows"]),
                read_jsonl(goals_path),
            )
        )
    if not by_parent:
        raise RuntimeError(f"No completed Stage-04 boundary artifacts under {run_dir}")

    all_trajectories: list[dict[str, Any]] = []
    all_rejected: list[dict[str, Any]] = []
    training_views_stage = ensure_empty_dir(output_root / "stage_02_training_views", overwrite=True)
    training_views_root = ensure_dir(training_views_stage / "clips")
    for parent, parts in sorted(by_parent.items()):
        base_dir = modalities_root / "clips" / parent / "stage_01_base"
        training_view_dir = training_views_root / parent
        materialize_observation_view(
            base_dir=base_dir,
            output_dir=training_view_dir,
            view_name="training",
            observation_fps=args.training_fps,
            idle_keep_head=args.training_idle_keep_head,
            idle_keep_tail=args.training_idle_keep_tail,
        )
        observations_path = training_view_dir / "observations.jsonl"
        observations = read_jsonl(observations_path)
        if not observations:
            print(f"skip {parent}: no observations", file=sys.stderr)
            continue
        goals = merge_window_goals(parts)
        trajectories, rejected = assemble_trajectories(
            observations, goals, min_observations=args.min_observations
        )
        manifest_rows = read_jsonl(
            modalities_root / "clips" / parent / "stage_00" / "manifest.jsonl"
        )
        source = manifest_rows[0]
        provenance = {key: source.get(key) for key in PROVENANCE_KEYS}
        provenance["parent_segment_id"] = parent
        for trajectory in trajectories:
            trajectory.update(provenance)
        for item in rejected:
            item["parent_segment_id"] = parent
        all_trajectories.extend(trajectories)
        all_rejected.extend(rejected)

    structured_dir = ensure_empty_dir(output_root / "stage_05_trajectories", overwrite=True)
    trajectories_path = structured_dir / "trajectories.jsonl"
    write_jsonl(trajectories_path, all_trajectories)
    write_jsonl(structured_dir / "rejected.jsonl", all_rejected)
    write_json(
        structured_dir / "manifest.json",
        {
            "stage": "structured_trajectory_assembly",
            "schema_version": 1,
            "source_run_dir": str(run_dir),
            "source_training_views": str(training_views_root.parent),
            "training_fps": args.training_fps,
            "training_idle_keep_head": args.training_idle_keep_head,
            "training_idle_keep_tail": args.training_idle_keep_tail,
            "n_parents": len(by_parent),
            "n_trajectories": len(all_trajectories),
            "n_rejected": len(all_rejected),
            "files": {"trajectories": "trajectories.jsonl", "rejected": "rejected.jsonl"},
        },
    )
    print(f"[stage 05] wrote {len(all_trajectories)} structured trajectories")
    if args.structured_only:
        return

    sft_dir = output_root / "stage_06_sft"
    manifest = project_sft(
        trajectories_path=trajectories_path,
        output_dir=sft_dir,
        include_variants=args.include_variants,
        split_group=args.split_group,
        val_frac=args.val_frac,
        seed=args.seed,
        image_path_mode="absolute",
        system_prompt_text=args.system_prompt_text,
        system_prompt_file=args.system_prompt_file,
        terminal_token=args.terminal_token,
        terminal_mode=args.terminal_mode,
        overwrite=True,
    )
    print(f"[stage 06] wrote {manifest['n_samples']} SFT samples to {sft_dir}")


if __name__ == "__main__":
    main()
