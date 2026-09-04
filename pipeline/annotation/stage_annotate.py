"""Annotate Crowd-Cast frames with the canonical describe/extract prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.annotation.lib.driver import run_driver
from pipeline.annotation.lib.labeler import Labeler, LabelerConfig
from pipeline.annotation.lib.prompts import PromptPack
from pipeline.annotation.lib.units import AnnotationUnit, build_units
from pipeline.annotation.methods.describe_extract.annotator import Context, run_unit
from pipeline.lib.action_format import WindowKeyboard, format_segment
from pipeline.lib.common import ensure_dir, write_json, write_jsonl
from pipeline.lib.events import load_events
from pipeline.lib.goals import validate_goal_row, view_span_to_master
from pipeline.lib.views import FilterArtifact

METHOD = "describe_extract"
PROMPTS_PATH = Path(__file__).parent / "methods" / METHOD / "prompts.yaml"
CONTEXT_LIMIT = 262144
CONTEXT_SAFETY_MARGIN = 28000
WINDOW_SNAP_SLACK = 25
WINDOW_TAIL_BUFFER = 5
EST_TOKENS_PER_FRAME = 1500


def _segment_keyboard(view, keylog_path: str | None) -> list[WindowKeyboard]:
    events, _ = load_events(Path(keylog_path)) if keylog_path else ([], None)
    return format_segment(
        events,
        view.windows(),
        view.dead_zones,
        master_fps=view.master_fps,
    ).keyboard


def _goal_rows(
    unit: AnnotationUnit,
    result: dict[str, Any],
    *,
    model: str,
    fps: float,
    prompt_sha: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, goal in enumerate(result["goals"]):
        start, end = view_span_to_master(
            unit.view,
            int(goal["start_frame"]),
            int(goal["end_frame"]) + 1,
        )
        row = {
            "goal_id": f"{unit.unit_id}_g{index:02d}",
            "segment_id": unit.view.segment_id,
            "recording_id": unit.view.recording_id,
            "start_master_idx": start,
            "end_master_idx": end,
            "instruction": goal["instruction"],
            "instruction_variants": goal["instruction_variants"],
            "anchor": goal["anchor"],
            "grounding": goal["grounding"],
            "method": METHOD,
            "model": model,
            "prompt_pack_sha": prompt_sha,
            "unit_id": unit.unit_id,
            "annotation_fps": fps,
        }
        validate_goal_row(row)
        rows.append(row)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter_dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target_tpm", type=float, required=True)
    parser.add_argument("--max_workers", type=int, default=64)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.target_tpm <= 0:
        raise SystemExit("--target-tpm must be positive")
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")
    artifact = FilterArtifact(args.filter_dir)
    stride = artifact.stride_for(args.fps)
    prompts = PromptPack(PROMPTS_PATH)
    labeler = Labeler(LabelerConfig.from_env(model=args.model))

    output = ensure_dir(args.output_dir)
    units_dir = ensure_dir(output / "units")
    describe_dir = ensure_dir(output / "describe")
    calls_dir = ensure_dir(output / "calls")
    progress_path = output / "progress.jsonl"
    items = [
        {"id": str(row["segment_id"]), "row": row} for row in artifact.usable_rows()
    ]
    if not items:
        raise ValueError(f"no usable Crowd-Cast segments in {artifact.dir}")

    def estimated_tokens(item: dict[str, Any]) -> float:
        kept = int(item["row"].get("n_kept") or 0)
        frames = max(1, kept // stride)
        return 2 * frames * EST_TOKENS_PER_FRAME + labeler.config.max_completion_tokens

    def process(item: dict[str, Any]) -> dict[str, Any]:
        segment_id = item["id"]
        view = artifact.segment_view(segment_id, args.fps)
        if not view.frames:
            raise ValueError(f"selected Crowd-Cast view is empty: {segment_id}")
        keyboard = _segment_keyboard(view, view.keylog_path)
        units = build_units(
            view,
            keyboard,
            context_limit=CONTEXT_LIMIT,
            completion_reserve=labeler.config.max_completion_tokens,
            safety_margin=CONTEXT_SAFETY_MARGIN,
            snap_slack=WINDOW_SNAP_SLACK,
            tail_buffer=WINDOW_TAIL_BUFFER,
        )
        goal_count = 0
        tokens = 0
        for unit in units:
            unit_path = units_dir / f"{unit.unit_id}.json"
            result = run_unit(
                unit,
                Context(
                    labeler=labeler,
                    prompts=prompts,
                    cache_dir=calls_dir / unit.unit_id,
                ),
            )
            goals = _goal_rows(
                unit,
                result,
                model=args.model,
                fps=args.fps,
                prompt_sha=prompts.sha,
            )
            narration = result["narration"]
            (describe_dir / f"{unit.unit_id}.txt").write_text(narration + "\n")
            write_json(
                unit_path,
                {
                    "unit_id": unit.unit_id,
                    "segment_id": view.segment_id,
                    "recording_id": view.recording_id,
                    "window_index": unit.window_index,
                    "n_windows": unit.n_windows,
                    "sent_view_range": [unit.lo, unit.hi + unit.tail_buffer],
                    "owned_view_range": [unit.lo, unit.hi],
                    "model": args.model,
                    "fps": args.fps,
                    "actual_tokens": result["actual_tokens"],
                    "goals": goals,
                },
            )
            goal_count += len(goals)
            tokens += int(result["actual_tokens"])
        return {"n_units": len(units), "n_goals": goal_count, "actual_tokens": tokens}

    run_driver(
        items,
        item_id=lambda item: item["id"],
        est_tokens=estimated_tokens,
        run_item=process,
        progress_path=progress_path,
        target_tpm=args.target_tpm,
        max_workers=args.max_workers,
    )

    goals: list[dict[str, Any]] = []
    for unit_path in sorted(units_dir.glob("*.json")):
        unit = json.loads(unit_path.read_text())
        for goal in unit["goals"]:
            validate_goal_row(goal)
            goals.append(goal)
    if not goals:
        raise ValueError("describe/extract produced no Crowd-Cast goals")
    goals.sort(
        key=lambda goal: (
            goal["segment_id"],
            goal["start_master_idx"],
            goal["end_master_idx"],
        )
    )
    write_jsonl(output / "goals.jsonl", goals)
    prompts.snapshot_to(output)
    write_json(
        output / "manifest.json",
        {
            "artifact_type": "crowdcast_describe_extract_goals",
            "schema_version": 1,
            "goals": "goals.jsonl",
            "method": METHOD,
            "input_kind": "frames",
            "prompt_pack_sha": prompts.sha,
            "model": args.model,
            "fps": args.fps,
            "stride": stride,
            "master_fps": artifact.master_fps,
            "n_goals": len(goals),
            "master_store_id": artifact.master_store_id,
            "filter_id": artifact.filter_id,
        },
    )


if __name__ == "__main__":
    main()
