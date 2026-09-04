"""Build canonical Crowd-Cast conversations from frames, actions, and goals."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import grammars
from image_domain import image_domain
from pipeline.lib.action_format import format_segment
from pipeline.lib.common import ensure_dir, write_json_atomic, write_jsonl
from pipeline.lib.events import Window, load_events
from pipeline.lib.goals import (
    assert_same_artifact,
    goals_by_segment,
    load_goals,
    project_goals,
)
from pipeline.lib.manifest import (
    file_sha256_short,
    make_artifact_id,
    resolve_chat_artifact,
)
from pipeline.lib.views import FilterArtifact, build_segment_view

ACTION_FORMAT = "canonical"
GRAMMAR = "deltatype_v2"
JPEG_QUALITY = 92


def _text(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def build_messages(
    turns: list[tuple[str, str]],
    *,
    instruction: str,
    system_prompt: str,
) -> list[dict[str, Any]]:
    if not turns:
        raise ValueError("a conversation requires at least one turn")
    if not instruction.strip():
        raise ValueError("a conversation requires a non-empty instruction")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [_text(system_prompt)]}
    ]
    for index, (image, action) in enumerate(turns):
        content = [{"type": "image", "image": image}]
        if index == 0:
            content.insert(0, _text(instruction))
        messages.extend(
            (
                {"role": "user", "content": content},
                {"role": "assistant", "content": [_text(action)]},
            )
        )
    return messages


def build_segment_conversations(task: dict[str, Any]) -> dict[str, Any]:
    segment_id = str(task["index_row"]["segment_id"])
    filter_segment = task["filter_segment"]
    view = build_segment_view(filter_segment, fps=task["fps"])
    base = {
        "segment_id": segment_id,
        "recording_id": view.recording_id,
        "segment_idx": view.segment_idx,
        "alignment_status": view.alignment_status,
    }
    segment_goals = task["goals_by_segment"].get(segment_id, [])
    if any(goal["recording_id"] != view.recording_id for goal in segment_goals):
        raise ValueError(f"Crowd-Cast goal recording mismatch: {segment_id}")
    if not view.frames:
        if segment_goals:
            raise ValueError(
                f"Crowd-Cast goals cannot project onto empty view: {segment_id}"
            )
        return {**base, "status": "empty_view", "rows": []}

    keylog = Path(view.keylog_path) if view.keylog_path else None
    events = load_events(keylog)[0] if keylog else []
    if not segment_goals:
        return {**base, "status": "no_goals", "rows": []}
    projections, projection_stats = project_goals(
        segment_goals,
        view,
    )
    rows: list[dict[str, Any]] = []
    for projection in projections:
        goal = projection.goal
        formatted = format_segment(
            events,
            [
                Window(
                    frame.master_idx,
                    frame.win_start,
                    frame.win_end,
                )
                for frame in projection.frames
            ],
            view.dead_zones,
            master_fps=view.master_fps,
        )
        turns = [
            (str(frame.image), formatted.labels[index])
            for index, frame in enumerate(projection.frames)
        ]
        instruction = goal["instruction"]
        rows.append(
            {
                "conversation_id": f"{segment_id}:{goal['goal_id']}",
                **base,
                "target_fps": task["fps"],
                "action_format": ACTION_FORMAT,
                "goal_id": goal["goal_id"],
                "instruction": instruction,
                "start_master_idx": goal["start_master_idx"],
                "end_master_idx": goal["end_master_idx"],
                "snapped_start": projection.snapped_start,
                "n_frames": len(turns),
                "n_turns": len(turns),
                "n_non_noop": sum(action != "NO_OP" for _, action in turns),
                "messages": build_messages(
                    turns,
                    instruction=instruction,
                    system_prompt=task["system_prompt"],
                ),
            }
        )
    return {
        **base,
        "status": "ok" if rows else "no_projected_goals",
        "rows": rows,
        "projection": asdict(projection_stats),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter_dir", type=Path, required=True)
    parser.add_argument("--goals_dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count())
    return parser.parse_args(argv)


def resolve_goals(
    art: FilterArtifact, goals_dir: Path
) -> tuple[dict[str, list[dict]], str]:
    manifest_path = goals_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read Crowd-Cast goals manifest {manifest_path}: {exc}"
        ) from exc
    expected_fields = {
        "artifact_type",
        "filter_id",
        "fps",
        "goals",
        "goals_sha256",
        "input_kind",
        "master_fps",
        "master_store_id",
        "method",
        "model",
        "n_goals",
        "prompt_pack_sha",
        "prompts",
        "prompts_sha256",
        "schema_version",
        "stride",
    }
    required = {
        "artifact_type": "crowdcast_describe_extract_goals",
        "schema_version": 1,
        "method": "describe_extract",
        "input_kind": "frames",
        "goals": "goals.jsonl",
        "prompts": "prompts.yaml",
    }
    observed = {key: manifest.get(key) for key in required}
    if set(manifest) != expected_fields or observed != required:
        raise ValueError(
            f"goals contract mismatch: expected {required}, got {observed}"
        )
    annotation_fps = manifest.get("fps")
    if (
        isinstance(annotation_fps, bool)
        or not isinstance(annotation_fps, (int, float))
        or annotation_fps <= 0
        or manifest.get("master_fps") != art.master_fps
        or manifest.get("stride") != art.stride_for(annotation_fps)
        or not isinstance(manifest.get("model"), str)
        or not manifest["model"]
    ):
        raise ValueError(
            f"Crowd-Cast goals sampling contract mismatch: {manifest_path}"
        )
    goals_path = goals_dir / "goals.jsonl"
    expected_sha = manifest.get("goals_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(f"goals artifact has no SHA-256: {manifest_path}")
    observed_sha = file_sha256_short(goals_path, n=64)
    if observed_sha != expected_sha:
        raise ValueError(
            f"goals digest mismatch: expected {expected_sha}, got {observed_sha}"
        )
    prompts_path = goals_dir / "prompts.yaml"
    prompt_sha = file_sha256_short(prompts_path, n=64)
    if prompt_sha != manifest.get("prompts_sha256") or prompt_sha != manifest.get(
        "prompt_pack_sha"
    ):
        raise ValueError(f"Crowd-Cast prompt artifact mismatch: {prompts_path}")
    assert_same_artifact(
        str(manifest.get("master_store_id")),
        art.master_store_id,
        what="master_store_id",
    )
    assert_same_artifact(
        str(manifest.get("filter_id")), art.filter_id, what="filter_id"
    )
    goals = load_goals(goals_path)
    if not goals:
        raise ValueError(f"goals artifact is empty: {goals_dir}")
    if manifest.get("n_goals") != len(goals) or any(
        goal["prompt_pack_sha"] != prompt_sha or goal["model"] != manifest.get("model")
        for goal in goals
    ):
        raise ValueError(f"Crowd-Cast goal receipt mismatch: {manifest_path}")
    return goals_by_segment(goals), make_artifact_id(goals_dir)


def main() -> None:
    args = parse_args()
    output = ensure_dir(args.output_dir)
    (output / "manifest.json").unlink(missing_ok=True)
    if args.num_workers <= 0:
        raise SystemExit("--num_workers must be positive")
    art = FilterArtifact(args.filter_dir)
    stride = art.stride_for(args.fps)
    goals, goals_id = resolve_goals(art, args.goals_dir)

    master = json.loads((art.master_dir / "manifest.json").read_text())
    if master.get("jpeg_quality") != JPEG_QUALITY:
        raise ValueError(
            f"Crowd-Cast master must use JPEG quality {JPEG_QUALITY}, "
            f"got {master.get('jpeg_quality')!r}"
        )
    target_height = master.get("target_height")
    if not isinstance(target_height, int) or target_height <= 0:
        raise ValueError(
            f"Crowd-Cast master has invalid target_height {target_height!r}"
        )
    frames_domain = image_domain(
        media="jpeg",
        quality=JPEG_QUALITY,
        geometry="height",
        extent=target_height,
    )

    system_prompt = grammars.describe(GRAMMAR)
    prompt_digest = hashlib.sha256(system_prompt.encode()).hexdigest()
    source_rows = art.usable_rows()
    if not source_rows:
        raise ValueError(f"no usable segments in {art.dir}")
    source_segments = {str(row["segment_id"]) for row in source_rows}
    unknown_goal_segments = set(goals) - source_segments
    if unknown_goal_segments:
        raise ValueError(
            f"Crowd-Cast goals reference unknown segments: {sorted(unknown_goal_segments)}"
        )
    tasks = [
        {
            "index_row": row,
            "filter_segment": art.load_segment(str(row["segment_id"])),
            "fps": args.fps,
            "goals_by_segment": goals,
            "system_prompt": system_prompt,
        }
        for row in source_rows
    ]
    workers = min(args.num_workers, len(tasks))
    if workers == 1:
        results = map(build_segment_conversations, tasks)
    else:
        pool = mp.Pool(workers)
        results = pool.imap_unordered(build_segment_conversations, tasks, chunksize=8)

    records: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    projection_totals: Counter[str] = Counter()
    try:
        for result in results:
            projection = result.get("projection")
            if isinstance(projection, dict) and projection.get(
                "n_projected"
            ) != projection.get("n_goals"):
                raise ValueError(
                    f"Crowd-Cast goal projection failed for {result['segment_id']}: "
                    f"{projection.get('rejected')}"
                )
            statuses[result["status"]] += 1
            records.extend(result["rows"])
            for key, value in result.get("projection", {}).items():
                if isinstance(value, int):
                    projection_totals[key] += value
    finally:
        if workers != 1:
            pool.close()
            pool.join()
    if not records:
        raise ValueError("no Crowd-Cast conversations survived goal projection")
    records.sort(key=lambda record: record["conversation_id"])
    expected_goal_ids = {
        goal["goal_id"] for segment_goals in goals.values() for goal in segment_goals
    }
    produced_goal_ids = {record["goal_id"] for record in records}
    if produced_goal_ids != expected_goal_ids or len(records) != len(expected_goal_ids):
        raise ValueError("Crowd-Cast conversations do not cover every source goal")

    write_jsonl(output / "chat.jsonl", records)
    write_json_atomic(
        output / "manifest.json",
        {
            "artifact_type": "crowdcast_stage_04_conversations",
            "schema_version": 1,
            "chat": "chat.jsonl",
            "chat_sha256": file_sha256_short(output / "chat.jsonl", n=64),
            "master_store_id": art.master_store_id,
            "filter_id": art.filter_id,
            "goals_id": goals_id,
            "grammar": GRAMMAR,
            "action_format": ACTION_FORMAT,
            "system_prompt_sha256": prompt_digest,
            "image_domain": frames_domain,
            "fps": args.fps,
            "stride": stride,
            "n_conversations": len(records),
            "n_turns": sum(record["n_turns"] for record in records),
            "status_counts": dict(sorted(statuses.items())),
            "projection_counts": dict(sorted(projection_totals.items())),
        },
    )
    try:
        resolve_chat_artifact(output)
    except Exception:
        (output / "manifest.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
