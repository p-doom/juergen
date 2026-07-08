#!/usr/bin/env python3
"""Stage 03: validate VLM intervals and assemble SFT trajectory rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from annotation_pipeline.common import (
    assistant_text,
    ensure_dir,
    read_jsonl,
    user_image,
    write_json,
    write_jsonl,
)


def load_trajectories(path: Path) -> tuple[list[dict[str, Any]], str]:
    data = json.loads(path.read_text())
    trajectories = data.get("trajectories", [])
    if not isinstance(trajectories, list):
        raise ValueError("trajectories_raw.json must contain a trajectories list")
    annotation_source = str(data.get("annotation_source", "unknown"))
    return trajectories, annotation_source


def is_generic_instruction(instruction: str) -> bool:
    """Reject placeholder or contentless instructions that carry no goal."""
    text = instruction.strip().lower().rstrip(".")
    if not text or len(text.split()) < 3:
        return True
    generic = (
        "complete the visible desktop task in this interval",
        "complete the task",
        "perform the visible task",
        "continue the current task",
        "do the task on screen",
    )
    return text in generic


def selected_frames(
    frame_records: list[dict[str, Any]], spans: list[list[int]]
) -> list[dict[str, Any]]:
    # Union of inclusive frame-index spans. Stage 02 emits the trajectory's
    # actual span-union (`frame_spans`), so frames belonging to other activities
    # in the gaps between an interleaved goal's spans are excluded.
    return [
        record
        for record in frame_records
        if any(s <= int(record["global_frame_idx"]) <= e for s, e in spans)
    ]


def instruction_variants(trajectory: dict[str, Any]) -> list[str]:
    """Primary instruction plus any register paraphrases, de-duplicated.

    Stage 02 (hindsight) emits ``instruction_variants`` so one interval yields
    several user-prompt phrasings at different abstraction levels; each becomes
    its own SFT sample over the same frames/actions. Legacy trajectories with no
    variants list still yield exactly one sample.
    """
    out: list[str] = []
    for text in [trajectory.get("instruction"), *trajectory.get("instruction_variants", [])]:
        text = str(text or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def make_sample(
    recording_id: str,
    clip_id: str,
    trajectory_idx: int,
    trajectory: dict[str, Any],
    frames: list[dict[str, Any]],
    instruction: str,
    variant_idx: int = 0,
    plan: str = "",
) -> dict[str, Any]:
    # Reason-before-action: the plan prose (stage 02b) prefixes the FIRST
    # assistant turn only — `<plan>\n<first action>` — matching the fold
    # pipeline's assemble_sft byte format. Later turns stay pure actions.
    messages = []
    for idx, frame in enumerate(frames):
        messages.append(
            user_image(frame["image_path"], instruction if idx == 0 else None)
        )
        first_turn_text = f"{plan}\n{frame['action']}" if (idx == 0 and plan) else frame["action"]
        messages.append(assistant_text(first_turn_text))

    suffix = f"_v{variant_idx}" if variant_idx else ""
    first, last = frames[0], frames[-1]
    return {
        "plan": plan,
        # clip_id is a separate field and stage 04 prefixes it onto the final
        # sample_id, so keep this one clip-relative to avoid doubling it.
        "sample_id": f"traj{trajectory_idx:04d}{suffix}",
        "recording_id": recording_id,
        "clip_id": clip_id,
        "instruction": instruction,
        "variant_idx": variant_idx,
        # WHERE-IN-CLIP: frame_idx is the sampled-stream index; the time/source
        # spans locate the goal in the actual recording (seconds into the segment
        # and original video frame numbers — seekable with video_path + fps).
        "start_frame_idx": int(first["global_frame_idx"]),
        "end_frame_idx": int(last["global_frame_idx"]),
        "start_time_s": first.get("global_time_s"),
        "end_time_s": last.get("global_time_s"),
        "source_frame_start": first.get("source_frame_idx"),
        "source_frame_end": last.get("source_frame_idx"),
        "n_frames": len(frames),
        "n_non_noop": sum(1 for frame in frames if frame["action"] != "NO_OP"),
        "source_trajectory": trajectory,
        "messages": messages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-records", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=1,
                        help="Drop goals shorter than this many frames. Default 1 "
                             "keeps tight short goals; stage 02 already bounds them.")
    parser.add_argument("--include-variants", action="store_true",
                        help="Emit one sample per instruction paraphrase (3x). Default "
                             "emits one sample from the main instruction only.")
    parser.add_argument("--require-plan", action="store_true",
                        help="Reject goals without a usable stage-02b plan instead of "
                             "emitting a plan-less first turn.")
    return parser.parse_args()


# Plans carrying these stage-02b quality flags are unusable as training prose;
# the sample falls back to a plan-less first turn (or is rejected under
# require_plan). "too_long"/"not_first_person" flags are kept — still valid.
DROP_PLAN_FLAGS = frozenset({"empty", "restates_instruction"})


def usable_plan(trajectory: dict[str, Any]) -> str:
    plan = str(trajectory.get("plan") or "").strip()
    if set(trajectory.get("plan_flags") or []) & DROP_PLAN_FLAGS:
        return ""
    return plan


def assemble_samples(
    frame_records: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    *,
    min_frames: int = 1,
    include_variants: bool = False,
    require_plan: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Turn one segment's frame_records + goal trajectories into SFT samples.

    Stage 02 already emits tight, non-overlapping, frame-accurate bounds (start =
    first keystroke, end = true end-state) and stage 01 already thinned NO_OPs, so
    this just slices [start_frame_idx, end_frame_idx] and assembles image->action
    turns — no NO_OP re-trim, no overlap clipping, no large min-frames floor.
    Returns (samples, rejected). Reused by both the CLI and the dataset driver."""
    recording_id = str(frame_records[0]["recording_id"])
    # clip_id groups all samples of one segment (stage 04 uses it for sample ids
    # and clip-level splits); windows of a split segment share the parent's id
    # because the driver feeds the parent's full frame_records here.
    clip_id = str(frame_records[0].get("segment_id") or recording_id)
    samples: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    ordered = sorted(
        enumerate(trajectories),
        key=lambda kv: (int(kv[1].get("start_frame_idx") or 0), int(kv[1].get("end_frame_idx") or 0)),
    )
    for idx, trajectory in ordered:
        def reject(reason: str) -> None:
            rejected.append({"trajectory_idx": idx, "reason": reason, "trajectory": trajectory})

        instruction = str(trajectory.get("instruction") or "").strip()
        if is_generic_instruction(instruction):
            reject("generic_or_empty_instruction")
            continue
        try:
            start_idx = int(trajectory["start_frame_idx"])
            end_idx = int(trajectory["end_frame_idx"])
        except (KeyError, TypeError, ValueError):
            reject("bad_frame_bounds")
            continue
        if end_idx < start_idx:
            reject("too_short")
            continue

        spans = trajectory.get("frame_spans") or [[start_idx, end_idx]]
        frames = selected_frames(frame_records, spans)
        if len(frames) < min_frames:
            reject("too_few_frames")
            continue
        plan = usable_plan(trajectory)
        if require_plan and not plan:
            reject("missing_plan")
            continue
        # Emit the main instruction (default) or fan out every paraphrase into its
        # own sample over the same frames/actions when include_variants is set.
        # Variants share the goal's one plan (it adds situation/method, not
        # phrasing, so it composes with any paraphrase).
        phrasings = instruction_variants(trajectory) if include_variants else [instruction]
        emitted = 0
        for variant in phrasings:
            if is_generic_instruction(variant):
                continue
            samples.append(make_sample(recording_id, clip_id, idx, trajectory, frames, variant,
                                       emitted, plan=plan))
            emitted += 1
        if emitted == 0:
            reject("generic_or_empty_instruction")
    return samples, rejected


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    frame_records = read_jsonl(args.frame_records)
    trajectories, annotation_source = load_trajectories(args.trajectories)
    if not frame_records:
        raise RuntimeError(f"No frame records: {args.frame_records}")

    recording_id = str(frame_records[0]["recording_id"])
    samples, rejected = assemble_samples(
        frame_records, trajectories,
        min_frames=args.min_frames, include_variants=args.include_variants,
        require_plan=args.require_plan,
    )

    write_jsonl(output_dir / "trajectories.jsonl", samples)
    write_jsonl(output_dir / "rejected_trajectories.jsonl", rejected)
    write_json(
        output_dir / "assemble_summary.json",
        {
            "n_input_trajectories": len(trajectories),
            "n_samples": len(samples),
            "n_rejected": len(rejected),
            "annotation_source": annotation_source,
            "recording_id": recording_id,
            "reject_reasons": {
                reason: sum(1 for row in rejected if row["reason"] == reason)
                for reason in sorted({row["reason"] for row in rejected})
            },
        },
    )
    print(f"Wrote {len(samples)} assembled trajectories to {output_dir}")


if __name__ == "__main__":
    main()
