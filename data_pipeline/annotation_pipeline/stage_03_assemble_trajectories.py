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


def trim_to_action_span(
    frames: list[dict[str, Any]], pre_context_frames: int
) -> tuple[list[dict[str, Any]], str | None]:
    non_noop_positions = [idx for idx, record in enumerate(frames) if record["action"] != "NO_OP"]
    if not non_noop_positions:
        return [], "all_noop"

    start = max(0, non_noop_positions[0] - pre_context_frames)
    end = non_noop_positions[-1]
    frames = frames[start : end + 1]
    if not frames:
        return [], "empty_after_trim"
    return frames, None


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
    trajectory_idx: int,
    trajectory: dict[str, Any],
    frames: list[dict[str, Any]],
    instruction: str,
    variant_idx: int = 0,
) -> dict[str, Any]:
    messages = []
    for idx, frame in enumerate(frames):
        messages.append(
            user_image(frame["image_path"], instruction if idx == 0 else None)
        )
        messages.append(assistant_text(frame["action"]))

    suffix = f"_v{variant_idx}" if variant_idx else ""
    return {
        "sample_id": f"{recording_id}_traj{trajectory_idx:04d}{suffix}",
        "recording_id": recording_id,
        "instruction": instruction,
        "variant_idx": variant_idx,
        "start_frame_idx": int(frames[0]["global_frame_idx"]),
        "end_frame_idx": int(frames[-1]["global_frame_idx"]),
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
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--pre-context-frames", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    frame_records = read_jsonl(args.frame_records)
    trajectories, annotation_source = load_trajectories(args.trajectories)
    if not frame_records:
        raise RuntimeError(f"No frame records: {args.frame_records}")

    recording_id = str(frame_records[0]["recording_id"])
    samples: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    last_emitted_end_idx = -1

    ordered = sorted(
        enumerate(trajectories),
        key=lambda kv: (
            int(kv[1].get("start_frame_idx") or 0),
            int(kv[1].get("end_frame_idx") or 0),
        ),
    )
    for idx, trajectory in ordered:
        def reject(reason: str) -> None:
            rejected.append({"trajectory_idx": idx, "reason": reason, "trajectory": trajectory})

        instruction = str(trajectory.get("instruction") or "").strip()
        if is_generic_instruction(instruction):
            reject("generic_or_empty_instruction")
            continue
        if "verified" in trajectory and not trajectory.get("verified"):
            reject("not_verified")
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
        # Enforce non-overlap with the previously emitted sample.
        n_before_clip = len(frames)
        frames = [f for f in frames if int(f["global_frame_idx"]) > last_emitted_end_idx]
        if len(frames) < args.min_frames:
            reject("overlaps_previous" if n_before_clip > len(frames) else "too_few_frames")
            continue
        frames, reject_reason = trim_to_action_span(
            frames,
            pre_context_frames=args.pre_context_frames,
        )
        if reject_reason:
            reject(reject_reason)
            continue
        if len(frames) < args.min_frames:
            reject("too_few_frames_after_trim")
            continue
        # Fan out the interval's user-prompt variants into one sample each over
        # the same frames/actions. Non-overlap is keyed on the interval (updated
        # once below), so same-span variants are not dropped as overlapping.
        emitted = 0
        for variant in instruction_variants(trajectory):
            if is_generic_instruction(variant):
                continue
            samples.append(make_sample(recording_id, idx, trajectory, frames, variant, emitted))
            emitted += 1
        if emitted == 0:
            reject("generic_or_empty_instruction")
            continue
        last_emitted_end_idx = int(frames[-1]["global_frame_idx"])

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
