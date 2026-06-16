#!/usr/bin/env python3
"""Stage 03: validate VLM intervals and assemble SFT trajectory rows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import (
    assistant_text,
    ensure_dir,
    read_jsonl,
    system_message,
    user_image,
    write_json,
    write_jsonl,
)
import json


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
    frame_records: list[dict[str, Any]], start_s: float, end_s: float
) -> list[dict[str, Any]]:
    # End-inclusive: end_time_s comes from the overlay label of the last
    # relevant frame, which an exclusive bound would drop.
    return [
        record
        for record in frame_records
        if start_s <= float(record["global_time_s"]) <= end_s + 0.25
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


def make_sample(
    recording_id: str,
    trajectory_idx: int,
    trajectory: dict[str, Any],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    instruction = str(trajectory.get("instruction") or "").strip()
    messages = [system_message()]
    for idx, frame in enumerate(frames):
        messages.append(
            user_image(frame["image_path"], instruction if idx == 0 else None)
        )
        messages.append(assistant_text(frame["action"]))

    start_s = float(frames[0]["global_time_s"])
    end_s = float(frames[-1]["global_time_s"]) + 0.5
    return {
        "sample_id": f"{recording_id}_traj{trajectory_idx:04d}",
        "recording_id": recording_id,
        "instruction": instruction,
        "start_time_s": round(start_s, 6),
        "end_time_s": round(end_s, 6),
        "duration_s": round(end_s - start_s, 6),
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
    parser.add_argument("--min-duration-s", type=float, default=8.0)
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--pre-context-frames", type=int, default=1)
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help=(
            "Assemble trajectories even if stage 02 pass C did not verify them. "
            "Plumbing/debug only - the verification gate is the quality filter."
        ),
    )
    parser.add_argument(
        "--allow-heuristic",
        action="store_true",
        help=(
            "Assemble non-VLM (heuristic/dry-run) trajectories anyway. "
            "Plumbing tests only - never for training data."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    frame_records = read_jsonl(args.frame_records)
    trajectories, annotation_source = load_trajectories(args.trajectories)
    if not frame_records:
        raise RuntimeError(f"No frame records: {args.frame_records}")
    if not annotation_source.startswith("vlm") and not args.allow_heuristic:
        raise RuntimeError(
            f"annotation_source={annotation_source!r}: refusing to assemble "
            "non-VLM instructions into SFT rows. Run stage 02 against a real "
            "VLM, or pass --allow-heuristic for plumbing tests only."
        )
    # SFT messages must reference the clean stage 01 frames (2fps, 720p),
    # never the timestamp-overlaid renders made for VLM annotation.
    first_image_path = str(frame_records[0]["image_path"])
    if "pass_a_frames" in first_image_path or "pass_b_frames" in first_image_path:
        raise RuntimeError(
            "frame records point at stage 02 annotation renders; pass the "
            "stage 01 frame_records.jsonl instead"
        )

    recording_id = str(frame_records[0]["recording_id"])
    samples: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    quality_gates = not args.allow_heuristic
    # Verification gate (stage 02 pass C): keep only trajectories a fresh
    # grounded judge accepted. Applied whenever the trajectories carry a
    # "verified" flag, unless explicitly bypassed.
    verify_gate = quality_gates and not args.allow_unverified
    last_emitted_end_s = float("-inf")

    ordered = sorted(
        enumerate(trajectories),
        key=lambda kv: (
            float(kv[1].get("start_time_s") or 0.0),
            float(kv[1].get("end_time_s") or 0.0),
        ),
    )
    for idx, trajectory in ordered:
        def reject(reason: str) -> None:
            rejected.append({"trajectory_idx": idx, "reason": reason, "trajectory": trajectory})

        instruction = str(trajectory.get("instruction") or "").strip()
        if quality_gates and is_generic_instruction(instruction):
            reject("generic_or_empty_instruction")
            continue
        if verify_gate and "verified" in trajectory and not trajectory.get("verified"):
            reject("not_verified")
            continue
        try:
            start_s = float(trajectory["start_time_s"])
            end_s = float(trajectory["end_time_s"])
        except (KeyError, TypeError, ValueError):
            reject("bad_time_bounds")
            continue
        if end_s <= start_s or end_s - start_s < args.min_duration_s:
            reject("too_short")
            continue

        frames = selected_frames(frame_records, start_s, end_s)
        # Enforce non-overlap with the previously emitted sample.
        n_before_clip = len(frames)
        frames = [f for f in frames if float(f["global_time_s"]) > last_emitted_end_s]
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
        samples.append(make_sample(recording_id, idx, trajectory, frames))
        last_emitted_end_s = float(frames[-1]["global_time_s"])

    write_jsonl(output_dir / "trajectories.jsonl", samples)
    write_jsonl(output_dir / "rejected_trajectories.jsonl", rejected)
    write_json(
        output_dir / "assemble_summary.json",
        {
            "n_input_trajectories": len(trajectories),
            "n_samples": len(samples),
            "n_rejected": len(rejected),
            "annotation_source": annotation_source,
            "verify_gated": verify_gate,
            "quality_gates": quality_gates,
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
