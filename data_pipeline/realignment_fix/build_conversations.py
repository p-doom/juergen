#!/usr/bin/env python3
"""Stage 04 (conversations): turn a stage-03 frame-sampling dataset into a
training ``conversations.jsonl`` -- NO VLM annotation, NO re-decoding.

Stage 03 (``sample_frames_actions``) emitted, per segment, a
``frame_records.jsonl`` of ordered ``(image_path, action)`` rows (frames already
fps-sampled, NO_OP-thinned, and black-filtered). This stage assembles each
segment into ONE interleaved screenshot->action conversation:

    [ system,
      user(screenshot_0 [+ instruction]), assistant(action_0),
      user(screenshot_1),                 assistant(action_1),
      ... ,
      user(screenshot_N),                 assistant(action_N) ]

i.e. the user shows a screenshot, the model replies with the action taken from
it, and that action is what produced the NEXT screenshot -- exactly the shape the
eval-side OSWorld runtime prompts with, but materialized for SFT. ``action_i`` is
the raw recorded action string from the frame record (``"<dx> <dy> <scroll>"``
optionally ``" ; +KEY -KEY"``, or ``"NO_OP"``); the image is the ``ar://`` grain
ref, passed through verbatim (already portable/absolute).

The message/content schema matches the annotation pipeline's canonical
``chat.jsonl`` (``stage_04_build_canonical_sft``): content is a list of
``{"type":"image","image":...}`` / ``{"type":"text","text":...}`` blocks, and on
the first user turn the instruction TEXT precedes the image. Unlike that builder,
this one does NOT require an instruction -- goal-free (system-prompt-only) is the
default -- so it runs straight off the sampled dataset with no labeling step.

Instruction (first user turn) is configurable:
  * default: goal-free (image only).
  * ``--instruction TEXT``: a fixed instruction on every segment's first turn.
  * ``--instruction-field KEY``: a PER-SEGMENT instruction read from KEY on the
    sample_index row (falling back to the first frame record), for when goals are
    joined in upstream (e.g. OSWorld task text). Falls back to --instruction, then
    goal-free, when the field is absent/empty.

One conversation per segment (no windowing): a long, high-fps segment becomes a
long conversation -- watch the trainee's context window at high --target-fps.

Input  (--sample-dir): a stage-03 output (``sample_index.jsonl`` +
        ``clips/<seg>/stage_01/frame_records.jsonl``).
Output (--output-dir):
  conversations.jsonl          one row per segment: {messages, + provenance}.
  conversations.train/val.jsonl split partitions (only when --val-fraction > 0).
  <split>/chat.jsonl           per-split canonical layout (train/, val/, ...) so
                               this dir is a drop-in source_path for the
                               grain_payload stage (omegalax stage_c reads
                               <source>/<split>/chat.jsonl). Same rows, same
                               schema as conversations.jsonl -- just partitioned.
  split_manifest.jsonl         segment_id -> split (only when --val-fraction > 0).
  conversations_summary.json   aggregate stats.
  manifest.json                artifact marker.

Run::

    cd data_pipeline
    uv run python realignment_fix/build_conversations.py \
        --sample-dir  <stage-03 --output-dir> \
        --output-dir  <dest> \
        [--instruction "..."] [--val-fraction 0.05]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Make the sibling ``annotation_pipeline`` package importable when run directly
# (mirrors build_frames_master.py / sample_frames_actions.py).
DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from annotation_pipeline import config  # noqa: E402
from annotation_pipeline.common import ensure_dir, read_jsonl, write_json, write_jsonl  # noqa: E402

# Stage-03 statuses that carry a usable frame_records.jsonl.
USABLE_STATUSES = {"ok", "cached"}

# Default system prompts. Goal-conditioned reuses the canonical one (it names a
# goal); goal-free drops the goal but keeps the action-format contract.
GOAL_FREE_SYSTEM_PROMPT = (
    "You operate a desktop computer. Each user turn shows the current screen. "
    "Reply with the next action as `<dx> <dy> <scroll>` optionally followed by "
    "` ; +KEY -KEY` events, or `NO_OP` if no action."
)
GOAL_SYSTEM_PROMPT = config.SYSTEM_PROMPT


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _image_block(image: str) -> dict[str, Any]:
    return {"type": "image", "image": image}


def build_messages(
    frames: list[dict[str, Any]],
    *,
    instruction: str | None,
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    """Assemble the interleaved conversation for one segment. Matches the canonical
    chat.jsonl schema: instruction TEXT before the image on the first user turn,
    image-only on later turns, one assistant turn per frame carrying its action."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": [_text_block(system_prompt)]})
    for idx, frame in enumerate(frames):
        content: list[dict[str, Any]] = []
        if idx == 0 and instruction:
            content.append(_text_block(instruction))
        content.append(_image_block(str(frame["image_path"])))
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [_text_block(str(frame["action"]))]})
    return messages


def _resolve_instruction(
    index_row: dict[str, Any],
    frames: list[dict[str, Any]],
    *,
    instruction: str | None,
    instruction_field: str | None,
) -> str | None:
    """Per-segment instruction from --instruction-field (sample_index row, then
    first frame record), else the fixed --instruction, else None (goal-free)."""
    if instruction_field:
        for src in (index_row, frames[0] if frames else {}):
            val = src.get(instruction_field)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return instruction


def build_conversation(
    index_row: dict[str, Any],
    *,
    instruction: str | None,
    instruction_field: str | None,
    system_prompt: str | None,
    min_frames: int,
) -> dict[str, Any] | None:
    """One segment -> one conversation record, or None if it has no usable frames."""
    fr_path = index_row.get("frame_records")
    if not fr_path or not Path(fr_path).exists():
        return None
    frames = read_jsonl(Path(fr_path))
    frames.sort(key=lambda r: int(r.get("global_frame_idx") or 0))
    if len(frames) < min_frames:
        return None

    seg_instruction = _resolve_instruction(
        index_row, frames, instruction=instruction, instruction_field=instruction_field
    )
    messages = build_messages(frames, instruction=seg_instruction, system_prompt=system_prompt)
    return {
        "conversation_id": str(index_row.get("segment_id")),
        "recording_id": index_row.get("recording_id"),
        "segment_id": index_row.get("segment_id"),
        "segment_idx": index_row.get("segment_idx"),
        "instruction": seg_instruction,
        "goal_conditioned": seg_instruction is not None,
        "n_frames": len(frames),
        "n_turns": len(frames),  # one user+assistant pair per frame
        "n_non_noop": sum(1 for f in frames if str(f.get("action")) != "NO_OP"),
        "target_fps": index_row.get("target_fps"),
        "alignment_status": index_row.get("alignment_status"),
        "messages": messages,
    }


def _split_of(recording_id: str | None, val_fraction: float) -> str:
    """Deterministic recording-level train/val split (whole recording -> one side,
    so frames from a recording never leak across the split). No RNG/seed needed."""
    if val_fraction <= 0.0 or not recording_id:
        return "train"
    bucket = int(hashlib.sha1(str(recording_id).encode()).hexdigest(), 16) % 1000
    return "val" if bucket < round(val_fraction * 1000) else "train"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample-dir", type=Path, required=True,
                   help="A stage-03 (sample_frames_actions) --output-dir: must contain "
                        "sample_index.jsonl and clips/<seg>/stage_01/frame_records.jsonl.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--instruction", type=str, default=None,
                   help="Fixed instruction placed on each segment's first user turn "
                        "(goal-conditioned). Omit for goal-free (system-prompt only).")
    p.add_argument("--instruction-field", type=str, default=None,
                   help="Per-segment instruction: read this key from the sample_index row "
                        "(then the first frame record). Falls back to --instruction, then goal-free.")
    p.add_argument("--system-prompt", type=str, default=None,
                   help="System message text. Default: a goal-free prompt, or the canonical "
                        "goal-conditioned prompt when an instruction is set.")
    p.add_argument("--no-system-prompt", action="store_true", help="Emit no system message.")
    p.add_argument("--min-frames", type=int, default=1,
                   help="Skip segments with fewer than this many frames.")
    p.add_argument("--val-fraction", type=float, default=0.0,
                   help="If > 0, also write train/val partitions split by recording_id.")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N segments (debug).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.val_fraction < 1.0):
        raise SystemExit("--val-fraction must be in [0, 1)")

    index_path = args.sample_dir / "sample_index.jsonl"
    if not index_path.is_file():
        raise SystemExit(f"no sample_index.jsonl under {args.sample_dir} (is it a stage-03 sample artifact?)")
    index_rows = read_jsonl(index_path)
    usable = [r for r in index_rows if r.get("status") in USABLE_STATUSES]
    if args.limit is not None:
        usable = usable[: args.limit]
    if not usable:
        raise SystemExit(f"no usable segments (status in {sorted(USABLE_STATUSES)}) in {index_path}")

    # System prompt: explicit override wins; else default by goal-conditioning. A
    # segment is goal-conditioned iff it resolves an instruction, but the system
    # prompt is chosen once for the run from whether ANY instruction source is set.
    goal_conditioned = bool(args.instruction or args.instruction_field)
    if args.no_system_prompt:
        system_prompt = None
    elif args.system_prompt is not None:
        system_prompt = args.system_prompt
    else:
        system_prompt = GOAL_SYSTEM_PROMPT if goal_conditioned else GOAL_FREE_SYSTEM_PROMPT

    out_dir = ensure_dir(args.output_dir)
    records: list[dict[str, Any]] = []
    split_manifest: list[dict[str, Any]] = []
    n_skipped = 0
    n_frames_total = 0
    n_turns_total = 0
    for i, row in enumerate(usable, 1):
        conv = build_conversation(
            row,
            instruction=args.instruction,
            instruction_field=args.instruction_field,
            system_prompt=system_prompt,
            min_frames=args.min_frames,
        )
        if conv is None:
            n_skipped += 1
            continue
        conv["split"] = _split_of(conv.get("recording_id"), args.val_fraction)
        records.append(conv)
        split_manifest.append({"segment_id": conv["segment_id"],
                               "recording_id": conv["recording_id"], "split": conv["split"]})
        n_frames_total += conv["n_frames"]
        n_turns_total += conv["n_turns"]
        if i % 1000 == 0:
            print(f"  {i}/{len(usable)} segments | {len(records)} conversations", flush=True)

    if not records:
        raise SystemExit("no conversations built (all segments empty or below --min-frames)")

    write_jsonl(out_dir / "conversations.jsonl", records)
    if args.val_fraction > 0.0:
        write_jsonl(out_dir / "conversations.train.jsonl", [r for r in records if r["split"] == "train"])
        write_jsonl(out_dir / "conversations.val.jsonl", [r for r in records if r["split"] == "val"])
        write_jsonl(out_dir / "split_manifest.jsonl", split_manifest)

    # Per-split chat.jsonl under <out>/<split>/ so this dataset is a drop-in
    # source_path for the grain_payload stage: omegalax's stage_c reads
    # <source>/<split>/chat.jsonl per split. The record schema already matches the
    # canonical chat.jsonl (a "messages" list + provenance; omegalax keeps every
    # other key as session metadata), so these are the same rows, just partitioned
    # into the canonical layout. Mirrors annotation_pipeline/build_sft.py. Always
    # written -- with --val-fraction 0 that is just <out>/train/chat.jsonl.
    by_split: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_split.setdefault(r["split"], []).append(r)
    for split, srows in sorted(by_split.items()):
        write_jsonl(ensure_dir(out_dir / split) / "chat.jsonl", srows)

    n_val = sum(1 for r in records if r["split"] == "val")
    summary = {
        "n_conversations": len(records),
        "n_train": len(records) - n_val,
        "n_val": n_val,
        "n_segments_skipped": n_skipped,
        "n_frames_total": n_frames_total,
        "n_turns_total": n_turns_total,
        "goal_conditioned": goal_conditioned,
        "instruction": args.instruction,
        "instruction_field": args.instruction_field,
        "has_system_prompt": system_prompt is not None,
        "val_fraction": args.val_fraction,
        "sample_dir": str(args.sample_dir),
    }
    write_json(out_dir / "conversations_summary.json", summary)
    write_json(out_dir / "manifest.json", {
        "artifact_type": "juergen_annotation_conversations",
        "schema_version": 1,
        "conversations": "conversations.jsonl",
        "chat": "<split>/chat.jsonl",  # drop-in source_path for the grain_payload stage
        "splits": sorted(by_split),
        **summary,
    })
    print(
        f"[conversations] {len(records)} conversations "
        f"({summary['n_train']} train / {n_val} val), {n_turns_total} turns, "
        f"{n_frames_total} frames, {n_skipped} skipped -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
