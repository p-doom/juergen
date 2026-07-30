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

GOAL-CONDITIONED mode (``--goal-index``): instead of one conversation per segment,
build one conversation PER GOAL from a stage-03b ``goal_frame_index.jsonl``. Each
goal's OUR frames are its ``--sample-dir`` frames windowed to the goal's
source-frame span ``[coll_source_frame_idx_lo, coll_source_frame_idx_hi]``, with
the goal ``instruction`` on the first user turn. When the goal index carries a
``context`` (a self-compaction ``[CONTEXT]`` rolling summary, present on non-first
chunks of a split goal), it is fused after the instruction on that first turn -- so a
resumed chunk sees its progress summary, the format the selfcompact set was built
with. The span is colleague-derived and
fps-independent, so ONE goal index goal-conditions ANY ``--sample-dir`` fps. Goals
whose window is all-idle (dropped by ``noop_mode=none``) fall below ``--min-frames``
and are skipped. Without ``--goal-index`` the per-segment behavior below is unchanged.
Optionally (``--terminate-token TERMINATE``) the final assistant turn's action is
overwritten with a terminate token, marking goal completion at the window's end (the
token the eval side's ``freeroll._is_terminate`` recognizes). This applies in
goal-conditioned mode ONLY -- a per-segment end is not a task completion -- and pairs
with a system prompt that describes the contract (e.g. ``--system-prompt-id yll_v1``).

One conversation per segment (no windowing): a long, high-fps segment becomes a
long conversation -- watch the trainee's context window at high --target-fps.

The train/val split is NOT applied here: this stage emits a single
split-agnostic ``chat.jsonl`` and the recording-level split is deferred to the
records stage (stage 06, via ``--val_fraction``). That keeps this stage -- and
the measure cache (stage 05) -- independent of the split, so changing the val
fraction re-runs only stage 06 and never re-tokenizes.

Input  (--sample-dir): a stage-03 output (``sample_index.jsonl`` +
        ``clips/<seg>/stage_01/frame_records.jsonl``).
Output (--output-dir):
  conversations.jsonl          one row per segment: {messages, + provenance}.
  chat.jsonl                   the canonical layout (same rows, same schema as
                               conversations.jsonl) -- a single split-agnostic
                               drop-in source_path for the measure/records stages
                               (stage 05 reads <source>/chat.jsonl, stage 06 reads
                               it and applies the split). Carries recording_id per
                               row so the downstream split can group by recording.
  conversations_summary.json   aggregate stats.
  manifest.json                artifact marker.

Run::

    cd data_pipeline
    uv run python realigned_pipeline/stage_04_build_conversations.py \
        --sample-dir  <stage-03 --output-dir> \
        --output-dir  <dest> \
        [--instruction "..."]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Make the ``realigned_pipeline`` package importable when run directly
# (mirrors build_frames_master.py / sample_frames_actions.py).
DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.lib import config  # noqa: E402
from realigned_pipeline.lib.common import ensure_dir, read_jsonl, write_json, write_jsonl  # noqa: E402

# Named system prompts are shared with the eval side (single source of truth):
# the OSWorld runners select from this same SYSTEM_PROMPTS dict by id, so a model
# can be trained and evaluated under an identical system message. ``eval/`` is a
# sibling of ``data_pipeline/`` (repo root) with no package init, so add it to
# sys.path and import the module directly -- exactly as the eval runners do. The
# module is dependency-free (just the dict), so this is cheap and safe.
EVAL_DIR = DATA_PIPELINE_DIR.parent / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.append(str(EVAL_DIR))

from osworld_system_prompts import SYSTEM_PROMPTS  # noqa: E402

# Stage-03 statuses that carry a usable frame_records.jsonl.
USABLE_STATUSES = {"ok", "cached"}

# Default system prompts. Goal-free reuses the verbatim training-time prompt
# ("training_v1") from the shared eval dict; goal-conditioned reuses the
# canonical one (it names a goal) but keeps the action-format contract.
GOAL_FREE_SYSTEM_PROMPT = SYSTEM_PROMPTS["yll_v1"]
GOAL_SYSTEM_PROMPT = config.SYSTEM_PROMPT


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _image_block(image: str) -> dict[str, Any]:
    return {"type": "image", "image": image}


def _join_instruction_context(instruction: str | None, context: str | None) -> str | None:
    """Fuse the goal instruction with its self-compaction ``[CONTEXT]`` block into a
    single first-turn text, reproducing the external (selfcompact) builder's layout
    ``"<instruction>\n\n[CONTEXT]…[/CONTEXT]"``. ``context`` is present only on
    non-first chunks of a split goal (carried through stage-03b's goal index); either
    argument may be absent (goal-free -> None; first/single chunk -> instruction only)."""
    parts = [p.strip() for p in (instruction, context) if p and str(p).strip()]
    return "\n\n".join(parts) if parts else None


def build_messages(
    frames: list[dict[str, Any]],
    *,
    instruction: str | None,
    system_prompt: str | None,
    context: str | None = None,
    terminate_token: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble the interleaved conversation for one segment. Matches the canonical
    chat.jsonl schema: instruction TEXT before the image on the first user turn,
    image-only on later turns, one assistant turn per frame carrying its action.

    ``context`` (goal-conditioned self-compaction only): the goal's ``[CONTEXT]`` block,
    fused after the instruction on the first user turn (see ``_join_instruction_context``)
    so a non-first chunk resumes from its rolling progress summary -- the format the
    selfcompact set was built with. None for goal-free / first-or-single chunks.

    ``terminate_token`` (goal-conditioned mode only): OVERWRITE the final assistant
    turn's action with this token (e.g. ``"TERMINATE"``), marking goal completion at
    the window's end -- the eval side (``freeroll._is_terminate``) treats the first
    stripped line ``== "TERMINATE"`` as end-of-episode. The last frame's real action
    label is dropped in exchange (the yll pilot's convention)."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": [_text_block(system_prompt)]})
    first_text = _join_instruction_context(instruction, context)
    for idx, frame in enumerate(frames):
        content: list[dict[str, Any]] = []
        if idx == 0 and first_text:
            content.append(_text_block(first_text))
        content.append(_image_block(str(frame["image_path"])))
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [_text_block(str(frame["action"]))]})
    if terminate_token and messages and messages[-1]["role"] == "assistant":
        messages[-1]["content"] = [_text_block(terminate_token)]
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


def build_goal_conversation(
    goal: dict[str, Any],
    seg_frames: list[dict[str, Any]],
    index_row: dict[str, Any],
    *,
    system_prompt: str | None,
    min_frames: int,
    terminate_token: str | None = None,
) -> dict[str, Any] | None:
    """One goal-conditioned conversation: OUR ``--sample-dir`` frames for the goal's
    segment, windowed to its source-frame span ``[lo, hi]`` (from stage-03b), with the
    goal text -- plus its self-compaction ``[CONTEXT]`` summary when the goal index
    carries one (non-first chunks of a split goal) -- as the first user turn. Returns
    None if the window holds fewer than ``min_frames`` frames (e.g. an all-idle goal
    ``noop_mode=none`` dropped).

    ``terminate_token`` marks the window's end as goal-complete: the final assistant
    turn's action is overwritten with the token (see ``build_messages``)."""
    lo, hi = goal.get("coll_source_frame_idx_lo"), goal.get("coll_source_frame_idx_hi")
    if lo is None or hi is None:
        return None
    lo, hi = int(lo), int(hi)
    frames = [
        f for f in seg_frames
        if f.get("source_frame_idx") is not None and lo <= int(f["source_frame_idx"]) <= hi
    ]
    if len(frames) < min_frames:
        return None
    frames.sort(key=lambda r: int(r["source_frame_idx"]))
    instruction = goal.get("instruction")
    context = goal.get("context")  # self-compaction rolling summary (non-first chunks)
    messages = build_messages(
        frames, instruction=instruction, system_prompt=system_prompt,
        context=context, terminate_token=terminate_token,
    )
    return {
        "conversation_id": goal.get("sample_id") or f"{goal.get('segment_id')}_g{goal.get('goal_id')}",
        "recording_id": goal.get("recording_id") or index_row.get("recording_id"),
        "segment_id": goal.get("segment_id"),
        "segment_idx": index_row.get("segment_idx"),
        "goal_id": goal.get("goal_id"),
        "sample_id": goal.get("sample_id"),
        "instruction": instruction,
        # self-compaction provenance (goal-conditioned selfcompact set); context is
        # also fused into messages[0], these mirror it as clean metadata.
        "context": context,
        "chunk_idx": goal.get("chunk_idx"),
        "n_chunks": goal.get("n_chunks"),
        "parent_sample_id": goal.get("parent_sample_id"),
        "context_tokens": goal.get("context_tokens"),
        "goal_conditioned": True,
        "long_ref": goal.get("long_ref"),
        "long_text": goal.get("long_text"),
        "kind": goal.get("kind"),
        "status": goal.get("status"),
        "split": goal.get("split"),
        "n_frames": len(frames),
        "n_turns": len(frames),  # one user+assistant pair per frame
        "n_non_noop": sum(1 for f in frames if str(f.get("action")) != "NO_OP"),
        "target_fps": index_row.get("target_fps"),
        "alignment_status": index_row.get("alignment_status"),
        "messages": messages,
    }


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
    sp = p.add_mutually_exclusive_group()
    sp.add_argument("--system-prompt", type=str, default=None,
                    help="Raw system message text. Default: a goal-free prompt, or the canonical "
                         "goal-conditioned prompt when an instruction is set.")
    sp.add_argument("--system-prompt-id", type=str, default=None,
                    help="Select a named system prompt from eval/osworld_system_prompts.py "
                         "(shared with the OSWorld eval runners). One of: "
                         f"{', '.join(SYSTEM_PROMPTS)}.")
    sp.add_argument("--no-system-prompt", action="store_true", help="Emit no system message.")
    p.add_argument("--goal-index", type=Path, default=None,
                   help="A stage-03b goal_frame_index.jsonl. Switches to GOAL-CONDITIONED "
                        "mode: one conversation per goal, --sample-dir frames windowed to "
                        "the goal's source-frame span, goal text as the first-turn "
                        "instruction. Ignores --instruction/--instruction-field.")
    p.add_argument("--terminate-token", type=str, default="TERMINATE",
                   help="GOAL-CONDITIONED (--goal-index) ONLY: overwrite the final assistant "
                        "turn's action with this token (canonical: \"TERMINATE\") to mark goal "
                        "completion at the window's end. Off by default; ignored (with a warning) "
                        "in per-segment mode, where a segment end is not a task completion. Pair "
                        "with a system prompt that describes the contract, e.g. "
                        "--system-prompt-id yll_v1.")
    p.add_argument("--min-frames", type=int, default=1,
                   help="Skip segments (or goals, in --goal-index mode) with fewer than "
                        "this many frames.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N segments (or goals, in --goal-index mode).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

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
    # prompt is chosen once for the run from whether ANY instruction source is set
    # (a --goal-index run is always goal-conditioned).
    goal_conditioned = bool(args.goal_index or args.instruction or args.instruction_field)
    system_prompt_id = None
    if args.no_system_prompt:
        system_prompt = None
    elif args.system_prompt is not None:
        system_prompt = args.system_prompt
    elif args.system_prompt_id is not None:
        if args.system_prompt_id not in SYSTEM_PROMPTS:
            raise SystemExit(
                f"unknown --system-prompt-id {args.system_prompt_id!r}; "
                f"available: {', '.join(SYSTEM_PROMPTS)}"
            )
        system_prompt_id = args.system_prompt_id
        system_prompt = SYSTEM_PROMPTS[system_prompt_id]
    else:
        system_prompt_id = "training_v1" if not goal_conditioned else None
        system_prompt = GOAL_SYSTEM_PROMPT if goal_conditioned else GOAL_FREE_SYSTEM_PROMPT

    # TERMINATE is goal-conditioned-only (a segment end is not a task completion).
    terminate_token = args.terminate_token
    if terminate_token and not args.goal_index:
        print("[conversations] WARNING: --terminate-token is ignored without --goal-index "
              "(per-segment ends are not goal completions).", flush=True)
        terminate_token = None

    out_dir = ensure_dir(args.output_dir)
    records: list[dict[str, Any]] = []
    n_skipped = 0
    n_frames_total = 0
    n_turns_total = 0

    if args.goal_index:
        # GOAL-CONDITIONED: one conversation per goal, OUR sampled frames windowed to
        # the goal's source-frame span. The span is colleague-derived (fps-independent),
        # so this goal index goal-conditions whatever fps --sample-dir holds. Group goals
        # by segment so each segment's frame_records is read once.
        index_by_seg = {str(r["segment_id"]): r for r in index_rows}
        goals = [g for g in read_jsonl(args.goal_index)
                 if g.get("match_status") == "ok" and g.get("segment_id")]
        if args.limit is not None:
            goals = goals[: args.limit]
        goals_by_seg: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for g in goals:
            goals_by_seg[str(g["segment_id"])].append(g)
        print(f"[conversations] goal-conditioned: {len(goals)} goals across "
              f"{len(goals_by_seg)} segments", flush=True)
        for j, (segment_id, seg_goals) in enumerate(goals_by_seg.items(), 1):
            row = index_by_seg.get(segment_id)
            if row is None or row.get("status") not in USABLE_STATUSES:
                n_skipped += len(seg_goals)  # segment not in --sample-dir (or unusable)
                continue
            fr_path = row.get("frame_records")
            seg_frames = read_jsonl(Path(fr_path)) if fr_path and Path(fr_path).exists() else []
            for g in seg_goals:
                conv = build_goal_conversation(
                    g, seg_frames, row, system_prompt=system_prompt,
                    min_frames=args.min_frames, terminate_token=terminate_token,
                )
                if conv is None:
                    n_skipped += 1
                    continue
                records.append(conv)
                n_frames_total += conv["n_frames"]
                n_turns_total += conv["n_turns"]
            if j % 500 == 0:
                print(f"  {j}/{len(goals_by_seg)} segments | {len(records)} goal conversations", flush=True)
    else:
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
            records.append(conv)
            n_frames_total += conv["n_frames"]
            n_turns_total += conv["n_turns"]
            if i % 1000 == 0:
                print(f"  {i}/{len(usable)} segments | {len(records)} conversations", flush=True)

    if not records:
        raise SystemExit("no conversations built (all segments empty or below --min-frames)")

    write_jsonl(out_dir / "conversations.jsonl", records)
    write_jsonl(out_dir / "chat.jsonl", records)

    summary = {
        "n_conversations": len(records),
        "n_skipped": n_skipped,  # goals (goal-index mode) or segments below --min-frames
        "n_frames_total": n_frames_total,
        "n_turns_total": n_turns_total,
        "mode": "goal" if args.goal_index else "segment",
        "goal_conditioned": goal_conditioned,
        "goal_index": str(args.goal_index) if args.goal_index else None,
        "terminate_token": terminate_token,
        "instruction": args.instruction,
        "instruction_field": args.instruction_field,
        "has_system_prompt": system_prompt is not None,
        "system_prompt_id": system_prompt_id,
        "sample_dir": str(args.sample_dir),
    }
    write_json(out_dir / "conversations_summary.json", summary)
    write_json(out_dir / "manifest.json", {
        "artifact_type": "juergen_annotation_conversations",
        "schema_version": 1,
        "conversations": "conversations.jsonl",
        "chat": "chat.jsonl",  # split-agnostic drop-in source_path for stages 05/06
        **summary,
    })
    print(
        f"[conversations] {len(records)} conversations, {n_turns_total} turns, "
        f"{n_frames_total} frames, {n_skipped} skipped -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
