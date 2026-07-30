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

ACTION FORMAT (``--action-format``): by default (``sampled``) the assistant turns
carry the stage-03 frame records' ``action`` strings verbatim (the canonical
``"<dx> <dy> <scroll> [; +KEY -KEY]"`` aggregate). Any registered formatter name
(``lib/action_format.FORMATTERS``: ``canonical``, ``ordered_events_v2``,
``ordered_events_v3``, ``computer_use_rel_v1``) instead RE-DERIVES every label
from the segment's REALIGNED keylog at build time, so the action format is a
stage-04 ablation flag, not a pipeline rerun. The re-derivation reconstructs the
lib/events view from the sample artifact: the kept frames' ``master_record_index``
ticks become label windows ``[tick_i, tick_{i+1})`` (the last runs to the end of
master coverage), and dead zones are the pre-first-frame span plus every
(near-)black master tick (from the master store's ``frame_manifest.jsonl``, using
the thresholds the sample was built with -- only when it was built with
``drop_black_frames``). Labels then follow the shared dead-zone policy
(``lib/events.apply_label_policy``): deltas in dead zones are discarded, straddling
press/release pairs are clamped to zone boundaries. NOTE this differs near black
gaps from the ``sampled`` strings (stage 03 dropped whole black bins); ``canonical``
via the formatter is byte-identical to ``sampled`` only on dead-zone-free
stretches. Per-segment dead-zone counters land on every row
(``dead_zone_counters`` / ``dead_zone_flagged``) as a realignment health signal.

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
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Make the ``realigned_pipeline`` package importable when run directly
# (mirrors build_frames_master.py / sample_frames_actions.py).
DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.lib import config  # noqa: E402
from realigned_pipeline.lib.action_format import (  # noqa: E402
    DEFAULT_CONTINUOUS_ACTION_HZ,
    FORMATTERS,
    ActionFormatter,
    get_formatter,
)
from realigned_pipeline.lib.common import ensure_dir, read_jsonl, write_json, write_jsonl  # noqa: E402
from realigned_pipeline.lib.events import DeadZone, Window, load_events  # noqa: E402
from realigned_pipeline.lib.image_store import (  # noqa: E402
    is_arrayrecord_image_uri,
    parse_arrayrecord_image_uri,
)

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

# --action-format value meaning "no formatter": pass the stage-03 frame records'
# canonical action strings through verbatim (the historical behavior).
SAMPLED_FORMAT = "sampled"

# Default system prompts (``sampled`` mode). Goal-free reuses the verbatim
# training-time prompt ("training_v1") from the shared eval dict; goal-conditioned
# reuses the canonical one (it names a goal) but keeps the action-format contract.
GOAL_FREE_SYSTEM_PROMPT = SYSTEM_PROMPTS["yll_v1"]
GOAL_SYSTEM_PROMPT = config.SYSTEM_PROMPT

# Formatter-mode default system prompts: a fixed framing prefix + the formatter's
# own reply contract, so the default prompt always describes the SELECTED action
# format (ported from main's injection-point stage 04; for ``canonical`` the
# composition is byte-identical to the historical prompts -- the regression gate
# in tests/test_action_format.py).
GOAL_FREE_PROMPT_PREFIX = (
    "You operate a desktop computer. Each user turn shows the current screen. "
)
GOAL_PROMPT_PREFIX = (
    "You operate a desktop computer. The first user turn shows the initial "
    "screen and the user's goal; subsequent user turns show the current screen. "
)


def default_system_prompt(formatter: ActionFormatter, *, goal_conditioned: bool) -> str:
    if goal_conditioned:
        return GOAL_PROMPT_PREFIX + formatter.reply_contract.format(
            what="the next action toward that goal"
        )
    return GOAL_FREE_PROMPT_PREFIX + formatter.reply_contract.format(
        what="the next action"
    )


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


def _load_segment_frames(index_row: dict[str, Any]) -> list[dict[str, Any]] | None:
    """One segment's kept frame records, in conversation order (or None when the
    stage-03 artifact has no records for it)."""
    fr_path = index_row.get("frame_records")
    if not fr_path or not Path(fr_path).exists():
        return None
    frames = read_jsonl(Path(fr_path))
    frames.sort(key=lambda r: int(r.get("global_frame_idx") or 0))
    return frames


def _is_black_record(mrec: dict[str, Any], luma_max: float, dark_frac_min: float) -> bool:
    """Same predicate as stage 03: (near-)black per the stage-01a luma metrics;
    records without metrics are never black (absence of evidence isn't blackness)."""
    ml, fd = mrec.get("mean_luma"), mrec.get("frac_dark")
    return (ml is not None and ml <= luma_max) or (fd is not None and fd >= dark_frac_min)


def _load_sample_config(sample_dir: Path) -> dict[str, Any]:
    """The stage-03 artifact's build provenance (black-filter flag + thresholds,
    frames_master_dir, master_fps) -- what the formatter path needs to reconstruct
    dead zones exactly as the sample was built."""
    for name in ("sample_summary.json", "manifest.json"):
        path = sample_dir / name
        if path.is_file():
            return json.loads(path.read_text())
    raise SystemExit(
        f"no sample_summary.json/manifest.json under {sample_dir} -- "
        "--action-format needs the sample artifact's build provenance"
    )


def _master_manifest_path(
    frames: list[dict[str, Any]], sample_cfg: dict[str, Any], segment_id: str
) -> Path:
    """Locate the segment's master ``frame_manifest.jsonl``: prefer the shard the
    frames actually reference (``ar://<shard>#idx`` -- robust to a moved store),
    falling back to the artifact's recorded ``frames_master_dir`` layout."""
    for f in frames:
        ref = str(f.get("image_path") or "")
        if is_arrayrecord_image_uri(ref):
            shard, _ = parse_arrayrecord_image_uri(ref)
            return Path(shard).parent / "frame_manifest.jsonl"
    root = sample_cfg.get("frames_master_dir")
    if root:
        return Path(root) / "frames" / segment_id / "frame_manifest.jsonl"
    raise FileNotFoundError(
        f"{segment_id}: cannot locate the master frame_manifest.jsonl "
        "(no ar:// image refs and no frames_master_dir in the sample summary)"
    )


def reformat_segment_actions(
    frames: list[dict[str, Any]],
    index_row: dict[str, Any],
    *,
    formatter: ActionFormatter,
    sample_cfg: dict[str, Any],
    dead_zone_flag_frac: float,
) -> dict[str, Any]:
    """Replace every kept frame's ``action`` IN PLACE with ``formatter``'s label,
    re-derived from the segment's realigned keylog under the shared dead-zone
    policy (see the module docstring). Windows are the kept frames' master ticks
    tiled to the end of master coverage; dead zones are the pre-first-frame span
    plus black master ticks (when the sample was built with drop_black_frames).

    Returns segment-level accounting: ``action_format`` / ``dead_zone_counters`` /
    ``dead_zone_flagged`` (row fields) + ``primitive_counts`` (summary totals)."""
    seg = str(index_row.get("segment_id"))
    ticks = [int(f["master_record_index"]) for f in frames]
    if any(b <= a for a, b in zip(ticks, ticks[1:])):
        raise ValueError(f"{seg}: master_record_index not strictly increasing")

    manifest_path = _master_manifest_path(frames, sample_cfg, seg)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{seg}: no master frame_manifest at {manifest_path}")
    master_manifest = read_jsonl(manifest_path)
    if not master_manifest:
        raise ValueError(f"{seg}: empty master frame_manifest {manifest_path}")
    n_records = len(master_manifest)

    # Windows tile from each kept tick to the next; the last runs to the end of
    # master coverage (events past it resolve to the implicit no_coverage zone).
    axis_end = max(n_records, ticks[-1] + 1)
    windows = [
        Window(t, t, ticks[i + 1] if i + 1 < len(ticks) else axis_end)
        for i, t in enumerate(ticks)
    ]

    first_tick = ticks[0]
    dead_zones: list[DeadZone] = []
    if first_tick > 0:
        dead_zones.append(DeadZone(0, first_tick, "pre_first_frame"))
    # Black zones only when the sample was built black-filtered (same thresholds),
    # clipped to the visible region -- everything before the first kept frame is
    # already one pre_first_frame zone. Kept ticks are non-black by construction
    # (stage 03 dropped black candidate bins), so no window start is swallowed.
    if sample_cfg.get("drop_black_frames"):
        luma_max = float(sample_cfg.get("black_luma_max", config.DEFAULT_BLACK_LUMA_MAX))
        dark_min = float(sample_cfg.get("black_dark_frac_min", config.DEFAULT_BLACK_DARK_FRAC_MIN))
        run_start: int | None = None
        for tick, mrec in enumerate(master_manifest):
            black = tick >= first_tick and _is_black_record(mrec, luma_max, dark_min)
            if black and run_start is None:
                run_start = tick
            elif not black and run_start is not None:
                dead_zones.append(DeadZone(run_start, tick, "black"))
                run_start = None
        if run_start is not None:
            dead_zones.append(DeadZone(run_start, n_records, "black"))

    keylog = index_row.get("keylog_path")
    if keylog and not Path(keylog).exists():
        # load_events would silently yield no events -> an all-NO_OP segment.
        raise FileNotFoundError(f"{seg}: realigned keylog missing: {keylog}")
    events, _ = load_events(Path(keylog)) if keylog else ([], None)

    master_fps = float(index_row.get("master_fps") or sample_cfg["master_fps"])
    result = formatter.format_segment(events, windows, dead_zones, master_fps=master_fps)
    for f, label in zip(frames, result.labels, strict=True):
        f["action"] = label

    counters = result.counters
    n_discarded = (
        counters.n_discarded_black
        + counters.n_discarded_no_coverage
        + counters.n_discarded_pre_first_frame
        + 2 * counters.n_pairs_dropped_dead_zone
        + counters.n_unreleased_press_dropped
    )
    return {
        "action_format": formatter.name,
        "dead_zone_counters": asdict(counters),
        "dead_zone_flagged": len(events) > 0 and (n_discarded / len(events)) > dead_zone_flag_frac,
        "primitive_counts": result.primitive_counts,
    }


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
    frames: list[dict[str, Any]],
    *,
    instruction: str | None,
    instruction_field: str | None,
    system_prompt: str | None,
    idle_fn: Callable[[str], bool],
    fmt_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One segment -> one conversation record (frames pre-loaded and pre-filtered
    by the caller; ``fmt_fields`` carries the formatter provenance when the
    actions were re-derived -- action_format / dead-zone accounting)."""
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
        **(fmt_fields or {}),
        "n_frames": len(frames),
        "n_turns": len(frames),  # one user+assistant pair per frame
        "n_non_noop": sum(1 for f in frames if not idle_fn(str(f.get("action")))),
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
    idle_fn: Callable[[str], bool],
    terminate_token: str | None = None,
    fmt_fields: dict[str, Any] | None = None,
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
        **(fmt_fields or {}),  # segment-level formatter provenance (dead-zone counters)
        "n_frames": len(frames),
        "n_turns": len(frames),  # one user+assistant pair per frame
        "n_non_noop": sum(1 for f in frames if not idle_fn(str(f.get("action")))),
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
    p.add_argument("--action-format", type=str, default=SAMPLED_FORMAT,
                   choices=[SAMPLED_FORMAT, *sorted(FORMATTERS)],
                   help="Assistant-turn action format. 'sampled' (default): pass the "
                        "stage-03 frame records' canonical action strings through "
                        "verbatim. Any registered formatter (lib/action_format) instead "
                        "re-derives every label from the realigned keylog at build time "
                        "under the shared dead-zone policy: 'canonical' (aggregate "
                        "'<dx> <dy> <scroll> ; +KEY -KEY'), 'ordered_events_v2' (ordered "
                        "'move(dx,dy); down(LMB); ...' mini-programs), 'ordered_events_v3' "
                        "(v2 + type(\"...\") typing runs), 'computer_use_rel_v1' (Qwen "
                        "native <tool_call> JSON, relative mouse).")
    p.add_argument("--continuous-action-hz", type=float,
                   default=DEFAULT_CONTINUOUS_ACTION_HZ,
                   help="ordered_events_* only: internal motor-grid rate for accumulating "
                        "move/scroll deltas within a window (NOT a frame rate; recorded as "
                        "null for formats that ignore it).")
    p.add_argument("--dead-zone-flag-frac", type=float, default=0.05,
                   help="Formatter modes: flag a segment (dead_zone_flagged) when more than "
                        "this fraction of its keylog events were discarded by the dead-zone "
                        "policy (realignment health).")
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

    # Action format: 'sampled' keeps the frame records' strings; a registered
    # formatter re-derives every label from the realigned keylog (fails fast on an
    # unknown name / invalid hz). The idle predicate feeds n_non_noop either way.
    formatter: ActionFormatter | None = None
    sample_cfg: dict[str, Any] | None = None
    if args.action_format != SAMPLED_FORMAT:
        formatter = get_formatter(
            args.action_format, continuous_action_hz=args.continuous_action_hz
        )
        sample_cfg = _load_sample_config(args.sample_dir)
    idle_fn: Callable[[str], bool] = (
        formatter.is_idle_label if formatter is not None else (lambda a: a == "NO_OP")
    )

    # System prompt: explicit override wins; else default by goal-conditioning. A
    # segment is goal-conditioned iff it resolves an instruction, but the system
    # prompt is chosen once for the run from whether ANY instruction source is set
    # (a --goal-index run is always goal-conditioned). With a formatter active the
    # default is COMPOSED from its reply contract, so it describes the format.
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
    elif formatter is not None:
        system_prompt = default_system_prompt(formatter, goal_conditioned=goal_conditioned)
    else:
        system_prompt_id = "training_v1" if not goal_conditioned else None
        system_prompt = GOAL_SYSTEM_PROMPT if goal_conditioned else GOAL_FREE_SYSTEM_PROMPT

    # TERMINATE is goal-conditioned-only (a segment end is not a task completion).
    # With a formatter active, the default token is the formatter's own terminate
    # line ("TERMINATE" for the text formats, the native terminate tool_call for
    # computer_use_rel_v1); an explicitly different --terminate-token still wins.
    terminate_token = args.terminate_token
    if terminate_token and not args.goal_index:
        print("[conversations] WARNING: --terminate-token is ignored without --goal-index "
              "(per-segment ends are not goal completions).", flush=True)
        terminate_token = None
    elif terminate_token == "TERMINATE" and formatter is not None:
        terminate_token = formatter.terminate_line()

    out_dir = ensure_dir(args.output_dir)
    records: list[dict[str, Any]] = []
    n_skipped = 0
    n_failed = 0
    n_frames_total = 0
    n_turns_total = 0
    dz_totals: Counter = Counter()
    prim_totals: Counter = Counter()
    n_dz_flagged = 0

    def _reformat(frames: list[dict[str, Any]], row: dict[str, Any]) -> dict[str, Any] | None:
        """Re-derive one segment's labels (formatter modes) and fold its
        accounting into the run totals; returns the per-row provenance fields."""
        nonlocal n_dz_flagged
        if formatter is None or not frames:
            return None
        info = reformat_segment_actions(
            frames, row, formatter=formatter, sample_cfg=sample_cfg,
            dead_zone_flag_frac=args.dead_zone_flag_frac,
        )
        for k, v in info["dead_zone_counters"].items():
            if k != "max_simultaneous_keys":
                dz_totals[k] += int(v)
        for k, v in (info.pop("primitive_counts", None) or {}).items():
            prim_totals[k] += int(v)
        if info["dead_zone_flagged"]:
            n_dz_flagged += 1
        return info

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
            try:
                seg_frames = _load_segment_frames(row) or []
                fmt_fields = _reformat(seg_frames, row)  # once per segment, all its goals share it
            except Exception as exc:  # noqa: BLE001 - one bad segment must not abort the run
                n_failed += len(seg_goals)
                print(f"  FAIL {segment_id}: {exc}", flush=True)
                continue
            for g in seg_goals:
                conv = build_goal_conversation(
                    g, seg_frames, row, system_prompt=system_prompt,
                    min_frames=args.min_frames, idle_fn=idle_fn,
                    terminate_token=terminate_token, fmt_fields=fmt_fields,
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
            try:
                frames = _load_segment_frames(row)
                if frames is None or len(frames) < args.min_frames:
                    n_skipped += 1
                    continue
                fmt_fields = _reformat(frames, row)
                conv = build_conversation(
                    row,
                    frames,
                    instruction=args.instruction,
                    instruction_field=args.instruction_field,
                    system_prompt=system_prompt,
                    idle_fn=idle_fn,
                    fmt_fields=fmt_fields,
                )
            except Exception as exc:  # noqa: BLE001 - one bad segment must not abort the run
                n_failed += 1
                print(f"  FAIL {row.get('segment_id')}: {exc}", flush=True)
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
        "n_failed": n_failed,
        "n_frames_total": n_frames_total,
        "n_turns_total": n_turns_total,
        "mode": "goal" if args.goal_index else "segment",
        "goal_conditioned": goal_conditioned,
        "goal_index": str(args.goal_index) if args.goal_index else None,
        "action_format": args.action_format,
        "continuous_action_hz": getattr(formatter, "continuous_action_hz", None),
        "primitive_counts": dict(prim_totals) if prim_totals else None,
        "dead_zone_totals": dict(dz_totals) if formatter is not None else None,
        "n_dead_zone_flagged": n_dz_flagged if formatter is not None else None,
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
        f"{n_frames_total} frames, {n_skipped} skipped, {n_failed} failed "
        f"| format={args.action_format} -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
