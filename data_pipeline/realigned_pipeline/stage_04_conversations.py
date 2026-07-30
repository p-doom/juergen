#!/usr/bin/env python3
"""Stage 04 (conversations): the single injection point that joins frames
(stage 01, through the stage-03 filter mask), actions (stage 02's realigned
keylogs, formatted via lib/action_format's FORMATTERS), and optionally goals
(stage 03b), and emits training conversations. One script, two modes:

``--mode action`` (per-segment, fps-selected windows — the historical
build_conversations):
  * frames @ ``--fps`` (``--fps-mode`` exact/nearest) within filter survivors;
  * one conversation per segment (goal-free) or per projected goal x phrasing
    (``--goals-dir`` -> lib/goals master-interval projection, ``--use-plans``,
    ``--include-variants``, ``--snap-start``, ``--min-frames``);
  * ``--terminal-token`` glued to the final assistant message; dead-zone
    accounting + ``--dead-zone-flag-frac`` health flag.

``--mode thinking`` (day-stream windows from a lumine_thinking(-goals) 03b
artifact — the former thinking_conversations):
  * GOAL-BOUNDED (``lumine_thinking_goals``): train only on frames inside goal
    spans; windows tile each span in ``--window-frames`` (a positive multiple
    of the annotation clip stride 15, so window edges land on clip edges);
    first user turn carries ``GOAL: {goal_text}`` and, from the SECOND window
    of a span onward, ``\\nSo far: {memory}`` — the memory_out of the clip
    ending exactly at ``window_first_frame_idx - 1`` (never a clip overlapping
    or extending into the window: leak-free by construction; a span/chunk start
    withholds memory — the in-distribution episode-start shape). TERMINATE on
    the outcome frame at goal boundaries (``--terminate-boundaries``
    clean/verified/all), always its own whole action line via
    ``formatter.terminate_line()``.
  * LEGACY (``lumine_thinking``): goal-free windows tiled over each day chunk,
    ``<think>`` before the anchored action, earlier same-chunk thoughts as
    context, ``--terminal-token`` glued to the final action.

Both modes emit the canonical chat.jsonl schema (content blocks, instruction/
goal TEXT before the image on the first user turn, one assistant turn per
frame) with the same manifest join-guards, so stages 05/06 run unchanged.

Run::

    cd data_pipeline
    uv run python realigned_pipeline/stage_04_conversations.py --mode action \\
        --filter-dir <stage-03> --clips-manifest <stage-00/02 manifest> \\
        --day-index-cache <cache.json> --fps 1 --output-dir <dest> \\
        [--goals-dir <stage-03b> --use-plans --include-variants]

    uv run python realigned_pipeline/stage_04_conversations.py --mode thinking \\
        --filter-dir <stage-03> --clips-manifest <stage-00/02 manifest> \\
        --day-index-cache <cache.json> --goals-dir <lumine_thinking(-goals)> \\
        --fps 0.5 --window-frames 30 --output-dir <dest>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Make the ``realigned_pipeline`` package importable when run directly
# (mirrors the other stages).
DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.annotation.lib.days import (  # noqa: E402
    DEFAULT_GAP_CUT_S,
    DEFAULT_TZ,
    DayStream,
    build_day_stream,
    fmt_t,
)
from realigned_pipeline.lib.action_format import (  # noqa: E402
    DEFAULT_CONTINUOUS_ACTION_HZ,
    FORMATTERS,
    ActionFormatter,
    get_formatter,
)
from realigned_pipeline.lib.common import (  # noqa: E402
    normalize_dashed_argv,
    read_jsonl,
    str2bool,
)
from realigned_pipeline.lib.conversations import (  # noqa: E402
    CLIP_STRIDE,
    check_day_selection_args,
    image_block,
    load_or_build_day_index,
    require_window_alignment,
    select_day_rows,
    text_block,
    write_conversation_artifact,
)
from realigned_pipeline.lib.events import load_events  # noqa: E402
from realigned_pipeline.lib.goals import (  # noqa: E402
    SNAP_START_MODES,
    assert_same_artifact,
    goals_by_segment,
    load_goals,
    project_goals,
)
from realigned_pipeline.lib.manifest import make_artifact_id  # noqa: E402
from realigned_pipeline.lib.views import (  # noqa: E402
    FPS_MODES,
    FilterArtifact,
    build_segment_view,
)

# Aliases: the two old scripts used these private spellings; keep them so the
# body reads like the originals and both modes share one block schema.
_text = text_block
_image = image_block


# ===========================================================================
# MODE: action  (per-segment fps-selected windows)
# ===========================================================================

# Default system prompts: a fixed framing prefix + the formatter's own reply
# contract, so the prompt always describes the selected action format. For
# ``canonical`` the composition is byte-identical to the historical prompts
# (goal-conditioned == config.SYSTEM_PROMPT — the regression gate in
# tests/test_action_format.py).
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


# Plans carrying these quality flags are unusable as training prose; the
# conversation falls back to a plan-less first turn (same as the fold
# pipeline's assemble step).
DROP_PLAN_FLAGS = frozenset({"empty", "restates_instruction"})


def build_messages(
    turns: list[tuple[str, str]],  # ordered (image, action) pairs
    *,
    instruction: str | None,
    system_prompt: str | None,
    plan: str | None = None,
    terminal_token: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble one interleaved conversation. Canonical chat.jsonl schema:
    instruction TEXT before the image on the first user turn, image-only on
    later turns, one assistant turn per frame carrying its action. The plan
    prefixes the FIRST assistant turn (``<plan>\\n<action>``); the terminal
    token rides at the END of the final assistant message."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": [_text(system_prompt)]})
    last = len(turns) - 1
    for idx, (image, action) in enumerate(turns):
        content: list[dict[str, Any]] = []
        if idx == 0 and instruction:
            content.append(_text(instruction))
        content.append(_image(image))
        messages.append({"role": "user", "content": content})
        text = action
        if idx == 0 and plan:
            text = f"{plan}\n{text}"
        if idx == last and terminal_token:
            text = f"{text}\n{terminal_token}"
        messages.append({"role": "assistant", "content": [_text(text)]})
    return messages


def usable_plan(goal: dict[str, Any]) -> str:
    plan = str(goal.get("plan") or "").strip()
    if set(goal.get("plan_flags") or []) & DROP_PLAN_FLAGS:
        return ""
    return plan


def instruction_phrasings(goal: dict[str, Any], include_variants: bool) -> list[str]:
    """Primary instruction plus register paraphrases, de-duplicated; each
    becomes its own conversation over the same frames/actions."""
    out: list[str] = []
    candidates = [goal.get("instruction")]
    if include_variants:
        candidates += list(goal.get("instruction_variants") or [])
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _resolve_instruction(
    index_row: dict[str, Any],
    filter_seg: dict[str, Any],
    *,
    instruction: str | None,
    instruction_field: str | None,
) -> str | None:
    """Goal-free per-segment instruction from --instruction-field (filter index
    row, then the segment filter doc), else the fixed --instruction, else None."""
    if instruction_field:
        for src in (index_row, filter_seg):
            val = src.get(instruction_field)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return instruction


def build_segment_conversations(task: dict[str, Any]) -> dict[str, Any]:
    """Worker: one segment -> conversation rows (one per segment goal-free;
    one per goal x phrasing in goal mode) + per-segment accounting. Failures
    are captured, never raised."""
    seg = str(task["index_row"]["segment_id"])
    try:
        filter_seg = json.loads(Path(task["filter_seg_path"]).read_text())
        view = build_segment_view(filter_seg, fps=task["fps"], fps_mode=task["fps_mode"])
        base = {
            "segment_id": seg,
            "recording_id": view.recording_id,
            "segment_idx": view.segment_idx,
            "alignment_status": view.alignment_status,
        }
        if not view.frames:
            return {**base, "status": "empty_view", "rows": []}

        keylog = Path(view.keylog_path) if view.keylog_path else None
        events, _ = load_events(keylog) if keylog else ([], None)
        fmt = get_formatter(
            task["action_format"], continuous_action_hz=task["continuous_action_hz"]
        )
        result = fmt.format_segment(
            events, view.windows(), view.dead_zones, master_fps=view.master_fps,
            frame_size=task.get("frame_size"),
        )
        counters = result.counters
        n_events = len(events)
        n_discarded = (
            counters.n_discarded_black
            + counters.n_discarded_no_coverage
            + counters.n_discarded_pre_first_frame
            + 2 * counters.n_pairs_dropped_dead_zone
            + counters.n_unreleased_press_dropped
        )
        flagged = n_events > 0 and (n_discarded / n_events) > task["dead_zone_flag_frac"]

        common = {
            **base,
            "target_fps": task["fps"],
            "action_format": task["action_format"],
            "dead_zone_counters": asdict(counters),
            "dead_zone_flagged": flagged,
        }
        rows: list[dict[str, Any]] = []

        if task["goals_by_segment"] is None:
            if len(view.frames) < task["min_frames"]:
                return {**base, "status": "below_min_frames", "rows": []}
            turns = [(str(f.image), result.labels[f.view_idx]) for f in view.frames]
            instruction = _resolve_instruction(
                task["index_row"],
                filter_seg,
                instruction=task["instruction"],
                instruction_field=task["instruction_field"],
            )
            messages = build_messages(
                turns,
                instruction=instruction,
                system_prompt=task["system_prompt"],
                terminal_token=task["terminal_token"],
            )
            rows.append({
                "conversation_id": seg,
                **common,
                "instruction": instruction,
                "goal_conditioned": instruction is not None,
                "n_frames": len(turns),
                "n_turns": len(turns),
                "n_non_noop": sum(1 for _, a in turns if a != "NO_OP"),
                "messages": messages,
            })
            return {
                **base,
                "status": "ok",
                "rows": rows,
                "primitive_counts": result.primitive_counts,
            }

        seg_goals = task["goals_by_segment"].get(seg, [])
        if not seg_goals:
            return {**base, "status": "no_goals", "rows": []}
        projections, proj_stats = project_goals(
            seg_goals, view, snap_start=task["snap_start"], min_frames=task["min_frames"]
        )
        for proj in projections:
            goal = proj.goal
            plan = usable_plan(goal) if task["use_plans"] else ""
            turns = [(str(f.image), result.labels[f.view_idx]) for f in proj.frames]
            for variant_idx, phrasing in enumerate(
                instruction_phrasings(goal, task["include_variants"])
            ):
                suffix = f"_v{variant_idx}" if variant_idx else ""
                messages = build_messages(
                    turns,
                    instruction=phrasing,
                    system_prompt=task["system_prompt"],
                    plan=plan or None,
                    terminal_token=task["terminal_token"],
                )
                rows.append({
                    "conversation_id": f"{seg}:{goal['goal_id']}{suffix}",
                    **common,
                    "goal_id": goal["goal_id"],
                    "instruction": phrasing,
                    "variant_idx": variant_idx,
                    "goal_conditioned": True,
                    "plan": plan,
                    "start_master_idx": int(goal["start_master_idx"]),
                    "end_master_idx": int(goal["end_master_idx"]),
                    "snapped_start": proj.snapped_start,
                    "annotation_method": goal.get("method"),
                    "n_frames": len(turns),
                    "n_turns": len(turns),
                    "n_non_noop": sum(1 for _, a in turns if a != "NO_OP"),
                    "messages": messages,
                })
        return {
            **base,
            "status": "ok" if rows else "no_projected_goals",
            "rows": rows,
            "primitive_counts": result.primitive_counts,
            "projection": {
                "n_goals": proj_stats.n_goals,
                "n_projected": proj_stats.n_projected,
                "n_empty_projection": proj_stats.n_empty_projection,
                "n_too_few_frames": proj_stats.n_too_few_frames,
                "n_snapped": proj_stats.n_snapped,
                "rejected": proj_stats.rejected,
            },
        }
    except Exception as exc:
        return {
            "segment_id": seg,
            "status": "failed",
            "rows": [],
            "error": f"{exc}",
            "traceback": traceback.format_exc(),
        }


def _segments_in_days(art: FilterArtifact, clips_manifest: Path,
                      day_index_cache: Path | None, *, tz: str,
                      day_filter: list[str] | None,
                      day_exclude: list[str] | None) -> set[str] | None:
    """Segment ids belonging to the selected days, or None when no day
    selection was requested (action mode then processes every usable segment,
    byte-identically to the historical builder)."""
    if not (day_filter or day_exclude):
        return None
    day_rows = load_or_build_day_index(art, clips_manifest,
                                       day_index_cache=day_index_cache, tz=tz)
    kept = select_day_rows(day_rows, day_filter=day_filter, day_exclude=day_exclude)
    return {str(s["segment_id"]) for d in kept for s in d["segments"]}


def _capture_dims(clips_manifest: Path) -> dict[str, tuple[int, int]]:
    """``segment_id -> (video_width, video_height)`` — the ORIGINAL capture
    size, which is what a normalized action format divides by. Rows without
    usable dims are omitted rather than defaulted."""
    dims: dict[str, tuple[int, int]] = {}
    for r in read_jsonl(clips_manifest):
        w, h = r.get("video_width"), r.get("video_height")
        if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
            dims[str(r["segment_id"])] = (w, h)
    return dims


def _require_capture_dims(
    rows: list[dict], dims: dict[str, tuple[int, int]]
) -> tuple[list[dict], int]:
    """Keep only rows whose segment has capture dims, plus the dropped count.

    A normalized format must never fall back to a default size: that would
    silently emit deltas on a different scale than the rest of the corpus."""
    kept = [r for r in rows if str(r["segment_id"]) in dims]
    return kept, len(rows) - len(kept)


def run_action(args: argparse.Namespace) -> None:
    art = FilterArtifact(args.filter_dir)
    stride = art.stride_for(args.fps, args.fps_mode)  # fail fast on invalid rates

    goals_id = None
    goals_map = None
    if args.goals_dir is not None:
        goals_manifest_path = args.goals_dir / "manifest.json"
        if not goals_manifest_path.is_file():
            raise SystemExit(f"no manifest.json under {args.goals_dir} (not a goals artifact?)")
        goals_manifest = json.loads(goals_manifest_path.read_text())
        assert_same_artifact(
            str(goals_manifest.get("master_store_id")), art.master_store_id, what="master_store_id"
        )
        assert_same_artifact(
            str(goals_manifest.get("filter_id")), art.filter_id, what="filter_id"
        )
        goals_map = goals_by_segment(load_goals(args.goals_dir / "goals.jsonl"))
        goals_id = make_artifact_id(args.goals_dir)

    # Fails fast on an unknown format / invalid hz; also carries the format's
    # provenance attributes (reply contract, hz) for the prompt and manifest.
    formatter = get_formatter(
        args.action_format, continuous_action_hz=args.continuous_action_hz
    )
    continuous_action_hz = getattr(formatter, "continuous_action_hz", None)

    goal_mode = goals_map is not None
    if args.no_system_prompt:
        system_prompt = None
    elif args.system_prompt_file is not None:
        system_prompt = args.system_prompt_file.read_text().strip()
    elif args.system_prompt is not None:
        system_prompt = args.system_prompt
    else:
        goal_conditioned = goal_mode or bool(args.instruction or args.instruction_field)
        system_prompt = default_system_prompt(formatter, goal_conditioned=goal_conditioned)

    day_segments = _segments_in_days(
        art, args.clips_manifest, args.day_index_cache, tz=args.tz,
        day_filter=args.day_filter, day_exclude=args.day_exclude)

    rows_in = art.usable_rows()
    if day_segments is not None:
        rows_in = [r for r in rows_in if str(r["segment_id"]) in day_segments]
    if args.limit is not None:
        rows_in = rows_in[: args.limit]
    if not rows_in:
        raise SystemExit(f"no usable segments in {art.dir}")

    seg_dims = _capture_dims(args.clips_manifest)
    if getattr(formatter, "normalize_moves", False):
        rows_in, n_no_dims = _require_capture_dims(rows_in, seg_dims)
        if n_no_dims:
            print(f"[conversations] dropped {n_no_dims} segment(s) with no "
                  f"capture dims in {args.clips_manifest} "
                  f"(required by {args.action_format})", flush=True)
        if not rows_in:
            raise SystemExit(
                f"no segment in {art.dir} has capture dims in "
                f"{args.clips_manifest}; {args.action_format} cannot normalize")

    tasks = [
        {
            "index_row": row,
            "filter_seg_path": str(art.segment_path(str(row["segment_id"]))),
            "frame_size": seg_dims.get(str(row["segment_id"])),
            "fps": args.fps,
            "fps_mode": args.fps_mode,
            "action_format": args.action_format,
            "continuous_action_hz": args.continuous_action_hz,
            "goals_by_segment": goals_map,
            "instruction": args.instruction,
            "instruction_field": args.instruction_field,
            "system_prompt": system_prompt,
            "use_plans": args.use_plans,
            "include_variants": args.include_variants,
            "min_frames": args.min_frames,
            "snap_start": args.snap_start,
            "terminal_token": args.terminal_token,
            "dead_zone_flag_frac": args.dead_zone_flag_frac,
        }
        for row in rows_in
    ]

    n_workers = max(1, min(args.num_workers or mp.cpu_count(), len(tasks)))
    print(
        f"[conversations] mode=action | {len(tasks)} segments | fps={args.fps} "
        f"({args.fps_mode}, stride {stride:g}) format={args.action_format} "
        f"goals={'yes' if goal_mode else 'no'} | workers={n_workers}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    counts: Counter = Counter()
    dz_totals: Counter = Counter()
    proj_totals: Counter = Counter()
    prim_totals: Counter = Counter()
    n_flagged = 0
    with mp.Pool(n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(build_segment_conversations, tasks, chunksize=8), 1):
            counts[res["status"]] += 1
            if res["status"] == "failed":
                print(f"  FAIL {res['segment_id']}: {res.get('error')}", flush=True)
            for row in res["rows"]:
                records.append(row)
                if row.get("dead_zone_flagged"):
                    n_flagged += 1
                for k, v in row.get("dead_zone_counters", {}).items():
                    if k != "max_simultaneous_keys":
                        dz_totals[k] += int(v)
            for k, v in (res.get("projection") or {}).items():
                if isinstance(v, int):
                    proj_totals[k] += v
            for k, v in (res.get("primitive_counts") or {}).items():
                prim_totals[k] += int(v)
            if i % 1000 == 0:
                print(f"  {i}/{len(tasks)} segments | {len(records)} conversations", flush=True)

    if not records:
        raise SystemExit("no conversations built (all segments empty/rejected)")

    summary = {
        "mode": "action",
        "n_conversations": len(records),
        "n_frames_total": sum(r["n_frames"] for r in records),
        "n_turns_total": sum(r["n_turns"] for r in records),
        "status_counts": dict(counts),
        "goal_mode": goal_mode,
        "fps": args.fps,
        "fps_mode": args.fps_mode,
        "stride": stride,
        "master_fps": art.master_fps,
        "action_format": args.action_format,
        "continuous_action_hz": continuous_action_hz,
        "primitive_counts": dict(prim_totals) if prim_totals else None,
        "n_noop_turns": sum(r["n_turns"] - r["n_non_noop"] for r in records),
        "instruction": args.instruction,
        "instruction_field": args.instruction_field,
        "has_system_prompt": system_prompt is not None,
        "use_plans": args.use_plans,
        "include_variants": args.include_variants,
        "min_frames": args.min_frames,
        "snap_start": args.snap_start,
        "terminal_token": args.terminal_token,
        "dead_zone_totals": dict(dz_totals),
        "n_dead_zone_flagged": n_flagged,
        "goal_projection_totals": dict(proj_totals),
        "filter_dir": str(art.dir),
        "goals_dir": str(args.goals_dir) if args.goals_dir else None,
    }
    out_dir = write_conversation_artifact(
        args.output_dir, records, summary,
        master_store_id=art.master_store_id, filter_id=art.filter_id, goals_id=goals_id)
    print(
        f"[conversations] {len(records)} conversations, {summary['n_turns_total']} turns, "
        f"{summary['n_frames_total']} frames | statuses {dict(counts)} | "
        f"dead-zone flagged {n_flagged} -> {out_dir}",
        flush=True,
    )


# ===========================================================================
# MODE: thinking  (day-stream windows from a lumine_thinking(-goals) artifact)
# ===========================================================================

# Goal-free (legacy) system prompt + the think sentences: one short first-person
# thought at commit/switch/reconsider moments, silence during steady execution,
# an earlier-thoughts recap block.
THINKING_SYSTEM_PROMPT = (
    "You operate a desktop computer. Each user turn shows the current screen. "
    "Reply with the next action as `<dx> <dy> <scroll>` optionally followed by "
    "` ; +KEY -KEY` events, or `NO_OP` if no action. At a decision point — when "
    "you commit to, switch, or reconsider what you are doing — first write one "
    "short thought in your own voice inside <think></think> tags, then the "
    "action on the next line. Steady execution needs no thought. A \"Your "
    "thoughts so far this session:\" note at the start of a session recaps "
    "your earlier thinking."
)
CONTEXT_HEADER = "Your thoughts so far this session:"

# Goal-bounded mode constants. The terminate turn is always the ENTIRE action
# payload of its own turn (see system_prompts/cua_v3_thinking.txt /
# cua_v4_thinking.txt), never glued to a recorded action the way the legacy
# --terminal-token is. The payload itself comes from the selected formatter's
# ``terminate_line()`` — the TERMINATE literal for the text formats, the native
# terminate tool_call block for computer_use_rel_v1.
TERMINATE_TOKEN = "TERMINATE"
TERMINATE_BOUNDARY_MODES = ("clean", "verified", "all")
DEFAULT_TERMINATE_MAX_LAG_S = 180.0
SYSTEM_PROMPTS_DIR = Path(__file__).resolve().parent / "system_prompts"
DEFAULT_GOAL_SYSTEM_PROMPT_FILE = SYSTEM_PROMPTS_DIR / "cua_v3_thinking.txt"
# Formats whose default goal prompt is NOT cua_v3 (the tool spec must match the
# emission format the formatter produces).
GOAL_SYSTEM_PROMPT_FILES = {
    "computer_use_rel_v1": SYSTEM_PROMPTS_DIR / "cua_v4_thinking.txt",
    # Same tool spec, but the mouse_move_rel delta is declared as a 0-1000
    # screen fraction — the model must be told the units it is trained on.
    "computer_use_rel_norm_v1": SYSTEM_PROMPTS_DIR / "cua_v4_thinking_norm.txt",
    "computer_use_rel_step_v1": SYSTEM_PROMPTS_DIR / "cua_rel_step_v1_thinking.txt",
    # ordered_events_v2 needs the type()-free ordered prompt (cua_v3 advertises
    # type(), which the v2 formatter never emits).
    "ordered_events_v2": SYSTEM_PROMPTS_DIR / "cua_oev2_thinking.txt",
}


def goal_system_prompt_file(action_format: str) -> Path:
    """Goal mode's default system prompt for the selected action format."""
    return GOAL_SYSTEM_PROMPT_FILES.get(action_format, DEFAULT_GOAL_SYSTEM_PROMPT_FILE)


def resolve_terminal_token(terminal_token: str | None, action_format: str) -> str | None:
    """Legacy-mode --terminal-token: the literal ``TERMINATE`` sentinel resolves
    through the formatter's ``terminate_line()`` (the same literal for
    canonical/ordered_events_* — byte-identical legacy behavior — and the native
    terminate block for computer_use_rel_v1). Anything else, including the
    default None, passes through untouched."""
    if terminal_token == TERMINATE_TOKEN:
        return get_formatter(action_format).terminate_line()
    return terminal_token


_WORKER: dict[str, Any] = {}


def _init_worker(filter_dir: str) -> None:
    _WORKER["art"] = FilterArtifact(Path(filter_dir))


def _norm(text: Any) -> str:
    return " ".join(str(text).split())


def context_block(prior: list[dict[str, Any]], keep: int) -> str:
    if not prior or keep <= 0:
        return ""
    lines = [f"[{fmt_t(float(t['t_day_s']))}] {t['instruction']}" for t in prior[-keep:]]
    return CONTEXT_HEADER + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Goal-bounded mode: sidecar loading + pure window construction (unit-tested)
# ---------------------------------------------------------------------------


def load_goal_memory_sidecars(goals_dir: Path, day_tag: str
                              ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the lumine_thinking_goals sidecars for one day:
      units/<day>/goals_active.jsonl -> per-clip active fold goal
      memory/<day>.jsonl            -> per-clip rolling memory
    Both are keyed by day_idx_range; empty lists if absent (unconditioned day)."""
    ga = goals_dir / "units" / day_tag / "goals_active.jsonl"
    mem = goals_dir / "memory" / f"{day_tag}.jsonl"
    active = read_jsonl(ga) if ga.is_file() else []
    memory = read_jsonl(mem) if mem.is_file() else []
    return active, memory


def resolve_boundaries_dir(boundaries_dir: Path) -> Path:
    """Accept either a lumine_goal_boundaries artifact root (containing
    ``boundaries/``) or the boundaries/ directory itself."""
    sub = boundaries_dir / "boundaries"
    return sub if sub.is_dir() else boundaries_dir


def load_boundaries(boundaries_dir: Path | None, day_tag: str) -> dict[Any, dict[str, Any]]:
    """{goal_id -> boundaries row} for one day (lumine_goal_boundaries sidecar:
    one row per goal span, fail-closed judgments)."""
    if boundaries_dir is None:
        return {}
    p = boundaries_dir / f"{day_tag}.jsonl"
    if not p.is_file():
        return {}
    return {r["goal_id"]: r for r in read_jsonl(p)}


def check_sidecar_alignment(row: dict[str, Any], day: DayStream) -> None:
    """Assert the rebuilt day stream addresses the sidecar's clip: the frames at
    its recorded day_idx_range must carry its recorded t_range (same guard as
    lumine_goal_boundaries) — a mismatch means wrong filter/fps/tz."""
    i0, i1 = (int(x) for x in row["day_idx_range"])
    if not (0 <= i0 <= i1 < len(day.frames)):
        raise ValueError(
            f"{day.day_tag}/{row.get('clip_key')}: day_idx_range [{i0}, {i1}] outside the "
            f"rebuilt day stream ({len(day.frames)} frames) — rerun with the producing "
            "run's --filter-dir/--fps/--tz/--gap-cut-s")
    want = row.get("t_range")
    got = [fmt_t(day.frames[i0].t_day_s), fmt_t(day.frames[i1].t_day_s)]
    if want and got != list(want):
        raise ValueError(
            f"{day.day_tag}/{row.get('clip_key')}: rebuilt day stream misaligned with the "
            f"goals sidecar (t_range {got} != {list(want)}) — rerun with the producing "
            "run's --filter-dir/--fps/--tz/--gap-cut-s")


def reindex_active_rows(
    active: list[dict[str, Any]],
    annotation_day: DayStream,
    decision_day: DayStream,
    *,
    annotation_fps: float,
) -> list[dict[str, Any]]:
    """Project annotation-fps goal clips onto a denser decision stream.

    Goal sidecars remain immutable evidence from the annotation run. Their
    dense ``day_idx_range`` addresses the annotation stream, so a 4 Hz action
    dataset must not interpret those integers on its own stream. We first
    validate them against the original day, derive each clip's half-open time
    interval, and select the decision frames in that interval.
    """
    if annotation_fps <= 0:
        raise ValueError(f"annotation_fps must be > 0, got {annotation_fps}")
    out: list[dict[str, Any]] = []
    for row in active:
        check_sidecar_alignment(row, annotation_day)
        i0, i1 = (int(x) for x in row["day_idx_range"])
        start_s = annotation_day.frames[i0].t_day_s
        end_s = (
            annotation_day.frames[i1 + 1].t_day_s
            if i1 + 1 < len(annotation_day.frames)
            else annotation_day.frames[i1].t_day_s + 1.0 / annotation_fps
        )
        mapped = [
            frame.day_idx
            for frame in decision_day.frames
            if start_s <= frame.t_day_s < end_s
        ]
        if not mapped:
            raise ValueError(
                f"{decision_day.day_tag}/{row.get('clip_key')}: no decision frames "
                f"in annotation interval [{start_s}, {end_s})"
            )
        projected = dict(row)
        projected["source_day_idx_range"] = list(row["day_idx_range"])
        projected["day_idx_range"] = [mapped[0], mapped[-1]]
        projected["t_range"] = [
            fmt_t(decision_day.frames[mapped[0]].t_day_s),
            fmt_t(decision_day.frames[mapped[-1]].t_day_s),
        ]
        out.append(projected)
    return out


def group_goal_runs(active: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Goal SPANS as contiguous runs: consecutive goals_active rows sharing a
    non-null goal_id. A goal_id that recurs after an interruption (no-goal or
    other-goal clips in between) yields SEPARATE runs. Each run carries
    ``next_goal_id`` — the goal_id of the clip right after the run (None at day
    end or before a no-goal clip) — for the clean-mode handoff test."""
    rows = sorted(active, key=lambda r: int(r["day_idx_range"][0]))
    runs: list[dict[str, Any]] = []
    for pos, r in enumerate(rows):
        gid = r.get("goal_id")
        if gid is None:
            continue
        if runs and runs[-1]["goal_id"] == gid and runs[-1]["_last_pos"] == pos - 1:
            runs[-1]["clips"].append(r)
            runs[-1]["_last_pos"] = pos
        else:
            runs.append({
                "goal_id": gid,
                "goal_text": r.get("goal_text"),
                "goal_t_start": r.get("goal_t_start"),
                "goal_t_end": r.get("goal_t_end"),
                "clips": [r],
                "_last_pos": pos,
            })
    for run in runs:
        nxt = run.pop("_last_pos") + 1
        run["start_idx"] = int(run["clips"][0]["day_idx_range"][0])
        run["end_idx"] = int(run["clips"][-1]["day_idx_range"][1])
        run["next_goal_id"] = rows[nxt].get("goal_id") if nxt < len(rows) else None
    return runs


def select_memory(memory_rows: list[dict[str, Any]], win_start_idx: int
                  ) -> tuple[str, str]:
    """Leak-free "So far:" selection. The memory MUST be the memory_out of the
    clip whose ``day_idx_range`` END equals ``win_start_idx - 1`` exactly — the
    clip immediately BEFORE the window's first frame, so nothing it summarizes
    lies inside the window (day_idx_range is inclusive on both ends; a window
    edge aligned to a clip edge makes this predecessor well defined). NEVER a
    clip that overlaps or extends into the window.

    Returns ``(memory_text, status)``:
      * ``"ok"``    — exact predecessor clip found, non-empty memory (attach);
      * ``"empty"`` — exact predecessor found but it summarizes nothing;
      * ``"none"``  — no memory row ends before the window at all;
      * ``"gap"``   — a row ends before the window but NOT exactly at
                      ``win_start-1`` (a missing/short sidecar clip): omit
                      rather than attach a gappy/leaky memory, count it.
    """
    want = win_start_idx - 1
    exact: dict[str, Any] | None = None
    any_before = False
    for row in memory_rows:
        rng = row.get("day_idx_range") or [None, None]
        if rng[1] is None:
            continue
        end = int(rng[1])
        if end < win_start_idx:
            any_before = True
        if end == want and (
            exact is None
            or int(row["day_idx_range"][0]) > int(exact["day_idx_range"][0])
        ):
            exact = row
    if exact is not None:
        mem = _norm(exact.get("memory", ""))
        return (mem, "ok") if mem else ("", "empty")
    return ("", "gap" if any_before else "none")


def goal_memory_block(goal_text: str, memory: str) -> str:
    """First-user-turn conditioning: the span's goal + rolling memory (the
    memory IS the contextual summary — no earlier-thoughts recap)."""
    block = f"GOAL: {goal_text}"
    if memory:
        block += f"\nSo far: {memory}"
    return block


def decide_terminate(
    mode: str,
    brow: dict[str, Any] | None,
    outcome_idx: int | None,
    outcome_lag_s: float | None,
    next_goal_id: Any,
    max_lag_s: float,
    *,
    terminate_line: str = TERMINATE_TOKEN,
) -> tuple[str | None, str | None]:
    """(terminate turn text, skip reason). A terminate needs the outcome frame
    to exist in every mode; the modes differ in what else they demand:
      verified — the boundaries judge said completed && confidence high
                 (target carries its final_thought);
      clean    — outcome within max_lag_s AND goal->goal clip handoff;
      all      — outcome frame alone (ablations).
    ``terminate_line`` is the formatter's complete goal-done action payload
    (``ActionFormatter.terminate_line()``)."""
    if mode == "verified":
        if brow is None:
            return None, "no_boundaries_row"
        if not (brow.get("completed") is True and brow.get("confidence") == "high"):
            return None, "not_completed_high"
        if outcome_idx is None:
            return None, "no_outcome_frame"
        thought = _norm(brow.get("final_thought") or "")
        if not thought:  # parse_judgment demotes these to low; belt and braces
            return None, "empty_final_thought"
        return f"<think>\n{thought}\n</think>\n{terminate_line}", None
    if outcome_idx is None:
        return None, "no_outcome_frame"
    if mode == "all":
        return terminate_line, None
    # clean: structurally-clean subset
    if outcome_lag_s is None or outcome_lag_s > max_lag_s:
        return None, "lag_exceeded"
    if next_goal_id is None:
        return None, "no_goal_handoff"
    return terminate_line, None


def build_goal_day_rows(
    day: DayStream,
    active: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
    boundaries: dict[Any, dict[str, Any]],
    anchored: dict[int, dict[str, Any]],
    *,
    window_frames: int,
    terminate_mode: str,
    terminate_max_lag_s: float,
    min_anchor_lead: int,
    system_prompt: str | None,
    fps: float,
    action_format: str,
    annotation_method: str,
    context_images: int = 0,
    omit_goal_memory: bool = False,
    idle_keep_fraction: float = 1.0,
) -> tuple[list[dict[str, Any]], Counter]:
    """Pure goal-bounded window construction over one rebuilt day stream.
    ``anchored`` maps day_idx -> verified-thought row (goals.jsonl). Windows
    tile each goal span in ``window_frames`` (a positive multiple of the clip
    stride) from each chunk-cut piece start, so every window edge lands on a
    clip edge and the "So far:" memory predecessor is exact. Returns
    (conversation rows, stats Counter)."""
    if terminate_mode not in TERMINATE_BOUNDARY_MODES:
        raise ValueError(f"terminate mode must be one of {TERMINATE_BOUNDARY_MODES}, "
                         f"got {terminate_mode!r}")
    if not 0.0 <= idle_keep_fraction <= 1.0:
        raise ValueError(
            f"idle_keep_fraction must be in [0,1], got {idle_keep_fraction}"
        )
    # the formatter's complete goal-done action payload (TERMINATE literal for
    # the text formats, the native terminate tool_call for computer_use_rel_v1)
    formatter = get_formatter(action_format)
    terminate_line = formatter.terminate_line()
    frames = day.frames
    stats: Counter = Counter()
    for row in active:
        check_sidecar_alignment(row, day)

    chunk_of: dict[int, int] = {}
    chunk_first: dict[int, int] = {}
    chunk_last: dict[int, int] = {}
    for ci, chunk in enumerate(day.chunks):
        chunk_first[ci] = chunk[0].day_idx
        chunk_last[ci] = chunk[-1].day_idx
        for fr in chunk:
            chunk_of[fr.day_idx] = ci

    runs = group_goal_runs(active)
    stats["n_spans"] = len(runs)
    stats["n_no_goal_frames_excluded"] = (
        len(frames) - sum(r["end_idx"] - r["start_idx"] + 1 for r in runs))

    # The terminate logic applies only to the run containing goal_t_end: the
    # LAST run of each goal_id whose frames start at or before it (recurring
    # goal_ids: earlier interrupted runs never terminate).
    terminate_run: dict[Any, int] = {}
    for ri, run in enumerate(runs):
        te = run.get("goal_t_end")
        if te is not None and frames[run["start_idx"]].t_day_s <= float(te):
            terminate_run[run["goal_id"]] = ri

    # ---- pass 1: per-run frame sets + terminate decisions -----------------
    plans: list[dict[str, Any]] = []
    for ri, run in enumerate(runs):
        gid = run["goal_id"]
        te = float(run["goal_t_end"]) if run.get("goal_t_end") is not None else None
        idxs = list(range(run["start_idx"], run["end_idx"] + 1))
        terminate_text = None
        outcome_idx = None
        if terminate_run.get(gid) == ri and te is not None:
            sup = [i for i in idxs if frames[i].t_day_s <= te]
            if sup:
                last_sup = sup[-1]
                ci = chunk_of[last_sup]
                outcome = next(
                    (j for j in range(last_sup + 1, chunk_last[ci] + 1)
                     if frames[j].t_day_s > te), None)
                lag = frames[outcome].t_day_s - te if outcome is not None else None
                terminate_text, skip = decide_terminate(
                    terminate_mode, boundaries.get(gid), outcome, lag,
                    run["next_goal_id"], terminate_max_lag_s,
                    terminate_line=terminate_line)
                if terminate_text is not None:
                    outcome_idx = outcome
                    stats["n_frames_dropped_post_goal"] += len(idxs) - len(sup)
                    idxs = sup
                else:
                    stats[f"n_terminate_skipped_{skip}"] += 1
            else:
                stats["n_terminate_skipped_no_frames_before_end"] += 1
        plans.append({"run": run, "run_index": ri, "idxs": idxs,
                      "terminate_text": terminate_text, "outcome_idx": outcome_idx})

    # ---- near-miss negatives (verified mode): thought replacement map ------
    # {(goal_id, day_idx of the near-miss clip's first frame) -> thought}
    nm_thoughts: dict[tuple[Any, int], str] = {}
    if terminate_mode == "verified":
        emitted: dict[Any, set[int]] = {}
        for plan in plans:
            emitted.setdefault(plan["run"]["goal_id"], set()).update(plan["idxs"])
        for gid, brow in boundaries.items():
            nm = brow.get("near_miss")
            if not (brow.get("completed") is True and brow.get("confidence") == "high" and nm):
                continue
            thought = _norm(nm.get("next_step_thought") or "")
            first = int(nm["day_idx_range"][0])
            if thought and first in emitted.get(gid, set()):
                nm_thoughts[(gid, first)] = thought
            else:
                stats["n_near_miss_unplaced"] += 1

    # ---- pass 2: tile + emit ------------------------------------------------
    rows: list[dict[str, Any]] = []
    for plan in plans:
        run, idxs = plan["run"], plan["idxs"]
        gid = run["goal_id"]
        gid_s = f"{gid:04d}" if isinstance(gid, int) else str(gid)
        goal_text = _norm(run.get("goal_text") or "")
        if not goal_text:
            raise ValueError(f"{day.day_tag}: goal {gid!r} has clips but no goal_text")
        if not idxs:
            continue

        # windows never cross a recording-gap chunk boundary
        pieces: list[list[int]] = [[idxs[0]]]
        for i in idxs[1:]:
            if chunk_of[i] != chunk_of[pieces[-1][-1]]:
                pieces.append([])
            pieces[-1].append(i)

        w_seq = 0
        for piece in pieces:
            if context_images > 0:
                # Decision-record mode: one supervised assistant decision per
                # record, conditioned only on GOAL + recent screenshots.  No
                # prior assistant text is replayed, so standardized fixed-step
                # labels never claim to have caused the demonstrator's next
                # device-specific cursor displacement.
                for pos_in_piece, current in enumerate(piece):
                    ctx = piece[max(0, pos_in_piece - context_images + 1):pos_in_piece + 1]
                    target = frames[current].action
                    thought_ids: list[str] = []
                    th = anchored.get(current)
                    if th is not None:
                        target = f"<think>\n{th['instruction']}\n</think>\n{target}"
                        thought_ids.append(str(th["goal_id"]))
                    record_key = (
                        f"{day.day_tag}:{gid_s}:{plan['run_index']}:{current}"
                    )
                    bucket = int(
                        hashlib.sha1(record_key.encode()).hexdigest()[:8], 16
                    ) / 0xFFFFFFFF
                    is_idle = formatter.is_idle_label(frames[current].action)
                    if (
                        is_idle
                        and not thought_ids
                        and bucket >= idle_keep_fraction
                    ):
                        stats["n_idle_decisions_dropped"] += 1
                        w_seq += 1
                        continue
                    goal_block = goal_memory_block(goal_text, "")
                    user_content = [_text(goal_block), *(_image(frames[i].image) for i in ctx)]
                    messages: list[dict[str, Any]] = []
                    if system_prompt:
                        messages.append({"role": "system", "content": [_text(system_prompt)]})
                    messages.append({"role": "user", "content": user_content})
                    messages.append({"role": "assistant", "content": [_text(target)]})
                    rows.append({
                        "conversation_id": (
                            f"{day.day_tag}_g{gid_s}_r{plan['run_index']:02d}_d{w_seq:05d}"
                        ),
                        "day_tag": day.day_tag,
                        "chunk_index": chunk_of[current],
                        "recording_id": frames[current].recording_id,
                        "segment_ids": sorted({frames[i].segment_id for i in ctx}),
                        "t_start": fmt_t(frames[ctx[0]].t_day_s),
                        "t_end": fmt_t(frames[current].t_day_s),
                        "target_fps": fps,
                        "window_frames": window_frames,
                        "context_images": context_images,
                        "action_format": action_format,
                        "goal_conditioned": True,
                        "annotation_method": annotation_method,
                        "goal_id": gid,
                        "goal_text": goal_text,
                        "run_index": plan["run_index"],
                        "terminate": None,
                        "has_memory": False,
                        "n_frames": len(ctx),
                        "n_turns": 1,
                        "n_thoughts": len(thought_ids),
                        "thought_ids": thought_ids,
                        "n_non_noop": int(not formatter.is_idle_label(frames[current].action)),
                        "messages": messages,
                    })
                    stats["n_thoughts_placed"] += len(thought_ids)
                    w_seq += 1

                # Completion is a separate fresh decision whose final image is
                # the verified/clean outcome frame.
                if plan["terminate_text"] is not None and piece[-1] == idxs[-1]:
                    outcome_idx = plan["outcome_idx"]
                    prior = piece[max(0, len(piece) - context_images + 1):]
                    ctx_images = [frames[i].image for i in prior] + [frames[outcome_idx].image]
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": [_text(system_prompt)]})
                    messages.append({
                        "role": "user",
                        "content": [_text(goal_memory_block(goal_text, "")),
                                    *(_image(image) for image in ctx_images[-context_images:])],
                    })
                    messages.append({
                        "role": "assistant",
                        "content": [_text(plan["terminate_text"])],
                    })
                    rows.append({
                        "conversation_id": (
                            f"{day.day_tag}_g{gid_s}_r{plan['run_index']:02d}_term"
                        ),
                        "day_tag": day.day_tag,
                        "chunk_index": chunk_of[piece[-1]],
                        "recording_id": frames[piece[-1]].recording_id,
                        "segment_ids": sorted({frames[i].segment_id for i in prior}
                                              | {frames[outcome_idx].segment_id}),
                        "t_start": fmt_t(frames[prior[0]].t_day_s),
                        "t_end": fmt_t(frames[outcome_idx].t_day_s),
                        "target_fps": fps,
                        "window_frames": window_frames,
                        "context_images": context_images,
                        "action_format": action_format,
                        "goal_conditioned": True,
                        "annotation_method": annotation_method,
                        "goal_id": gid,
                        "goal_text": goal_text,
                        "run_index": plan["run_index"],
                        "terminate": terminate_mode,
                        "has_memory": False,
                        "n_frames": min(context_images, len(ctx_images)),
                        "n_turns": 1,
                        "n_thoughts": 0,
                        "thought_ids": [],
                        "n_non_noop": 1,
                        "messages": messages,
                    })
                    stats["n_terminate_turns"] += 1
                continue

            for k in range(0, len(piece), window_frames):
                win = piece[k: k + window_frames]
                is_final = win[-1] == idxs[-1]
                terminate_text = plan["terminate_text"] if is_final else None
                outcome_idx = plan["outcome_idx"] if is_final else None
                if len(win) < 2 and terminate_text is None:
                    stats["n_windows_skipped_short"] += 1
                    continue

                # Leak-free "So far:" memory. The FIRST window of a piece (the
                # span's own start, or a chunk start after a recording gap) has
                # no in-span in-chunk predecessor clip, so it withholds memory —
                # bare "GOAL: ..." conditioning, the in-distribution shape of an
                # inference episode start. Every later window carries the
                # memory_out of the clip ending exactly at win[0]-1.
                if omit_goal_memory or k == 0:
                    memory, mem_status = "", "piece_start"
                else:
                    memory, mem_status = select_memory(memory_rows, win[0])
                    if mem_status == "gap":
                        stats["n_windows_memory_omitted_boundary"] += 1
                block = goal_memory_block(goal_text, memory)
                if memory:
                    stats["n_windows_with_memory"] += 1

                messages: list[dict[str, Any]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": [_text(system_prompt)]})
                thought_ids: list[str] = []
                at_chunk_start = win[0] == chunk_first[chunk_of[win[0]]]
                for pos, i in enumerate(win):
                    content: list[dict[str, Any]] = []
                    if pos == 0:
                        content.append(_text(block))
                    content.append(_image(frames[i].image))
                    messages.append({"role": "user", "content": content})
                    text = frames[i].action
                    nm = nm_thoughts.get((gid, i))
                    if nm is not None:
                        # deliberate negative from the boundaries pass — replaces
                        # any annotation thought, exempt from the evidence-
                        # boundary demotion
                        text = f"<think>\n{nm}\n</think>\n{text}"
                        thought_ids.append(f"near_miss:g{gid_s}")
                        stats["n_near_miss_attached"] += 1
                    else:
                        th = anchored.get(i)
                        if th is not None:
                            if not at_chunk_start and pos < min_anchor_lead:
                                stats["n_demoted"] += 1
                            else:
                                text = f"<think>\n{th['instruction']}\n</think>\n{text}"
                                thought_ids.append(str(th["goal_id"]))
                    messages.append({"role": "assistant", "content": [_text(text)]})

                if terminate_text is not None:
                    messages.append({"role": "user",
                                     "content": [_image(frames[outcome_idx].image)]})
                    messages.append({"role": "assistant", "content": [_text(terminate_text)]})
                    stats["n_terminate_turns"] += 1

                turn_idxs = [*win, outcome_idx] if terminate_text is not None else win
                n_frames = len(turn_idxs)
                stats["n_thoughts_placed"] += len(thought_ids)
                rows.append({
                    "conversation_id": (f"{day.day_tag}_g{gid_s}_r{plan['run_index']:02d}"
                                        f"_w{w_seq:03d}"),
                    "day_tag": day.day_tag,
                    "chunk_index": chunk_of[win[0]],
                    "recording_id": frames[win[0]].recording_id,
                    "segment_ids": sorted({frames[i].segment_id for i in turn_idxs}),
                    "t_start": fmt_t(frames[turn_idxs[0]].t_day_s),
                    "t_end": fmt_t(frames[turn_idxs[-1]].t_day_s),
                    "target_fps": fps,
                    "window_frames": window_frames,
                    "action_format": action_format,
                    "goal_conditioned": True,
                    "annotation_method": annotation_method,
                    "goal_id": gid,
                    "goal_text": goal_text,
                    "run_index": plan["run_index"],
                    "terminate": terminate_mode if terminate_text is not None else None,
                    "has_memory": "\nSo far: " in block,
                    "n_frames": n_frames,
                    "n_turns": n_frames,
                    "n_thoughts": len(thought_ids),
                    "thought_ids": thought_ids,
                    "n_non_noop": sum(
                        1 for i in win if not formatter.is_idle_label(frames[i].action)
                    ),
                    "messages": messages,
                })
                w_seq += 1

    stats["n_windows"] = len(rows)
    stats["n_frames"] = sum(r["n_frames"] for r in rows)
    return rows, stats


# ---------------------------------------------------------------------------
# Legacy mode: goal-free thinking windows (byte-identical to the pre-goal
# builder; the regression gate in tests/test_stage04_goal_windows.py)
# ---------------------------------------------------------------------------


def build_legacy_day_rows(
    day: DayStream,
    anchored: dict[int, dict[str, Any]],
    *,
    window_frames: int,
    context_thoughts: int,
    min_anchor_lead: int,
    thinking_only: bool,
    system_prompt: str | None,
    terminal_token: str | None,
    fps: float,
) -> tuple[list[dict[str, Any]], Counter]:
    """The original goal-free window builder: tile every chunk, keep windows
    with >= 1 verified thought (``thinking_only``), earlier-thoughts context
    block, ``terminal_token`` glued to every window's final action."""
    stats: Counter = Counter()
    rows: list[dict[str, Any]] = []
    wf = window_frames
    for ci, chunk in enumerate(day.chunks):
        chunk_thoughts = [anchored[fr.day_idx] | {"day_idx": fr.day_idx}
                          for fr in chunk if fr.day_idx in anchored]
        for w0 in range(0, len(chunk), wf):
            win = chunk[w0: w0 + wf]
            if len(win) < 2:
                continue
            win_ids = {fr.day_idx for fr in win}
            wt = [t for t in chunk_thoughts if t["day_idx"] in win_ids]
            if thinking_only and not wt:
                continue
            # evidence boundary: anchors too close after a mid-chunk cut lose
            # their <think> (the audit's evidence sits in the previous window);
            # chunk-start windows are exempt.
            usable: dict[int, dict[str, Any]] = {}
            for t in wt:
                pos = t["day_idx"] - win[0].day_idx
                if w0 > 0 and pos < min_anchor_lead:
                    stats["n_demoted"] += 1
                    continue
                usable[t["day_idx"]] = t
            if thinking_only and not usable:
                continue
            prior = [t for t in chunk_thoughts if t["day_idx"] < win[0].day_idx]
            block = context_block(prior, context_thoughts)

            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": [_text(system_prompt)]})
            thought_ids: list[str] = []
            last = len(win) - 1
            for i, fr in enumerate(win):
                content: list[dict[str, Any]] = []
                if i == 0 and block:
                    content.append(_text(block))
                content.append(_image(fr.image))
                messages.append({"role": "user", "content": content})
                text = fr.action
                th = usable.get(fr.day_idx)
                if th is not None:
                    text = f"<think>\n{th['instruction']}\n</think>\n{text}"
                    thought_ids.append(str(th["goal_id"]))
                if i == last and terminal_token:
                    text = f"{text}\n{terminal_token}"
                messages.append({"role": "assistant", "content": [_text(text)]})
            stats["n_thoughts_placed"] += len(thought_ids)
            rows.append({
                "conversation_id": f"{day.day_tag}_c{ci:02d}_w{w0 // wf:03d}",
                "day_tag": day.day_tag,
                "chunk_index": ci,
                "recording_id": win[0].recording_id,
                "segment_ids": sorted({fr.segment_id for fr in win}),
                "t_start": fmt_t(win[0].t_day_s),
                "t_end": fmt_t(win[-1].t_day_s),
                "target_fps": fps,
                "window_frames": wf,
                "goal_conditioned": False,
                "annotation_method": "lumine_thinking",
                "n_frames": len(win),
                "n_turns": len(win),
                "n_thoughts": len(thought_ids),
                "n_context_thoughts": min(len(prior), context_thoughts),
                "thought_ids": thought_ids,
                "n_non_noop": sum(1 for fr in win if fr.action != "NO_OP"),
                "messages": messages,
            })
    stats["n_windows"] = len(rows)
    stats["n_frames"] = sum(r["n_frames"] for r in rows)
    return rows, stats


def build_day_windows(task: dict[str, Any]) -> dict[str, Any]:
    """Worker: one day -> conversation rows. Failures captured, never raised."""
    day_row = task["day_row"]
    day_tag = str(day_row["day_tag"])
    try:
        day: DayStream = build_day_stream(
            day_row,
            _WORKER["art"],
            fps=task["decision_fps"],
            fps_mode="nearest" if task["decision_fps"] != task["fps"] else "exact",
            gap_cut_s=task["gap_cut_s"], action_format=task["action_format"],
            continuous_action_hz=task["continuous_action_hz"],
        )
        if not day.frames:
            return {"day_tag": day_tag, "status": "empty_day", "rows": [], "stats": {}}
        active = task["active"]
        if task["decision_fps"] != task["fps"]:
            if not (task["goal_conditioned"] and task["context_images"] > 0
                    and task["omit_goal_memory"]):
                raise ValueError(
                    "--decision-fps differing from --fps requires fresh goal decision "
                    "records (--context-images > 0 --omit-goal-memory)"
                )
            annotation_day = build_day_stream(
                day_row,
                _WORKER["art"],
                fps=task["fps"],
                fps_mode="exact",
                gap_cut_s=task["gap_cut_s"],
                action_format="canonical",
            )
            active = reindex_active_rows(
                active,
                annotation_day,
                day,
                annotation_fps=task["fps"],
            )

        # Thought anchors use persistent master coordinates, so they remain
        # exact when the decision stream is denser than annotation.
        by_coord = {(fr.segment_id, fr.master_idx): fr.day_idx for fr in day.frames}
        anchored: dict[int, dict[str, Any]] = {}
        n_unmapped = 0
        for g in task["goals"]:
            di = by_coord.get((str(g["segment_id"]), int(g["start_master_idx"])))
            if di is None:
                n_unmapped += 1
                continue
            anchored[di] = g

        if task["goal_conditioned"]:
            rows, stats = build_goal_day_rows(
                day, active, task["memory"],
                {r["goal_id"]: r for r in task["boundaries"]}, anchored,
                window_frames=task["window_frames"],
                terminate_mode=task["terminate_boundaries"],
                terminate_max_lag_s=task["terminate_max_lag_s"],
                min_anchor_lead=task["min_anchor_lead"],
                system_prompt=task["system_prompt"],
                fps=task["decision_fps"],
                action_format=task["action_format"],
                annotation_method=task["annotation_method"],
                context_images=task["context_images"],
                omit_goal_memory=task["omit_goal_memory"],
                idle_keep_fraction=task["idle_keep_fraction"],
            )
            status = "ok" if rows else "no_goal_spans"
        else:
            rows, stats = build_legacy_day_rows(
                day, anchored,
                window_frames=task["window_frames"],
                context_thoughts=task["context_thoughts"],
                min_anchor_lead=task["min_anchor_lead"],
                thinking_only=task["thinking_only"],
                system_prompt=task["system_prompt"],
                terminal_token=resolve_terminal_token(
                    task["terminal_token"], task["action_format"]),
                fps=task["fps"],
            )
            status = "ok"
        stats["n_unmapped"] = n_unmapped
        return {"day_tag": day_tag, "status": status, "rows": rows,
                "stats": dict(stats)}
    except Exception as exc:
        return {"day_tag": day_tag, "status": "failed", "rows": [], "stats": {},
                "error": f"{exc}", "traceback": traceback.format_exc()}


def run_thinking(args: argparse.Namespace) -> None:
    art = FilterArtifact(args.filter_dir)

    if args.goals_dir is None:
        raise SystemExit("--mode thinking requires --goals-dir (a lumine_thinking / "
                         "lumine_thinking_goals stage-03b artifact)")
    gm_path = args.goals_dir / "manifest.json"
    if not gm_path.is_file():
        raise SystemExit(f"no manifest.json under {args.goals_dir}")
    gm = json.loads(gm_path.read_text())
    if gm.get("method") not in ("lumine_thinking", "lumine_thinking_goals"):
        raise SystemExit(f"--goals-dir is method {gm.get('method')!r}; --mode thinking "
                         "consumes lumine_thinking / lumine_thinking_goals artifacts only")
    goal_conditioned = gm.get("method") == "lumine_thinking_goals"
    assert_same_artifact(str(gm.get("master_store_id")), art.master_store_id,
                         what="master_store_id")
    assert_same_artifact(str(gm.get("filter_id")), art.filter_id, what="filter_id")
    if float(gm.get("fps") or 0) != args.fps:
        raise SystemExit(f"--fps {args.fps} != artifact annotation fps {gm.get('fps')} "
                         "(v1 locks training fps to annotation fps)")
    decision_fps = args.decision_fps or args.fps
    if decision_fps <= 0:
        raise SystemExit("--decision-fps must be > 0")
    if decision_fps != args.fps:
        if not (goal_conditioned and args.context_images > 0 and args.omit_goal_memory):
            raise SystemExit(
                "a denser --decision-fps requires goal-conditioned fresh records: "
                "--context-images > 0 --omit-goal-memory"
            )
        if args.terminate_boundaries == "verified":
            raise SystemExit(
                "denser --decision-fps currently supports clean/all termination; "
                "verified near-miss ranges remain annotation-indexed"
            )

    # Windows must be a positive multiple of the annotation clip stride so their
    # edges land on clip edges (the leak-free memory predecessor precondition).
    require_window_alignment(args.window_frames)
    goals_id = make_artifact_id(args.goals_dir)

    boundaries_dir = None
    if goal_conditioned and args.terminate_boundaries == "verified":
        if args.boundaries_dir is None:
            raise SystemExit("--terminate-boundaries verified needs --boundaries-dir "
                             "(a lumine_goal_boundaries artifact)")
        boundaries_dir = resolve_boundaries_dir(args.boundaries_dir)
        if not boundaries_dir.is_dir():
            raise SystemExit(f"no boundaries sidecars under {args.boundaries_dir}")

    goals = load_goals(args.goals_dir / "goals.jsonl")
    by_day: dict[str, list[dict[str, Any]]] = {}
    for g in goals:
        by_day.setdefault(str(g.get("day_tag") or ""), []).append(g)
    by_day.pop("", None)

    if args.no_system_prompt:
        system_prompt = None
    elif args.system_prompt_file is not None:
        system_prompt = args.system_prompt_file.read_text().strip()
    elif args.system_prompt is not None:
        system_prompt = args.system_prompt
    elif goal_conditioned:
        system_prompt = goal_system_prompt_file(args.action_format).read_text().strip()
    else:
        system_prompt = THINKING_SYSTEM_PROMPT

    day_rows = load_or_build_day_index(
        art, args.clips_manifest, day_index_cache=args.day_index_cache, tz=args.tz)

    if goal_conditioned:
        # goal mode trains on goal spans — restrict to days that HAVE a
        # goals_active sidecar (a day with goals but zero verified thoughts
        # still trains).
        units = args.goals_dir / "units"
        restrict = {d.name for d in units.iterdir()
                    if d.is_dir() and (d / "goals_active.jsonl").is_file()} if units.is_dir() else set()
    else:
        restrict = set(by_day)
    day_rows = select_day_rows(day_rows, day_filter=args.day_filter,
                               day_exclude=args.day_exclude, restrict_to=restrict)
    if args.limit is not None:
        day_rows = day_rows[: args.limit]
    if not day_rows:
        raise SystemExit("no days to process")

    def _mk_task(d: dict[str, Any]) -> dict[str, Any]:
        active, memory, bounds = [], [], []
        if goal_conditioned:
            active, memory = load_goal_memory_sidecars(args.goals_dir, d["day_tag"])
            bounds = list(load_boundaries(boundaries_dir, d["day_tag"]).values())
        return {
            "day_row": d,
            "goals": by_day.get(d["day_tag"], []),
            "fps": args.fps,
            "decision_fps": args.decision_fps or args.fps,
            "gap_cut_s": args.gap_cut_s,
            "window_frames": args.window_frames,
            "action_format": args.action_format,
            "continuous_action_hz": (
                0.0 if args.action_format == "computer_use_rel_step_v1"
                else args.continuous_action_hz
            ),
            "context_thoughts": args.context_thoughts,
            "min_anchor_lead": args.min_anchor_lead,
            "thinking_only": bool(args.thinking_only),
            "system_prompt": system_prompt,
            "terminal_token": args.terminal_token,
            "goal_conditioned": goal_conditioned,
            "annotation_method": gm.get("method"),
            "context_images": args.context_images,
            "omit_goal_memory": bool(args.omit_goal_memory),
            "idle_keep_fraction": args.idle_keep_fraction,
            "terminate_boundaries": args.terminate_boundaries,
            "terminate_max_lag_s": args.terminate_max_lag_s,
            "active": active,
            "memory": memory,
            "boundaries": bounds,
        }

    tasks = [_mk_task(d) for d in day_rows]

    n_workers = max(1, min(args.num_workers, len(tasks)))
    mode = "goal_windows" if goal_conditioned else "thinking_windows"
    print(f"[conversations] mode=thinking ({mode}) | {len(tasks)} days, "
          f"{sum(len(t['goals']) for t in tasks):,} thoughts | window={args.window_frames}f "
          f"(stride {CLIP_STRIDE}) annotation_fps={args.fps} "
          f"decision_fps={args.decision_fps or args.fps} "
          f"format={args.action_format}"
          + (f" terminate={args.terminate_boundaries}" if goal_conditioned else "")
          + f" | workers={n_workers}", flush=True)

    records: list[dict[str, Any]] = []
    counts: Counter = Counter()
    totals: Counter = Counter()
    with mp.Pool(n_workers, initializer=_init_worker,
                 initargs=(str(args.filter_dir),)) as pool:
        for i, res in enumerate(pool.imap_unordered(build_day_windows, tasks, chunksize=1), 1):
            counts[res["status"]] += 1
            if res["status"] == "failed":
                print(f"  FAIL {res['day_tag']}: {res.get('error')}", flush=True)
            records.extend(res["rows"])
            for k, v in (res.get("stats") or {}).items():
                totals[k] += int(v)
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} days | {len(records)} windows", flush=True)

    if not records:
        raise SystemExit("no windows built")

    summary = {
        "mode": mode,
        "goal_conditioned": goal_conditioned,
        "n_conversations": len(records),
        "n_windows": len(records),
        "n_frames_total": sum(r["n_frames"] for r in records),
        "n_frames": sum(r["n_frames"] for r in records),
        "n_thoughts": totals["n_thoughts_placed"],
        "n_thoughts_placed": totals["n_thoughts_placed"],
        "n_thoughts_demoted_boundary": totals["n_demoted"],
        "n_anchors_unmapped": totals["n_unmapped"],
        "status_counts": dict(counts),
        "fps": args.decision_fps or args.fps,
        "annotation_fps": args.fps,
        "decision_fps": args.decision_fps or args.fps,
        "window_frames": args.window_frames,
        "clip_stride": CLIP_STRIDE,
        "action_format": args.action_format,
        "continuous_action_hz": (
            0.0 if args.action_format == "computer_use_rel_step_v1"
            else args.continuous_action_hz
        ),
        "context_thoughts": args.context_thoughts,
        "context_images": args.context_images,
        "omit_goal_memory": bool(args.omit_goal_memory),
        "idle_keep_fraction": args.idle_keep_fraction,
        "n_idle_decisions_dropped": totals["n_idle_decisions_dropped"],
        "min_anchor_lead": args.min_anchor_lead,
        "thinking_only": bool(args.thinking_only),
        "has_system_prompt": system_prompt is not None,
        "terminal_token": args.terminal_token,
        "gap_cut_s": args.gap_cut_s,
        "tz": args.tz,
        "filter_dir": str(art.dir),
        "goals_dir": str(args.goals_dir),
    }
    if goal_conditioned:
        summary.update({
            "terminate_boundaries": args.terminate_boundaries,
            "terminate_max_lag_s": args.terminate_max_lag_s,
            "boundaries_dir": str(boundaries_dir) if boundaries_dir else None,
            "n_goal_spans": totals["n_spans"],
            "n_terminate": totals["n_terminate_turns"],
            "n_terminate_turns": totals["n_terminate_turns"],
            "terminate_skips": {k.removeprefix("n_terminate_skipped_"): v
                                for k, v in sorted(totals.items())
                                if k.startswith("n_terminate_skipped_")},
            "n_windows_with_memory": totals["n_windows_with_memory"],
            "n_windows_memory_omitted_boundary": totals["n_windows_memory_omitted_boundary"],
            "n_near_miss_attached": totals["n_near_miss_attached"],
            "n_near_miss_unplaced": totals["n_near_miss_unplaced"],
            "n_frames_dropped_post_goal": totals["n_frames_dropped_post_goal"],
            "n_no_goal_frames_excluded": totals["n_no_goal_frames_excluded"],
            "n_windows_skipped_short": totals["n_windows_skipped_short"],
        })
    out_dir = write_conversation_artifact(
        args.output_dir, records, summary,
        master_store_id=art.master_store_id, filter_id=art.filter_id, goals_id=goals_id)
    tail = (f"{totals['n_terminate_turns']} terminate, "
            f"{totals['n_windows_with_memory']} with-memory "
            f"({totals['n_windows_memory_omitted_boundary']} memory omitted at boundary), "
            if goal_conditioned else "")
    print(f"[conversations] {len(records)} windows, {totals['n_thoughts_placed']:,} thoughts placed "
          f"({totals['n_demoted']} demoted, {totals['n_unmapped']} unmapped), {tail}"
          f"-> {out_dir}", flush=True)


# ===========================================================================
# CLI
# ===========================================================================


def parse_args() -> argparse.Namespace:
    normalize_dashed_argv()  # accept pmanager's --foo_bar=value arg form
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("action", "thinking"), required=True,
                   help="'action': per-segment fps-selected windows (the historical "
                        "build_conversations). 'thinking': day-stream windows from a "
                        "lumine_thinking(-goals) 03b artifact (goal/memory/terminate).")

    # ---- shared ----
    p.add_argument("--filter-dir", type=Path, required=True,
                   help="A stage-03 (filter) --output-dir: manifest.json + "
                        "filter_index.jsonl + filter/<seg>.json.")
    p.add_argument("--clips-manifest", type=Path, required=True,
                   help="Stage-00/02 realigned clips_manifest.jsonl (day grouping; used "
                        "for --day-filter/--day-exclude, and always in thinking mode).")
    p.add_argument("--day-index-cache", type=Path, required=True,
                   help="JSON cache of the mvhd day index (written on a miss; reused when "
                        "filter_id + tz match). Skips the ~minutes probe on repeat runs.")
    p.add_argument("--fps", type=float, default=0.5,
                   help="Training frame rate. Action mode: any integer divisor of the "
                        "master fps (--fps-mode). Thinking mode: MUST equal the artifact's "
                        "annotation fps (v1 lock).")
    p.add_argument(
        "--decision-fps",
        type=float,
        default=None,
        help="Thinking fresh-decision mode only: optional denser observation/action "
             "rate. Goal evidence remains anchored at --fps and is projected by "
             "time. A differing value requires --context-images > 0 and "
             "--omit-goal-memory; 4 Hz uses causal nearest-tick sampling on the "
             "15 Hz master stream.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--day-filter", nargs="+", default=None, metavar="DAY",
                   help="Include only these day tags (mutually exclusive with --day-exclude).")
    p.add_argument("--day-exclude", nargs="+", default=None, metavar="DAY",
                   help="Exclude these day tags (mutually exclusive with --day-filter).")
    p.add_argument("--action-format", type=str, default=None, choices=sorted(FORMATTERS),
                   help="Assistant-turn action formatter (lib/action_format registry). "
                        "Default: 'canonical' in action mode, 'computer_use_rel_v1' in "
                        "thinking mode.")
    p.add_argument("--continuous-action-hz", type=float, default=DEFAULT_CONTINUOUS_ACTION_HZ,
                   help="ordered_events_* only: internal motor-grid rate (NOT a frame rate).")
    p.add_argument("--system-prompt", type=str, default=None, help="System message text.")
    p.add_argument("--system-prompt-file", type=Path, default=None,
                   help="Read the system message from a file (wins over --system-prompt). "
                        "Thinking goal mode defaults per action-format "
                        "(cua_v4_thinking.txt for computer_use_rel_v1, else cua_v3_thinking.txt).")
    p.add_argument("--no-system-prompt", action="store_true", help="Emit no system message.")
    p.add_argument("--goals-dir", type=Path, default=None,
                   help="Action mode: a stage-03b goals artifact (goals.jsonl master "
                        "intervals) -> one conversation per projected goal. Thinking mode "
                        "(REQUIRED): a lumine_thinking(-goals) artifact -> goal-conditioned "
                        "when method is lumine_thinking_goals.")
    p.add_argument("--tz", default=DEFAULT_TZ, help="Day-grouping timezone.")
    p.add_argument("--num-workers", type=int, default=0, help="0 = cpu_count().")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N units (segments in action mode, days in "
                        "thinking mode) — debug/smoke.")

    # ---- action mode ----
    a = p.add_argument_group("action mode")
    a.add_argument("--fps-mode", choices=FPS_MODES, default="exact",
                   help="'exact' (default): fps must divide the master fps. 'nearest': any "
                        "fps <= master; each slot takes the nearest master tick.")
    a.add_argument("--instruction", type=str, default=None,
                   help="Goal-free: fixed instruction on each segment's first user turn.")
    a.add_argument("--instruction-field", type=str, default=None,
                   help="Goal-free: read a per-segment instruction from this key on the "
                        "filter index row (then the segment filter doc).")
    a.add_argument("--use-plans", nargs="?", const=True, type=str2bool, default=False,
                   metavar="BOOL",
                   help="Goal mode: prefix the goal's plan prose to the first assistant "
                        f"turn (plans flagged {sorted(DROP_PLAN_FLAGS)} fall back).")
    a.add_argument("--include-variants", nargs="?", const=True, type=str2bool, default=False,
                   metavar="BOOL",
                   help="Goal mode: one conversation per instruction phrasing (primary + "
                        "instruction_variants), sharing frames/actions/plan.")
    a.add_argument("--min-frames", type=int, default=1,
                   help="Skip segments (goal-free) / reject goals (goal mode) with fewer frames.")
    a.add_argument("--snap-start", choices=SNAP_START_MODES, default="before",
                   help="Goal projection: 'before' (default) includes the last selected "
                        "frame at-or-before the goal start; 'inside' keeps only interior frames.")
    a.add_argument("--dead-zone-flag-frac", type=float, default=0.05,
                   help="Flag a segment when more than this fraction of its keylog events "
                        "were discarded by the dead-zone policy (realignment health).")

    # ---- thinking mode ----
    t = p.add_argument_group("thinking mode")
    t.add_argument("--window-frames", type=int, default=None,
                   help="REQUIRED in thinking mode: frames per window. MUST be a positive "
                        f"multiple of the annotation clip stride ({CLIP_STRIDE}) so window "
                        "edges land on clip edges (leak-free 'So far:' memory).")
    t.add_argument("--gap-cut-s", type=float, default=DEFAULT_GAP_CUT_S,
                   help="A recording gap > this splits the day into chunks.")
    t.add_argument("--terminate-boundaries", choices=TERMINATE_BOUNDARY_MODES, default="clean",
                   help="Goal mode TERMINATE gating: 'clean' (default, structural "
                        "goal->goal handoff within --terminate-max-lag-s), 'verified' "
                        "(lumine_goal_boundaries completed+high only; needs "
                        "--boundaries-dir), 'all' (every span with an outcome frame).")
    t.add_argument("--boundaries-dir", type=Path, default=None,
                   help="A lumine_goal_boundaries artifact (or its boundaries/ dir); "
                        "required for --terminate-boundaries verified.")
    t.add_argument("--terminate-max-lag-s", type=float, default=DEFAULT_TERMINATE_MAX_LAG_S,
                   help="'clean' mode: max seconds between goal_t_end and the outcome frame.")
    t.add_argument("--min-anchor-lead", type=int, default=0,
                   help="A thought anchored closer than this after a mid-chunk window cut "
                        "loses its <think> (audit evidence sits in the previous window).")
    t.add_argument("--context-thoughts", type=int, default=8,
                   help="Legacy (goal-free) mode: earlier same-chunk thoughts on the first "
                        "user turn.")
    t.add_argument("--context-images", type=int, default=0,
                   help="Goal mode only: >0 emits one fresh decision record per target "
                        "with GOAL plus up to this many chronological screenshots and no "
                        "prior assistant messages. 0 keeps transcript windows.")
    t.add_argument("--omit-goal-memory", action="store_true",
                   help="Never emit annotation-generated 'So far:' memory. Required by "
                        "fresh relative-step recipes so train and rollout context match.")
    t.add_argument(
        "--idle-keep-fraction",
        type=float,
        default=1.0,
        help="Fresh decision records only: deterministic fraction of idle wait targets "
             "to retain (thought-bearing and terminate records are always retained).",
    )
    t.add_argument("--thinking-only", nargs="?", const=True, type=str2bool, default=True,
                   metavar="BOOL",
                   help="Legacy mode: emit only windows with >=1 thought (ignored in goal "
                        "mode — goal windows without thoughts still supervise actions).")
    t.add_argument("--terminal-token", type=str, default=None,
                   help="Action mode: append to the final assistant message (as "
                        "'<action>\\n<token>'). Legacy thinking mode: same (the literal "
                        "TERMINATE resolves through the formatter). Goal thinking mode "
                        "ignores this — TERMINATE is --terminate-boundaries policy.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    check_day_selection_args(args.day_filter, args.day_exclude)
    if args.action_format is None:
        args.action_format = "canonical" if args.mode == "action" else "computer_use_rel_v1"
    if args.mode == "thinking" and args.window_frames is None:
        raise SystemExit("--mode thinking requires --window-frames "
                         f"(a positive multiple of the clip stride {CLIP_STRIDE})")
    if args.mode == "action":
        run_action(args)
    else:
        run_thinking(args)


if __name__ == "__main__":
    main()
