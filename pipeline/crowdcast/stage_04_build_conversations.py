#!/usr/bin/env python3
"""Stage 04 (conversations): the single injection point — join frames (stage
01, via the stage-03 filter mask), actions (stage 02's realigned keylogs), and
optionally goals (stage 03b), and emit training conversations.

Everything trainable is decided here, so ablations are flags, not pipeline
reruns:
  * frame rate     --fps X (any integer divisor of the master fps; the shared
                   selector in lib/views picks frames within filter survivors)
  * action format  --action-format (lib/action_format registry; ``canonical``
                   is byte-identical to the historical format on dead-zone-free
                   stretches; ``ordered_events_v2`` renders each window as an
                   ordered mini-program on a --continuous-action-hz motor grid;
                   ``ordered_events_v3`` is that plus ``type("...")`` for
                   balanced typing runs)
  * goals          --goals-dir (a stage-03b artifact; goals are half-open
                   master-tick intervals, projected onto the actual selected
                   frames — one conversation per goal, instruction on the
                   first user turn)
  * plan prose     --use-plans (goal's ``plan`` prefixes the first assistant
                   turn as ``<plan>\\n<action>``; unusable plan_flags fall back
                   to a plan-less first turn)
  * terminal token --terminal-token (appended to the final assistant message
                   as ``<action>\\n<token>`` — never a standalone turn, which
                   would train an out-of-distribution state)

The system prompt is not a flag. It is ``grammars.describe()`` of the grammar the
selected formatter renders its labels in, so the prompt trained against a label
is the prompt the eval harness and the RL rollout put in front of the model. Its
sha256 is recorded in the manifest as ``system_prompt_sha256``, and the grammar's
conformance vectors pin the same digest, so neither side can move alone.

Dead-zone accounting: every segment row carries the label-policy counters
(discarded deltas, clamped/dropped pairs); segments whose discard fraction
exceeds --dead-zone-flag-frac are flagged (``dead_zone_flagged``) — a
realignment health signal.

The conversation shape is the canonical chat.jsonl schema: content blocks,
instruction text before the image on the first user turn, one assistant turn
per frame. Stages 05/06 consume the output directly.

Output (--output-dir):
  conversations.jsonl          one row per conversation {messages, provenance}.
  chat.jsonl                   the same rows — the split-agnostic drop-in
                               source_path for stages 05 (measure) and 06
                               (records; split applied there by recording_id).
  conversations_summary.json   aggregate stats (incl. dead-zone counters and
                               goal-projection rejections).
  manifest.json                artifact marker; carries master_store_id +
                               filter_id (+ goals_id) so joins are auditable.

Run::

    cd <repo root>
    uv run python pipeline/stage_04_build_conversations.py \
        --filter-dir <stage-03 --output-dir> --fps 1 --output-dir <dest> \
        [--goals-dir <stage-03b --output-dir> --use-plans --include-variants] \
        [--terminal-token '<terminate>']
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Make the ``pipeline`` and ``grammars`` packages importable when run directly
# (mirrors the other stages).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import grammars  # noqa: E402
import prompts  # noqa: E402

from pipeline.crowdcast.lib.action_format import (  # noqa: E402
    DEFAULT_CONTINUOUS_ACTION_HZ,
    FORMATTERS,
    get_formatter,
)
from pipeline.crowdcast.lib.app_filter import (  # noqa: E402
    APP_UNKNOWN_MODES,
    AppFilter,
    AppSpan,
    app_stats,
    label_view_frames,
    plan_app_spans,
)
from pipeline.crowdcast.lib.common import (  # noqa: E402
    ensure_dir,
    normalize_dashed_argv,
    str2bool,
    write_json,
    write_jsonl,
)
from pipeline.crowdcast.lib.events import EventStats, load_events  # noqa: E402
from pipeline.crowdcast.lib.goals import (  # noqa: E402
    SNAP_START_MODES,
    assert_same_artifact,
    goals_by_segment,
    load_goals,
    project_goals,
)
from pipeline.crowdcast.lib.manifest import make_artifact_id  # noqa: E402
from pipeline.crowdcast.lib.views import (  # noqa: E402
    FPS_MODES,
    FilterArtifact,
    build_segment_view,
)

# Plans carrying these quality flags are unusable as training prose; the
# conversation falls back to a plan-less first turn (same as the fold
# pipeline's assemble step).
DROP_PLAN_FLAGS = frozenset({"empty", "restates_instruction"})


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _image_block(image: str) -> dict[str, Any]:
    return {"type": "image", "image": image}


def build_messages(
    turns: list[tuple[str, str]],  # ordered (image, action) pairs
    *,
    instruction: str | None,
    system_prompt: str,
    plan: str | None = None,
    terminal_token: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble one interleaved conversation. Canonical chat.jsonl schema:
    instruction text before the image on the first user turn, image-only on
    later turns, one assistant turn per frame carrying its action. The plan
    prefixes the first assistant turn (``<plan>\\n<action>``); the terminal
    token rides at the end of the final assistant message."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [_text_block(system_prompt)]}
    ]
    last = len(turns) - 1
    for idx, (image, action) in enumerate(turns):
        content: list[dict[str, Any]] = []
        if idx == 0 and instruction:
            content.append(_text_block(instruction))
        content.append(_image_block(image))
        messages.append({"role": "user", "content": content})
        text = action
        if idx == 0 and plan:
            text = f"{plan}\n{text}"
        if idx == last and terminal_token:
            text = f"{text}\n{terminal_token}"
        messages.append({"role": "assistant", "content": [_text_block(text)]})
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
        events, event_stats = load_events(keylog) if keylog else ([], EventStats())
        fmt = get_formatter(
            task["action_format"],
            continuous_action_hz=task["continuous_action_hz"],
            jitter_deadband_px=task.get("jitter_deadband_px", 0),
        )
        result = fmt.format_segment(
            events, view.windows(), view.dead_zones, master_fps=view.master_fps
        )
        counters = result.counters
        # Resolved once per segment: the filter config is plain data on the task
        # (a frozen dataclass does not survive the pool's pickling as itself), and
        # the per-frame app labels are derived only when something reads them.
        app_cfg = AppFilter(**task["app_filter"])
        app_frames = label_view_frames(view) if app_cfg.active else []
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
            all_turns = [(str(f.image), result.labels[f.view_idx]) for f in view.frames]
            instruction = _resolve_instruction(
                task["index_row"],
                filter_seg,
                instruction=task["instruction"],
                instruction_field=task["instruction_field"],
            )
            spans = [AppSpan(None, 0, len(all_turns))]
            if app_cfg.active:
                spans = [
                    sp
                    for sp in plan_app_spans(
                        app_frames, app_cfg, min_frames=task["min_frames"]
                    )
                ]
                if not spans:
                    return {**base, "status": "app_filtered", "rows": []}
            for span_idx, span in enumerate(spans):
                turns = all_turns[span.lo : span.hi]
                if len(turns) < task["min_frames"]:
                    continue
                messages = build_messages(
                    turns,
                    instruction=instruction,
                    system_prompt=task["system_prompt"],
                    terminal_token=task["terminal_token"],
                )
                # A split segment yields several conversations, so the id has to
                # carry the span. An unsplit run keeps the bare segment id, so
                # every existing artifact's ids are unchanged.
                conv_id = seg if len(spans) == 1 and span.lo == 0 else f"{seg}#a{span_idx}"
                rows.append({
                    "conversation_id": conv_id,
                    **common,
                    "instruction": instruction,
                    "goal_conditioned": instruction is not None,
                    "app": span.app,
                    "app_seam_trimmed": span.seam_trimmed,
                    "n_frames": len(turns),
                    "n_turns": len(turns),
                    "n_non_noop": sum(1 for _, a in turns if a != "NO_OP"),
                    "messages": messages,
                })
            return {
                **base,
                "status": "ok" if rows else "app_filtered",
                "rows": rows,
                "primitive_counts": result.primitive_counts,
                "parse_counters": asdict(event_stats),
            }

        seg_goals = task["goals_by_segment"].get(seg, [])
        if not seg_goals:
            return {**base, "status": "no_goals", "rows": []}
        projections, proj_stats = project_goals(
            seg_goals, view, snap_start=task["snap_start"], min_frames=task["min_frames"]
        )
        n_app_rejected = 0
        for proj in projections:
            goal = proj.goal
            # The gate reads the goal's OWN frames, not the segment's: a goal is
            # the unit being trained here, and a segment-wide verdict would admit
            # a terminal goal inside a mostly-browser recording.
            if app_cfg.active:
                goal_apps = [app_frames[f.view_idx] for f in proj.frames]
                stats = app_stats(goal_apps)
                if not app_cfg.accepts(stats["app"], float(stats["app_frac"])):
                    n_app_rejected += 1
                    continue
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
            "parse_counters": asdict(event_stats),
            "projection": {
                "n_goals": proj_stats.n_goals,
                "n_projected": proj_stats.n_projected,
                "n_empty_projection": proj_stats.n_empty_projection,
                "n_too_few_frames": proj_stats.n_too_few_frames,
                "n_snapped": proj_stats.n_snapped,
                "n_app_rejected": n_app_rejected,
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


#: Cap for the affinity-mask fallback, so an interactive run on a big login node
#: stays neighbourly. A Slurm allocation is never capped — it was granted.
AUTO_WORKER_CAP = 32


def resolve_workers(requested: int) -> int:
    """``--num-workers 0`` (the default) means "the CPUs this run was given".

    ``SLURM_CPUS_PER_TASK`` wins wherever it is set. `mp.cpu_count()` reports the
    MACHINE, not the allocation, so on a shared node it forks a pool sized to
    someone else's cores; and the affinity mask is no better, because a Slurm
    allocation's mask includes the SMT siblings of the allocated cores — `-c 16`
    reports 32, so the mask silently doubles the pool. The mask is the fallback
    for interactive runs only, capped by ``AUTO_WORKER_CAP``.
    """
    if requested > 0:
        return requested
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit() and int(slurm) > 0:
        return int(slurm)
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        available = os.cpu_count() or 1
    return max(1, min(available, AUTO_WORKER_CAP))


def resolve_system_prompt(prompt_id: str | None, grammar: str) -> tuple[str, str | None]:
    """This run's system prompt, and the prompt id recorded beside it.

    Unset is the default and means the grammar's own ``describe()`` — the prompt
    the eval harness and the RL rollout put in front of the model with no
    justification needed. A prompt id selects a registered edit of that same
    spec (`prompts/`), for the framing the grammar does not own: whether a goal
    is stated, whether the corpus contains idle turns.

    The grammar check is the guard that matters. A prompt composed over another
    grammar would describe a syntax these labels are not written in, and nothing
    downstream could tell: the labels would parse, the digests would differ from
    nothing in particular, and the model would be trained to answer in a format
    it was told not to use. Refusing here is the only place that is cheap.
    """
    if prompt_id is None:
        return grammars.describe(grammar), None
    named = prompts.get(prompt_id)
    if named.grammar != grammar:
        raise SystemExit(
            f"--system-prompt-id {prompt_id!r} is composed over grammar "
            f"{named.grammar!r}, but --action-format renders {grammar!r}. The "
            "prompt would describe a syntax these labels are not written in."
        )
    return named.describe(), prompt_id


def parse_args() -> argparse.Namespace:
    normalize_dashed_argv()  # accept pmanager's --foo_bar=value arg form
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--filter-dir", type=Path, required=True,
                   help="A stage-03 (filter) --output-dir: manifest.json + filter_index.jsonl "
                        "+ filter/<seg>.json.")
    p.add_argument("--fps", type=float, required=True,
                   help="Training frame rate. With --fps-mode exact, master_fps/fps must "
                        "be an integer.")
    p.add_argument("--fps-mode", choices=FPS_MODES, default="exact",
                   help="'exact' (default): fps must divide the master fps; even spacing. "
                        "'nearest': any fps <= master; each slot takes the master tick "
                        "nearest its ideal time (spacing jitters by up to half a tick, "
                        "e.g. 4 fps on a 15 fps master -> ticks 0,4,8,11,...).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--action-format", type=str, required=True,
                   choices=sorted(FORMATTERS),
                   help="Registered assistant-turn action formatter (lib/action_format). "
                        "Required: it picks the grammar that every label AND the system "
                        "prompt are rendered in, so a default would silently decide the "
                        "dataset's target format for a caller who forgot the flag.")
    p.add_argument("--continuous-action-hz", type=float,
                   default=DEFAULT_CONTINUOUS_ACTION_HZ,
                   help="ordered formats only: internal motor-grid rate for accumulating "
                        "move/scroll deltas within a window (NOT a frame rate; recorded as "
                        "null for formats that ignore it).")
    p.add_argument("--jitter-deadband-px", type=int, default=0,
                   help="ordered formats only: drop single-axis move/scroll deltas within "
                        "this many pixels after summing each move/scroll-only run — the "
                        "hand tremor a resting hand contributes while clicking, scrolling "
                        "or typing. A diagonal is never dropped, and a window is never "
                        "emptied. 0 (default) is off and cannot move an existing label.")
    p.add_argument("--system-prompt-id", type=str, default=None,
                   choices=sorted(prompts.names()),
                   help="A registered prompt (prompts/) composed over the action format's "
                        "grammar — e.g. goal-conditioned vs not. Must name the same grammar "
                        "the formatter renders. Default: the grammar's own describe(), "
                        "which is what the eval harness accepts without a justification.")
    p.add_argument("--goals-dir", type=Path, default=None,
                   help="A stage-03b annotation artifact (goals.jsonl in master intervals). "
                        "Sets goal mode: one conversation per projected goal. Its manifest's "
                        "master_store_id/filter_id must match --filter-dir's.")
    p.add_argument("--instruction", type=str, default=None,
                   help="Goal-free: fixed instruction on each segment's first user turn.")
    p.add_argument("--instruction-field", type=str, default=None,
                   help="Goal-free: read a per-segment instruction from this key on the "
                        "filter index row (then the segment filter doc).")
    p.add_argument("--use-plans", nargs="?", const=True, type=str2bool, default=False,
                   metavar="BOOL",
                   help="Goal mode: prefix the goal's plan prose to the first assistant turn "
                        f"(plans flagged {sorted(DROP_PLAN_FLAGS)} fall back to plan-less). "
                        "Bare flag or --use-plans=true (labctl's --key=value form).")
    p.add_argument("--include-variants", nargs="?", const=True, type=str2bool, default=False,
                   metavar="BOOL",
                   help="Goal mode: one conversation per instruction phrasing (primary + "
                        "instruction_variants), sharing frames/actions/plan. Bare flag or "
                        "--include-variants=true.")
    p.add_argument("--min-frames", type=int, default=1,
                   help="Skip segments (goal-free) / reject goals (goal mode) with fewer frames.")
    p.add_argument("--snap-start", choices=SNAP_START_MODES, default="before",
                   help="Goal projection: 'before' (default) includes the last selected frame "
                        "at-or-before the goal start — the observation its first action was "
                        "taken from; 'inside' keeps only frames within the interval.")
    p.add_argument("--terminal-token", type=str, default=None,
                   help="Append this token to the final assistant message "
                        "(as '<action>\\n<token>', never a standalone turn).")
    p.add_argument("--dead-zone-flag-frac", type=float, default=0.05,
                   help="Flag a segment when more than this fraction of its keylog events "
                        "were discarded by the dead-zone policy (realignment health).")
    g = p.add_argument_group(
        "foreground app",
        "Filter or split conversations by the application that had focus "
        "(lib/app_context, from the recorder's ContextChanged events).",
    )
    g.add_argument("--include-app", action="append", default=None, metavar="APP",
                   help="Keep only conversations whose dominant app is one of these. "
                        "Repeatable AND comma-separated (labctl renders every recipe "
                        "arg as one --key=value, so it cannot repeat a flag). An "
                        "unresolved app NEVER satisfies an include list.")
    g.add_argument("--exclude-app", action="append", default=None, metavar="APP",
                   help="Drop conversations whose dominant app is one of these.")
    g.add_argument("--app-min-frac", type=float, default=0.0,
                   help="Require the dominant app to hold at least this fraction of a "
                        "conversation's labelled frames (0 = no requirement).")
    g.add_argument("--app-unknown", choices=APP_UNKNOWN_MODES, default="keep",
                   help="'keep' (default) or 'drop' conversations with no resolvable "
                        "app — UNCAPTURED (the recorder's privacy blackout) or UNKNOWN.")
    g.add_argument("--split-by-app", nargs="?", const=True, type=str2bool, default=False,
                   metavar="BOOL",
                   help="Cut each segment at every app switch: one conversation per "
                        "application. Without it a conversation can teach the model to "
                        "switch apps mid-episode for a reason that was the "
                        "demonstrator's, not the screen's.")
    g.add_argument("--app-min-run-frames", type=int, default=1,
                   help="--split-by-app: drop a same-app run shorter than this.")
    g.add_argument("--app-drop-seam-turns", nargs="?", const=True, type=str2bool,
                   default=True, metavar="BOOL",
                   help="--split-by-app: drop the boundary turn whose action window "
                        "straddles the switch (default true). Its keystrokes belong "
                        "partly to each app, so kept it is the one turn in a split "
                        "conversation guaranteed to be mislabelled.")
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 (default) = SLURM_CPUS_PER_TASK, else the CPU affinity "
                        f"mask capped at {AUTO_WORKER_CAP}. Never cpu_count(), which "
                        "reports the machine rather than this run's allocation.")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N segments (debug).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

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

    # Fails fast on an unknown format / invalid hz / negative deadband, and
    # resolves the grammar whose codec renders this run's labels and prompt.
    formatter = get_formatter(
        args.action_format,
        continuous_action_hz=args.continuous_action_hz,
        jitter_deadband_px=args.jitter_deadband_px,
    )
    continuous_action_hz = getattr(formatter, "continuous_action_hz", None)
    system_prompt, system_prompt_id = resolve_system_prompt(
        args.system_prompt_id, formatter.grammar
    )
    system_prompt_sha256 = hashlib.sha256(system_prompt.encode()).hexdigest()

    app_filter = AppFilter.from_args(args)
    # frozenset is not JSON-able and the pool pickles the task dicts; sorted
    # tuples also make the manifest's record of the policy readable.
    app_filter_payload = {
        "include": frozenset(app_filter.include),
        "exclude": frozenset(app_filter.exclude),
        "min_frac": app_filter.min_frac,
        "unknown": app_filter.unknown,
        "split": app_filter.split,
        "min_run_frames": app_filter.min_run_frames,
        "drop_seam_turns": app_filter.drop_seam_turns,
    }

    goal_mode = goals_map is not None
    rows_in = art.usable_rows()
    if args.limit is not None:
        rows_in = rows_in[: args.limit]
    if not rows_in:
        raise SystemExit(f"no usable segments in {art.dir}")

    tasks = [
        {
            "index_row": row,
            "filter_seg_path": str(art.segment_path(str(row["segment_id"]))),
            "fps": args.fps,
            "fps_mode": args.fps_mode,
            "action_format": args.action_format,
            "continuous_action_hz": args.continuous_action_hz,
            "jitter_deadband_px": args.jitter_deadband_px,
            "app_filter": app_filter_payload,
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

    n_workers = max(1, min(resolve_workers(args.num_workers), len(tasks)))
    print(
        f"[conversations] {len(tasks)} segments | fps={args.fps} ({args.fps_mode}, stride {stride:g}) "
        f"format={args.action_format} mode={'goals' if goal_mode else 'goal-free'} "
        f"| workers={n_workers}",
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
    records.sort(key=lambda r: str(r["conversation_id"]))

    out_dir = ensure_dir(args.output_dir)
    write_jsonl(out_dir / "conversations.jsonl", records)
    write_jsonl(out_dir / "chat.jsonl", records)

    summary = {
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
        "grammar": formatter.grammar,
        "continuous_action_hz": continuous_action_hz,
        "jitter_deadband_px": args.jitter_deadband_px,
        "app_filter": {
            "include": sorted(app_filter.include),
            "exclude": sorted(app_filter.exclude),
            "min_frac": app_filter.min_frac,
            "unknown": app_filter.unknown,
            "split": app_filter.split,
            "min_run_frames": app_filter.min_run_frames,
            "drop_seam_turns": app_filter.drop_seam_turns,
        } if app_filter.active else None,
        "primitive_counts": dict(prim_totals) if prim_totals else None,
        "n_noop_turns": sum(r["n_turns"] - r["n_non_noop"] for r in records),
        "instruction": args.instruction,
        "instruction_field": args.instruction_field,
        "system_prompt_id": system_prompt_id,
        "system_prompt_sha256": system_prompt_sha256,
        # The grammar's own digest beside the prompt's. They are equal when no
        # --system-prompt-id was given, and differ by exactly the prompt's edits
        # when one was, so a reader can tell an edited prompt from a moved spec.
        "grammar_sha256": grammars.load(formatter.grammar).digest,
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
    write_json(out_dir / "conversations_summary.json", summary)
    write_json(out_dir / "manifest.json", {
        "artifact_type": "juergen_annotation_conversations",
        "schema_version": 2,
        "conversations": "conversations.jsonl",
        "chat": "chat.jsonl",  # split-agnostic drop-in source_path for stages 05/06
        "master_store_id": art.master_store_id,
        "filter_id": art.filter_id,
        "goals_id": goals_id,
        **summary,
    })
    print(
        f"[conversations] {len(records)} conversations, {summary['n_turns_total']} turns, "
        f"{summary['n_frames_total']} frames | statuses {dict(counts)} | "
        f"dead-zone flagged {n_flagged} -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
