#!/usr/bin/env python3
"""Stage 04t (thinking conversations): lumine_thinking(-goals) -> window SFT.

AD-HOC / INTERIM sibling of stage 04 (see HANDOFF_stage04_thinking_sft.md).
Two modes, branched on the goals artifact's manifest ``method``:

LEGACY (``lumine_thinking``) — unchanged goal-free windows:
  * each user-day's stream (lib/days) is tiled per CHUNK into fixed
    ``--window-frames`` windows; one conversation per window that contains
    >= 1 verified thought (``--thinking-only``, default);
  * THINK-THEN-ACT: the anchor frame's assistant turn becomes
    ``<think>\\n{thought}\\n</think>\\n{action}``; training fps MUST equal
    the annotation fps (v1 lock);
  * first user turn carries the last ``--context-thoughts`` earlier VERIFIED
    same-chunk thoughts; ``--terminal-token`` (default None) rides glued at
    the end of every window's final assistant message.

GOAL-BOUNDED (``lumine_thinking_goals``) — goal-conditioned windows:
  * train ONLY on frames inside GOAL SPANS (contiguous runs of clips sharing
    a ``goal_id`` in the artifact's ``units/<day>/goals_active.jsonl``
    sidecar); no-goal stretches are excluded entirely;
  * windows tile each span in order, up to ``--window-frames`` frames each,
    never crossing a recording-gap chunk boundary;
  * every window's first user turn gets ``GOAL: {goal_text}`` and, when a
    strictly-earlier memory row exists (``memory/<day>.jsonl``, latest row
    whose day_idx_range ends BEFORE the window's first frame — no overlap,
    no future leak), ``So far: {memory}``;
  * TERMINATE supervision on the OUTCOME frame — the first frame strictly
    after ``goal_t_end`` in the same chunk. The final window of the span
    containing goal_t_end is extended by that one extra frame whose assistant
    target is the terminate turn (never glued to an action). Gating is
    ``--terminate-boundaries``:
      - ``clean`` (default): plain ``TERMINATE`` only when the outcome frame
        exists <= --terminate-max-lag-s after goal_t_end AND the next clip
        belongs to a different goal (goal->goal handoff);
      - ``verified``: needs a lumine_goal_boundaries sidecar
        (``--boundaries-dir``); only spans judged completed && confidence
        high terminate, with ``<think>\\n{final_thought}\\n</think>\\n
        TERMINATE``; their near-miss clip (when mined and in-span) gets
        ``next_step_thought`` as the thought on its first frame;
      - ``all``: terminate whenever the outcome frame exists (ablations).
    When a terminate turn is emitted, span frames after goal_t_end are
    dropped (post-completion actions under a completed goal are
    contradictory supervision); spans without a terminate keep all frames
    and just end on the last in-span frame's normal action.
  * windows without thoughts are KEPT (--thinking-only is ignored);
  * ``--memory-update-samples``: non-terminate windows whose last fully-
    contained clip has a memory row get a trailing
    ``MEMORY UPDATE: ...`` user turn (no image) + the clip's memory verbatim
    as the assistant target — trains the model to write its own memory.

``--action-format`` selects the assistant-turn action label from the
lib/action_format FORMATTERS registry (``canonical`` default;
``ordered_events_v3`` for the type()-collapsed ordered mini-programs;
``computer_use_rel_v1`` for Qwen3-VL-native <tool_call> blocks — goal mode
then defaults the system prompt to cua_v4_thinking.txt and terminate turns
render the formatter's native terminate block via ``terminate_line()``).

The output is the canonical chat.jsonl schema (content blocks, one assistant
turn per frame) with the same manifest join-guards as stage 04, so stages
05/06 run unchanged.

Run::

    cd data_pipeline
    uv run python realigned_pipeline/stage_04_thinking_conversations.py \
        --filter-dir <stage-03> --goals-dir <lumine_thinking(-goals) artifact> \
        --clips-manifest <stage-00/02 manifest> --fps 0.5 \
        --window-frames 24 --output-dir <dest>
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.annotation.lib.days import (  # noqa: E402
    DEFAULT_GAP_CUT_S,
    DEFAULT_TZ,
    DayStream,
    build_day_index,
    build_day_stream,
    fmt_t,
)
from realigned_pipeline.lib.action_format import (  # noqa: E402
    DEFAULT_CONTINUOUS_ACTION_HZ,
    FORMATTERS,
    get_formatter,
)
from realigned_pipeline.lib.common import (  # noqa: E402
    ensure_dir,
    normalize_dashed_argv,
    read_jsonl,
    str2bool,
    write_json,
    write_jsonl,
)
from realigned_pipeline.lib.goals import assert_same_artifact, load_goals  # noqa: E402
from realigned_pipeline.lib.manifest import make_artifact_id  # noqa: E402
from realigned_pipeline.lib.views import FilterArtifact  # noqa: E402

# Goal-free system prompt + the think sentences (the samples' actual
# distribution: one short first-person thought at commit/switch/reconsider
# moments, silence during steady execution, an earlier-thoughts recap block).
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
# ``terminate_line()`` — the TERMINATE literal for the text formats, the
# native terminate tool_call block for computer_use_rel_v1.
TERMINATE_TOKEN = "TERMINATE"
TERMINATE_BOUNDARY_MODES = ("clean", "verified", "all")
DEFAULT_TERMINATE_MAX_LAG_S = 180.0
MEMORY_UPDATE_PROMPT = ("MEMORY UPDATE: Summarize your progress toward the "
                        "goal so far and anything still open.")
SYSTEM_PROMPTS_DIR = Path(__file__).resolve().parent / "system_prompts"
DEFAULT_GOAL_SYSTEM_PROMPT_FILE = SYSTEM_PROMPTS_DIR / "cua_v3_thinking.txt"
# Formats whose default goal prompt is NOT cua_v3 (the tool spec must match
# the emission format the formatter produces).
GOAL_SYSTEM_PROMPT_FILES = {
    "computer_use_rel_v1": SYSTEM_PROMPTS_DIR / "cua_v4_thinking.txt",
}


def goal_system_prompt_file(action_format: str) -> Path:
    """Goal mode's default system prompt for the selected action format."""
    return GOAL_SYSTEM_PROMPT_FILES.get(action_format, DEFAULT_GOAL_SYSTEM_PROMPT_FILE)


def resolve_terminal_token(terminal_token: str | None, action_format: str) -> str | None:
    """Legacy-mode --terminal-token: the literal ``TERMINATE`` sentinel
    resolves through the formatter's ``terminate_line()`` (the same literal
    for canonical/ordered_events_* — byte-identical legacy behavior — and the
    native terminate block for computer_use_rel_v1). Anything else, including
    the default None, passes through untouched."""
    if terminal_token == TERMINATE_TOKEN:
        return get_formatter(action_format).terminate_line()
    return terminal_token

_WORKER: dict[str, Any] = {}


def _init_worker(filter_dir: str) -> None:
    _WORKER["art"] = FilterArtifact(Path(filter_dir))


def _norm(text: Any) -> str:
    return " ".join(str(text).split())


def _text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


def _image(i: str) -> dict[str, Any]:
    return {"type": "image", "image": i}


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
    """{goal_id -> boundaries row} for one day (lumine_goal_boundaries
    sidecar: one row per goal span, fail-closed judgments)."""
    if boundaries_dir is None:
        return {}
    p = boundaries_dir / f"{day_tag}.jsonl"
    if not p.is_file():
        return {}
    return {r["goal_id"]: r for r in read_jsonl(p)}


def check_sidecar_alignment(row: dict[str, Any], day: DayStream) -> None:
    """Assert the rebuilt day stream addresses the sidecar's clip: the frames
    at its recorded day_idx_range must carry its recorded t_range (same guard
    as lumine_goal_boundaries) — a mismatch means wrong filter/fps/tz."""
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


def group_goal_runs(active: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Goal SPANS as contiguous runs: consecutive goals_active rows sharing a
    non-null goal_id. A goal_id that recurs after an interruption (no-goal or
    other-goal clips in between) yields SEPARATE runs. Each run carries
    ``next_goal_id`` — the goal_id of the clip right after the run (None at
    day end or before a no-goal clip) — for the clean-mode handoff test."""
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


def select_memory(memory_rows: list[dict[str, Any]], win_start_idx: int) -> str:
    """Rolling memory as of the window start: the memory row with the LATEST
    day_idx_range that ends strictly BEFORE the window's first frame index —
    no overlap with the window, so nothing the memory summarizes is inside
    it (the future-leak fix; day_idx_range is inclusive on both ends)."""
    best = None
    for row in memory_rows:
        rng = row.get("day_idx_range") or [None, None]
        if rng[1] is None or int(rng[1]) >= win_start_idx:
            continue
        if best is None or int(rng[1]) > int(best["day_idx_range"][1]):
            best = row
    return _norm(best.get("memory", "")) if best and str(best.get("memory") or "").strip() else ""


def goal_memory_block(goal_text: str, memory: str) -> str:
    """First-user-turn conditioning: the span's goal + rolling memory (the
    memory IS the contextual summary — no earlier-thoughts recap)."""
    block = f"GOAL: {goal_text}"
    if memory:
        block += f"\nSo far: {memory}"
    return block


def memory_update_row(memory_rows: list[dict[str, Any]], w0: int, w1: int
                      ) -> dict[str, Any] | None:
    """The LAST clip fully contained in the window [w0, w1] (day_idx_range
    entirely inside) that carries a non-empty memory — the target of a
    memory-update appendix."""
    best = None
    for row in memory_rows:
        rng = row.get("day_idx_range") or [None, None]
        if rng[0] is None or rng[1] is None:
            continue
        if w0 <= int(rng[0]) and int(rng[1]) <= w1 and str(row.get("memory") or "").strip():
            if best is None or int(rng[1]) > int(best["day_idx_range"][1]):
                best = row
    return best


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
    memory_update_samples: bool,
    system_prompt: str | None,
    fps: float,
    action_format: str,
    annotation_method: str,
) -> tuple[list[dict[str, Any]], Counter]:
    """Pure goal-bounded window construction over one rebuilt day stream.
    ``anchored`` maps day_idx -> verified-thought row (goals.jsonl). Returns
    (conversation rows, stats Counter)."""
    if terminate_mode not in TERMINATE_BOUNDARY_MODES:
        raise ValueError(f"terminate mode must be one of {TERMINATE_BOUNDARY_MODES}, "
                         f"got {terminate_mode!r}")
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
            for k in range(0, len(piece), window_frames):
                win = piece[k: k + window_frames]
                is_final = win[-1] == idxs[-1]
                terminate_text = plan["terminate_text"] if is_final else None
                outcome_idx = plan["outcome_idx"] if is_final else None
                if len(win) < 2 and terminate_text is None:
                    stats["n_windows_skipped_short"] += 1
                    continue

                block = goal_memory_block(goal_text, select_memory(memory_rows, win[0]))
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
                        # deliberate negative from the boundaries pass —
                        # replaces any annotation thought, exempt from the
                        # evidence-boundary demotion
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

                mem_update = None
                if memory_update_samples and terminate_text is None:
                    mem_update = memory_update_row(memory_rows, win[0], win[-1])
                    if mem_update is not None:
                        messages.append({"role": "user",
                                         "content": [_text(MEMORY_UPDATE_PROMPT)]})
                        messages.append({"role": "assistant",
                                         "content": [_text(str(mem_update["memory"]))]})
                        stats["n_memory_update_samples"] += 1

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
                    "memory_update": mem_update is not None,
                    "n_frames": n_frames,
                    "n_turns": n_frames + (1 if mem_update is not None else 0),
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
            # evidence boundary: anchors too close after a mid-chunk cut
            # lose their <think> (the audit's evidence sits in the
            # previous window); chunk-start windows are exempt.
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


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def build_day_windows(task: dict[str, Any]) -> dict[str, Any]:
    """Worker: one day -> conversation rows. Failures captured, never raised."""
    day_row = task["day_row"]
    day_tag = str(day_row["day_tag"])
    try:
        day: DayStream = build_day_stream(
            day_row, _WORKER["art"], fps=task["fps"], fps_mode="exact",
            gap_cut_s=task["gap_cut_s"], action_format=task["action_format"],
            continuous_action_hz=task["continuous_action_hz"],
        )
        if not day.frames:
            return {"day_tag": day_tag, "status": "empty_day", "rows": [], "stats": {}}
        # thought anchor -> day frame (exact: training fps == annotation fps)
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
                day, task["active"], task["memory"],
                {r["goal_id"]: r for r in task["boundaries"]}, anchored,
                window_frames=task["window_frames"],
                terminate_mode=task["terminate_boundaries"],
                terminate_max_lag_s=task["terminate_max_lag_s"],
                min_anchor_lead=task["min_anchor_lead"],
                memory_update_samples=task["memory_update_samples"],
                system_prompt=task["system_prompt"],
                fps=task["fps"],
                action_format=task["action_format"],
                annotation_method=task["annotation_method"],
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


def parse_args() -> argparse.Namespace:
    normalize_dashed_argv()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--filter-dir", type=Path, required=True)
    p.add_argument("--goals-dir", type=Path, required=True,
                   help="A lumine_thinking / lumine_thinking_goals stage-03b artifact.")
    p.add_argument("--clips-manifest", type=Path, required=True,
                   help="Stage-00/02 realigned clips_manifest.jsonl (day grouping).")
    p.add_argument("--day-index-cache", type=Path, default=None)
    p.add_argument("--tz", default=DEFAULT_TZ)
    p.add_argument("--gap-cut-s", type=float, default=DEFAULT_GAP_CUT_S)
    p.add_argument("--fps", type=float, required=True,
                   help="Training fps; MUST equal the artifact's annotation fps (v1).")
    p.add_argument("--window-frames", type=int, default=24,
                   help="Frames per window (24 ~ 32k tokens at 720p/0.5fps).")
    p.add_argument("--action-format", type=str, default="canonical",
                   choices=sorted(FORMATTERS),
                   help="Assistant-turn action label from the lib/action_format registry "
                        "('canonical' default; 'ordered_events_v3' for typed ordered "
                        "mini-programs).")
    p.add_argument("--continuous-action-hz", type=float,
                   default=DEFAULT_CONTINUOUS_ACTION_HZ,
                   help="ordered_events_* only: internal motor-grid rate (NOT a frame rate).")
    p.add_argument("--context-thoughts", type=int, default=8,
                   help="Legacy mode: earlier same-chunk thoughts on the first user turn.")
    p.add_argument("--min-anchor-lead", type=int, default=12,
                   help="A thought anchored closer than this after a mid-chunk window cut "
                        "loses its <think> (audit evidence sits in the previous window).")
    p.add_argument("--thinking-only", nargs="?", const=True, type=str2bool, default=True,
                   metavar="BOOL", help="Legacy mode: emit only windows with >=1 thought "
                                        "(default; ignored in goal mode — goal windows "
                                        "without thoughts still supervise actions).")
    p.add_argument("--terminate-boundaries", choices=TERMINATE_BOUNDARY_MODES,
                   default="clean",
                   help="Goal mode TERMINATE gating: 'clean' (default, structural "
                        "goal->goal handoff within --terminate-max-lag-s), 'verified' "
                        "(lumine_goal_boundaries completed+high only; needs "
                        "--boundaries-dir), 'all' (every span with an outcome frame).")
    p.add_argument("--boundaries-dir", type=Path, default=None,
                   help="A lumine_goal_boundaries artifact (or its boundaries/ dir); "
                        "required for --terminate-boundaries verified.")
    p.add_argument("--terminate-max-lag-s", type=float, default=DEFAULT_TERMINATE_MAX_LAG_S,
                   help="'clean' mode: max seconds between goal_t_end and the outcome "
                        "frame for a terminate to be emitted.")
    p.add_argument("--memory-update-samples", nargs="?", const=True, type=str2bool,
                   default=False, metavar="BOOL",
                   help="Goal mode: append a MEMORY UPDATE user turn + the last fully-"
                        "contained clip's memory as the assistant target to non-terminate "
                        "windows (trains the model to write its own 'So far' memory).")
    p.add_argument("--system-prompt", type=str, default=None)
    p.add_argument("--system-prompt-file", type=Path, default=None,
                   help="Read the system message from a file (e.g. "
                        "system_prompts/cua_v1_thinking.txt / cua_v2_thinking.txt). "
                        "Goal mode defaults to system_prompts/cua_v3_thinking.txt "
                        "(cua_v4_thinking.txt for --action-format computer_use_rel_v1).")
    p.add_argument("--no-system-prompt", action="store_true")
    p.add_argument("--terminal-token", type=str, default=None,
                   help="Legacy mode only (default None: windows are arbitrary cuts, not "
                        "completions); the literal TERMINATE resolves through the "
                        "formatter's terminate_line(). Goal mode ignores this — "
                        "TERMINATE supervision is --terminate-boundaries policy on "
                        "outcome frames.")
    p.add_argument("--day-filter", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None, help="First N days (debug).")
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    art = FilterArtifact(args.filter_dir)

    gm_path = args.goals_dir / "manifest.json"
    if not gm_path.is_file():
        raise SystemExit(f"no manifest.json under {args.goals_dir}")
    gm = json.loads(gm_path.read_text())
    if gm.get("method") not in ("lumine_thinking", "lumine_thinking_goals"):
        raise SystemExit(f"--goals-dir is method {gm.get('method')!r}; this stage "
                         "consumes lumine_thinking / lumine_thinking_goals artifacts only")
    goal_conditioned = gm.get("method") == "lumine_thinking_goals"
    assert_same_artifact(str(gm.get("master_store_id")), art.master_store_id,
                         what="master_store_id")
    assert_same_artifact(str(gm.get("filter_id")), art.filter_id, what="filter_id")
    if float(gm.get("fps") or 0) != args.fps:
        raise SystemExit(f"--fps {args.fps} != artifact annotation fps {gm.get('fps')} "
                         "(v1 locks training fps to annotation fps)")
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

    cache = args.day_index_cache
    day_rows = None
    if cache is not None and cache.is_file():
        doc = json.loads(cache.read_text())
        if doc.get("filter_id") == art.filter_id and doc.get("tz") == args.tz:
            day_rows = doc["days"]
    if day_rows is None:
        day_rows, counters = build_day_index(art, args.clips_manifest, tz=args.tz)
        print(f"[04t] day index: {counters}", flush=True)
        if cache is not None:
            write_json(cache, {"filter_id": art.filter_id, "tz": args.tz,
                               "clips_manifest": str(args.clips_manifest),
                               "counters": counters, "days": day_rows})

    if goal_conditioned:
        # goal mode trains on goal spans — days come from the goals_active
        # sidecars (a day with goals but zero verified thoughts still trains)
        units = args.goals_dir / "units"
        wanted = {d.name for d in units.iterdir()
                  if d.is_dir() and (d / "goals_active.jsonl").is_file()} if units.is_dir() else set()
    else:
        wanted = set(by_day)
    if args.day_filter:
        wanted &= set(args.day_filter)
    day_rows = [d for d in day_rows if d["day_tag"] in wanted]
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
            "gap_cut_s": args.gap_cut_s,
            "window_frames": args.window_frames,
            "action_format": args.action_format,
            "continuous_action_hz": args.continuous_action_hz,
            "context_thoughts": args.context_thoughts,
            "min_anchor_lead": args.min_anchor_lead,
            "thinking_only": bool(args.thinking_only),
            "system_prompt": system_prompt,
            "terminal_token": args.terminal_token,
            "goal_conditioned": goal_conditioned,
            "annotation_method": gm.get("method"),
            "terminate_boundaries": args.terminate_boundaries,
            "terminate_max_lag_s": args.terminate_max_lag_s,
            "memory_update_samples": bool(args.memory_update_samples),
            "active": active,
            "memory": memory,
            "boundaries": bounds,
        }

    tasks = [_mk_task(d) for d in day_rows]

    n_workers = max(1, min(args.num_workers, len(tasks)))
    mode = "goal_windows" if goal_conditioned else "thinking_windows"
    print(f"[04t] {len(tasks)} days, {sum(len(t['goals']) for t in tasks):,} thoughts | "
          f"mode={mode} window={args.window_frames}f fps={args.fps} "
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
    records.sort(key=lambda r: str(r["conversation_id"]))

    out_dir = ensure_dir(args.output_dir)
    write_jsonl(out_dir / "conversations.jsonl", records)
    write_jsonl(out_dir / "chat.jsonl", records)
    summary = {
        "mode": mode,
        "goal_conditioned": goal_conditioned,
        "n_conversations": len(records),
        "n_frames_total": sum(r["n_frames"] for r in records),
        "n_thoughts_placed": totals["n_thoughts_placed"],
        "n_thoughts_demoted_boundary": totals["n_demoted"],
        "n_anchors_unmapped": totals["n_unmapped"],
        "status_counts": dict(counts),
        "fps": args.fps,
        "window_frames": args.window_frames,
        "action_format": args.action_format,
        "continuous_action_hz": args.continuous_action_hz,
        "context_thoughts": args.context_thoughts,
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
            "memory_update_samples": bool(args.memory_update_samples),
            "n_goal_spans": totals["n_spans"],
            "n_terminate_turns": totals["n_terminate_turns"],
            "terminate_skips": {k.removeprefix("n_terminate_skipped_"): v
                                for k, v in sorted(totals.items())
                                if k.startswith("n_terminate_skipped_")},
            "n_memory_update_samples": totals["n_memory_update_samples"],
            "n_near_miss_attached": totals["n_near_miss_attached"],
            "n_near_miss_unplaced": totals["n_near_miss_unplaced"],
            "n_frames_dropped_post_goal": totals["n_frames_dropped_post_goal"],
            "n_no_goal_frames_excluded": totals["n_no_goal_frames_excluded"],
            "n_windows_skipped_short": totals["n_windows_skipped_short"],
        })
    write_json(out_dir / "conversations_summary.json", summary)
    write_json(out_dir / "manifest.json", {
        "artifact_type": "juergen_annotation_conversations",
        "schema_version": 2,
        "conversations": "conversations.jsonl",
        "chat": "chat.jsonl",
        "master_store_id": art.master_store_id,
        "filter_id": art.filter_id,
        "goals_id": goals_id,
        **summary,
    })
    tail = (f"{totals['n_terminate_turns']} terminate, "
            f"{totals['n_memory_update_samples']} memory-update, "
            if goal_conditioned else "")
    print(f"[04t] {len(records)} windows, {totals['n_thoughts_placed']:,} thoughts placed "
          f"({totals['n_demoted']} demoted, {totals['n_unmapped']} unmapped), {tail}"
          f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
