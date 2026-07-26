"""lumine_thinking_goals: goal-conditioned, dense variant of lumine_thinking.

Same sequential day-watching with carried memory and future-blind verification
as lumine_thinking (Track A) — the helpers are imported verbatim from that
method so the audit gate, self-test canary, resume ledger, and coordinate
handling are byte-identical. Two deliberate changes:

  1. GOAL CONDITIONING. Each clip's writer prompt carries the person's active
     short-horizon goal (fold goals from ``--param goals_fold_dir``, joined by
     day-second interval — the goal with max overlap of the clip's time span).
     The goal is intent-only (no plan); it disambiguates what the person is
     doing without being treated as on-screen evidence. The active goal per
     clip is recorded to a ``goals_active.jsonl`` sidecar (write/train
     consistency: the training window later conditions on the SAME goal text
     the writer saw). Days with no fold goal run unconditioned (graceful).

  2. DENSITY BY DESIGN. 15-frame clips (was 30) and up to 5 thoughts/clip
     (was 3), with the writer prompt broadened so outcome reactions and
     corrections are first-class thought moments — no second densify pass.

Everything downstream (goal rows, verify, memory sidecar) is produced by the
shared stage exactly as for lumine_thinking; only the writer prompt gains a
GOAL block and the method emits the extra goals_active sidecar.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from realigned_pipeline.annotation.lib.days import DayStream, fmt_t, frame_label
from realigned_pipeline.annotation.lib.labeler import ContentFilteredError
from realigned_pipeline.annotation.lib.registry import MethodContext
from realigned_pipeline.annotation.lib.units import frames_to_data_urls
from realigned_pipeline.annotation.methods.lumine_thinking.annotator import (
    MEMORY_SEED,
    VERIFY_MODES,
    _audit,
    _clean_thoughts,
    _norm,
    _self_test,
    _tokens,
    plan_clips,
)
from realigned_pipeline.lib.common import write_json, write_jsonl

INPUT_KIND = "days"
LABELER_DEFAULTS = {"temperature": 0.2, "reasoning_effort": "low"}

# Dense-by-design defaults (validated in the goal_dense smoke series). Override
# via --param key=value. cap 5 (NOT higher — cap 8 over-generates and collapses
# the verify pass-rate); 32K budgets so dense writer/verify calls never
# truncate-then-retry.
DEFAULT_CLIP_FRAMES = 15
DEFAULT_MAX_THOUGHTS_PER_CLIP = 5
DEFAULT_CTX_FRAMES = 12
DEFAULT_SELFTEST_MIN_SEP_S = 1800.0
DEFAULT_WRITER_MAX_TOKENS = 32000
DEFAULT_VERIFY_MAX_TOKENS = 32000

_NO_GOAL = "(no recovered goal for this stretch — annotate from the frames and memory alone)"


def _render(ctx: MethodContext, frames: list) -> tuple[list[str], list[str]]:
    imgs = frames_to_data_urls([fr.image for fr in frames],
                               target_height=ctx.vlm_frame_height,
                               jpeg_quality=ctx.jpeg_quality)
    return imgs, [frame_label(fr) for fr in frames]


def _load_short_goals(goals_fold_dir: Path, day_tag: str) -> list[dict[str, Any]]:
    """Short-horizon fold goals for a day: [{text, t_start, t_end, id, ...}].
    Missing/empty file -> [] (day runs unconditioned)."""
    p = goals_fold_dir / day_tag / "goals" / "goals.json"
    if not p.is_file():
        return []
    try:
        goals = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return [g for g in goals if isinstance(g, dict) and g.get("horizon") == "short"
            and g.get("t_start") is not None and g.get("t_end") is not None]


def _active_goal(short_goals: list[dict[str, Any]], t0: float, t1: float) -> dict[str, Any] | None:
    """The short goal with the greatest overlap of the clip span [t0, t1]."""
    best, best_ov = None, 0.0
    for g in short_goals:
        ov = min(t1, float(g["t_end"])) - max(t0, float(g["t_start"]))
        if ov > best_ov:
            best, best_ov = g, ov
    return best


def run_unit(item: dict[str, Any], ctx: MethodContext) -> dict[str, Any]:
    """Goal-conditioned twin of lumine_thinking.run_unit. Walks the day's clips
    in order (resuming from the per-clip ledger), injects the active fold goal
    into each clip's writer prompt, verifies every thought future-blind, runs
    the day-end self-test, and writes memory + goals_active sidecars."""
    day: DayStream = item["day"]
    p = ctx.params
    clip_frames = int(p.get("clip_frames", DEFAULT_CLIP_FRAMES))
    max_per_clip = int(p.get("max_thoughts_per_clip", DEFAULT_MAX_THOUGHTS_PER_CLIP))
    ctx_frames = int(p.get("ctx_frames", DEFAULT_CTX_FRAMES))
    verify_mode = str(p.get("verify_mode", "batched"))
    if verify_mode not in VERIFY_MODES:
        raise ValueError(f"verify_mode must be one of {VERIFY_MODES}, got {verify_mode!r}")
    writer_max = int(p.get("writer_max_tokens", DEFAULT_WRITER_MAX_TOKENS))
    verify_max = int(p.get("verify_max_tokens", DEFAULT_VERIFY_MAX_TOKENS))
    selftest_min_sep_s = float(p.get("selftest_min_sep_s", DEFAULT_SELFTEST_MIN_SEP_S))
    goals_fold_dir = p.get("goals_fold_dir")
    if not goals_fold_dir:
        raise ValueError("lumine_thinking_goals needs --param goals_fold_dir=<hindsight_fold "
                         "pipeline_runs/days dir> (per-day <tag>/goals/goals.json)")
    day_units_dir = Path(p["day_units_dir"])
    memory_path = Path(p["memory_path"])
    force = bool(p.get("force"))
    report = p.get("report_tokens") or (lambda n: None)
    spent = {"total": 0}

    def track(n: int) -> None:
        spent["total"] += int(n)
        report(n)

    short_goals = _load_short_goals(Path(goals_fold_dir), day.day_tag)
    clips = plan_clips(day, clip_frames)
    system = ctx.prompts.get("system")
    memory = MEMORY_SEED
    all_thoughts: list[dict[str, Any]] = []
    mem_rows: list[dict[str, Any]] = []
    goal_rows: list[dict[str, Any]] = []
    n_dropped = 0
    n_resumed = 0
    n_goal_clips = 0

    for clip_key, chunk_idx, is_chunk_start, clip in clips:
        goal = _active_goal(short_goals, clip[0].t_day_s, clip[-1].t_day_s)
        goal_text = _norm(goal["text"]) if goal else _NO_GOAL
        if goal:
            n_goal_clips += 1
        goal_rows.append({
            "clip_key": clip_key,
            "day_idx_range": [clip[0].day_idx, clip[-1].day_idx],
            "t_range": [fmt_t(clip[0].t_day_s), fmt_t(clip[-1].t_day_s)],
            "segments": sorted({fr.segment_id for fr in clip}),
            "goal_id": (goal.get("id") if goal else None),
            "goal_text": goal_text if goal else None,
            "goal_t_start": (goal.get("t_start") if goal else None),
            "goal_t_end": (goal.get("t_end") if goal else None),
            "goal_long_ref": (goal.get("long_ref") if goal else None),
        })

        rec_path = day_units_dir / f"{clip_key}.json"
        if rec_path.exists() and not force:
            rec = json.loads(rec_path.read_text())
            memory = str(rec["memory_out"])
            all_thoughts.extend(rec.get("thoughts", []))
            n_dropped += int(rec.get("n_dropped_anchor") or 0)
            n_resumed += 1
        else:
            gap_note = ""
            if is_chunk_start and chunk_idx > 0:
                gap_note = (f"\n(NOTE: the recording resumed at {fmt_t(clip[0].t_day_s)} "
                            "after a break — treat this clip as a fresh sit-down.)")
            imgs, labels = _render(ctx, clip)
            user = ctx.prompts.render("clip", max_thoughts=max_per_clip,
                                      goal=goal_text, memory=memory + gap_note)
            content_filtered = False
            try:
                parsed, res = ctx.labeler.call_json_full(
                    system, user, images=imgs, image_labels=labels,
                    cache_path=ctx.cache_dir / f"{clip_key}.txt", no_cache=ctx.no_cache,
                    max_completion_tokens=writer_max)
            except ContentFilteredError:
                parsed, res = {}, None
                content_filtered = True
            clip_tokens = _tokens(res.usage) if res else 0
            track(clip_tokens)

            thoughts, dropped = _clean_thoughts(parsed, clip, max_per_clip)
            memory_out = _norm(parsed.get("memory", "")) or memory
            log = str(parsed.get("log", "")).strip()

            _audit(ctx, day, thoughts, verify_mode, ctx_frames, clip_key, verify_max, track)
            rec = {
                "clip_key": clip_key,
                "chunk_index": chunk_idx,
                "day_idx_range": [clip[0].day_idx, clip[-1].day_idx],
                "t_range": [fmt_t(clip[0].t_day_s), fmt_t(clip[-1].t_day_s)],
                "segments": sorted({fr.segment_id for fr in clip}),
                "n_frames": len(clip),
                "gap_note": bool(gap_note),
                "content_filtered": content_filtered,
                "goal_text": goal_text,
                "memory_in": memory,
                "memory_out": memory_out,
                "log": log,
                "thoughts": thoughts,
                "n_dropped_anchor": dropped,
                "writer_finish": res.finish_reason if res else "content_filter",
                "writer_tokens": clip_tokens,
            }
            write_json(rec_path, rec)
            memory = memory_out
            all_thoughts.extend(thoughts)
            n_dropped += dropped
        mem_rows.append({
            "clip_key": clip_key,
            "chunk_index": rec["chunk_index"],
            "day_idx_range": rec["day_idx_range"],
            "t_range": rec["t_range"],
            "memory": rec["memory_out"],
            "log": rec.get("log", ""),
        })

    selftest = _self_test(ctx, day, all_thoughts, verify_mode, ctx_frames,
                          verify_max, selftest_min_sep_s, track) \
        if all_thoughts else {"status": "skipped", "reason": "no thoughts"}
    if selftest["status"] == "failed":
        write_json(day_units_dir / "selftest_failed.json", selftest)
        raise RuntimeError(
            f"{day.day_tag}: verifier SELF-TEST FAILED (a planted anachronism passed) — "
            "the gate is broken; aborting the day, nothing emitted")

    write_jsonl(memory_path, mem_rows)
    write_jsonl(day_units_dir / "goals_active.jsonl", goal_rows)
    n_pass = sum(1 for t in all_thoughts if (t.get("verify") or {}).get("verdict") == "pass")
    return {
        "thoughts": all_thoughts,
        "n_clips": len(clips),
        "n_clips_resumed": n_resumed,
        "n_clips_with_goal": n_goal_clips,
        "n_thoughts": len(all_thoughts),
        "n_pass": n_pass,
        "n_dropped_anchor": n_dropped,
        "verify_mode": verify_mode,
        "selftest": selftest,
        "actual_tokens": spent["total"],
    }
