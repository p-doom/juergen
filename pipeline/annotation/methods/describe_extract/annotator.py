"""describe_extract: two-pass vision-only hindsight annotation (v2 port).

  A. DESCRIBE — feed every sent frame and get a faithful, fine-grained,
     factual prose narration of what happens (no goals/intent).
  B. EXTRACT — feed that narration + the frames again (each labelled
     ``frame <N>``, N = the segment's dense view-local index) and recover the
     instruction(s) a person would type to a computer-use agent, with per-goal
     start/end frame bounds.

Plus the deterministic keystroke-burst start snap (method-internal, from the
derived per-frame action labels): the labeler is vision-only and the
screenshot of the first-keystroke tick often predates the text rendering, so
it anchors typed goals ~1 frame late; the keylog knows when typing began.

Outputs view-local inclusive [start_frame, end_frame] spans — the stage
converts them to master intervals at write time; view indices never persist.
"""

from __future__ import annotations

from typing import Any

from pipeline.annotation.lib.registry import MethodContext
from pipeline.annotation.lib.units import (
    AnnotationUnit,
    _is_submission,
    frame_activity,
    frames_to_data_urls,
)

INPUT_KIND = "frames"


def _fmt_period(unit: AnnotationUnit) -> str:
    period = unit.view.stride / unit.view.master_fps
    return f"{period:g}"


def clean_goals(parsed: dict[str, Any], frame_lo: int, frame_hi: int,
                own_hi: int) -> list[dict[str, Any]]:
    """Validate/clamp the model's goals. Bounds are the interleaved
    ``frame <N>`` labels == view indices; clamp to the indices actually sent.
    ``own_hi`` is the last view index this unit owns: a goal whose start is
    past it began in the trailing context buffer and belongs to the next
    window — drop it; surviving goals' ends clamp to own_hi."""
    goals: list[dict[str, Any]] = []
    for g in (parsed.get("goals", []) if isinstance(parsed, dict) else []):
        if not isinstance(g, dict):
            continue
        instr = str(g.get("instruction", "")).strip()
        if not instr:
            continue
        sf = ef = None
        try:
            sf, ef = int(g["start_frame"]), int(g["end_frame"])
            if ef < sf:
                sf, ef = ef, sf
            if sf > own_hi:
                continue  # opened in the tail buffer -> next window owns it
            sf = max(frame_lo, min(sf, frame_hi))
            ef = max(frame_lo, min(ef, own_hi))
        except (KeyError, TypeError, ValueError):
            sf = ef = None
        goals.append({
            "instruction": instr,
            "instruction_variants": [str(v).strip() for v in g.get("instruction_variants", [])
                                     if str(v).strip()],
            "anchor": str(g.get("anchor", "")).strip(),
            "grounding": str(g.get("grounding", "")).strip(),
            "start_frame": sf,
            "end_frame": ef,
        })
    return goals


def snap_goal_starts(goals: list[dict[str, Any]], unit: AnnotationUnit) -> list[dict[str, Any]]:
    """Pull each typed goal's start back to the first keystroke of its input
    burst, using the derived per-frame action labels. Walk back over the
    contiguous typing run, stopping at a submission (Return/Enter — that ended
    the PREVIOUS action) or any non-typing frame. Mouse/scroll-initiated goals
    are untouched. Finally re-enforce non-overlap."""
    acts = unit.actions
    sent = set(unit.sent_view_indices)
    first_vi = unit.sent_view_indices[0] if unit.sent_view_indices else 0
    for g in goals:
        sf = g.get("start_frame")
        if sf is None or sf not in sent:
            continue
        p = sf
        if frame_activity(acts[p]) != "type" and not (
            p > first_vi and frame_activity(acts[p - 1]) == "type"
        ):
            continue
        while p > first_vi and frame_activity(acts[p - 1]) == "type" and not _is_submission(acts[p - 1]):
            p -= 1
        g["start_frame"] = p
        if g.get("end_frame") is not None and g["end_frame"] < p:
            g["end_frame"] = p
    ordered = sorted((g for g in goals if isinstance(g.get("start_frame"), int)
                      and isinstance(g.get("end_frame"), int)), key=lambda g: g["start_frame"])
    for a, b in zip(ordered, ordered[1:], strict=False):  # noqa: RUF007 - dicts, not pairs math
        if a["end_frame"] >= b["start_frame"]:
            a["end_frame"] = max(a["start_frame"], b["start_frame"] - 1)
    return goals


def _tokens(usage: dict[str, Any] | None) -> int:
    if not isinstance(usage, dict):
        return 0
    return usage.get("total_tokens") or ((usage.get("prompt_tokens") or 0)
                                         + (usage.get("completion_tokens") or 0))


def run_unit(unit: AnnotationUnit, ctx: MethodContext) -> dict[str, Any]:
    imgs = frames_to_data_urls(unit.image_refs(), target_height=ctx.vlm_frame_height,
                               jpeg_quality=ctx.jpeg_quality)
    n = len(imgs)
    vis = unit.sent_view_indices
    labels = [f"frame {vi}" for vi in vis]
    period = _fmt_period(unit)

    system = ctx.prompts.render("system", frame_period_s=period)
    describe_prompt = ctx.prompts.render("describe_prose", n_frames=n, frame_period_s=period)
    # Interleave the same `frame <N>` labels as extract, so the narration's
    # frame references are grounded in the printed index rather than the
    # model's own running count (which drifts and poisons extract's bounds).
    res_d = ctx.labeler.call_full(system, describe_prompt, images=imgs, image_labels=labels,
                                  cache_path=ctx.cache_dir / "describe_prose.txt",
                                  no_cache=ctx.no_cache)
    description = res_d.content

    out: dict[str, Any] = {"narration": description, "n_images_sent": n}
    goals: list[dict[str, Any]] = []
    extract_usage = None
    if description.strip():
        extract_system = ctx.prompts.render("extract_system", frame_period_s=period)
        extract_prompt = ctx.prompts.render("extract", description=description,
                                            n_frames=n, frame_period_s=period)
        try:
            parsed, res_e = ctx.labeler.call_json_full(
                extract_system, extract_prompt, images=imgs, image_labels=labels,
                cache_path=ctx.cache_dir / "extract_from_prose.txt", no_cache=ctx.no_cache)
            goals = clean_goals(parsed, frame_lo=vis[0], frame_hi=vis[-1],
                                own_hi=unit.owned_hi_view_idx)
            snap_goal_starts(goals, unit)
            extract_usage = res_e.usage
            out["extract_finish"] = res_e.finish_reason
        except Exception as exc:
            out["extract_error"] = f"{type(exc).__name__}: {exc}"
    else:
        out["extract_error"] = "empty_description"

    out["goals"] = goals
    out["describe_finish"] = res_d.finish_reason
    out["actual_tokens"] = _tokens(res_d.usage) + _tokens(extract_usage)
    return out
