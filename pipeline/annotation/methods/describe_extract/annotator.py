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

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from pipeline.annotation.lib.labeler import Labeler
from pipeline.annotation.lib.prompts import PromptPack
from pipeline.annotation.lib.units import (
    AnnotationUnit,
    _is_submission,
    frames_to_data_urls,
    is_typing,
)


@dataclass(frozen=True)
class Context:
    labeler: Labeler
    prompts: PromptPack
    cache_dir: Path


def _fmt_period(unit: AnnotationUnit) -> str:
    period = unit.view.stride / unit.view.master_fps
    return f"{period:g}"


def clean_goals(
    parsed: dict[str, Any], frame_lo: int, frame_hi: int, own_hi: int
) -> list[dict[str, Any]]:
    """Validate goal fields against the owned frame window."""
    if not isinstance(parsed, dict) or not isinstance(parsed.get("goals"), list):
        raise TypeError("extract response must contain a goals list")
    if set(parsed) != {"goals"}:
        raise ValueError("extract response must contain only goals")
    goals: list[dict[str, Any]] = []
    for index, g in enumerate(parsed["goals"]):
        if not isinstance(g, dict):
            raise TypeError(f"goal {index} must be an object")
        expected = {
            "instruction",
            "anchor",
            "grounding",
            "start_frame",
            "end_frame",
        }
        if set(g) != expected:
            raise ValueError(
                f"goal {index} fields must be exactly {sorted(expected)}, got {sorted(g)}"
            )
        instr = g.get("instruction")
        if not isinstance(instr, str):
            raise TypeError(f"goal {index} instruction must be text")
        instr = instr.strip()
        if not instr:
            raise ValueError(f"goal {index} instruction is empty")
        start = g["start_frame"]
        end = g["end_frame"]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (start, end)
        ):
            raise ValueError(f"goal {index} has invalid frame bounds")
        if end < start:
            raise ValueError(f"goal {index} end_frame precedes start_frame")
        if not frame_lo <= start <= end <= frame_hi or end > own_hi:
            raise ValueError(
                f"goal {index} bounds [{start}, {end}] are outside "
                f"the owned frame range [{frame_lo}, {own_hi}]"
            )
        anchor = g.get("anchor")
        grounding = g.get("grounding")
        if not isinstance(anchor, str) or not anchor.strip():
            raise ValueError(f"goal {index} anchor is empty")
        if not isinstance(grounding, str) or not grounding.strip():
            raise ValueError(f"goal {index} grounding is empty")
        goals.append(
            {
                "instruction": instr,
                "anchor": anchor.strip(),
                "grounding": grounding.strip(),
                "start_frame": start,
                "end_frame": end,
            }
        )
    for previous, current in pairwise(goals):
        if current["start_frame"] <= previous["end_frame"]:
            raise ValueError("goals must be strictly ordered and nonoverlapping")
    return goals


def snap_goal_starts(
    goals: list[dict[str, Any]], unit: AnnotationUnit
) -> list[dict[str, Any]]:
    """Pull each typed goal's start back to the first keystroke of its input
    burst, using what the keylog says was typed per frame. Walk back over the
    contiguous typing run, stopping at a submission (Return/Enter — that ended
    the PREVIOUS action) or any non-typing frame. Mouse/scroll-initiated goals
    are untouched. Finally re-enforce non-overlap."""
    kb = unit.keyboard
    sent = set(unit.sent_view_indices)
    first_vi = unit.sent_view_indices[0] if unit.sent_view_indices else 0
    for g in goals:
        sf = g.get("start_frame")
        if sf is None or sf not in sent:
            continue
        p = sf
        if not is_typing(kb[p]) and not (p > first_vi and is_typing(kb[p - 1])):
            continue
        while p > first_vi and is_typing(kb[p - 1]) and not _is_submission(kb[p - 1]):
            p -= 1
        g["start_frame"] = p
    for previous, current in pairwise(goals):
        if current["start_frame"] <= previous["end_frame"]:
            raise ValueError("snapped goals overlap")
    return goals


def _tokens(usage: dict[str, Any]) -> int:
    return int(usage["total_tokens"])


def run_unit(unit: AnnotationUnit, ctx: Context) -> dict[str, Any]:
    imgs = frames_to_data_urls(unit.image_refs())
    n = len(imgs)
    vis = unit.sent_view_indices
    labels = [f"frame {vi}" for vi in vis]
    period = _fmt_period(unit)

    system = ctx.prompts.render("system", frame_period_s=period)
    describe_prompt = ctx.prompts.render(
        "describe_prose", n_frames=n, frame_period_s=period
    )
    # Interleave the same `frame <N>` labels as extract, so the narration's
    # frame references are grounded in the printed index rather than the
    # model's own running count (which drifts and poisons extract's bounds).
    res_d = ctx.labeler.call_full(
        system,
        describe_prompt,
        images=imgs,
        image_labels=labels,
        cache_path=ctx.cache_dir / "describe_prose.json",
    )
    description = res_d.content

    out: dict[str, Any] = {"narration": description, "n_images_sent": n}
    if not description.strip():
        raise ValueError(f"describe pass returned an empty response for {unit.unit_id}")
    extract_system = ctx.prompts.render("extract_system", frame_period_s=period)
    extract_prompt = ctx.prompts.render(
        "extract", description=description, n_frames=n, frame_period_s=period
    )
    parsed, res_e = ctx.labeler.call_json_full(
        extract_system,
        extract_prompt,
        images=imgs,
        image_labels=labels,
        cache_path=ctx.cache_dir / "extract_from_prose.json",
    )
    goals = clean_goals(
        parsed,
        frame_lo=vis[0],
        frame_hi=vis[-1],
        own_hi=unit.owned_hi_view_idx,
    )
    snap_goal_starts(goals, unit)
    out["extract_finish"] = res_e.finish_reason

    out["goals"] = goals
    out["describe_finish"] = res_d.finish_reason
    out["actual_tokens"] = _tokens(res_d.usage) + _tokens(res_e.usage)
    return out
