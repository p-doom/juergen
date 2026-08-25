"""plans: enrichment pass — goals artifact in, + plan/plan_flags out.

For each annotated unit it makes one cached labeler call: the unit's describe
narration (sidecar from the producing method) + its goals in time order + each
goal's start-tick screenshot, and gets back a 1-2 sentence first-person plan
per goal — written strictly from the information state at that goal's start
(no outcome/clairvoyance, no restatement; situation + method). Stage 04
renders the first assistant turn as ``plan\\n<first action>`` under
``--use-plans``.

Deterministic quality flags (recorded, not enforced — stage 04 decides):
``empty``, ``restates_instruction``, ``too_long``, ``not_first_person``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pipeline.crowdcast.annotation.lib.registry import MethodContext
from pipeline.crowdcast.annotation.lib.units import frames_to_data_urls
from pipeline.crowdcast.lib.views import SegmentView

INPUT_KIND = "goals"

STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "then", "this", "to", "with", "i", "ill", "i'll", "my", "so"]
)


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9'/._-]*", text.lower()) if w not in STOPWORDS}


def plan_flags(plan: str, instruction: str) -> list[str]:
    flags: list[str] = []
    if not plan.strip():
        return ["empty"]
    iw, pw = content_words(instruction), content_words(plan)
    novel = pw - iw
    if iw and len(iw & pw) / len(iw) > 0.6 and len(novel) < 4:
        flags.append("restates_instruction")
    if len(re.findall(r"[.!?](?:\s|$)", plan.strip())) > 3 or len(plan) > 500:
        flags.append("too_long")
    if not re.search(r"\b(i|i'll|i'm|my)\b", plan.lower()):
        flags.append("not_first_person")
    return flags


def goal_start_frame(view: SegmentView, start_master_idx: int):
    """The view frame at the goal's start tick — exact match, else the nearest
    selected frame after it (the start tick can be masked), else before."""
    exact = [f for f in view.frames if f.master_idx == start_master_idx]
    if exact:
        return exact[0]
    later = [f for f in view.frames if f.master_idx > start_master_idx]
    if later:
        return later[0]
    earlier = [f for f in view.frames if f.master_idx < start_master_idx]
    return earlier[-1] if earlier else None


def build_goals_block(goals: list[dict[str, Any]]) -> str:
    lines = []
    for k, g in enumerate(goals, start=1):
        anchor = str(g.get("anchor") or "").strip().replace("\n", " ")
        line = (f"Goal {k} [master ticks {g['start_master_idx']}-{g['end_master_idx']}] "
                f"instruction: {json.dumps(str(g.get('instruction') or ''))}")
        if anchor:
            line += f"  (anchor: {json.dumps(anchor[:200])})"
        lines.append(line)
    return "\n".join(lines)


def _tokens(usage: dict[str, Any] | None) -> int:
    if not isinstance(usage, dict):
        return 0
    return usage.get("total_tokens") or ((usage.get("prompt_tokens") or 0)
                                         + (usage.get("completion_tokens") or 0))


def run_unit(item: dict[str, Any], ctx: MethodContext) -> dict[str, Any]:
    """``item``: {"unit_id", "view": SegmentView, "goals": [rows], "narration": str}.
    Returns the input rows enriched with plan/plan_flags (order preserved)."""
    goals: list[dict[str, Any]] = item["goals"]
    view: SegmentView = item["view"]
    narration = str(item.get("narration") or "").strip()
    if not goals:
        return {"goals": [], "actual_tokens": 0}
    if not narration:
        raise RuntimeError(f"unit {item['unit_id']} has no describe narration sidecar")

    ordered = sorted(goals, key=lambda g: (int(g["start_master_idx"]), int(g["end_master_idx"])))
    frames = []
    labels = []
    for k, g in enumerate(ordered, start=1):
        f = goal_start_frame(view, int(g["start_master_idx"]))
        if f is None:
            raise RuntimeError(f"no view frame for goal {g.get('goal_id')} start {g['start_master_idx']}")
        frames.append(str(f.image))
        labels.append(f"Goal {k} start screen (master tick {f.master_idx}):")
    imgs = frames_to_data_urls(frames, target_height=ctx.vlm_frame_height,
                               jpeg_quality=ctx.jpeg_quality)

    prompt = ctx.prompts.render("plan", description=narration, n_goals=str(len(ordered)),
                                goals_block=build_goals_block(ordered))
    parsed, res = ctx.labeler.call_json_full(
        ctx.prompts.get("plan_system"), prompt, images=imgs, image_labels=labels,
        cache_path=ctx.cache_dir / "plan_from_prose.txt", no_cache=ctx.no_cache)

    by_goal: dict[int, str] = {}
    for entry in (parsed.get("plans", []) if isinstance(parsed, dict) else []):
        if isinstance(entry, dict):
            try:
                by_goal[int(entry["goal"])] = str(entry.get("plan") or "").strip()
            except (KeyError, TypeError, ValueError):
                continue

    n_flagged = 0
    for k, g in enumerate(ordered, start=1):
        plan = by_goal.get(k, "")
        flags = plan_flags(plan, str(g.get("instruction") or ""))
        g["plan"] = plan
        g["plan_flags"] = flags
        n_flagged += 1 if flags else 0

    return {
        "goals": goals,  # same objects, enriched in place; original order
        "n_plans": sum(1 for g in goals if g.get("plan")),
        "n_flagged": n_flagged,
        "finish_reason": res.finish_reason,
        "actual_tokens": _tokens(res.usage),
    }
