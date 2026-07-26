"""lumine_goal_boundaries: goal-END verification for TERMINATE supervision.

Enrichment pass over a lumine_thinking_goals artifact's per-day sidecars
(``units/<day>/goals_active.jsonl`` + ``memory/<day>.jsonl``): ~47% of fold
goal ends fall into no-goal stretches and spot-checks show mid-activity /
distraction ends — supervising TERMINATE on every goal_t_end would teach the
model to stop mid-task. Per goal span this method:

  1. JUDGES the end. One call on the last ``n_end_frames`` of the span plus
     the first ``n_after_frames`` after it (never across a recording gap),
     with the boundary clip's factual log as context -> strict JSON
     {completed, confidence, evidence, final_thought}. Uncertainty is forced
     downward by prompt AND parser (fail-closed: non-bool completed -> false,
     unknown confidence -> low, a completed verdict without evidence or
     final_thought is demoted to low). Downstream (stage 04's 'verified'
     terminate-boundary mode) trusts ONLY completed && confidence=high.
  2. MINES A NEAR-MISS negative. For completed+high spans with >= 2 clips,
     one call on the clip BEFORE the completion clip -> {not_done_reason,
     next_step_thought} (future-blind: what is visibly still missing and the
     immediate next action — the hard negative one clip earlier). Single-clip
     spans get no near-miss: the preceding clip belongs to another goal or a
     no-goal stretch, so "not done yet" would be unsupported there.

INPUT_KIND is "days": the stage rebuilds the SAME day streams the producing
run used (same --filter-dir / --fps / --tz / --gap-cut-s), so the sidecars'
day-global frame indices address the rebuilt stream directly. Every span's
clips are checked against their recorded ``t_range`` before any call — a
mismatch (wrong filter/fps) aborts the day loudly instead of judging the
wrong frames. The input artifact comes in as ``--param goals_dir=<dir>``
(same pattern as lumine_thinking_goals' ``goals_fold_dir``).

Output: ``boundaries/<day>.jsonl`` — one row per goal span (schema in
``assemble_row``) — plus the framework's usual units/ ledger (one record per
span: the resume level below whole days), calls/ cache, progress.jsonl and
manifest. ``goals.jsonl`` stays empty (this method emits no thoughts).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from realigned_pipeline.annotation.lib.days import DayStream, fmt_t, frame_label
from realigned_pipeline.annotation.lib.labeler import ContentFilteredError
from realigned_pipeline.annotation.lib.registry import MethodContext
from realigned_pipeline.annotation.lib.units import frames_to_data_urls
from realigned_pipeline.annotation.methods.lumine_thinking.annotator import _norm, _tokens
from realigned_pipeline.lib.common import read_jsonl, write_json, write_jsonl

INPUT_KIND = "days"
# Same locked Kimi discipline as the producing run (never temperature 0; low
# effort or the completion budget burns on reasoning). CLI flags win.
LABELER_DEFAULTS = {"temperature": 0.2, "reasoning_effort": "low"}

CONFIDENCES = ("high", "low")

# Defaults, not dogma — override via --param key=value.
DEFAULT_N_END_FRAMES = 3        # last frames of the span shown to the judge
DEFAULT_N_AFTER_FRAMES = 2      # first frames after the span (same chunk only)
DEFAULT_NEAR_MISS_FRAMES = 0    # 0 = the whole preceding clip; else its last N
DEFAULT_MAX_SPANS_PER_DAY = 0   # 0 = all (smoke/validation cap)
DEFAULT_JUDGE_MAX_TOKENS = 8192
DEFAULT_NEAR_MISS_MAX_TOKENS = 8192

_NO_LOG = "(no log recorded for this clip)"


def _render(ctx: MethodContext, frames: list) -> tuple[list[str], list[str]]:
    imgs = frames_to_data_urls([fr.image for fr in frames],
                               target_height=ctx.vlm_frame_height,
                               jpeg_quality=ctx.jpeg_quality)
    return imgs, [frame_label(fr) for fr in frames]


def _model_name(ctx: MethodContext) -> str:
    cfg = getattr(ctx.labeler, "config", None)
    return getattr(cfg, "model", None) or "env"


# ---------------------------------------------------------------------------
# Input sidecars (the producing lumine_thinking_goals artifact)
# ---------------------------------------------------------------------------


def load_goal_clips(goals_dir: Path, day_tag: str) -> list[dict[str, Any]]:
    """The day's per-clip active-goal rows ({clip_key, day_idx_range, t_range,
    goal_id, goal_text, goal_t_start, goal_t_end, goal_long_ref, ...}).
    Missing file -> [] (the day had no goal sidecar; nothing to judge)."""
    p = goals_dir / "units" / day_tag / "goals_active.jsonl"
    return read_jsonl(p) if p.is_file() else []


def load_clip_logs(goals_dir: Path, day_tag: str) -> dict[str, str]:
    """{clip_key -> factual log text} from the day's memory sidecar."""
    p = goals_dir / "memory" / f"{day_tag}.jsonl"
    if not p.is_file():
        return {}
    return {str(r["clip_key"]): str(r.get("log") or "") for r in read_jsonl(p)}


def group_goal_spans(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group a day's goals_active rows into GOAL SPANS: one span per distinct
    goal_id (fold ids are day-local ints), clips in day order. A goal
    interrupted and resumed contributes ONE span whose boundary is its LAST
    clip — the clip nearest goal_t_end, which is the terminate boundary; the
    interruption gap is not an end. No-goal clips (goal_id null) separate
    spans and are never judged."""
    spans: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    for r in rows:
        gid = r.get("goal_id")
        if gid is None:
            continue
        if gid not in spans:
            spans[gid] = {
                "goal_id": gid,
                "goal_text": r.get("goal_text"),
                "goal_t_start": r.get("goal_t_start"),
                "goal_t_end": r.get("goal_t_end"),
                "goal_long_ref": r.get("goal_long_ref"),
                "clips": [],
            }
            order.append(gid)
        spans[gid]["clips"].append(r)
    for gid in order:
        spans[gid]["clips"].sort(key=lambda c: int(c["day_idx_range"][0]))
    # Day order by boundary clip (== insertion order unless goals interleave).
    return sorted((spans[g] for g in order),
                  key=lambda s: int(s["clips"][-1]["day_idx_range"][1]))


# ---------------------------------------------------------------------------
# Frame index math (pure; unit-tested)
# ---------------------------------------------------------------------------


def chunk_end_idx(day: DayStream, day_idx: int) -> int:
    """The last day index of the chunk containing ``day_idx`` (after-frames
    never cross a recording gap — house rule from lib/days)."""
    for chunk in day.chunks:
        if chunk[0].day_idx <= day_idx <= chunk[-1].day_idx:
            return chunk[-1].day_idx
    raise KeyError(f"frame {day_idx} not in any chunk of {day.day_tag}")


def judge_frame_indices(span_start: int, span_end: int, n_end: int, n_after: int,
                        chunk_end: int) -> tuple[list[int], list[int]]:
    """(end_indices, after_indices) for the judge call: the last ``n_end``
    frames of the span [span_start, span_end] (clamped so a short span never
    reaches before its own start) and the first ``n_after`` frames after it,
    clamped to the chunk end (empty when the span ends the chunk)."""
    if span_end < span_start:
        raise ValueError(f"span_end {span_end} < span_start {span_start}")
    end_idxs = list(range(max(span_start, span_end - n_end + 1), span_end + 1))
    after_idxs = list(range(span_end + 1, min(chunk_end, span_end + n_after) + 1))
    return end_idxs, after_idxs


def check_clip_alignment(row: dict[str, Any], day: DayStream) -> None:
    """Assert the rebuilt day stream addresses the sidecar's clip: the frames
    at its recorded day_idx_range must carry its recorded t_range. A mismatch
    means the run was pointed at a different filter/fps/tz than the producing
    run — abort loudly, never judge the wrong frames."""
    i0, i1 = (int(x) for x in row["day_idx_range"])
    if not (0 <= i0 <= i1 < len(day.frames)):
        raise ValueError(
            f"{day.day_tag}/{row.get('clip_key')}: day_idx_range [{i0}, {i1}] outside the "
            f"rebuilt day stream ({len(day.frames)} frames) — rerun with the producing "
            "run's --filter-dir/--fps/--fps-mode/--tz/--gap-cut-s")
    want = row.get("t_range")
    got = [fmt_t(day.frames[i0].t_day_s), fmt_t(day.frames[i1].t_day_s)]
    if want and got != list(want):
        raise ValueError(
            f"{day.day_tag}/{row.get('clip_key')}: rebuilt day stream misaligned with the "
            f"input sidecar (t_range {got} != {list(want)}) — rerun with the producing "
            "run's --filter-dir/--fps/--fps-mode/--tz/--gap-cut-s")


# ---------------------------------------------------------------------------
# Response parsing (fail-closed; unit-tested)
# ---------------------------------------------------------------------------


def parse_judgment(parsed: dict[str, Any]) -> dict[str, Any]:
    """Validate the judge's JSON, biased CLOSED: anything not a clean
    ``completed: true`` is false; any confidence not literally "high" is low;
    a completed verdict missing its evidence or final_thought is demoted to
    low (the downstream consumer trusts only completed && high, so demotion
    == exclusion). ``final_thought`` is cleared on not-completed."""
    parsed = parsed if isinstance(parsed, dict) else {}
    raw = parsed.get("completed")
    completed = raw if isinstance(raw, bool) else str(raw).strip().lower() == "true"
    conf = str(parsed.get("confidence", "")).strip().lower()
    confidence = conf if conf in CONFIDENCES else "low"
    evidence = _norm(parsed.get("evidence", ""))
    final_thought = _norm(parsed.get("final_thought", ""))
    if completed and (not evidence or not final_thought):
        confidence = "low"
    if not completed:
        final_thought = ""
    return {"completed": completed, "confidence": confidence,
            "evidence": evidence, "final_thought": final_thought}


def parse_near_miss(parsed: dict[str, Any]) -> dict[str, str] | None:
    """Validate the near-miss JSON; a response without a usable
    next_step_thought yields None (partial negatives never ship)."""
    parsed = parsed if isinstance(parsed, dict) else {}
    reason = _norm(parsed.get("not_done_reason", ""))
    thought = _norm(parsed.get("next_step_thought", ""))
    if not thought or not reason:
        return None
    return {"not_done_reason": reason, "next_step_thought": thought}


def assemble_row(span: dict[str, Any], judgment: dict[str, Any],
                 near_miss: dict[str, Any] | None, *, model: str,
                 ts: str) -> dict[str, Any]:
    """One boundaries/<day>.jsonl row for a goal span. The boundary clip is
    the span's last clip; ``near_miss`` (when present) carries its own
    clip_key/day_idx_range plus the mined negative."""
    boundary = span["clips"][-1]
    return {
        "goal_id": span["goal_id"],
        "goal_text": span["goal_text"],
        "goal_t_start": span["goal_t_start"],
        "goal_t_end": span["goal_t_end"],
        "goal_long_ref": span["goal_long_ref"],
        "clip_key": boundary["clip_key"],
        "day_idx_range": boundary["day_idx_range"],
        "t_range": boundary.get("t_range"),
        "n_clips_in_span": len(span["clips"]),
        "completed": judgment["completed"],
        "confidence": judgment["confidence"],
        "evidence": judgment["evidence"],
        "final_thought": judgment["final_thought"],
        "content_filtered": bool(judgment.get("content_filtered")),
        "near_miss": near_miss,
        "model": model,
        "ts": ts,
    }


# ---------------------------------------------------------------------------
# run_unit
# ---------------------------------------------------------------------------


def run_unit(item: dict[str, Any], ctx: MethodContext) -> dict[str, Any]:
    """``item``: {"id": day_tag, "day": DayStream, "row": day-index row}.
    Judges every goal span of the day (resuming from the per-span ledger),
    mines near-miss negatives for the verified completions, and writes the
    day's ``boundaries/<day>.jsonl``. Returns no thoughts — goals.jsonl stays
    empty; the boundaries sidecar is the artifact's payload."""
    day: DayStream = item["day"]
    p = ctx.params
    goals_dir = p.get("goals_dir")
    if not goals_dir:
        raise ValueError("lumine_goal_boundaries needs --param goals_dir=<the producing "
                         "lumine_thinking_goals artifact dir> (units/<day>/goals_active.jsonl "
                         "+ memory/<day>.jsonl sidecars)")
    n_end = int(p.get("n_end_frames", DEFAULT_N_END_FRAMES))
    n_after = int(p.get("n_after_frames", DEFAULT_N_AFTER_FRAMES))
    near_miss_frames = int(p.get("near_miss_frames", DEFAULT_NEAR_MISS_FRAMES))
    max_spans = int(p.get("max_spans_per_day", DEFAULT_MAX_SPANS_PER_DAY))
    judge_max = int(p.get("judge_max_tokens", DEFAULT_JUDGE_MAX_TOKENS))
    near_max = int(p.get("near_miss_max_tokens", DEFAULT_NEAR_MISS_MAX_TOKENS))
    day_units_dir = Path(p["day_units_dir"])
    force = bool(p.get("force"))
    report = p.get("report_tokens") or (lambda n: None)
    spent = {"total": 0}

    def track(n: int) -> None:
        spent["total"] += int(n)
        report(n)

    # units/<day>/ sits at <out>/units/<day>/ -> the boundaries sidecar tree
    # is its sibling <out>/boundaries/ (same shape as the stage's memory/).
    boundaries_dir = day_units_dir.parents[1] / "boundaries"
    boundaries_dir.mkdir(parents=True, exist_ok=True)

    clip_rows = load_goal_clips(Path(goals_dir), day.day_tag)
    logs = load_clip_logs(Path(goals_dir), day.day_tag)
    spans = group_goal_spans(clip_rows)
    if max_spans:
        spans = spans[:max_spans]

    system = ctx.prompts.get("system")
    nm_system = ctx.prompts.get("near_miss_system")
    model = _model_name(ctx)
    rows: list[dict[str, Any]] = []
    n_resumed = 0

    for si, span in enumerate(spans):
        span_key = f"span_{si:03d}_g{span['goal_id']}"
        rec_path = day_units_dir / f"{span_key}.json"
        if rec_path.exists() and not force:
            rows.append(json.loads(rec_path.read_text())["row"])
            n_resumed += 1
            continue

        boundary = span["clips"][-1]
        check_clip_alignment(boundary, day)
        span_start = int(span["clips"][0]["day_idx_range"][0])
        span_end = int(boundary["day_idx_range"][1])
        end_idxs, after_idxs = judge_frame_indices(
            span_start, span_end, n_end, n_after, chunk_end_idx(day, span_end))
        frames = [day.frames[i] for i in end_idxs + after_idxs]
        imgs, labels = _render(ctx, frames)
        for j in range(len(end_idxs), len(frames)):
            labels[j] = f"(AFTER the goal stretch) {labels[j]}"
        user = ctx.prompts.render(
            "judge", goal=_norm(span["goal_text"]),
            log=logs.get(str(boundary["clip_key"]), "") or _NO_LOG,
            n_end=len(end_idxs), n_after=len(after_idxs),
            t_end_label=(boundary.get("t_range") or ["?", "?"])[1])
        content_filtered = False
        try:
            parsed, res = ctx.labeler.call_json_full(
                system, user, images=imgs, image_labels=labels,
                cache_path=ctx.cache_dir / f"{span_key}_judge.txt", no_cache=ctx.no_cache,
                max_completion_tokens=judge_max)
        except ContentFilteredError:
            # The judge itself was blocked — fail CLOSED: an unjudgeable end
            # is not a verified completion.
            parsed, res = {}, None
            content_filtered = True
        track(_tokens(res.usage) if res else 0)
        judgment = parse_judgment(parsed)
        judgment["content_filtered"] = content_filtered

        near_miss = None
        nm_status = "not_eligible"
        if judgment["completed"] and judgment["confidence"] == "high":
            if len(span["clips"]) < 2:
                # Single-clip span: the preceding clip belongs to another
                # goal / no-goal stretch — no clean negative exists.
                nm_status = "single_clip_span"
            else:
                nm_clip = span["clips"][-2]
                check_clip_alignment(nm_clip, day)
                a, b = (int(x) for x in nm_clip["day_idx_range"])
                if near_miss_frames > 0:
                    a = max(a, b - near_miss_frames + 1)
                imgs, labels = _render(ctx, day.frames[a: b + 1])
                user = ctx.prompts.render(
                    "near_miss", goal=_norm(span["goal_text"]),
                    log=logs.get(str(nm_clip["clip_key"]), "") or _NO_LOG)
                try:
                    parsed_nm, res_nm = ctx.labeler.call_json_full(
                        nm_system, user, images=imgs, image_labels=labels,
                        cache_path=ctx.cache_dir / f"{span_key}_nearmiss.txt",
                        no_cache=ctx.no_cache, max_completion_tokens=near_max)
                except ContentFilteredError:
                    parsed_nm, res_nm = {}, None
                track(_tokens(res_nm.usage) if res_nm else 0)
                mined = parse_near_miss(parsed_nm)
                if mined is None:
                    nm_status = "unusable_response" if res_nm else "content_filtered"
                else:
                    nm_status = "ok"
                    near_miss = {"clip_key": nm_clip["clip_key"],
                                 "day_idx_range": nm_clip["day_idx_range"],
                                 "t_range": nm_clip.get("t_range"),
                                 **mined}

        ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        row = assemble_row(span, judgment, near_miss, model=model, ts=ts)
        write_json(rec_path, {
            "span_key": span_key,
            "judge_frames": {"end": end_idxs, "after": after_idxs},
            "judge_finish": res.finish_reason if res else "content_filter",
            "near_miss_status": nm_status,
            "row": row,
        })
        rows.append(row)

    write_jsonl(boundaries_dir / f"{day.day_tag}.jsonl", rows)
    n_completed = sum(1 for r in rows if r["completed"])
    n_high = sum(1 for r in rows if r["completed"] and r["confidence"] == "high")
    return {
        "thoughts": [],  # no goal rows — boundaries/ is the payload
        "n_spans": len(spans),
        "n_spans_resumed": n_resumed,
        "n_completed": n_completed,
        "n_completed_high": n_high,
        "n_near_miss": sum(1 for r in rows if r["near_miss"]),
        "n_content_filtered": sum(1 for r in rows if r["content_filtered"]),
        "actual_tokens": spent["total"],
    }
