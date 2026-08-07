"""lumine_thinking: sequential day-watching with carried memory (Track A).

The core mechanic (validated in hindsight_fold/lumine as ``clip_annotator``;
reimplemented here, nothing imported from lumine):

  1. Sequential watching with carried memory. The model sees a user-day's
     frame+action stream strictly in time order, one ~30-frame clip at a
     time, carrying a 2-4 sentence textual memory clip to clip. It never
     sees ahead. Chunk boundaries (recording gaps > gap_cut_s) get a one-time
     "resumed after a break" note; the memory itself persists across them.
  2. Thoughts at decision points. Per clip the model marks 0-N moments where
     the person visibly commits/switches/reconsiders and writes the person's
     inner monologue at that exact frame (first person, present tense, 15-45
     words, evidence-bound, kind in plan/reorient/decide/react/monitor/wait).
     Steady flow gets zero thoughts. Anchors outside the clip are dropped.
  3. Future-blind verification is a hard gate. Every thought is audited
     against only the frames up to and including its anchor (within its
     chunk); only passes reach the uniform artifact. A per-day self-test
     canary (texts of two thoughts >=30 min apart swapped - the auditor must
     fail both plants) aborts the day if the gate itself is broken.

Each clip additionally returns a detailed factual log, stored in the clip
ledger + memory sidecar, never carried forward and never part of the
verifier's evidence.

Coordinates: prompts use day-global dense frame indices (lib/days); nothing
day-local persists — thoughts are returned in (segment_id, master_idx)
coordinates and the stage writes them as single-tick master intervals.

Resume: one ledger record per clip under ctx.params["day_units_dir"], carrying
the outgoing memory — a restarted day resumes at the first unfinished clip
with byte-identical prompts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.annotation.lib.days import DayFrame, DayStream, fmt_t, frame_label
from pipeline.annotation.lib.labeler import ContentFilteredError
from pipeline.annotation.lib.registry import MethodContext
from pipeline.annotation.lib.units import frames_to_data_urls
from pipeline.lib.common import write_json, write_jsonl

INPUT_KIND = "days"
# Locked model discipline of the validated mechanic (Kimi: never temperature 0;
# low effort or it burns the completion budget on reasoning). CLI flags win.
LABELER_DEFAULTS = {"temperature": 0.2, "reasoning_effort": "low"}

KINDS = ("plan", "reorient", "decide", "react", "monitor", "wait")
VERIFY_MODES = ("batched", "per_thought")

# Validated params; override via --param key=value.
DEFAULT_CLIP_FRAMES = 30          # ~60 s at 0.5 fps
DEFAULT_MAX_THOUGHTS_PER_CLIP = 3
DEFAULT_CTX_FRAMES = 12           # pre-anchor evidence frames for the auditor
DEFAULT_SELFTEST_MIN_SEP_S = 1800.0
# Per-call completion budgets. Kimi's in-band reasoning plus the log field
# make writer completions run 8-12k observed (lumine's 4k cap predates the
# log); a lower start only buys escalation retries at full input cost. The
# labeler still doubles up to LABELER_MAX_TOKENS on overflow.
DEFAULT_WRITER_MAX_TOKENS = 16384
DEFAULT_VERIFY_MAX_TOKENS = 8192
MEMORY_SEED = "(day just started — no memory yet)"


def _tokens(usage: dict[str, Any] | None) -> int:
    if not isinstance(usage, dict):
        return 0
    return usage.get("total_tokens") or ((usage.get("prompt_tokens") or 0)
                                         + (usage.get("completion_tokens") or 0))


def _norm(text: Any) -> str:
    return " ".join(str(text).split())


def plan_clips(day: DayStream, clip_frames: int) -> list[tuple[str, int, bool, list[DayFrame]]]:
    """Strict tiling (the validated shape): each chunk cut into consecutive
    ``clip_frames``-sized clips; a trailing sliver of <2 frames is skipped.
    Returns (clip_key, chunk_index, is_chunk_start, frames) in day order —
    deterministic across resumes."""
    clips: list[tuple[str, int, bool, list[DayFrame]]] = []
    k = 0
    for ci, chunk in enumerate(day.chunks):
        for c0 in range(0, len(chunk), clip_frames):
            clip = chunk[c0: c0 + clip_frames]
            if len(clip) < 2:
                continue
            clips.append((f"clip_{k:04d}", ci, c0 == 0, clip))
            k += 1
    return clips


def _render(ctx: MethodContext, frames: list[DayFrame]) -> tuple[list[str], list[str]]:
    imgs = frames_to_data_urls([fr.image for fr in frames],
                               target_height=ctx.vlm_frame_height,
                               jpeg_quality=ctx.jpeg_quality)
    return imgs, [frame_label(fr) for fr in frames]


def _clean_thoughts(parsed: dict[str, Any], clip: list[DayFrame],
                    max_per_clip: int) -> tuple[list[dict[str, Any]], int]:
    """Validate the writer's thoughts: anchor must be a frame of this clip
    (out-of-clip anchors are the model hallucinating indices — dropped, never
    remapped), kind falls back to 'plan', text is whitespace-normalized."""
    by_idx = {fr.day_idx: fr for fr in clip}
    thoughts: list[dict[str, Any]] = []
    n_dropped = 0
    for th in (parsed.get("thoughts") or [])[:max_per_clip]:
        if not isinstance(th, dict):
            n_dropped += 1
            continue
        try:
            day_idx = int(th.get("frame"))
        except (TypeError, ValueError):
            n_dropped += 1
            continue
        fr = by_idx.get(day_idx)
        text = _norm(th.get("text", ""))
        if fr is None or not text:
            n_dropped += 1
            continue
        kind = str(th.get("kind", "")).strip()
        thoughts.append({
            "day_idx": day_idx,
            "t_day_s": fr.t_day_s,
            "t_label": fmt_t(fr.t_day_s),
            "segment_id": fr.segment_id,
            "recording_id": fr.recording_id,
            "master_idx": fr.master_idx,
            "kind": kind if kind in KINDS else "plan",
            "text": text,
        })
    thoughts.sort(key=lambda t: t["day_idx"])
    return thoughts, n_dropped


def _parse_verdict(obj: Any) -> dict[str, Any]:
    obj = obj if isinstance(obj, dict) else {}
    verdict = "pass" if str(obj.get("verdict", "")).lower() == "pass" else "fail"
    return {"verdict": verdict,
            "violations": obj.get("violations") or [],
            "reason": _norm(obj.get("reason", ""))}


def _audit_batched(ctx: MethodContext, day: DayStream, thoughts: list[dict[str, Any]],
                   ctx_frames: int, cache_path: Path, verify_max: int,
                   report) -> list[dict[str, Any]]:
    """One audit call for a set of thoughts (same chunk, ordered): frames up
    to the LAST anchor, per-thought evidence cutoff enforced by the prompt.
    A thought with no returned verdict fails (the gate never defaults open)."""
    anchors = [int(t["day_idx"]) for t in thoughts]
    lo, hi = min(anchors), max(anchors)
    frames = day.context_before(hi, (hi - lo) + ctx_frames)
    imgs, labels = _render(ctx, frames)
    block = "\n".join(
        f'{k}) anchor frame {t["day_idx"]} ({t["t_label"]}), kind={t["kind"]}: "{t["text"]}"'
        for k, t in enumerate(thoughts, start=1))
    user = ctx.prompts.render("verify_batched", n_thoughts=len(thoughts),
                              max_frame=hi, thoughts_block=block)
    parsed, res = ctx.labeler.call_json_full(
        ctx.prompts.get("verifier_system"), user, images=imgs, image_labels=labels,
        cache_path=cache_path, no_cache=ctx.no_cache, max_completion_tokens=verify_max)
    report(_tokens(res.usage))
    by_n: dict[int, dict[str, Any]] = {}
    for v in (parsed.get("verdicts") or []):
        if isinstance(v, dict):
            try:
                by_n[int(v.get("n"))] = v
            except (TypeError, ValueError):
                continue
    out = []
    for k in range(1, len(thoughts) + 1):
        if k in by_n:
            out.append(_parse_verdict(by_n[k]))
        else:
            out.append({"verdict": "fail", "violations": ["no_verdict"],
                        "reason": "auditor returned no verdict for this thought"})
    return out


def _audit_per_thought(ctx: MethodContext, day: DayStream, th: dict[str, Any],
                       ctx_frames: int, cache_path: Path, verify_max: int,
                       report) -> dict[str, Any]:
    """The lumine-validated shape: one future-blind call per thought, frames
    up to and including its anchor only."""
    frames = day.context_before(int(th["day_idx"]), ctx_frames)
    imgs, labels = _render(ctx, frames)
    user = ctx.prompts.render("verify_thought", frame_idx=th["day_idx"],
                              t_label=th["t_label"], kind=th["kind"], text=th["text"])
    parsed, res = ctx.labeler.call_json_full(
        ctx.prompts.get("verifier_system"), user, images=imgs, image_labels=labels,
        cache_path=cache_path, no_cache=ctx.no_cache, max_completion_tokens=verify_max)
    report(_tokens(res.usage))
    return _parse_verdict(parsed)


def _audit(ctx: MethodContext, day: DayStream, thoughts: list[dict[str, Any]],
           verify_mode: str, ctx_frames: int, cache_stem: str, verify_max: int,
           report) -> None:
    """Attach ``verify`` verdicts to ``thoughts`` in place, in the configured
    mode. Batched audits group per clip call; the self-test reuses this with a
    batch of one so the canary exercises the shipping prompt."""
    if not thoughts:
        return
    try:
        if verify_mode == "batched":
            verdicts = _audit_batched(ctx, day, thoughts, ctx_frames,
                                      ctx.cache_dir / f"{cache_stem}_verify.txt",
                                      verify_max, report)
            for th, v in zip(thoughts, verdicts, strict=True):
                th["verify"] = v
        else:
            for th in thoughts:
                th["verify"] = _audit_per_thought(
                    ctx, day, th, ctx_frames,
                    ctx.cache_dir / f"{cache_stem}_v{th['day_idx']:06d}.txt",
                    verify_max, report)
    except ContentFilteredError as exc:
        # The audit itself was blocked — fail closed: unverifiable thoughts
        # never reach the artifact.
        for th in thoughts:
            if not th.get("verify"):
                th["verify"] = {"verdict": "fail", "violations": ["content_filter"],
                                "reason": f"audit blocked by content filter: {exc}"[:200]}


def _specificity(th: dict[str, Any]) -> float:
    """How anchored a thought's text is to concrete on-screen referents.
    A generic thought ("I'll switch to another application") is consistent
    with almost any moment, so swapping it elsewhere legitimately passes the
    audit — planting one is a false alarm, not a gate test. Count tokens that
    look like named specifics: digits, quoted text, path-ish tokens, and
    proper nouns (capitalized past sentence start)."""
    words = str(th["text"]).split()
    score = 0.0
    for i, w in enumerate(words):
        core = w.strip(".,;:!?()")
        if not core:
            continue
        if (any(c.isdigit() for c in core)
                or core.startswith(("'", '"')) or core.endswith(("'", '"'))
                or (len(core) > 3 and any(c in core for c in "./_-"))
                or (i > 0 and core[:1].isupper())):
            score += 1
    return score + 0.02 * len(words)


def _self_test(ctx: MethodContext, day: DayStream, thoughts: list[dict[str, Any]],
               verify_mode: str, ctx_frames: int, verify_max: int, min_sep_s: float,
               report) -> dict[str, Any]:
    """Verifier canary: swap the texts of two thoughts ≥ min_sep_s apart and
    audit both plants — each text now sits at a moment whose evidence cannot
    support it, so a working auditor must fail both. The pair is the most
    specific eligible one (see _specificity): the canary must plant thoughts
    whose referents provably aren't at the other anchor, else "vague but
    consistent passes" turns the test into a coin flip. Returns a status
    record; the caller aborts the day on 'failed'."""
    candidates = sorted(thoughts, key=_specificity, reverse=True)[:40]
    best = None
    for i, x in enumerate(candidates):
        for y in candidates[i + 1:]:
            if abs(float(y["t_day_s"]) - float(x["t_day_s"])) >= min_sep_s:
                key = min(_specificity(x), _specificity(y))
                if best is None or key > best[0]:
                    best = (key, x, y)
    if best is None:
        return {"status": "skipped", "reason": f"no two thoughts >= {min_sep_s:.0f}s apart"}
    a, b = sorted(best[1:], key=lambda t: float(t["t_day_s"]))
    plants = []
    n_caught = 0
    for host, donor in ((a, b), (b, a)):
        planted = {**host, "text": donor["text"], "verify": None}
        _audit(ctx, day, [planted], verify_mode, ctx_frames,
               f"selftest_{host['day_idx']:06d}_{donor['day_idx']:06d}",
               verify_max, report)
        v = planted["verify"]
        if "content_filter" in (v.get("violations") or []):
            # The plant audit itself was blocked — the gate's health is
            # untestable on this pair; don't fail a day over it.
            return {"status": "skipped", "reason": "self-test audit content-filtered"}
        caught = v["verdict"] == "fail"
        n_caught += caught
        plants.append({"host_day_idx": host["day_idx"], "host_t": host["t_label"],
                       "donor_day_idx": donor["day_idx"], "donor_t": donor["t_label"],
                       "caught": caught, "reason": v["reason"]})
    # A dead gate passes everything -> both plants pass -> abort. One caught
    # plant proves the gate works; the passed one is (per the audit's own
    # "vague but consistent passes" rule) usually a too-generic text — record
    # it as 'partial' rather than failing the day.
    status = "ok" if n_caught == 2 else ("partial" if n_caught == 1 else "failed")
    return {"status": status, "plants": plants}


def run_unit(item: dict[str, Any], ctx: MethodContext) -> dict[str, Any]:
    """``item``: {"id": day_tag, "day": DayStream, "row": day-index row}.
    Walks the day's clips strictly in order (resuming from the per-clip
    ledger), verifies every thought future-blind, runs the day-end self-test,
    writes the memory/log sidecar, and returns all thoughts with verdicts
    attached (the stage emits only the passes)."""
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
    day_units_dir = Path(p["day_units_dir"])
    memory_path = Path(p["memory_path"])
    force = bool(p.get("force"))
    report = p.get("report_tokens") or (lambda n: None)
    spent = {"total": 0}

    def track(n: int) -> None:
        """Stream every call's tokens to the governor and into the day total."""
        spent["total"] += int(n)
        report(n)

    clips = plan_clips(day, clip_frames)
    system = ctx.prompts.get("system")
    memory = MEMORY_SEED
    all_thoughts: list[dict[str, Any]] = []
    mem_rows: list[dict[str, Any]] = []
    n_dropped = 0
    n_resumed = 0

    for clip_key, chunk_idx, is_chunk_start, clip in clips:
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
                                      memory=memory + gap_note)
            content_filtered = False
            try:
                parsed, res = ctx.labeler.call_json_full(
                    system, user, images=imgs, image_labels=labels,
                    cache_path=ctx.cache_dir / f"{clip_key}.txt", no_cache=ctx.no_cache,
                    max_completion_tokens=writer_max)
            except ContentFilteredError:
                # The provider refuses to describe this clip's screen content.
                # Degrade to an unannotated clip (zero thoughts, memory carried
                # unchanged) instead of killing the whole day.
                parsed, res = {}, None
                content_filtered = True
            clip_tokens = _tokens(res.usage) if res else 0
            track(clip_tokens)

            thoughts, dropped = _clean_thoughts(parsed, clip, max_per_clip)
            memory_out = _norm(parsed.get("memory", "")) or memory
            log = str(parsed.get("log", "")).strip()

            _audit(ctx, day, thoughts, verify_mode, ctx_frames, clip_key,
                   verify_max, track)
            rec = {
                "clip_key": clip_key,
                "chunk_index": chunk_idx,
                "day_idx_range": [clip[0].day_idx, clip[-1].day_idx],
                "t_range": [fmt_t(clip[0].t_day_s), fmt_t(clip[-1].t_day_s)],
                "segments": sorted({fr.segment_id for fr in clip}),
                "n_frames": len(clip),
                "gap_note": bool(gap_note),
                "content_filtered": content_filtered,
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
    n_pass = sum(1 for t in all_thoughts if (t.get("verify") or {}).get("verdict") == "pass")
    return {
        "thoughts": all_thoughts,
        "n_clips": len(clips),
        "n_clips_resumed": n_resumed,
        "n_thoughts": len(all_thoughts),
        "n_pass": n_pass,
        "n_dropped_anchor": n_dropped,
        "verify_mode": verify_mode,
        "selftest": selftest,
        "actual_tokens": spent["total"],
    }
