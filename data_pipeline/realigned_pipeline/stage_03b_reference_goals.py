#!/usr/bin/env python3
"""Stage 03b (goal reference): attach an EXTERNAL goal-annotation set onto the
stage-03 sampled frames, keyed on the shared ``(segment_id, source_frame_idx)``
axis.

Motivation. A goal-annotation dataset (here: a ``hindsight_fold`` canonical SFT
build) defines goals as time windows on a per-recording/day **cumulative** clock
(``t_start``/``t_end`` running into the thousands of seconds -- it includes the
paused-out idle the recorder collapses). Our sampler (stage 03) indexes frames on
a per-segment **video-PTS** clock that resets to ~0 each segment. Those two clocks
do NOT line up, so you cannot join goals to our frames by timestamp.

But both pipelines decode the SAME source mp4s and both record ``source_frame_idx``
(the original mp4 frame number, local to a segment) and share the SAME
``segment_id`` (``<recording_id>_seg<NNNN>``). ``source_frame_idx`` is
decode-deterministic and alignment-independent -- realignment only moves ACTIONS,
never which pixels a source frame is. So ``(segment_id, source_frame_idx)`` is an
exact, clock-free bridge between the two datasets. This stage uses it.

What it does, per goal (a row of the external ``chat.jsonl``):
  * resolve the goal's frames -- its ``image_paths`` (``ar://…/images.array_record#N``
    into the external per-segment frame store) -- to their ``source_frame_idx`` via
    the external day-level ``frame_records.jsonl`` (which carries
    ``image_path`` + ``segment_id`` + ``source_frame_idx`` for every frame),
  * take the goal's ``source_frame_idx`` span ``[lo, hi]`` in its segment,
  * select OUR sampled frames for that ``segment_id`` whose ``source_frame_idx``
    falls in ``[lo, hi]`` -- our (denser) frames covering the goal window, each with
    its ``master_record_index`` / ``ar://`` image / realigned ``action``,
  * emit one ``goal_frame_index.jsonl`` row: the goal metadata + the matched OUR
    frames + a match certificate (exact-source-match count, coverage).

Why our frames, not theirs: the external ``action`` was binned from the RAW
(pre-realignment) keylog; OUR stage-03 frames carry the REALIGNED action. So the
product is goals (theirs) on realigned frames/actions (ours), joined on the frame
identity -- gate downstream trust on ``alignment_status``.

Two clocks never need reconciling because we match FRAMES, not time. The external
cumulative clock is only read implicitly, through the goal's own ``image_paths``.

Inputs:
  --sample-dir     a stage-03 (sample_frames) --output-dir: sample_index.jsonl +
                   clips/<seg>/stage_01/frame_records.jsonl. The frames to annotate.
  --goals-chat     the external goal set's chat.jsonl (one goal per line, with
                   recording_id, sid, goal_id, instruction, long_ref/long_text,
                   kind, status, start/end_time_s, image_paths, day_tag, split).
  --hindsight-days-root  the external ``.../pipeline_runs/days`` dir holding
                   <day_tag>/frame_records.jsonl (the source_frame_idx provenance).
                   Auto-derived from the goals' ar:// image paths when omitted.

Outputs (under --output-dir):
  goal_frame_index.jsonl   one row per goal -> our matched frames + certificate.
  verify_conversations.jsonl  a spot-check subset as GOAL-CONDITIONED conversations
                   (stage-04 chat schema) -- open directly in
                   visualize_frame_records.py to eyeball goal<->frame correctness.
  verify_source_conversations.jsonl  the SAME sampled goals as the ORIGINAL external
                   conversations (their frames + raw actions), keyed to line up with
                   verify_conversations.jsonl -- open both to cross-reference our
                   realigned frames against theirs without loading the full chat.jsonl.
  summary.json     match-status counts + coverage stats.
  manifest.json    artifact marker.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Make the ``realigned_pipeline`` package importable when run directly
# (mirrors the other stages' PYTHONPATH setup).
DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.lib.common import (  # noqa: E402
    ensure_dir,
    read_jsonl,
    write_json,
    write_jsonl,
)
from realigned_pipeline.lib.image_store import (  # noqa: E402
    is_arrayrecord_image_uri,
    parse_arrayrecord_image_uri,
)


def _read_goals(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Stream goals from the external chat.jsonl, stopping at ``limit``. The file
    can be hundreds of MB (full messages per goal), so we never slurp it whole."""
    goals: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            goals.append(json.loads(line))
            if limit is not None and len(goals) >= limit:
                break
    return goals


def _uri_key(uri: str) -> tuple[str, int] | None:
    """Normalize an ``ar://…/images.array_record#idx`` ref to ``(shard, idx)`` so
    the external chat.jsonl image and the external frame_records image match even
    if their strings differ trivially. Non-ar refs -> None."""
    if not is_arrayrecord_image_uri(uri):
        return None
    try:
        shard, idx = parse_arrayrecord_image_uri(uri)
    except ValueError:
        return None
    return (str(shard), int(idx))


def _days_root_from_goal(goal: dict[str, Any]) -> Path | None:
    """Derive ``.../pipeline_runs/days`` from a goal's first ar:// image path
    (``…/days/<day_tag>/frames/<sid>/images.array_record#N``)."""
    for uri in goal.get("image_paths") or []:
        key = _uri_key(str(uri))
        if key is None:
            continue
        shard = Path(key[0])  # …/days/<day_tag>/frames/<sid>/images.array_record
        # parents: [0]=<sid>dir [1]=frames [2]=<day_tag> [3]=days
        try:
            days = shard.parents[3]
        except IndexError:
            return None
        if days.name == "days":
            return days
        return days  # tolerate a differently named leaf; caller can override
    return None


def _load_day_provenance(day_fr_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Build ``{(shard, idx) -> {segment_id, source_frame_idx, local_time_s}}`` from
    an external day-level ``frame_records.jsonl`` (every frame carries image_path +
    segment_id + source_frame_idx). This is the source_frame_idx provenance the
    goal's ar:// image refs resolve through."""
    prov: dict[tuple[str, int], dict[str, Any]] = {}
    for r in read_jsonl(day_fr_path):
        ref = r.get("image_path") or r.get("image")
        if not ref:
            continue
        key = _uri_key(str(ref))
        if key is None:
            continue
        prov[key] = {
            "segment_id": r.get("segment_id"),
            "source_frame_idx": r.get("source_frame_idx"),
            "local_time_s": r.get("local_time_s"),
        }
    return prov


def _my_frames_in_source_span(
    frame_records: list[dict[str, Any]], src_lo: int, src_hi: int
) -> list[dict[str, Any]]:
    """OUR sampled frames whose ``source_frame_idx`` falls in ``[src_lo, src_hi]``,
    in source order. These are our (denser) frames covering the goal window."""
    out = [
        r
        for r in frame_records
        if r.get("source_frame_idx") is not None
        and src_lo <= int(r["source_frame_idx"]) <= src_hi
    ]
    out.sort(key=lambda r: int(r["source_frame_idx"]))
    return out


def _extract_context(goal: dict[str, Any]) -> str | None:
    """Pull the self-compaction ``[CONTEXT]…[/CONTEXT]`` block out of a goal's first
    user turn. The external (selfcompact) builder fuses it into the first user
    message's text as ``"<instruction>\n\n[CONTEXT]…[/CONTEXT]"`` -- but ONLY on
    non-first chunks of a split goal (chunk_idx>0). First/single chunks carry no
    block -> None. We capture it separately here so stage-04 can re-fuse it onto OUR
    first frame, reproducing the prompt the external set was built with (otherwise the
    whole point of selfcompact -- the rolling progress summary -- is dropped)."""
    for msg in goal.get("messages") or []:
        if msg.get("role") != "user":
            continue
        # first user turn: its (single) text part is "<instruction>\n\n[CONTEXT]…".
        for part in msg.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "text":
                txt = part.get("text") or ""
                i = txt.find("[CONTEXT]")
                return txt[i:].strip() if i != -1 else None
        return None  # first user turn had no text part
    return None


def _join_instruction_context(instruction: str | None, context: str | None) -> str | None:
    """Fuse the goal instruction and its self-compaction ``[CONTEXT]`` block into one
    first-turn text, reproducing the external layout ``"<instruction>\n\n[CONTEXT]…"``.
    Either may be absent (goal-free -> None; first/single chunk -> instruction only)."""
    parts = [p.strip() for p in (instruction, context) if p and str(p).strip()]
    return "\n\n".join(parts) if parts else None


def _goal_message_block(
    instruction: str | None, context: str | None, frames: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One interleaved screenshot->action conversation for a goal's matched frames.
    Mirrors stage_04_build_conversations' canonical chat schema: the instruction TEXT
    (fused with the self-compaction ``[CONTEXT]`` block when present) before the image
    on the first user turn, image-only after, one assistant turn per frame carrying
    its (realigned) action."""
    messages: list[dict[str, Any]] = []
    first_text = _join_instruction_context(instruction, context)
    for idx, fr in enumerate(frames):
        content: list[dict[str, Any]] = []
        if idx == 0 and first_text:
            content.append({"type": "text", "text": first_text})
        content.append({"type": "image", "image": str(fr["image_path"])})
        messages.append({"role": "user", "content": content})
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": str(fr.get("action", ""))}]}
        )
    return messages


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sample-dir",
        type=Path,
        required=True,
        help="A stage-03 (sample_frames) --output-dir: sample_index.jsonl + "
        "clips/<seg>/stage_01/frame_records.jsonl. The frames to annotate.",
    )
    p.add_argument(
        "--goals-chat",
        type=Path,
        required=True,
        help="External goal set chat.jsonl (one goal per line).",
    )
    p.add_argument(
        "--hindsight-days-root",
        type=Path,
        default=None,
        help="External .../pipeline_runs/days dir (holds <day_tag>/frame_records.jsonl). "
        "Auto-derived from the goals' ar:// image paths if omitted.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--min-frames",
        type=int,
        default=1,
        help="Skip goals matching fewer than this many of OUR frames (default 1).",
    )
    p.add_argument(
        "--verify-sample",
        type=int,
        default=40,
        help="Emit up to this many matched goals as goal-conditioned conversations "
        "for visual spot-checking (evenly spaced across the corpus). 0 = all.",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Process only the first N goals (debug)."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    # OUR sampled dataset: segment_id -> index row (frame_records path, fps, alignment).
    sample_index = read_jsonl(args.sample_dir / "sample_index.jsonl")
    if not sample_index:
        raise SystemExit(f"no sample_index.jsonl under {args.sample_dir}")
    my_by_seg = {str(r["segment_id"]): r for r in sample_index}
    target_fps = next((r.get("target_fps") for r in sample_index if r.get("target_fps")), None)
    print(f"[goal-ref] {len(my_by_seg)} of our segments; target_fps={target_fps}", flush=True)

    goals = _read_goals(args.goals_chat, args.limit)
    if not goals:
        raise SystemExit(f"no goals in {args.goals_chat}")

    # Original external goal rows by sample_id, so verification can emit a PAIRED
    # source-side spot-check (the same goals on THEIR frames) alongside ours.
    goal_by_sid = {str(g.get("sample_id")): g for g in goals if g.get("sample_id")}

    # Group goals by day so each external day frame_records.jsonl is read once.
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in goals:
        by_day[str(g.get("day_tag") or "")].append(g)
    print(f"[goal-ref] {len(goals)} goals across {len(by_day)} days", flush=True)

    # Cache OUR per-segment frame_records (loaded lazily; small per segment).
    my_frames_cache: dict[str, list[dict[str, Any]]] = {}

    def my_frames(segment_id: str) -> list[dict[str, Any]] | None:
        if segment_id in my_frames_cache:
            return my_frames_cache[segment_id]
        row = my_by_seg.get(segment_id)
        fr_path = Path(row["frame_records"]) if row and row.get("frame_records") else None
        recs = read_jsonl(fr_path) if fr_path and fr_path.exists() else None
        my_frames_cache[segment_id] = recs  # may be None -> "no_segment"
        return recs

    counts: Counter = Counter()
    index_rows: list[dict[str, Any]] = []

    for day_tag in sorted(by_day):
        day_goals = by_day[day_tag]
        # Locate the external day frame_records (source_frame_idx provenance).
        days_root = args.hindsight_days_root or _days_root_from_goal(day_goals[0])
        day_fr_path = (days_root / day_tag / "frame_records.jsonl") if days_root else None
        prov = (
            _load_day_provenance(day_fr_path)
            if day_fr_path and day_fr_path.exists()
            else {}
        )
        if not prov:
            print(f"  WARN {day_tag}: no external frame_records ({day_fr_path})", flush=True)

        for g in day_goals:
            base = {
                "sample_id": g.get("sample_id"),
                "goal_id": g.get("goal_id"),
                "recording_id": g.get("recording_id"),
                "day_tag": day_tag,
                "split": g.get("split"),
                "instruction": g.get("instruction"),
                # self-compaction: the rolling [CONTEXT] summary fused into the goal's
                # first user turn (non-first chunks only; None otherwise) + chunk
                # provenance, so stage-04 can rebuild the trained prompt and group chunks.
                "context": _extract_context(g),
                "chunk_idx": g.get("chunk_idx"),
                "n_chunks": g.get("n_chunks"),
                "parent_sample_id": g.get("parent_sample_id"),
                "context_tokens": g.get("context_tokens"),
                "long_ref": g.get("long_ref"),
                "long_text": g.get("long_text"),
                "kind": g.get("kind"),
                "status": g.get("status"),
                "coll_start_time_s": g.get("start_time_s"),
                "coll_end_time_s": g.get("end_time_s"),
                "coll_n_frames": g.get("n_frames") or len(g.get("image_paths") or []),
            }

            # Resolve the goal's external frames -> (segment_id, source_frame_idx).
            resolved = [prov[k] for k in (
                _uri_key(str(u)) for u in (g.get("image_paths") or [])
            ) if k is not None and k in prov]
            coll_src = sorted({int(p["source_frame_idx"]) for p in resolved
                               if p.get("source_frame_idx") is not None})
            seg_ids = {p["segment_id"] for p in resolved if p.get("segment_id")}

            if not coll_src or not seg_ids:
                counts["unresolved_coll_frames"] += 1
                index_rows.append({**base, "match_status": "unresolved_coll_frames",
                                   "n_my_frames": 0})
                continue
            # A goal maps to one segment in this dataset (its own sid).
            segment_id = sorted(seg_ids)[0]
            base["segment_id"] = segment_id

            recs = my_frames(segment_id)
            if recs is None:
                counts["no_segment"] += 1
                index_rows.append({**base, "match_status": "no_segment", "n_my_frames": 0})
                continue

            src_lo, src_hi = coll_src[0], coll_src[-1]
            matched = _my_frames_in_source_span(recs, src_lo, src_hi)
            if len(matched) < args.min_frames:
                # No OUR frames in the goal's source span. Usually the goal window is
                # all-idle: our stage-03 dropped it (noop_mode=none drops every idle
                # frame; empty segment == whole segment idle), whereas the external
                # set keeps idle-run head/tail. Surface our segment status so it's
                # distinguishable from a real miss.
                counts["no_frames"] += 1
                index_rows.append({**base, "match_status": "no_frames", "n_my_frames": len(matched),
                                   "my_segment_status": (my_by_seg.get(segment_id) or {}).get("status"),
                                   "coll_source_frame_idx_lo": src_lo,
                                   "coll_source_frame_idx_hi": src_hi})
                continue

            my_src = {int(r["source_frame_idx"]) for r in matched}
            n_exact = len(set(coll_src) & my_src)
            counts["ok"] += 1
            index_rows.append({
                **base,
                "match_status": "ok",
                "alignment_status": (my_by_seg.get(segment_id) or {}).get("alignment_status"),
                "target_fps": (my_by_seg.get(segment_id) or {}).get("target_fps"),
                "coll_source_frame_idx_lo": src_lo,
                "coll_source_frame_idx_hi": src_hi,
                "n_my_frames": len(matched),
                # certificate: how many external goal frames have an EXACT
                # source-frame twin in ours (the rest sit between our denser bins).
                "n_exact_source_match": n_exact,
                "exact_source_match_frac": round(n_exact / len(coll_src), 4),
                "my_frames": [
                    {
                        "master_record_index": r.get("master_record_index"),
                        "source_frame_idx": r.get("source_frame_idx"),
                        "local_time_s": r.get("local_time_s"),
                        "image_path": r.get("image_path"),
                        "action": r.get("action"),
                    }
                    for r in matched
                ],
            })

    write_jsonl(out_dir / "goal_frame_index.jsonl", index_rows)

    # --- verification: a spread of matched goals as goal-conditioned conversations,
    # directly openable in visualize_frame_records.py (auto-detected: filename has
    # "conversations"). Each conversation is one goal's OUR frames + the goal text. ---
    ok_rows = [r for r in index_rows if r.get("match_status") == "ok"]
    if args.verify_sample and len(ok_rows) > args.verify_sample:
        step = len(ok_rows) / args.verify_sample
        sample = [ok_rows[int(i * step)] for i in range(args.verify_sample)]
    else:
        sample = ok_rows
    verify_convs = [
        {
            "conversation_id": r.get("sample_id"),
            "segment_id": r.get("segment_id"),
            "recording_id": r.get("recording_id"),
            "instruction": r.get("instruction"),
            "context": r.get("context"),
            "chunk_idx": r.get("chunk_idx"),
            "n_chunks": r.get("n_chunks"),
            "goal_conditioned": True,
            "target_fps": r.get("target_fps") or target_fps,
            "alignment_status": r.get("alignment_status"),
            "split": r.get("split"),
            "goal_id": r.get("goal_id"),
            "long_ref": r.get("long_ref"),
            "kind": r.get("kind"),
            "status": r.get("status"),
            "n_exact_source_match": r.get("n_exact_source_match"),
            "coll_n_frames": r.get("coll_n_frames"),
            "n_frames": r.get("n_my_frames"),
            "messages": _goal_message_block(
                r.get("instruction"), r.get("context"), r.get("my_frames") or []
            ),
        }
        for r in sample
    ]
    write_jsonl(out_dir / "verify_conversations.jsonl", verify_convs)

    # Paired ORIGINAL-side spot-check: the external goal conversations for EXACTLY
    # the sampled goals, in the same order. Both files then hold the same goals and
    # load instantly in the viewer (no need to open the full multi-hundred-MB
    # chat.jsonl split -- and the goal you sampled may live in a DIFFERENT split).
    # We inject our matched segment_id as the key so each goal lines up with its
    # verify_conversations twin in the viewer's segment dropdown; origin marks it
    # as theirs (external frames + RAW, pre-realignment actions).
    verify_source = []
    for r in sample:
        g = goal_by_sid.get(str(r.get("sample_id")))
        if g is None:
            continue
        src = dict(g)
        src["segment_id"] = r.get("segment_id")        # share the dropdown key with ours
        src["conversation_id"] = r.get("sample_id")
        src["goal_conditioned"] = True
        src["alignment_status"] = r.get("alignment_status")
        src["split"] = r.get("split")
        src["origin"] = "external_source"
        verify_source.append(src)
    write_jsonl(out_dir / "verify_source_conversations.jsonl", verify_source)

    n_ok = counts.get("ok", 0)
    exact_fracs = [r["exact_source_match_frac"] for r in ok_rows]
    my_frame_total = sum(r["n_my_frames"] for r in ok_rows)
    summary = {
        "n_goals": len(goals),
        "match_status_counts": dict(counts),
        "n_ok": n_ok,
        "n_my_frames_total": my_frame_total,
        "mean_exact_source_match_frac": round(sum(exact_fracs) / len(exact_fracs), 4)
        if exact_fracs else None,
        "n_verify_conversations": len(verify_convs),
        "n_verify_source_conversations": len(verify_source),
        "sample_dir": str(args.sample_dir),
        "goals_chat": str(args.goals_chat),
        "target_fps": target_fps,
    }
    write_json(out_dir / "summary.json", summary)
    write_json(
        out_dir / "manifest.json",
        {
            "artifact_type": "juergen_goal_frame_index",
            "schema_version": 1,
            "goal_frame_index": "goal_frame_index.jsonl",
            "verify_conversations": "verify_conversations.jsonl",
            "verify_source_conversations": "verify_source_conversations.jsonl",
            **summary,
        },
    )
    print(f"[goal-ref] done: {dict(counts)} | {my_frame_total} frames matched "
          f"| verify={len(verify_convs)} (+{len(verify_source)} source) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
