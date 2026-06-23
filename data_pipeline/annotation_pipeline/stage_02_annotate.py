#!/usr/bin/env python3
"""Stage 02 (redesign): two-pass hindsight instruction annotation.

The objective is NOT to describe the recording but to **recover the prompt a
user would have typed** to make a computer-use agent perform the observed
trajectory. Two passes, anchored on FRAME INDEX (each frame is labelled
``frame <N>``; after sampling/NO_OP-filtering wall-clock time is no longer exact,
but the kept-frame ``global_frame_idx`` is stable):

  PASS 1  (describe + segment) -> runs once over the whole post-NO_OP-cut clip
     (sub-sampled to an image budget). The VLM returns a list of ACTIVITIES,
     each with one or more FRAME-INDEX spans (interleaved activity allowed), a
     detailed *factual* description (no goal/intent), a per-span user_state
     (actively_working / idle_waiting), and onset/completion flags. Output
     ``activities.jsonl``.

  PASS 2  (hindsight instruction) -> for each activity, send the frames across
     all its spans + the pass-1 description, and write the user-prompt
     instruction (mixed intent level) + register variants. One trajectory is
     emitted PER CONTIGUOUS SPAN, sharing the activity's instruction/variants;
     idle-only spans fall out downstream (all-NO_OP trim in stage 03).

There is intentionally no verify/repair pass here — this is the raw output of
pass1+pass2; the independent judge (judge.py) is the quality measurement.

Output ``trajectories_raw.json`` is schema-compatible with stage 03 (each
trajectory has ``start_time_s/end_time_s/instruction/instruction_variants``;
times are derived from the span's first/last frame ``global_time_s``). Frame
indices (``start_frame_idx``/``end_frame_idx``) are carried for traceability.
Intermediate artifacts (activities, input transcript, raw responses) are written
for the inspector and free re-runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from annotation_pipeline import config, prompts
from annotation_pipeline.common import ensure_dir, read_jsonl, write_json, write_jsonl
from annotation_pipeline.keylog_transcript import build_transcript
from annotation_pipeline.labeler import Labeler, LabelerConfig
from annotation_pipeline.frames_render import (
    load_vlm_video_sources,
    records_in_index_span,
    render_frames,
    select_naming_frames,
)

# All prompt text lives in prompts.yaml (loaded via annotation_pipeline.prompts).
SYSTEM = prompts.get("system")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def pass1_prompt(transcript_text: str, n_frames: int) -> str:
    return prompts.render("pass1", transcript_text=transcript_text, n_frames=n_frames)


def pass2_prompt(spans_text: str, description: str, onset: str, completion: str,
                 transcript_text: str, n_frames: int) -> str:
    return prompts.render(
        "pass2", spans_text=spans_text, description=description,
        onset=onset, completion=completion,
        transcript_text=transcript_text, n_frames=n_frames,
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache(output_dir: Path, name: str) -> Path:
    """Stable per-call cache path. Reused on re-runs so unchanged calls never
    re-spend tokens. After editing a pass's prompt, invalidate just that pass
    with --refresh (e.g. --refresh pass2)."""
    return output_dir / "cache" / f"{name}.txt"


def refresh_cache(output_dir: Path, prefixes: list[str]) -> int:
    cdir = output_dir / "cache"
    if not cdir.is_dir() or not prefixes:
        return 0
    n = 0
    for f in cdir.glob("*.txt"):
        if any(f.name.startswith(p) for p in prefixes):
            f.unlink()
            n += 1
    return n


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def frame_labels(records: list[dict[str, Any]]) -> list[str]:
    """Per-frame text labels interleaved before each image: the stable kept-frame
    index. Pass 1 reports boundaries in these units."""
    return [f"frame {int(r['global_frame_idx'])}" for r in records]


def render_for(records, out_dir, args, vlm_video_by_segment) -> list[Path]:
    # Clean frames from the raw MP4; the frame index goes to the model as
    # interleaved text (frame_labels), not burned in.
    return render_frames(
        records, out_dir,
        jpeg_quality=args.jpeg_quality,
        video_by_segment=vlm_video_by_segment,
        target_height=args.vlm_frame_height,
    )


def _time_of_idx(by_idx: dict[int, dict[str, Any]], idx: int, default: float) -> float:
    r = by_idx.get(int(idx))
    return float(r["global_time_s"]) if r else default


# ---------------------------------------------------------------------------
# Pass 1 — describe + segment
# ---------------------------------------------------------------------------


def step_pass1(lab, args, frame_records, transcript, output_dir, vlm_video_by_segment):
    idx_min = int(frame_records[0]["global_frame_idx"])
    idx_max = int(frame_records[-1]["global_frame_idx"])

    sampled = select_naming_frames(frame_records, args.pass1_image_max)
    if len(sampled) < len(frame_records):
        print(f"  pass1: sub-sampled {len(frame_records)} kept frames -> {len(sampled)} images "
              f"(budget {args.pass1_image_max}); spans returned over the full index range "
              f"{idx_min}..{idx_max}")
    imgs = render_for(sampled, output_dir / "pass1_frames", args, vlm_video_by_segment)
    prompt = pass1_prompt(transcript.render(max_text_chars=8000), len(sampled))
    parsed = lab.call_json(
        SYSTEM, prompt, images=imgs, image_labels=frame_labels(sampled),
        cache_path=_cache(output_dir, "pass1"),
    )

    activities: list[dict[str, Any]] = []
    for a in parsed.get("activities", []):
        spans: list[dict[str, Any]] = []
        for sp in a.get("spans", []):
            try:
                sf = int(sp["start_frame"]); ef = int(sp["end_frame"])
            except (KeyError, TypeError, ValueError):
                continue
            if ef < sf:
                sf, ef = ef, sf
            sf = max(idx_min, min(sf, idx_max))
            ef = max(idx_min, min(ef, idx_max))
            spans.append({"start_frame": sf, "end_frame": ef,
                          "user_state": str(sp.get("user_state", ""))})
        if not spans:
            continue
        spans.sort(key=lambda s: (s["start_frame"], s["end_frame"]))
        activities.append({
            "id": a.get("id"),
            "app": str(a.get("app", "")),
            "spans": spans,
            "description": str(a.get("description", "")),
            "onset": str(a.get("onset", "unknown")),
            "completion": str(a.get("completion", "unknown")),
        })
    write_jsonl(output_dir / "activities.jsonl", activities)
    return activities


# ---------------------------------------------------------------------------
# Pass 2 — hindsight instruction (+ per-span trajectory emission)
# ---------------------------------------------------------------------------


def label_activity(lab, args, frame_records, transcript, act, tag, output_dir, vlm_video_by_segment):
    # Frames across ALL of the activity's spans, so the labeler sees the whole
    # (possibly interleaved) line of work before writing one shared instruction.
    span_records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for sp in act["spans"]:
        for r in records_in_index_span(frame_records, sp["start_frame"], sp["end_frame"]):
            gi = int(r["global_frame_idx"])
            if gi not in seen:
                seen.add(gi)
                span_records.append(r)
    span_records.sort(key=lambda r: int(r["global_frame_idx"]))
    if len(span_records) < 2:
        return None, span_records

    frames = select_naming_frames(span_records, args.label_image_max)
    imgs = render_for(frames, output_dir / "pass2_frames" / tag, args, vlm_video_by_segment)
    spans_text = "frames " + ", ".join(
        f"{s['start_frame']}–{s['end_frame']}" for s in act["spans"]
    )
    t0 = float(span_records[0]["global_time_s"]); t1 = float(span_records[-1]["global_time_s"])
    prompt = pass2_prompt(
        spans_text, act.get("description", ""), act.get("onset", "unknown"),
        act.get("completion", "unknown"), transcript.render(t0, t1, max_text_chars=2000), len(frames),
    )
    parsed = lab.call_json(SYSTEM, prompt, images=imgs, image_labels=frame_labels(frames),
                           cache_path=_cache(output_dir, f"pass2_{tag}"))
    return parsed, span_records


def step_pass2(lab, args, frame_records, transcript, activities, output_dir, vlm_video_by_segment):
    trajectories: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for ai, act in enumerate(activities):
        tag = f"{ai:04d}"
        parsed, _ = label_activity(lab, args, frame_records, transcript, act, tag, output_dir, vlm_video_by_segment)
        if not parsed:
            rejected.append({"activity": act, "reason": "too_few_frames_in_span"})
            continue
        instruction = str(parsed.get("instruction", "")).strip()
        if not instruction:
            rejected.append({"activity": act, "reason": "empty_instruction", "label": parsed})
            continue
        variants = [str(v).strip() for v in parsed.get("instruction_variants", []) if str(v).strip()]
        grounding = str(parsed.get("grounding", ""))

        # One trajectory per contiguous span, sharing the activity's instruction.
        for si, sp in enumerate(act["spans"]):
            recs = records_in_index_span(frame_records, sp["start_frame"], sp["end_frame"])
            if len(recs) < 2:
                continue
            trajectories.append({
                "start_time_s": round(float(recs[0]["global_time_s"]), 1),
                "end_time_s": round(float(recs[-1]["global_time_s"]), 1),
                "start_frame_idx": int(recs[0]["global_frame_idx"]),
                "end_frame_idx": int(recs[-1]["global_frame_idx"]),
                "instruction": instruction,
                "instruction_variants": variants,
                "description": act.get("description", ""),
                "grounding": grounding,
                "app": act.get("app", ""),
                "user_state": sp.get("user_state", ""),
                "onset": act.get("onset", "unknown"),
                "completion": act.get("completion", "unknown"),
                "activity_id": act.get("id"),
                "span_idx": si,
            })
    return trajectories, rejected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frame-records", type=Path, required=True)
    p.add_argument("--keylog", type=Path, required=True, help="msgpack keylog for this segment")
    p.add_argument("--manifest", type=Path, required=True, help="stage 00 manifest (render VLM frames from raw MP4s)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--segment-offset-s", type=float, default=0.0)
    # Pass 1
    p.add_argument("--pass1-image-max", type=int, default=96,
                   help="Max frames sent to the single pass-1 describe+segment call "
                        "(the whole clip is sub-sampled to this; spans still range over all kept frames).")
    # Pass 2
    p.add_argument("--label-image-max", type=int, default=config.DEFAULT_NAME_IMAGE_MAX,
                   help="Max frames sent per pass-2 label call (across the activity's spans).")
    # Frame render
    p.add_argument("--vlm-frame-height", type=int, default=config.DEFAULT_VLM_FRAME_HEIGHT)
    p.add_argument("--jpeg-quality", type=int, default=80)
    # Labeler
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--refresh", default=None,
                   help="Comma-separated cache prefixes to invalidate before running "
                        "(e.g. 'pass2' or 'pass1,pass2'); only those calls re-run.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    if args.refresh:
        n = refresh_cache(output_dir, [p.strip() for p in args.refresh.split(",") if p.strip()])
        print(f"refresh: invalidated {n} cached responses ({args.refresh})")
    frame_records = read_jsonl(args.frame_records)
    if not frame_records:
        raise RuntimeError(f"No frame records: {args.frame_records}")

    cfg = LabelerConfig.from_env(model=args.model, base_url=args.base_url, reasoning_effort=args.reasoning_effort)
    lab = Labeler(cfg)

    transcript = build_transcript(args.keylog, segment_offset_s=args.segment_offset_s)
    write_jsonl(output_dir / "input_transcript.jsonl",
                [{"t_s": round(e.t_s, 3), "t_end_s": round(e.t_end_s, 3), "kind": e.kind, **e.data}
                 for e in transcript.events])

    vlm_video_by_segment = load_vlm_video_sources(args.manifest)

    activities = step_pass1(lab, args, frame_records, transcript, output_dir, vlm_video_by_segment)
    trajectories, rejected = step_pass2(lab, args, frame_records, transcript, activities, output_dir, vlm_video_by_segment)

    n_spans = sum(len(a["spans"]) for a in activities)
    result = {
        "recording_id": str(frame_records[0]["recording_id"]),
        "annotation_source": "vlm_hindsight",
        "trajectories": sorted(trajectories, key=lambda t: (t["start_time_s"], t["end_time_s"])),
        "response_meta": {"model": cfg.model, "base_url": cfg.base_url,
                          "vlm_frame_height": args.vlm_frame_height},
    }
    write_json(output_dir / "trajectories_raw.json", result)
    write_jsonl(output_dir / "rejected.jsonl", rejected)
    write_json(output_dir / "stage02_summary.json", {
        "n_frames": len(frame_records),
        "n_pass1_activities": len(activities),
        "n_spans": n_spans,
        "n_active_spans": sum(1 for a in activities for s in a["spans"] if s.get("user_state") == "actively_working"),
        "n_idle_spans": sum(1 for a in activities for s in a["spans"] if s.get("user_state") == "idle_waiting"),
        "n_trajectories": len(trajectories),
        "n_rejected": len(rejected),
        "n_variants_total": sum(1 + len(t["instruction_variants"]) for t in trajectories),
    })
    print(f"activities={len(activities)} spans={n_spans} "
          f"trajectories={len(trajectories)} rejected={len(rejected)} -> {output_dir/'trajectories_raw.json'}")


if __name__ == "__main__":
    main()
