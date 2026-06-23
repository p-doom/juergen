#!/usr/bin/env python3
"""Stage 02 (redesign): hindsight instruction annotation.

Replaces the old segment->name->verify two-pass. The objective is NOT to
describe the recording but to **recover the prompt a user would have typed** to
make a computer-use agent perform the observed trajectory.

Four steps, per segment:

  A. PERCEIVE  -> a fine-grained textual *timeline* of what happens, built from
     dense frames FUSED with the exact keylog transcript (typed text, chords,
     clicks, app switches). Describing, not boundary-drawing.
  B. SEGMENT   -> cut the timeline into goal-coherent intervals (one achievable
     user intent each); tight starts; split non-monotonic "do->undo->redo" spans
     keeping the final achieving sub-span; never bundle unrelated tasks.
  C. LABEL     -> for each interval, write the achieved goal AS A USER PROMPT at
     mixed intent level (+ a couple of varied-register paraphrases). Hindsight:
     the goal is whatever end-state the trajectory reached, so the trajectory
     satisfies it by construction.
  D. VERIFY+REPAIR -> grounded check (achieved / monotonic / start-achievable /
     reads-like-a-user-prompt). On failure, try to TRIM to the achieving
     monotonic sub-interval and re-label once before discarding (recovers yield).

Output ``trajectories_raw.json`` is schema-compatible with stage 03
(``annotation_source`` starts with ``vlm``; each trajectory has
``start_time_s/end_time_s/instruction/verified/verify_checks`` plus
``instruction_variants``). Intermediate artifacts (timeline, intervals, the
input transcript, raw responses) are written for the review report and free
re-runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from annotation_pipeline import config, prompts
from annotation_pipeline.common import ensure_dir, read_jsonl, write_json, write_jsonl
from annotation_pipeline.keylog_transcript import Transcript, build_transcript
from annotation_pipeline.labeler import Labeler, LabelerConfig
# Reuse the (pure) frame rendering + sampling utilities from the legacy module.
from annotation_pipeline.frames_render import (
    evenly,
    load_vlm_video_sources,
    render_frames,
    sample_window_frames,
    segment_windows,
    select_naming_frames,
)

# ---------------------------------------------------------------------------
# Shared framing
# ---------------------------------------------------------------------------

# All prompt text lives in prompts.yaml (loaded via annotation_pipeline.prompts).
SYSTEM = prompts.get("system")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _cache(output_dir: Path, name: str) -> Path:
    """Stable per-step cache path. Reused on re-runs so unchanged steps never
    re-spend tokens. After editing a step's prompt, invalidate just that step
    with --refresh (e.g. --refresh verify) to re-run only those calls."""
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


def perceive_prompt(win_start: float, win_end: float, transcript_text: str, n_frames: int) -> str:
    return prompts.render(
        "perceive",
        win_start=f"{win_start:.1f}", win_end=f"{win_end:.1f}",
        transcript_text=transcript_text, n_frames=n_frames,
    )


def segment_prompt(timeline_text: str, transcript_text: str, span: tuple[float, float]) -> str:
    return prompts.render(
        "segment",
        timeline_text=timeline_text, transcript_text=transcript_text,
        span0=f"{span[0]:.1f}", span1=f"{span[1]:.1f}",
    )


def label_prompt(
    start_s: float, end_s: float, achieved_state: str, transcript_text: str, n_frames: int
) -> str:
    return prompts.render(
        "label",
        start_s=f"{start_s:.1f}", end_s=f"{end_s:.1f}",
        achieved_state=achieved_state, transcript_text=transcript_text, n_frames=n_frames,
    )


def verify_prompt(instruction: str, start_s: float, end_s: float, n_frames: int,
                  transcript_text: str) -> str:
    return prompts.render(
        "verify",
        instruction=repr(instruction), transcript_text=transcript_text,
        start_s=f"{start_s:.1f}", end_s=f"{end_s:.1f}", n_frames=n_frames,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def frames_in_span(frame_records: list[dict[str, Any]], start_s: float, end_s: float) -> list[dict[str, Any]]:
    return [r for r in frame_records if start_s - 0.5 <= float(r["global_time_s"]) <= end_s + 0.5]


def select_verify_frames(frames: list[dict[str, Any]], max_images: int) -> list[dict[str, Any]]:
    """Like select_naming_frames but tail-biased: completion/streamed results
    land near the end, so spend ~60% of the budget on the last third."""
    n = len(frames)
    if n <= max_images:
        return frames
    picks: set[int] = {0, n - 1}
    tail = list(range(max(1, (2 * n) // 3), n))
    budget = max_images - len(picks)
    picks |= set(evenly(tail, min(len(tail), max(1, int(budget * 0.6)))))
    rest = [i for i in range(n) if i not in picks]
    picks |= set(evenly(rest, max(0, max_images - len(picks))))
    return [frames[i] for i in sorted(picks)]


def render_for(records, out_dir, args, vlm_video_by_segment) -> list[Path]:
    # Clean frames from the raw MP4; timestamps go to the model as interleaved
    # text (frame_labels), not burned in.
    return render_frames(
        records, out_dir,
        jpeg_quality=args.jpeg_quality,
        video_by_segment=vlm_video_by_segment,
        target_height=args.vlm_frame_height,
    )


def frame_labels(records: list[dict[str, Any]]) -> list[str]:
    """Per-frame text labels interleaved before each image in the VLM request."""
    return [f"original_t={float(r['global_time_s']):.1f}s" for r in records]


# Hard reject gate: a trajectory is kept only if ALL of these pass. NOTE
# boundary_tight is deliberately NOT here — as a hard conjunctive boolean a
# skeptical labeler marks it false on almost everything (it nuked yield 22->7
# with no judge-agreement gain), so loose boundaries instead drive the
# repair-trim (see needs_repair) and are fixed rather than discarded.
VERIFY_AXES = ("achieved", "monotonic", "grounded",
               "start_achievable", "user_prompt_register")


def accept(checks: dict[str, Any]) -> bool:
    return all(bool(checks.get(a)) for a in VERIFY_AXES)


def needs_repair(checks: dict[str, Any]) -> bool:
    """Trim+re-label when a hard axis fails OR the boundary is just loose."""
    return (not accept(checks)) or (not checks.get("boundary_tight"))


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_a_timeline(lab, args, frame_records, transcript, output_dir, vlm_video_by_segment):
    t_min = float(frame_records[0]["global_time_s"])
    t_max = float(frame_records[-1]["global_time_s"])
    windows = segment_windows(t_min, t_max, args.perceive_window_s, args.perceive_overlap_s)
    events: list[dict[str, Any]] = []
    for wi, (ws, we) in enumerate(windows):
        in_win = [r for r in frame_records if ws <= float(r["global_time_s"]) < we]
        if not in_win:
            continue
        sampled = sample_window_frames(in_win, ws, we, args.perceive_image_max)
        imgs = render_for(sampled, output_dir / "perceive_frames" / f"win_{wi:04d}", args, vlm_video_by_segment)
        ttext = transcript.render(ws, we, max_text_chars=400)
        prompt = perceive_prompt(ws, we, ttext, len(sampled))
        parsed = lab.call_json(
            SYSTEM, prompt, images=imgs, image_labels=frame_labels(sampled),
            cache_path=_cache(output_dir, f"perceive_{wi:04d}"),
        )
        for e in parsed.get("events", []):
            try:
                e["t_start"] = max(ws, min(float(e["t_start"]), we))
                e["t_end"] = max(ws, min(float(e["t_end"]), we))
            except (KeyError, TypeError, ValueError):
                continue
            e["window"] = wi
            events.append(e)
    events.sort(key=lambda e: (e["t_start"], e["t_end"]))
    write_jsonl(output_dir / "timeline.jsonl", events)
    return events


def timeline_text(events: list[dict[str, Any]], max_events: int = 400) -> str:
    lines = [
        f"[{e['t_start']:7.1f}-{e['t_end']:.1f}s] {e.get('app','?')}: {e.get('observation','')}"
        for e in events[:max_events]
    ]
    return "\n".join(lines) if lines else "(empty timeline)"


def step_b_segment(lab, args, events, transcript, span, output_dir):
    prompt = segment_prompt(timeline_text(events), transcript.render(max_text_chars=4000), span)
    parsed = lab.call_json(SYSTEM, prompt, cache_path=_cache(output_dir, "segment"))
    intervals = []
    for iv in parsed.get("intervals", []):
        try:
            s = float(iv["start_time_s"]); e = float(iv["end_time_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if e - s < args.min_segment_s:
            continue
        intervals.append({
            "start_time_s": round(max(span[0], s), 1),
            "end_time_s": round(min(span[1], e), 1),
            "achieved_state": str(iv.get("achieved_state", "")),
            "monotonic_flag": bool(iv.get("monotonic", True)),
            "rationale": str(iv.get("rationale", "")),
        })
    intervals.sort(key=lambda iv: (iv["start_time_s"], iv["end_time_s"]))
    write_jsonl(output_dir / "intervals.jsonl", intervals)
    return intervals


def label_interval(lab, args, frame_records, transcript, iv, tag, output_dir, vlm_video_by_segment):
    s, e = iv["start_time_s"], iv["end_time_s"]
    in_span = frames_in_span(frame_records, s, e)
    if len(in_span) < 2:
        return None, None
    frames = select_naming_frames(in_span, args.label_image_max)
    imgs = render_for(frames, output_dir / "label_frames" / tag, args, vlm_video_by_segment)
    prompt = label_prompt(s, e, iv.get("achieved_state", ""), transcript.render(s, e, max_text_chars=1500), len(frames))
    parsed = lab.call_json(SYSTEM, prompt, images=imgs, image_labels=frame_labels(frames),
                           cache_path=_cache(output_dir, f"label_{tag}"))
    return parsed, imgs


def verify_interval(lab, args, frame_records, transcript, instruction, s, e, tag, output_dir, vlm_video_by_segment):
    # Extend the tail so a streamed/async result lands in-frame, and bias frame
    # selection toward the end where completion is visible.
    in_span = frames_in_span(frame_records, s, e + args.verify_tail_extend_s)
    frames = select_verify_frames(in_span, args.verify_image_max)
    imgs = render_for(frames, output_dir / "verify_frames" / tag, args, vlm_video_by_segment)
    prompt = verify_prompt(instruction, s, e, len(frames), transcript.render(s, e, max_text_chars=1500))
    checks = lab.call_json(SYSTEM, prompt, images=imgs, image_labels=frame_labels(frames),
                           cache_path=_cache(output_dir, f"verify_{tag}"))
    return checks


def step_cd(lab, args, frame_records, transcript, intervals, output_dir, vlm_video_by_segment):
    trajectories, rejected = [], []
    for i, iv in enumerate(intervals):
        tag = f"{i:04d}"
        parsed, _ = label_interval(lab, args, frame_records, transcript, iv, tag, output_dir, vlm_video_by_segment)
        if not parsed:
            rejected.append({"interval": iv, "reason": "too_few_frames_in_span"})
            continue
        instruction = str(parsed.get("instruction", "")).strip()
        if not instruction:
            rejected.append({"interval": iv, "reason": "empty_instruction", "label": parsed})
            continue
        variants = [str(v).strip() for v in parsed.get("instruction_variants", []) if str(v).strip()]
        s, e = iv["start_time_s"], iv["end_time_s"]

        checks = verify_interval(lab, args, frame_records, transcript, instruction, s, e, tag, output_dir, vlm_video_by_segment)
        repaired = False
        # Verify-and-repair: trim to the cleaner sub-interval and re-label once,
        # whenever a hard axis fails OR the boundary is merely loose.
        rep = checks.get("repair") if isinstance(checks, dict) else None
        if needs_repair(checks) and isinstance(rep, dict):
            try:
                rs = max(s, float(rep["start_time_s"])); re_ = min(e, float(rep["end_time_s"]))
            except (KeyError, TypeError, ValueError):
                rs, re_ = s, e
            if re_ - rs >= args.min_segment_s and (rs, re_) != (s, e):
                iv2 = {**iv, "start_time_s": round(rs, 1), "end_time_s": round(re_, 1)}
                parsed2, _ = label_interval(lab, args, frame_records, transcript, iv2, f"{tag}_r", output_dir, vlm_video_by_segment)
                if parsed2 and str(parsed2.get("instruction", "")).strip():
                    instruction = str(parsed2["instruction"]).strip()
                    variants = [str(v).strip() for v in parsed2.get("instruction_variants", []) if str(v).strip()]
                    parsed = parsed2
                    s, e = round(rs, 1), round(re_, 1)
                    checks = verify_interval(lab, args, frame_records, transcript, instruction, s, e, f"{tag}_r", output_dir, vlm_video_by_segment)
                    repaired = True

        verified = accept(checks)
        traj = {
            "start_time_s": s, "end_time_s": e,
            "instruction": instruction,
            "instruction_variants": variants,
            "achieved_state": str(parsed.get("achieved_state", "")),
            "grounding": str(parsed.get("grounding", "")),
            "verified": verified,
            "verify_checks": checks,
            "repaired": repaired,
            "monotonic_flag": iv.get("monotonic_flag", True),
            "rationale": iv.get("rationale", ""),
        }
        if verified:
            trajectories.append(traj)
        else:
            rejected.append({"interval": iv, "reason": "verify_failed", "trajectory": traj})
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
    # Step A
    p.add_argument("--perceive-window-s", type=float, default=120.0)
    p.add_argument("--perceive-overlap-s", type=float, default=20.0)
    p.add_argument("--perceive-image-max", type=int, default=30)
    # Step B/C/D
    p.add_argument("--min-segment-s", type=float, default=8.0)
    p.add_argument("--label-image-max", type=int, default=config.DEFAULT_NAME_IMAGE_MAX)
    p.add_argument("--verify-image-max", type=int, default=config.DEFAULT_NAME_IMAGE_MAX)
    p.add_argument("--verify-tail-extend-s", type=float, default=8.0,
                   help="Extend the verify frame window past the interval end to capture streamed/async results.")
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
                        "(e.g. 'verify' or 'label,verify'); only those steps re-run.")
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
    span = (float(frame_records[0]["global_time_s"]), float(frame_records[-1]["global_time_s"]))

    events = step_a_timeline(lab, args, frame_records, transcript, output_dir, vlm_video_by_segment)
    intervals = step_b_segment(lab, args, events, transcript, span, output_dir)
    trajectories, rejected = step_cd(lab, args, frame_records, transcript, intervals, output_dir, vlm_video_by_segment)

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
        "n_timeline_events": len(events),
        "n_intervals": len(intervals),
        "n_trajectories": len(trajectories),
        "n_verified": sum(1 for t in trajectories if t["verified"]),
        "n_repaired": sum(1 for t in trajectories if t.get("repaired")),
        "n_rejected": len(rejected),
        "n_variants_total": sum(1 + len(t["instruction_variants"]) for t in trajectories),
    })
    print(f"timeline={len(events)} intervals={len(intervals)} "
          f"trajectories={len(trajectories)} rejected={len(rejected)} -> {output_dir/'trajectories_raw.json'}")


if __name__ == "__main__":
    main()
