#!/usr/bin/env python3
"""Stage 02 (redesign): two-pass hindsight instruction annotation.

The objective is NOT to describe the recording but to **recover the prompt a
user would have typed** to make a computer-use agent perform the observed
trajectory. Two passes, anchored on FRAME INDEX (each frame is labelled
``frame <N>``; after sampling/NO_OP-filtering wall-clock time is no longer exact,
but the kept-frame ``global_frame_idx`` is stable):

  PASS 1  (describe + segment) -> runs over fixed-duration windows of the
     post-NO_OP-cut clip (5 minutes by default, all kept frames in each window).
     The VLM returns a list of ACTIVITIES,
     each with one or more FRAME-INDEX spans (interleaved activity allowed), a
     detailed *factual* description (no goal/intent), a per-span user_state
     (actively_working / idle_waiting), and onset/completion flags. Output
     ``activities.jsonl``.

  PASS 2  (hindsight instruction) -> for each activity, send the frames across
     all its spans + the pass-1 description, and write a candidate user-prompt
     instruction (mixed intent level) + register variants. Output
     ``pass2_labels.jsonl`` (per-activity labels, NOT final trajectories).

  PASS 3  (merge into goals) -> TEXT ONLY (no frames): given each activity's
     description + candidate instruction, the VLM groups activities that are
     really ONE user goal (e.g. one task worked across several panes/apps, or a
     request + the wait for it) vs. separate goals. Each goal becomes ONE
     trajectory spanning the union of its members' frames, with one merged
     instruction. Output ``goals.jsonl``; final trajectories in
     ``trajectories_raw.json``.

There is intentionally no verify/repair pass here — this is the raw output of
pass1->pass2->pass3 (the independent judge.py can be run separately).

Output ``trajectories_raw.json`` feeds stage 03 (each trajectory has
``start_frame_idx/end_frame_idx/instruction/instruction_variants``; the whole
pipeline is anchored on frame index — wall-clock seconds are not emitted).
Intermediate artifacts (activities, input transcript, raw responses) are written
for the inspector and free re-runs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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


def pass3_prompt(activities_text_str: str) -> str:
    return prompts.render("pass3", activities_text=activities_text_str)


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


def parallel_map(fn, items: list, concurrency: int) -> list:
    """Map fn over items, order-preserving. Labeler calls are independent stateless
    HTTP requests, so this just fires up to ``concurrency`` of them at once."""
    if not items:
        return []
    n = max(1, int(concurrency))
    if n == 1 or len(items) == 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(n, len(items))) as ex:
        return list(ex.map(fn, items))


def render_for(records, out_dir, args, vlm_video_by_segment) -> list[Path]:
    # Clean frames from the raw MP4; the frame index goes to the model as
    # interleaved text (frame_labels), not burned in.
    return render_frames(
        records, out_dir,
        jpeg_quality=args.jpeg_quality,
        video_by_segment=vlm_video_by_segment,
        target_height=args.vlm_frame_height,
    )


def pass1_record_windows(frame_records: list[dict[str, Any]], window_frames: int) -> list[list[dict[str, Any]]]:
    """Split the kept-frame stream into fixed-size windows of at most
    ``window_frames`` frames (0 disables windowing)."""
    if window_frames <= 0:
        return [frame_records]
    return [frame_records[i:i + window_frames] for i in range(0, len(frame_records), window_frames)]


# ---------------------------------------------------------------------------
# Pass 1 — describe + segment
# ---------------------------------------------------------------------------


def _pass1_window(lab, args, wi, records, transcript, output_dir, vlm_video_by_segment):
    """Describe + segment ONE window. Independent of the other windows, so these
    run concurrently."""
    idx_min = int(records[0]["global_frame_idx"])
    idx_max = int(records[-1]["global_frame_idx"])
    if args.pass1_image_max and args.pass1_image_max > 0:
        sampled = select_naming_frames(records, args.pass1_image_max)
    else:
        sampled = records
    print(f"  pass1[{wi:04d}]: {len(sampled)} images "
          f"({'all kept' if len(sampled)==len(records) else f'sub-sampled from {len(records)}'}); "
          f"frames {idx_min}..{idx_max}")

    imgs = render_for(sampled, output_dir / "pass1_frames" / f"window_{wi:04d}", args, vlm_video_by_segment)
    prompt = pass1_prompt(transcript.render(idx_min, idx_max, max_text_chars=8000), len(sampled))
    parsed = lab.call_json(SYSTEM, prompt, images=imgs, image_labels=frame_labels(sampled),
                           cache_path=_cache(output_dir, f"pass1_{wi:04d}"))
    acts: list[dict[str, Any]] = []
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
            spans.append({"start_frame": sf, "end_frame": ef, "user_state": str(sp.get("user_state", ""))})
        if not spans:
            continue
        spans.sort(key=lambda s: (s["start_frame"], s["end_frame"]))
        acts.append({
            "id": f"{wi}:{a.get('id')}", "pass1_window_idx": wi, "app": str(a.get("app", "")),
            "spans": spans, "description": str(a.get("description", "")),
            "onset": str(a.get("onset", "unknown")), "completion": str(a.get("completion", "unknown")),
        })
    window_row = {"window_idx": wi, "start_frame_idx": idx_min, "end_frame_idx": idx_max,
                  "n_kept_frames": len(records), "n_sent_frames": len(sampled),
                  "n_activities": len(acts), "cache_name": f"pass1_{wi:04d}"}
    return acts, window_row


def step_pass1(lab, args, frame_records, transcript, output_dir, vlm_video_by_segment):
    windows = list(enumerate(pass1_record_windows(frame_records, args.pass1_window_frames)))
    results = parallel_map(
        lambda iw: _pass1_window(lab, args, iw[0], iw[1], transcript, output_dir, vlm_video_by_segment),
        windows, args.concurrency,
    )
    activities: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for acts, row in results:  # order-preserving -> windows stay in order
        activities.extend(acts)
        window_rows.append(row)
    write_jsonl(output_dir / "activities.jsonl", activities)
    write_jsonl(output_dir / "pass1_windows.jsonl", window_rows)
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
    f0 = int(span_records[0]["global_frame_idx"]); f1 = int(span_records[-1]["global_frame_idx"])
    prompt = pass2_prompt(
        spans_text, act.get("description", ""), act.get("onset", "unknown"),
        act.get("completion", "unknown"), transcript.render(f0, f1, max_text_chars=2000), len(frames),
    )
    parsed = lab.call_json(SYSTEM, prompt, images=imgs, image_labels=frame_labels(frames),
                           cache_path=_cache(output_dir, f"pass2_{tag}"))
    return parsed, span_records


def step_pass2(lab, args, frame_records, transcript, activities, output_dir, vlm_video_by_segment):
    """Per activity, write the candidate user-prompt instruction. Returns the
    labelled activities (NOT final trajectories — pass 3 groups them into goals)."""
    def do(ai_act):
        ai, act = ai_act
        parsed, span_records = label_activity(
            lab, args, frame_records, transcript, act, f"{ai:04d}", output_dir, vlm_video_by_segment
        )
        if not parsed:
            return ("reject", {"activity": act, "reason": "too_few_frames_in_span"})
        instruction = str(parsed.get("instruction", "")).strip()
        if not instruction:
            return ("reject", {"activity": act, "reason": "empty_instruction", "label": parsed})
        return ("ok", {
            "activity_index": ai,
            "activity": act,
            "instruction": instruction,
            "instruction_variants": [str(v).strip() for v in parsed.get("instruction_variants", []) if str(v).strip()],
            "grounding": str(parsed.get("grounding", "")),
            "start_frame_idx": int(span_records[0]["global_frame_idx"]),
            "end_frame_idx": int(span_records[-1]["global_frame_idx"]),
        })

    labelled: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for kind, val in parallel_map(do, list(enumerate(activities)), args.concurrency):
        (labelled if kind == "ok" else rejected).append(val)
    write_jsonl(output_dir / "pass2_labels.jsonl", labelled)
    return labelled, rejected


# ---------------------------------------------------------------------------
# Pass 3 — merge activities into goals (text only)
# ---------------------------------------------------------------------------


def activities_text(labelled: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for L in labelled:
        a = L["activity"]
        spans = ", ".join(f"{s['start_frame']}–{s['end_frame']}" for s in a.get("spans", []))
        states = "/".join(sorted({s.get("user_state", "") for s in a.get("spans", [])}))
        lines.append(f"[activity {L['activity_index']}] app={a.get('app','')} frames {spans} "
                     f"state={states} onset={a.get('onset','')} completion={a.get('completion','')}")
        lines.append(f"  description: {a.get('description','')}")
        lines.append(f"  instruction: {L['instruction']}")
        for v in L["instruction_variants"]:
            lines.append(f"    variant: {v}")
    return "\n".join(lines)


def step_pass3(lab, args, frame_records, labelled, output_dir):
    """Group the labelled activities into goals (text only) and emit one merged
    trajectory per goal. Activities the model omits fall back to singleton goals."""
    trajectories: list[dict[str, Any]] = []
    if not labelled:
        write_jsonl(output_dir / "goals.jsonl", [])
        return trajectories
    by_index = {L["activity_index"]: L for L in labelled}

    parsed = lab.call_json(SYSTEM, prompts.render("pass3", activities_text=activities_text(labelled)),
                           cache_path=_cache(output_dir, "pass3"))
    raw_goals = parsed.get("goals", []) if isinstance(parsed, dict) else []

    used: set[int] = set()
    goals: list[dict[str, Any]] = []
    for g in raw_goals:
        members = [int(m) for m in g.get("members", []) if isinstance(m, (int, float)) and int(m) in by_index]
        members = [m for m in members if m not in used]
        instr = str(g.get("instruction", "")).strip()
        if not members or not instr:
            continue
        variants = [str(v).strip() for v in g.get("instruction_variants", []) if str(v).strip()]
        goals.append({"members": members, "instruction": instr, "instruction_variants": variants,
                      "rationale": str(g.get("rationale", "")), "merged": len(members) > 1})
        used.update(members)
    # Any labelled activity the model dropped becomes its own singleton goal,
    # carrying its pass-2 instruction (never silently lose a labelled activity).
    for L in labelled:
        if L["activity_index"] not in used:
            goals.append({"members": [L["activity_index"]], "instruction": L["instruction"],
                          "instruction_variants": L["instruction_variants"],
                          "rationale": "(unmerged: not grouped by pass 3)", "merged": False})

    for gi, g in enumerate(goals):
        member_acts = [by_index[m]["activity"] for m in g["members"]]
        # Trajectory frames = UNION of the members' ACTUAL spans (not the
        # [min,max] envelope), so interleaved frames belonging to other
        # activities in the gaps are excluded. Merge overlapping/adjacent spans.
        raw_spans = sorted((int(s["start_frame"]), int(s["end_frame"]))
                           for a in member_acts for s in a.get("spans", []))
        frame_spans: list[list[int]] = []
        for s, e in raw_spans:
            if frame_spans and s <= frame_spans[-1][1] + 1:
                frame_spans[-1][1] = max(frame_spans[-1][1], e)
            else:
                frame_spans.append([s, e])
        recs: list[dict[str, Any]] = []
        seen: set[int] = set()
        for s, e in frame_spans:
            for r in records_in_index_span(frame_records, s, e):
                ix = int(r["global_frame_idx"])
                if ix not in seen:
                    seen.add(ix)
                    recs.append(r)
        recs.sort(key=lambda r: int(r["global_frame_idx"]))
        if len(recs) < 2:
            continue
        states = {s.get("user_state", "") for a in member_acts for s in a.get("spans", [])}
        user_state = "actively_working" if "actively_working" in states else "idle_waiting"
        apps = sorted({a.get("app", "") for a in member_acts if a.get("app")})
        trajectories.append({
            "start_frame_idx": int(recs[0]["global_frame_idx"]),
            "end_frame_idx": int(recs[-1]["global_frame_idx"]),
            "instruction": g["instruction"],
            "instruction_variants": g["instruction_variants"],
            "merged": g["merged"],
            "members": g["members"],
            "rationale": g["rationale"],
            "frame_spans": frame_spans,
            "app": ", ".join(apps),
            "user_state": user_state,
            "descriptions": [by_index[m]["activity"].get("description", "") for m in g["members"]],
            "groundings": [by_index[m]["grounding"] for m in g["members"]],
            "member_spans": [{"activity_index": m, "start_frame_idx": by_index[m]["start_frame_idx"],
                              "end_frame_idx": by_index[m]["end_frame_idx"]} for m in g["members"]],
        })
    write_jsonl(output_dir / "goals.jsonl", goals)
    return trajectories


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
    p.add_argument("--pass1-window-frames", type=int, default=150,
                   help="Split pass 1 into fixed windows of at most this many kept frames "
                        "(150 = a 5-min window at 0.5 fps). With --pass1-image-max 0 (send all) "
                        "this is the effective image cap, so it must stay within the model's "
                        "context (~150 frames at 720p fits a 262K-token window). 0 disables windowing.")
    p.add_argument("--pass1-image-max", type=int, default=0,
                   help="Max frames sent to each pass-1 describe+segment call. "
                        "0 means no cap: send all kept stage-01 frames in that window.")
    # Pass 2
    p.add_argument("--label-image-max", type=int, default=config.DEFAULT_NAME_IMAGE_MAX,
                   help="Max frames sent per pass-2 label call (across the activity's spans).")
    # Frame render
    p.add_argument("--vlm-frame-height", type=int, default=config.DEFAULT_VLM_FRAME_HEIGHT)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--concurrency", type=int, default=4,
                   help="Max in-flight labeler calls within this clip (pass-1 windows and "
                        "pass-2 activities run concurrently; they're independent HTTP requests).")
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

    transcript = build_transcript(args.keylog, frame_records=frame_records,
                                  segment_offset_s=args.segment_offset_s)
    write_jsonl(output_dir / "input_transcript.jsonl",
                [{"frame": transcript.frame_of(e.t_s),
                  "end_frame": transcript.frame_of(e.t_end_s) if e.t_end_s else transcript.frame_of(e.t_s),
                  "kind": e.kind, **e.data}
                 for e in transcript.events])

    vlm_video_by_segment = load_vlm_video_sources(args.manifest)

    activities = step_pass1(lab, args, frame_records, transcript, output_dir, vlm_video_by_segment)
    labelled, rejected = step_pass2(lab, args, frame_records, transcript, activities, output_dir, vlm_video_by_segment)
    trajectories = step_pass3(lab, args, frame_records, labelled, output_dir)

    n_spans = sum(len(a["spans"]) for a in activities)
    n_merged = sum(1 for t in trajectories if t.get("merged"))
    result = {
        "recording_id": str(frame_records[0]["recording_id"]),
        "annotation_source": "vlm_hindsight",
        "trajectories": sorted(trajectories, key=lambda t: (t["start_frame_idx"], t["end_frame_idx"])),
        "response_meta": {"model": cfg.model, "base_url": cfg.base_url,
                          "vlm_frame_height": args.vlm_frame_height},
    }
    write_json(output_dir / "trajectories_raw.json", result)
    write_jsonl(output_dir / "rejected.jsonl", rejected)
    write_json(output_dir / "stage02_summary.json", {
        "n_frames": len(frame_records),
        "n_pass1_activities": len(activities),
        "n_pass1_windows": len(pass1_record_windows(frame_records, args.pass1_window_frames)),
        "n_spans": n_spans,
        "n_active_spans": sum(1 for a in activities for s in a["spans"] if s.get("user_state") == "actively_working"),
        "n_idle_spans": sum(1 for a in activities for s in a["spans"] if s.get("user_state") == "idle_waiting"),
        "n_pass2_labelled": len(labelled),
        "n_goals": len(trajectories),
        "n_merged_goals": n_merged,
        "n_trajectories": len(trajectories),
        "n_rejected": len(rejected),
        "n_variants_total": sum(1 + len(t["instruction_variants"]) for t in trajectories),
    })
    print(f"activities={len(activities)} labelled={len(labelled)} "
          f"goals={len(trajectories)} (merged={n_merged}) rejected={len(rejected)} "
          f"-> {output_dir/'trajectories_raw.json'}")


if __name__ == "__main__":
    main()
