#!/usr/bin/env python3
"""Stage 02 (v2 redesign): vision-only, timestamp-free hindsight annotation.

Two passes over ONE clip (frames sampled at 0.5 fps in stage 01, NO_OP runs
thinned to head/tail). The frames in order are the entire evidence.

  A. DESCRIBE — feed every kept frame and get a faithful, fine-grained, factual
     prose narration of what happens (no goals/intent), under the `system`
     framing.

  B. EXTRACT — feed that narration + the frames again (each labelled `frame <N>`)
     and recover the instruction(s) a person would type to a computer-use agent
     to reproduce the work, with per-goal start/end frame bounds.

Everything is cached per call (cache/<name>.txt + .reasoning.txt + .meta.json),
so a prompt edit + ``--refresh <name>`` re-runs only the changed call. Outputs:
  - stage02_result.json : self-contained record (prompt / reasoning / raw
    response / narration / goals) for the inspector and downstream stages.
  - describe_prose.txt, goals_prose.jsonl, stage02_summary.json
  - trajectories_raw.json : goals + frame bounds, consumed by stage 03.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from annotation_pipeline import config, prompts
from annotation_pipeline.common import ensure_dir, read_jsonl, write_json, write_jsonl
from annotation_pipeline.labeler import Labeler, LabelerConfig, LabelResult
from annotation_pipeline.frames_render import (
    frames_to_data_urls,
    select_naming_frames,
)

SYSTEM = prompts.get("system")
EXTRACT_SYSTEM = prompts.get("extract_system")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def describe_prose_prompt(n_frames: int) -> str:
    return prompts.render("describe_prose", n_frames=n_frames)


def extract_prompt(description: str, n_frames: int) -> str:
    return prompts.render("extract", description=description, n_frames=n_frames)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache(output_dir: Path, name: str) -> Path:
    return output_dir / "cache" / f"{name}.txt"


def refresh_cache(output_dir: Path, prefixes: list[str]) -> int:
    cdir = output_dir / "cache"
    if not cdir.is_dir() or not prefixes:
        return 0
    n = 0
    for f in cdir.iterdir():
        if f.is_file() and any(f.name.startswith(p) for p in prefixes):
            f.unlink()
            n += 1
    return n


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------


def clean_goals(parsed: dict[str, Any], frame_lo: int | None = None,
                frame_hi: int | None = None, own_hi: int | None = None) -> list[dict[str, Any]]:
    """`frame_lo`/`frame_hi` are the global indices of the frames actually sent
    (used to clamp reported boundaries). `own_hi`, when set, is the last frame this
    window OWNS: any extra frames sent beyond it are a trailing CONTEXT buffer (so
    the model can see where this window's last goal ends without ending on the very
    last frame). A goal whose start_frame is past `own_hi` began in the buffer and
    belongs to the next window — drop it; surviving goals' ends clamp to own_hi."""
    goals: list[dict[str, Any]] = []
    for g in (parsed.get("goals", []) if isinstance(parsed, dict) else []):
        if not isinstance(g, dict):
            continue
        instr = str(g.get("instruction", "")).strip()
        if not instr:
            continue
        # Frame boundaries the model reports against the interleaved `frame <N>`
        # labels (extract only). Clamp to the indices actually sent; None if absent.
        sf = ef = None
        try:
            sf, ef = int(g["start_frame"]), int(g["end_frame"])
            if ef < sf:
                sf, ef = ef, sf
            if own_hi is not None and sf > own_hi:
                continue                     # goal opened in the trailing buffer -> next window owns it
            if frame_lo is not None and frame_hi is not None:
                sf = max(frame_lo, min(sf, frame_hi))
                ef = max(frame_lo, min(ef, frame_hi))
            if own_hi is not None:
                ef = min(ef, own_hi)         # this window's work ends by its owned range
        except (KeyError, TypeError, ValueError):
            sf = ef = None
        goals.append({
            "instruction": instr,
            "instruction_variants": [str(v).strip() for v in g.get("instruction_variants", []) if str(v).strip()],
            "anchor": str(g.get("anchor", "")).strip(),
            "grounding": str(g.get("grounding", "")).strip(),
            "start_frame": sf,
            "end_frame": ef,
        })
    return goals


def snap_goal_starts(goals: list[dict[str, Any]], sent: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pull each TYPED goal's start_frame back to the first keystroke of its input
    burst, using the per-frame keylog action labels.

    Why: the labeler is vision-only and the screenshot for the first-keystroke bin
    often predates the text rendering, so the model anchors start ~1 frame late
    (the frame where the text first becomes legible). The keylog knows when typing
    actually began — snap to it. Walk back over the contiguous typing run, stopping
    at a submission (Return/Enter — that ended the PREVIOUS action) or any
    non-typing frame, so we never merge into the previous goal. Mouse/scroll/arrow-
    initiated goals (start frame isn't a keystroke) are left untouched. Finally
    re-enforce non-overlap so an earlier start can't collide with a neighbor."""
    from annotation_pipeline.frames_render import _is_submission, frame_activity  # noqa: PLC0415

    if not sent:
        return goals
    pos = {int(r["global_frame_idx"]): i for i, r in enumerate(sent)}
    acts = [r.get("action") for r in sent]
    for g in goals:
        sf = g.get("start_frame")
        if sf is None or sf not in pos:
            continue
        p = pos[sf]
        # Only snap when this is a typing burst (start frame, or the one before it,
        # is a keystroke frame). Otherwise leave the model's boundary as-is.
        if frame_activity(acts[p]) != "type" and not (p > 0 and frame_activity(acts[p - 1]) == "type"):
            continue
        while p > 0 and frame_activity(acts[p - 1]) == "type" and not _is_submission(acts[p - 1]):
            p -= 1
        g["start_frame"] = int(sent[p]["global_frame_idx"])
        if g.get("end_frame") is not None and g["end_frame"] < g["start_frame"]:
            g["end_frame"] = g["start_frame"]
    # Non-overlap: after pulling starts earlier, clamp each goal's end to just
    # before the next goal's start (chronological order) so none nest/overlap.
    ordered = sorted((g for g in goals if isinstance(g.get("start_frame"), int)
                      and isinstance(g.get("end_frame"), int)), key=lambda g: g["start_frame"])
    for a, b in zip(ordered, ordered[1:]):
        if a["end_frame"] >= b["start_frame"]:
            a["end_frame"] = max(a["start_frame"], b["start_frame"] - 1)
    return goals


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------


def run_describe_prose(lab: Labeler, imgs: list[Path | str], n: int, output_dir: Path, no_cache: bool,
                       cache_name: str = "describe_prose") -> dict[str, Any]:
    prompt = describe_prose_prompt(n)
    res = lab.call_full(SYSTEM, prompt, images=imgs, cache_path=_cache(output_dir, cache_name), no_cache=no_cache)
    return {"prompt": prompt, "reasoning": res.reasoning, "content": res.content,
            "finish_reason": res.finish_reason, "usage": res.usage, "description": res.content}


def run_extract(lab: Labeler, description: str, imgs: list[Path], n: int, cache_name: str,
                output_dir: Path, no_cache: bool, image_labels: list[str] | None = None,
                frame_lo: int | None = None, frame_hi: int | None = None,
                own_hi: int | None = None) -> dict[str, Any]:
    if not description.strip():
        return {"prompt": "", "goals": [], "error": "empty_description"}
    prompt = extract_prompt(description, n)
    out: dict[str, Any] = {"prompt": prompt}
    try:
        # Frame index labels are interleaved before each image (extract only) so
        # the model can report each goal's start_frame/end_frame.
        parsed, res = lab.call_json_full(EXTRACT_SYSTEM, prompt, images=imgs, image_labels=image_labels,
                                         cache_path=_cache(output_dir, cache_name), no_cache=no_cache)
        out.update({"reasoning": res.reasoning, "content": res.content, "finish_reason": res.finish_reason,
                    "usage": res.usage, "goals": clean_goals(parsed, frame_lo, frame_hi, own_hi)})
    except Exception as exc:  # noqa: BLE001
        out.update({"error": f"{type(exc).__name__}: {exc}", "goals": []})
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frame-records", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True,
                   help="stage 00 manifest (kept for provenance; frames now come "
                        "from the stage-01 array_record, not the raw MP4)")
    p.add_argument("--output-dir", type=Path, required=True)
    # Frames
    p.add_argument("--image-max", type=int, default=0,
                   help="Max frames sent per call. 0 = send all kept stage-01 frames "
                        "(a ~5-min clip at 0.5 fps is ~150 frames).")
    p.add_argument("--vlm-frame-height", type=int, default=config.DEFAULT_VLM_FRAME_HEIGHT)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--concurrency", type=int, default=4,
                   help="Max in-flight labeler calls within this clip.")
    # Provenance for window-segments (set by the driver when this unit is one
    # window of a larger segment that was split upstream).
    p.add_argument("--parent-segment-id", default="",
                   help="Original segment this unit stems from (defaults to its own).")
    p.add_argument("--window-index", type=int, default=0)
    p.add_argument("--n-windows", type=int, default=1)
    p.add_argument("--tail-buffer", type=int, default=0,
                   help="Number of trailing frames (of the frames passed) that are a "
                        "CONTEXT buffer beyond this window's owned range: the model "
                        "sees them to judge where its last goal ends, but goals that "
                        "START in them are dropped (the next window owns them).")
    # Labeler
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--refresh", default=None,
                   help="Comma-separated cache prefixes to invalidate before running "
                        "(describe_prose, describe_steps, extract_from_prose, extract_from_steps, "
                        "or 'describe' / 'extract').")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    if args.refresh:
        n = refresh_cache(output_dir, [p.strip() for p in args.refresh.split(",") if p.strip()])
        print(f"refresh: invalidated {n} cached files ({args.refresh})")

    frame_records = read_jsonl(args.frame_records)
    if not frame_records:
        raise RuntimeError(f"No frame records: {args.frame_records}")

    cfg = LabelerConfig.from_env(model=args.model, base_url=args.base_url, reasoning_effort=args.reasoning_effort)
    lab = Labeler(cfg)

    # Frames for the whole clip come straight from the stage-01 array_record
    # (each record's ar:// image_path) as in-memory data URLs — no re-render, no
    # loose jpegs on disk. Downscales only if vlm_frame_height < the stored height.
    sent = frame_records
    if args.image_max and args.image_max > 0 and len(frame_records) > args.image_max:
        sent = select_naming_frames(frame_records, args.image_max)
    imgs = frames_to_data_urls(sent, target_height=args.vlm_frame_height, jpeg_quality=args.jpeg_quality)
    n = len(imgs)
    sent_frame_indices = [int(r["global_frame_idx"]) for r in sent]
    # Frame-index labels interleaved before each image in the EXTRACT call only,
    # so goals can carry start_frame/end_frame. Describe stays label-free.
    frame_labels = [f"frame {i}" for i in sent_frame_indices]
    frame_lo = min(sent_frame_indices) if sent_frame_indices else None
    frame_hi = max(sent_frame_indices) if sent_frame_indices else None
    # The last `tail_buffer` frames are trailing CONTEXT (lookahead so the model
    # can see where this window's final goal ends): this window OWNS up to own_hi.
    own_hi = (sent_frame_indices[n - 1 - args.tail_buffer]
              if (args.tail_buffer and n - 1 - args.tail_buffer >= 0) else frame_hi)
    print(f"  v2 annotate: {n} frames "
          f"({'all kept' if n == len(frame_records) else f'sub-sampled from {len(frame_records)}'})")

    # One describe (prose narration) + one extract (goals) over ALL frames given.
    # The driver guarantees these frames fit the context (it splits oversized
    # segments into independent __wN window-segments upstream), so there is no
    # windowing or merging here.
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        describe = ex.submit(run_describe_prose, lab, imgs, n, output_dir, args.no_cache).result()
    extract = run_extract(lab, describe["description"], imgs, n, "extract_from_prose",
                          output_dir, args.no_cache, frame_labels, frame_lo, frame_hi, own_hi)

    # Deterministically correct the ~1-frame-late start of typed goals using the
    # keylog action labels (the vision model anchors to where text becomes legible,
    # one frame after the first keystroke). Operates on the frames actually sent.
    snap_goal_starts(extract["goals"], sent)

    # ---- write outputs ----
    (output_dir / "describe_prose.txt").write_text(describe.get("content", "") + "\n")
    write_jsonl(output_dir / "goals_prose.jsonl", extract["goals"])

    parent_seg = args.parent_segment_id or str(frame_records[0].get("segment_id", ""))
    result = {
        "recording_id": str(frame_records[0]["recording_id"]),
        "segment_id": str(frame_records[0].get("segment_id", "")),
        # Provenance: when this unit is a window of a larger segment, these track
        # it back to the parent and the exact frames it covers.
        "parent_segment_id": parent_seg,
        "window_index": args.window_index,
        "n_windows": args.n_windows,
        "source_frame_range": [sent_frame_indices[0], sent_frame_indices[-1]] if sent_frame_indices else None,
        "annotation_source": "vlm_describe_extract_v2",
        "n_frames": len(frame_records),
        "n_images_sent": n,
        "sent_frame_indices": sent_frame_indices,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "vlm_frame_height": args.vlm_frame_height,
        "variants": {"prose": {"describe": describe, "extract": extract}},
    }
    write_json(output_dir / "stage02_result.json", result)

    # Trajectories consumed by stage 03 (SFT assembly). Each goal carries its
    # frame bounds as start_frame_idx/end_frame_idx (global frame indices) so
    # stage 03 can slice the exact frames; goals with no bounds are skipped.
    write_json(output_dir / "trajectories_raw.json", {
        "recording_id": result["recording_id"],
        "annotation_source": result["annotation_source"],
        "trajectories": [
            {"instruction": g["instruction"], "instruction_variants": g["instruction_variants"],
             "anchor": g["anchor"], "grounding": g["grounding"], "variant": "prose",
             "start_frame_idx": g["start_frame"], "end_frame_idx": g["end_frame"]}
            for g in extract["goals"]
            if g.get("start_frame") is not None and g.get("end_frame") is not None
        ],
    })

    summary = {
        "n_frames": len(frame_records),
        "n_images_sent": n,
        "prose_chars": len(describe.get("description", "")),
        "n_goals_prose": len(extract["goals"]),
        "n_variants_total": sum(1 + len(g["instruction_variants"]) for g in extract["goals"]),
        "describe_prose_finish": describe.get("finish_reason"),
        "errors": {k: v.get("error") for k, v in [("extract_prose", extract)] if v.get("error")},
    }
    write_json(output_dir / "stage02_summary.json", summary)
    print(f"frames={n} prose_goals={summary['n_goals_prose']} -> {output_dir/'stage02_result.json'}")


if __name__ == "__main__":
    main()
