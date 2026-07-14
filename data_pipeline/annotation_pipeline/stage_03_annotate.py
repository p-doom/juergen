#!/usr/bin/env python3
"""Stage 03: vision-only, timestamp-free hindsight annotation.

Two passes over one Stage-02 observation view. The frames in order are the
entire evidence; structured input events are never shown to the labeler.

  A. DESCRIBE — feed every kept frame and get a faithful, fine-grained, factual
     prose narration of what happens (no goals/intent), under the `system`
     framing.

  B. EXTRACT — feed that narration + the frames again (each labelled `frame <N>`)
     and recover the instruction(s) a person would type to a computer-use agent
     to reproduce the work, with per-goal start/end frame bounds.

Everything is cached per call (cache/<name>.txt + .reasoning.txt + .meta.json),
so a prompt edit + ``--refresh <name>`` re-runs only the changed call. Outputs:
  - annotation.json : self-contained record (prompt / reasoning / raw
    response / narration / goals) for the inspector and downstream stages.
  - describe_prose.txt, goal_proposals.jsonl, manifest.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from annotation_pipeline import config, prompts
from annotation_pipeline.common import ensure_dir, read_jsonl, write_json, write_jsonl
from annotation_pipeline.frames_render import (
    frames_to_data_urls,
    select_naming_frames,
)
from annotation_pipeline.labeler import Labeler, LabelerConfig

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


def clean_goals(
    parsed: dict[str, Any],
    frame_lo: int | None = None,
    frame_hi: int | None = None,
    own_hi: int | None = None,
) -> list[dict[str, Any]]:
    """`frame_lo`/`frame_hi` are the global indices of the frames actually sent
    (used to clamp reported boundaries). `own_hi`, when set, is the last frame this
    window OWNS: any extra frames sent beyond it are a trailing CONTEXT buffer (so
    the model can see where this window's last goal ends without ending on the very
    last frame). A goal whose start_frame is past `own_hi` began in the buffer and
    belongs to the next window — drop it; surviving goals' ends clamp to own_hi."""
    goals: list[dict[str, Any]] = []
    for g in parsed.get("goals", []) if isinstance(parsed, dict) else []:
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
                continue  # goal opened in the trailing buffer -> next window owns it
            if frame_lo is not None and frame_hi is not None:
                sf = max(frame_lo, min(sf, frame_hi))
                ef = max(frame_lo, min(ef, frame_hi))
            if own_hi is not None:
                ef = min(ef, own_hi)  # this window's work ends by its owned range
        except (KeyError, TypeError, ValueError):
            sf = ef = None
        goals.append(
            {
                "instruction": instr,
                "instruction_variants": [
                    str(v).strip() for v in g.get("instruction_variants", []) if str(v).strip()
                ],
                "anchor": str(g.get("anchor", "")).strip(),
                "grounding": str(g.get("grounding", "")).strip(),
                "start_frame": sf,
                "end_frame": ef,
            }
        )
    return goals


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------


def run_describe_prose(
    lab: Labeler,
    imgs: list[Path | str],
    n: int,
    output_dir: Path,
    no_cache: bool,
    cache_name: str = "describe_prose",
    image_labels: list[str] | None = None,
) -> dict[str, Any]:
    prompt = describe_prose_prompt(n)
    # Interleave the same `frame <N>` labels as extract, so the narration's frame
    # references are GROUNDED in the printed index rather than the model's own
    # running count (which drifts early on idle/black-heavy clips and then
    # poisons extract's goal boundaries).
    res = lab.call_full(
        SYSTEM,
        prompt,
        images=imgs,
        image_labels=image_labels,
        cache_path=_cache(output_dir, cache_name),
        no_cache=no_cache,
    )
    return {
        "prompt": prompt,
        "reasoning": res.reasoning,
        "content": res.content,
        "finish_reason": res.finish_reason,
        "usage": res.usage,
        "description": res.content,
    }


def run_extract(
    lab: Labeler,
    description: str,
    imgs: list[Path],
    n: int,
    cache_name: str,
    output_dir: Path,
    no_cache: bool,
    image_labels: list[str] | None = None,
    frame_lo: int | None = None,
    frame_hi: int | None = None,
    own_hi: int | None = None,
) -> dict[str, Any]:
    if not description.strip():
        return {"prompt": "", "goals": [], "error": "empty_description"}
    prompt = extract_prompt(description, n)
    out: dict[str, Any] = {"prompt": prompt}
    try:
        # Frame index labels are interleaved before each image (extract only) so
        # the model can report each goal's start_frame/end_frame.
        parsed, res = lab.call_json_full(
            EXTRACT_SYSTEM,
            prompt,
            images=imgs,
            image_labels=image_labels,
            cache_path=_cache(output_dir, cache_name),
            no_cache=no_cache,
        )
        out.update(
            {
                "reasoning": res.reasoning,
                "content": res.content,
                "finish_reason": res.finish_reason,
                "usage": res.usage,
                "goals": clean_goals(parsed, frame_lo, frame_hi, own_hi),
            }
        )
    except Exception as exc:
        out.update({"error": f"{type(exc).__name__}: {exc}", "goals": []})
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    # Frames
    p.add_argument(
        "--image-max",
        type=int,
        default=0,
        help="Max frames sent per call. 0 = send all Stage-02 observations "
        "(a ~5-min clip at 0.5 fps is ~150 frames).",
    )
    p.add_argument("--vlm-frame-height", type=int, default=config.DEFAULT_VLM_FRAME_HEIGHT)
    p.add_argument("--jpeg-quality", type=int, default=80)
    # Provenance for window-segments (set by the driver when this unit is one
    p.add_argument(
        "--parent-segment-id",
        default="",
        help="Original segment this unit stems from (defaults to its own).",
    )
    p.add_argument("--window-index", type=int, default=0)
    p.add_argument("--n-windows", type=int, default=1)
    p.add_argument(
        "--tail-buffer",
        type=int,
        default=0,
        help="Number of trailing frames (of the frames passed) that are a "
        "CONTEXT buffer beyond this window's owned range: the model "
        "sees them to judge where its last goal ends, but goals that "
        "START in them are dropped (the next window owns them).",
    )
    # Labeler
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument(
        "--refresh",
        default=None,
        help="Comma-separated cache prefixes to invalidate before running "
        "(describe_prose or extract_from_prose).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    if args.refresh:
        n = refresh_cache(output_dir, [p.strip() for p in args.refresh.split(",") if p.strip()])
        print(f"refresh: invalidated {n} cached files ({args.refresh})")

    observations = read_jsonl(args.observations)
    if not observations:
        raise RuntimeError(f"No observations: {args.observations}")
    if args.tail_buffer < 0 or args.tail_buffer >= len(observations):
        raise ValueError("tail_buffer must be non-negative and smaller than the view")

    cfg = LabelerConfig.from_env(
        model=args.model, base_url=args.base_url, reasoning_effort=args.reasoning_effort
    )
    lab = Labeler(cfg)

    # Frames for the whole clip come through the Stage-02 observation view and
    # point at the Stage-01 array_record
    # (each record's ar:// image_path) as in-memory data URLs — no re-render, no
    # loose jpegs on disk. Downscales only if vlm_frame_height < the stored height.
    sent = observations
    if args.image_max and args.image_max > 0 and len(observations) > args.image_max:
        sent = select_naming_frames(observations, args.image_max)
    imgs = frames_to_data_urls(
        sent, target_height=args.vlm_frame_height, jpeg_quality=args.jpeg_quality
    )
    n = len(imgs)
    sent_frame_indices = [int(r["global_frame_idx"]) for r in sent]
    # Frame-index labels interleaved before each image in the EXTRACT call only,
    # so goals can carry start_frame/end_frame. Describe stays label-free.
    frame_labels = [f"frame {i}" for i in sent_frame_indices]
    frame_lo = min(sent_frame_indices) if sent_frame_indices else None
    frame_hi = max(sent_frame_indices) if sent_frame_indices else None
    # The last `tail_buffer` frames are trailing CONTEXT (lookahead so the model
    # can see where this window's final goal ends): this window OWNS up to own_hi.
    owned_position = len(observations) - 1 - args.tail_buffer
    own_hi = (
        int(observations[owned_position]["global_frame_idx"])
        if args.tail_buffer and owned_position >= 0
        else frame_hi
    )
    print(
        f"  annotate: {n} frames "
        f"({'all kept' if n == len(observations) else f'sub-sampled from {len(observations)}'})"
    )

    # One describe (prose narration) + one extract (goals) over ALL frames given.
    # The driver guarantees these frames fit the context (it splits oversized
    # segments into independent __wN window-segments upstream), so there is no
    # windowing or merging here.
    describe = run_describe_prose(
        lab,
        imgs,
        n,
        output_dir,
        args.no_cache,
        "describe_prose",
        frame_labels,
    )
    extract = run_extract(
        lab,
        describe["description"],
        imgs,
        n,
        "extract_from_prose",
        output_dir,
        args.no_cache,
        frame_labels,
        frame_lo,
        frame_hi,
        own_hi,
    )

    # ---- write outputs ----
    (output_dir / "describe_prose.txt").write_text(describe.get("content", "") + "\n")
    write_jsonl(output_dir / "goal_proposals.jsonl", extract["goals"])

    parent_seg = args.parent_segment_id or str(observations[0].get("segment_id", ""))
    result = {
        "stage": "visual_annotation",
        "schema_version": 1,
        "recording_id": str(observations[0]["recording_id"]),
        "segment_id": str(observations[0].get("segment_id", "")),
        # Provenance: when this unit is a window of a larger segment, these track
        # it back to the parent and the exact frames it covers.
        "parent_segment_id": parent_seg,
        "window_index": args.window_index,
        "n_windows": args.n_windows,
        "tail_buffer": args.tail_buffer,
        "source_frame_range": [sent_frame_indices[0], sent_frame_indices[-1]]
        if sent_frame_indices
        else None,
        "annotation_source": "visual_describe_extract_v1",
        "source_observations": str(args.observations.resolve()),
        "n_observations": len(observations),
        "n_images_sent": n,
        "sent_frame_indices": sent_frame_indices,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "vlm_frame_height": args.vlm_frame_height,
        "variants": {"prose": {"describe": describe, "extract": extract}},
    }
    write_json(output_dir / "annotation.json", result)

    summary = {
        "stage": "visual_annotation",
        "schema_version": 1,
        "source_observations": str(args.observations.resolve()),
        "n_observations": len(observations),
        "n_images_sent": n,
        "prose_chars": len(describe.get("description", "")),
        "n_goals_prose": len(extract["goals"]),
        "n_variants_total": sum(1 + len(g["instruction_variants"]) for g in extract["goals"]),
        "describe_prose_finish": describe.get("finish_reason"),
        "errors": {k: v.get("error") for k, v in [("extract_prose", extract)] if v.get("error")},
    }
    write_json(output_dir / "manifest.json", summary)
    print(f"frames={n} prose_goals={summary['n_goals_prose']} -> {output_dir / 'annotation.json'}")


if __name__ == "__main__":
    main()
