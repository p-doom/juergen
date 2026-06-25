#!/usr/bin/env python3
"""Stage 02 (v2 redesign): vision-only, timestamp-free hindsight annotation.

Two passes over ONE ~5-min clip (frames sampled at 0.5 fps in stage 01, NO_OP
runs capped 2 head / 2 tail). No keylog, no frame-index anchoring — the frames
in order are the entire evidence.

  A. DESCRIBE — feed every kept frame and get a faithful, fine-grained, factual
     log of what happens (no goals/intent). Run in TWO variants we compare
     downstream:
       - describe_prose : free-form chronological narration (raw text)
       - describe_steps : structured step list (JSON)
     Both share the `system` framing.

  B. EXTRACT — feed that account + the frames again and recover the
     instruction(s) a person would type to a computer-use agent to reproduce the
     work. Run once per describe variant (`extract` + `extract_system`).

Everything is cached per call (cache/<name>.txt + .reasoning.txt + .meta.json),
so a prompt edit + ``--refresh <name>`` re-runs only the changed call. Outputs:
  - stage02_result.json : self-contained record for the inspector — both
    variants' prompt / reasoning (thinking) / raw response / parsed description /
    goals.
  - describe_prose.txt, describe_steps.json, goals_prose.jsonl, goals_steps.jsonl
  - stage02_summary.json (read by run_iteration)
  - trajectories_raw.json (legacy-shaped: prose goals, for downstream/skip-check)
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


def describe_steps_prompt(n_frames: int) -> str:
    return prompts.render("describe_steps", n_frames=n_frames)


def extract_prompt(description: str, n_frames: int) -> str:
    return prompts.render("extract", description=description, n_frames=n_frames)


def render_steps_text(steps: list[dict[str, Any]]) -> str:
    """Render the structured step list into the plain-text account the extract
    pass consumes (so both variants feed extract the same way)."""
    lines: list[str] = []
    for i, s in enumerate(steps, 1):
        actor = str(s.get("actor", "")).strip()
        ctx = str(s.get("context", "")).strip()
        action = str(s.get("action", "")).strip()
        result = str(s.get("result", "")).strip()
        text = str(s.get("text", "")).strip()
        src = str(s.get("text_source", "")).strip()
        note = str(s.get("note", "")).strip()
        head = f"{i}. [{actor}] {ctx}: {action}" if ctx else f"{i}. [{actor}] {action}"
        if result:
            head += f"  ->  {result}"
        lines.append(head)
        if text and src not in ("", "none"):
            lines.append(f'    text ({src}): "{text}"')
        elif text:
            lines.append(f'    text: "{text}"')
        if note:
            lines.append(f"    note: {note}")
    return "\n".join(lines)


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
                frame_hi: int | None = None) -> list[dict[str, Any]]:
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
            if frame_lo is not None and frame_hi is not None:
                sf = max(frame_lo, min(sf, frame_hi))
                ef = max(frame_lo, min(ef, frame_hi))
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


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------


def run_describe_prose(lab: Labeler, imgs: list[Path], n: int, output_dir: Path, no_cache: bool) -> dict[str, Any]:
    prompt = describe_prose_prompt(n)
    res = lab.call_full(SYSTEM, prompt, images=imgs, cache_path=_cache(output_dir, "describe_prose"), no_cache=no_cache)
    return {"prompt": prompt, "reasoning": res.reasoning, "content": res.content,
            "finish_reason": res.finish_reason, "usage": res.usage, "description": res.content}


def run_describe_steps(lab: Labeler, imgs: list[Path], n: int, output_dir: Path, no_cache: bool) -> dict[str, Any]:
    prompt = describe_steps_prompt(n)
    out: dict[str, Any] = {"prompt": prompt}
    try:
        parsed, res = lab.call_json_full(SYSTEM, prompt, images=imgs,
                                         cache_path=_cache(output_dir, "describe_steps"), no_cache=no_cache)
        steps = parsed.get("steps", []) if isinstance(parsed, dict) else []
        out.update({"reasoning": res.reasoning, "content": res.content, "finish_reason": res.finish_reason,
                    "usage": res.usage, "steps": steps, "description": render_steps_text(steps)})
    except Exception as exc:  # noqa: BLE001 - keep the clip alive if steps JSON is bad
        out.update({"error": f"{type(exc).__name__}: {exc}", "steps": [], "description": ""})
    return out


def run_extract(lab: Labeler, description: str, imgs: list[Path], n: int, cache_name: str,
                output_dir: Path, no_cache: bool, image_labels: list[str] | None = None,
                frame_lo: int | None = None, frame_hi: int | None = None) -> dict[str, Any]:
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
                    "usage": res.usage, "goals": clean_goals(parsed, frame_lo, frame_hi)})
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
    # Accepted for run_iteration compatibility; unused in v2 (vision-only).
    p.add_argument("--keylog", type=Path, default=None, help="(ignored in v2 — no keylog)")
    p.add_argument("--segment-offset-s", type=float, default=0.0, help="(ignored in v2)")
    # Frames
    p.add_argument("--image-max", type=int, default=0,
                   help="Max frames sent per call. 0 = send all kept stage-01 frames "
                        "(a ~5-min clip at 0.5 fps is ~150 frames).")
    p.add_argument("--vlm-frame-height", type=int, default=config.DEFAULT_VLM_FRAME_HEIGHT)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--concurrency", type=int, default=4,
                   help="Max in-flight labeler calls within this clip.")
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
    print(f"  v2 annotate: {n} frames "
          f"({'all kept' if n == len(frame_records) else f'sub-sampled from {len(frame_records)}'})")

    # Pass A — both describe variants, concurrently.
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        f_prose = ex.submit(run_describe_prose, lab, imgs, n, output_dir, args.no_cache)
        f_steps = ex.submit(run_describe_steps, lab, imgs, n, output_dir, args.no_cache)
        describe = {"prose": f_prose.result(), "steps": f_steps.result()}

    # Pass B — extract on each variant's account, concurrently.
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        f_ep = ex.submit(run_extract, lab, describe["prose"]["description"], imgs, n,
                         "extract_from_prose", output_dir, args.no_cache, frame_labels, frame_lo, frame_hi)
        f_es = ex.submit(run_extract, lab, describe["steps"]["description"], imgs, n,
                         "extract_from_steps", output_dir, args.no_cache, frame_labels, frame_lo, frame_hi)
        extract = {"prose": f_ep.result(), "steps": f_es.result()}

    # ---- write outputs ----
    (output_dir / "describe_prose.txt").write_text(describe["prose"].get("content", "") + "\n")
    write_json(output_dir / "describe_steps.json", describe["steps"].get("steps", []))
    write_jsonl(output_dir / "goals_prose.jsonl", extract["prose"]["goals"])
    write_jsonl(output_dir / "goals_steps.jsonl", extract["steps"]["goals"])

    result = {
        "recording_id": str(frame_records[0]["recording_id"]),
        "segment_id": str(frame_records[0].get("segment_id", "")),
        "annotation_source": "vlm_describe_extract_v2",
        "n_frames": len(frame_records),
        "n_images_sent": n,
        "sent_frame_indices": sent_frame_indices,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "vlm_frame_height": args.vlm_frame_height,
        "variants": {
            "prose": {"describe": describe["prose"], "extract": extract["prose"]},
            "steps": {"describe": describe["steps"], "extract": extract["steps"]},
        },
    }
    write_json(output_dir / "stage02_result.json", result)

    # Legacy-shaped trajectories (prose goals) for downstream / skip-check.
    write_json(output_dir / "trajectories_raw.json", {
        "recording_id": result["recording_id"],
        "annotation_source": result["annotation_source"],
        "trajectories": [
            {"instruction": g["instruction"], "instruction_variants": g["instruction_variants"],
             "anchor": g["anchor"], "grounding": g["grounding"], "variant": "prose"}
            for g in extract["prose"]["goals"]
        ],
        "variants": {"prose": extract["prose"]["goals"], "steps": extract["steps"]["goals"]},
    })

    summary = {
        "n_frames": len(frame_records),
        "n_images_sent": n,
        "n_steps": len(describe["steps"].get("steps", [])),
        "prose_chars": len(describe["prose"].get("description", "")),
        "n_goals_prose": len(extract["prose"]["goals"]),
        "n_goals_steps": len(extract["steps"]["goals"]),
        "n_variants_total": sum(1 + len(g["instruction_variants"]) for g in extract["prose"]["goals"]),
        "describe_prose_finish": describe["prose"].get("finish_reason"),
        "describe_steps_finish": describe["steps"].get("finish_reason"),
        "errors": {k: v.get("error") for k, v in
                   [("describe_steps", describe["steps"]), ("extract_prose", extract["prose"]),
                    ("extract_steps", extract["steps"])] if v.get("error")},
    }
    write_json(output_dir / "stage02_summary.json", summary)
    print(f"frames={n} prose_goals={summary['n_goals_prose']} steps_goals={summary['n_goals_steps']} "
          f"steps={summary['n_steps']} -> {output_dir/'stage02_result.json'}")


if __name__ == "__main__":
    main()
