#!/usr/bin/env python3
"""Stage 02b: pre-action PLAN prose per goal (the reason-before-action turn).

Runs AFTER stage 02, on an annotation run laid out as <run-root>/<model>/clips/
<unit>/ (run_dataset output — e.g. the labctl juergen_<tag>_annotations
artifact, which is read-only). For each annotated window-unit it makes ONE
cached labeler call: the unit's describe narration + its goals in time order +
each goal's start-frame screenshot (from the unit's own stage_01
frame_records, ar:// grain URIs), and gets back a 1-2 sentence first-person
PLAN per goal — written strictly from the information state at that goal's
start (no outcome/clairvoyance, no restatement, situation + method). Stage 03
then renders the first assistant turn of each sample as ``plan\\nfirst_action``.

Outputs a MIRRORED tree under --out-root so build_sft consumes it as its
--run-dir unchanged:
    <out-root>/<model>/clips/<unit>/stage_02/trajectories_raw.json  goals + "plan"
    <out-root>/<model>/clips/<unit>/stage_02/stage02_result.json    slim provenance
    <out-root>/<model>/clips/<unit>/stage_02b/plans.json            full call record
    <out-root>/<model>/clips/<unit>/stage_02b/cache/plan_from_prose.*
Units with zero goals are mirrored verbatim (empty trajectories). Resumable: a
unit whose output trajectories_raw.json exists is skipped (--force to redo);
the labeler response is cached per unit (--force --no-cache to re-call after a
prompt edit).

  PYTHONPATH=. python3 -m annotation_pipeline.stage_02b_plans \
      --run-root .../juergen_ccast0618d_annotations/ccast0618d \
      --out-root .../juergen_ccast0618d_annotations_plans/ccast0618d \
      --concurrency 16
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from annotation_pipeline import config, prompts
from annotation_pipeline.common import ensure_dir, read_jsonl, write_json
from annotation_pipeline.frames_render import frames_to_data_urls
from annotation_pipeline.labeler import Labeler, LabelerConfig

PLAN_SYSTEM = prompts.get("plan_system")

# Provenance keys build_sft reads from each unit's stage02_result.json.
SLIM_RESULT_KEYS = (
    "recording_id", "segment_id", "parent_segment_id", "window_index",
    "n_windows", "source_frame_range", "annotation_source", "model",
)

STOPWORDS = frozenset(
    "a an and are as at be by for from in into is it its of on or that the "
    "then this to with i ill i'll my so".split()
)


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9'/._-]*", text.lower()) if w not in STOPWORDS}


def plan_flags(plan: str, instruction: str) -> list[str]:
    """Cheap deterministic quality flags — recorded, not enforced; stage 03
    decides what to do with flagged plans."""
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


def goal_start_record(frame_records: list[dict[str, Any]], start_idx: int) -> dict[str, Any] | None:
    """The unit's frame record at the goal's start_frame_idx — exact match, else
    the nearest kept frame after it (stage 01 thinning can drop the exact bin),
    else the nearest before."""
    by_idx = {int(r["global_frame_idx"]): r for r in frame_records}
    if start_idx in by_idx:
        return by_idx[start_idx]
    later = [i for i in by_idx if i > start_idx]
    if later:
        return by_idx[min(later)]
    earlier = [i for i in by_idx if i < start_idx]
    return by_idx[max(earlier)] if earlier else None


def build_goals_block(trajectories: list[dict[str, Any]]) -> str:
    lines = []
    for k, t in enumerate(trajectories, start=1):
        anchor = str(t.get("anchor") or "").strip().replace("\n", " ")
        line = (f"Goal {k} [frames {t.get('start_frame_idx')}-{t.get('end_frame_idx')}] "
                f"instruction: {json.dumps(str(t.get('instruction') or ''))}")
        if anchor:
            line += f"  (anchor: {json.dumps(anchor[:200])})"
        lines.append(line)
    return "\n".join(lines)


def process_unit(unit_dir: Path, out_dir: Path, lab: Labeler,
                 vlm_frame_height: int, jpeg_quality: int, no_cache: bool) -> dict[str, Any]:
    stage02 = unit_dir / "stage_02"
    raw = json.loads((stage02 / "trajectories_raw.json").read_text())
    trajectories = raw.get("trajectories", [])
    result_src = json.loads((stage02 / "stage02_result.json").read_text())
    slim = {k: result_src.get(k) for k in SLIM_RESULT_KEYS}
    slim["plans_source"] = "stage_02b_plans"

    out_stage02 = ensure_dir(out_dir / "stage_02")
    summary: dict[str, Any] = {"unit_id": unit_dir.name, "n_goals": len(trajectories)}

    if not trajectories:
        write_json(out_stage02 / "trajectories_raw.json", raw)
        write_json(out_stage02 / "stage02_result.json", slim)
        summary["skipped"] = "no_goals"
        return summary

    # Sort chronologically for the prompt but write plans back onto the
    # original trajectory objects (order in the file is preserved).
    ordered = sorted(trajectories, key=lambda t: (int(t.get("start_frame_idx") or 0),
                                                  int(t.get("end_frame_idx") or 0)))
    narration = (stage02 / "describe_prose.txt").read_text().strip()
    if not narration:
        raise RuntimeError("empty describe_prose.txt")

    frame_records = read_jsonl(unit_dir / "stage_01" / "frame_records.jsonl")
    recs, labels = [], []
    for k, t in enumerate(ordered, start=1):
        rec = goal_start_record(frame_records, int(t["start_frame_idx"]))
        if rec is None:
            raise RuntimeError(f"no frame record for goal {k} start {t['start_frame_idx']}")
        recs.append(rec)
        labels.append(f"Goal {k} start screen (frame {rec['global_frame_idx']}):")
    imgs = frames_to_data_urls(recs, target_height=vlm_frame_height, jpeg_quality=jpeg_quality)

    prompt = prompts.render("plan", description=narration, n_goals=str(len(ordered)),
                            goals_block=build_goals_block(ordered))
    stage02b = ensure_dir(out_dir / "stage_02b")
    parsed, res = lab.call_json_full(PLAN_SYSTEM, prompt, images=imgs, image_labels=labels,
                                     cache_path=stage02b / "cache" / "plan_from_prose.txt",
                                     no_cache=no_cache)

    by_goal: dict[int, str] = {}
    for entry in (parsed.get("plans", []) if isinstance(parsed, dict) else []):
        if isinstance(entry, dict):
            try:
                by_goal[int(entry["goal"])] = str(entry.get("plan") or "").strip()
            except (KeyError, TypeError, ValueError):
                continue

    plan_records = []
    for k, t in enumerate(ordered, start=1):
        plan = by_goal.get(k, "")
        flags = plan_flags(plan, str(t.get("instruction") or ""))
        t["plan"] = plan
        if flags:
            t["plan_flags"] = flags
        plan_records.append({"goal": k, "instruction": t.get("instruction"),
                             "start_frame_idx": t.get("start_frame_idx"),
                             "end_frame_idx": t.get("end_frame_idx"),
                             "plan": plan, "flags": flags})

    write_json(out_stage02 / "trajectories_raw.json", raw)
    write_json(out_stage02 / "stage02_result.json", slim)
    write_json(stage02b / "plans.json", {
        "unit_id": unit_dir.name, "model": res.model, "usage": res.usage,
        "finish_reason": res.finish_reason, "prompt": prompt,
        "reasoning": res.reasoning, "content": res.content, "plans": plan_records,
    })
    summary.update({
        "n_plans": sum(1 for p in plan_records if p["plan"]),
        "n_flagged": sum(1 for p in plan_records if p["flags"]),
        "tokens": (res.usage or {}).get("total_tokens"),
    })
    return summary


def find_units(run_root: Path) -> list[Path]:
    """All <model>/clips/<unit> dirs with a stage_02/trajectories_raw.json;
    also accepts a root that itself contains clips/ (single-model layouts)."""
    model_dirs = [d for d in sorted(run_root.iterdir())
                  if d.is_dir() and d.name != "_frames" and (d / "clips").is_dir()]
    if not model_dirs and (run_root / "clips").is_dir():
        model_dirs = [run_root]
    units = []
    for m in model_dirs:
        for u in sorted((m / "clips").iterdir()):
            if (u / "stage_02" / "trajectories_raw.json").exists():
                units.append(u)
    return units


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True,
                   help="Annotation run root containing <model>/clips/<unit> (read-only).")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Mirrored output root (becomes build_sft's --run-dir).")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="Process at most N units.")
    p.add_argument("--shuffle-seed", type=int, default=None)
    p.add_argument("--only-with-goals", action="store_true",
                   help="Skip zero-goal units entirely (smoke tests) instead of mirroring them.")
    p.add_argument("--vlm-frame-height", type=int, default=config.DEFAULT_VLM_FRAME_HEIGHT)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Re-process units whose output already exists (cache still applies).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    out_root = ensure_dir(args.out_root)
    units = find_units(run_root)
    if args.shuffle_seed is not None:
        import random
        random.Random(args.shuffle_seed).shuffle(units)
    if args.only_with_goals:
        units = [u for u in units
                 if json.loads((u / "stage_02" / "trajectories_raw.json").read_text()).get("trajectories")]
    if args.limit is not None:
        units = units[: args.limit]

    def out_dir_for(u: Path) -> Path:
        return out_root / u.parent.parent.name / "clips" / u.name

    todo = [u for u in units
            if args.force or not (out_dir_for(u) / "stage_02" / "trajectories_raw.json").exists()]
    print(f"[stage_02b] {len(units)} units, {len(units) - len(todo)} done, {len(todo)} to do "
          f"-> {out_root}", flush=True)
    if not todo:
        return

    cfg = LabelerConfig.from_env(model=args.model, base_url=args.base_url,
                                 reasoning_effort=args.reasoning_effort)
    lab = Labeler(cfg)
    lock = threading.Lock()
    progress_path = out_root / "plans_progress.jsonl"
    counts = {"done": 0, "fail": 0, "goals": 0, "plans": 0, "flagged": 0}

    def work(u: Path) -> dict[str, Any]:
        return process_unit(u, ensure_dir(out_dir_for(u)), lab,
                            args.vlm_frame_height, args.jpeg_quality, args.no_cache)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = {ex.submit(work, u): u for u in todo}
        for fut in as_completed(futures):
            u = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001 - record and continue
                rec = {"unit_id": u.name, "error": f"{type(exc).__name__}: {exc}"}
            with lock:
                with progress_path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                counts["done"] += 1
                counts["fail"] += 1 if rec.get("error") else 0
                counts["goals"] += rec.get("n_goals") or 0
                counts["plans"] += rec.get("n_plans") or 0
                counts["flagged"] += rec.get("n_flagged") or 0
                if rec.get("error"):
                    print(f"  FAIL {u.name}: {rec['error']}", flush=True)
                elif counts["done"] % 25 == 0:
                    print(f"  [{counts['done']}/{len(todo)}] plans={counts['plans']}/"
                          f"{counts['goals']} flagged={counts['flagged']} fails={counts['fail']}",
                          flush=True)

    print(f"[stage_02b] finished: {counts['done']} units ({counts['fail']} failed), "
          f"{counts['plans']}/{counts['goals']} goals planned, {counts['flagged']} flagged. "
          f"Progress: {progress_path}", flush=True)


if __name__ == "__main__":
    main()
