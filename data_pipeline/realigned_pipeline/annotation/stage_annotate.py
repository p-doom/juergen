#!/usr/bin/env python3
"""Stage 03b (annotate): pluggable hindsight annotation over a filter view.

A fully decoupled side branch of the realigned pipeline: it reads the stage-03
filter mask, selects frames at ``--fps k`` via the shared selector
(lib/views), hands ``AnnotationUnit``s to the chosen method, and writes ONE
uniform artifact regardless of method:

  goals.jsonl     rows {goal_id, segment_id, recording_id, start_master_idx,
                  end_master_idx, instruction, instruction_variants, anchor,
                  grounding, plan?, plan_flags?, method, model,
                  prompt_pack_sha, unit_id} — goals are half-open MASTER-tick
                  intervals; view-local indices are never persisted.
  describe/       per-unit narration sidecars (reused by enrichment methods).
  units/          per-unit result records (the resume ledger + audit trail).
  calls/          per-model response caches (re-runs never re-spend tokens).
  prompts.yaml    snapshot of the method's prompt pack (sha in the manifest).
  manifest.json   method, prompt_pack_sha, models, fps, filter_id,
                  master_store_id — stage 04 refuses mismatched joins.

Methods (annotation/methods/<name>/, discovered by lib/registry):
  describe_extract  vision-only two-pass describe -> extract (INPUT: frames).
  plans             enrichment: goals artifact in -> + plan/plan_flags out
                    (INPUT: goals; needs --input-goals-dir with describe/
                    sidecars from the producing run).
  lumine_thinking   sequential day-watching with carried memory: thoughts at
                    decision points, future-blind verified (INPUT: days;
                    needs --clips-manifest for wall-clock day grouping;
                    thoughts land as single-tick goal rows, memory/log
                    trajectory as memory/<day>.jsonl sidecars).

Throughput: units are routed across ``--models`` by a closed-loop TPM governor
(lib/driver); the labeler is configured by env (LABELER_BASE_URL /
LABELER_API_KEY, see lib/labeler.py). Resumable at three levels: finished
segments (progress.jsonl), finished units (units/<id>.json), and cached calls.

Run::

    cd data_pipeline
    uv run python realigned_pipeline/annotation/stage_annotate.py \
        --filter-dir <stage-03 output> --fps 0.5 --method describe_extract \
        --models Kimi-K2.6 --output-dir <dest> --limit 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[2]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.annotation.lib.days import (  # noqa: E402
    DEFAULT_GAP_CUT_S,
    DEFAULT_TZ,
    build_day_index,
    build_day_stream,
    fmt_t,
)
from realigned_pipeline.annotation.lib.driver import model_slug, run_driver  # noqa: E402
from realigned_pipeline.annotation.lib.labeler import (  # noqa: E402
    Labeler,
    LabelerConfig,
    labeler_model,
)
from realigned_pipeline.annotation.lib.registry import (  # noqa: E402
    Method,
    MethodContext,
    discover_methods,
    load_method,
)
from realigned_pipeline.annotation.lib.units import build_units  # noqa: E402
from realigned_pipeline.lib.action_format import get_formatter  # noqa: E402
from realigned_pipeline.lib.common import (  # noqa: E402
    ensure_dir,
    normalize_dashed_argv,
    write_json,
    write_jsonl,
)
from realigned_pipeline.lib.events import load_events  # noqa: E402
from realigned_pipeline.lib.goals import (  # noqa: E402
    assert_same_artifact,
    load_goals,
    validate_goal_row,
    view_span_to_master,
)
from realigned_pipeline.lib.manifest import make_artifact_id  # noqa: E402
from realigned_pipeline.lib.views import FPS_MODES, FilterArtifact  # noqa: E402

# Routing-only per-frame token guess (720p-ish); the governor corrects with
# measured actuals, so precision doesn't matter here.
EST_TOKENS_PER_FRAME = 1500


def _segment_actions(view, keylog_path: str | None) -> list[str]:
    """Derived canonical action labels per view frame (method-internal: window
    planning + keystroke-burst snapping; never persisted)."""
    events, _ = load_events(Path(keylog_path)) if keylog_path else ([], None)
    result = get_formatter("canonical").format_segment(
        events, view.windows(), view.dead_zones, master_fps=view.master_fps
    )
    return result.labels


def _goal_rows_from_unit(unit, result: dict[str, Any], *, method: Method,
                         model: str | None, fps: float) -> tuple[list[dict[str, Any]], int]:
    """Convert a frames-method's view-local goal spans to master-interval rows
    (the ONLY place view indices become coordinates on disk)."""
    rows: list[dict[str, Any]] = []
    n_unbounded = 0
    for k, g in enumerate(result.get("goals", [])):
        sf, ef = g.get("start_frame"), g.get("end_frame")
        if sf is None or ef is None:
            n_unbounded += 1
            continue
        start_m, end_m = view_span_to_master(unit.view, int(sf), int(ef) + 1)
        row = {
            "goal_id": f"{unit.unit_id}_g{k:02d}",
            "segment_id": unit.view.segment_id,
            "recording_id": unit.view.recording_id,
            "start_master_idx": start_m,
            "end_master_idx": end_m,
            "instruction": g["instruction"],
            "instruction_variants": g.get("instruction_variants", []),
            "anchor": g.get("anchor", ""),
            "grounding": g.get("grounding", ""),
            "method": method.name,
            "model": model or "env",
            "prompt_pack_sha": method.prompts.sha,
            "unit_id": unit.unit_id,
            "annotation_fps": fps,
        }
        validate_goal_row(row)
        rows.append(row)
    return rows, n_unbounded


def _goal_rows_from_day(day_tag: str, thoughts: list[dict[str, Any]], *, method: Method,
                        model: str | None, fps: float) -> list[dict[str, Any]]:
    """Compose uniform goal rows from a day method's VERIFIED thoughts (the
    artifact only ever contains passes; raw verdicts live in the units/
    records). A thought anchors to a single master tick: [m, m+1)."""
    rows: list[dict[str, Any]] = []
    k = 0
    for th in thoughts:
        if (th.get("verify") or {}).get("verdict") != "pass":
            continue
        row = {
            "goal_id": f"{day_tag}_t{k:04d}",
            "segment_id": str(th["segment_id"]),
            "recording_id": th.get("recording_id"),
            "start_master_idx": int(th["master_idx"]),
            "end_master_idx": int(th["master_idx"]) + 1,
            "instruction": str(th["text"]),
            "kind": str(th.get("kind") or ""),
            "anchor": f"{day_tag} {fmt_t(float(th['t_day_s']))}",
            "grounding": str((th.get("verify") or {}).get("reason") or ""),
            "verify": th.get("verify"),
            "method": method.name,
            "model": model or "env",
            "prompt_pack_sha": method.prompts.sha,
            "unit_id": day_tag,
            "annotation_fps": fps,
            "day_tag": day_tag,
            "t_day_s": th.get("t_day_s"),
        }
        validate_goal_row(row)
        rows.append(row)
        k += 1
    return rows


def parse_args() -> argparse.Namespace:
    normalize_dashed_argv()  # accept pmanager's --foo_bar=value arg form
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--filter-dir", type=Path, required=True,
                   help="A stage-03 (filter) --output-dir.")
    p.add_argument("--fps", type=float, required=True,
                   help="Annotation frame rate k, independent of the training fps. With "
                        "--fps-mode exact, master_fps/k must be an integer.")
    p.add_argument("--fps-mode", choices=FPS_MODES, default="exact",
                   help="'exact' (default): k must divide the master fps. 'nearest': any "
                        "k <= master; slots take the nearest master tick (jittered spacing).")
    p.add_argument("--method", required=True, choices=sorted(discover_methods()),
                   help="Annotation method (annotation/methods/<name>/).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--models", default="Kimi-K2.6,Kimi-K2.5",
                   help="Comma list to split work across. Default: both Kimi models "
                        "(on-par quality; combined ~5M TPM quota). Pass '' to fall "
                        "back to the env-configured LABELER_MODEL.")
    p.add_argument("--input-goals-dir", type=Path, default=None,
                   help="For INPUT_KIND=goals methods (e.g. plans): the producing "
                        "stage-03b artifact to enrich.")
    # Day mode (INPUT_KIND=days methods)
    p.add_argument("--clips-manifest", type=Path, default=None,
                   help="For INPUT_KIND=days methods: the stage-00 clip manifest "
                        "(video_path/user_id per segment) used to group segments "
                        "into wall-clock user-days via mp4 mvhd creation_time.")
    p.add_argument("--tz", default=DEFAULT_TZ,
                   help="Timezone deciding where one user-day ends (day mode).")
    p.add_argument("--gap-cut-s", type=float, default=DEFAULT_GAP_CUT_S,
                   help="A recording gap longer than this splits a day into "
                        "independent chunks (day mode).")
    p.add_argument("--day-filter", nargs="*", default=None,
                   help="Day mode: only these day tags (smoke/validation runs).")
    p.add_argument("--day-t1", type=float, default=None,
                   help="Day mode: cap each day stream at this many day-seconds "
                        "(e.g. 3600 = first hour; matches the lumine 3-track "
                        "comparison scope). Partial runs — never corpus results.")
    p.add_argument("--day-index-cache", type=Path, default=None,
                   help="Day mode: cache file for the day index (the mvhd probe of "
                        "every segment video costs minutes per invocation). Reused "
                        "only when its recorded filter_id matches --filter-dir.")
    # Windowing (frames methods)
    p.add_argument("--context-limit", type=int, default=262144)
    p.add_argument("--window-safety-margin", type=int, default=28000)
    p.add_argument("--window-snap-slack", type=int, default=25,
                   help="Snap window cuts within ±this many frames to a submission "
                        "(Return/Enter) or real time-gap, never mid typing-burst.")
    p.add_argument("--window-tail-buffer", type=int, default=5,
                   help="Trailing context frames each non-final window also sees; goals "
                        "that START in them belong to the next window.")
    p.add_argument("--max-frames-per-window", type=int, default=0,
                   help="0 = derive from the context budget and measured frame size.")
    # Rendering / labeler
    p.add_argument("--vlm-frame-height", type=int, default=720,
                   help="Height fed to the labeler (<= stored height; downscales in memory).")
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--reasoning-effort", default=None,
                   help="Unset: the method's LABELER_DEFAULTS (if any), else env.")
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature. Unset: the method's "
                        "LABELER_DEFAULTS (if any), else omitted from calls.")
    p.add_argument("--param", action="append", default=None, metavar="KEY=VALUE",
                   help="Method knob override(s), passed through in ctx.params "
                        "(e.g. --param clip_frames=30 --param verify_mode=batched). "
                        "Values are strings; the method casts.")
    p.add_argument("--no-cache", action="store_true")
    # Governor
    p.add_argument("--target-tpm", default="Kimi-K2.6=2700000,Kimi-K2.5=1800000",
                   help="Per-model sustained-TPM target: 'model=tpm' comma pairs, or one "
                        "bare number for every model. Defaults leave 429 headroom under "
                        "the Azure quotas (K2.6 3M, K2.5 2M); unlisted models get 1.8M.")
    p.add_argument("--max-workers", type=int, default=64)
    p.add_argument("--start-limit", type=int, default=24)
    p.add_argument("--max-limit", type=int, default=80)
    p.add_argument("--tpm-window-s", type=float, default=180.0)
    p.add_argument("--init-call-s", type=float, default=200.0)
    # Selection
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shuffle-seed", type=int, default=None)
    p.add_argument("--shard", default=None, help="'i/N': process only rows with index mod N == i.")
    p.add_argument("--force", action="store_true",
                   help="Re-run finished units/segments (response cache still applies).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    art = FilterArtifact(args.filter_dir)
    stride = art.stride_for(args.fps, args.fps_mode)
    method = load_method(args.method)
    # Resolve the env default to its actual name so provenance (goal rows,
    # cache dirs) never records a placeholder.
    models: list[str | None] = (
        [m.strip() for m in args.models.split(",") if m.strip()] or [labeler_model()]
    )

    out_dir = ensure_dir(args.output_dir)
    units_dir = ensure_dir(out_dir / "units")
    describe_dir = ensure_dir(out_dir / "describe")
    calls_dir = ensure_dir(out_dir / "calls")
    progress_path = out_dir / "progress.jsonl"

    # Model discipline: an unset flag falls back to the method's declared
    # defaults (e.g. lumine_thinking pins temperature 0.2 / low effort).
    reasoning_effort = (args.reasoning_effort
                        if args.reasoning_effort is not None
                        else method.labeler_defaults.get("reasoning_effort"))
    temperature = (args.temperature
                   if args.temperature is not None
                   else method.labeler_defaults.get("temperature"))
    labelers = {m: Labeler(LabelerConfig.from_env(model=m, reasoning_effort=reasoning_effort,
                                                  temperature=temperature))
                for m in models}

    target_tpm: float | dict[str | None, float]
    if "=" in args.target_tpm:
        spec: dict[str, float] = {}
        for pair in args.target_tpm.split(","):
            key, sep, value = pair.partition("=")
            if not sep:
                raise SystemExit(f"--target-tpm: mixed forms in {args.target_tpm!r}")
            spec[key.strip()] = float(value)
        target_tpm = {m: spec.get(m or "", 1_800_000.0) for m in models}
    else:
        target_tpm = float(args.target_tpm)

    method_params: dict[str, Any] = {}
    for spec in args.param or []:
        key, sep, value = spec.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--param must be KEY=VALUE, got {spec!r}")
        method_params[key.strip()] = value.strip()

    def ctx_for(model: str | None, unit_id: str,
                extra_params: dict[str, Any] | None = None) -> MethodContext:
        return MethodContext(
            labeler=labelers[model],
            prompts=method.prompts,
            cache_dir=calls_dir / model_slug(model) / unit_id,
            vlm_frame_height=args.vlm_frame_height,
            jpeg_quality=args.jpeg_quality,
            no_cache=args.no_cache,
            params={**method_params, **(extra_params or {})},
        )

    input_goals_id = None
    if method.input_kind == "frames":
        rows = art.usable_rows()
        items, est_fn, run_fn = _frames_mode(args, art, method, rows, units_dir,
                                             describe_dir, ctx_for)
    elif method.input_kind == "days":
        if args.clips_manifest is None:
            raise SystemExit(f"method {method.name!r} consumes days: pass --clips-manifest")
        items, est_fn, run_fn = _days_mode(args, art, method, units_dir, out_dir, ctx_for)
    else:
        if args.input_goals_dir is None:
            raise SystemExit(f"method {method.name!r} consumes goals: pass --input-goals-dir")
        items, est_fn, run_fn, input_goals_id = _goals_mode(args, art, method, units_dir,
                                                            describe_dir, ctx_for)

    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(items)
    if args.shard:
        try:
            si, sn = (int(x) for x in args.shard.split("/"))
        except ValueError as exc:
            raise SystemExit(f"--shard must be 'i/N', got {args.shard!r}") from exc
        if not (sn > 0 and 0 <= si < sn):
            raise SystemExit(f"--shard out of range: {args.shard}")
        items = [it for idx, it in enumerate(items) if idx % sn == si]
        progress_path = out_dir / f"progress.shard{si}_of_{sn}.jsonl"
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise SystemExit("no items to process")

    print(f"[annotate] method={method.name} (input: {method.input_kind}) "
          f"fps={args.fps} ({args.fps_mode}, stride {stride:g}) prompt_pack={method.prompts.sha} "
          f"| {len(items)} items", flush=True)

    run_driver(
        items,
        item_id=lambda it: str(it["id"]),
        est_tokens=est_fn,
        run_item=run_fn,
        models=models,
        progress_path=progress_path,
        target_tpm=target_tpm,
        max_workers=args.max_workers,
        init_call_s=args.init_call_s,
        tpm_window_s=args.tpm_window_s,
        start_limit=args.start_limit,
        max_limit=args.max_limit,
        force=args.force,
    )

    # ---- finalize: units/ ledger -> one uniform goals.jsonl ----------------
    all_goals: list[dict[str, Any]] = []
    n_units = 0
    for upath in sorted(units_dir.glob("*.json")):
        n_units += 1
        unit_doc = json.loads(upath.read_text())
        all_goals.extend(unit_doc.get("goals", []))
    all_goals.sort(key=lambda g: (str(g["segment_id"]), int(g["start_master_idx"]),
                                  int(g["end_master_idx"])))
    write_jsonl(out_dir / "goals.jsonl", all_goals)
    method.prompts.snapshot_to(out_dir)

    summary = {
        "method": method.name,
        "input_kind": method.input_kind,
        "prompt_pack_sha": method.prompts.sha,
        "models": [m or "env" for m in models],
        "fps": args.fps,
        "fps_mode": args.fps_mode,
        "stride": stride,
        "master_fps": art.master_fps,
        "n_units": n_units,
        "n_goals": len(all_goals),
        "n_goals_with_plan": sum(1 for g in all_goals if g.get("plan")),
        "vlm_frame_height": args.vlm_frame_height,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "method_params": method_params,
        "filter_dir": str(art.dir),
        "input_goals_dir": str(args.input_goals_dir) if args.input_goals_dir else None,
        "clips_manifest": str(args.clips_manifest) if args.clips_manifest else None,
        "tz": args.tz if method.input_kind == "days" else None,
        "gap_cut_s": args.gap_cut_s if method.input_kind == "days" else None,
        "day_t1": args.day_t1,
    }
    write_json(out_dir / "annotate_summary.json", summary)
    write_json(out_dir / "manifest.json", {
        "artifact_type": "realigned_goals",
        "schema_version": 1,
        "goals": "goals.jsonl",
        "master_store_id": art.master_store_id,
        "filter_id": art.filter_id,
        "input_goals_id": input_goals_id,
        **summary,
    })
    print(f"[annotate] {len(all_goals)} goals from {n_units} units -> {out_dir}", flush=True)


def _frames_mode(args, art: FilterArtifact, method: Method, rows, units_dir: Path,
                 describe_dir: Path, ctx_for):
    """Items = segments; each worker builds the view + units and runs every
    not-yet-done unit of its segment (governor admission is per segment)."""
    items = [{"id": str(r["segment_id"]), "row": r} for r in rows]

    def est_tokens(item) -> float:
        # Rough: frame count at k fps ~ n_kept/stride; 2 passes + reserve.
        n_kept = int(item["row"].get("n_kept") or 0)
        n_frames = max(1, n_kept // art.stride_for(args.fps))
        return 2 * n_frames * EST_TOKENS_PER_FRAME + 32000

    def run_item(item, model: str | None) -> dict[str, Any]:
        seg = item["id"]
        view = art.segment_view(seg, args.fps, args.fps_mode)
        if not view.frames:
            return {"skipped": "empty_view", "n_units": 0, "n_goals": 0, "actual_tokens": 0}
        actions = _segment_actions(view, view.keylog_path)
        units = build_units(
            view, actions,
            context_limit=args.context_limit,
            completion_reserve=int(os.environ.get("LABELER_MAX_TOKENS") or 32000),
            safety_margin=args.window_safety_margin,
            max_frames_per_window=args.max_frames_per_window,
            snap_slack=args.window_snap_slack,
            tail_buffer=args.window_tail_buffer,
        )
        n_goals = 0
        tokens = 0
        for unit in units:
            unit_path = units_dir / f"{unit.unit_id}.json"
            if unit_path.exists() and not args.force:
                continue
            result = method.run_unit(unit, ctx_for(model, unit.unit_id))
            goal_rows, n_unbounded = _goal_rows_from_unit(
                unit, result, method=method, model=model, fps=args.fps)
            narration = str(result.get("narration") or "")
            if narration.strip():
                (describe_dir / f"{unit.unit_id}.txt").write_text(narration + "\n")
            write_json(unit_path, {
                "unit_id": unit.unit_id,
                "segment_id": view.segment_id,
                "recording_id": view.recording_id,
                "window_index": unit.window_index,
                "n_windows": unit.n_windows,
                "sent_view_range": [unit.lo, unit.hi + unit.tail_buffer],
                "owned_view_range": [unit.lo, unit.hi],
                "model": model or "env",
                "fps": args.fps,
                "n_goals_unbounded": n_unbounded,
                "extract_error": result.get("extract_error"),
                "actual_tokens": result.get("actual_tokens"),
                "goals": goal_rows,
            })
            n_goals += len(goal_rows)
            tokens += int(result.get("actual_tokens") or 0)
        return {"n_units": len(units), "n_goals": n_goals, "actual_tokens": tokens}

    return items, est_tokens, run_item


def _days_mode(args, art: FilterArtifact, method: Method, units_dir: Path,
               out_dir: Path, ctx_for):
    """Items = user-days (lib/days: wall-clock groups of the filter's usable
    segments). A day is inherently sequential — the method carries memory
    call-to-call — so the driver's parallelism axis is ACROSS days; the item
    estimate is ONE clip call (a day never has more than one call in flight)
    and live TPM comes from the per-call report hook."""
    cache = args.day_index_cache
    cached = None
    if cache is not None and cache.is_file():
        doc = json.loads(cache.read_text())
        if doc.get("filter_id") == art.filter_id and doc.get("tz") == args.tz:
            cached = doc
        else:
            print("[annotate] day-index cache is stale (filter_id/tz mismatch); rebuilding.",
                  flush=True)
    if cached is not None:
        day_rows, counters = cached["days"], cached["counters"]
        print(f"[annotate] day index (cached): {counters}", flush=True)
    else:
        day_rows, counters = build_day_index(art, args.clips_manifest, tz=args.tz)
        print(f"[annotate] day index: {counters}", flush=True)
        if cache is not None:
            write_json(cache, {"filter_id": art.filter_id, "tz": args.tz,
                               "clips_manifest": str(args.clips_manifest),
                               "counters": counters, "days": day_rows})
    if args.day_filter:
        wanted = set(args.day_filter)
        missing = wanted - {d["day_tag"] for d in day_rows}
        if missing:
            raise SystemExit(f"--day-filter tags not in the day index: {sorted(missing)}")
        day_rows = [d for d in day_rows if d["day_tag"] in wanted]
    memory_dir = ensure_dir(out_dir / "memory")
    items = [{"id": d["day_tag"], "row": d} for d in day_rows]

    def est_tokens(item) -> float:
        # Admission cost of ONE in-flight call, not the whole day: the day's
        # calls run strictly one at a time.
        return 30 * EST_TOKENS_PER_FRAME + 16000

    def run_item(item, model: str | None, report_tokens) -> dict[str, Any]:
        day_tag = str(item["id"])
        unit_path = units_dir / f"{day_tag}.json"
        if unit_path.exists() and not args.force:
            unit_doc = json.loads(unit_path.read_text())
            return {"n_clips": unit_doc.get("n_clips"), "n_goals": len(unit_doc.get("goals", [])),
                    "actual_tokens": 0, "resumed": True}
        day = build_day_stream(item["row"], art, fps=args.fps, fps_mode=args.fps_mode,
                               gap_cut_s=args.gap_cut_s, t1=args.day_t1)
        if not day.frames:
            return {"skipped": "empty_day", "n_clips": 0, "n_goals": 0, "actual_tokens": 0}
        ctx = ctx_for(model, day_tag, {
            "day_units_dir": ensure_dir(units_dir / day_tag),
            "memory_path": memory_dir / f"{day_tag}.jsonl",
            "force": args.force,
            "report_tokens": report_tokens,
        })
        result = method.run_unit({"id": day_tag, "day": day, "row": item["row"]}, ctx)
        goal_rows = _goal_rows_from_day(day_tag, result.get("thoughts", []),
                                        method=method, model=model, fps=args.fps)
        write_json(unit_path, {
            "unit_id": day_tag,
            "day_tag": day_tag,
            "user_id": day.user_id,
            "date": day.date,
            "n_segments": day.n_segments,
            "n_frames": len(day.frames),
            "n_chunks": len(day.chunks),
            "model": model or "env",
            "fps": args.fps,
            "n_clips": result.get("n_clips"),
            "n_thoughts": result.get("n_thoughts"),
            "n_pass": result.get("n_pass"),
            "n_dropped_anchor": result.get("n_dropped_anchor"),
            "verify_mode": result.get("verify_mode"),
            "selftest": result.get("selftest"),
            "actual_tokens": result.get("actual_tokens"),
            "goals": goal_rows,
        })
        return {"n_clips": result.get("n_clips"), "n_thoughts": result.get("n_thoughts"),
                "n_pass": result.get("n_pass"), "n_goals": len(goal_rows),
                "actual_tokens": int(result.get("actual_tokens") or 0)}

    return items, est_tokens, run_item


def _goals_mode(args, art: FilterArtifact, method: Method, units_dir: Path,
                describe_dir: Path, ctx_for):
    """Items = the input artifact's units (goals grouped by unit_id)."""
    in_dir = args.input_goals_dir
    manifest_path = in_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no manifest.json under {in_dir} (not a goals artifact?)")
    in_manifest = json.loads(manifest_path.read_text())
    assert_same_artifact(str(in_manifest.get("master_store_id")), art.master_store_id,
                         what="master_store_id")
    assert_same_artifact(str(in_manifest.get("filter_id")), art.filter_id, what="filter_id")
    in_fps = float(in_manifest.get("fps") or args.fps)
    input_goals_id = make_artifact_id(in_dir)

    by_unit: dict[str, list[dict[str, Any]]] = {}
    for g in load_goals(in_dir / "goals.jsonl"):
        by_unit.setdefault(str(g.get("unit_id") or g["segment_id"]), []).append(g)
    items = [{"id": uid, "goals": gs} for uid, gs in sorted(by_unit.items())]

    def est_tokens(item) -> float:
        return len(item["goals"]) * EST_TOKENS_PER_FRAME + 16000

    def run_item(item, model: str | None) -> dict[str, Any]:
        uid = item["id"]
        goals = [dict(g) for g in item["goals"]]  # never mutate the input rows
        seg = str(goals[0]["segment_id"])
        view = art.segment_view(seg, in_fps, str(in_manifest.get("fps_mode") or "exact"))
        narration_path = in_dir / "describe" / f"{uid}.txt"
        narration = narration_path.read_text() if narration_path.exists() else ""
        result = method.run_unit(
            {"unit_id": uid, "view": view, "goals": goals, "narration": narration},
            ctx_for(model, uid),
        )
        enriched = result.get("goals", [])
        for row in enriched:
            row["plan_method"] = method.name
            row["plan_model"] = model or "env"
            row["plan_prompt_pack_sha"] = method.prompts.sha
            validate_goal_row(row)
        if narration.strip():
            (describe_dir / f"{uid}.txt").write_text(narration)
        write_json(units_dir / f"{uid}.json", {
            "unit_id": uid,
            "segment_id": seg,
            "model": model or "env",
            "fps": in_fps,
            "n_plans": result.get("n_plans"),
            "n_flagged": result.get("n_flagged"),
            "actual_tokens": result.get("actual_tokens"),
            "goals": enriched,
        })
        return {"n_goals": len(enriched), "n_plans": result.get("n_plans"),
                "actual_tokens": int(result.get("actual_tokens") or 0)}

    return items, est_tokens, run_item, input_goals_id


if __name__ == "__main__":
    main()
