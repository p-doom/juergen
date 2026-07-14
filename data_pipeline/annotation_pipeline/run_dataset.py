#!/usr/bin/env python3
"""Run stages 01-04 over a whole manifest,
maximizing throughput under a fixed per-model TPM cap.

Throughput model (no hardcoded per-model concurrency):
  * stages 01-02 run once per segment into a shared _modalities/ directory;
    oversized observation views are split into
    independent __wN window-units that reference that same array_record.
  * Each window-unit is routed at dispatch time to whichever model has the most
    TPM headroom, by a closed-loop governor: it projects each model's
    tokens/minute from the verified per-frame cost (ceil(h/28)*ceil(w/28)) and
    that model's OWN measured call latency, and only admits a unit if the model
    stays under --target-tpm. A faster model gets fewer concurrent units that
    cycle quicker; a slower one gets more — the asymmetric split emerges live.

  PYTHONPATH=. python3 -m annotation_pipeline.run_dataset \
      --manifest manifest.crowd-cast-2026-06-18.jsonl \
      --run-name full --models Kimi-K2.6,Kimi-K2.5 \
      --target-tpm 1800000 --max-workers 64

Per-model isolation: each model annotates under <run>/<model>/clips/<unit>/ with
its own response cache; shared Stages 00-02 live under <run>/_modalities/. Resumable:
a unit whose stage_04_boundaries/goals.jsonl exists (under any model) is skipped;
one phase-specific progress file records each finished parent segment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from annotation_pipeline import config
from annotation_pipeline.common import ensure_dir, read_jsonl, write_json, write_jsonl
from annotation_pipeline.frames_render import est_frame_tokens, plan_windows

PIPELINE_DIR = Path(__file__).resolve().parent
DATA_PIPELINE_DIR = PIPELINE_DIR.parent


def model_slug(model: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model) if model else "default"


def run_module(module: str, mod_args: list[str], model: str | None) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(DATA_PIPELINE_DIR) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    if model:
        env["LABELER_MODEL"] = model
    cmd = [sys.executable, "-m", f"annotation_pipeline.{module}", *mod_args]
    subprocess.run(
        cmd,
        check=True,
        env=env,
        cwd=str(DATA_PIPELINE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


# ---------------------------------------------------------------------------
# Closed-loop TPM governor — projects each model's tokens/min and admits a unit
# only if it stays under target. Dynamic, asymmetric, no hardcoded splits.
# ---------------------------------------------------------------------------


class TpmGovernor:
    """Adaptive-concurrency governor. Each model has an in-flight LIMIT that a
    control loop raises while the model's MEASURED sustained TPM is below target
    and lowers when it exceeds — so each model is driven to ~target_tpm
    regardless of token-estimate error. A loose projection ceiling guards
    against runaway admission during the measurement lag."""

    def __init__(
        self,
        models: list[str | None],
        target_tpm: float,
        init_call_s: float = 200.0,
        window_s: float = 180.0,
        start_limit: int = 24,
        max_limit: int = 80,
    ):
        self.models = list(models)
        self.target = float(target_tpm)
        self.window_s = float(window_s)
        self.max_limit = int(max_limit)
        self.cv = threading.Condition()
        self._next = 0
        self.inflight: dict[Any, dict[int, float]] = {m: {} for m in self.models}  # id -> est
        self.dur: dict[Any, float] = {m: float(init_call_s) for m in self.models}
        self.recent: dict[Any, deque] = {m: deque() for m in self.models}  # (ts, actual_tokens)
        self.limit: dict[Any, int] = {m: int(start_limit) for m in self.models}
        self.tokens: dict[Any, int] = dict.fromkeys(self.models, 0)
        self.done: dict[Any, int] = dict.fromkeys(self.models, 0)

    def _measured_tpm(self, m: Any, now: float) -> float:
        dq = self.recent[m]
        while dq and now - dq[0][0] > self.window_s:
            dq.popleft()
        return sum(t for _, t in dq) / (self.window_s / 60.0)

    def _proj_tpm(self, m: Any) -> float:
        return sum(self.inflight[m].values()) / max(20.0, self.dur[m]) * 60.0

    def acquire(self, est_tokens: float) -> tuple[Any, int]:
        """Block until a model has a free in-flight slot (under its adaptive
        limit) and isn't wildly over-committed; route to the most-free model."""
        with self.cv:
            while True:
                cands = []
                for m in self.models:
                    free = self.limit[m] - len(self.inflight[m])
                    # loose projection ceiling (1.5x target) as a runaway guard
                    proj_ok = (
                        self._proj_tpm(m) + est_tokens / max(20.0, self.dur[m]) * 60.0
                        <= self.target * 1.5
                    )
                    if free > 0 and proj_ok:
                        cands.append((free, m))
                if not cands and all(not self.inflight[m] for m in self.models):
                    cands = [
                        (1, min(self.models, key=lambda m: self.dur[m]))
                    ]  # never stall when idle
                if cands:
                    _, m = max(cands, key=lambda x: x[0])
                    h = self._next
                    self._next += 1
                    self.inflight[m][h] = float(est_tokens)
                    return m, h
                self.cv.wait(timeout=0.5)

    def release(self, m: Any, handle: int, actual_tokens: int, dur_s: float, now: float) -> None:
        with self.cv:
            self.inflight[m].pop(handle, None)
            if dur_s > 0:
                self.dur[m] = 0.75 * self.dur[m] + 0.25 * dur_s
            self.recent[m].append((now, int(actual_tokens)))
            self.tokens[m] += int(actual_tokens)
            self.done[m] += 1
            self.cv.notify_all()

    def control_tick(self, now: float) -> None:
        """AIMD on measured TPM: grow the limit when under target and saturated,
        shrink (x0.9) when over."""
        with self.cv:
            for m in self.models:
                mt = self._measured_tpm(m, now)
                if mt > self.target:
                    self.limit[m] = max(2, int(self.limit[m] * 0.9))
                elif mt < 0.9 * self.target and len(self.inflight[m]) >= self.limit[m] - 1:
                    self.limit[m] = min(self.max_limit, self.limit[m] + 2)
            self.cv.notify_all()

    def snapshot(self, now: float) -> str:
        with self.cv:
            return "  ".join(
                f"{model_slug(m)}: meas~{self._measured_tpm(m, now) / 1e6:.2f}M tpm, "
                f"inflight={len(self.inflight[m])}/{self.limit[m]}, dur~{self.dur[m]:.0f}s, done={self.done[m]}"
                for m in self.models
            )


# ---------------------------------------------------------------------------
# Per-segment work
# ---------------------------------------------------------------------------


def _unit_actual_tokens(annotation_dir: Path, variants: list[str]) -> int:
    res = annotation_dir / "annotation.json"
    if not res.exists():
        return 0
    try:
        d = json.loads(res.read_text())
    except Exception:
        return 0
    tot = 0
    for v in variants:
        for key in ("describe", "extract"):
            u = (d.get("variants", {}).get(v, {}).get(key, {}) or {}).get("usage")
            for x in u if isinstance(u, list) else [u]:
                if isinstance(x, dict):
                    tot += x.get("total_tokens") or (
                        (x.get("prompt_tokens") or 0) + (x.get("completion_tokens") or 0)
                    )
    return tot


def _boundaries_exist(run_root: Path, uid: str) -> bool:
    return any(
        (model_dir / "clips" / uid / "stage_04_boundaries" / "goals.jsonl").exists()
        for model_dir in run_root.iterdir()
        if model_dir.is_dir() and model_dir.name != "_modalities"
    )


def process_segment(
    row: dict[str, Any],
    run_root: Path,
    modalities_root: Path,
    run_dirs: dict[Any, Path],
    gov: TpmGovernor,
    ffmpeg_sem: threading.Semaphore,
    variants: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    seg = str(row["segment_id"])
    segment_dir = modalities_root / "clips" / seg
    manifest_path = segment_dir / "stage_00" / "manifest.jsonl"
    base_dir = segment_dir / "stage_01_base"
    annotation_view_dir = segment_dir / "stage_02_views" / "annotation"
    observations_path = annotation_view_dir / "observations.jsonl"

    if args.phase in ("prepare", "all"):
        ensure_dir(segment_dir / "stage_00")
        write_jsonl(manifest_path, [row])
        if args.force_prepare or not (base_dir / "manifest.json").exists():
            with ffmpeg_sem:
                s1 = [
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(base_dir),
                    "--base-fps",
                    str(args.base_fps),
                ]
                run_module("stage_01_base_modalities", s1, model=None)
        if args.force_prepare or not observations_path.exists():
            run_module(
                "stage_02_observation_view",
                [
                    "--base-dir",
                    str(base_dir),
                    "--output-dir",
                    str(annotation_view_dir),
                    "--view-name",
                    "annotation",
                    "--observation-fps",
                    str(args.annotation_fps),
                    "--idle-keep-head",
                    str(args.idle_keep_head),
                    "--idle-keep-tail",
                    str(args.idle_keep_tail),
                ],
                model=None,
            )
        if args.phase == "prepare":
            recs0 = read_jsonl(observations_path) if observations_path.exists() else []
            return {
                "n_frames": len(recs0),
                "n_windows": 0,
                "units": [],
                "n_goals_prose": 0,
                "phase": "prepare",
            }

    if not observations_path.exists():
        return {
            "n_frames": 0,
            "n_windows": 0,
            "units": [],
            "n_goals_prose": 0,
            "skipped": "no_observations",
        }
    recs = read_jsonl(observations_path)
    n = len(recs)
    if n == 0:
        # No user-activity frames (e.g. a fully-idle/blank segment now that
        # NO_OPs are dropped). Nothing to annotate — skip cleanly, not a failure.
        return {
            "n_frames": 0,
            "n_windows": 0,
            "units": [],
            "n_goals_prose": 0,
            "skipped": "no_active_frames",
        }
    per_frame = est_frame_tokens(recs[0]["image_path"])
    reserve = int(os.environ.get("LABELER_MAX_TOKENS") or 32000)
    budget = max(1, args.context_limit - reserve - args.window_safety_margin)
    max_fpw = args.max_frames_per_window or max(1, int(budget / (per_frame * 1.05)))
    # A segment is split into windows ONLY when it exceeds the context budget.
    # When it must be split, snap each cut to a command/prompt SUBMISSION or a
    # real time-gap (never mid typing-burst) so one action — and its goal — is not
    # split across two windows (which truncates the second goal's start_frame).
    times = [float(r.get("global_time_s") or 0.0) for r in recs]
    windows = plan_windows(
        n,
        max_fpw,
        args.window_overlap,
        observations=recs,
        times=times,
        slack=args.window_snap_slack,
    )
    nw = len(windows)

    units = (
        [(seg, 0, 0, n)]
        if nw <= 1
        else [(f"{seg}__w{i}", i, lo, hi) for i, (lo, hi) in enumerate(windows)]
    )

    unit_summaries = []
    for uid, wi, lo, hi in units:
        if not args.force and _boundaries_exist(run_root, uid):
            continue
        # Trailing CONTEXT buffer: a non-final window also gets the next
        # `window_tail_buffer` frames so the model can see where its last goal
        # ends (and not be forced to end on the very last frame). Stage 03 keeps
        # those frames as context only — goals that START in them are dropped.
        tail_buf = min(args.window_tail_buffer, n - hi) if (nw > 1 and wi < nw - 1) else 0
        sub = recs if nw <= 1 else recs[lo : hi + tail_buf]
        # per-unit token estimate (describe + extract per variant, + fixed overhead)
        f = len(sub)
        est = len(variants) * (2 * per_frame * f + 32000)
        model, handle = gov.acquire(est)
        t0 = time.time()
        actual: int | None = None
        try:
            udir = ensure_dir(run_dirs[model] / "clips" / uid)
            unit_view = ensure_dir(udir / "stage_02_annotation_view")
            unit_observations = unit_view / "observations.jsonl"
            write_jsonl(unit_observations, sub)
            write_json(
                unit_view / "manifest.json",
                {
                    "stage": "observation_view_subset",
                    "schema_version": 1,
                    "view_name": "annotation",
                    "source_observation_view": str(observations_path),
                    "parent_segment_id": seg,
                    "window_index": wi,
                    "n_windows": nw,
                    "tail_buffer": tail_buf,
                    "n_observations": len(sub),
                    "files": {"observations": "observations.jsonl"},
                },
            )
            if nw > 1:
                owned_hi_idx = len(sub) - 1 - tail_buf
                write_json(
                    udir / "window.json",
                    {
                        "parent_segment_id": seg,
                        "window_index": wi,
                        "n_windows": nw,
                        "source_frame_range": [
                            int(sub[0]["global_frame_idx"]),
                            int(sub[-1]["global_frame_idx"]),
                        ],
                        "owned_frame_range": [
                            int(sub[0]["global_frame_idx"]),
                            int(sub[owned_hi_idx]["global_frame_idx"]),
                        ],
                        "tail_buffer": tail_buf,
                        "n_frames": f,
                        "observation_view_shared_from": str(observations_path),
                    },
                )
            annotation_dir = udir / "stage_03_annotation"
            if args.force or not (annotation_dir / "annotation.json").exists():
                s3 = [
                    "--observations",
                    str(unit_observations),
                    "--output-dir",
                    str(annotation_dir),
                    "--vlm-frame-height",
                    str(args.vlm_frame_height),
                    "--parent-segment-id",
                    seg,
                    "--window-index",
                    str(wi),
                    "--n-windows",
                    str(nw),
                    "--tail-buffer",
                    str(tail_buf),
                ]
                if model:
                    s3 += ["--model", model]
                if args.reasoning_effort:
                    s3 += ["--reasoning-effort", args.reasoning_effort]
                run_module("stage_03_annotate", s3, model=model)
            boundaries_dir = udir / "stage_04_boundaries"
            if args.force or not (boundaries_dir / "goals.jsonl").exists():
                run_module(
                    "stage_04_refine_boundaries",
                    [
                        "--annotation-dir",
                        str(annotation_dir),
                        "--observations",
                        str(unit_observations),
                        "--output-dir",
                        str(boundaries_dir),
                        "--policy",
                        args.boundary_policy,
                    ],
                    model=None,
                )
            summ = json.loads((annotation_dir / "manifest.json").read_text())
            actual = _unit_actual_tokens(annotation_dir, variants)
            unit_summaries.append(
                {
                    "unit_id": uid,
                    "window_index": wi,
                    "model": model,
                    "n_goals_prose": summ.get("n_goals_prose"),
                    "actual_tokens": actual,
                }
            )
        finally:
            gov.release(
                model,
                handle,
                actual if actual is not None else int(est),
                time.time() - t0,
                time.time(),
            )

    return {
        "n_frames": n,
        "n_windows": nw,
        "per_frame_tokens": per_frame,
        "units": unit_summaries,
        "n_goals_prose": sum((u.get("n_goals_prose") or 0) for u in unit_summaries),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument(
        "--models", default="", help="comma list to split work across, e.g. 'Kimi-K2.6,Kimi-K2.5'"
    )
    p.add_argument("--out-root", type=Path, default=PIPELINE_DIR / "dataset_runs")
    # Throughput governor
    p.add_argument(
        "--target-tpm",
        type=float,
        default=1_800_000,
        help="Per-model tokens/min ceiling the governor stays under (buffer below the hard cap).",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=160,
        help="Total worker threads (upper bound on in-flight segments; the governor throttles below this).",
    )
    p.add_argument(
        "--ffmpeg-concurrency",
        type=int,
        default=10,
        help="Max concurrent stage-01 ffmpeg decodes (CPU bound).",
    )
    p.add_argument(
        "--init-call-s",
        type=float,
        default=200.0,
        help="Initial per-unit wall-time estimate (s); refined live per model.",
    )
    p.add_argument(
        "--start-limit",
        type=int,
        default=24,
        help="Initial per-model in-flight unit limit (AIMD grows/shrinks it toward target-tpm).",
    )
    p.add_argument(
        "--max-limit",
        type=int,
        default=80,
        help="Max per-model in-flight units the AIMD controller may grow to.",
    )
    p.add_argument(
        "--tpm-window-s",
        type=float,
        default=180.0,
        help="Sliding window (s) for measuring sustained TPM (>= a typical call duration).",
    )
    p.add_argument("--vlm-frame-height", type=int, default=720)
    # Windowing
    p.add_argument("--context-limit", type=int, default=262144)
    p.add_argument("--window-safety-margin", type=int, default=28000)
    p.add_argument("--window-overlap", type=int, default=0)
    p.add_argument(
        "--window-snap-slack",
        type=int,
        default=25,
        help="When a segment must be split, snap each window boundary "
        "within ±this many frames to a command/prompt submission "
        "(Return/Enter) or real time-gap, never mid typing-burst, "
        "so an action isn't split across windows. 0 disables.",
    )
    p.add_argument(
        "--window-tail-buffer",
        type=int,
        default=5,
        help="Trailing context frames each non-final window also sees "
        "past its owned range, so the model can judge where its "
        "last goal ends instead of ending on the final frame. "
        "Goals that START in the buffer are dropped (next window "
        "owns them). 0 disables.",
    )
    p.add_argument("--max-frames-per-window", type=int, default=0)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--base-fps", type=float, default=config.DEFAULT_BASE_FPS)
    p.add_argument("--annotation-fps", type=float, default=config.DEFAULT_ANNOTATION_FPS)
    p.add_argument("--idle-keep-head", type=int, default=config.DEFAULT_IDLE_KEEP_HEAD)
    p.add_argument("--idle-keep-tail", type=int, default=config.DEFAULT_IDLE_KEEP_TAIL)
    p.add_argument(
        "--boundary-policy",
        choices=("vision_only", "keylog_refined"),
        default="keylog_refined",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shuffle-seed", type=int, default=None)
    p.add_argument("--phase", choices=("prepare", "annotate", "all"), default="all")
    p.add_argument("--modalities-root", type=Path, default=None)
    p.add_argument(
        "--shard",
        default=None,
        help="'i/N': process only segments with (index mod N == i). For multi-node "
        "fan-out (one srun task per shard); disjoint subsets, per-shard progress file.",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--force-prepare", action="store_true")
    return p.parse_args()


def progress_path_for_phase(run_root: Path, phase: str, shard_tag: str) -> Path:
    return run_root / f"progress.{phase}{shard_tag}.jsonl"


def completed_segment_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row["segment_id"]) for row in rows if row.get("segment_id") and not row.get("error")
    }


def main() -> None:
    args = parse_args()
    models: list[str | None] = [m.strip() for m in args.models.split(",") if m.strip()] or [None]
    variants = ["prose"]  # Stage 03 is prose-only; kept for token accounting below.

    rows = read_jsonl(args.manifest)
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    # Multi-node fan-out: --shard "i/N" keeps only rows whose index mod N == i, so
    # N independent workers (e.g. srun tasks) cover disjoint segment subsets. Each
    # segment writes to its own clips/<seg>/ dir, so shards never collide; only the
    # progress file is made shard-specific to avoid concurrent-append races.
    shard_tag = ""
    if args.shard:
        try:
            si, sn = (int(x) for x in args.shard.split("/"))
        except ValueError as exc:
            raise SystemExit(f"--shard must be 'i/N', got {args.shard!r}") from exc
        if not (sn > 0 and 0 <= si < sn):
            raise SystemExit(f"--shard out of range: {args.shard}")
        rows = [r for idx, r in enumerate(rows) if idx % sn == si]
        shard_tag = f".shard{si}_of_{sn}"
    if not rows:
        raise SystemExit("no rows to process")

    run_root = ensure_dir(args.out_root / args.run_name)
    modalities_root = (
        ensure_dir(args.modalities_root)
        if args.modalities_root
        else ensure_dir(run_root / "_modalities")
    )
    # Model annotation dirs are only needed for the annotate/all phases.
    run_dirs = (
        {} if args.phase == "prepare" else {m: ensure_dir(run_root / model_slug(m)) for m in models}
    )
    progress_path = progress_path_for_phase(run_root, args.phase, shard_tag)
    done_ids = completed_segment_ids(read_jsonl(progress_path)) if progress_path.exists() else set()
    if args.force or args.force_prepare:
        done_ids = set()
    todo = [r for r in rows if str(r["segment_id"]) not in done_ids]

    gov = TpmGovernor(
        models,
        args.target_tpm,
        args.init_call_s,
        window_s=args.tpm_window_s,
        start_limit=args.start_limit,
        max_limit=args.max_limit,
    )
    ffmpeg_sem = threading.Semaphore(max(1, args.ffmpeg_concurrency))
    print(
        f"[run_dataset] run={args.run_name} phase={args.phase} "
        f"models={[m or 'env' for m in models]} variants={variants} | "
        f"modalities_root={modalities_root} | "
        f"{len(rows)} rows, {len(done_ids)} done, {len(todo)} to do | "
        f"target_tpm={args.target_tpm:,.0f}/model max_workers={args.max_workers}"
    )

    q: Queue[dict[str, Any]] = Queue()
    for r in todo:
        q.put(r)
    lock = threading.Lock()
    counter = {"done": 0, "fail": 0, "goals": 0}
    total = len(todo)
    stop = threading.Event()

    def reporter() -> None:
        i = 0
        while not stop.wait(8.0):
            gov.control_tick(time.time())  # AIMD adjust every 8s
            i += 1
            if i % 2 == 0:  # report every ~16s
                with lock:
                    d, f, g = counter["done"], counter["fail"], counter["goals"]
                print(
                    f"  [{d}/{total}] fails={f} goals={g} | {gov.snapshot(time.time())}", flush=True
                )

    def worker() -> None:
        while True:
            try:
                row = q.get_nowait()
            except Empty:
                return
            seg = str(row["segment_id"])
            try:
                rec = process_segment(
                    row,
                    run_root,
                    modalities_root,
                    run_dirs,
                    gov,
                    ffmpeg_sem,
                    variants,
                    args,
                )
                rec["segment_id"] = seg
                status = "ok"
            except subprocess.CalledProcessError as exc:
                err = (
                    exc.stderr.decode("utf-8", "replace")[-600:]
                    if isinstance(exc.stderr, bytes)
                    else str(exc.stderr or "")[-600:]
                )
                rec = {"segment_id": seg, "error": f"subprocess rc={exc.returncode}", "stderr": err}
                status = "fail"
            except Exception as exc:
                rec = {"segment_id": seg, "error": f"{type(exc).__name__}: {exc}"}
                status = "fail"
            with lock:
                with progress_path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                counter["done"] += 1
                counter["goals"] += rec.get("n_goals_prose") or 0
                if status == "fail":
                    counter["fail"] += 1
                    print(
                        f"  FAIL {seg}: {rec.get('error')} | {str(rec.get('stderr', ''))[-200:]}",
                        flush=True,
                    )

    rep = threading.Thread(target=reporter, daemon=True)
    rep.start()
    threads = [
        threading.Thread(target=worker, daemon=True) for _ in range(max(1, args.max_workers))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()

    print(
        f"[run_dataset] finished: {counter['done']} segments, {counter['fail']} failed, "
        f"{counter['goals']} goals. tokens/model: "
        f"{ {model_slug(m): gov.tokens[m] for m in models} }. Progress: {progress_path}"
    )


if __name__ == "__main__":
    main()
