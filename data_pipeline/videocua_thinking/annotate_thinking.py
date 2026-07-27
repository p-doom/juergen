"""VideoCUA per-task thinking annotator (stage B).

Standalone, DECOUPLED from the crowd-cast annotation method: no day-long context
thread, no hindsight goal-recovery, no FilterArtifact / clips-manifest DayStream,
no stage_04t SFT build. It borrows only the *technique* (first-person reasoning
at decision points + a future-blind verify gate) and the low-level tooling
(labeler client, ArrayRecord frame reader) from ``realigned_pipeline``.

Inputs:
  --frames-dir  a stage_01_master_frames output dir (15fps JPEG ArrayRecord
                master store; per-segment frames/<segment_id>/frame_manifest.jsonl)
  --tasks       tasks.jsonl from build_manifest.py (goal + normalized actions)

Per task:
  1. read the 15fps master frames, SUBSAMPLE to --vlm-fps (default 5), force-including
     the frame nearest each action (so every decision point has a frame);
  2. assign each action to the frame-slot it falls in and render a readable label;
  3. slide a >=15-frame window over the sampled frames, carrying light memory;
  4. per window, ask the labeler (Qwen local / Kimi via env) for the person's thoughts
     conditioned on the task_instruction (the goal);
  5. verify each thought future-blind (per-anchor evidence cutoff);
  6. emit thinking.jsonl -- one row per PASSED thought, with its timestamp, the frame
     it sits on, and the action it precedes.

Output ``thinking.jsonl`` row:
  {task_id, segment_id, platform, task_instruction, t_s, frame_idx, image,
   kind, thought, before_action, verify, window_idx, model}
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.annotation.lib.labeler import Labeler, LabelerConfig  # noqa: E402
from realigned_pipeline.annotation.lib.prompts import PromptPack  # noqa: E402
from realigned_pipeline.annotation.lib.units import frames_to_data_urls  # noqa: E402
from realigned_pipeline.lib.common import ensure_dir, read_jsonl, write_json  # noqa: E402

KINDS = {"plan", "reorient", "decide", "react", "monitor", "wait"}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def fmt_t(s: float) -> str:
    s = max(0, int(round(s)))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"+{h}:{m:02d}:{sec:02d}" if h else f"+{m:02d}:{sec:02d}"


def render_action(a: dict[str, Any]) -> str:
    """One VideoCUA action -> a compact human-readable label."""
    t = str(a.get("type") or "").upper()
    p = a.get("params") or {}
    xy = ""
    if "x" in p and "y" in p:
        try:
            xy = f"({int(p['x'])},{int(p['y'])})"
        except (TypeError, ValueError):
            xy = ""
    btn = str(p.get("text") or "").strip()
    if t == "CLICK":
        try:
            n = int(p.get("numClicks") or 1)
        except (TypeError, ValueError):
            n = 1
        name = "DOUBLE_CLICK" if n == 2 else ("TRIPLE_CLICK" if n >= 3 else "CLICK")
        return " ".join(x for x in (name, btn, xy) if x)
    if t == "MOVE_TO":
        return f"MOVE {xy}".strip()
    if t == "DRAG_TO":
        return f"DRAG_TO {xy}".strip()
    if t in ("MOUSE_DOWN", "MOUSE_UP"):
        return f"{t} {btn}".strip()
    if t == "PRESS":
        return f"PRESS {btn}".strip()
    if t == "HOTKEY":
        return f"HOTKEY {btn}".strip()
    if t in ("KEY_DOWN", "KEY_UP"):
        return f"{t} {btn}".strip()
    if t == "TYPING":
        return 'TYPE "%s"' % str(p.get("text") or "")
    # unknown (e.g. a SCROLL the card mentions but the logs never contain): passthrough
    extra = xy or btn or (json.dumps(p, ensure_ascii=False) if p else "")
    return f"{t} {extra}".strip()


def render_slot(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "NO_OP"
    return " ; ".join(render_action(a) for a in actions)


# --------------------------------------------------------------------------- #
# Frames: load master store, subsample, assign actions
# --------------------------------------------------------------------------- #
def load_master_frames(frames_dir: Path, segment_id: str) -> list[dict[str, Any]]:
    fm = frames_dir / "frames" / segment_id / "frame_manifest.jsonl"
    if not fm.is_file():
        return []
    rows = read_jsonl(fm)
    frames = [{"record_index": int(r["record_index"]),
               "t_s": float(r["source_time_s"]),
               "image": str(r["image"])} for r in rows]
    frames.sort(key=lambda f: f["record_index"])
    return frames


def master_fps_of(frames_dir: Path, frames: list[dict[str, Any]]) -> float:
    mf = frames_dir / "manifest.json"
    if mf.is_file():
        try:
            v = float(json.loads(mf.read_text()).get("master_fps") or 0.0)
            if v > 0:
                return v
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    # fall back to the median spacing of the store
    dts = [b["t_s"] - a["t_s"] for a, b in zip(frames, frames[1:]) if b["t_s"] > a["t_s"]]
    dts.sort()
    return (1.0 / dts[len(dts) // 2]) if dts else 15.0


# Cursor-approach moves carry no decision; they inflate frame count ~10x on
# graphical apps without adding reasoning signal. Everything else is a real input.
IDLE_ACTION_TYPES = {"MOVE_TO"}


def subsample(frames: list[dict[str, Any]], master_fps: float, vlm_fps: float,
              action_times: list[float]) -> list[dict[str, Any]]:
    """Uniform subsample of the master store at ~vlm_fps, force-including the
    frame nearest each action time and the first/last frames."""
    if not frames:
        return []
    stride = max(1, round(master_fps / max(0.01, vlm_fps)))
    keep = set(range(0, len(frames), stride))
    keep.add(0)
    keep.add(len(frames) - 1)
    times = [f["t_s"] for f in frames]
    for at in action_times:
        j = bisect.bisect_right(times, at) - 1  # last frame at or before the action
        keep.add(max(0, j))
    return [frames[i] for i in sorted(keep)]


def event_anchored(frames: list[dict[str, Any]], actions: list[dict[str, Any]],
                   *, idle_floor_s: float) -> list[dict[str, Any]]:
    """Sample the frame at (or just before) each MEANINGFUL action -- the state
    the operator observed to make that decision -- plus the first/last frames and
    a low idle-floor that fills gaps longer than ``idle_floor_s`` (so a large state
    change with no input, e.g. command output appearing, still gets a frame).
    Bare cursor moves (MOVE_TO) never anchor a frame. Cost scales with the number
    of decisions, not with duration x fps."""
    if not frames:
        return []
    times = [f["t_s"] for f in frames]
    keep = {0, len(frames) - 1}
    for a in actions:
        if a["type"] in IDLE_ACTION_TYPES:
            continue
        keep.add(max(0, bisect.bisect_right(times, a["t_s"]) - 1))
    if idle_floor_s and idle_floor_s > 0:
        anchored = sorted(keep)
        for ai, bi in zip(anchored, anchored[1:]):
            gap = frames[bi]["t_s"] - frames[ai]["t_s"]
            for k in range(1, int(gap // idle_floor_s) + 1):
                keep.add(max(0, bisect.bisect_right(times, frames[ai]["t_s"] + k * idle_floor_s) - 1))
    return [frames[i] for i in sorted(keep)]


def assign_slots(sampled: list[dict[str, Any]], actions: list[dict[str, Any]]) -> None:
    """Attach to each sampled frame the actions produced in its slot
    [t_i, t_{i+1}) (last frame: [t_last, +inf))."""
    times = [f["t_s"] for f in sampled]
    for f in sampled:
        f["slot"] = []
    for a in actions:
        j = bisect.bisect_right(times, a["t_s"]) - 1
        if j < 0:
            j = 0
        sampled[j]["slot"].append(a)


# --------------------------------------------------------------------------- #
# Labeler passes
# --------------------------------------------------------------------------- #
def labels_for(window: list[dict[str, Any]]) -> list[str]:
    return [f"frame {i} | {fmt_t(f['t_s'])} | action: {render_slot(f['slot'])}"
            for i, f in enumerate(window)]


def write_thoughts(labeler: Labeler, prompts: PromptPack, *, goal: str, memory: str,
                   window: list[dict[str, Any]], max_thoughts: int,
                   cache_path: Path, no_cache: bool) -> tuple[list[dict[str, Any]], str, str]:
    urls = frames_to_data_urls([f["image"] for f in window])
    labels = labels_for(window)
    user = prompts.render("clip", goal=goal, memory=memory or "(nothing yet)",
                          max_thoughts=max_thoughts)
    parsed, _ = labeler.call_json_full(prompts.get("system"), user, images=urls,
                                       image_labels=labels, cache_path=cache_path,
                                       no_cache=no_cache)
    raw = parsed.get("thoughts") or []
    clean: list[dict[str, Any]] = []
    for th in raw:
        try:
            fi = int(th.get("frame"))
        except (TypeError, ValueError):
            continue
        if not (0 <= fi < len(window)):
            continue
        text = str(th.get("text") or "").strip()
        if not text:
            continue
        kind = str(th.get("kind") or "plan").strip().lower()
        clean.append({"frame": fi, "kind": kind if kind in KINDS else "plan", "text": text})
    memory_out = str(parsed.get("memory") or memory or "")
    log = str(parsed.get("log") or "")
    return clean, memory_out, log


def verify_thoughts(labeler: Labeler, prompts: PromptPack, *, goal: str,
                    window: list[dict[str, Any]], thoughts: list[dict[str, Any]],
                    cache_path: Path, no_cache: bool) -> None:
    """Attach a ``verify`` dict to each thought (future-blind, per-anchor cutoff)."""
    if not thoughts:
        return
    max_frame = max(t["frame"] for t in thoughts)
    ctx = window[: max_frame + 1]
    urls = frames_to_data_urls([f["image"] for f in ctx])
    labels = labels_for(ctx)
    block = "\n".join(
        f'{n + 1}. frame {t["frame"]} ({fmt_t(window[t["frame"]]["t_s"])}), kind={t["kind"]}: "{t["text"]}"'
        for n, t in enumerate(thoughts)
    )
    user = prompts.render("verify_batched", goal=goal, n_thoughts=len(thoughts),
                          max_frame=max_frame, thoughts_block=block)
    parsed, _ = labeler.call_json_full(prompts.get("verifier_system"), user, images=urls,
                                       image_labels=labels, cache_path=cache_path,
                                       no_cache=no_cache)
    verdicts = {int(v.get("n")): v for v in (parsed.get("verdicts") or []) if v.get("n") is not None}
    for n, t in enumerate(thoughts, start=1):
        v = verdicts.get(n) or {}
        t["verify"] = {
            "verdict": "pass" if str(v.get("verdict")) == "pass" else "fail",
            "violations": v.get("violations") or [],
            "reason": str(v.get("reason") or ("no verdict returned" if not v else "")),
        }


# --------------------------------------------------------------------------- #
# Per-task driver
# --------------------------------------------------------------------------- #
def process_task(task: dict[str, Any], frames_dir: Path, labeler: Labeler,
                 prompts: PromptPack, args: argparse.Namespace) -> dict[str, Any]:
    seg = task["segment_id"]
    goal = task.get("task_instruction") or ""
    actions = sorted(task.get("actions") or [], key=lambda a: a["t_s"])
    action_times = [a["t_s"] for a in actions]

    frames = load_master_frames(frames_dir, seg)
    if not frames:
        return {"segment_id": seg, "status": "no_frames", "rows": []}

    mf = master_fps_of(frames_dir, frames)
    if args.sampling == "uniform":
        sampled = subsample(frames, mf, args.vlm_fps, action_times)
    else:
        sampled = event_anchored(frames, actions, idle_floor_s=args.idle_floor_s)
    assign_slots(sampled, actions)

    cache_dir = ensure_dir(args.output_dir / "calls" / seg)
    windows = [sampled[i:i + args.window] for i in range(0, len(sampled), args.window)]

    rows: list[dict[str, Any]] = []
    all_thoughts: list[dict[str, Any]] = []
    memory = ""
    for wi, window in enumerate(windows):
        thoughts, memory, log = write_thoughts(
            labeler, prompts, goal=goal, memory=memory, window=window,
            max_thoughts=args.max_thoughts,
            cache_path=cache_dir / f"clip_{wi:03d}_write.txt", no_cache=args.no_cache)
        if not args.no_verify:
            verify_thoughts(labeler, prompts, goal=goal, window=window, thoughts=thoughts,
                            cache_path=cache_dir / f"clip_{wi:03d}_verify.txt",
                            no_cache=args.no_cache)
        for th in thoughts:
            frame = window[th["frame"]]
            t_s = round(frame["t_s"], 3)
            j = bisect.bisect_left(action_times, t_s)
            before = actions[j] if j < len(actions) else None
            # persist resolved coordinates on the thought so the cached unit
            # file is self-sufficient for the resume path.
            th["window_idx"] = wi
            th["t_s"] = t_s
            th["frame_idx"] = frame["record_index"]
            th["image"] = frame["image"]
            all_thoughts.append(th)
            verify = th.get("verify") or {"verdict": "pass" if args.no_verify else "fail"}
            if not args.keep_fails and verify.get("verdict") != "pass":
                continue
            rows.append({
                "task_id": task.get("task_id"),
                "segment_id": seg,
                "platform": task.get("platform"),
                "task_instruction": goal,
                "t_s": t_s,
                "frame_idx": frame["record_index"],
                "image": frame["image"],
                "kind": th["kind"],
                "thought": th["text"],
                "before_action": before,
                "verify": verify,
                "window_idx": wi,
                "model": labeler.config.model,
            })

    n_pass = sum(1 for t in all_thoughts if (t.get("verify") or {}).get("verdict") == "pass")
    unit = {"segment_id": seg, "goal": goal, "n_windows": len(windows),
            "n_sampled_frames": len(sampled), "n_thoughts": len(all_thoughts),
            "n_pass": n_pass, "master_fps": mf, "vlm_fps": args.vlm_fps,
            "thoughts": all_thoughts}
    write_json(ensure_dir(args.output_dir / "units") / f"{seg}.json", unit)
    return {"segment_id": seg, "status": "ok", "rows": rows,
            "n_thoughts": len(all_thoughts), "n_pass": n_pass}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-dir", type=Path, required=True,
                   help="stage_01_master_frames output dir (15fps master ArrayRecord store).")
    p.add_argument("--tasks", type=Path, required=True, help="tasks.jsonl from build_manifest.py.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prompts", type=Path, default=Path(__file__).with_name("prompts.yaml"))
    p.add_argument("--sampling", choices=["event", "uniform"], default="event",
                   help="'event' (default): one frame per meaningful action + idle floor. "
                        "'uniform': fixed --vlm-fps subsample (denser, ~10x costlier).")
    p.add_argument("--idle-floor-s", type=float, default=3.0,
                   help="event mode: also sample a frame every N seconds of input-free time.")
    p.add_argument("--vlm-fps", type=float, default=5.0, help="uniform mode: subsample rate fed to the VLM.")
    p.add_argument("--window", type=int, default=15, help="Frames per VLM clip (<= vision-image cap).")
    p.add_argument("--max-thoughts", type=int, default=5, help="Max thoughts per clip.")
    p.add_argument("--platforms", nargs="*", default=None, help="Optional platform filter.")
    p.add_argument("--limit", type=int, default=None, help="First N tasks only (debug).")
    p.add_argument("--max-workers", type=int, default=16,
                   help="Tasks annotated concurrently (they are independent). The real ceiling "
                        "is the Kimi per-model concurrent-call limit (~10) + TPM; the labeler's "
                        "Retry-After backoff absorbs 429s above it.")
    p.add_argument("--no-verify", action="store_true", help="Skip the future-blind verify gate.")
    p.add_argument("--keep-fails", action="store_true", help="Emit failed thoughts too (audit).")
    p.add_argument("--no-cache", action="store_true", help="Ignore the per-clip response cache.")
    p.add_argument("--force", action="store_true", help="Re-annotate tasks that already have a unit file.")
    # labeler overrides (else from env: LABELER_MODEL / LABELER_BASE_URL / LABELER_API_KEY)
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--reasoning-effort", default="low")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    prompts = PromptPack(args.prompts)
    prompts.snapshot_to(out_dir)

    labeler = Labeler(LabelerConfig.from_env(
        model=args.model, base_url=args.base_url, api_key=args.api_key,
        temperature=args.temperature, reasoning_effort=args.reasoning_effort))

    tasks = read_jsonl(args.tasks)
    if args.platforms:
        want = {p.lower() for p in args.platforms}
        tasks = [t for t in tasks if str(t.get("platform", "")).lower() in want]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    units_dir = out_dir / "units"
    all_rows: list[dict[str, Any]] = []
    n_ok = n_thoughts = n_pass = 0

    # Resume: replay already-annotated tasks from their unit files (no API), and
    # queue the rest. Tasks are independent -> annotate them concurrently; the one
    # shared Labeler is safe across threads (stateless calls, per-task cache paths).
    todo: list[dict[str, Any]] = []
    for task in tasks:
        unit_path = units_dir / f"{task['segment_id']}.json"
        if unit_path.is_file() and not args.force:
            unit = json.loads(unit_path.read_text())
            all_rows.extend(_rows_from_unit(unit, task, labeler.config.model, args.keep_fails))
            n_ok += 1
            n_thoughts += int(unit.get("n_thoughts") or 0)
            n_pass += int(unit.get("n_pass") or 0)
        else:
            todo.append(task)
    print(f"[vcua_thinking] {len(tasks)} tasks: {len(tasks) - len(todo)} cached, "
          f"{len(todo)} to annotate | workers={args.max_workers} model={labeler.config.model}",
          flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futs = {ex.submit(process_task, t, args.frames_dir, labeler, prompts, args): t for t in todo}
        for fut in as_completed(futs):
            seg = futs[fut]["segment_id"]
            done += 1
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001 -- one bad task must not abort the run
                print(f"[vcua_thinking] ({done}/{len(todo)}) {seg}: ERROR {e!r}", flush=True)
                continue
            if res["status"] != "ok":
                print(f"[vcua_thinking] ({done}/{len(todo)}) {seg}: {res['status']}", flush=True)
                continue
            all_rows.extend(res["rows"])
            n_ok += 1
            n_thoughts += res["n_thoughts"]
            n_pass += res["n_pass"]
            print(f"[vcua_thinking] ({done}/{len(todo)}) {seg}: "
                  f"{res['n_pass']}/{res['n_thoughts']} thoughts pass -> {len(res['rows'])} rows",
                  flush=True)

    with (out_dir / "thinking.jsonl").open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_json(out_dir / "manifest.json", {
        "artifact_type": "videocua_thinking",
        "schema_version": 1,
        "thinking": "thinking.jsonl",
        "frames_dir": str(args.frames_dir.resolve()),
        "tasks": str(args.tasks.resolve()),
        "prompt_pack_sha": prompts.sha,
        "model": labeler.config.model,
        "base_url": labeler.config.base_url,
        "sampling": args.sampling,
        "idle_floor_s": args.idle_floor_s,
        "vlm_fps": args.vlm_fps,
        "window": args.window,
        "max_workers": args.max_workers,
        "max_thoughts": args.max_thoughts,
        "verify": not args.no_verify,
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        "n_tasks": n_ok,
        "n_thoughts": n_thoughts,
        "n_pass": n_pass,
        "n_rows": len(all_rows),
    })
    print(f"[vcua_thinking] done: {n_ok} tasks, {n_pass}/{n_thoughts} thoughts pass, "
          f"{len(all_rows)} rows -> {out_dir / 'thinking.jsonl'}", flush=True)


def _rows_from_unit(unit: dict[str, Any], task: dict[str, Any], model: str,
                    keep_fails: bool) -> list[dict[str, Any]]:
    """Reconstruct thinking.jsonl rows from a cached unit file (resume path)."""
    actions = sorted(task.get("actions") or [], key=lambda a: a["t_s"])
    action_times = [a["t_s"] for a in actions]
    rows = []
    for th in unit.get("thoughts") or []:
        verify = th.get("verify") or {}
        if not keep_fails and verify.get("verdict") != "pass":
            continue
        # the unit stores frame's local idx per window; t_s/frame_idx aren't kept,
        # so recover via the stored thought's own fields if present
        t_s = th.get("t_s")
        rows.append({
            "task_id": task.get("task_id"),
            "segment_id": unit["segment_id"],
            "platform": task.get("platform"),
            "task_instruction": unit.get("goal") or task.get("task_instruction") or "",
            "t_s": t_s,
            "frame_idx": th.get("frame_idx"),
            "image": th.get("image"),
            "kind": th.get("kind"),
            "thought": th.get("text"),
            "before_action": (actions[bisect.bisect_left(action_times, t_s)]
                              if t_s is not None and bisect.bisect_left(action_times, t_s) < len(actions)
                              else None),
            "verify": verify,
            "window_idx": th.get("window_idx"),
            "model": model,
        })
    return rows


if __name__ == "__main__":
    main()
