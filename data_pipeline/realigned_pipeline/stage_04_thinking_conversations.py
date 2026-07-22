#!/usr/bin/env python3
"""Stage 04t (thinking conversations): lumine_thinking goals -> window SFT.

AD-HOC / INTERIM: a standalone sibling of stage 04 built to produce training
data from a partial-corpus snapshot immediately. The real stage 04 (goal
mode, plans, canonical SFT) is untouched; whether think-insertion later
becomes a mode inside it or stays a sibling is an open design decision — see
HANDOFF_stage04_thinking_sft.md.

Turns a stage-03b ``lumine_thinking`` artifact (verified thoughts as
single-tick master intervals) into training conversations, following the
lumine-validated sample shape in the pipeline's own conventions:

  * WINDOWS, not goals: each user-day's stream (lib/days — same day index,
    chunking and canonical actions as stage 03b) is tiled per CHUNK into
    fixed ``--window-frames`` windows; one conversation per window that
    contains >= 1 verified thought (``--thinking-only``, default).
  * THINK-THEN-ACT: the anchor frame's assistant turn becomes
    ``<think>\\n{thought}\\n</think>\\n{action}``; actions verbatim
    (DayFrame.action, the canonical per-frame label). Training fps MUST
    equal the annotation fps (v1 lock): the anchor tick is then exactly a
    selected frame — placement is 1:1, no snapping.
  * CONTEXT BLOCK: the first user turn carries the last ``--context-thoughts``
    earlier VERIFIED thoughts of the SAME chunk ("Your thoughts so far this
    session:", one ``[+HH:MM:SS] text`` line each). Never crosses a
    recording gap.
  * EVIDENCE BOUNDARY: a thought anchored fewer than ``--min-anchor-lead``
    frames after a window cut (except at a chunk start, where the verifier's
    evidence also began) loses its <think> (the action stays; counted as
    ``n_demoted``) — its ~12-frame audit evidence would sit in the previous
    window.

The output is the canonical chat.jsonl schema (content blocks, one assistant
turn per frame) with the same manifest join-guards as stage 04, so stages
05/06 run unchanged. System prompt / terminal token stay CLI policy;
default: no terminal token (windows are arbitrary cuts, not completions).

Run::

    cd data_pipeline
    uv run python realigned_pipeline/stage_04_thinking_conversations.py \
        --filter-dir <stage-03> --goals-dir <lumine_thinking artifact> \
        --clips-manifest <stage-00/02 manifest> --fps 0.5 \
        --window-frames 48 --output-dir <dest>
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.annotation.lib.days import (  # noqa: E402
    DEFAULT_GAP_CUT_S,
    DEFAULT_TZ,
    DayStream,
    build_day_index,
    build_day_stream,
    fmt_t,
)
from realigned_pipeline.lib.common import (  # noqa: E402
    ensure_dir,
    normalize_dashed_argv,
    str2bool,
    write_json,
    write_jsonl,
)
from realigned_pipeline.lib.goals import assert_same_artifact, load_goals  # noqa: E402
from realigned_pipeline.lib.manifest import make_artifact_id  # noqa: E402
from realigned_pipeline.lib.views import FilterArtifact  # noqa: E402

# Goal-free system prompt + the think sentences (the samples' actual
# distribution: one short first-person thought at commit/switch/reconsider
# moments, silence during steady execution, an earlier-thoughts recap block).
THINKING_SYSTEM_PROMPT = (
    "You operate a desktop computer. Each user turn shows the current screen. "
    "Reply with the next action as `<dx> <dy> <scroll>` optionally followed by "
    "` ; +KEY -KEY` events, or `NO_OP` if no action. At a decision point — when "
    "you commit to, switch, or reconsider what you are doing — first write one "
    "short thought in your own voice inside <think></think> tags, then the "
    "action on the next line. Steady execution needs no thought. A \"Your "
    "thoughts so far this session:\" note at the start of a session recaps "
    "your earlier thinking."
)
CONTEXT_HEADER = "Your thoughts so far this session:"

_WORKER: dict[str, Any] = {}


def _init_worker(filter_dir: str) -> None:
    _WORKER["art"] = FilterArtifact(Path(filter_dir))


def _text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


def _image(i: str) -> dict[str, Any]:
    return {"type": "image", "image": i}


def context_block(prior: list[dict[str, Any]], keep: int) -> str:
    if not prior or keep <= 0:
        return ""
    lines = [f"[{fmt_t(float(t['t_day_s']))}] {t['instruction']}" for t in prior[-keep:]]
    return CONTEXT_HEADER + "\n" + "\n".join(lines)


def build_day_windows(task: dict[str, Any]) -> dict[str, Any]:
    """Worker: one day -> conversation rows. Failures captured, never raised."""
    day_row = task["day_row"]
    day_tag = str(day_row["day_tag"])
    try:
        day: DayStream = build_day_stream(
            day_row, _WORKER["art"], fps=task["fps"], fps_mode="exact",
            gap_cut_s=task["gap_cut_s"],
        )
        if not day.frames:
            return {"day_tag": day_tag, "status": "empty_day", "rows": []}
        # thought anchor -> day frame (exact: training fps == annotation fps)
        by_coord = {(fr.segment_id, fr.master_idx): fr.day_idx for fr in day.frames}
        anchored: dict[int, dict[str, Any]] = {}
        n_unmapped = 0
        for g in task["goals"]:
            di = by_coord.get((str(g["segment_id"]), int(g["start_master_idx"])))
            if di is None:
                n_unmapped += 1
                continue
            anchored[di] = g

        rows: list[dict[str, Any]] = []
        n_thoughts_placed = 0
        n_demoted = 0
        wf = task["window_frames"]
        for ci, chunk in enumerate(day.chunks):
            chunk_thoughts = [anchored[fr.day_idx] | {"day_idx": fr.day_idx}
                              for fr in chunk if fr.day_idx in anchored]
            for w0 in range(0, len(chunk), wf):
                win = chunk[w0: w0 + wf]
                if len(win) < 2:
                    continue
                win_ids = {fr.day_idx for fr in win}
                wt = [t for t in chunk_thoughts if t["day_idx"] in win_ids]
                if task["thinking_only"] and not wt:
                    continue
                # evidence boundary: anchors too close after a mid-chunk cut
                # lose their <think> (the audit's evidence sits in the
                # previous window); chunk-start windows are exempt.
                usable: dict[int, dict[str, Any]] = {}
                for t in wt:
                    pos = t["day_idx"] - win[0].day_idx
                    if w0 > 0 and pos < task["min_anchor_lead"]:
                        n_demoted += 1
                        continue
                    usable[t["day_idx"]] = t
                if task["thinking_only"] and not usable:
                    continue
                prior = [t for t in chunk_thoughts if t["day_idx"] < win[0].day_idx]
                block = context_block(prior, task["context_thoughts"])

                messages: list[dict[str, Any]] = []
                if task["system_prompt"]:
                    messages.append({"role": "system",
                                     "content": [_text(task["system_prompt"])]})
                thought_ids: list[str] = []
                last = len(win) - 1
                for i, fr in enumerate(win):
                    content: list[dict[str, Any]] = []
                    if i == 0 and block:
                        content.append(_text(block))
                    content.append(_image(fr.image))
                    messages.append({"role": "user", "content": content})
                    text = fr.action
                    th = usable.get(fr.day_idx)
                    if th is not None:
                        text = f"<think>\n{th['instruction']}\n</think>\n{text}"
                        thought_ids.append(str(th["goal_id"]))
                    if i == last and task["terminal_token"]:
                        text = f"{text}\n{task['terminal_token']}"
                    messages.append({"role": "assistant", "content": [_text(text)]})
                n_thoughts_placed += len(thought_ids)
                rows.append({
                    "conversation_id": f"{day_tag}_c{ci:02d}_w{w0 // wf:03d}",
                    "day_tag": day_tag,
                    "chunk_index": ci,
                    "recording_id": win[0].recording_id,
                    "segment_ids": sorted({fr.segment_id for fr in win}),
                    "t_start": fmt_t(win[0].t_day_s),
                    "t_end": fmt_t(win[-1].t_day_s),
                    "target_fps": task["fps"],
                    "window_frames": wf,
                    "goal_conditioned": False,
                    "annotation_method": "lumine_thinking",
                    "n_frames": len(win),
                    "n_turns": len(win),
                    "n_thoughts": len(thought_ids),
                    "n_context_thoughts": min(len(prior), task["context_thoughts"]),
                    "thought_ids": thought_ids,
                    "n_non_noop": sum(1 for fr in win if fr.action != "NO_OP"),
                    "messages": messages,
                })
        return {"day_tag": day_tag, "status": "ok", "rows": rows,
                "n_unmapped": n_unmapped, "n_demoted": n_demoted,
                "n_thoughts_placed": n_thoughts_placed}
    except Exception as exc:
        return {"day_tag": day_tag, "status": "failed", "rows": [],
                "error": f"{exc}", "traceback": traceback.format_exc()}


def parse_args() -> argparse.Namespace:
    normalize_dashed_argv()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--filter-dir", type=Path, required=True)
    p.add_argument("--goals-dir", type=Path, required=True,
                   help="A lumine_thinking stage-03b artifact (or snapshot).")
    p.add_argument("--clips-manifest", type=Path, required=True,
                   help="Stage-00/02 realigned clips_manifest.jsonl (day grouping).")
    p.add_argument("--day-index-cache", type=Path, default=None)
    p.add_argument("--tz", default=DEFAULT_TZ)
    p.add_argument("--gap-cut-s", type=float, default=DEFAULT_GAP_CUT_S)
    p.add_argument("--fps", type=float, required=True,
                   help="Training fps; MUST equal the artifact's annotation fps (v1).")
    p.add_argument("--window-frames", type=int, required=True,
                   help="Frames per window (12 ~ 16k tokens, 48 ~ 64k at 720p/0.5fps).")
    p.add_argument("--context-thoughts", type=int, default=8,
                   help="Earlier same-chunk thoughts on the first user turn (lumine: 8).")
    p.add_argument("--min-anchor-lead", type=int, default=12,
                   help="A thought anchored closer than this after a mid-chunk window cut "
                        "loses its <think> (audit evidence sits in the previous window).")
    p.add_argument("--thinking-only", nargs="?", const=True, type=str2bool, default=True,
                   metavar="BOOL", help="Emit only windows with >=1 thought (default).")
    p.add_argument("--system-prompt", type=str, default=None)
    p.add_argument("--system-prompt-file", type=Path, default=None)
    p.add_argument("--no-system-prompt", action="store_true")
    p.add_argument("--terminal-token", type=str, default=None,
                   help="Default None: windows are arbitrary cuts, not completions.")
    p.add_argument("--day-filter", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None, help="First N days (debug).")
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    art = FilterArtifact(args.filter_dir)

    gm_path = args.goals_dir / "manifest.json"
    if not gm_path.is_file():
        raise SystemExit(f"no manifest.json under {args.goals_dir}")
    gm = json.loads(gm_path.read_text())
    if gm.get("method") != "lumine_thinking":
        raise SystemExit(f"--goals-dir is method {gm.get('method')!r}; this stage "
                         "consumes lumine_thinking artifacts only")
    assert_same_artifact(str(gm.get("master_store_id")), art.master_store_id,
                         what="master_store_id")
    assert_same_artifact(str(gm.get("filter_id")), art.filter_id, what="filter_id")
    if float(gm.get("fps") or 0) != args.fps:
        raise SystemExit(f"--fps {args.fps} != artifact annotation fps {gm.get('fps')} "
                         "(v1 locks training fps to annotation fps)")
    goals_id = make_artifact_id(args.goals_dir)

    goals = load_goals(args.goals_dir / "goals.jsonl")
    by_day: dict[str, list[dict[str, Any]]] = {}
    for g in goals:
        by_day.setdefault(str(g.get("day_tag") or ""), []).append(g)
    by_day.pop("", None)

    if args.no_system_prompt:
        system_prompt = None
    elif args.system_prompt_file is not None:
        system_prompt = args.system_prompt_file.read_text().strip()
    elif args.system_prompt is not None:
        system_prompt = args.system_prompt
    else:
        system_prompt = THINKING_SYSTEM_PROMPT

    cache = args.day_index_cache
    day_rows = None
    if cache is not None and cache.is_file():
        doc = json.loads(cache.read_text())
        if doc.get("filter_id") == art.filter_id and doc.get("tz") == args.tz:
            day_rows = doc["days"]
    if day_rows is None:
        day_rows, counters = build_day_index(art, args.clips_manifest, tz=args.tz)
        print(f"[04t] day index: {counters}", flush=True)
        if cache is not None:
            write_json(cache, {"filter_id": art.filter_id, "tz": args.tz,
                               "clips_manifest": str(args.clips_manifest),
                               "counters": counters, "days": day_rows})

    wanted = set(by_day)
    if args.day_filter:
        wanted &= set(args.day_filter)
    day_rows = [d for d in day_rows if d["day_tag"] in wanted]
    if args.limit is not None:
        day_rows = day_rows[: args.limit]
    if not day_rows:
        raise SystemExit("no days with thoughts to process")

    tasks = [{
        "day_row": d,
        "goals": by_day[d["day_tag"]],
        "fps": args.fps,
        "gap_cut_s": args.gap_cut_s,
        "window_frames": args.window_frames,
        "context_thoughts": args.context_thoughts,
        "min_anchor_lead": args.min_anchor_lead,
        "thinking_only": bool(args.thinking_only),
        "system_prompt": system_prompt,
        "terminal_token": args.terminal_token,
    } for d in day_rows]

    n_workers = max(1, min(args.num_workers, len(tasks)))
    print(f"[04t] {len(tasks)} days, {sum(len(t['goals']) for t in tasks):,} thoughts | "
          f"window={args.window_frames}f ctx={args.context_thoughts} fps={args.fps} "
          f"| workers={n_workers}", flush=True)

    records: list[dict[str, Any]] = []
    counts: Counter = Counter()
    totals: Counter = Counter()
    with mp.Pool(n_workers, initializer=_init_worker,
                 initargs=(str(args.filter_dir),)) as pool:
        for i, res in enumerate(pool.imap_unordered(build_day_windows, tasks, chunksize=1), 1):
            counts[res["status"]] += 1
            if res["status"] == "failed":
                print(f"  FAIL {res['day_tag']}: {res.get('error')}", flush=True)
            records.extend(res["rows"])
            for k in ("n_unmapped", "n_demoted", "n_thoughts_placed"):
                totals[k] += int(res.get(k) or 0)
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} days | {len(records)} windows", flush=True)

    if not records:
        raise SystemExit("no windows built")
    records.sort(key=lambda r: str(r["conversation_id"]))

    out_dir = ensure_dir(args.output_dir)
    write_jsonl(out_dir / "conversations.jsonl", records)
    write_jsonl(out_dir / "chat.jsonl", records)
    summary = {
        "mode": "thinking_windows",
        "n_conversations": len(records),
        "n_frames_total": sum(r["n_frames"] for r in records),
        "n_thoughts_placed": totals["n_thoughts_placed"],
        "n_thoughts_demoted_boundary": totals["n_demoted"],
        "n_anchors_unmapped": totals["n_unmapped"],
        "status_counts": dict(counts),
        "fps": args.fps,
        "window_frames": args.window_frames,
        "context_thoughts": args.context_thoughts,
        "min_anchor_lead": args.min_anchor_lead,
        "thinking_only": bool(args.thinking_only),
        "has_system_prompt": system_prompt is not None,
        "terminal_token": args.terminal_token,
        "gap_cut_s": args.gap_cut_s,
        "tz": args.tz,
        "filter_dir": str(art.dir),
        "goals_dir": str(args.goals_dir),
    }
    write_json(out_dir / "conversations_summary.json", summary)
    write_json(out_dir / "manifest.json", {
        "artifact_type": "juergen_annotation_conversations",
        "schema_version": 2,
        "conversations": "conversations.jsonl",
        "chat": "chat.jsonl",
        "master_store_id": art.master_store_id,
        "filter_id": art.filter_id,
        "goals_id": goals_id,
        **summary,
    })
    print(f"[04t] {len(records)} windows, {totals['n_thoughts_placed']:,} thoughts placed "
          f"({totals['n_demoted']} demoted at boundaries, {totals['n_unmapped']} unmapped) "
          f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
