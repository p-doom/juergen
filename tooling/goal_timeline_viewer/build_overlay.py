#!/usr/bin/env python3
"""Convert the extracted goal hierarchy (restructured.json + task_spans.json) into an
overlay_goals.json the timeline viewer renders: overarching goals -> tasks -> intervals,
with each interval's segment-local (seg, t0, t1) resolved to absolute t_day via day.json.

  python build_overlay.py --restructured <restructured.json> --spans <task_spans.json> \
      --day <DATA>/day.json --out <DATA>/overlay_goals.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# distinct, high-contrast hues per overarching goal (tasks get tints client-side)
OG_COLORS = ["#4c9be8", "#e8924c", "#5ec27a", "#c45e9e", "#b0a040", "#7a6ee0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restructured", required=True)
    ap.add_argument("--spans", required=True)
    ap.add_argument("--day", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chat", default=None,
                    help="canonical SFT chat.jsonl — if given, each interval carries its micro-goal instruction")
    args = ap.parse_args()

    restr = json.loads(Path(args.restructured).read_text())
    spans = json.loads(Path(args.spans).read_text())
    day = json.loads(Path(args.day).read_text())

    # optional: sample_id -> micro-goal instruction text (for the sidebar drill-down)
    instr: dict[str, str] = {}
    if args.chat:
        rec_ids = {s["recording_id"] for s in day["segments"]}   # day may span many recordings
        for line in Path(args.chat).open():
            if not any(r in line for r in rec_ids):
                continue
            o = json.loads(line)
            sid = o.get("sample_id", "")
            if not sid:
                continue
            for m in o.get("messages", []):
                if m.get("role") == "user":
                    c = m.get("content")
                    txt = (" ".join(p.get("text", "") for p in c
                                    if isinstance(p, dict) and p.get("type") == "text")
                           if isinstance(c, list) else (c or ""))
                    instr[sid] = txt.strip().replace("\n", " ")
                    break

    # segment -> absolute t_day base. Key by global sid (<rec8>_s####) so the same seg index
    # in different recordings doesn't collide; fall back to seg for single-recording days.
    base_sid = {s["sid"]: s["t_day"] for s in day["segments"] if s.get("sid")}
    base_seg = {s["seg"]: s["t_day"] for s in day["segments"]}

    def resolve(ivs):
        out = []
        for iv in ivs:
            b = base_sid.get(iv["sid"]) if iv.get("sid") else None
            if b is None:
                b = base_seg.get(iv["seg"])
            if b is None:
                continue
            rec = {"seg": iv["seg"], "t_start": round(b + (iv["t0"] or 0), 2),
                   "t_end": round(b + (iv["t1"] or 0), 2)}
            if iv.get("sid"):
                rec["sid"] = iv["sid"]
            if iv.get("recording_id"):
                rec["recording_id"] = iv["recording_id"]
            samp = iv.get("sample_id")
            if samp:
                rec["sample_id"] = samp
                if instr.get(samp):
                    rec["instruction"] = instr[samp]
            out.append(rec)
        out.sort(key=lambda x: x["t_start"])
        return out

    ogs = []
    for oi, og in enumerate(restr.get("overarching_goals", []), 1):
        color = OG_COLORS[(oi - 1) % len(OG_COLORS)]
        tasks = []
        for ti, t in enumerate(og.get("tasks", []), 1):
            tid = f"{oi}.{ti}"
            sp = spans.get(tid, {})
            ivs = resolve(sp.get("intervals", []))
            if not ivs:
                continue
            tasks.append({"id": tid, "title": t["title"], "status": t.get("status", ""),
                          "n_leaves": len(ivs), "n_runs": sp.get("n_runs"),
                          "t_start": ivs[0]["t_start"], "t_end": ivs[-1]["t_end"],
                          "intervals": ivs})
        if tasks:
            ogs.append({"id": str(oi), "title": og["title"], "color": color, "tasks": tasks})

    # tangents + idle as their own muted lanes (optional context)
    extra = []
    for tid, label, color in [("T", "Tangents / dead-ends", "#8a6d3b"), ("X", "Idle / navigation", "#555")]:
        sp = spans.get(tid)
        if sp and sp.get("intervals"):
            ivs = resolve(sp["intervals"])
            extra.append({"id": tid, "title": label, "color": color, "tasks": [
                {"id": tid, "title": label, "status": "", "n_leaves": len(ivs), "n_runs": sp.get("n_runs"),
                 "t_start": ivs[0]["t_start"], "t_end": ivs[-1]["t_end"], "intervals": ivs}]})

    overlay = {"recording": day.get("recording"), "span_seconds": day.get("span_seconds"),
               "overarching_goals": ogs, "extra": extra}
    Path(args.out).write_text(json.dumps(overlay, indent=2))
    nt = sum(len(o["tasks"]) for o in ogs)
    ni = sum(len(t["intervals"]) for o in ogs for t in o["tasks"])
    print(f"wrote {args.out}: {len(ogs)} overarching goals, {nt} tasks, {ni} intervals "
          f"+ {len(extra)} extra lanes")


if __name__ == "__main__":
    main()
