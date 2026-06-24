#!/usr/bin/env python3
"""Build a self-contained HTML review report for an iteration run.

For every clip and every annotated trajectory it shows, side by side: the
instruction + its register variants, the interval bounds and verify checks, the
achieved-state + grounding the labeler claimed, the reconstructed keylog
transcript for that interval, and a strip of thumbnails sampled from the raw
video across the interval. This is the human-eyeball tool: read the prompt, look
at the frames, and decide "is this what a user would type, and did the
trajectory achieve it?".
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any

import cv2

from annotation_pipeline.common import read_jsonl
from annotation_pipeline.frames_render import records_in_index_span
from annotation_pipeline.keylog_transcript import build_transcript


def _thumbs(video_path: Path, recs: list[dict[str, Any]], n: int = 8, height: int = 150) -> list[str]:
    """Thumbnails for the kept frames in a trajectory's frame-index span, read
    from the raw video at each record's source_frame_idx."""
    cap = cv2.VideoCapture(str(video_path))
    out: list[str] = []
    try:
        if not cap.isOpened() or not recs:
            return out
        picks = recs if len(recs) <= n else [recs[int(i * (len(recs) - 1) / (n - 1))] for i in range(n)]
        for r in picks:
            fidx = int(r.get("source_frame_idx", -1))
            if fidx < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            scale = height / frame.shape[0]
            frame = cv2.resize(frame, (max(2, int(frame.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                out.append(f'<img title="frame {int(r.get("global_frame_idx", -1))}" src="data:image/jpeg;base64,{b64}">')
    finally:
        cap.release()
    return out


def _metabadge(traj: dict[str, Any]) -> str:
    bits = []
    for k in ("app", "user_state", "onset", "completion"):
        v = str(traj.get(k, "") or "")
        if v:
            cls = "ok" if k == "user_state" and v == "actively_working" else ""
            bits.append(f'<span class="badge {cls}">{k}={html.escape(v)}</span>')
    return " ".join(bits)


def _traj_card(idx: int, traj: dict[str, Any], video_path: Path, transcript,
               frame_records: list[dict[str, Any]]) -> str:
    fi = int(traj.get("start_frame_idx", 0)); fj = int(traj.get("end_frame_idx", 0))
    recs = records_in_index_span(frame_records, fi, fj)
    thumbs = "".join(_thumbs(video_path, recs))
    variants = "".join(f"<li>{html.escape(v)}</li>" for v in traj.get("instruction_variants", []))
    tslice = html.escape(transcript.render(fi, fj, max_text_chars=800))
    # Color cue only: actively-working spans carry the user's own actions; idle
    # spans (agent/build running) usually drop out downstream as all-NO_OP.
    vclass = "verified" if traj.get("user_state") == "actively_working" else "unverified"
    return f"""
    <div class="card {vclass}">
      <div class="hd">#{idx} &nbsp; frames {fi}–{fj} ({len(recs)} kept)
        &nbsp; {_metabadge(traj)}</div>
      <div class="instr">{html.escape(traj.get('instruction',''))}</div>
      <ul class="variants">{variants}</ul>
      <div class="thumbs">{thumbs}</div>
      <details><summary>pass-1 description · grounding</summary>
        <div class="meta"><b>description:</b> {html.escape(str(traj.get('description','')))}<br>
        <b>grounding:</b> {html.escape(str(traj.get('grounding','')))}</div></details>
      <details><summary>keylog transcript for this interval</summary>
        <pre class="tr">{tslice}</pre></details>
    </div>"""


def _clip_section(clip_dir: Path) -> str:
    stage02 = clip_dir / "stage_02"
    traj_path = stage02 / "trajectories_raw.json"
    if not traj_path.exists():
        return f"<section><h2>{clip_dir.name}</h2><p>no stage 02 output</p></section>"
    data = json.loads(traj_path.read_text())
    summary = json.loads((stage02 / "stage02_summary.json").read_text())
    manifest = read_jsonl(clip_dir / "stage_00" / "manifest.jsonl")
    row = manifest[0]
    video_path = Path(row["video_path"])
    frame_records = read_jsonl(clip_dir / "stage_01" / "frame_records.jsonl")
    transcript = build_transcript(Path(row["keylog_path"]), frame_records=frame_records)

    cards = "".join(_traj_card(i, t, video_path, transcript, frame_records)
                    for i, t in enumerate(data.get("trajectories", [])))
    rej = stage02 / "rejected.jsonl"
    n_rej = len(read_jsonl(rej)) if rej.exists() else 0
    stat = (f"activities={summary.get('n_pass1_activities','?')} · spans={summary.get('n_spans','?')} "
            f"(active={summary.get('n_active_spans','?')}/idle={summary.get('n_idle_spans','?')}) "
            f"· trajectories={summary.get('n_trajectories','?')} · rejected={n_rej} "
            f"· variants_total={summary.get('n_variants_total','?')}")
    return f"""
    <section><h2>{clip_dir.name} <span class="sub">{row['segment_id']}</span></h2>
      <div class="stat">{stat}</div>{cards}</section>"""


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
header{padding:14px 20px;background:#161a22;position:sticky;top:0;border-bottom:1px solid #2a2f3a}
section{padding:8px 20px 24px}
h2{margin:18px 0 4px} .sub{font-size:12px;color:#8a93a6;font-weight:400}
.stat{color:#9aa3b5;font-size:13px;margin-bottom:10px}
.card{border:1px solid #2a2f3a;border-radius:8px;padding:12px;margin:10px 0;background:#161a22}
.card.verified{border-left:4px solid #36c47a} .card.unverified{border-left:4px solid #c4543a}
.hd{font-size:12px;color:#9aa3b5;margin-bottom:6px}
.instr{font-size:17px;font-weight:600;margin:4px 0;color:#fff}
.variants{margin:4px 0;color:#b9c2d4;font-size:14px}
.thumbs{display:flex;gap:4px;overflow-x:auto;margin:8px 0;padding-bottom:4px}
.thumbs img{height:150px;border-radius:4px;border:1px solid #2a2f3a}
.badge{font-size:11px;padding:1px 6px;border-radius:4px;margin-right:2px}
.badge.ok{background:#15402b;color:#5ce39a} .badge.bad{background:#451c15;color:#ff8a6a}
.badge.rep{background:#3a3415;color:#e3d05c}
details{margin-top:6px} summary{cursor:pointer;color:#8a93a6;font-size:13px}
.meta{font-size:13px;color:#b9c2d4;margin:6px 0;line-height:1.5}
pre.tr{white-space:pre-wrap;font-size:12px;background:#0c0e12;padding:8px;border-radius:6px;color:#c6cfdf;max-height:340px;overflow:auto}
"""


def build(run_dir: Path) -> Path:
    clips_root = run_dir / "clips"
    sections = "".join(_clip_section(d) for d in sorted(clips_root.iterdir()) if d.is_dir())
    out = run_dir / "review.html"
    out.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style>"
        f"<title>annotation review · {run_dir.name}</title></head><body>"
        f"<header><b>Annotation review</b> — {run_dir.name} "
        f"<span class='sub'>green=actively_working span · amber=idle_waiting span (usually dropped downstream)</span></header>"
        f"{sections}</body></html>"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    out = build(args.run_dir)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
