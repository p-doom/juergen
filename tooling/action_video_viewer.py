#!/usr/bin/env python3
"""Realtime video/action viewer for one crowd-cast clip.

This intentionally uses the raw encoded video timeline: video time t is paired
with keylog action bin t. It does not apply any wall-clock stretch or drift
correction.

Example:
    cd /fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline
    uv run python tooling/action_video_viewer.py \
        --clip-dir annotation_pipeline/iteration_runs/merge_v2/clips/ghostty_term_402fe670_s38 \
        --port 8772
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import socketserver
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

# ``annotation_pipeline`` lives under ``data_pipeline/``; this viewer was split
# out to ``tooling/`` in the data-layer restructure, so put that root on the path
# when run directly (``uv run python tooling/action_video_viewer.py``).
import sys as _sys
from pathlib import Path as _Path

_DATA_PIPELINE_DIR = _Path(__file__).resolve().parents[1] / "data_pipeline"
if str(_DATA_PIPELINE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_DATA_PIPELINE_DIR))

from annotation_pipeline.common import (
    aggregate_actions,
    ceil_frames,
    format_action,
    keylog_summary,
)


VIDEO = Path()
KEYLOG = Path()
CLIP_NAME = ""
VIDEO_DURATION_S = 0.0
KEYLOG_DURATION_S = 0.0
VIDEO_FPS = 30.0
TARGET_FPS = 1.0
ACTIONS: list[str] = []


INDEX = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>action video viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111418;
      --panel: #181d23;
      --line: #303741;
      --text: #ecf0f3;
      --muted: #94a0ad;
      --accent: #43c979;
      --warn: #f2b84b;
      --code: #0c0f13;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    }
    header {
      height: 52px;
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 0 16px;
      background: #15191f;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    header strong { font-size: 14px; font-weight: 650; }
    header span { color: var(--muted); font-variant-numeric: tabular-nums; }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 430px;
      min-height: calc(100vh - 52px);
    }
    .videoPane {
      min-width: 0;
      padding: 14px;
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      gap: 10px;
    }
    video {
      width: 100%;
      max-height: calc(100vh - 132px);
      align-self: start;
      background: #000;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .controls {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      min-height: 38px;
    }
    button, select {
      height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #1d232b;
      color: var(--text);
      padding: 0 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    button:hover, select:hover { border-color: #566170; }
    .sidePane {
      min-width: 0;
      border-left: 1px solid var(--line);
      background: var(--panel);
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
    }
    .now {
      padding: 14px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 10px;
    }
    .metrics {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #14191f;
      min-width: 0;
    }
    .metric b {
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
    }
    .metric span {
      display: block;
      margin-top: 2px;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .currentAction {
      min-height: 74px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--code);
      padding: 10px;
      font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }
    .noop { color: #77818f; }
    .active { color: #f4d58a; }
    .listHead {
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .timeline {
      overflow: auto;
      padding: 6px 0;
      font: 13px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .row {
      display: grid;
      grid-template-columns: 74px minmax(0, 1fr);
      gap: 10px;
      padding: 4px 14px;
      border-left: 3px solid transparent;
    }
    .row .t { color: var(--muted); font-variant-numeric: tabular-nums; }
    .row .a {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .row.current {
      background: #202832;
      border-left-color: var(--accent);
    }
    .row.hasAction .a { color: #f4d58a; }
    .row:not(.hasAction) .a { color: #77818f; }
    .warn { color: var(--warn); }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .sidePane { border-left: 0; border-top: 1px solid var(--line); }
      video { max-height: 56vh; }
    }
  </style>
</head>
<body>
  <header>
    <strong id="clip">loading</strong>
    <span id="timelineLabel"></span>
  </header>
  <main>
    <section class="videoPane">
      <video id="video" controls preload="metadata" src="/video"></video>
      <div class="controls">
        <button id="back1">-1s</button>
        <button id="fwd1">+1s</button>
        <button id="back10">-10s</button>
        <button id="fwd10">+10s</button>
        <select id="rate">
          <option value="0.25">0.25x</option>
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
          <option value="4">4x</option>
        </select>
      </div>
    </section>
    <aside class="sidePane">
      <section class="now">
        <div class="metrics">
          <div class="metric"><b>video time</b><span id="videoTime">0.000s</span></div>
          <div class="metric"><b>action bin</b><span id="actionBin">0s</span></div>
          <div class="metric"><b>source frame</b><span id="sourceFrame">0</span></div>
          <div class="metric"><b>duration ratio</b><span id="ratio">-</span></div>
        </div>
        <div id="currentAction" class="currentAction noop">NO_OP</div>
      </section>
      <div class="listHead">
        <span>same-second action bins</span>
        <span id="rangeLabel"></span>
      </div>
      <section id="timeline" class="timeline"></section>
    </aside>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const video = $("video");
    let meta = null;

    function esc(text) {
      return String(text).replace(/[&<>]/g, (ch) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;"
      }[ch]));
    }

    function fmtTime(sec) {
      if (!Number.isFinite(sec)) return "0.000s";
      return sec.toFixed(3) + "s";
    }

    function actionAt(idx) {
      if (!meta || idx < 0 || idx >= meta.actions.length) return "NO_OP";
      return meta.actions[idx] || "NO_OP";
    }

    function seek(delta) {
      video.currentTime = Math.max(0, Math.min(video.duration || Infinity, video.currentTime + delta));
      update();
    }

    function update() {
      if (!meta) return;
      const t = video.currentTime || 0;
      const idx = Math.max(0, Math.floor(t * meta.target_fps));
      const wholeSec = idx / meta.target_fps;
      const action = actionAt(idx);
      $("videoTime").textContent = fmtTime(t);
      $("actionBin").textContent = wholeSec.toFixed(meta.target_fps === 1 ? 0 : 3) + "s";
      $("sourceFrame").textContent = Math.round(t * meta.video_fps).toString();
      $("currentAction").className = "currentAction " + (action === "NO_OP" ? "noop" : "active");
      $("currentAction").innerHTML = esc(action);

      const before = 8;
      const after = 22;
      const start = Math.max(0, idx - before);
      const end = Math.min(meta.actions.length - 1, idx + after);
      $("rangeLabel").textContent = (start / meta.target_fps).toFixed(0) + "s-" + (end / meta.target_fps).toFixed(0) + "s";
      let html = "";
      for (let i = start; i <= end; i++) {
        const a = actionAt(i);
        const classes = ["row"];
        if (i === idx) classes.push("current");
        if (a !== "NO_OP") classes.push("hasAction");
        html += `<div class="${classes.join(" ")}"><div class="t">${(i / meta.target_fps).toFixed(0)}s</div><div class="a">${esc(a)}</div></div>`;
      }
      $("timeline").innerHTML = html;
    }

    async function init() {
      meta = await (await fetch("/meta")).json();
      $("clip").textContent = meta.clip_name;
      $("timelineLabel").textContent =
        `video ${meta.video_duration_s.toFixed(1)}s, keylog ${meta.keylog_duration_s.toFixed(1)}s, bins @ ${meta.target_fps} fps`;
      const ratio = meta.video_duration_s > 0 ? meta.keylog_duration_s / meta.video_duration_s : 0;
      $("ratio").innerHTML = ratio >= 1.15 || ratio <= 0.85
        ? `<span class="warn">${ratio.toFixed(3)}</span>`
        : ratio.toFixed(3);
      $("back1").onclick = () => seek(-1);
      $("fwd1").onclick = () => seek(1);
      $("back10").onclick = () => seek(-10);
      $("fwd10").onclick = () => seek(10);
      $("rate").onchange = () => { video.playbackRate = Number($("rate").value); };
      video.addEventListener("timeupdate", update);
      video.addEventListener("seeked", update);
      video.addEventListener("loadedmetadata", update);
      addEventListener("keydown", (event) => {
        if (event.target && ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
        if (event.key === "ArrowLeft") { seek(event.shiftKey ? -10 : -1); event.preventDefault(); }
        if (event.key === "ArrowRight") { seek(event.shiftKey ? 10 : 1); event.preventDefault(); }
      });
      update();
    }

    init();
  </script>
</body>
</html>
"""


def read_first_jsonl(path: Path) -> dict[str, Any]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                return value
    raise ValueError(f"No rows in {path}")


def load_clip_row(clip_dir: Path) -> dict[str, Any]:
    manifest = clip_dir / "stage_00" / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing clip manifest: {manifest}")
    return read_first_jsonl(manifest)


def choose(args: argparse.Namespace, row: dict[str, Any] | None, key: str) -> Any:
    value = getattr(args, key)
    if value is not None:
        return value
    if row is not None and key in row:
        return row[key]
    return None


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    if not header:
        return None
    match = re.match(r"bytes=(\d*)-(\d*)$", header)
    if not match:
        return None
    start_s, end_s = match.groups()
    if start_s == "" and end_s == "":
        return None
    if start_s == "":
        suffix = int(end_s)
        if suffix <= 0:
            return None
        start = max(0, size - suffix)
        end = size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    if start >= size:
        return None
    return start, min(end, size - 1)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            self.send_bytes(INDEX.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/meta":
            self.send_json(
                {
                    "clip_name": CLIP_NAME,
                    "video_duration_s": VIDEO_DURATION_S,
                    "keylog_duration_s": KEYLOG_DURATION_S,
                    "video_fps": VIDEO_FPS,
                    "target_fps": TARGET_FPS,
                    "n_actions": len(ACTIONS),
                    "video_path": str(VIDEO),
                    "keylog_path": str(KEYLOG),
                    "actions": ACTIONS,
                }
            )
        elif self.path == "/video":
            self.send_file(VIDEO)
        elif self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def send_json(self, value: Any) -> None:
        self.send_bytes(json.dumps(value).encode("utf-8"), "application/json")

    def send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        byte_range = parse_range(self.headers.get("Range"), size)
        if byte_range is None:
            start, end = 0, size - 1
            status = HTTPStatus.OK
        else:
            start, end = byte_range
            status = HTTPStatus.PARTIAL_CONTENT

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, *_args: Any) -> None:
        pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    global ACTIONS, CLIP_NAME, KEYLOG, KEYLOG_DURATION_S, TARGET_FPS
    global VIDEO, VIDEO_DURATION_S, VIDEO_FPS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", type=Path, help="Clip dir containing stage_00/manifest.jsonl.")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--keylog", type=Path)
    parser.add_argument("--video_duration_s", "--duration", dest="video_duration_s", type=float)
    parser.add_argument("--keylog_duration_s", dest="keylog_duration_s", type=float)
    parser.add_argument("--video_fps", "--video-fps", dest="video_fps", type=float)
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    args = parser.parse_args()

    row = load_clip_row(args.clip_dir) if args.clip_dir else None
    video = choose(args, row, "video")
    keylog = choose(args, row, "keylog")
    if video is None and row is not None:
        video = row.get("video_path")
    if keylog is None and row is not None:
        keylog = row.get("keylog_path")
    if video is None or keylog is None:
        raise SystemExit("Provide --clip-dir or both --video and --keylog.")

    VIDEO = Path(video)
    KEYLOG = Path(keylog)
    VIDEO_DURATION_S = float(choose(args, row, "video_duration_s") or 0.0)
    VIDEO_FPS = float(choose(args, row, "video_fps") or 30.0)
    TARGET_FPS = float(args.target_fps)
    if not VIDEO.exists():
        raise FileNotFoundError(VIDEO)
    if not KEYLOG.exists():
        raise FileNotFoundError(KEYLOG)

    summary = keylog_summary(KEYLOG)
    KEYLOG_DURATION_S = float(
        choose(args, row, "keylog_duration_s") or summary.get("keylog_duration_s") or 0.0
    )
    action_duration_s = max(VIDEO_DURATION_S, KEYLOG_DURATION_S)
    n_bins = ceil_frames(action_duration_s, TARGET_FPS)
    bins, stats = aggregate_actions(KEYLOG, n_bins, TARGET_FPS)
    ACTIONS = [format_action(action_bin) for action_bin in bins]
    CLIP_NAME = (
        str(row.get("segment_id") or row.get("clip_id") or args.clip_dir.name)
        if row is not None
        else VIDEO.name
    )
    n_active = sum(1 for action in ACTIONS if action != "NO_OP")
    ratio = (KEYLOG_DURATION_S / VIDEO_DURATION_S) if VIDEO_DURATION_S > 0 else 0.0
    print(
        f"action video viewer: {CLIP_NAME} | "
        f"video={VIDEO_DURATION_S:.3f}s keylog={KEYLOG_DURATION_S:.3f}s r={ratio:.3f} | "
        f"{len(ACTIONS)} bins @ {TARGET_FPS:g} fps, {n_active} active, "
        f"{stats.n_events} parsed events"
    )

    with Server((args.host, args.port), Handler) as httpd:
        url = f"http://{args.host}:{args.port}/"
        print(f"open {url}")
        print(f"ssh tunnel if needed: ssh -L {args.port}:127.0.0.1:{args.port} <host>")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
