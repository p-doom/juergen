#!/usr/bin/env python3
"""Frame-by-frame stepper for one clip — for eyeballing action↔screen alignment.

Samples the raw video at ``--target-fps`` (the pipeline's bin rate, default 1 fps)
and pairs each sampled frame with the keylog action bin for that second (the exact
``aggregate_actions``/``format_action`` stage 01 uses). Step with the LEFT/RIGHT
arrow keys; the current frame index, time, and action string are shown.

    cd .../data_pipeline
    uv run python tooling/frame_stepper.py \
        --video <mp4> --keylog <msgpack> --video-fps 30 --duration 841.8 --port 8771
"""

from __future__ import annotations

import argparse
import json
import socketserver
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import cv2

# ``pipeline`` lives at the repo root; this viewer was split out to ``tooling/``
# in the data-layer restructure, so put that root on the path when run directly
# (``uv run python tooling/frame_stepper.py``).
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from pipeline.lib.common import aggregate_actions, ceil_frames, format_action

VIDEO = Path()
VIDEO_FPS = 30.0
TARGET_FPS = 1.0
ACTIONS: list[str] = []
HEIGHT = 720


INDEX = r"""<!doctype html><html><head><meta charset="utf-8"><title>frame stepper</title>
<style>
 body{margin:0;background:#0f1115;color:#e6e6e6;font:14px -apple-system,Segoe UI,Roboto,sans-serif}
 header{padding:8px 14px;background:#161a22;border-bottom:1px solid #2a2f3a;display:flex;gap:16px;align-items:center;position:sticky;top:0}
 #wrap{display:grid;place-items:center;padding:10px}
 img{max-width:100%;max-height:78vh;background:#000;border:1px solid #2a2f3a;border-radius:6px}
 .meta{font:13px ui-monospace,Menlo,monospace;color:#c6cfdf}
 .idx{font-weight:700;color:#5ce39a;font-size:16px}
 code{background:#0c0e12;padding:2px 6px;border-radius:4px;color:#ffd479}
 .noop{color:#6b7280}
 input[type=range]{width:60vw}
 button{background:transparent;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px;height:30px;cursor:pointer;padding:0 10px}
 .hint{color:#8a93a6;font-size:12px}
</style></head><body>
<header>
 <button id="prev">◀ (←)</button><button id="next">(→) ▶</button>
 <span class="idx" id="idx"></span>
 <input type="range" id="rng" min="0" value="0">
 <span class="hint">← / → to step · Home/End jump · hold Shift+→ to skip 10</span>
</header>
<div id="wrap"><img id="img" alt=""></div>
<div style="padding:8px 14px" class="meta" id="meta"></div>
<script>
let A=[], N=0, i=0, fps=1;
const $=id=>document.getElementById(id);
async function init(){
  const m=await (await fetch('/meta')).json();
  A=m.actions; N=m.n; fps=m.target_fps;
  $('rng').max=N-1;
  addEventListener('keydown',e=>{
    if(e.key==='ArrowRight')show(i+(e.shiftKey?10:1));
    else if(e.key==='ArrowLeft')show(i-(e.shiftKey?10:1));
    else if(e.key==='Home')show(0);
    else if(e.key==='End')show(N-1);
    else return; e.preventDefault();
  });
  $('prev').onclick=()=>show(i-1); $('next').onclick=()=>show(i+1);
  $('rng').oninput=()=>show(+$('rng').value);
  show(0);
}
function show(k){
  i=Math.max(0,Math.min(N-1,k));
  $('img').src='/f?i='+i;
  $('rng').value=i;
  const t=(i/fps).toFixed(1);
  $('idx').textContent='frame '+i+'/'+(N-1)+'  ·  t='+t+'s';
  const a=A[i]||'NO_OP';
  $('meta').innerHTML = a==='NO_OP' ? '<span class="noop">NO_OP (no input this second)</span>' : '<code>'+a.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))+'</code>';
}
init();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/":
            body = INDEX.encode()
            self._send(body, "text/html; charset=utf-8")
        elif parsed.path == "/meta":
            self._send(json.dumps({"n": len(ACTIONS), "target_fps": TARGET_FPS,
                                   "actions": ACTIONS}).encode(),
                       "application/json")
        elif parsed.path == "/f":
            i = int(q.get("i", ["0"])[0])
            data = self._render(i)
            if data is None:
                self.send_response(HTTPStatus.NOT_FOUND); self.end_headers(); return
            self._send(data, "image/jpeg")
        else:
            self.send_response(HTTPStatus.NOT_FOUND); self.end_headers()

    def _render(self, i: int) -> bytes | None:
        cap = cv2.VideoCapture(str(VIDEO))
        try:
            if not cap.isOpened():
                return None
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            src = min(max(0, round((i / TARGET_FPS) * VIDEO_FPS)), max(0, count - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, src)
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            if frame.shape[0] != HEIGHT:
                sc = HEIGHT / frame.shape[0]
                frame = cv2.resize(frame, (max(2, int(frame.shape[1] * sc)), HEIGHT), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return buf.tobytes() if ok else None
        finally:
            cap.release()

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_a) -> None:  # quiet
        pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    global VIDEO, VIDEO_FPS, TARGET_FPS, ACTIONS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--keylog", type=Path, required=True)
    ap.add_argument("--video-fps", type=float, required=True)
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--target-fps", type=float, default=1.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8771)
    args = ap.parse_args()

    VIDEO = args.video
    VIDEO_FPS = args.video_fps
    TARGET_FPS = args.target_fps
    n_bins = ceil_frames(args.duration, args.target_fps)
    bins, _ = aggregate_actions(args.keylog, n_bins, args.target_fps)
    ACTIONS = [format_action(b) for b in bins]
    n_active = sum(1 for a in ACTIONS if a != "NO_OP")
    print(f"frame stepper: {len(ACTIONS)} bins @ {args.target_fps}fps ({n_active} active) "
          f"over {args.duration:.0f}s of {VIDEO.name}")

    with Server((args.host, args.port), Handler) as httpd:
        url = f"http://{args.host}:{args.port}/"
        print(f"open {url}  (ssh -L {args.port}:127.0.0.1:{args.port} <host>)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
