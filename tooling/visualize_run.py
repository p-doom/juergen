#!/usr/bin/env python3
"""Interactive inspector for the v2 (describe→extract, timestamp-free) pipeline.

Serves a single-page dashboard over an iteration run (default
``annotation_pipeline/iteration_runs/<run>``). For each clip you can:

  - play the RAW recording (HTTP Range streaming) and step through the exact
    0.5-fps frames the VLM was sent (the kept stage-01 stream);
  - the describe pass: the full narration, the model's THINKING (reasoning),
    the full raw response, and the prompt that was sent;
  - the list of GOALS the extract pass recovered (instruction + register
    variants + anchor + grounding + start/end frame).

Reads ``stage_02/stage02_result.json`` (self-contained) per clip.

Run:
    cd .../data_pipeline
    uv run python tooling/visualize_run.py --port 8765
    # then SSH-forward the port and open http://127.0.0.1:8765/
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import socketserver
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

# ``pipeline`` lives at the repo root; this viewer was split out to ``tooling/``
# in the data-layer restructure, so put that root on the path when run directly
# (``uv run python tooling/visualize_run.py``).
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from pipeline.lib.common import read_jsonl
from pipeline.lib.image_store import (
    is_arrayrecord_image_uri,
    parse_arrayrecord_image_uri,
    read_jpeg_bytes,
)

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = PIPELINE_DIR / "iteration_runs"
# Media (frames/videos) may only be served from under these roots.
MEDIA_ROOTS = [PIPELINE_DIR.resolve(), Path("/fast/project/HFMI_SynergyUnit").resolve()]
MEDIA_EXTS = {".jpg", ".jpeg", ".png", ".mp4"}

RUN_ROOT = DEFAULT_RUN_ROOT  # overridden in main()


# ---------------------------------------------------------------------------
# Loading / assembly
# ---------------------------------------------------------------------------


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def list_runs() -> list[dict[str, Any]]:
    if not RUN_ROOT.is_dir():
        return []
    out = []
    for d in (p for p in RUN_ROOT.iterdir() if (p / "clips").is_dir()):
        clips = sorted(c.name for c in (d / "clips").iterdir() if c.is_dir())
        marker_times = [
            (d / "run_summary.json").stat().st_mtime if (d / "run_summary.json").exists() else 0.0,
            (d / "judge.json").stat().st_mtime if (d / "judge.json").exists() else 0.0,
        ]
        for clip in clips:
            for rel in ("stage_02/stage02_result.json", "stage_02/stage02_summary.json"):
                p = d / "clips" / clip / rel
                if p.exists():
                    marker_times.append(p.stat().st_mtime)
        out.append({"name": d.name, "n_clips": len(clips), "mtime": max(marker_times)})
    out.sort(key=lambda r: (r["mtime"], r["name"]), reverse=True)
    return out


def build_clip(clip_dir: Path) -> dict[str, Any]:
    s00 = read_jsonl(clip_dir / "stage_00" / "manifest.jsonl")
    row = s00[0] if s00 else {}
    s02 = clip_dir / "stage_02"
    summary = read_json(s02 / "stage02_summary.json", {})
    result = read_json(s02 / "stage02_result.json", {}) or {}
    frame_records = read_jsonl(clip_dir / "stage_01" / "frame_records.jsonl")

    # Kept-frame stream = exactly what stage 01 sampled (0.5 fps, NO_OP-capped).
    # Play it back to SEE the clip; the same frames are what the VLM was sent.
    s01_summary = read_json(clip_dir / "stage_01" / "frames_actions_summary.json", {})
    kept_frames = [
        {"idx": r.get("global_frame_idx"), "action": r.get("action"),
         # ar:// grain URIs are served by /image as-is; loose files are resolved.
         "img": r["image_path"] if is_arrayrecord_image_uri(r["image_path"]) else str(Path(r["image_path"]).resolve())}
        for r in frame_records[:6000]
        if r.get("image_path")
    ]
    sampling = {
        "target_fps": s01_summary.get("target_fps"),
        "noop_keep_head": s01_summary.get("noop_keep_head"),
        "noop_keep_tail": s01_summary.get("noop_keep_tail"),
        "n_frames": len(frame_records),
        "n_non_noop": sum(1 for r in frame_records if r.get("action") != "NO_OP"),
        "n_noop_dropped": s01_summary.get("n_noop_dropped"),
    }

    video = row.get("video_path")
    return {
        "clip_key": clip_dir.name,
        "segment_id": row.get("segment_id") or result.get("segment_id"),
        "video": str(Path(video).resolve()) if video and Path(video).exists() else None,
        "n_kept_frames": len(frame_records),
        "fps": row.get("video_fps"),
        "summary": summary,
        "sampling": sampling,
        "kept_frames": kept_frames,
        "n_images_sent": result.get("n_images_sent"),
        "model": result.get("model"),
        "variants": result.get("variants", {}),
    }


def build_run(name: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / name
    clips_root = run_dir / "clips"
    clip_keys = sorted(c.name for c in clips_root.iterdir() if c.is_dir()) if clips_root.is_dir() else []
    judge_path = run_dir / "judge.json"
    judge = read_json(judge_path, {})
    if judge and judge_path.exists():
        # Do not show stale judge numbers after regenerating trajectories without
        # re-running the judge.
        judge_mtime = judge_path.stat().st_mtime
        for key in clip_keys:
            traj_path = clips_root / key / "stage_02" / "trajectories_raw.json"
            if traj_path.exists() and traj_path.stat().st_mtime > judge_mtime:
                judge = {}
                break
    return {
        "run": name,
        "judge": {"pass_rate": judge.get("pass_rate"), "n_pass": judge.get("n_pass"),
                  "n_examples": judge.get("n_examples"), "diversity": judge.get("diversity")},
        "clips": [build_clip(clips_root / k) for k in clip_keys],
    }


# ---------------------------------------------------------------------------
# Media serving (root-gated)
# ---------------------------------------------------------------------------


def _under_roots(p: Path) -> bool:
    return any(p == r or r in p.parents for r in MEDIA_ROOTS)


def safe_media(raw_path: str) -> Path | None:
    if not raw_path:
        return None
    try:
        p = Path(raw_path).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    if p.suffix.lower() not in MEDIA_EXTS:
        return None
    if not _under_roots(p):
        return None
    return p if p.exists() else None


def grain_jpeg(ar_uri: str) -> bytes | None:
    """Decode a JPEG from an ar:///shard.array_record#idx URI, root-gated."""
    try:
        shard, _ = parse_arrayrecord_image_uri(ar_uri)
    except ValueError:
        return None
    shard = shard.resolve()
    if not _under_roots(shard) or not shard.exists():
        return None
    return read_jpeg_bytes(ar_uri)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Annotation Run Inspector</title>
<style>
  :root{--bg:#0f1115;--panel:#161a22;--ink:#e6e6e6;--muted:#8a93a6;--line:#2a2f3a;--accent:#5ce39a;--bad:#ff8a6a;--code:#0c0e12;}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
  header{position:sticky;top:0;z-index:5;background:#13161d;border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:14px;align-items:center}
  header b{font-size:16px} select{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;height:32px;padding:0 8px}
  .judge{color:var(--muted);font-size:13px}
  .layout{display:grid;grid-template-columns:230px minmax(0,1fr);gap:16px;max-width:1500px;margin:0 auto;padding:16px}
  nav{position:sticky;top:60px;align-self:start;border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:8px;max-height:88vh;overflow:auto}
  .clip-btn{display:block;width:100%;text-align:left;background:transparent;border:1px solid transparent;color:var(--ink);border-radius:6px;padding:7px 8px;margin:2px 0;cursor:pointer;font:inherit}
  .clip-btn.active{background:rgba(92,227,154,.1);border-color:rgba(92,227,154,.3);color:var(--accent);font-weight:600}
  .clip-btn .sub{display:block;color:var(--muted);font-size:11px}
  h2{font-size:15px;margin:18px 0 6px;border-bottom:1px solid var(--line);padding-bottom:4px}
  video{width:100%;max-height:60vh;background:#000;border-radius:8px}
  .keptplayer{border:1px solid var(--line);border-radius:8px;background:#000;overflow:hidden}
  .kp-screen{position:relative;display:grid;place-items:center;min-height:300px;background:#000}
  .kp-screen img{width:100%;max-height:60vh;object-fit:contain;background:#000}
  .kp-count{position:absolute;top:8px;right:10px;background:rgba(0,0,0,.5);padding:2px 7px;border-radius:4px;font-size:12px}
  .kp-ctrls{display:grid;grid-template-columns:auto auto 1fr auto;gap:10px;align-items:center;padding:8px 12px;background:var(--panel);border-top:1px solid var(--line)}
  .kp-ctrls button{background:transparent;color:var(--ink);border:1px solid var(--line);border-radius:6px;height:30px;cursor:pointer}
  .kp-meta{padding:8px 12px;color:var(--muted);font-size:13px;font-family:ui-monospace,Menlo,monospace}
  .stat{color:var(--muted);font-size:13px;margin:6px 0 12px}
  details{border:1px solid var(--line);border-radius:8px;margin:8px 0;background:var(--panel)}
  details>summary{cursor:pointer;padding:9px 12px;font-weight:600;list-style:none}
  details>summary::-webkit-details-marker{display:none}
  details>summary:before{content:"▸ ";color:var(--muted)} details[open]>summary:before{content:"▾ "}
  .inner{padding:0 12px 12px}
  .thumbs{display:flex;gap:4px;overflow-x:auto;padding:6px 0}
  .thumbs img{height:120px;border-radius:4px;border:1px solid var(--line);cursor:zoom-in}
  pre{white-space:pre-wrap;background:var(--code);border-radius:6px;padding:9px;font:12px/1.45 ui-monospace,Menlo,monospace;color:#c6cfdf;max-height:320px;overflow:auto}
  .json{color:#9ad}
  .card{border:1px solid var(--line);border-radius:8px;margin:10px 0;background:var(--panel);padding:12px}
  .card.ok{border-left:4px solid var(--accent)} .card.no{border-left:4px solid var(--bad)}
  .instr{font-size:16px;font-weight:600;margin:2px 0 6px}
  .variants{color:#b9c2d4;font-size:13px;margin:0 0 6px;padding-left:18px}
  .badge{font-size:11px;padding:1px 6px;border-radius:4px;margin-right:3px;background:#23262e;color:#9aa3b5}
  .badge.t{background:#15402b;color:#5ce39a} .badge.f{background:#451c15;color:#ff8a6a}
  .row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .lab{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.05em;margin:8px 0 2px}
  @media(max-width:980px){.layout{grid-template-columns:1fr}.row2{grid-template-columns:1fr}}
</style></head><body>
<header><b>Annotation Run Inspector</b><select id="run"></select><span class="judge" id="judge"></span></header>
<div class="layout"><nav id="nav"></nav><div id="main"></div></div>
<script>
const $=id=>document.getElementById(id);
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const img=p=>`/image?path=${encodeURIComponent(p)}`;
const vid=p=>`/video?path=${encodeURIComponent(p)}`;
const S={run:null,clip:0};
const api=async p=>{const r=await fetch(p);if(!r.ok)throw new Error(await r.text());return r.json();};

async function init(){
  const runs=await api('/api/runs');
  $('run').innerHTML=runs.map(r=>`<option value="${esc(r.name)}">${esc(r.name)} (${r.n_clips} clips)</option>`).join('');
  if(!runs.length){$('main').innerHTML='<p>No runs under iteration_runs/.</p>';return;}
  $('run').onchange=()=>load($('run').value);
  await load(runs[0].name);
}
async function load(name){S.run=await api(`/api/run?name=${encodeURIComponent(name)}`);S.clip=0;
  const j=S.run.judge||{};
  $('judge').textContent=j.pass_rate!=null?`judge ${j.n_pass}/${j.n_examples} = ${Math.round(j.pass_rate*100)}% · diversity banned=${j.diversity?.banned_opening_rate} distinct=${j.diversity?.distinct_first_word_ratio}`:'';
  renderNav();renderClip();}
function renderNav(){$('nav').innerHTML=S.run.clips.map((c,i)=>{
  const s=c.summary||{};
  return `<button class="clip-btn ${i===S.clip?'active':''}" data-i="${i}">${esc(c.clip_key)}
    <span class="sub">${c.n_kept_frames??0} frames · goals ${s.n_goals_prose??0}</span></button>`;
}).join('');
document.querySelectorAll('.clip-btn').forEach(b=>b.onclick=()=>{S.clip=+b.dataset.i;renderNav();renderClip();window.scrollTo(0,0);});}

function pre(t){return `<pre>${esc(t||'')}</pre>`;}
function block(title,open,inner){return `<details ${open?'open':''}><summary>${esc(title)}</summary><div class="inner">${inner}</div></details>`;}
function truncBadge(fr){return fr==='length'?' <span class="badge f">TRUNCATED</span>':'';}

function framesInSpan(frames,a,b){return (frames||[]).filter(f=>Number(f.idx)>=a&&Number(f.idx)<=b);}
function goalsBlock(goals){
  if(!goals||!goals.length)return '<p class="lab">no goals</p>';
  return goals.map(g=>{
    const hasSpan = g.start_frame!=null && g.end_frame!=null;
    return `<div class="card ok">
    <div class="instr">${esc(g.instruction)}</div>
    ${(g.instruction_variants||[]).length?`<ul class="variants">${g.instruction_variants.map(v=>`<li>${esc(v)}</li>`).join('')}</ul>`:''}
    ${hasSpan?`<div class="lab">goal trajectory · frames ${g.start_frame}–${g.end_frame}</div>
       <div class="keptplayer goalfp" data-start="${g.start_frame}" data-end="${g.end_frame}"></div>`
      :'<div class="lab">no frame span returned</div>'}
    ${g.anchor?`<div class="lab">anchor</div><div style="color:#b9c2d4;font-size:13px">${esc(g.anchor)}</div>`:''}
    ${g.grounding?`<div class="lab">grounding</div><div style="color:#8a93a6;font-size:12px">${esc(g.grounding)}</div>`:''}
  </div>`;}).join('');
}
function describeBlock(d){
  d=d||{};
  let body=d.error?`<div class="card no">error: ${esc(d.error)}</div>`:'';
  body+=`<div class="lab">narration</div>${pre(d.description||d.content)}`;
  body+=block('thinking (reasoning)',false,pre(d.reasoning||'(none returned)'));
  body+=block('full raw response',false,pre(d.content));
  body+=block('prompt sent',false,pre(d.prompt));
  return `<div class="lab" style="margin-top:6px">DESCRIBE${truncBadge(d.finish_reason)}</div>${body}`;
}
function extractBlock(e){
  e=e||{};
  let body=e.error?`<div class="card no">error: ${esc(e.error)}</div>`:'';
  body+=`<div class="lab">${(e.goals||[]).length} goals</div>`+goalsBlock(e.goals);
  body+=block('thinking (reasoning)',false,pre(e.reasoning||'(none returned)'));
  body+=block('full raw response',false,pre(e.content));
  body+=block('prompt sent',false,pre(e.prompt));
  return `<div class="lab" style="margin-top:12px">EXTRACT${truncBadge(e.finish_reason)}</div>${body}`;
}
function variantSection(v){
  if(!v)return `<p class="lab">no annotation on disk</p>`;
  return describeBlock(v.describe)+extractBlock(v.extract);
}

// Generic frame-player: plays the ACTUAL kept (stage-01) JPEGs a sample contains.
let TIMERS=new Set();
function clearTimers(){for(const t of TIMERS)clearInterval(t);TIMERS.clear();}
function mountFramePlayer(el, frames, fps){
  if(!el)return;
  if(!frames||!frames.length){el.innerHTML='<div class="kp-meta">no kept frames in this span</div>';return;}
  fps=fps||1; const period=1/fps; let i=0,timer=null;
  el.innerHTML=`<div class="kp-screen"><img alt=""><div class="kp-count"></div></div>
    <div class="kp-ctrls"><button class="b-prev" title="prev">◀</button><button class="b-play" title="play/pause">▶</button>
      <input class="b-range" type="range" min="0" max="${frames.length-1}" value="0"><button class="b-next" title="next">▶</button></div>
    <div class="kp-meta"></div>`;
  const im=el.querySelector('img'),cnt=el.querySelector('.kp-count'),rng=el.querySelector('.b-range'),
        meta=el.querySelector('.kp-meta'),play=el.querySelector('.b-play');
  function show(k){i=Math.max(0,Math.min(frames.length-1,k));const f=frames[i];im.src=img(f.img);rng.value=i;
    cnt.textContent=`${i+1}/${frames.length}`;
    const fid=f.idx!=null?`frame ${f.idx}`:`#${i}`;
    meta.innerHTML=`${fid} · <code>${esc(f.action)}</code>`;}
  function stop(){if(timer){clearInterval(timer);TIMERS.delete(timer);timer=null;}play.textContent='▶';}
  function toggle(){if(timer){stop();return;}if(i>=frames.length-1)show(0);play.textContent='⏸';
    timer=setInterval(()=>{if(i>=frames.length-1){stop();return;}show(i+1);},Math.max(80,Math.round(1000/fps)));TIMERS.add(timer);}
  el.querySelector('.b-prev').onclick=()=>{stop();show(i-1);};
  el.querySelector('.b-next').onclick=()=>{stop();show(i+1);};
  play.onclick=toggle; rng.oninput=()=>{stop();show(Number(rng.value));};
  show(0);
}
function renderClip(){
  clearTimers();
  const c=S.run.clips[S.clip];if(!c){$('main').innerHTML='';return;}
  const s=c.summary||{};
  let h=`<h2>${esc(c.clip_key)} <span class="lab">${esc(c.segment_id||'')}</span></h2>
    <div class="stat">${c.n_images_sent??s.n_images_sent??0} frames sent ·
      goals ${s.n_goals_prose??0}${c.model?` · ${esc(c.model)}`:''}</div>`;
  h+= c.video?`<video controls preload="metadata" src="${vid(c.video)}"></video>`:'<p class="lab">raw video not found</p>';

  const sm=c.sampling||{};
  h+=`<h2>Clip — sampled frames (what the VLM saw)</h2>
    <div class="stat">${sm.n_frames??0} kept frames (${sm.n_non_noop??0} active) · base ${sm.target_fps??'?'} fps ·
      NO_OP keep head/tail ${sm.noop_keep_head??'?'}/${sm.noop_keep_tail??'?'} · dropped idle ${sm.n_noop_dropped??0}</div>
    <div class="keptplayer" id="kept-stream"></div>`;

  const V=c.variants||{};
  h+=`<div><h2>Describe → Extract</h2>${variantSection(V.prose)}</div>`;

  $('main').innerHTML=h;
  const fps=Number(c.sampling?.target_fps)||0.5;
  mountFramePlayer($('kept-stream'), c.kept_frames||[], fps);
  // Per-goal trajectory players, scoped to each goal's [start_frame,end_frame].
  document.querySelectorAll('.goalfp').forEach(el=>{
    mountFramePlayer(el, framesInSpan(c.kept_frames, Number(el.dataset.start), Number(el.dataset.end)), 3);
  });
}
init().catch(e=>{$('main').innerHTML=`<pre>${esc(e.stack||e.message)}</pre>`;});
</script></body></html>"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "AnnotationRunInspector/1.0"

    def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _serve_file(self, path: Path, content_type: str) -> None:
        size = path.stat().st_size
        rng = self.headers.get("Range")
        if rng and (m := re.match(r"bytes=(\d+)-(\d*)", rng)):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
            end = min(end, size - 1)
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        else:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(path.read_bytes())

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        parsed = urllib.parse.urlparse(self.path)
        path, query = parsed.path, urllib.parse.parse_qs(parsed.query)
        try:
            if path == "/":
                body = INDEX_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/runs":
                self._json(list_runs())
            elif path == "/api/run":
                name = query.get("name", [""])[0]
                if not re.match(r"^[A-Za-z0-9_.-]+$", name):
                    self._json({"error": "bad run name"}, HTTPStatus.BAD_REQUEST); return
                self._json(build_run(name))
            elif path == "/image" and is_arrayrecord_image_uri(query.get("path", [""])[0]):
                data = grain_jpeg(query.get("path", [""])[0])
                if data is None:
                    self.send_response(HTTPStatus.NOT_FOUND); self.end_headers(); return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path in ("/image", "/video"):
                media = safe_media(query.get("path", [""])[0])
                if media is None:
                    self.send_response(HTTPStatus.NOT_FOUND); self.end_headers(); return
                ctype = mimetypes.guess_type(media.name)[0] or "application/octet-stream"
                self._serve_file(media, ctype)
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001 - report in-browser
            try:
                self._json({"error": type(exc).__name__, "message": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            except Exception:  # noqa: BLE001
                pass

    def log_message(self, *_a: Any) -> None:  # quiet
        pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    global RUN_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    RUN_ROOT = args.run_root.resolve()
    with Server((args.host, args.port), Handler) as httpd:
        url = f"http://{args.host}:{args.port}/"
        print(f"Annotation run inspector on {url}  (run-root: {RUN_ROOT})")
        print("If on a remote node: ssh -L {0}:127.0.0.1:{0} <host>  then open the URL.".format(args.port))
        if args.open:
            webbrowser.open(url)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
