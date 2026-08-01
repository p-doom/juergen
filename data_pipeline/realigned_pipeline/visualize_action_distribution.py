#!/usr/bin/env python3
"""Action-distribution viewer — aggregate stats over a dataset's action tokens.

The companion to ``visualize_frame_records.py``. That tool browses ONE trajectory
at a time (frame + action per step); this one answers the *aggregate* questions
across the whole dataset:

  * How often is a mouse click present? Which button?
  * What exact mouse movements dominate (e.g. is ``-100 10 0`` over-represented)?
  * Which keys are pressed the most? Which key *combinations* (Ctrl+C, Shift+…)?
  * What does the dx / dy / scroll magnitude distribution look like?
  * For any token you type (``LMB``, ``+KeyEnter``, ``-100 10 0``): what fraction
    of frames / segments contain it, and which segments contain it the most?

It reads the SAME datasets ``visualize_frame_records.py`` does — stage-01b
``frame_records.jsonl``, stage-04 ``conversations.jsonl``, stage-06 inline SFT
records (ArrayRecord ``train``/``val`` shards), all auto-detected — by importing
that module's loaders and its ``format_action`` parser, so a segment's stats here
match exactly what the frame viewer reconstructs when you open it. (A stage-01a
frames-master store is keylog-free, so it has no actions to aggregate; it's
reported as empty.)

The action grammar it aggregates over
(``realigned_pipeline.lib.common.format_action``): ``NO_OP``, or ``"<dx> <dy>
<scroll>"`` optionally followed by ``" ; "`` and space-separated ``+Name`` /
``-Name`` press/release tokens (rdev key names + mouse buttons ``LMB``/``RMB``/
``MMB``). Conversation/inline rows whose assistant turn prefixes the action with a
natural-language plan are handled by the shared tolerant parser.

Run::

    cd .../data_pipeline
    uv run python realigned_pipeline/visualize_action_distribution.py \
        --dataset <dir_or_file> [<dir_or_file> ...] \
        --port 8780
    # then SSH-forward the port and open http://127.0.0.1:8780/
    #   ssh -L 8780:127.0.0.1:8780 <host>

Pass several datasets and switch between them in the UI's "dataset" dropdown; each
is aggregated lazily on first selection and cached.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, OrderedDict
from http import HTTPStatus  # noqa: F401  (kept for parity / future use)
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# Make the ``realigned_pipeline`` package importable when run directly, then reuse
# the frame-records viewer's dataset loaders + action parser wholesale (single
# source of truth for the action grammar and the 4 dataset formats).
DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline import visualize_frame_records as V  # noqa: E402

# Dataset registry: display-name -> {"path": Path, "mode": str, "obj": built|None,
# "dist": aggregated|None}. Built lazily on first selection.
DATASETS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
DATASET_SAMPLE_LIMIT: "int | None" = None

# rdev modifier keys, canonicalized so "ControlLeft"/"ControlRight" fold into one
# "Ctrl" label in chords (the per-key chart keeps the raw left/right names).
_MOD_CANON = {
    "ControlLeft": "Ctrl", "ControlRight": "Ctrl",
    "ShiftLeft": "Shift", "ShiftRight": "Shift",
    "Alt": "Alt", "AltGr": "AltGr",
    "MetaLeft": "Meta", "MetaRight": "Meta",
}
_MODIFIERS = set(_MOD_CANON)
_BUTTONS = ("LMB", "RMB", "MMB")

# Histogram bin edges (non-uniform; rendered as equal-width categorical bars). The
# ±100 boundary is a bin edge so the "cursor moves 100px" quantum lands cleanly and
# can be highlighted. Outer ±1e5 edges are overflow catch-alls.
_DXDY_EDGES = [-100000, -250, -160, -120, -100, -80, -60, -40, -25, -10, -1,
               1, 10, 25, 40, 60, 80, 100, 120, 160, 250, 100000]
_MAG_EDGES = [0, 5, 10, 20, 40, 60, 80, 100, 140, 200, 300, 100000]
_DIR_LABELS = ["E →", "SE ↘", "S ↓", "SW ↙", "W ←", "NW ↖", "N ↑", "NE ↗"]


def _bin_index(value: float, edges: list[int]) -> int:
    """Index of the ``[edges[i], edges[i+1])`` bin containing ``value`` (clamped)."""
    if value <= edges[0]:
        return 0
    if value >= edges[-1]:
        return len(edges) - 2
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2


def _bin_labels(edges: list[int]) -> list[str]:
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo <= -100000:
            labels.append(f"<{hi}")
        elif hi >= 100000:
            labels.append(f"≥{lo}")
        else:
            labels.append(f"{lo}..{hi}")
    return labels


def _hist_spec(edges: list[int], counts: list[int], highlight_at: "list[int]") -> dict[str, Any]:
    """A renderable histogram: bin labels, counts, and which bins to highlight
    (the bins whose LEFT edge is one of ``highlight_at``, e.g. ±100)."""
    hi_idx = [i for i in range(len(edges) - 1) if edges[i] in highlight_at]
    return {"labels": _bin_labels(edges), "counts": counts, "highlight": hi_idx}


def _dir8(dx: float, dy: float) -> int:
    """8-way compass bucket of a (dx, dy) move (screen coords: +dy is DOWN)."""
    ang = math.degrees(math.atan2(dy, dx))  # 0=right, 90=down, -90=up, 180=left
    return int(round(ang / 45.0)) % 8


def _canonical_action(dx: int, dy: int, scroll: int,
                      events: list[tuple[str, str]], noop: bool) -> str:
    """Rebuild the clean ``format_action`` string from parsed parts — strips any
    natural-language plan prefix so identical actions collapse to one bucket."""
    if noop:
        return "NO_OP"
    head = f"{dx} {dy} {scroll}"
    if events:
        return head + " ; " + " ".join(f"{s}{n}" for s, n in events)
    return head


def _collect_segments(ds: Any) -> list[tuple[str, list[str]]]:
    """``[(segment_id, [action_str, ...]), ...]`` in dataset order.

    Frames-master stores are keylog-free (no per-frame actions), so they yield an
    empty list; every other loader exposes ``.segments`` of ``Segment`` objects
    whose ``.frames`` carry the action strings."""
    if isinstance(ds, V.FramesMasterDataset):
        return []
    out: list[tuple[str, list[str]]] = []
    for sid, seg in ds.segments.items():
        out.append((sid, [str(f.get("action") or "") for f in seg.frames]))
    return out


def build_distribution(ds: Any) -> dict[str, Any]:
    """Aggregate the full action distribution over a dataset in a single pass.

    Every count is frame-level unless named ``*_frames`` (frames CONTAINING at
    least one such event) or ``*_segments``. Chords are press-triggered: a chord is
    emitted when a non-modifier key or a mouse button is pressed while ≥1 modifier
    is held (tracked across frames within a segment, mirroring the frame viewer's
    press/release bookkeeping), so Ctrl+C spanning two turns still counts once."""
    segs = _collect_segments(ds)
    n_segments = len(segs)
    n_frames = n_noop = 0
    n_move = n_scroll = n_click = n_key = 0

    key_press: Counter = Counter()          # +Key presses, per raw rdev name
    key_frames: Counter = Counter()         # frames containing ≥1 press of a key
    key_segments: Counter = Counter()       # segments containing ≥1 press of a key
    btn_press = {b: 0 for b in _BUTTONS}
    btn_frames = {b: 0 for b in _BUTTONS}
    chords: Counter = Counter()
    triples: Counter = Counter()            # exact "dx dy scroll" over active frames
    full_actions: Counter = Counter()       # exact canonical action over active frames
    scroll_vals: Counter = Counter()        # exact nonzero scroll amounts
    directions: Counter = Counter()
    dx_counts = [0] * (len(_DXDY_EDGES) - 1)
    dy_counts = [0] * (len(_DXDY_EDGES) - 1)
    mag_counts = [0] * (len(_MAG_EDGES) - 1)
    total_scroll_mag = 0

    for sid, actions in segs:
        held: set[str] = set()
        seg_keys: set[str] = set()
        for a in actions:
            n_frames += 1
            dx_f, dy_f, scroll_f, events = V._parse_action_str(a)
            dx, dy, scroll = int(round(dx_f)), int(round(dy_f)), int(round(scroll_f))
            noop = (dx == 0 and dy == 0 and scroll == 0 and not events)
            if noop:
                n_noop += 1
            moved = dx != 0 or dy != 0
            if moved:
                n_move += 1
                dx_counts[_bin_index(dx, _DXDY_EDGES)] += 1
                dy_counts[_bin_index(dy, _DXDY_EDGES)] += 1
                mag_counts[_bin_index(math.hypot(dx, dy), _MAG_EDGES)] += 1
                directions[_dir8(dx, dy)] += 1
            if scroll != 0:
                n_scroll += 1
                total_scroll_mag += abs(scroll)
                scroll_vals[scroll] += 1
            if moved or scroll != 0 or events:
                triples[f"{dx} {dy} {scroll}"] += 1
                full_actions[_canonical_action(dx, dy, scroll, events, False)] += 1

            frame_buttons: set[str] = set()
            has_key = False
            for sign, name in events:
                if name in _BUTTONS:
                    if sign == "+":
                        btn_press[name] += 1
                        frame_buttons.add(name)
                        mods = sorted({_MOD_CANON[m] for m in held if m in _MODIFIERS})
                        if mods:
                            chords["+".join(mods + [name])] += 1
                        held.add(name)
                    else:
                        held.discard(name)
                else:
                    if sign == "+":
                        key_press[name] += 1
                        has_key = True
                        key_frames[name] += 1
                        seg_keys.add(name)
                        if name not in _MODIFIERS:
                            mods = sorted({_MOD_CANON[m] for m in held if m in _MODIFIERS})
                            if mods:
                                chords["+".join(mods + [name])] += 1
                        held.add(name)
                    else:
                        held.discard(name)
            if frame_buttons:
                n_click += 1
                for b in frame_buttons:
                    btn_frames[b] += 1
            if has_key:
                n_key += 1
        for k in seg_keys:
            key_segments[k] += 1

    n_active = n_frames - n_noop
    keys_sorted = [
        {"name": k, "presses": key_press[k], "frames": key_frames[k],
         "segments": key_segments[k]}
        for k, _ in key_press.most_common()
    ]
    return {
        "n_segments": n_segments,
        "n_frames": n_frames,
        "n_noop": n_noop,
        "n_active": n_active,
        "present": {  # frames CONTAINING each (non-exclusive)
            "move": n_move, "scroll": n_scroll, "click": n_click, "key": n_key,
            "noop": n_noop,
        },
        "totals": {
            "clicks": sum(btn_press.values()),
            "keypresses": sum(key_press.values()),
            "scroll_mag": total_scroll_mag,
        },
        "buttons": [
            {"name": b, "presses": btn_press[b], "frames": btn_frames[b]}
            for b in _BUTTONS
        ],
        "keys": keys_sorted,
        "chords": [{"combo": c, "count": n} for c, n in chords.most_common(60)],
        "triples": [{"move": t, "count": n} for t, n in triples.most_common(60)],
        "full_actions": [
            {"action": a, "count": n} for a, n in full_actions.most_common(60)
        ],
        "scroll_values": [
            {"amount": v, "count": n} for v, n in scroll_vals.most_common(40)
        ],
        "directions": [
            {"label": _DIR_LABELS[i], "count": directions.get(i, 0)} for i in range(8)
        ],
        "dx_hist": _hist_spec(_DXDY_EDGES, dx_counts, highlight_at=[-100, 100]),
        "dy_hist": _hist_spec(_DXDY_EDGES, dy_counts, highlight_at=[-100, 100]),
        "mag_hist": _hist_spec(_MAG_EDGES, mag_counts, highlight_at=[100]),
    }


def search_token(ds: Any, query: str) -> dict[str, Any]:
    """How often ``query`` (case-insensitive substring) appears across the dataset.

    Matches against the raw action strings, so it reaches tokens verbatim —
    ``LMB``, ``+KeyEnter``, ``-100 10 0`` — even when a conversation turn wraps the
    action in a natural-language plan. Returns frame/segment coverage and the
    segments where it occurs most."""
    q = query.strip().lower()
    segs = _cache_segments(ds)
    n_frames_total = sum(len(a) for _, a in segs)
    n_seg_total = len(segs)
    if not q:
        return {"query": query, "n_frames": 0, "frames_total": n_frames_total,
                "n_segments": 0, "segments_total": n_seg_total, "top_segments": []}
    n_frames = 0
    per_seg: list[tuple[str, int]] = []
    for sid, actions in segs:
        c = sum(1 for a in actions if q in a.lower())
        if c:
            n_frames += c
            per_seg.append((sid, c))
    per_seg.sort(key=lambda x: x[1], reverse=True)
    return {
        "query": query,
        "n_frames": n_frames,
        "frames_total": n_frames_total,
        "n_segments": len(per_seg),
        "segments_total": n_seg_total,
        "top_segments": [{"segment_id": s, "count": c} for s, c in per_seg[:25]],
    }


def _cache_segments(ds: Any) -> list[tuple[str, list[str]]]:
    segs = getattr(ds, "_dist_segments", None)
    if segs is None:
        segs = _collect_segments(ds)
        ds._dist_segments = segs  # type: ignore[attr-defined]
    return segs


# --------------------------------------------------------------------------- #
# Dataset registration / lazy build (delegates format detection to the sibling).
# --------------------------------------------------------------------------- #
def register_datasets(paths: list[str]) -> None:
    for raw in paths:
        p = Path(raw).expanduser()
        name = p.name or str(p)
        base, k = name, 2
        while name in DATASETS:
            name, k = f"{base}#{k}", k + 1
        DATASETS[name] = {"path": p, "mode": V.detect_mode(p), "obj": None, "dist": None}


def get_distribution(name: str) -> dict[str, Any] | None:
    """Build (or return cached) aggregate distribution for a registered dataset."""
    entry = DATASETS.get(name)
    if entry is None:
        return None
    if entry["dist"] is None:
        V.DATASET_SAMPLE_LIMIT = DATASET_SAMPLE_LIMIT
        try:
            ds = V._build_dataset(entry["path"])
        except SystemExit as exc:
            raise RuntimeError(str(exc)) from exc
        entry["obj"] = ds
        entry["dist"] = build_distribution(ds)
        entry["dist"]["mode"] = getattr(ds, "mode", entry["mode"])
        _cache_segments(ds)  # warm the search cache
    return entry["dist"]


def get_dataset_obj(name: str) -> Any | None:
    entry = DATASETS.get(name)
    if entry is None:
        return None
    if entry["obj"] is None:
        get_distribution(name)  # builds + caches the object as a side effect
    return entry["obj"]


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a: Any) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _dsname(self, q: dict[str, list[str]]) -> str:
        vals = q.get("ds")
        if vals and vals[0]:
            return vals[0]
        return next(iter(DATASETS), "")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        q = parse_qs(parsed.query)
        try:
            if route == "/":
                self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
            elif route == "/api/datasets":
                names = list(DATASETS.keys())
                self._send_json({
                    "datasets": [{"name": n, "mode": DATASETS[n]["mode"]} for n in names],
                    "default": names[0] if names else None,
                })
            elif route == "/api/dist":
                name = self._dsname(q)
                if name not in DATASETS:
                    self._send_json({"error": f"unknown dataset {name!r}"}, 404)
                    return
                try:
                    self._send_json(get_distribution(name))
                except Exception as exc:  # noqa: BLE001 — report, keep UI alive
                    self._send_json({"error": f"failed to load {name!r}: {exc}"}, 500)
            elif route == "/api/search":
                name = self._dsname(q)
                query = (q.get("q") or [""])[0]
                ds = get_dataset_obj(name) if name in DATASETS else None
                if ds is None:
                    self._send_json({"error": f"unknown dataset {name!r}"}, 404)
                else:
                    self._send_json(search_token(ds, query))
            else:
                self._send(404, b"not found", "text/plain")
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001
            self._send(500, f"{type(exc).__name__}: {exc}".encode(), "text/plain")


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>action-distribution viewer</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font:13px/1.45 ui-monospace,"SF Mono",Menlo,Consolas,monospace;
         background:#14161a; color:#d7dae0; }
  header { position:sticky; top:0; z-index:10; padding:8px 14px; border-bottom:1px solid #2a2e36;
           display:flex; gap:10px; align-items:center; flex-wrap:wrap; background:#191c21; }
  header .title { font-weight:700; color:#e8eef7; }
  select,input,button { background:#22262e; color:#d7dae0; border:1px solid #343a44;
                  border-radius:4px; padding:4px 8px; font:inherit; }
  select,button { cursor:pointer; }
  button:hover { border-color:#5b9dd9; }
  .hint { margin-left:auto; color:#6b7280; font-size:12px; }
  main { padding:14px; max-width:1500px; margin:0 auto; }
  #err { color:#f7a6a6; padding:6px 0; }
  #loading { color:#8b93a1; padding:10px 0; }

  .tiles { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; }
  .tile { background:#191c21; border:1px solid #2a2e36; border-radius:6px; padding:8px 12px; min-width:120px; }
  .tile .k { color:#8b93a1; font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
  .tile .v { color:#e8eef7; font-size:19px; font-weight:700; margin-top:2px; }
  .tile .s { color:#7fd6a2; font-size:11px; }

  /* search */
  #searchbar { display:flex; gap:8px; align-items:center; margin-bottom:8px; flex-wrap:wrap; }
  #q { min-width:280px; flex:1; }
  #searchres { background:#171b22; border:1px solid #2a2e36; border-radius:6px; padding:10px 12px;
               margin-bottom:16px; display:none; }
  #searchres.show { display:block; }
  #searchres .big { font-size:15px; color:#e8eef7; }
  #searchres .big b.hl { color:#f5b544; }
  #searchres .seglist { margin-top:8px; display:flex; flex-direction:column; gap:2px; max-height:230px; overflow:auto; }
  #searchres .segrow { display:grid; grid-template-columns:1fr 60px; gap:8px; }
  #searchres .segrow .bar { background:#20242b; border-radius:3px; position:relative; overflow:hidden; }
  #searchres .segrow .bar > i { position:absolute; inset:0 auto 0 0; background:#2d4a75; }
  #searchres .segrow .lab { position:relative; padding:1px 6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#cbd3df; }
  #searchres .segrow .cnt { text-align:right; color:#8fc4f2; }

  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(430px, 1fr)); gap:14px; }
  .card { background:#191c21; border:1px solid #2a2e36; border-radius:8px; padding:10px 12px; }
  .card h3 { margin:0 0 8px; font-size:12px; color:#aeb6c2; text-transform:uppercase; letter-spacing:.05em;
             display:flex; align-items:baseline; gap:8px; }
  .card h3 .sub { color:#6b7280; font-size:11px; text-transform:none; letter-spacing:0; font-weight:400; }

  /* horizontal bar list */
  .blist { display:flex; flex-direction:column; gap:3px; max-height:340px; overflow:auto; }
  .brow { display:grid; grid-template-columns:1fr 74px; gap:8px; align-items:center; cursor:pointer; }
  .brow:hover .lab { color:#fff; }
  .brow .bar { height:18px; background:#20242b; border-radius:3px; position:relative; overflow:hidden; }
  .brow .bar > i { position:absolute; inset:0 auto 0 0; background:#3564a0; }
  .brow.hl .bar > i { background:#c08a2a; }
  .brow.click .bar > i { background:#2d6a45; }
  .brow .lab { position:relative; padding:0 7px; line-height:18px; white-space:nowrap; overflow:hidden;
               text-overflow:ellipsis; color:#cbd3df; font-size:12px; }
  .brow .cnt { text-align:right; color:#8fc4f2; font-size:12px; }
  .brow .cnt small { color:#6b7280; }

  /* svg histogram */
  .hist { width:100%; height:150px; }
  .hist .bar { fill:#3564a0; }
  .hist .bar.hl { fill:#c08a2a; }
  .hist text { fill:#7a828e; font-size:9px; }
  .histx { display:flex; justify-content:space-between; color:#6b7280; font-size:10px; margin-top:2px; }
  .empty { color:#6b7280; font-style:italic; }
</style>
</head><body>
<header>
  <span class="title">action distribution</span>
  <select id="ds"></select>
  <span id="mode" class="hint"></span>
</header>
<main>
  <div id="err"></div>
  <div id="loading">select a dataset…</div>
  <div id="content" style="display:none">
    <div class="tiles" id="tiles"></div>
    <div id="searchbar">
      <input id="q" placeholder="filter: type a token — LMB, +KeyEnter, ControlLeft, -100 10 0 — substring match">
      <button id="qgo">count</button>
      <button id="qclear">clear</button>
      <span class="hint">click any bar to filter by it</span>
    </div>
    <div id="searchres"></div>
    <div class="grid" id="grid"></div>
  </div>
</main>
<script>
const $ = s => document.querySelector(s);
let CUR = null;      // current dataset name
let DIST = null;     // current distribution

function fmt(n){ return (n==null?0:n).toLocaleString(); }
function pct(a,b){ return b? (100*a/b).toFixed(1)+'%' : '0%'; }

async function loadDatasets(){
  const r = await fetch('/api/datasets'); const d = await r.json();
  const sel = $('#ds'); sel.innerHTML='';
  for(const ds of d.datasets){
    const o=document.createElement('option'); o.value=ds.name; o.textContent=`${ds.name}  [${ds.mode}]`;
    sel.appendChild(o);
  }
  sel.onchange = ()=> selectDataset(sel.value);
  if(d.default) selectDataset(d.default);
}

async function selectDataset(name){
  CUR = name; DIST = null;
  $('#err').textContent=''; $('#content').style.display='none';
  $('#loading').style.display=''; $('#loading').textContent=`aggregating ${name} … (first load builds the dataset)`;
  $('#mode').textContent='';
  try{
    const r = await fetch('/api/dist?ds='+encodeURIComponent(name));
    const d = await r.json();
    if(d.error){ $('#loading').style.display='none'; $('#err').textContent=d.error; return; }
    DIST = d; render(d);
  }catch(e){ $('#loading').style.display='none'; $('#err').textContent=String(e); }
}

function render(d){
  $('#loading').style.display='none'; $('#content').style.display='';
  $('#mode').textContent = `${d.mode} · ${fmt(d.n_segments)} segments · ${fmt(d.n_frames)} frames`;
  if(d.n_frames===0){ $('#grid').innerHTML='<div class="empty">no action tokens in this dataset (a frames-master store is keylog-free — run stage 01b / 04 to get actions).</div>'; $('#tiles').innerHTML=''; return; }
  renderTiles(d); renderGrid(d);
  $('#searchres').className=''; $('#searchres').innerHTML=''; $('#q').value='';
}

function renderTiles(d){
  const t = [
    ['segments', fmt(d.n_segments), ''],
    ['frames', fmt(d.n_frames), ''],
    ['active', fmt(d.n_active), pct(d.n_active,d.n_frames)+' of frames'],
    ['NO_OP', fmt(d.n_noop), pct(d.n_noop,d.n_frames)+' of frames'],
    ['move frames', fmt(d.present.move), pct(d.present.move,d.n_frames)],
    ['clicks', fmt(d.totals.clicks), pct(d.present.click,d.n_frames)+' of frames'],
    ['key presses', fmt(d.totals.keypresses), pct(d.present.key,d.n_frames)+' of frames'],
    ['scroll frames', fmt(d.present.scroll), pct(d.present.scroll,d.n_frames)],
  ];
  $('#tiles').innerHTML = t.map(([k,v,s])=>`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join('');
}

// one horizontal bar-list card. items:[{lab, cnt, sub?, cls?, token?}]
function barCard(title, sub, items, total){
  const max = items.reduce((m,it)=>Math.max(m,it.cnt),0) || 1;
  const rows = items.map(it=>{
    const w = (100*it.cnt/max).toFixed(1);
    const cls = it.cls? ' '+it.cls : '';
    const tok = it.token!=null? ` data-token="${encodeURIComponent(it.token)}"` : '';
    const sub = it.sub? ` <small>${it.sub}</small>` : '';
    return `<div class="brow${cls}"${tok}><div class="bar"><i style="width:${w}%"></i><span class="lab">${esc(it.lab)}</span></div><div class="cnt">${fmt(it.cnt)}${sub}</div></div>`;
  }).join('');
  const body = items.length? `<div class="blist">${rows}</div>` : `<div class="empty">none</div>`;
  return `<div class="card"><h3>${title}${sub?` <span class="sub">${sub}</span>`:''}</h3>${body}</div>`;
}

function histCard(title, sub, hist){
  const n = hist.counts.length, max = Math.max(1,...hist.counts);
  const W=100/n;
  const bars = hist.counts.map((c,i)=>{
    const h = 100*c/max, hl = hist.highlight.includes(i)?' hl':'';
    return `<rect class="bar${hl}" x="${(i*W).toFixed(2)}%" y="${(100-h).toFixed(2)}%" width="${(W*0.86).toFixed(2)}%" height="${h.toFixed(2)}%"><title>${esc(hist.labels[i])}: ${fmt(c)}</title></rect>`;
  }).join('');
  // sparse x labels (first, ~mid, last, and highlighted)
  const marks = new Set([0, Math.floor(n/2), n-1, ...hist.highlight]);
  const xl = hist.labels.map((l,i)=> marks.has(i)? `<span>${esc(l)}</span>`:'').join('');
  return `<div class="card"><h3>${title}${sub?` <span class="sub">${sub}</span>`:''}</h3>`+
         `<svg class="hist" preserveAspectRatio="none">${bars}</svg><div class="histx">${xl}</div></div>`;
}

function renderGrid(d){
  const cards = [];
  // exact actions & movements — the "why is X so common" panels
  cards.push(barCard('top full actions','exact canonical string',
     d.full_actions.map(x=>({lab:x.action, cnt:x.count, token:x.action})) ));
  cards.push(barCard('top mouse movements','dx dy scroll (any keys ignored)',
     d.triples.map(x=>({lab:x.move, cnt:x.count, token:x.move})) ));
  // buttons
  cards.push(barCard('mouse buttons','presses · frames-with',
     d.buttons.map(b=>({lab:b.name, cnt:b.presses, sub:pct(b.frames,d.n_frames), cls:'click', token:b.name})) ));
  // keys
  cards.push(barCard('keys pressed','presses · '+d.keys.length+' distinct',
     d.keys.map(k=>({lab:k.name, cnt:k.presses, sub:`${fmt(k.segments)} seg`, token:'+'+k.name})) ));
  // chords
  cards.push(barCard('key combinations','modifier + key/button chords',
     d.chords.map(c=>({lab:c.combo, cnt:c.count, token:null})) ));
  // directions
  cards.push(barCard('move direction','8-way, of '+fmt(d.present.move)+' move frames',
     d.directions.map(x=>({lab:x.label, cnt:x.count})) ));
  // scroll amounts
  cards.push(barCard('scroll amounts','exact nonzero values',
     d.scroll_values.map(x=>({lab:String(x.amount), cnt:x.count, token:null})) ));
  // histograms
  cards.push(histCard('dx distribution','over move frames; ±100 highlighted', d.dx_hist));
  cards.push(histCard('dy distribution','over move frames; ±100 highlighted', d.dy_hist));
  cards.push(histCard('|move| magnitude','px per frame; 100 highlighted', d.mag_hist));
  $('#grid').innerHTML = cards.join('');
  // clicking a bar with a token filters
  document.querySelectorAll('.brow[data-token]').forEach(el=>{
    el.onclick = ()=>{ const tok=decodeURIComponent(el.dataset.token); $('#q').value=tok; runSearch(tok); };
  });
}

async function runSearch(query){
  if(!query || !query.trim()){ $('#searchres').className=''; return; }
  const r = await fetch(`/api/search?ds=${encodeURIComponent(CUR)}&q=${encodeURIComponent(query)}`);
  const s = await r.json();
  if(s.error){ $('#searchres').className='show'; $('#searchres').innerHTML=`<span style="color:#f7a6a6">${s.error}</span>`; return; }
  const max = s.top_segments.reduce((m,x)=>Math.max(m,x.count),0)||1;
  const segs = s.top_segments.map(x=>{
    const w=(100*x.count/max).toFixed(1);
    return `<div class="segrow"><div class="bar"><i style="width:${w}%"></i><span class="lab">${esc(x.segment_id)}</span></div><div class="cnt">${fmt(x.count)}</div></div>`;
  }).join('');
  $('#searchres').className='show';
  $('#searchres').innerHTML =
    `<div class="big"><b class="hl">${esc(s.query)}</b> appears in `+
    `<b>${fmt(s.n_frames)}</b> / ${fmt(s.frames_total)} frames (${pct(s.n_frames,s.frames_total)}) · `+
    `<b>${fmt(s.n_segments)}</b> / ${fmt(s.segments_total)} segments (${pct(s.n_segments,s.segments_total)})</div>`+
    (s.top_segments.length? `<div class="hint" style="margin-top:6px">top segments by count</div><div class="seglist">${segs}</div>`:'');
}

function esc(s){ return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

$('#qgo').onclick = ()=> runSearch($('#q').value);
$('#q').addEventListener('keydown', e=>{ if(e.key==='Enter') runSearch($('#q').value); });
$('#qclear').onclick = ()=>{ $('#q').value=''; $('#searchres').className=''; };

loadDatasets();
</script>
</body></html>
"""


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset", required=True, nargs="+", metavar="PATH",
        help="one or more datasets (same shapes as visualize_frame_records.py: a "
             "stage-01b frame_records dir/file, a stage-04 conversations dir/file, "
             "or a stage-06 inline-records dir) — auto-detected; choose in the UI",
    )
    p.add_argument(
        "--limit", "--limit-samples", dest="limit", type=_positive_int, default=None,
        help="aggregate at most the first K samples per dataset",
    )
    p.add_argument("--port", type=int, default=8780, help="HTTP port (default 8780)")
    p.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    return p.parse_args()


def main() -> None:
    global DATASET_SAMPLE_LIMIT
    args = parse_args()
    DATASET_SAMPLE_LIMIT = args.limit
    register_datasets(args.dataset)
    if not DATASETS:
        raise SystemExit("no datasets given")
    print(f"registered {len(DATASETS)} dataset(s):", flush=True)
    for name, entry in DATASETS.items():
        print(f"  {name}  [{entry['mode']}]  {entry['path']}", flush=True)
    if DATASET_SAMPLE_LIMIT is not None:
        print(f"sample limit: first {DATASET_SAMPLE_LIMIT} samples per dataset", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"serving on http://{args.host}:{args.port}/  "
        f"(datasets aggregate on first selection; Ctrl-C to stop)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)


if __name__ == "__main__":
    main()
