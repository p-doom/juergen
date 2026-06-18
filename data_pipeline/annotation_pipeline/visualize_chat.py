cd ..
#!/usr/bin/env python3
"""Serve a video-style dashboard for inspecting a canonical SFT chat.jsonl.

Each row of the chat file is one training sample: an instruction plus an
alternating sequence of frame images (user turns) and actions (assistant
turns). This tool lets you browse/search the samples and scrub through the
frames like a video, with the per-frame action shown alongside.

Usage:
    python visualize_chat.py [CHAT_JSONL] [--port 8766] [--open]
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import socketserver
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

DEFAULT_CHAT = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom/crowd-cast/crowd-cast-2026-05-19/"
    "processed/runs/dataset_full_20260615_runlevel_migrated/"
    "stage_04_canonical_sft/chat.jsonl"
)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


class ChatIndex:
    """Byte-offset index over a large chat.jsonl so individual rows can be
    fetched on demand without holding every full record in memory.

    The index is built lazily on a background thread: the server can start
    serving (and rows become queryable) as soon as the first lines are read,
    while the rest of the file is scanned behind the scenes.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offsets: list[int] = []
        self.meta: list[dict[str, Any]] = []
        self.splits: dict[str, int] = {}
        self.clips: set[str] = set()
        self.done = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._build, name="chat-index", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _build(self) -> None:
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                split = str(row.get("split") or "—")
                clip_id = str(row.get("clip_id") or "—")
                entry = {
                    "sample_id": row.get("sample_id"),
                    "clip_id": clip_id,
                    "recording_id": row.get("recording_id"),
                    "split": split,
                    "instruction": row.get("instruction"),
                    "n_frames": row.get("n_frames"),
                    "n_non_noop": row.get("n_non_noop"),
                    "duration_s": row.get("duration_s"),
                    "_search": " ".join(
                        str(row.get(field) or "")
                        for field in ("sample_id", "clip_id", "recording_id", "instruction")
                    ).lower(),
                }
                with self._lock:
                    entry["idx"] = len(self.offsets)
                    self.offsets.append(offset)
                    self.meta.append(entry)
                    self.splits[split] = self.splits.get(split, 0) + 1
                    self.clips.add(clip_id)
        with self._lock:
            self.done = True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total": len(self.offsets),
                "splits": dict(self.splits),
                "clips": len(self.clips),
                "indexing": not self.done,
            }

    def row(self, idx: int) -> dict[str, Any] | None:
        with self._lock:
            if idx < 0 or idx >= len(self.offsets):
                return None
            offset = self.offsets[idx]
        with self.path.open("rb") as handle:
            handle.seek(offset)
            line = handle.readline()
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def query(self, q: str, split: str, offset: int, limit: int) -> dict[str, Any]:
        q = (q or "").strip().lower()
        with self._lock:
            rows = list(self.meta)
            indexing = not self.done
        if split and split != "all":
            rows = [row for row in rows if row["split"] == split]
        if q:
            rows = [row for row in rows if q in row["_search"]]
        total = len(rows)
        window = rows[offset : offset + limit]
        return {
            "total": total,
            "indexing": indexing,
            "rows": [{k: v for k, v in row.items() if k != "_search"} for row in window],
        }


def frames_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Walk the chat messages and pair each frame image with the action that
    follows it (assistant turn)."""
    frames: list[dict[str, Any]] = []
    pending_image: str | None = None
    image_paths = row.get("image_paths") or []
    fallback_iter = iter(image_paths)
    for message in row.get("messages", []):
        role = message.get("role")
        content = message.get("content", [])
        if not isinstance(content, list):
            content = [{"type": "text", "text": content}]
        if role == "user":
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image" and block.get("image"):
                    pending_image = str(block["image"])
        elif role == "assistant":
            action = ""
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    action = str(block.get("text") or "")
                    break
            image = pending_image
            if image is None:
                image = next(fallback_iter, None)
            frames.append({"index": len(frames), "image_path": image, "action": action})
            pending_image = None
    # Fall back to raw image_paths if the message structure was unexpected.
    if not frames and image_paths:
        frames = [
            {"index": i, "image_path": str(p), "action": ""} for i, p in enumerate(image_paths)
        ]
    return frames


def build_sample(index: ChatIndex, idx: int) -> dict[str, Any] | None:
    row = index.row(idx)
    if row is None:
        return None
    return {
        "idx": idx,
        "sample_id": row.get("sample_id"),
        "raw_sample_id": row.get("raw_sample_id"),
        "clip_id": row.get("clip_id"),
        "recording_id": row.get("recording_id"),
        "group_id": row.get("group_id"),
        "split": row.get("split"),
        "instruction": row.get("instruction"),
        "start_time_s": row.get("start_time_s"),
        "end_time_s": row.get("end_time_s"),
        "duration_s": row.get("duration_s"),
        "n_frames": row.get("n_frames"),
        "n_non_noop": row.get("n_non_noop"),
        "source_trajectory": row.get("source_trajectory") or {},
        "frames": frames_from_row(row),
    }


# ---------------------------------------------------------------------------
# Image serving (restricted to a root for safety)
# ---------------------------------------------------------------------------

IMAGE_ROOT: Path = Path("/")


def safe_image_path(raw_path: str) -> Path | None:
    if not raw_path:
        return None
    try:
        path = Path(raw_path).expanduser().resolve()
    except (RuntimeError, OSError):
        return None
    root = IMAGE_ROOT.resolve()
    if path == root or root in path.parents:
        return path
    return None


def infer_image_root(chat_path: Path) -> Path:
    """Derive a sensible serving root: the ``processed`` ancestor if present,
    otherwise the chat file's grandparent."""
    resolved = chat_path.resolve()
    for parent in resolved.parents:
        if parent.name == "processed":
            return parent
    return resolved.parent.parent


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SFT Chat Viewer</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #151922;
      --muted: #697386;
      --line: #d9dee8;
      --accent: #126f84;
      --bad: #b42318;
      --ok: #18794e;
      --code: #edf1f5;
      --shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(246,247,249,0.94);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--line);
    }
    .topbar {
      display: grid;
      grid-template-columns: auto 1fr auto auto;
      gap: 12px;
      align-items: center;
      max-width: 1520px;
      margin: 0 auto;
      padding: 14px 18px;
    }
    h1 { margin: 0; font-size: 18px; font-weight: 720; }
    .topbar .path { font-size: 12px; }
    select, button, input[type=search] {
      height: 34px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 6px;
      padding: 0 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    main { max-width: 1520px; margin: 0 auto; padding: 18px; }
    .layout {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }
    nav {
      position: sticky;
      top: 74px;
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow);
      border-radius: 8px;
      padding: 10px;
      max-height: calc(100vh - 96px);
      display: flex;
      flex-direction: column;
    }
    .nav-controls { display: grid; gap: 8px; margin-bottom: 8px; }
    .nav-controls .row { display: grid; grid-template-columns: 1fr auto; gap: 8px; }
    .sample-scroll { overflow-y: auto; flex: 1; margin: 0 -4px; padding: 0 4px; }
    .sample-btn {
      width: 100%;
      display: block;
      text-align: left;
      margin: 5px 0;
      border-color: transparent;
      background: transparent;
      height: auto;
      min-height: 34px;
      padding: 7px 8px;
      border-radius: 6px;
      border: 1px solid transparent;
    }
    .sample-btn:hover { background: #fbfcfd; border-color: var(--line); }
    .sample-btn.active {
      border-color: rgba(18,111,132,0.35);
      background: rgba(18,111,132,0.08);
      color: var(--accent);
      font-weight: 650;
    }
    .sample-btn .label { font-size: 11px; }
    .pager {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-top: 8px;
      border-top: 1px solid var(--line);
      padding-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .pager button { height: 28px; }
    .content { min-width: 0; }
    .summary {
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric, section {
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow);
      border-radius: 8px;
    }
    .metric { padding: 12px; min-height: 70px; }
    .label { color: var(--muted); font-size: 12px; }
    .value { font-size: 22px; font-weight: 760; margin-top: 4px; }
    section { margin-bottom: 14px; overflow: hidden; }
    .stage-head {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }
    h2 { margin: 0; font-size: 15px; }
    .path { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; overflow-wrap: anywhere; }
    .stage-body { padding: 14px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 7px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 650; background: #fbfcfd; }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      background: var(--code);
      padding: 1px 4px;
      border-radius: 4px;
    }
    .pill {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      background: #fbfcfd;
      color: var(--muted);
    }
    .pill.ok { color: var(--ok); border-color: rgba(24,121,78,0.3); background: rgba(24,121,78,0.06); }
    .pill.bad { color: var(--bad); border-color: rgba(180,35,24,0.3); background: rgba(180,35,24,0.06); }
    .instruction-box {
      margin: 0 0 12px;
      padding: 10px 12px;
      border-left: 3px solid var(--accent);
      background: rgba(18,111,132,0.06);
      font-size: 15px;
    }
    .subhead {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin: 0 0 8px;
    }
    .muted { color: var(--muted); }
    .empty {
      color: var(--muted);
      padding: 14px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #fbfcfd;
    }
    /* ---- player (matches pipeline inspector style) ---- */
    .player {
      border: 1px solid #22231f;
      border-radius: 8px;
      overflow: hidden;
      background: #151713;
      color: #e8e4dc;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .player-screen {
      position: relative;
      display: grid;
      place-items: center;
      min-height: 420px;
      background: #050608;
    }
    .player-screen img {
      display: block;
      width: 100%;
      max-height: 72vh;
      object-fit: contain;
      background: #050608;
    }
    .player-frame-count {
      position: absolute;
      top: 8px;
      right: 10px;
      padding: 2px 7px;
      color: #f2eee7;
      background: rgba(0, 0, 0, 0.42);
      border-radius: 4px;
      font-size: 12px;
    }
    .player-controls {
      display: grid;
      grid-template-columns: auto auto minmax(130px, 1fr) auto auto auto;
      gap: 10px;
      align-items: center;
      padding: 12px 14px;
      background: #151713;
      border-top: 1px solid #26281f;
    }
    .player-controls button {
      width: 32px;
      height: 32px;
      padding: 0;
      background: transparent;
      color: #e8e4dc;
      border-color: transparent;
      font-size: 18px;
      line-height: 1;
    }
    .player-controls button:hover { background: #24261f; }
    .player-controls .fps {
      color: #c5beb2;
      font-size: 12px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .player-controls .fps select {
      height: 28px;
      background: #24261f;
      color: #e8e4dc;
      border-color: #36382f;
    }
    .player-range { width: 100%; }
    .player-action {
      color: #f1ece5;
      font-size: 15px;
      padding: 10px 14px;
      overflow-wrap: anywhere;
      border-top: 1px solid #26281f;
    }
    .player-meta { padding: 0 14px 10px; color: #c5beb2; font-size: 12px; }
    .action-table-wrap { max-height: 280px; overflow: auto; border-top: 1px solid #26281f; }
    .action-table { width: 100%; border-collapse: collapse; color: #bfb8ad; font-size: 12px; }
    .action-table th, .action-table td {
      padding: 7px 14px;
      border-bottom: 1px solid #24261f;
      background: #151713;
    }
    .action-table th {
      color: #8f897f;
      font-weight: 650;
      text-align: left;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    .action-table td:first-child, .action-table th:first-child {
      width: 56px;
      text-align: right;
      color: #d1cabf;
    }
    .action-table tr.noop td { color: #6f6a61; }
    .action-table tr.active td { background: #45251d; color: #f4eee8; }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      nav { position: static; max-height: none; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .topbar { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <h1>SFT Chat Viewer</h1>
      <div class="path" id="chatPath"></div>
      <div class="muted" id="metaSummary"></div>
      <button id="refreshBtn">Refresh</button>
    </div>
  </header>
  <main>
    <div class="layout">
      <nav>
        <div class="nav-controls">
          <input type="search" id="searchBox" placeholder="Search instruction / clip / sample id…" />
          <div class="row">
            <select id="splitSelect"></select>
            <button id="searchBtn">Search</button>
          </div>
        </div>
        <div class="sample-scroll" id="sampleList"></div>
        <div class="pager">
          <button id="prevPage">‹ Prev</button>
          <span id="pageInfo">—</span>
          <button id="nextPage">Next ›</button>
        </div>
      </nav>
      <div class="content"><div id="content"></div></div>
    </div>
  </main>
  <script>
    const PAGE = 50;
    const state = {
      meta: null, offset: 0, total: 0, lastCount: 0, q: "", split: "all",
      selectedIdx: null, fps: 2, playerTimers: new Set(), activePlayer: null,
    };
    const $ = (id) => document.getElementById(id);
    const esc = (v) => String(v ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const fmt = (v) => v === null || v === undefined || v === "" ? "—" : esc(v);
    const num = (v) => Number.isFinite(Number(v)) ? Number(v).toLocaleString() : "—";
    const imgUrl = (path) => `/image?path=${encodeURIComponent(path || "")}`;

    function secs(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return "—";
      if (n >= 60) {
        const minutes = Math.floor(n / 60);
        return `${minutes}m ${(n - minutes * 60).toFixed(1)}s`;
      }
      return `${n.toFixed(1)}s`;
    }
    const isNoop = (a) => !a || String(a).trim().toUpperCase() === "NO_OP";

    async function api(path) {
      const res = await fetch(path);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    async function init() {
      state.meta = await api("/api/meta");
      $("chatPath").textContent = state.meta.path;
      renderMeta(state.meta);
      $("searchBtn").addEventListener("click", () => runSearch());
      $("searchBox").addEventListener("keydown", e => { if (e.key === "Enter") runSearch(); });
      $("splitSelect").addEventListener("change", () => runSearch());
      $("refreshBtn").addEventListener("click", () => loadList());
      $("prevPage").addEventListener("click", () => { if (state.offset > 0) { state.offset -= PAGE; loadList(); } });
      $("nextPage").addEventListener("click", () => { if (state.offset + PAGE < state.total) { state.offset += PAGE; loadList(); } });
      document.addEventListener("keydown", handlePlayerKeys);
      await loadList(true);
      pollMeta();
    }

    function renderMeta(meta) {
      const indexing = meta.indexing;
      $("metaSummary").textContent =
        `${num(meta.total)} samples · ${num(meta.clips)} clips${indexing ? " · indexing…" : ""}`;
      // Preserve the user's current split selection while refreshing counts.
      const current = $("splitSelect").value || "all";
      const splits = ["all", ...Object.keys(meta.splits || {})];
      $("splitSelect").innerHTML = splits.map(s =>
        `<option value="${esc(s)}"${s === current ? " selected" : ""}>${esc(s)}${s === "all" ? "" : ` (${num(meta.splits[s])})`}</option>`
      ).join("");
    }

    async function pollMeta() {
      while (true) {
        await new Promise(r => setTimeout(r, 1200));
        let meta;
        try { meta = await api("/api/meta"); } catch { continue; }
        state.meta = meta;
        renderMeta(meta);
        // While indexing, keep filling the first page if it isn't full yet.
        if (state.offset === 0 && !state.selectedIdx && state.lastCount < PAGE) {
          await loadList(state.lastCount === 0);
        }
        if (!meta.indexing) break;
      }
    }

    function runSearch() {
      state.q = $("searchBox").value;
      state.split = $("splitSelect").value;
      state.offset = 0;
      loadList(true);
    }

    async function loadList(selectFirst = false) {
      const url = `/api/samples?offset=${state.offset}&limit=${PAGE}` +
        `&q=${encodeURIComponent(state.q)}&split=${encodeURIComponent(state.split)}`;
      const data = await api(url);
      state.total = data.total;
      state.lastCount = data.rows.length;
      renderList(data.rows);
      $("pageInfo").textContent = data.total
        ? `${state.offset + 1}–${Math.min(state.offset + PAGE, data.total)} of ${num(data.total)}${data.indexing ? "+" : ""}`
        : (data.indexing ? "indexing…" : "0 of 0");
      if (selectFirst && data.rows.length) selectSample(data.rows[0].idx);
      else if (!data.rows.length) {
        $("content").innerHTML = `<div class="empty">${data.indexing ? "Indexing — samples loading…" : "No matching samples."}</div>`;
      }
    }

    function renderList(rows) {
      $("sampleList").innerHTML = rows.map(r => `
        <button class="sample-btn ${r.idx === state.selectedIdx ? "active" : ""}" data-idx="${r.idx}">
          <div>${esc((r.instruction || "(no instruction)").slice(0, 90))}</div>
          <div class="label">${esc(r.clip_id)} · ${num(r.n_frames)}f · ${num(r.n_non_noop)} active · ${esc(r.split)}</div>
        </button>
      `).join("");
      document.querySelectorAll(".sample-btn").forEach(btn => {
        btn.addEventListener("click", () => selectSample(Number(btn.dataset.idx)));
      });
    }

    async function selectSample(idx) {
      state.selectedIdx = idx;
      document.querySelectorAll(".sample-btn").forEach(btn =>
        btn.classList.toggle("active", Number(btn.dataset.idx) === idx));
      stopPlayerTimers();
      state.activePlayer = null;
      $("content").innerHTML = `<div class="empty">Loading sample…</div>`;
      try {
        const sample = await api(`/api/sample?idx=${idx}`);
        renderSample(sample);
      } catch (err) {
        $("content").innerHTML = `<div class="empty">${esc(err.message || err)}</div>`;
      }
    }

    function metric(label, value, detail = "") {
      return `<div class="metric"><div class="label">${esc(label)}</div><div class="value">${fmt(value)}</div><div class="label">${esc(detail)}</div></div>`;
    }

    function section(title, sub, body) {
      return `<section>
        <div class="stage-head"><div><h2>${esc(title)}</h2>${sub ? `<div class="path">${esc(sub)}</div>` : ""}</div><div></div></div>
        <div class="stage-body">${body}</div>
      </section>`;
    }

    function kvTable(obj, keys) {
      const entries = (keys || Object.keys(obj || {})).filter(k => obj && obj[k] !== undefined);
      if (!entries.length) return `<div class="empty">No data.</div>`;
      return `<table><tbody>` + entries.map(k => {
        const v = obj[k];
        const text = (v && typeof v === "object") ? JSON.stringify(v) : v;
        return `<tr><th style="width:160px">${esc(k)}</th><td><code>${esc(text)}</code></td></tr>`;
      }).join("") + `</tbody></table>`;
    }

    function renderSample(sample) {
      const st = sample.source_trajectory || {};
      const verify = st.verify_checks || {};
      $("content").innerHTML = `
        <div class="instruction-box">${esc(sample.instruction || "(no instruction)")}</div>
        <div class="summary">
          ${metric("Frames", num(sample.n_frames), `${num(sample.n_non_noop)} active`)}
          ${metric("Duration", secs(sample.duration_s), `${secs(sample.start_time_s)} → ${secs(sample.end_time_s)}`)}
          ${metric("Split", sample.split, "")}
          ${metric("Completed", fmt(st.completed), st.verified !== undefined ? `verified: ${fmt(st.verified)}` : "")}
          ${metric("Clip", sample.clip_id, "")}
        </div>
        ${section("Frame player", sample.sample_id, `<div class="player-shell"></div>`)}
        ${section("Source trajectory", "stage 02 / 03 provenance", `
          <div class="subhead">Reason</div>
          <p>${esc(st.reason || "—")}</p>
          ${st.evidence ? `<div class="subhead">Evidence</div><p>${esc(st.evidence)}</p>` : ""}
          <div class="subhead" style="margin-top:12px">Fields</div>
          ${kvTable(st, ["segment_label", "start_time_s", "end_time_s", "segment_span_s", "completed", "verified", "source_windows"])}
          ${Object.keys(verify).length ? `<div class="subhead" style="margin-top:12px">Verify checks</div>${kvTable(verify)}` : ""}
        `)}
        ${section("Identifiers", "", kvTable(sample, ["sample_id", "raw_sample_id", "clip_id", "recording_id", "group_id"]))}
      `;
      mountPlayer(document.querySelector(".player-shell"), sample);
    }

    function stopPlayerTimers() {
      for (const t of state.playerTimers) clearInterval(t);
      state.playerTimers.clear();
    }

    function handlePlayerKeys(event) {
      if (!state.activePlayer) return;
      const tag = event.target?.tagName;
      const typing = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
      if (typing && !event.target?.classList?.contains("player-range")) return;
      if (event.key === "ArrowLeft") { event.preventDefault(); state.activePlayer.step(-1); }
      else if (event.key === "ArrowRight") { event.preventDefault(); state.activePlayer.step(1); }
      else if (event.key === " ") { event.preventDefault(); state.activePlayer.toggle(); }
    }

    function mountPlayer(shell, sample) {
      const frames = sample.frames || [];
      if (!frames.length) { shell.innerHTML = `<div class="empty">No frames in this sample.</div>`; return; }
      const actionRows = frames.map((f, i) => `
        <tr data-frame-index="${i}" class="${isNoop(f.action) ? "noop" : ""}">
          <td>${num(i)}</td><td>${esc(f.action || "—")}</td>
        </tr>`).join("");
      shell.innerHTML = `
        <div class="player" tabindex="0">
          <div class="player-screen">
            <img class="player-img" alt="" />
            <div class="player-frame-count"></div>
          </div>
          <div class="player-controls">
            <button class="player-prev" title="Previous frame (Left arrow)">&#9664;</button>
            <button class="player-play" title="Play / pause (Space)">&#9654;</button>
            <input class="player-range" type="range" min="0" max="${frames.length - 1}" value="0" />
            <button class="player-next" title="Next frame (Right arrow)">&#9654;</button>
            <span class="fps">fps
              <select class="player-fps">
                <option>1</option><option selected>2</option><option>4</option><option>8</option>
              </select>
            </span>
            <button class="player-restart" title="Restart">&#8635;</button>
          </div>
          <div class="player-action"></div>
          <div class="player-meta"></div>
          <div class="action-table-wrap">
            <table class="action-table">
              <thead><tr><th>#</th><th>action&nbsp;(dx dy button)</th></tr></thead>
              <tbody>${actionRows}</tbody>
            </table>
          </div>
        </div>`;
      const player = shell.querySelector(".player");
      const img = shell.querySelector(".player-img");
      const frameCount = shell.querySelector(".player-frame-count");
      const currentAction = shell.querySelector(".player-action");
      const range = shell.querySelector(".player-range");
      const meta = shell.querySelector(".player-meta");
      const play = shell.querySelector(".player-play");
      const tableBody = shell.querySelector(".action-table tbody");
      const fpsSelect = shell.querySelector(".player-fps");
      fpsSelect.value = String(state.fps);
      let idx = 0, timer = null;
      const delay = () => Math.max(60, Math.round(1000 / (Number(fpsSelect.value) || 2)));

      function pause() {
        if (timer) { clearInterval(timer); state.playerTimers.delete(timer); timer = null; }
        play.innerHTML = "&#9654;";
      }
      function setIndex(next) {
        idx = Math.max(0, Math.min(frames.length - 1, next));
        const f = frames[idx] || {};
        img.src = imgUrl(f.image_path);
        img.alt = `frame ${idx + 1}`;
        range.value = idx;
        frameCount.textContent = `frame ${idx + 1} / ${frames.length}`;
        currentAction.innerHTML = `action: <code>${esc(f.action || "—")}</code>`;
        meta.innerHTML = `<b>${esc(sample.sample_id)}</b> · frame #${num(idx)} / ${num(frames.length - 1)}`;
        const prevActive = tableBody.querySelector("tr.active");
        if (prevActive) prevActive.classList.remove("active");
        const active = tableBody.querySelector(`tr[data-frame-index="${idx}"]`);
        if (active) { active.classList.add("active"); active.scrollIntoView({ block: "nearest" }); }
      }
      function step(delta) { pause(); setIndex(idx + delta); }
      function toggle() {
        if (timer) { pause(); return; }
        if (idx >= frames.length - 1) setIndex(0);
        play.innerHTML = "&#10074;&#10074;";
        timer = setInterval(() => {
          if (idx >= frames.length - 1) { pause(); return; }
          setIndex(idx + 1);
        }, delay());
        state.playerTimers.add(timer);
      }

      shell.querySelector(".player-prev").addEventListener("click", () => step(-1));
      shell.querySelector(".player-next").addEventListener("click", () => step(1));
      shell.querySelector(".player-restart").addEventListener("click", () => { pause(); setIndex(0); });
      range.addEventListener("input", () => { pause(); setIndex(Number(range.value)); });
      play.addEventListener("click", () => toggle());
      fpsSelect.addEventListener("change", () => {
        state.fps = Number(fpsSelect.value) || 2;
        if (timer) { pause(); toggle(); }
      });
      tableBody.addEventListener("click", (e) => {
        const row = e.target.closest("tr[data-frame-index]");
        if (!row) return;
        pause(); setIndex(Number(row.dataset.frameIndex));
      });
      state.activePlayer = { step, toggle };
      setIndex(0);
      player.focus({ preventScroll: true });
    }

    init().catch(err => {
      $("content").innerHTML = `<div class="empty">${esc(err.stack || err.message)}</div>`;
    });
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

INDEX: ChatIndex | None = None


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "SftChatViewer/0.1"

    def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_text(self, value: str, content_type: str = "text/html; charset=utf-8") -> None:
        payload = value.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        assert INDEX is not None
        try:
            if path == "/":
                self.send_text(INDEX_HTML)
            elif path == "/api/meta":
                self.send_json(
                    {
                        "path": str(INDEX.path),
                        "image_root": str(IMAGE_ROOT),
                        **INDEX.stats(),
                    }
                )
            elif path == "/api/samples":
                self.send_json(
                    INDEX.query(
                        q=query.get("q", [""])[0],
                        split=query.get("split", ["all"])[0],
                        offset=_safe_int(query.get("offset", ["0"])[0], 0),
                        limit=min(200, _safe_int(query.get("limit", ["50"])[0], 50)),
                    )
                )
            elif path == "/api/sample":
                idx = _safe_int(query.get("idx", [""])[0], -1)
                payload = build_sample(INDEX, idx)
                if payload is None:
                    self.send_json({"error": "sample not found"}, HTTPStatus.NOT_FOUND)
                    return
                self.send_json(payload)
            elif path == "/image":
                image_path = safe_image_path(query.get("path", [""])[0])
                if image_path is None or not image_path.exists():
                    self.send_response(HTTPStatus.NOT_FOUND)
                    self.end_headers()
                    return
                content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                data = image_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=60")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - report errors in-browser
            self.send_json({"error": type(exc).__name__, "message": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chat", nargs="?", default=str(DEFAULT_CHAT), help="Path to chat.jsonl")
    parser.add_argument("--image-root", default=None, help="Root directory under which frame images may be served")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()


def main() -> None:
    global INDEX, IMAGE_ROOT
    args = parse_args()
    chat_path = Path(args.chat).expanduser()
    if not chat_path.exists():
        raise SystemExit(f"chat file not found: {chat_path}")
    IMAGE_ROOT = Path(args.image_root).expanduser() if args.image_root else infer_image_root(chat_path)
    INDEX = ChatIndex(chat_path)
    INDEX.start()  # builds in the background; serving begins immediately
    print(f"Indexing {chat_path} in the background · image root={IMAGE_ROOT}")
    with ReusableThreadingTCPServer((args.host, args.port), ChatHandler) as httpd:
        httpd.daemon_threads = True
        url = f"http://{args.host}:{args.port}/"
        print(f"Serving SFT chat viewer at {url} (samples appear as they index)")
        if args.open:
            webbrowser.open(url)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
