#!/usr/bin/env python3
"""Browse a CUA-Gym teacher-rollout dataset (``trajectories.jsonl`` + WebDataset
``screenshots-*.tar`` shards) frame by frame in the browser.

This is the rollout-corpus sibling of ``visualize_frame_records.py``. It reads the
HF layout published as ``p-doom/cuagym-qwen35-rollouts`` directly — no conversion,
no stage-06 records, no ``ar://`` store:

    <root>/
      trajectories.jsonl     one JSON record per rollout
      screenshots-NNNN.tar   <task_id>/step_NNN.png
      stats.json  README.md  (optional)

One rollout record looks like::

    {"task_id": "...__r2", "instruction": "...", "app": "libreoffice_calc",
     "reward": 0.0, "reward_raw": "...", "screen": [1920, 1080], "steps_taken": 45,
     "duration_s": 395.4, "worker": "hkn0425:w12", "_shard": "screenshots-0000.tar",
     "steps": [{"step": 0, "latency_s": 5.82, "cursor_before": [1728, 972],
                "raw": "<think-ish CoT>...<tool_call>{...}</tool_call>",
                "action": "left_click",
                "raw_action_args": {"action": "left_click", "coordinate": [389, 308]},
                "meta": {"action": "left_click", "pixel": [747, 333]},
                "coordinate_screen": [747, 333],
                "shard": "screenshots-0000.tar", "member": "<task_id>/step_000.png"}]}

Teacher coordinates in ``raw_action_args`` are Qwen-native 0-1000 normalized;
``meta.pixel`` / ``coordinate_screen`` are the pixels actually clicked. The viewer
overlays the PIXEL coordinate on the screenshot and prints both.

What you get
------------
* rollout list with per-rollout reward, app, step count, wall time and outcome;
* filters: app, reward range, instruction substring, "has error step",
  "terminated", and a deep search over every step's action / CoT text;
* the step's screenshot with the action drawn on it — click crosshair, move
  arrow from ``cursor_before``, scroll direction, typed text, key chord;
* the teacher's verbatim reasoning for that step, ``<think>`` split from the
  ``<tool_call>`` it emitted;
* the rollout's ``reward_raw`` grader log, so a 0.0 is explainable.

Screenshots are read straight out of the tar shards on demand (offset index per
shard, LRU of open handles) — nothing is unpacked to disk.

Usage
-----
    python3 realigned_pipeline/visualize_cua_rollouts.py \
        --dataset /…/cuagym-qwen35-rollouts/…/p2_9b_think \
        --port 9995

Several ``--dataset`` roots can be given; switch between them in the UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tarfile
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# index cache

CACHE_DIR = Path(
    os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
) / "juergen_cua_rollouts"

class Sampling:
    """Which slice of a dataset to present: ``first N`` or a deterministic ``random N``.

    Sampling is a VIEW over the full index, not a different index — ``i`` stays the
    rollout's absolute position in trajectories.jsonl, so a link, a deep-search hit
    and a cached shard offset all stay valid when you change N. ``random`` is seeded,
    so the same (n, seed) always yields the same rollouts, and the picks are returned
    in corpus order — a spread through the whole 17k rather than a reshuffle.
    """

    __slots__ = ("mode", "n", "seed")

    def __init__(self, mode: str = "first", n: "int | None" = None, seed: int = 0) -> None:
        self.mode = mode if mode in ("first", "random") else "first"
        self.n = n if (n is None or n > 0) else None
        self.seed = seed

    def apply(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.n is None or self.n >= len(rows):
            return rows
        if self.mode == "random":
            picks = random.Random(self.seed).sample(range(len(rows)), self.n)
            picks.sort()
            return [rows[i] for i in picks]
        return rows[: self.n]

    def public(self) -> dict[str, Any]:
        return {"mode": self.mode, "n": self.n or 0, "seed": self.seed}


# Actions whose payload is a screen position; everything else is drawn as a chip.
POINT_ACTIONS = {"left_click", "right_click", "double_click", "click", "mouse_move",
                 "middle_click", "triple_click", "left_double_click"}


def _cache_path(jsonl: Path) -> Path:
    st = jsonl.stat()
    key = f"{jsonl.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    return CACHE_DIR / (hashlib.sha256(key.encode()).hexdigest()[:20] + ".json")


# ---------------------------------------------------------------------------
# tar-backed screenshot store


class ShardStore:
    """Random access to ``<root>/screenshots-*.tar`` members, by name.

    A shard is 2 GB and holds ~7.5k members. Walking its headers costs 25-40 s on
    this filesystem — it is ~3 ms of network latency per header seek, not CPU — so
    doing it per server start, let alone per request, is what makes the viewer feel
    stuck on the previous screenshot. Instead:

    * each shard's ``name -> (offset, size)`` map is built ONCE and cached on disk,
      so every later run resolves a member with no tar work at all;
    * a member is then served with a plain ``open``/``seek``/``read`` — no shared
      ``TarFile`` handle, so concurrent frame requests never queue behind one
      another (the old global lock was why one slow shard froze every image);
    * a background pool pre-indexes the remaining shards, latency-bound work that
      parallelises well, so browsing gets progressively instant instead of paying
      the cost shard by shard as you happen to click into them.
    """

    def __init__(self, root: Path, workers: int = 12) -> None:
        self.root = root
        self.workers = workers
        self._maps: dict[str, dict[str, tuple[int, int]]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()          # guards _maps/_locks only, never I/O
        self.shards = sorted(p.name for p in root.glob("screenshots-*.tar"))
        self._done = 0

    # -- on-disk index cache ------------------------------------------------

    def _cache_file(self, shard: str) -> Path:
        key = hashlib.sha256(str(self.root.resolve()).encode()).hexdigest()[:16]
        return CACHE_DIR / "shards" / key / (shard + ".json")

    def _build(self, shard: str) -> dict[str, tuple[int, int]]:
        """The shard's member map, from cache if we have it, else by scanning."""
        cf = self._cache_file(shard)
        if cf.exists():
            try:
                return {k: (v[0], v[1]) for k, v in json.loads(cf.read_text()).items()}
            except Exception:  # noqa: BLE001 — a bad cache just means a rescan
                pass
        path = self.root / shard
        if not path.exists():
            raise FileNotFoundError(f"no such shard: {path}")
        with tarfile.open(path) as tf:
            out = {m.name: (m.offset_data, m.size) for m in tf if m.isfile()}
        try:
            cf.parent.mkdir(parents=True, exist_ok=True)
            tmp = cf.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(out))
            tmp.replace(cf)                     # atomic: a torn cache is worse than none
        except Exception:  # noqa: BLE001 — caching is best-effort
            pass
        return out

    def _map(self, shard: str) -> dict[str, tuple[int, int]]:
        got = self._maps.get(shard)
        if got is not None:
            return got
        with self._guard:
            lock = self._locks.setdefault(shard, threading.Lock())
        with lock:                              # per shard: others keep serving
            got = self._maps.get(shard)
            if got is None:
                got = self._build(shard)
                with self._guard:
                    self._maps[shard] = got
                    self._done = len(self._maps)
        return got

    # -- background warmer --------------------------------------------------

    def warm(self) -> None:
        """Pre-index every shard in the background. Latency-bound, so a wide pool
        helps; daemon threads so it never holds up shutdown."""
        todo = list(self.shards)

        def run() -> None:
            while True:
                with self._guard:
                    if not todo:
                        return
                    shard = todo.pop(0)
                try:
                    self._map(shard)
                except Exception:  # noqa: BLE001 — a bad shard must not kill the pool
                    pass

        for _ in range(min(self.workers, max(1, len(todo)))):
            threading.Thread(target=run, daemon=True).start()

    def progress(self) -> dict[str, int]:
        return {"indexed": self._done, "total": len(self.shards)}

    # -- read ---------------------------------------------------------------

    def read(self, shard: str, member: str) -> bytes:
        members = self._map(shard)
        got = members.get(member)
        if got is None:
            # Some shards were written with a "./" prefix; try the variants.
            for alt in (member.lstrip("./"), "./" + member):
                got = members.get(alt)
                if got is not None:
                    break
        if got is None:
            raise KeyError(f"{member} not in {shard}")
        offset, size = got
        with (self.root / shard).open("rb") as fh:   # own handle: no cross-request lock
            fh.seek(offset)
            return fh.read(size)


# ---------------------------------------------------------------------------
# dataset


class RolloutDataset:
    """A rollout root: the jsonl indexed by byte offset, shards read lazily."""

    def __init__(self, root: Path, name: str, sampling: "Sampling | None" = None) -> None:
        self.root = root
        self.name = name
        self.jsonl = root / "trajectories.jsonl"
        if not self.jsonl.exists():
            raise FileNotFoundError(f"{root} has no trajectories.jsonl")
        self.shards = ShardStore(root)
        self.stats = self._read_json(root / "stats.json")
        self.readme = (root / "README.md").read_text(errors="replace") \
            if (root / "README.md").exists() else ""
        self.rows: list[dict[str, Any]] = []
        self.partial = False          # True if --limit cut the index short
        self.sampling = sampling or Sampling()
        # Only "first N" can be answered without reading the whole file; "random N"
        # has to know what it is choosing from, so it still needs the full index
        # (~15 s once, then cached — and a warm cache makes both instant anyway).
        self._build_index(self.sampling.n if self.sampling.mode == "first" else None)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001 — stats.json is decoration, never fatal
            return {}

    # -- index -------------------------------------------------------------

    def _build_index(self, limit: int | None) -> None:
        # A cached full index is always preferable to a truncated read: it costs
        # nothing to load and it keeps the WHOLE corpus available, so raising N or
        # switching to random in the UI does not need a restart. --limit only buys
        # a short read when there is no cache yet, i.e. on the very first run.
        cache = _cache_path(self.jsonl)
        if cache.exists():
            try:
                self.rows = json.loads(cache.read_text())
                self.partial = False
                print(f"[{self.name}] index from cache: {len(self.rows)} rollouts")
                return
            except Exception:  # noqa: BLE001 — a bad cache just means a rebuild
                pass
        print(f"[{self.name}] indexing {self.jsonl} …", flush=True)
        rows: list[dict[str, Any]] = []
        off = 0
        with self.jsonl.open("rb") as fh:
            for i, raw in enumerate(fh):
                if limit is not None and i >= limit:
                    break
                start = off
                off += len(raw)
                try:
                    d = json.loads(raw)
                except Exception:  # noqa: BLE001 — skip a torn line, keep going
                    continue
                steps = d.get("steps") or []
                acts = [s.get("action") for s in steps]
                rows.append({
                    "i": len(rows),
                    "off": start,
                    "len": len(raw),
                    "task_id": d.get("task_id", ""),
                    "base_id": re.sub(r"__r\d+$", "", d.get("task_id", "")),
                    "instruction": d.get("instruction", ""),
                    "app": d.get("app") or "?",
                    "reward": d.get("reward"),
                    "steps": len(steps),
                    "duration_s": d.get("duration_s"),
                    "terminated": bool(d.get("terminated")),
                    "complete": bool(d.get("complete")),
                    "setup_ok": bool(d.get("setup_ok", True)),
                    "worker": d.get("worker", ""),
                    "shard": d.get("_shard", ""),
                    "n_err": sum(1 for s in steps if s.get("error")),
                    "n_noact": sum(1 for a in acts if not a),
                    "final": next((s.get("meta", {}).get("status")
                                   for s in reversed(steps)
                                   if (s.get("meta") or {}).get("status")), ""),
                })
                if len(rows) % 2000 == 0:
                    print(f"[{self.name}]   {len(rows)} …", flush=True)
        self.rows = rows
        self.partial = limit is not None
        if self.partial:
            print(f"[{self.name}] indexed the FIRST {len(rows)} rollouts only "
                  f"(--limit); re-run without it for the whole corpus", flush=True)
        else:
            print(f"[{self.name}] indexed {len(rows)} rollouts", flush=True)
        if limit is None:
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(rows))
            except Exception:  # noqa: BLE001 — caching is best-effort
                pass

    # -- access ------------------------------------------------------------

    def record(self, i: int) -> dict[str, Any]:
        row = self.rows[i]
        with self.jsonl.open("rb") as fh:
            fh.seek(row["off"])
            return json.loads(fh.read(row["len"]))

    def info(self, sampling: "Sampling | None" = None) -> dict[str, Any]:
        sampling = sampling or self.sampling
        rows = sampling.apply(self.rows)
        apps: dict[str, int] = {}
        rewarded = [r["reward"] for r in rows if isinstance(r["reward"], (int, float))]
        for r in rows:
            apps[r["app"]] = apps.get(r["app"], 0) + 1
        # Reported the way the dataset's own README/stats.json report them: every
        # rate is over the NON-NULL rollouts (a null reward is a task whose
        # reward.py produced nothing parseable, ~5%), never over all 17k.
        return {
            "name": self.name,
            "root": str(self.root),
            "n": len(rows),
            "n_total": len(self.rows),
            "partial": self.partial,
            "sampling": sampling.public(),
            "apps": sorted(apps.items(), key=lambda kv: -kv[1]),
            "stats": self.stats,
            "n_scored": len(rewarded),
            "n_null": len(rows) - len(rewarded),
            "mean_reward": (sum(rewarded) / len(rewarded)) if rewarded else None,
            "golden": sum(1 for v in rewarded if v > 0),
            "perfect": sum(1 for v in rewarded if v >= 1.0),
            "rows": rows,
        }

    def detail(self, i: int) -> dict[str, Any]:
        d = self.record(i)
        row = self.rows[i]
        steps = []
        for s in d.get("steps") or []:
            meta = s.get("meta") or {}
            args = s.get("raw_action_args") or {}
            steps.append({
                "step": s.get("step"),
                "action": s.get("action"),
                "latency_s": s.get("latency_s"),
                "cursor_before": s.get("cursor_before"),
                "pixel": s.get("coordinate_screen") or meta.get("pixel"),
                "norm": args.get("coordinate"),
                "meta": meta,
                "args": args,
                "raw": s.get("raw") or s.get("assistant_raw") or "",
                "error": s.get("error"),
                "sub": s.get("sub"),
                "shard": s.get("shard") or row["shard"],
                "member": s.get("member"),
                "label": _label(s),
            })
        return {
            "i": i,
            "task_id": d.get("task_id"),
            "instruction": d.get("instruction", ""),
            "app": d.get("app"),
            "reward": d.get("reward"),
            "reward_raw": d.get("reward_raw") or "",
            "screen": d.get("screen") or [1920, 1080],
            "worker": d.get("worker", ""),
            "duration_s": d.get("duration_s"),
            "steps_taken": d.get("steps_taken"),
            "terminated": d.get("terminated"),
            "complete": d.get("complete"),
            "setup_ok": d.get("setup_ok"),
            "steps": steps,
        }


def _label(s: dict[str, Any]) -> str:
    """One verbatim line for a step's action — what the teacher actually asked for."""
    a = s.get("action")
    if not a:
        err = s.get("error")
        return f"<no action: {err}>" if err else "<no action>"
    args = dict(s.get("raw_action_args") or {})
    args.pop("action", None)
    px = s.get("coordinate_screen") or (s.get("meta") or {}).get("pixel")
    if a in POINT_ACTIONS and px:
        norm = args.get("coordinate")
        return f"{a}({px[0]}, {px[1]}) px" + (f"  [norm {norm[0]},{norm[1]}]" if norm else "")
    if a == "type":
        return f'type({json.dumps(args.get("text", ""))})'
    if a == "key":
        keys = args.get("keys") or args.get("key") or []
        return f"key({'+'.join(keys) if isinstance(keys, list) else keys})"
    if a == "scroll":
        return f"scroll({args.get('pixels', args.get('scroll_amount', '?'))})"
    if a == "wait":
        return f"wait({args.get('time', '?')}s)"
    if a == "terminate":
        return f"terminate({args.get('status', '?')})"
    return f"{a}({json.dumps(args)})" if args else f"{a}()"


THINK_RE = re.compile(r"(?:<think>)?(.*?)</think>", re.S)
TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def split_raw(raw: str) -> dict[str, str]:
    """Split a teacher turn into its reasoning and its emitted tool call."""
    think = ""
    m = THINK_RE.search(raw)
    if m:
        think = m.group(1).strip()
    tool = ""
    t = TOOL_RE.search(raw)
    if t:
        tool = t.group(1).strip()
    rest = raw
    if m:
        rest = rest[m.end():]
    if t:
        rest = rest.replace(t.group(0), "")
    return {"think": think or (raw if not m and not t else ""),
            "tool": tool, "rest": rest.strip()}


# ---------------------------------------------------------------------------
# search


def deep_search(ds: RolloutDataset, needle: str, subset: list[int],
                cap: int = 400) -> dict[str, Any]:
    """Substring search over every step's action label and CoT, within ``subset``.

    Streams the records rather than holding them; stops at ``cap`` hit rollouts so
    a two-letter needle cannot walk the whole 2 GB corpus.
    """
    if not needle:
        return {"hits": [], "scanned": 0, "capped": False}
    low = needle.lower()
    hits: list[dict[str, Any]] = []
    scanned = 0
    for i in subset:
        scanned += 1
        d = ds.record(i)
        marks = []
        for s in d.get("steps") or []:
            hay = (_label(s) + "\n" + (s.get("raw") or "")).lower()
            if low in hay:
                marks.append(s.get("step"))
        if marks:
            hits.append({"i": i, "steps": marks})
            if len(hits) >= cap:
                return {"hits": hits, "scanned": scanned, "capped": True}
    return {"hits": hits, "scanned": scanned, "capped": False}


# ---------------------------------------------------------------------------
# server

DATASETS: "OrderedDict[str, RolloutDataset]" = OrderedDict()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a: Any) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _ds(self, q: dict[str, list[str]]) -> RolloutDataset | None:
        name = (q.get("ds") or [""])[0] or next(iter(DATASETS), "")
        return DATASETS.get(name)

    @staticmethod
    def _sampling(q: dict[str, list[str]], ds: RolloutDataset) -> Sampling:
        """The sampling the client asked for — ``sm`` (first|random), ``n``
        (0/blank = no cap, which overrides --limit for this request), ``seed``.
        Anything missing or unparseable falls back to the CLI defaults rather
        than failing the request."""
        def _int(key: str, default: "int | None") -> "int | None":
            raw = (q.get(key) or [""])[0].strip()
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        base = ds.sampling
        mode = (q.get("sm") or [base.mode])[0]
        n = _int("n", base.n)
        if n is not None and n <= 0:
            n = None
        return Sampling(mode, n, _int("seed", base.seed) or 0)

    def do_GET(self) -> None:  # noqa: N802
        p = urlparse(self.path)
        q = parse_qs(p.query)
        try:
            if p.path == "/":
                self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
            elif p.path == "/api/datasets":
                first = next(iter(DATASETS.values()), None)
                self._json({"datasets": list(DATASETS.keys()),
                            "default": next(iter(DATASETS), None),
                            "sampling": first.sampling.public() if first else
                                        Sampling().public()})
            elif p.path == "/api/index":
                ds = self._ds(q)
                if ds is None:
                    self._json({"error": "unknown dataset"}, 404)
                else:
                    self._json(ds.info(self._sampling(q, ds)))
            elif p.path == "/api/rollout":
                ds = self._ds(q)
                i = int((q.get("i") or ["-1"])[0])
                if ds is None or not (0 <= i < len(ds.rows)):
                    self._json({"error": "unknown rollout"}, 404)
                else:
                    self._json(ds.detail(i))
            elif p.path == "/api/shards":
                ds = self._ds(q)
                self._json(ds.shards.progress() if ds else {"indexed": 0, "total": 0})
            elif p.path == "/frame":
                self._frame(q)
            else:
                self._send(404, b"not found", "text/plain")
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001 — surface, keep the server up
            self._send(500, f"{type(exc).__name__}: {exc}".encode(), "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        """Deep search. POST, not GET: the subset is up to 17k rollout indices,
        which overruns the 64 KB request-line cap as a query string."""
        p = urlparse(self.path)
        try:
            if p.path != "/api/search":
                self._send(404, b"not found", "text/plain")
                return
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            ds = DATASETS.get(body.get("ds") or next(iter(DATASETS), ""))
            if ds is None:
                self._json({"error": "unknown dataset"}, 404)
                return
            subset = body.get("subset")
            if not subset:
                subset = list(range(len(ds.rows)))
            self._json(deep_search(ds, body.get("q") or "", subset))
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001 — surface, keep the server up
            self._send(500, f"{type(exc).__name__}: {exc}".encode(), "text/plain")

    def _frame(self, q: dict[str, list[str]]) -> None:
        ds = self._ds(q)
        shard = (q.get("shard") or [""])[0]
        member = (q.get("member") or [""])[0]
        if ds is None or not shard or not member:
            self._send(404, b"no frame", "text/plain")
            return
        try:
            self._send(200, ds.shards.read(shard, member), "image/png", cache=True)
        except (KeyError, FileNotFoundError) as exc:
            self._send(404, str(exc).encode(), "text/plain")


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cua-gym rollout viewer</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font:13px/1.45 ui-monospace,"SF Mono",Menlo,Consolas,monospace;
         background:#14161a; color:#d7dae0; height:100vh; display:flex; flex-direction:column; }
  header { padding:7px 12px; border-bottom:1px solid #2a2e36; display:flex; gap:10px;
           align-items:center; flex-wrap:wrap; background:#191c21; flex:none; }
  select,button,input { background:#22262e; color:#d7dae0; border:1px solid #343a44;
                        border-radius:4px; padding:3px 8px; font:inherit; }
  button { cursor:pointer; }
  button:hover { border-color:#5b9dd9; }
  button.on { background:#2d4a75; border-color:#5b9dd9; }
  input:focus,select:focus { outline:none; border-color:#5b9dd9; }
  .num { width:66px; }
  .hint { margin-left:auto; color:#6b7280; font-size:12px; }
  kbd { background:#22262e; border:1px solid #343a44; border-radius:3px; padding:0 4px; }
  main { flex:1; display:flex; min-height:0; }

  #list { width:340px; flex:none; overflow-y:auto; border-right:1px solid #2a2e36;
          background:#171a1f; }
  .row { padding:5px 8px; border-bottom:1px solid #22262e; cursor:pointer; }
  .row:hover { background:#1e222a; }
  .row.cur { background:#233149; }
  .row .l1 { display:flex; gap:6px; align-items:center; }
  .row .ins { color:#8b93a1; font-size:11px; white-space:nowrap; overflow:hidden;
              text-overflow:ellipsis; }
  .row .tid { color:#5b6270; font-size:10px; }
  .rw { display:inline-block; min-width:38px; text-align:center; padding:0 5px;
        border-radius:3px; font-size:11px; background:#26292f; color:#8b93a1; }
  .rw.zero { background:#3a2226; color:#e88b93; }
  .rw.part { background:#3d3418; color:#e8c877; }
  .rw.gold { background:#1e3a2a; color:#7fd6a2; }
  .rw.null { background:#2b2440; color:#c3b3f5; }
  .app { color:#7aa7d4; font-size:11px; }
  .meta { margin-left:auto; color:#5b6270; font-size:10px; }
  .row.hit { box-shadow:inset 3px 0 0 #d9b95b; }

  #resizer { width:6px; flex:none; cursor:col-resize; background:#20242b;
             border-left:1px solid #2a2e36; border-right:1px solid #2a2e36; }
  #resizer:hover,#resizer.drag { background:#5b9dd9; }

  #screen { flex:1; min-width:0; padding:10px; display:flex; flex-direction:column;
            gap:6px; overflow:hidden; }
  #stage { flex:1 1 auto; min-height:0; position:relative; background:#000;
           border-radius:4px; overflow:hidden; }
  #frameimg { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }
  /* A cold shard costs one ~30 s index; dim the stale frame rather than leaving it
     looking current, which is exactly how "the screenshot never changes" reads. */
  #stage.loading #frameimg { opacity:.25; filter:grayscale(1); }
  #stage.loading::after { content:"indexing shard…"; position:absolute; left:50%; top:50%;
    transform:translate(-50%,-50%); background:rgba(20,22,26,.9); border:1px solid #5b9dd9;
    border-radius:6px; padding:6px 12px; color:#8fc4f2; font-size:12px; }
  #stage.err::after { content:"frame not found"; border-color:#e88b93; color:#e88b93; }
  #ov { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
  #status { display:flex; gap:12px; color:#aeb6c2; flex-wrap:wrap; align-items:center;
            flex:none; }
  #status b { color:#fff; }
  #label { color:#d7dae0; font-size:12px; white-space:pre-wrap; word-break:break-word; }
  .badge { display:inline-block; padding:0 6px; border-radius:3px; font-size:11px; }
  .badge.act { background:#1e3a2a; color:#7fd6a2; }
  .badge.noop { background:#26292f; color:#8b93a1; }
  .badge.err { background:#3a2226; color:#e88b93; }
  .badge.term { background:#2b2440; color:#c3b3f5; }
  #strip { display:flex; gap:2px; overflow-x:auto; padding:4px 0 1px; flex:none; }
  .cell { flex:none; width:11px; height:24px; border-radius:2px; background:#2a2e36;
          cursor:pointer; }
  .cell.act { background:#3f7d5b; }
  .cell.type { background:#3a6ea5; }
  .cell.key { background:#5a5a8c; }
  .cell.scroll { background:#8a6d2f; }
  .cell.wait { background:#3a3f49; }
  .cell.err { background:#7d3a3f; }
  .cell.term { background:#5b4a8c; }
  .cell.match { box-shadow:inset 0 0 0 2px #d9b95b; }
  .cell.cur { outline:2px solid #5b9dd9; outline-offset:1px; }

  #side { width:430px; flex:none; border-left:1px solid #2a2e36; background:#171a1f;
          display:flex; flex-direction:column; min-height:0; }
  .sec { border-bottom:1px solid #2a2e36; padding:7px 10px; }
  .sec h4 { margin:0 0 4px; font-size:11px; color:#6b7280; font-weight:normal;
            text-transform:uppercase; letter-spacing:.06em; }
  #instr { color:#d7dae0; font-size:12px; max-height:110px; overflow-y:auto; }
  #rmeta { display:flex; flex-wrap:wrap; gap:4px 10px; color:#8b93a1; font-size:11px; }
  #rmeta b { color:#d7dae0; }
  #think { flex:1 1 auto; overflow-y:auto; padding:7px 10px; white-space:pre-wrap;
           word-break:break-word; color:#b6bdc9; font-size:12px; }
  #tool { color:#7fd6a2; font-size:11px; white-space:pre-wrap; word-break:break-word;
          max-height:130px; overflow-y:auto; }
  #grader { color:#8b93a1; font-size:11px; white-space:pre-wrap; max-height:150px;
            overflow-y:auto; display:none; }
  #grader.on { display:block; }
</style></head><body>

<header>
  <select id="ds"></select>
  <select id="sm"><option value="first">first</option><option value="random">random</option></select>
  <input id="n" class="num" placeholder="N" title="how many rollouts to load (blank/0 = all)">
  <input id="seed" class="num" placeholder="seed" title="seed for random N — same n+seed, same rollouts">
  <select id="app"><option value="">app: all</option></select>
  <input id="rmin" class="num" placeholder="rew ≥">
  <input id="rmax" class="num" placeholder="rew ≤">
  <input id="q" placeholder="instruction contains…" style="width:180px">
  <input id="dq" placeholder="deep: action / CoT contains…" style="width:200px">
  <button id="godeep">search</button>
  <button id="errs">errors only</button>
  <span id="count" style="color:#8b93a1"></span>
  <span id="shardprog" style="color:#6b7280;font-size:11px"></span>
  <span class="hint"><kbd>j</kbd>/<kbd>k</kbd> step · <kbd>n</kbd>/<kbd>p</kbd> rollout ·
    <kbd>g</kbd> grader log · <kbd>/</kbd> filter</span>
</header>

<main>
  <div id="list"></div>
  <div id="resizer"></div>
  <div id="screen">
    <div id="stage"><img id="frameimg" alt=""><svg id="ov" preserveAspectRatio="xMidYMid meet"></svg></div>
    <div id="status"></div>
    <div id="label"></div>
    <div id="strip"></div>
  </div>
  <div id="side">
    <div class="sec"><h4>instruction</h4><div id="instr"></div></div>
    <div class="sec"><h4>rollout</h4><div id="rmeta"></div>
      <pre id="grader" style="margin:6px 0 0"></pre></div>
    <div class="sec"><h4>tool call</h4><div id="tool"></div></div>
    <div class="sec" style="border:0;padding-bottom:2px"><h4>teacher reasoning</h4></div>
    <div id="think"></div>
  </div>
</main>

<script>
const $ = s => document.querySelector(s);
let DS = "", INFO = null, ROWS = [], VIEW = [], CUR = -1, ROLL = null, STEP = 0;
let HITS = new Map();          // rollout index -> matching step numbers

const fmt = (v, d=2) => (v===null||v===undefined) ? "–" : (+v).toFixed(d);
const esc = s => (s||"").replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function rwClass(r){
  if (r === null || r === undefined) return "null";
  if (r >= 0.99) return "gold";
  if (r > 0) return "part";
  return "zero";
}

async function boot(){
  const d = await (await fetch("/api/datasets")).json();
  $("#ds").innerHTML = d.datasets.map(n=>`<option>${esc(n)}</option>`).join("");
  DS = d.default || "";
  $("#ds").value = DS;
  // Seed the controls from the CLI flags, so the UI shows what is actually loaded.
  const sp = d.sampling || {mode:"first", n:0, seed:0};
  $("#sm").value = sp.mode;
  $("#n").value = sp.n || "";
  $("#seed").value = sp.seed || 0;
  $("#ds").onchange = () => { DS = $("#ds").value; loadIndex(); };
  await loadIndex();
}

function sampleQS(){
  return `sm=${encodeURIComponent($("#sm").value)}` +
         `&n=${encodeURIComponent($("#n").value.trim() || 0)}` +
         `&seed=${encodeURIComponent($("#seed").value.trim() || 0)}`;
}

async function loadIndex(){
  $("#count").textContent = "loading…";
  INFO = await (await fetch(`/api/index?ds=${encodeURIComponent(DS)}&${sampleQS()}`)).json();
  ROWS = INFO.rows || [];
  $("#app").innerHTML = `<option value="">app: all (${ROWS.length})</option>` +
    (INFO.apps||[]).map(([a,n])=>`<option value="${esc(a)}">${esc(a)} (${n})</option>`).join("");
  HITS = new Map();
  applyFilter();
}

function applyFilter(){
  const app = $("#app").value, q = $("#q").value.trim().toLowerCase();
  const rmin = parseFloat($("#rmin").value), rmax = parseFloat($("#rmax").value);
  const errOnly = $("#errs").classList.contains("on");
  VIEW = ROWS.filter(r => {
    if (app && r.app !== app) return false;
    if (q && !(r.instruction||"").toLowerCase().includes(q)) return false;
    if (!isNaN(rmin) && !(r.reward !== null && r.reward >= rmin)) return false;
    if (!isNaN(rmax) && !(r.reward !== null && r.reward <= rmax)) return false;
    if (errOnly && !r.n_err && !r.n_noact) return false;
    if (HITS.size && !HITS.has(r.i)) return false;
    return true;
  });
  const mr = INFO && INFO.mean_reward;
  const pct = (k) => `${k} (${fmt(100*k/INFO.n_scored,1)}%)`;
  const sp = (INFO && INFO.sampling) || {mode:"first", n:0};
  const of = (INFO && INFO.n_total && INFO.n_total !== ROWS.length)
    ? ` (${sp.mode} ${ROWS.length} of ${INFO.n_total})` : "";
  const part = (INFO && INFO.partial)
    ? " · index truncated by --limit: restart without it to reach the rest" : "";
  $("#count").textContent = `${VIEW.length} / ${ROWS.length} rollouts${of}` +
    (mr!=null ? ` · scored ${INFO.n_scored}, null ${INFO.n_null}` +
                ` · mean reward ${fmt(mr,4)} · golden(>0) ${pct(INFO.golden)}` +
                ` · perfect(1.0) ${pct(INFO.perfect)}` : "") + part;
  renderList();
  if (VIEW.length) select(VIEW[0].i); else { ROLL = null; render(); }
}

function renderList(){
  const html = VIEW.slice(0, 3000).map(r => `
    <div class="row ${r.i===CUR?'cur':''} ${HITS.has(r.i)?'hit':''}" data-i="${r.i}">
      <div class="l1">
        <span class="rw ${rwClass(r.reward)}">${r.reward===null?'null':fmt(r.reward,2)}</span>
        <span class="app">${esc(r.app)}</span>
        <span class="meta">${r.steps} steps · ${fmt(r.duration_s,0)}s${r.n_err?` · ${r.n_err} err`:''}</span>
      </div>
      <div class="ins">${esc(r.instruction)}</div>
      <div class="tid">${esc(r.task_id)}</div>
    </div>`).join("");
  $("#list").innerHTML = html +
    (VIEW.length > 3000 ? `<div class="row" style="color:#6b7280">… ${VIEW.length-3000} more (narrow the filter)</div>` : "");
  $("#list").querySelectorAll(".row[data-i]").forEach(el =>
    el.onclick = () => select(+el.dataset.i));
}

async function select(i){
  CUR = i;
  renderList();
  ROLL = await (await fetch(`/api/rollout?ds=${encodeURIComponent(DS)}&i=${i}`)).json();
  const marks = HITS.get(i);
  STEP = (marks && marks.length) ? Math.max(0, ROLL.steps.findIndex(s=>s.step===marks[0])) : 0;
  render();
  const el = $(`.row[data-i="${i}"]`);
  if (el) el.scrollIntoView({block:"nearest"});
}

function render(){
  if (!ROLL){ $("#frameimg").removeAttribute("src"); $("#ov").innerHTML=""; return; }
  const s = ROLL.steps[STEP];
  const [W,H] = ROLL.screen;
  $("#instr").textContent = ROLL.instruction;
  $("#rmeta").innerHTML = [
    `<span>reward <b class="rw ${rwClass(ROLL.reward)}">${ROLL.reward===null?'null':fmt(ROLL.reward,3)}</b></span>`,
    `<span>app <b>${esc(ROLL.app)}</b></span>`,
    `<span>steps <b>${ROLL.steps.length}</b></span>`,
    `<span>wall <b>${fmt(ROLL.duration_s,0)}s</b></span>`,
    `<span>terminated <b>${ROLL.terminated}</b></span>`,
    `<span>setup_ok <b>${ROLL.setup_ok}</b></span>`,
    `<span>worker <b>${esc(ROLL.worker)}</b></span>`,
    `<span style="color:#5b6270">${esc(ROLL.task_id)}</span>`,
  ].join("");
  $("#grader").textContent = ROLL.reward_raw || "(no grader output)";

  if (!s){ return; }
  const src = `/frame?ds=${encodeURIComponent(DS)}&shard=${encodeURIComponent(s.shard)}&member=${encodeURIComponent(s.member)}`;
  const img = $("#frameimg"), stage = $("#stage");
  if (img.getAttribute("src") !== src){
    stage.classList.remove("err");
    stage.classList.add("loading");
    img.onload = () => stage.classList.remove("loading");
    img.onerror = () => { stage.classList.remove("loading"); stage.classList.add("err"); };
    img.src = src;
  }
  const kind = !s.action ? "noop" : s.action === "terminate" ? "term" : s.error ? "err" : "act";
  $("#status").innerHTML =
    `<span>step <b>${STEP+1}/${ROLL.steps.length}</b></span>` +
    `<span class="badge ${kind}">${esc(s.action||"none")}</span>` +
    `<span>latency <b>${fmt(s.latency_s)}s</b></span>` +
    `<span>cursor_before <b>${s.cursor_before?s.cursor_before.join(", "):"–"}</b></span>` +
    (s.error?`<span class="badge err">${esc(s.error)}</span>`:"");
  $("#label").textContent = s.label;
  drawOverlay(s, W, H);
  renderStrip();

  const raw = s.raw || "";
  const tm = raw.match(/([\s\S]*?)<\/think>/);
  const cm = raw.match(/<tool_call>([\s\S]*?)<\/tool_call>/);
  $("#think").textContent = tm ? tm[1].replace(/^<think>/,"").trim() : raw.trim();
  $("#tool").textContent = cm ? cm[1].trim() : "(no <tool_call> in this turn)";
  $("#think").scrollTop = 0;
}

function drawOverlay(s, W, H){
  const ov = $("#ov");
  ov.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const parts = [];
  const cb = s.cursor_before;
  if (cb) parts.push(`<circle cx="${cb[0]}" cy="${cb[1]}" r="9" fill="none"
      stroke="#5b9dd9" stroke-width="3"/><circle cx="${cb[0]}" cy="${cb[1]}" r="2.5" fill="#5b9dd9"/>`);
  const p = s.pixel;
  if (p){
    if (cb && (cb[0]!==p[0] || cb[1]!==p[1]))
      parts.push(`<line x1="${cb[0]}" y1="${cb[1]}" x2="${p[0]}" y2="${p[1]}"
        stroke="#5b9dd9" stroke-width="2" stroke-dasharray="8 6" opacity=".7"/>`);
    const col = s.action === "mouse_move" ? "#e8c877" : "#ff5f6d";
    parts.push(`
      <line x1="${p[0]-34}" y1="${p[1]}" x2="${p[0]+34}" y2="${p[1]}" stroke="${col}" stroke-width="3"/>
      <line x1="${p[0]}" y1="${p[1]-34}" x2="${p[0]}" y2="${p[1]+34}" stroke="${col}" stroke-width="3"/>
      <circle cx="${p[0]}" cy="${p[1]}" r="17" fill="none" stroke="${col}" stroke-width="3"/>
      <circle cx="${p[0]}" cy="${p[1]}" r="34" fill="none" stroke="${col}" stroke-width="1.5" opacity=".55"/>`);
  }
  if (s.action === "scroll"){
    const px = (s.args && (s.args.pixels ?? s.args.scroll_amount)) || 0;
    const cx = p ? p[0] : W/2, cy = p ? p[1] : H/2;
    const dir = px < 0 ? 1 : -1;          // negative pixels scroll the page down
    parts.push(`<line x1="${cx}" y1="${cy-120*dir}" x2="${cx}" y2="${cy+120*dir}"
        stroke="#e8c877" stroke-width="5"/>
      <polygon points="${cx-22},${cy+120*dir-26*dir} ${cx+22},${cy+120*dir-26*dir} ${cx},${cy+130*dir}"
        fill="#e8c877"/>`);
  }
  if (s.action === "type" || s.action === "key"){
    const txt = s.action === "type" ? (s.args.text||"")
                                    : ((s.args.keys||[]).join(" + "));
    parts.push(`<rect x="24" y="${H-96}" width="${Math.min(W-48, 26+txt.length*17)}" height="58"
        rx="8" fill="rgba(20,22,26,.86)" stroke="#5b9dd9" stroke-width="2"/>
      <text x="42" y="${H-56}" fill="#d7dae0" font-family="monospace" font-size="30">${
        esc(txt).slice(0,80)}</text>`);
  }
  ov.innerHTML = parts.join("");
}

function renderStrip(){
  const marks = new Set(HITS.get(CUR) || []);
  $("#strip").innerHTML = ROLL.steps.map((s,ix) => {
    const a = s.action;
    const cls = !a ? (s.error ? "err" : "") :
      ["left_click","right_click","double_click","click","mouse_move"].includes(a) ? "act" :
      a === "type" ? "type" : a === "key" ? "key" : a === "scroll" ? "scroll" :
      a === "wait" ? "wait" : a === "terminate" ? "term" : "act";
    return `<div class="cell ${cls} ${ix===STEP?'cur':''} ${marks.has(s.step)?'match':''}"
      data-ix="${ix}" title="${esc(String(s.step))}: ${esc(s.label)}"></div>`;
  }).join("");
  $("#strip").querySelectorAll(".cell").forEach(el =>
    el.onclick = () => { STEP = +el.dataset.ix; render(); });
  const cur = $("#strip .cell.cur");
  if (cur) cur.scrollIntoView({block:"nearest", inline:"nearest"});
}

async function runDeep(){
  const needle = $("#dq").value.trim();
  if (!needle){ HITS = new Map(); applyFilter(); return; }
  // Search only what the cheap filters already left standing.
  HITS = new Map();
  applyFilter();
  const subset = VIEW.map(r=>r.i);
  $("#count").textContent = `deep search over ${subset.length} rollouts…`;
  const r = await (await fetch("/api/search", {
    method: "POST",
    body: JSON.stringify({ds: DS, q: needle, subset}),
  })).json();
  HITS = new Map((r.hits||[]).map(h=>[h.i, h.steps]));
  applyFilter();
  if (r.capped) $("#count").textContent += " (capped at 400 hits)";
}

// --- wiring ---------------------------------------------------------------
["#app","#rmin","#rmax"].forEach(s => $(s).onchange = applyFilter);
// Sampling is server-side (it changes WHICH rollouts are loaded), so it reloads
// the index rather than filtering what is already here.
["#sm","#n","#seed"].forEach(s => $(s).onchange = () => { HITS = new Map(); loadIndex(); });
$("#q").oninput = () => { clearTimeout(window._t); window._t = setTimeout(applyFilter, 200); };
$("#errs").onclick = () => { $("#errs").classList.toggle("on"); applyFilter(); };
$("#godeep").onclick = runDeep;
$("#dq").onkeydown = e => { if (e.key === "Enter") runDeep(); };

document.onkeydown = e => {
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)){
    if (e.key === "Escape") e.target.blur();
    return;
  }
  if (e.key === "/"){ e.preventDefault(); $("#q").focus(); return; }
  if (!ROLL) return;
  const at = VIEW.findIndex(r => r.i === CUR);
  if (e.key === "j" || e.key === "ArrowRight"){ STEP = Math.min(ROLL.steps.length-1, STEP+1); render(); }
  else if (e.key === "k" || e.key === "ArrowLeft"){ STEP = Math.max(0, STEP-1); render(); }
  else if (e.key === "n"){ if (at >= 0 && at+1 < VIEW.length) select(VIEW[at+1].i); }
  else if (e.key === "p"){ if (at > 0) select(VIEW[at-1].i); }
  else if (e.key === "g"){ $("#grader").classList.toggle("on"); }
  else return;
  e.preventDefault();
};

// draggable list/screen split
(() => {
  const rz = $("#resizer"); let on = false;
  rz.onmousedown = e => { on = true; rz.classList.add("drag"); e.preventDefault(); };
  document.onmousemove = e => { if (on) $("#list").style.width = Math.max(220, e.clientX) + "px"; };
  document.onmouseup = () => { on = false; rz.classList.remove("drag"); };
})();

// Shard indexing runs in the background; show how far along it is, since a frame
// on a not-yet-indexed shard is the one thing here that can take ~30 s.
async function pollShards(){
  try {
    const p = await (await fetch(`/api/shards?ds=${encodeURIComponent(DS)}`)).json();
    $("#shardprog").textContent = (p.total && p.indexed < p.total)
      ? `· shards indexed ${p.indexed}/${p.total}` : "";
    if (!p.total || p.indexed < p.total) setTimeout(pollShards, 3000);
  } catch (e) { /* the server going away is not worth a console full of errors */ }
}

boot().then(pollShards);
</script></body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser(
        description="Browse CUA-Gym rollout datasets (trajectories.jsonl + screenshot tars).")
    p.add_argument("--dataset", nargs="+", required=True,
                   help="one or more rollout roots (a dir holding trajectories.jsonl "
                        "and screenshots-*.tar); switch between them in the UI")
    p.add_argument("--limit", type=int, default=0,
                   help="present only N rollouts per dataset (0 = all). With "
                        "--sample-mode first this also stops reading the jsonl "
                        "after N, so a cold start is instant; change it live in the UI")
    p.add_argument("--sample-mode", choices=("first", "random"), default="first",
                   help="take the first N rollouts, or a deterministic random N "
                        "spread across the whole corpus (default: first)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for --sample-mode random (default 0) — the same "
                        "n+seed always yields the same rollouts")
    p.add_argument("--no-warm", action="store_true",
                   help="do not pre-index the screenshot shards in the background; "
                        "each shard is then indexed (~30 s, once, cached) the first "
                        "time you open a rollout that lives on it")
    p.add_argument("--port", type=int, default=9995, help="HTTP port (default 9995)")
    p.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    args = p.parse_args()

    sampling = Sampling(args.sample_mode, args.limit or None, args.seed)
    for raw in args.dataset:
        root = Path(raw).expanduser().resolve()
        name = root.name if root.name not in DATASETS else str(root)
        try:
            DATASETS[name] = RolloutDataset(root, name, sampling)
        except Exception as exc:  # noqa: BLE001 — report and keep the others
            print(f"[skip] {root}: {type(exc).__name__}: {exc}")
    if not DATASETS:
        raise SystemExit("no usable dataset")

    if not args.no_warm:
        for ds in DATASETS.values():
            ds.shards.warm()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {len(DATASETS)} dataset(s) on http://{args.host}:{args.port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
