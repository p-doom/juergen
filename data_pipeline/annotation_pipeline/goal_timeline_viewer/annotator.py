#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["av>=12", "msgpack>=1"]
# ///
"""Manual goal-annotation tool for crowd-cast working days — one script, one frontend.

The whole tool is this file plus `annotator.html` (the viewer, served from the same
directory). Four subcommands cover the full flow:

  index    order one participant's segmented recordings into days; freeze one -> day.json
  align    (re)build the keylog<->video realignment map -> alignment.json   (serve auto-runs it)
  prewarm  parallel-remux a day to faststart mp4s -> <DATA>/cache_faststart/ (serve self-warms too)
  serve    run the local viewer: HEVC <video> playback + synchronized keylog + goal authoring

Typical flow:
  uv run annotator.py index --user <UID>                                 # survey the days
  uv run annotator.py index --user <UID> --date <YYYY-MM-DD> --out <DATA>/day.json
  uv run annotator.py prewarm --data_dir <DATA>                          # optional, on a compute node
  uv run annotator.py serve   --data_dir <DATA> --port 8753              # auto-builds alignment + warms
  # from your laptop:  ssh -L 8753:localhost:8753 <node>   ->   http://localhost:8753

The realignment (`align`) corrects the recorder's idle-pause keylog drift; see
REALIGNMENT.md. Source uploads are READ-ONLY — everything written goes under <DATA>.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import datetime as dt
import json
import os
import re
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import av
import msgpack

DEFAULT_UPLOADS = "/fast/project/HFMI_SynergyUnit/p-doom/crowd-cast/crowd-cast-2026-06-18/uploads"
SEG_RE = re.compile(r"recording_(?P<rid>[0-9a-f-]+)_seg(?P<seg>\d+)\.mp4$")
INPUT = {"KeyPress", "KeyRelease", "MousePress", "MouseRelease", "MouseScroll", "MouseMove"}
METHOD = "continuous_overhang_v2"
MOUSEMOVE_STRIDE = 4          # thin the dominant MouseMove events for payload size
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")

STATE: dict = {}
LOCK = threading.Lock()
REMUX_LOCK = threading.Lock()
VERBOSE = False


# ======================================================================
# index — uploads -> ordered day.json
# ======================================================================
def find_user_segments(uploads: Path, user: str, version: str | None = None) -> list[Path]:
    # version=None globs every recorder-version dir; pass e.g. "1.0.5" to restrict to one
    # (a participant can appear under several versions; this isolates a single recorder).
    return sorted(uploads.glob(f"{version or '*'}/{user}/recordings/recording_*_seg*.mp4"))


def keylog_for(video: Path) -> Path | None:
    m = SEG_RE.search(video.name)
    if not m:
        return None
    kl = video.parent.parent / "keylogs" / f"input_{m['rid']}_seg{m['seg']}.msgpack"
    return kl if kl.exists() else None


def probe(video: Path) -> dict | None:
    """Read absolute start (OBS creation_time) + duration from mp4 metadata only."""
    m = SEG_RE.search(video.name)
    if not m:
        return None
    try:
        with av.open(str(video)) as c:
            ct = c.metadata.get("creation_time") or c.streams.video[0].metadata.get("creation_time")
            s = c.streams.video[0]
            dur = float(s.duration * s.time_base) if s.duration else 0.0
            if dur == 0.0 and c.duration:
                dur = c.duration / 1_000_000.0
            w, h = s.codec_context.width, s.codec_context.height
    except Exception as e:  # noqa: BLE001
        return {"video": str(video), "error": str(e)}
    if not ct:
        return {"video": str(video), "error": "no creation_time"}
    start = dt.datetime.fromisoformat(ct.replace("Z", "+00:00"))
    kl = keylog_for(video)
    return {
        "recording_id": m["rid"], "seg": int(m["seg"]),
        "video": str(video), "keylog": str(kl) if kl else None,
        "start_utc": start.isoformat(), "_start_ts": start.timestamp(),
        "duration_s": round(dur, 3), "width": w, "height": h,
        "version": video.parts[video.parts.index("uploads") + 1],
    }


def event_count(keylog: str | None) -> int:
    if not keylog or not os.path.exists(keylog):
        return 0
    try:
        with open(keylog, "rb") as fh:
            return len(msgpack.unpack(fh, raw=False))
    except Exception:  # noqa: BLE001
        return 0


def cmd_index(args) -> None:
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(args.tz)
    except Exception:  # noqa: BLE001
        tz = dt.timezone.utc

    uploads = Path(args.uploads)
    vids = find_user_segments(uploads, args.user, args.version)
    vtag = f" (version {args.version})" if args.version else ""
    print(f"[index] {len(vids)} mp4 for user {args.user}{vtag}; reading metadata ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        probed = [p for p in ex.map(probe, vids) if p]

    good = [p for p in probed if "error" not in p]
    errs = [p for p in probed if "error" in p]
    # dedupe re-exports across versions: keep one per (recording_id, seg),
    # preferring the row that has a keylog, then the latest version.
    by_key: dict[tuple, dict] = {}
    for p in good:
        k = (p["recording_id"], p["seg"])
        cur = by_key.get(k)
        if cur is None or (p["keylog"] and not cur["keylog"]) or \
           (bool(p["keylog"]) == bool(cur["keylog"]) and p["version"] > cur["version"]):
            by_key[k] = p
    uniq = sorted(by_key.values(), key=lambda r: r["_start_ts"])
    print(f"[index] {len(good)} readable, {len(uniq)} unique segments, {len(errs)} errors", flush=True)

    days = collections.defaultdict(list)
    for r in uniq:
        d = dt.datetime.fromtimestamp(r["_start_ts"], tz).date().isoformat()
        days[d].append(r)

    print(f"\n[index] days for user {args.user} (tz={args.tz}):")
    print(f"  {'date':12} {'segs':>5} {'span':>13} {'video_h':>8} {'gaps>5m':>8}")
    for d in sorted(days):
        rows = days[d]
        vid_h = sum(r["duration_s"] for r in rows) / 3600
        big_gaps = sum(
            1 for a, b in zip(rows, rows[1:])
            if b["_start_ts"] - (a["_start_ts"] + a["duration_s"]) > 300
        )
        first = dt.datetime.fromtimestamp(rows[0]["_start_ts"], tz).strftime("%H:%M")
        last = dt.datetime.fromtimestamp(rows[-1]["_start_ts"] + rows[-1]["duration_s"], tz).strftime("%H:%M")
        print(f"  {d:12} {len(rows):>5} {first}-{last:>5} {vid_h:>7.2f}h {big_gaps:>8}")

    if not args.date:
        print("\n[index] re-run with --date YYYY-MM-DD --out day.json to freeze a day.")
        return

    rows = days.get(args.date)
    if not rows:
        raise SystemExit(f"no segments on {args.date}")
    day_start = rows[0]["_start_ts"]
    out_rows = []
    prev_end = None
    for r in rows:
        gap = 0.0 if prev_end is None else max(0.0, r["_start_ts"] - prev_end)
        out_rows.append({
            "recording_id": r["recording_id"], "seg": r["seg"],
            "video": r["video"], "keylog": r["keylog"],
            "start_utc": r["start_utc"],
            "t_day": round(r["_start_ts"] - day_start, 3),
            "duration_s": r["duration_s"],
            "gap_before_s": round(gap, 3),
            "width": r["width"], "height": r["height"],
            "n_events": event_count(r["keylog"]),
        })
        prev_end = r["_start_ts"] + r["duration_s"]

    out = args.out or f"day_{args.user[:8]}_{args.date}.json"
    payload = {
        "user": args.user, "date": args.date, "tz": args.tz,
        "day_start_utc": dt.datetime.fromtimestamp(day_start, dt.timezone.utc).isoformat(),
        "n_segments": len(out_rows),
        "video_seconds": round(sum(r["duration_s"] for r in out_rows), 1),
        "segments": out_rows,
    }
    Path(out).write_text(json.dumps(payload, indent=2))
    print(f"\n[index] wrote {out}: {len(out_rows)} segments, {payload['video_seconds']/3600:.2f}h video.")


# ======================================================================
# align — keylog<->video realignment map (see REALIGNMENT.md)
# ======================================================================
def input_times(keylog_path: str):
    data = msgpack.unpack(open(keylog_path, "rb"), raw=False)
    span = data[-1][0] / 1e6 if data else 0.0
    ev = sorted(e[0] / 1e6 for e in data if e[1][0] in INPUT)
    return ev, span


def build_day(data_dir, idle: float = 120.0, tol: float = 2.0, write: bool = True) -> dict:
    """Per-segment keylog->video realignment. THE BUG: keylog timestamps come from
    OBS's global frame clock, which advances while the recording OUTPUT is paused on
    idle (idle_timeout, default 120s); OBS pause collapses the paused span out of the
    mp4, so after each idle pause the keylog runs AHEAD of the video by (gap-120).
    Idle is recording-continuous (carries across segments), so we stitch each
    recording's segments into one continuous obs-clock timeline (via t_day) and
    recover an exact piecewise time-map. See REALIGNMENT.md for the full spec."""
    data_dir = Path(data_dir)
    day = json.loads((data_dir / "day.json").read_text())
    segs = day["segments"]

    by_rec: dict[str, list[int]] = {}
    for i, s in enumerate(segs):
        if s.get("keylog") and s["duration_s"] >= 1:
            by_rec.setdefault(s["recording_id"], []).append(i)

    out_segs = {}
    counts = collections.Counter()
    for rec, idxs in by_rec.items():
        idxs.sort(key=lambda i: segs[i]["seg"])
        localev = {}
        span = {}
        cont = []                                  # continuous obs-time of every input event
        for i in idxs:
            ev, sp = input_times(segs[i]["keylog"])
            localev[i] = ev
            span[i] = sp
            cont.extend(segs[i]["t_day"] + r for r in ev)
        cont.sort()
        rec_start = segs[idxs[0]]["t_day"]         # recorder's idle timer origin

        for i in idxs:
            s = segs[i]
            t0 = s["t_day"]
            ev = localev[i]
            vid = s["duration_s"]
            overhang = round(max(0.0, span[i] - vid), 3)

            # fresh (within-segment) idle pauses -- exact, microsecond
            fresh = [(round(a + idle, 3), round(b - a - idle, 3))
                     for a, b in zip(ev, ev[1:]) if b - a > idle]      # (kp_local, collapse)
            sum_fresh = round(sum(c for _, c in fresh), 3)

            # leading / overhanging pause: gap from the previous continuous input
            # (in a prior segment, or the recording start) to this segment's first input
            has_leading = False
            lead_collapse_cont = 0.0
            resume_local = None
            if ev:
                first_cont = t0 + ev[0]
                k = bisect.bisect_left(cont, first_cont)
                prev = cont[k - 1] if k > 0 else rec_start
                gap_lead = first_cont - prev
                if gap_lead > idle:
                    has_leading = True
                    lead_collapse_cont = gap_lead - idle
                    resume_local = ev[0]

            total_cont = round(sum_fresh + (lead_collapse_cont if has_leading else 0.0), 3)
            no_trail = abs(total_cont - overhang) <= tol     # overhang == total collapse?

            # build splices: leading first (refined), then fresh in order
            splices = []
            cum = 0.0
            leading_method = "n/a"
            if has_leading:
                if no_trail:
                    coll = round(overhang - sum_fresh, 3)    # PRECISE (overhang is creation-time-free)
                    leading_method = "overhang"
                else:
                    coll = round(lead_collapse_cont, 3)      # trailing-idle confound: keep continuous
                    leading_method = "creation"
                coll = max(0.0, coll)
                kp = round(resume_local - coll, 3)
                splices.append({"kp": kp, "vp": kp, "collapse": coll})
                cum += coll
            for kp_local, coll in sorted(fresh):
                splices.append({"kp": kp_local, "vp": round(kp_local - cum, 3), "collapse": coll})
                cum += coll

            total = round(cum, 3)
            residual = round(total - overhang, 3)
            corr_end = round(span[i] - total, 3)
            under = corr_end > vid + tol
            if not splices and abs(overhang) <= tol:
                status = "aligned"
            elif under:
                status = "UNDER"
            elif leading_method == "creation":
                status = "needs_review"      # leading idle not verifiable via overhang -- inspect
            elif abs(residual) <= tol:
                status = "exact"
            else:
                status = "benign_idle"
            counts[status] += 1
            out_segs[str(i)] = {
                "n_pauses": len(splices), "total_collapse_s": total, "overhang_s": overhang,
                "residual_s": residual, "corr_end_s": corr_end, "video_s": round(vid, 3),
                "status": status, "closed": status in ("aligned", "exact", "benign_idle"),
                "leading_method": leading_method, "splices": splices,
            }

    payload = {
        "idle_timeout_s": idle, "closure_tol_s": tol, "method": METHOD,
        "built_for_day": str(data_dir / "day.json"),
        "ok": True, "counts": dict(counts), "segments": out_segs,
    }
    if write:
        (data_dir / "alignment.json").write_text(json.dumps(payload, indent=2))
    return payload


def cmd_align(args) -> None:
    p = build_day(Path(args.data_dir), args.idle, args.tol, write=True)
    c = p["counts"]
    nover = sum(1 for v in p["segments"].values() if v["leading_method"] == "overhang")
    ncre = sum(1 for v in p["segments"].values() if v["leading_method"] == "creation")
    print(f"[align] wrote {Path(args.data_dir) / 'alignment.json'}  (method={METHOD})")
    print(f"[align] {len(p['segments'])} segments: {c.get('aligned',0)} already-aligned, "
          f"{c.get('exact',0)} exact, {c.get('benign_idle',0)} benign-idle, "
          f"{c.get('needs_review',0)} needs-review, {c.get('UNDER',0)} UNDER")
    print(f"[align] leading idle: {nover} overhang-refined (frame-exact), {ncre} creation-based")


# ======================================================================
# prewarm — parallel faststart remux of a whole day
# ======================================================================
def remux(job: dict) -> dict:
    """Stream-copy a segment to a faststart mp4 (moov-first). Picklable for the pool."""
    src, dst = job["src"], Path(job["dst"])
    if dst.exists() and dst.stat().st_size > 0:
        return {"i": job["i"], "status": "cached", "size": dst.stat().st_size}
    tmp = dst.with_suffix(".tmp.mp4")
    try:
        inp = av.open(src)
        out = av.open(str(tmp), "w", format="mp4", options={"movflags": "faststart"})
        smap = {}
        for s in inp.streams:
            if s.type not in ("video", "audio"):
                continue
            os_ = out.add_stream_from_template(s)
            if s.type == "video":
                os_.codec_tag = "hvc1"   # keep hvc1; mp4 muxer otherwise retags to hev1 (Firefox can't decode)
            smap[s.index] = os_
        for pkt in inp.demux(*[s for s in inp.streams if s.index in smap]):
            if pkt.dts is None:
                continue
            pkt.stream = smap[pkt.stream.index]
            out.mux(pkt)
        out.close()
        inp.close()
        tmp.replace(dst)
        return {"i": job["i"], "status": "ok", "size": dst.stat().st_size}
    except Exception as e:  # noqa: BLE001
        if tmp.exists():
            tmp.unlink()
        return {"i": job["i"], "status": "error", "error": str(e)[:120]}


def cmd_prewarm(args) -> None:
    data = Path(args.data_dir)
    day = json.loads((data / "day.json").read_text())
    cache = data / "cache_faststart"
    cache.mkdir(exist_ok=True)
    jobs = [{"i": i, "src": s["video"], "dst": str(cache / f"seg_{i:04d}.mp4")}
            for i, s in enumerate(day["segments"])]
    print(f"[prewarm] {len(jobs)} segments -> {cache}  ({args.workers} workers)", flush=True)
    ok = cached = err = 0
    total_bytes = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(remux, jobs):
            if r["status"] == "ok":
                ok += 1; total_bytes += r["size"]
            elif r["status"] == "cached":
                cached += 1; total_bytes += r["size"]
            else:
                err += 1
                print(f"[prewarm] seg {r['i']} ERROR: {r.get('error')}", flush=True)
            done = ok + cached + err
            if done % 10 == 0 or done == len(jobs):
                print(f"[prewarm] {done}/{len(jobs)}  ok={ok} cached={cached} err={err}", flush=True)
    print(f"[prewarm] done: ok={ok} cached={cached} err={err}, cache {total_bytes/1e9:.1f} GB", flush=True)
    if err:
        sys.exit(1)


# ======================================================================
# serve — local annotation viewer
# ======================================================================
def faststart_path(seg_idx: int) -> str:
    """Lazily stream-copy one segment to a faststart mp4 (moov-first) and cache it.
    Serialized via REMUX_LOCK so live clicks queue behind one remux. Lossless."""
    seg = STATE["day"]["segments"][seg_idx]
    cache_dir = STATE["data_dir"] / "cache_faststart"
    cache_dir.mkdir(exist_ok=True)
    dst = cache_dir / f"seg_{seg_idx:04d}.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        return str(dst)
    with REMUX_LOCK:
        if dst.exists() and dst.stat().st_size > 0:
            return str(dst)
        tmp = dst.with_suffix(".tmp.mp4")
        inp = av.open(seg["video"])
        out = av.open(str(tmp), "w", format="mp4", options={"movflags": "faststart"})
        smap = {}
        for s in inp.streams:
            if s.type not in ("video", "audio"):
                continue
            os_ = out.add_stream_from_template(s)  # lossless packet copy
            if s.type == "video":
                os_.codec_tag = "hvc1"             # keep hvc1; mp4 muxer otherwise retags to hev1
            smap[s.index] = os_
        for pkt in inp.demux(*[s for s in inp.streams if s.index in smap]):
            if pkt.dts is None:
                continue
            pkt.stream = smap[pkt.stream.index]
            out.mux(pkt)
        out.close()
        inp.close()
        tmp.replace(dst)
    if VERBOSE:
        print(f"[faststart] seg {seg_idx} -> {dst} ({dst.stat().st_size} bytes)", flush=True)
    return str(dst)


def load_day(data_dir: Path) -> dict:
    day = json.loads((data_dir / "day.json").read_text())
    goals_path = data_dir / "goals.jsonl"
    goals = []
    if goals_path.exists():
        for line in goals_path.read_text().splitlines():
            if line.strip():
                goals.append(json.loads(line))
    next_id = 1 + max([g["id"] for g in goals], default=0)
    # read-only overlay of auto-extracted goals (fold->restructure->assign->stitch),
    # already resolved to t_day by build_overlay.py. Optional; {} if absent.
    overlay_path = data_dir / "overlay_goals.json"
    overlay = json.loads(overlay_path.read_text()) if overlay_path.exists() else {}
    align = ensure_alignment(data_dir, day)
    return {"data_dir": data_dir, "day": day, "goals_path": goals_path, "overlay": overlay,
            "goals": goals, "next_id": next_id, "keylog_cache": {}, "align": align}


def ensure_alignment(data_dir: Path, day: dict) -> dict:
    """Load alignment.json; (re)build it (via build_day, above) if missing or stale.
    The viewer gates on the "ok" flag. Writes only to the derived data dir."""
    align_path = data_dir / "alignment.json"
    expected = {str(i) for i, s in enumerate(day["segments"])
                if s.get("keylog") and s["duration_s"] >= 1}
    align = None
    if align_path.exists():
        try:
            align = json.loads(align_path.read_text())
        except Exception:  # noqa: BLE001
            align = None
    stale = (align is None
             or align.get("method") != METHOD                          # upgrade older maps
             or align.get("built_for_day") != str(data_dir / "day.json")
             or set(align.get("segments", {}).keys()) != expected)
    if not stale:
        align["ok"] = True
        return align
    try:
        print(f"[serve] alignment.json {'missing' if align is None else 'stale'} — building ...", flush=True)
        align = build_day(data_dir, write=True)
        align["autobuilt"] = True
        return align
    except Exception as e:  # noqa: BLE001
        print(f"[serve] FAILED to build alignment.json: {e}", flush=True)
        return {"segments": {}, "ok": False, "error": str(e)}


def keylog_to_video(kt: float, splices: list) -> float:
    """Map a keylog timestamp to recorded-video time by removing the collapsed idle
    spans. splices sorted by kp; an event inside a collapsed span clamps to vp."""
    cum = 0.0
    for s in splices:
        kp, coll = s["kp"], s["collapse"]
        if kt <= kp:
            break
        if kt < kp + coll:
            return round(s["vp"], 3)
        cum += coll
    return round(kt - cum, 3)


def decode_keylog(seg_idx: int, corrected: bool) -> dict:
    key = (seg_idx, corrected)
    cache = STATE["keylog_cache"]
    if key in cache:
        return cache[key]
    seg = STATE["day"]["segments"][seg_idx]
    info = STATE["align"].get("segments", {}).get(str(seg_idx), {})
    splices = info.get("splices", []) if corrected else []
    out = {"seg_idx": seg_idx, "t_day0": seg["t_day"], "duration_s": seg["duration_s"],
           "corrected": corrected and bool(info.get("splices")),
           "alignment": {
               "n_pauses": info.get("n_pauses", 0),
               "closed": info.get("closed", True),
               "residual_s": info.get("residual_s", 0.0),
               "splices_video": [s["vp"] for s in info.get("splices", [])],
           },
           "events": []}
    kl = seg.get("keylog")
    if kl and os.path.exists(kl):
        data = msgpack.unpack(open(kl, "rb"), raw=False)
        mm = 0
        ax = ay = 0.0
        for ts, (name, args) in ((e[0], e[1]) for e in data):
            if name == "MouseMove":
                ax += args[0]
                ay += args[1]
                mm += 1
                if mm % MOUSEMOVE_STRIDE:
                    continue
                args = [round(ax, 2), round(ay, 2)]
                ax = ay = 0.0
            t = ts / 1e6
            if splices:
                t = keylog_to_video(t, splices)
            out["events"].append([round(t, 3), name, args])
    cache[key] = out
    return out


def persist_goals() -> None:
    gp: Path = STATE["goals_path"]
    tmp = gp.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(g) + "\n"
                           for g in sorted(STATE["goals"], key=lambda g: g["t_start"])))
    tmp.replace(gp)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400" if ctype.startswith("video") else "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _serve_video(self, seg_idx: int, faststart: bool):
        path = faststart_path(seg_idx) if faststart else STATE["day"]["segments"][seg_idx]["video"]
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:
            m = RANGE_RE.search(rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                # open-ended ranges (bytes=N-) are served to EOF: capping them
                # truncates GOPs mid-stream and the HEVC decoder errors out.
        end = min(end, size - 1)
        length = end - start + 1
        if VERBOSE:
            print(f"[video] seg={seg_idx} fs={int(faststart)} range={rng or '-'} "
                  f"-> {start}-{end}/{size} ({length}B)", flush=True)
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", 'video/mp4; codecs="hvc1"')
        self.send_header("Accept-Ranges", "bytes")
        # faststart segments are immutable per seg_idx -> let the browser cache them
        # (so the prefetch pane's range fetches are reused, and re-visits are instant)
        self.send_header("Cache-Control", "private, max-age=86400")
        self.send_header("Content-Length", str(length))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                return self._send(200, (Path(__file__).parent / "annotator.html").read_bytes(),
                                  "text/html; charset=utf-8")
            if u.path == "/api/day":
                return self._json(STATE["day"])
            if u.path == "/api/goals":
                return self._json(STATE["goals"])
            if u.path == "/api/overlay":
                return self._json(STATE.get("overlay", {}))
            if u.path == "/api/alignment":
                return self._json(STATE["align"])
            if u.path == "/api/keylog":
                corrected = q.get("corrected", ["1"])[0] != "0"   # default: apply realignment
                return self._json(decode_keylog(int(q["seg"][0]), corrected))
            if u.path == "/video":
                return self._serve_video(int(q["seg"][0]), q.get("fs", ["0"])[0] == "1")
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:  # noqa: BLE001
            return self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        with LOCK:
            if u.path == "/api/goals":
                if not body.get("id"):
                    body["id"] = STATE["next_id"]
                    STATE["next_id"] += 1
                    STATE["goals"].append(body)
                else:
                    STATE["goals"] = [g for g in STATE["goals"] if g["id"] != body["id"]] + [body]
                persist_goals()
                return self._json({"ok": True, "id": body["id"], "n": len(STATE["goals"])})
            if u.path == "/api/goals/delete":
                STATE["goals"] = [g for g in STATE["goals"] if g["id"] != body.get("id")]
                persist_goals()
                return self._json({"ok": True, "n": len(STATE["goals"])})
        return self._send(404, b"not found", "text/plain")


def cmd_serve(args) -> None:
    global VERBOSE
    VERBOSE = args.verbose
    STATE.update(load_day(Path(args.data_dir)))

    if not args.no_prewarm:
        def warm():
            segs = STATE["day"]["segments"]
            for i in range(len(segs)):
                try:
                    faststart_path(i)   # serialized via REMUX_LOCK; live clicks queue behind one remux
                except Exception as e:  # noqa: BLE001
                    print(f"[faststart] seg {i} FAILED: {e}", flush=True)
            print(f"[faststart] pre-warm complete ({len(segs)} segments cached)", flush=True)
        threading.Thread(target=warm, daemon=True).start()
        print(f"[serve] pre-warming faststart cache in background -> {Path(args.data_dir)/'cache_faststart'}")
    d = STATE["day"]
    print(f"[serve] user {d['user'][:8]} {d['date']} | {d['n_segments']} segs, "
          f"{d['video_seconds']/3600:.2f}h | {len(STATE['goals'])} goals loaded")
    print(f"[serve] goals -> {STATE['goals_path']}  (write-through autosave)")
    al = STATE["align"]
    if not al.get("ok"):
        print(f"[serve] ⚠ alignment UNAVAILABLE ({al.get('error', '?')}) — viewer is GATED (won't load).", flush=True)
    else:
        asegs = al.get("segments", {})
        npause = sum(1 for v in asegs.values() if v.get("n_pauses"))
        nclosed = sum(1 for v in asegs.values() if v.get("n_pauses") and v.get("closed"))
        tag = " (auto-built)" if al.get("autobuilt") else ""
        print(f"[serve] keylog realignment{tag}: {npause} segments have idle-pauses "
              f"({nclosed} closed/exact, {npause - nclosed} flagged). Toggle in viewer with 'c'.", flush=True)
    print(f"[serve] http://{args.host}:{args.port}   tunnel: ssh -L {args.port}:localhost:{args.port} <node>")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


# ======================================================================
# CLI
# ======================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="uploads -> ordered day.json")
    pi.add_argument("--uploads", default=DEFAULT_UPLOADS)
    pi.add_argument("--user", required=True)
    pi.add_argument("--version", default=None, help="restrict to one recorder-version dir, e.g. 1.0.5")
    pi.add_argument("--tz", default="UTC", help="IANA tz for day bucketing + display")
    pi.add_argument("--date", default=None, help="YYYY-MM-DD: emit ordered day.json for this day")
    pi.add_argument("--out", default=None, help="output path for --date")
    pi.add_argument("--workers", type=int, default=16)
    pi.set_defaults(func=cmd_index)

    pa = sub.add_parser("align", help="(re)build alignment.json (serve auto-runs this)")
    pa.add_argument("--data_dir", required=True)
    pa.add_argument("--idle", type=float, default=120.0, help="recorder idle_timeout_secs")
    pa.add_argument("--tol", type=float, default=2.0)
    pa.set_defaults(func=cmd_align)

    pp = sub.add_parser("prewarm", help="parallel faststart remux of the day")
    pp.add_argument("--data_dir", required=True)
    pp.add_argument("--workers", type=int, default=8)
    pp.set_defaults(func=cmd_prewarm)

    ps = sub.add_parser("serve", help="run the local annotation viewer")
    ps.add_argument("--data_dir", required=True)
    ps.add_argument("--port", type=int, default=8753)
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--verbose", action="store_true", help="log every video/range request + remux")
    ps.add_argument("--no_prewarm", action="store_true", help="don't pre-remux the faststart cache on startup")
    ps.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
