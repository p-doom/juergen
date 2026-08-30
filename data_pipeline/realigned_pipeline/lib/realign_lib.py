"""Keylog<->video realignment (crowd-cast recorder pause-clock bug).

Self-contained port of the validated ``realignment/align_lib.py`` primitives plus
the spec-v2 per-recording algorithm (the reference ``realignment/`` scripts only
implement the creation_time *fallback* and a coarse 4-way category; this module
adds the frame-exact overhang refinement and the full status taxonomy).

Root cause (spec v1): keylog event timestamps come from OBS's global compositor
clock (``obs_get_video_frame_time() - recording_start_ns``), which keeps advancing
while the recording *output* is paused on idle. The mp4 collapses the paused span
out, so for every point in time ``keylog_time = video_pts + cumulative_paused``.
The map back is piecewise-linear slope-1 with a downward ``collapse`` at each idle
pause (``collapse = idle_gap - idle_timeout``).

Per recording the pauses are of two kinds (spec sec.2-3):
  * **fresh** -- the idle started within this segment. ``collapse = gap - timeout``
    from the microsecond keylog (segment offset cancels). Exact; always trusted.
  * **leading / overhanging** -- the idle started before the segment's first input
    (inherited across a segment boundary, or the recording's opening idle). At most
    one, always the segment's first pause. Its precise collapse is recovered from
    the mp4 stream duration: ``leading = overhang - Sum(fresh)`` where
    ``overhang = keylog_span - video_dur`` (frame-exact), provided the segment has
    no trailing-idle confound (``|continuous_total - overhang| <= tol``); otherwise
    we keep the creation_time-stitched estimate and flag it.

Status per segment (spec sec.5) -- gate trust on ``closed``:
  aligned       no pause, keylog == video.                              (trust)
  exact         pauses, residual ~ 0; frame-exact.                      (trust)
  benign_idle   Sum(collapse) > overhang but corr_end <= video_dur:
                uncaptured trailing/leading idle *video* (no keylog).   (trust)
  needs_review  leading idle whose collapse the overhang could NOT
                verify (creation_time fallback). Best-effort, inspect.  (flag)
  UNDER         corr_end > video_dur: keylog still past the video, a
                real under-collapse (e.g. a manual/non-idle pause).     (flag)

Source keylogs/mp4s are read-only inputs; this module never modifies them.
"""
from __future__ import annotations

import datetime
import struct
from dataclasses import dataclass, field
from typing import Any

import msgpack

IDLE_TIMEOUT = 120.0  # recorder idle_timeout_secs default
CLOSURE_TOL = 2.0  # closure threshold (s)

INPUT_TYPES = {
    "KeyPress", "KeyRelease", "MousePress",
    "MouseRelease", "MouseScroll", "MouseMove",
}

CLOSED_STATUSES = {"aligned", "exact", "benign_idle"}

_EPOCH_1904 = datetime.datetime(1904, 1, 1)


# ---------------------------------------------------------------------------
# mp4 container header (no frame decode)
# ---------------------------------------------------------------------------

def _find_box(f, end, target):
    while f.tell() < end:
        start = f.tell()
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        size = struct.unpack(">I", hdr[:4])[0]
        typ = hdr[4:8]
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            hsize = 16
        elif size == 0:
            size = end - start
            hsize = 8
        else:
            hsize = 8
        if typ == target:
            return start + hsize, start + size
        f.seek(start + size)
    return None, None


def mp4_mvhd(path: str) -> tuple[datetime.datetime | None, float | None]:
    """Return (creation_time, duration_s) from moov/mvhd. Header-only read."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        fend = f.tell()
        f.seek(0)
        ms, me = _find_box(f, fend, b"moov")
        if ms is None:
            return None, None
        f.seek(ms)
        vs, _ve = _find_box(f, me, b"mvhd")
        if vs is None:
            return None, None
        f.seek(vs)
        ver = f.read(1)[0]
        f.read(3)  # flags
        if ver == 1:
            ct = struct.unpack(">Q", f.read(8))[0]
            f.read(8)  # modification_time
            ts = struct.unpack(">I", f.read(4))[0]
            du = struct.unpack(">Q", f.read(8))[0]
        else:
            ct = struct.unpack(">I", f.read(4))[0]
            f.read(4)
            ts = struct.unpack(">I", f.read(4))[0]
            du = struct.unpack(">I", f.read(4))[0]
        if ts == 0:
            return None, None
        ctime = _EPOCH_1904 + datetime.timedelta(seconds=ct) if ct else None
        return ctime, du / ts


# ---------------------------------------------------------------------------
# keylog primitives
# ---------------------------------------------------------------------------

def load_keylog(path: str) -> list[Any]:
    """Raw msgpack event list: [[ts_us, [type, args]], ...]. [] on missing/empty."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return []
    if not raw:
        return []
    events = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    return events if isinstance(events, list) else []


def input_timestamps_s(events: list[Any]) -> list[float]:
    """Sorted input-event timestamps (s) -- the 6 idle-gating event types."""
    ts = [e[0] / 1e6 for e in events
          if isinstance(e, list) and len(e) >= 2
          and isinstance(e[1], list) and e[1] and e[1][0] in INPUT_TYPES]
    ts.sort()
    return ts


def keylog_span_s(events: list[Any]) -> float:
    """Last event timestamp (s) -- the keylog span."""
    ts = [e[0] for e in events if isinstance(e, list) and e]
    return (max(ts) / 1e6) if ts else 0.0


# ---------------------------------------------------------------------------
# splice model (spec sec.3-4)
# ---------------------------------------------------------------------------

@dataclass
class Splice:
    kp: float  # keylog-time pause point (local s)
    vp: float  # video-time pause point (local s)
    collapse: float  # collapsed seconds
    leading: bool = False  # True for the (single) leading/overhanging idle


def compute_splices(input_ts: list[float], idle_timeout: float = IDLE_TIMEOUT,
                    prior_idle: float = 0.0) -> list[dict]:
    """Spec sec.3 on a single (segment-local or global) input stream.

    prior_idle credits idle already accrued before t=0 (cross-segment leading).
    Returns [{kp, vp, collapse}] sorted by kp.
    """
    splices: list[dict] = []
    cum = 0.0
    bounds = [0.0] + list(input_ts)
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        lead = prior_idle if i == 0 else 0.0
        gap = (b - a) + lead
        if gap <= idle_timeout:
            continue
        kp = a + (idle_timeout - lead)
        collapse = gap - idle_timeout
        splices.append({"kp": kp, "vp": kp - cum, "collapse": collapse})
        cum += collapse
    return splices


def keylog_to_video(kt: float, splices: list[dict]) -> float:
    """Spec sec.4: map a keylog timestamp (s) to recorded-video time (s)."""
    cum = 0.0
    for s in splices:
        if kt <= s["kp"]:
            break
        if kt < s["kp"] + s["collapse"]:
            return s["vp"]  # inside collapsed span -> clamp to vp
        cum += s["collapse"]
    return kt - cum


def closure(splices: list[dict], keylog_span: float, video_dur: float,
            tol: float = CLOSURE_TOL) -> dict:
    """Spec sec.5 closure certificate."""
    total = sum(s["collapse"] for s in splices)
    overhang = max(0.0, keylog_span - video_dur)
    return {
        "total_collapse_s": total,
        "overhang_s": overhang,
        "residual_s": total - overhang,
        "closed": abs(total - overhang) <= tol,
    }


# ---------------------------------------------------------------------------
# per-recording realignment (spec sec.3, with overhang refinement + sec.5 status)
# ---------------------------------------------------------------------------

def _classify(splices: list[dict], overhang: float, residual: float,
              corr_end: float, video_dur: float, has_leading: bool,
              refined: bool, tol: float) -> str:
    if not splices:
        return "aligned"
    if abs(residual) <= tol:
        return "exact"
    if corr_end > video_dur + tol:
        return "UNDER"
    if has_leading and not refined:
        return "needs_review"
    if residual > tol:  # over-collapse w/ corr_end <= video -> uncaptured trailing idle
        return "benign_idle"
    return "needs_review"


def _refine_and_classify(local: list[dict], S: float, V: float | None,
                         first_input: float, model: str, tol: float) -> dict:
    """Given a chosen candidate splice list (local kp/collapse), apply the
    overhang refinement to the leading idle and compute the spec-v2 status."""
    # leading/overhanging = pause that fires at/before this segment's first input
    # (spec: at most one, the segment's first); the rest are fresh (exact gap-T).
    local = [dict(s) for s in sorted(local, key=lambda x: x["kp"])]
    for s in local:
        s["leading"] = s["kp"] < first_input - 1e-9
    has_leading = any(s["leading"] for s in local)
    fresh_collapse = sum(s["collapse"] for s in local if not s["leading"])
    total = sum(s["collapse"] for s in local)
    overhang = max(0.0, S - (V or 0.0)) if V is not None else 0.0

    refined = False
    leading_method = "n/a"
    if V is not None and has_leading:
        # frame-exact leading collapse from the mp4 duration, when the segment has
        # no trailing-idle confound (continuous total closes against overhang).
        if abs(total - overhang) <= tol and (overhang - fresh_collapse) >= -tol:
            for s in local:
                if s["leading"]:
                    s["collapse"] = max(0.0, overhang - fresh_collapse)
                    break
            refined = True
            leading_method = "overhang"
        else:
            leading_method = model  # creation-stitched / naive estimate, unverified

    cum = 0.0
    for s in local:
        s["vp"] = s["kp"] - cum
        cum += s["collapse"]

    total_collapse = sum(s["collapse"] for s in local)
    residual = total_collapse - overhang
    corr_end = keylog_to_video(S, local)
    status = (_classify(local, overhang, residual, corr_end, V, has_leading,
                        refined, tol) if V is not None else "no_video")
    return {
        "splices": [{"kp": s["kp"], "vp": s["vp"], "collapse": s["collapse"],
                     "leading": s["leading"]} for s in local],
        "n_pauses": len(local), "status": status,
        "closed": status in CLOSED_STATUSES, "leading_method": leading_method,
        "total_collapse_s": total_collapse, "overhang_s": overhang,
        "residual_s": residual, "corr_end_s": corr_end,
    }


def realign_recording(segs: list[dict], idle_timeout: float = IDLE_TIMEOUT,
                      tol: float = CLOSURE_TOL) -> dict[str, dict]:
    """Realign every segment of ONE recording.

    Per segment, two candidate reconstructions are computed and the more
    trustworthy is chosen (the validated ``hybrid.py`` strategy):
      * **naive** -- per-segment ``compute_splices`` on the local input stream
        (no cross-segment offset noise; exact when the idle is fully internal or
        the segment's own opening idle).
      * **global** -- one splice pass on the recording's merged, creation_time-
        stitched global input stream, then attributed back per segment (recovers
        idles that *overhang* a segment boundary, which naive cannot see).
    Prefer naive when it closes; else global when it closes; else the smaller
    residual. The chosen leading collapse is then overhang-refined (frame-exact)
    and the segment gets its spec-v2 status.

    segs: list of {segment_id, segment_idx, keylog_path, video_path,
                   video_dur_s?(float|None), creation_time?(datetime|None)}.
          Pass ALL segments of the recording present in the *source* tree (not
          just dataset-kept ones) so overhanging idles thread correctly.

    Returns {segment_id: result}; result keys: splices ([{kp,vp,collapse,leading}]),
      status, closed, model, leading_method, n_pauses, total_collapse_s,
      overhang_s, residual_s, corr_end_s, keylog_span_s, video_dur_s,
      first_input_s, segment_idx.
    """
    segs = sorted(segs, key=lambda s: int(s["segment_idx"]))
    n = len(segs)

    its_l: list[list[float]] = []
    span_l: list[float] = []
    vdur_l: list[float | None] = []
    ct_l: list[datetime.datetime | None] = []
    for s in segs:
        ev = load_keylog(s["keylog_path"])
        its_l.append(input_timestamps_s(ev))
        span_l.append(keylog_span_s(ev))
        vd = s.get("video_dur_s")
        ct = s.get("creation_time")
        if (vd is None or ct is None) and s.get("video_path"):
            mct, mdu = mp4_mvhd(s["video_path"])
            vd = vd if vd is not None else mdu
            ct = ct if ct is not None else mct
        vdur_l.append(vd)
        ct_l.append(ct)

    # continuous obs-clock: wall span per segment (creation_time delta preferred,
    # else the segment's own keylog/video extent), then cumulative offsets.
    wall = [0.0] * n
    for i in range(n - 1):
        if ct_l[i] and ct_l[i + 1]:
            wall[i] = (ct_l[i + 1] - ct_l[i]).total_seconds()
        else:
            wall[i] = max(span_l[i], vdur_l[i] or 0.0)
    if n:
        wall[n - 1] = max(span_l[n - 1], vdur_l[n - 1] or 0.0)
    offset = [0.0] * n
    for i in range(1, n):
        offset[i] = offset[i - 1] + wall[i - 1]

    # one splice pass on the merged global input stream, attributed back to the
    # segment whose wall span contains each pause point.
    global_inputs: list[float] = []
    for i in range(n):
        for t in its_l[i]:
            global_inputs.append(offset[i] + t)
    global_inputs.sort()
    gsplices = compute_splices(global_inputs, idle_timeout)
    seg_g: list[list[dict]] = [[] for _ in range(n)]
    for g in gsplices:
        idx = n - 1
        for i in range(n):
            if offset[i] <= g["kp"] < offset[i] + wall[i]:
                idx = i
                break
        seg_g[idx].append({"kp": g["kp"] - offset[i], "collapse": g["collapse"]})

    out: dict[str, dict] = {}
    for i in range(n):
        seg = segs[i]
        S, V = span_l[i], vdur_l[i]
        first_input = its_l[i][0] if its_l[i] else 0.0

        naive = compute_splices(its_l[i], idle_timeout)
        glob = sorted(seg_g[i], key=lambda x: x["kp"])

        if V is None:
            model, chosen = "naive", naive
        else:
            cn, cg = closure(naive, S, V, tol), closure(glob, S, V, tol)
            if cn["closed"]:
                model, chosen = "naive", naive
            elif cg["closed"]:
                model, chosen = "global", glob
            elif abs(cn["residual_s"]) <= abs(cg["residual_s"]):
                model, chosen = "naive", naive
            else:
                model, chosen = "global", glob

        res = _refine_and_classify(chosen, S, V, first_input, model, tol)
        res.update({
            "segment_id": seg["segment_id"],
            "segment_idx": int(seg["segment_idx"]),
            "model": model,
            "keylog_span_s": S,
            "video_dur_s": V,
            "first_input_s": first_input,
        })
        out[seg["segment_id"]] = res
    return out
