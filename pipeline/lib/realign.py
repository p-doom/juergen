"""Correct Crowd-Cast keylog timestamps for recorder idle pauses."""

from __future__ import annotations

import datetime
import math
import struct
from itertools import pairwise
from pathlib import Path
from typing import Any

from pipeline.lib.common import load_keylog_entries

IDLE_TIMEOUT = 120.0
CLOSURE_TOL = 2.0

INPUT_TYPES = {
    "KeyPress",
    "KeyRelease",
    "MousePress",
    "MouseRelease",
    "MouseScroll",
    "MouseMove",
}

CLOSED_STATUSES = {"aligned", "exact", "benign_idle"}
EXCLUSION_REASONS = {
    "no_closed_candidate",
    "nonincreasing_creation_time",
    "unattributed_global_splice",
    "unreadable_creation_time",
}

_EPOCH_1904 = datetime.datetime(1904, 1, 1, tzinfo=datetime.UTC)


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


def mp4_creation_time(path: Path) -> datetime.datetime | None:
    """Read the creation timestamp from an MP4 movie header."""
    with path.open("rb") as f:
        f.seek(0, 2)
        fend = f.tell()
        f.seek(0)
        ms, me = _find_box(f, fend, b"moov")
        if ms is None:
            return None
        f.seek(ms)
        vs, _ve = _find_box(f, me, b"mvhd")
        if vs is None:
            return None
        f.seek(vs)
        ver = f.read(1)[0]
        f.read(3)
        if ver == 1:
            ct = struct.unpack(">Q", f.read(8))[0]
            f.read(8)
            ts = struct.unpack(">I", f.read(4))[0]
            struct.unpack(">Q", f.read(8))[0]
        elif ver == 0:
            ct = struct.unpack(">I", f.read(4))[0]
            f.read(4)
            ts = struct.unpack(">I", f.read(4))[0]
            struct.unpack(">I", f.read(4))[0]
        else:
            return None
        if ct == 0 or ts == 0:
            return None
        return _EPOCH_1904 + datetime.timedelta(seconds=ct)


def load_keylog(path: str) -> list[Any]:
    """Load the exact Crowd-Cast event-list schema."""
    return load_keylog_entries(Path(path))


def input_timestamps_s(events: list[Any]) -> list[float]:
    """Input-event timestamps (s) for the six idle-gating event types."""
    return [entry[0] / 1e6 for entry in events if entry[1][0] in INPUT_TYPES]


def keylog_span_s(events: list[Any]) -> float:
    """Last event timestamp (s) -- the keylog span."""
    return events[-1][0] / 1e6


def compute_splices(input_ts: list[float]) -> list[dict]:
    """Infer recorder pauses from an ordered input-event clock."""
    splices: list[dict] = []
    cum = 0.0
    bounds = [0.0] + list(input_ts)
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        gap = b - a
        if gap <= IDLE_TIMEOUT:
            continue
        kp = a + IDLE_TIMEOUT
        collapse = gap - IDLE_TIMEOUT
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


def _classify(
    splices: list[dict],
    residual: float,
    corr_end: float,
    video_dur: float,
    has_leading: bool,
    refined: bool,
) -> str:
    if abs(residual) <= CLOSURE_TOL:
        return "exact" if splices else "aligned"
    if corr_end > video_dur + CLOSURE_TOL:
        return "UNDER"
    if has_leading and not refined:
        return "needs_review"
    if residual > CLOSURE_TOL:
        return "benign_idle"
    return "needs_review"


def _refine_and_classify(
    local: list[dict],
    S: float,
    V: float,
    first_input: float,
    model: str,
) -> dict:
    """Refine and classify one candidate reconstruction."""
    local = [dict(s) for s in sorted(local, key=lambda x: x["kp"])]
    for s in local:
        s["leading"] = s["kp"] < first_input - 1e-9
    has_leading = any(s["leading"] for s in local)
    fresh_collapse = sum(s["collapse"] for s in local if not s["leading"])
    total = sum(s["collapse"] for s in local)
    overhang = max(0.0, S - V)

    refined = False
    leading_method = "n/a"
    if has_leading:
        if (
            abs(total - overhang) <= CLOSURE_TOL
            and overhang - fresh_collapse >= -CLOSURE_TOL
        ):
            for s in local:
                if s["leading"]:
                    s["collapse"] = max(0.0, overhang - fresh_collapse)
                    break
            refined = True
            leading_method = "overhang"
        else:
            leading_method = model

    cum = 0.0
    for s in local:
        s["vp"] = s["kp"] - cum
        cum += s["collapse"]

    total_collapse = sum(s["collapse"] for s in local)
    residual = total_collapse - overhang
    corr_end = keylog_to_video(S, local)
    status = _classify(local, residual, corr_end, V, has_leading, refined)
    return {
        "splices": [
            {
                "kp": s["kp"],
                "vp": s["vp"],
                "collapse": s["collapse"],
                "leading": s["leading"],
            }
            for s in local
        ],
        "n_pauses": len(local),
        "status": status,
        "closed": status in CLOSED_STATUSES,
        "leading_method": leading_method,
        "total_collapse_s": total_collapse,
        "overhang_s": overhang,
        "residual_s": residual,
        "corr_end_s": corr_end,
    }


def _exclude_recording(segs: list[dict], reason: str) -> dict[str, dict]:
    return {
        segment["segment_id"]: {
            "segment_id": segment["segment_id"],
            "segment_idx": segment["segment_idx"],
            "closed": False,
            "exclusion_reason": reason,
            "candidates": {},
        }
        for segment in segs
    }


def realign_recording(segs: list[dict]) -> dict[str, dict]:
    """Realign every segment of ONE recording.

    The local candidate captures pauses within a segment. The recording-global
    candidate uses MP4 creation timestamps to capture pauses crossing segment
    boundaries. Segments without a closed candidate receive an exclusion receipt.
    """
    if not segs:
        raise ValueError("recording has no segments")
    expected_fields = {
        "segment_id",
        "segment_idx",
        "keylog_path",
        "video_path",
        "video_dur_s",
    }
    for s in segs:
        if set(s) != expected_fields:
            raise ValueError(f"invalid realignment segment fields: {sorted(s)}")
        if (
            not isinstance(s["segment_id"], str)
            or not s["segment_id"]
            or isinstance(s["segment_idx"], bool)
            or not isinstance(s["segment_idx"], int)
            or s["segment_idx"] < 0
            or not isinstance(s["keylog_path"], str)
            or not Path(s["keylog_path"]).is_absolute()
            or not isinstance(s["video_path"], str)
            or not Path(s["video_path"]).is_absolute()
        ):
            raise ValueError(f"invalid realignment segment: {s!r}")
        duration = s["video_dur_s"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError(f"invalid manifest video duration: {duration!r}")

    segs = sorted(segs, key=lambda s: s["segment_idx"])
    if len({s["segment_id"] for s in segs}) != len(segs) or len(
        {s["segment_idx"] for s in segs}
    ) != len(segs):
        raise ValueError("recording contains duplicate segments")
    n = len(segs)

    its_l: list[list[float]] = []
    span_l: list[float] = []
    vdur_l: list[float] = []
    for s in segs:
        duration = s["video_dur_s"]
        ev = load_keylog(s["keylog_path"])
        its_l.append(input_timestamps_s(ev))
        span_l.append(keylog_span_s(ev))
        vdur_l.append(float(duration))

    offsets = [0.0] * n
    if n > 1:
        try:
            creation_times = [mp4_creation_time(Path(s["video_path"])) for s in segs]
        except (IndexError, OSError, OverflowError, struct.error):
            return _exclude_recording(segs, "unreadable_creation_time")
        if any(value is None for value in creation_times):
            return _exclude_recording(segs, "unreadable_creation_time")
        exact_times = [value for value in creation_times if value is not None]
        if any(right <= left for left, right in pairwise(exact_times)):
            return _exclude_recording(segs, "nonincreasing_creation_time")
        offsets = [
            (creation_time - exact_times[0]).total_seconds()
            for creation_time in exact_times
        ]

    global_inputs: list[float] = []
    for i in range(n):
        for t in its_l[i]:
            global_inputs.append(offsets[i] + t)
    global_inputs.sort()
    gsplices = compute_splices(global_inputs)
    seg_g: list[list[dict]] = [[] for _ in range(n)]
    segment_ends = offsets[1:] + [offsets[-1] + span_l[-1]]
    for g in gsplices:
        matches = [i for i in range(n) if offsets[i] <= g["kp"] < segment_ends[i]]
        if len(matches) != 1:
            return _exclude_recording(segs, "unattributed_global_splice")
        index = matches[0]
        seg_g[index].append({"kp": g["kp"] - offsets[index], "collapse": g["collapse"]})

    out: dict[str, dict] = {}
    for i in range(n):
        seg = segs[i]
        S, V = span_l[i], vdur_l[i]
        first_input = its_l[i][0] if its_l[i] else 0.0

        naive = compute_splices(its_l[i])
        glob = sorted(seg_g[i], key=lambda x: x["kp"])
        candidates = [
            (model, _refine_and_classify(splices, S, V, first_input, model))
            for model, splices in (("naive", naive), ("global", glob))
        ]
        accepted = next(
            ((model, result) for model, result in candidates if result["closed"]),
            None,
        )
        if accepted is None:
            out[seg["segment_id"]] = {
                "segment_id": seg["segment_id"],
                "segment_idx": seg["segment_idx"],
                "closed": False,
                "exclusion_reason": "no_closed_candidate",
                "candidates": {
                    model: {
                        "status": result["status"],
                        "residual_s": result["residual_s"],
                        "corr_end_s": result["corr_end_s"],
                    }
                    for model, result in candidates
                },
            }
            continue
        model, res = accepted
        res.update(
            {
                "segment_id": seg["segment_id"],
                "segment_idx": seg["segment_idx"],
                "model": model,
                "keylog_span_s": S,
                "video_dur_s": V,
                "first_input_s": first_input,
                "exclusion_reason": None,
            }
        )
        out[seg["segment_id"]] = res
    return out
