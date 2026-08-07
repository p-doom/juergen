"""Day grouping + day streams for day-scope annotation methods.

A day is the work unit for sequential-watching methods
(``INPUT_KIND="days"``): all of one user's segments whose recordings started
on the same local calendar date, laid out on a wall-clock day axis. Wall-clock
placement comes from each segment video's mp4 ``mvhd creation_time``
(header-only read via ``lib.realign.mp4_mvhd``), joined through the
stage-00 clip manifest (``video_path``/``user_id`` per ``segment_id``); the
stage-03 filter artifact remains the sole authority for which segments and
ticks are usable.

A ``DayStream`` concatenates the day's ``SegmentView``s (lib/views, @k fps
within filter survivors) onto the day clock: every selected frame gets a
day-global dense index (0..N-1 in time order — the ``frame <N>`` the model is
shown; never persisted), a day time, its canonical per-frame action label,
and its ``(segment_id, master_idx)`` master coordinates for emit time. Frames
are cut into chunks at real gaps > ``gap_cut_s`` (the recording stopped);
evidence context never crosses a chunk boundary.
"""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pipeline.lib.action_format import get_formatter
from pipeline.lib.common import read_jsonl
from pipeline.lib.events import load_events
from pipeline.lib.realign import mp4_mvhd
from pipeline.lib.views import FilterArtifact, SegmentView

DEFAULT_TZ = "Europe/Berlin"
DEFAULT_GAP_CUT_S = 180.0  # a recording gap > this splits the day into chunks


def fmt_t(seconds: float) -> str:
    """Day-clock label ``+HH:MM:SS`` (elapsed since the day's first frame)."""
    s = round(seconds)
    return f"+{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


@dataclass(frozen=True)
class DayFrame:
    """One selected frame on the day axis. ``day_idx`` is the dense prompt
    index (what the model sees and anchors thoughts to); ``segment_id`` +
    ``master_idx`` are the persistent coordinates it maps back to at emit
    time. ``action`` is the frame's canonical label (its ownership window's
    input), shown verbatim in the frame label."""

    day_idx: int
    t_day_s: float
    segment_id: str
    recording_id: str | None
    master_idx: int
    image: str
    action: str


def frame_label(fr: DayFrame) -> str:
    return f"frame {fr.day_idx} | {fmt_t(fr.t_day_s)} | action: {fr.action}"


@dataclass
class DayStream:
    day_tag: str
    user_id: str
    date: str
    frames: list[DayFrame]              # dense day_idx == list position
    chunks: list[list[DayFrame]]        # frames split at gaps > gap_cut_s
    gap_cut_s: float
    n_segments: int

    def context_before(self, day_idx: int, n: int) -> list[DayFrame]:
        """The n frames up to and including day_idx, within its chunk (never
        across a recording gap) — the future-blind evidence window."""
        for chunk in self.chunks:
            if chunk[0].day_idx <= day_idx <= chunk[-1].day_idx:
                pos = day_idx - chunk[0].day_idx
                return chunk[max(0, pos - n + 1): pos + 1]
        raise KeyError(f"frame {day_idx} not in any chunk of {self.day_tag}")


def segment_actions(view: SegmentView) -> list[str]:
    """Canonical per-frame action labels for a segment view (same derivation
    as stage 03b's frames mode; method-internal, never persisted)."""
    events, _ = (load_events(Path(view.keylog_path)) if view.keylog_path else ([], None))
    result = get_formatter("canonical").format_segment(
        events, view.windows(), view.dead_zones, master_fps=view.master_fps
    )
    return result.labels


def load_clips_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """{segment_id -> stage-00 row}; needs video_path + user_id per segment."""
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"empty/missing clips manifest: {path}")
    by_seg = {str(r["segment_id"]): r for r in rows}
    probe = next(iter(by_seg.values()))
    for key in ("video_path", "user_id"):
        if key not in probe:
            raise ValueError(f"clips manifest rows lack {key!r} (not a stage-00 manifest?): {path}")
    return by_seg


def build_day_index(
    art: FilterArtifact,
    clips_manifest_path: Path,
    *,
    tz: str = DEFAULT_TZ,
    workers: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Group the filter artifact's usable segments into user-days.

    Returns (day rows, counters). A day row:
      {day_tag, user_id, date, tz, n_segments, n_kept,
       segments: [{segment_id, recording_id, segment_idx, t_day_s, n_kept}]}
    ordered by wall clock within the day. Segments whose video lacks a
    readable mvhd creation_time are counted (``n_undatable``) and skipped —
    never silently placed.
    """
    zone = ZoneInfo(tz)
    manifest = load_clips_manifest(clips_manifest_path)
    usable = art.usable_rows()
    missing = [r for r in usable if str(r["segment_id"]) not in manifest]

    def probe(row: dict[str, Any]) -> dict[str, Any] | None:
        seg = str(row["segment_id"])
        m = manifest[seg]
        created, _dur = mp4_mvhd(str(m["video_path"]))
        if created is None:
            return None
        return {
            "segment_id": seg,
            "recording_id": row.get("recording_id"),
            "segment_idx": row.get("segment_idx"),
            "user_id": str(m["user_id"]),
            "start_ts": created.timestamp(),
            "n_kept": int(row.get("n_kept") or 0),
        }

    candidates = [r for r in usable if str(r["segment_id"]) in manifest]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        probed = [p for p in ex.map(probe, candidates) if p is not None]
    n_undatable = len(candidates) - len(probed)

    by_day: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for p in probed:
        date = dt.datetime.fromtimestamp(p["start_ts"], zone).date().isoformat()
        by_day.setdefault((p["user_id"], date), []).append(p)

    days: list[dict[str, Any]] = []
    for (user, date), segs in sorted(by_day.items()):
        segs.sort(key=lambda s: s["start_ts"])
        day_start = segs[0]["start_ts"]
        for s in segs:
            s["t_day_s"] = round(s["start_ts"] - day_start, 3)
            del s["start_ts"], s["user_id"]
        days.append({
            "day_tag": f"u{user[:8]}_{date.replace('-', '')}",
            "user_id": user,
            "date": date,
            "tz": tz,
            "n_segments": len(segs),
            "n_kept": sum(s["n_kept"] for s in segs),
            "segments": segs,
        })
    counters = {
        "n_usable_segments": len(usable),
        "n_not_in_manifest": len(missing),
        "n_undatable": n_undatable,
        "n_days": len(days),
    }
    return days, counters


def build_day_stream(
    day_row: dict[str, Any],
    art: FilterArtifact,
    *,
    fps: float,
    fps_mode: str = "exact",
    gap_cut_s: float = DEFAULT_GAP_CUT_S,
    t1: float | None = None,
) -> DayStream:
    """Concatenate the day's segment views onto the day clock. Frames sort by
    day time (segments already come wall-clock-ordered), day_idx is assigned
    densely, and chunks split at gaps > gap_cut_s. ``t1`` caps the stream at
    that many day-seconds (validation/smoke slices — a capped run is a
    partial artifact, never a corpus result)."""
    raw: list[tuple[float, str, str | None, int, str, str]] = []
    for seg in day_row["segments"]:
        if t1 is not None and float(seg["t_day_s"]) > t1:
            continue
        view = art.segment_view(str(seg["segment_id"]), fps, fps_mode)
        if not view.frames:
            continue
        actions = segment_actions(view)
        t0 = float(seg["t_day_s"])
        for f in view.frames:
            t = t0 + f.t_s
            if t1 is not None and t > t1:
                continue
            raw.append((t, view.segment_id, view.recording_id,
                        f.master_idx, str(f.image), actions[f.view_idx]))
    raw.sort(key=lambda r: r[0])

    frames = [
        DayFrame(day_idx=i, t_day_s=t, segment_id=sid, recording_id=rid,
                 master_idx=mi, image=img, action=act)
        for i, (t, sid, rid, mi, img, act) in enumerate(raw)
    ]
    chunks: list[list[DayFrame]] = []
    for fr in frames:
        if not chunks or fr.t_day_s - chunks[-1][-1].t_day_s > gap_cut_s:
            chunks.append([])
        chunks[-1].append(fr)
    return DayStream(
        day_tag=str(day_row["day_tag"]),
        user_id=str(day_row["user_id"]),
        date=str(day_row["date"]),
        frames=frames,
        chunks=chunks,
        gap_cut_s=gap_cut_s,
        n_segments=int(day_row["n_segments"]),
    )
