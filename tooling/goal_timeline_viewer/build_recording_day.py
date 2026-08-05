#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["av>=12", "msgpack>=1"]
# ///
"""Build a day.json for ONE recording (all its segments), schema-compatible with
annotator.py serve. Unlike `annotator.py index` (which buckets a participant's whole
DAY across recordings), this gives a clean single-recording timeline whose segment
indices match the annotation pipeline's per-recording seg numbers — so our extracted
goals (keyed by seg) overlay directly.

t_day is derived from each segment's OBS container creation_time (absolute UTC), so
recorder-off gaps between segments show up as real gaps on the timeline.

  uv run build_recording_day.py --uploads <ROOT> --recording <RECID> --out <DATA>/day.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import av
import msgpack

SEG_RE = re.compile(r"recording_(?P<rec>.+)_seg(?P<idx>\d+)\.mp4$")


def creation_time(container) -> dt.datetime | None:
    ct = container.metadata.get("creation_time")
    if not ct:
        for s in container.streams:
            ct = s.metadata.get("creation_time")
            if ct:
                break
    if not ct:
        return None
    return dt.datetime.fromisoformat(ct.replace("Z", "+00:00"))


def probe(video: Path):
    with av.open(str(video)) as c:
        v = next((s for s in c.streams if s.type == "video"), None)
        dur = float(c.duration / 1_000_000) if c.duration else (
            float(v.duration * v.time_base) if v and v.duration else 0.0)
        w = v.codec_context.width if v else 0
        h = v.codec_context.height if v else 0
        codec = v.codec_context.name if v else "?"
        return creation_time(c), dur, w, h, codec


def n_events(keylog: Path | None) -> int:
    if not keylog or not keylog.exists():
        return 0
    try:
        data = msgpack.unpackb(keylog.read_bytes(), raw=False, strict_map_key=False)
        return len(data) if hasattr(data, "__len__") else 0
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uploads", required=True)
    ap.add_argument("--recording", help="single recording id")
    ap.add_argument("--day-file", help="file of recording ids (one per line; trailing time/#comments "
                    "ignored) — build a whole-day timeline across all of them, ordered by creation_time")
    ap.add_argument("--day-name", default="day", help="recording/name field for a --day-file day")
    ap.add_argument("--tz", default="Europe/Berlin")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.day_file:
        recordings = [ln.split()[0] for ln in Path(args.day_file).read_text().splitlines()
                      if ln.strip() and not ln.lstrip().startswith("#")]
        day_label = args.day_name
    elif args.recording:
        recordings, day_label = [args.recording], args.recording
    else:
        raise SystemExit("provide --recording or --day-file")

    up = Path(args.uploads)
    vids = []
    for rec in recordings:
        v = sorted(up.glob(f"*/*/recordings/recording_{rec}*_seg*.mp4"),
                   key=lambda p: int(SEG_RE.search(p.name).group("idx")))
        vids.extend(v)
    if not vids:
        raise SystemExit(f"no segments for {recordings} under {up}")

    rows = []
    for vp in vids:
        m = SEG_RE.search(vp.name)
        rec, idx = m.group("rec"), int(m.group("idx"))
        kp = vp.parent.parent / "keylogs" / f"input_{rec}_seg{idx:04d}.msgpack"
        ct, dur, w, h, codec = probe(vp)
        rows.append(dict(recording_id=rec, seg=idx, video=str(vp),
                         keylog=str(kp) if kp.exists() else None,
                         start=ct, duration_s=round(dur, 3), width=w, height=h,
                         codec=codec, n_events=n_events(kp if kp.exists() else None)))

    rows = [r for r in rows if r["start"] is not None] or rows
    rows.sort(key=lambda r: (r["start"] or dt.datetime.min, r["seg"]))
    day_start = rows[0]["start"]
    segments = []
    prev_end = None
    for r in rows:
        t_day = (r["start"] - day_start).total_seconds() if r["start"] else (
            segments[-1]["t_day"] + segments[-1]["duration_s"] if segments else 0.0)
        gap = 0.0 if prev_end is None or r["start"] is None else max(0.0, (r["start"] - prev_end).total_seconds())
        segments.append(dict(
            recording_id=r["recording_id"], seg=r["seg"],
            sid=f"{r['recording_id'][:8]}_s{r['seg']:04d}", video=r["video"], keylog=r["keylog"],
            start_utc=r["start"].isoformat() if r["start"] else None,
            t_day=round(t_day, 3), duration_s=r["duration_s"],
            gap_before_s=round(gap, 3), width=r["width"], height=r["height"],
            codec=r["codec"], n_events=r["n_events"]))
        if r["start"]:
            prev_end = r["start"] + dt.timedelta(seconds=r["duration_s"])

    day = dict(
        user=str(Path(vids[0]).parent.parent.parent.name), recording=day_label,
        date=(day_start.isoformat()[:10] if day_start else None), tz=args.tz,
        day_start_utc=day_start.isoformat() if day_start else None,
        n_segments=len(segments),
        video_seconds=round(sum(s["duration_s"] for s in segments), 1),
        span_seconds=round(segments[-1]["t_day"] + segments[-1]["duration_s"], 1),
        segments=segments)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(day, indent=2))
    codecs = {s["codec"] for s in segments}
    print(f"wrote {out}: {len(segments)} segments | recorded {day['video_seconds']/3600:.2f}h | "
          f"span {day['span_seconds']/3600:.2f}h | codecs={codecs} | date={day['date']}")
    gaps = [s for s in segments if s["gap_before_s"] > 120]
    print(f"  {len(gaps)} gaps >2min" + (f"; largest {max(s['gap_before_s'] for s in gaps):.0f}s" if gaps else ""))


if __name__ == "__main__":
    main()
