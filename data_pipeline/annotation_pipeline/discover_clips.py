#!/usr/bin/env python3
"""Discover runnable clips across the whole upload tree, pre-gating out idle ones.

One clip = one recording (its segments form a contiguous timeline). A recording
is kept only if its keylogs carry enough *actionable* input (keypresses, clicks,
scrolls — not mouse-move/idle), which drops the ~46% idle/black recordings before
any VLM cost. Writes a clips file in the same schema as clips.json plus a report.

    python discover_clips.py                       # all versions -> clips_dataset.json
    python discover_clips.py --versions 1.0.2 1.0.3 --min-actionable 15
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import config
from common import keylog_summary

REC_RE = re.compile(r"^recording_(?P<rec>[0-9a-fA-F-]+)_seg(?P<seg>\d+).*\.mp4$")
# Presses + scroll are user intent; mouse-move and ContextChanged alone are idle.
ACTIONABLE = ("KeyPress", "MousePress", "MouseScroll")


def keylog_path(rec_dir: Path, rec: str, seg: int) -> Path:
    return rec_dir.parent / "keylogs" / f"input_{rec}_seg{seg:04d}.msgpack"


def actionable_events(path: Path) -> int:
    if not path.exists():
        return 0
    counts = keylog_summary(path).get("event_counts", {})
    return sum(int(counts.get(name, 0)) for name in ACTIONABLE)


def discover(raw_root: Path, versions: list[str], min_actionable: int, workers: int,
             max_seg: int):
    uploads = raw_root / "uploads"
    # (version, user, rec) -> sorted segment indices
    recordings: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for version in versions:
        for rec_dir in sorted(uploads.glob(f"{version}/*/recordings")):
            user = rec_dir.parent.name
            for mp4 in rec_dir.glob("recording_*_seg*.mp4"):
                m = REC_RE.match(mp4.name)
                if m:
                    recordings[(version, user, m.group("rec"))].append(int(m.group("seg")))

    def score(item):
        (version, user, rec), segs = item
        rec_dir = uploads / version / user / "recordings"
        total = sum(actionable_events(keylog_path(rec_dir, rec, s)) for s in segs)
        return (version, user, rec), sorted(segs), total

    with ThreadPoolExecutor(max_workers=workers) as pool:
        scored = list(pool.map(score, recordings.items()))

    clips: dict[str, dict[str, Any]] = {}
    kept = skipped = 0
    for (version, user, rec), segs, total in scored:
        if total < min_actionable:
            skipped += 1
            continue
        kept += 1
        # Chunk long recordings into bounded clips (one clip = <= max_seg
        # contiguous segments) for bounded per-task time, finer resume, and
        # balanced shards. rec[:8] keeps clip ids unique across recordings.
        for start in range(0, len(segs), max_seg):
            chunk = segs[start:start + max_seg]
            clip_id = f"{rec[:8]}_s{chunk[0]:04d}-{chunk[-1]:04d}"
            clips[clip_id] = {
                "version": version, "user_id": user, "recording_id": rec,
                "segment_start": chunk[0], "segment_end": chunk[-1],
            }
    report = {
        "n_recordings": len(recordings),
        "n_kept_recordings": kept,
        "n_skipped_idle": skipped,
        "n_clips": len(clips),
        "min_actionable": min_actionable,
        "max_segments_per_clip": max_seg,
        "versions": versions,
    }
    return clips, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=config.RAW_DATA_ROOT)
    ap.add_argument("--versions", nargs="+",
                    default=["0.1.0", "0.1.1", "1.0.0", "1.0.1", "1.0.2", "1.0.3"])
    ap.add_argument("--min-actionable", type=int, default=15,
                    help="Keep a recording only if its keylogs carry >= this many press/click/scroll events.")
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--max-segments-per-clip", type=int, default=12,
                    help="Split long recordings into clips of at most this many segments (~60 min).")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "clips_dataset.json")
    args = ap.parse_args()

    clips, report = discover(args.raw_root, args.versions, args.min_actionable,
                             args.max_workers, args.max_segments_per_clip)
    args.out.write_text(json.dumps({"_comment": "Auto-discovered by discover_clips.py", "clips": clips}, indent=2) + "\n")
    (args.out.parent / "discovery_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {len(clips)} clips to {args.out}")


if __name__ == "__main__":
    main()
