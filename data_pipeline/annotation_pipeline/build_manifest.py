#!/usr/bin/env python3
"""Build a stage-00 manifest over a raw crowd-cast uploads tree.

The annotation pipeline consumes a JSONL manifest with one row per segment
(see ``manifest.selected_segments.jsonl`` in the curated subsets). The raw
``crowd-cast-2026-06-18`` dataset ships no such manifest — just
``uploads/<version>/<user_id>/recordings/recording_<rid>_seg<NNNN>.mp4`` with a
sibling ``keylogs/input_<rid>_seg<NNNN>.msgpack``. This walks that tree, probes
each MP4 with OpenCV for fps/frame-count/duration/dims, pairs it with its
keylog, and emits the rows stage 01 needs:

    video_path, keylog_path, segment_id, segment_idx, recording_id,
    video_duration_s, video_fps, video_frame_count, video_width, video_height,
    video_ok  (+ version / user_id for provenance)

Segments without a keylog are skipped by default because they cannot provide
action supervision. Pass ``--keep-missing-keylog`` to retain them for visual
inspection; their Stage-01 event timeline will be empty.

    PYTHONPATH=. python3 -m annotation_pipeline.build_manifest \
        --dataset-root /fast/project/HFMI_SynergyUnit/p-doom/crowd-cast/crowd-cast-2026-06-18 \
        --out manifest.crowd-cast-2026-06-18.jsonl --workers 32
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# recording_<rid>_seg<NNNN>.mp4  ->  rid, NNNN
_NAME_RE = re.compile(r"^recording_(?P<rid>.+)_seg(?P<idx>\d+)\.mp4$")


def probe_video(path: Path) -> dict[str, Any]:
    """fps/frame-count/duration/dims via OpenCV. video_ok=False on any failure."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return {"video_ok": False}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok = fps > 0 and n > 0 and w > 0 and h > 0
        return {
            "video_ok": ok,
            "video_fps": fps,
            "video_frame_count": n,
            "video_duration_s": (n / fps) if fps > 0 else 0.0,
            "video_width": w,
            "video_height": h,
        }
    finally:
        cap.release()


def build_row(video: Path) -> dict[str, Any] | None:
    m = _NAME_RE.match(video.name)
    if not m:
        return None
    rid = m.group("rid")
    idx = int(m.group("idx"))
    seg_tag = f"seg{m.group('idx')}"  # preserve the exact zero-padding from the file
    rec_dir = video.parent                  # .../recordings
    user_dir = rec_dir.parent               # .../<user_id>
    keylog = user_dir / "keylogs" / f"input_{rid}_{seg_tag}.msgpack"
    row: dict[str, Any] = {
        "segment_id": f"{rid}_{seg_tag}",
        "segment_idx": idx,
        "recording_id": rid,
        "video_path": str(video.resolve()),
        "keylog_path": str(keylog.resolve()),
        "keylog_exists": keylog.exists() and keylog.stat().st_size > 0,
        "user_id": user_dir.name,
        "version": user_dir.parent.name if user_dir.parent.name != "uploads" else user_dir.parent.parent.name,
    }
    row.update(probe_video(video))
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=32, help="parallel cv2 probes")
    p.add_argument("--keep-missing-keylog", action="store_true",
                   help="include segments with no keylog (collapse to ~4 frames; off by default)")
    p.add_argument("--keep-bad-video", action="store_true",
                   help="include segments whose video failed to probe (off by default)")
    p.add_argument("--limit", type=int, default=None,
                   help="Probe at most this many videos (smoke runs). Applied after --shuffle-seed.")
    p.add_argument("--shuffle-seed", type=int, default=None,
                   help="Deterministically shuffle the video list before --limit (else sorted order).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root
    videos = sorted(root.glob("uploads/**/recordings/*.mp4"))
    print(f"found {len(videos)} mp4 under {root}/uploads", file=sys.stderr)
    if not videos:
        raise SystemExit("no videos found")

    if args.shuffle_seed is not None:
        import random
        random.Random(args.shuffle_seed).shuffle(videos)
    if args.limit is not None:
        videos = videos[: args.limit]
        print(f"  limited to {len(videos)} videos (limit={args.limit}, seed={args.shuffle_seed})",
              file=sys.stderr)

    rows: list[dict[str, Any]] = []
    n_bad_name = n_no_keylog = n_bad_video = 0
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_row, v): v for v in videos}
        for fut in as_completed(futs):
            done += 1
            if done % 2000 == 0:
                print(f"  probed {done}/{len(videos)}", file=sys.stderr)
            row = fut.result()
            if row is None:
                n_bad_name += 1
                continue
            if not row.get("video_ok"):
                n_bad_video += 1
                if not args.keep_bad_video:
                    continue
            if not row.get("keylog_exists"):
                n_no_keylog += 1
                if not args.keep_missing_keylog:
                    continue
            rows.append(row)

    rows.sort(key=lambda r: (r["recording_id"], r["segment_idx"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    total_dur = sum(r.get("video_duration_s", 0.0) for r in rows)
    print(
        f"wrote {len(rows)} rows -> {args.out}\n"
        f"  skipped: bad_name={n_bad_name} bad_video={n_bad_video} no_keylog={n_no_keylog}\n"
        f"  total duration kept: {total_dur/3600:.1f} h",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
