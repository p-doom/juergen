"""Build the canonical manifest for a complete Crowd-Cast uploads tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# recording_<rid>_seg<NNNN>.mp4  ->  rid, NNNN
_NAME_RE = re.compile(r"^recording_(?P<rid>.+)_seg(?P<idx>\d{4})\.mp4$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, Any]:
    """fps/frame-count/duration/dims via OpenCV. video_ok=False on any failure."""
    import cv2

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


def build_row(video: Path) -> dict[str, Any]:
    m = _NAME_RE.match(video.name)
    if not m:
        raise ValueError(f"invalid Crowd-Cast video name: {video.name}")
    rid = m.group("rid")
    idx = int(m.group("idx"))
    seg_tag = f"seg{m.group('idx')}"
    rec_dir = video.parent
    user_dir = rec_dir.parent
    keylog = user_dir / "keylogs" / f"input_{rid}_{seg_tag}.msgpack"
    if not keylog.is_file() or keylog.stat().st_size == 0:
        raise FileNotFoundError(f"Crowd-Cast keylog is missing or empty: {keylog}")
    video_info = probe_video(video)
    if not video_info["video_ok"]:
        raise ValueError(f"Crowd-Cast video is not decodable: {video}")
    row: dict[str, Any] = {
        "segment_id": f"{rid}_{seg_tag}",
        "segment_idx": idx,
        "recording_id": rid,
        "video_path": str(video.resolve()),
        "keylog_path": str(keylog.resolve()),
        "video_sha256": _sha256(video),
        "keylog_sha256": _sha256(keylog),
        "user_id": user_dir.name,
        "version": user_dir.parent.name,
    }
    row.update(video_info)
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=32, help="parallel cv2 probes")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out.parent / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.out.name != "clips_manifest.jsonl":
        raise SystemExit("--out must end in clips_manifest.jsonl")
    root = args.dataset_root.resolve()
    videos = sorted(root.glob("uploads/*/*/recordings/*.mp4"))
    print(f"found {len(videos)} mp4 under {root}/uploads", file=sys.stderr)
    if not videos:
        raise SystemExit("no videos found")
    expected_keylogs: set[Path] = set()
    for video in videos:
        match = _NAME_RE.fullmatch(video.name)
        if match is None:
            raise ValueError(f"invalid Crowd-Cast video name: {video.name}")
        expected_keylogs.add(
            video.parent.parent
            / "keylogs"
            / f"input_{match.group('rid')}_seg{match.group('idx')}.msgpack"
        )
    observed_keylogs = set(root.glob("uploads/*/*/keylogs/*.msgpack"))
    if observed_keylogs != expected_keylogs:
        raise ValueError(
            "Crowd-Cast keylog inventory does not match the video inventory: "
            f"missing={sorted(map(str, expected_keylogs - observed_keylogs))}, "
            f"orphan={sorted(map(str, observed_keylogs - expected_keylogs))}"
        )

    rows: list[dict[str, Any]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_row, v): v for v in videos}
        for fut in as_completed(futs):
            done += 1
            if done % 2000 == 0:
                print(f"  probed {done}/{len(videos)}", file=sys.stderr)
            rows.append(fut.result())

    rows.sort(key=lambda r: (r["recording_id"], r["segment_idx"]))
    segment_ids = [row["segment_id"] for row in rows]
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("Crowd-Cast uploads contain duplicate segment IDs")
    temporary = args.out.with_suffix(".tmp")
    with temporary.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    temporary.replace(args.out)

    clips_sha256 = _sha256(args.out)
    manifest = {
        "artifact_type": "crowdcast_source_clips",
        "schema_version": 1,
        "clips_file": "clips_manifest.jsonl",
        "clips_sha256": clips_sha256,
        "source_root": str(root),
        "n_segments": len(rows),
        "n_recordings": len({row["recording_id"] for row in rows}),
    }
    temporary_manifest = args.out.parent / ".manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)

    total_dur = sum(r.get("video_duration_s", 0.0) for r in rows)
    print(
        f"wrote {len(rows)} rows -> {args.out}\n"
        f"  total duration: {total_dur / 3600:.1f} h",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
