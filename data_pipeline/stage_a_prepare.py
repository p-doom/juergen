"""Stage A: crowd-cast S3 sync → omegalax-ready chat.jsonl.

Reads raw recordings + keylogs from --source_path (the S3 sync), extracts
JPEG frames at target fps/height, parses keylog msgpacks, builds per-frame
action strings, writes per-segment chat_line.json + meta.json + per-split
chat.jsonl + the dataset manifest.

Output layout (under --output_dir):
  <output_dir>/
    manifest.json                       # required: pmanager completion marker
    {train,val,test}/
      chat.jsonl                        # concatenated per-segment lines
      <segment_id>/
        frames/frame_<N>.jpg            # frames at target fps/height
        chat_line.json                  # one structured-message record
        meta.json                       # per-segment audit counts

Action string format (event-stream, lossless to recorder):
  'NO_OP'                              all-zero frame, no key transitions
  '<dx> <dy> <scroll>'                 mouse-only frame
  '<dx> <dy> <scroll> ; +K1 -K2'       mouse + key transitions in keylog order
"""

from __future__ import annotations

import json
import multiprocessing as mp
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
from absl import app, flags
from PIL import Image

from _manifest import write_manifest

FLAGS = flags.FLAGS

# pmanager-injected:
flags.DEFINE_string("output_dir", None, "Output dataset root.", required=True)
flags.DEFINE_string(
    "source_path",
    None,
    "S3 sync root (must contain <contributor>/recordings + <contributor>/keylogs).",
    required=True,
)
# Stage-specific:
flags.DEFINE_integer("target_fps", None, "Frame extraction fps.", required=True)
flags.DEFINE_integer("target_height", None, "Frame height in px.", required=True)
flags.DEFINE_integer("jpeg_quality", None, "JPEG quality 1-100.", required=True)
flags.DEFINE_float("train_ratio", None, "", required=True)
flags.DEFINE_float("val_ratio", None, "", required=True)
flags.DEFINE_integer("seed", None, "Shuffle seed for splits.", required=True)
flags.DEFINE_integer(
    "num_workers", None, "Multiprocessing workers (0 = mp.cpu_count()).", required=True
)
flags.DEFINE_integer(
    "max_segments", None, "Cap on segments processed (0 = unlimited).", required=True
)
flags.DEFINE_float(
    "black_frame_threshold",
    5.0,
    "Mean pixel intensity (0-255) below which a frame is considered black "
    "and dropped. 0 = disable black-frame filtering.",
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECORDING_RE = re.compile(r"^recording_([0-9a-fA-F-]+)_seg(\d+)(_[a-z]+)?\.mp4$")

# rdev's macOS keycode table is incomplete; these codes appear as
# Key::Unknown(N). Patch them back to canonical names so the tokenizer sees
# stable identifiers instead of BPE-fragmenting "Unknown(N)" strings.
MACOS_UNKNOWN_NAME_BY_CODE: dict[int, str] = {
    10: "ISO_Section",
    62: "ControlRight",
    84: "Keypad2",
    86: "Keypad4",
    88: "Keypad6",
    91: "Keypad8",
    114: "Help",
    115: "Home",
    116: "PageUp",
    117: "ForwardDelete",
    119: "End",
    121: "PageDown",
}


def _resolve_ffmpeg_binary() -> tuple[str, str]:
    """Return (ffmpeg, ffprobe). Prefer system PATH; fall back to imageio_ffmpeg."""
    sys_ffmpeg = shutil.which("ffmpeg")
    sys_ffprobe = shutil.which("ffprobe")
    if sys_ffmpeg and sys_ffprobe:
        return sys_ffmpeg, sys_ffprobe
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]  # noqa: PLC0415 — lazy fallback when system ffmpeg is absent

        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError("ffmpeg not on PATH and imageio_ffmpeg unavailable") from e
    # imageio_ffmpeg only ships ffmpeg, not ffprobe; we fall back to parsing -i stderr.
    return ff, ""


_FFMPEG_BIN, _FFPROBE_BIN = _resolve_ffmpeg_binary()


# ---------------------------------------------------------------------------
# Discovery & splits
# ---------------------------------------------------------------------------


def _collect_videos(source_path: Path) -> list[Path]:
    return sorted(p for p in source_path.rglob("*.mp4") if p.is_file())


def _keylog_path_for(video_path: Path) -> Path:
    """<contrib>/recordings/recording_<sess>_seg<N><suffix>.mp4
    → <contrib>/keylogs/input_<sess>_seg<N><suffix>.msgpack

    Also supports flat layout where mp4 and msgpack live in the same directory.
    """
    m = RECORDING_RE.match(video_path.name)
    assert m is not None, f"Unexpected video filename: {video_path.name}"
    sess, seg_str, suffix = m.group(1), m.group(2), (m.group(3) or "")
    msgpack_name = f"input_{sess}_seg{int(seg_str):04d}{suffix}.msgpack"
    hierarchical = video_path.parent.parent / "keylogs" / msgpack_name
    if hierarchical.exists():
        return hierarchical
    return video_path.parent / msgpack_name


def _split_videos(
    videos: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    rng = np.random.default_rng(seed)
    shuffled = list(videos)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


# ---------------------------------------------------------------------------
# Action format
# ---------------------------------------------------------------------------


@dataclass
class FrameEvents:
    move_dx: float = 0.0
    move_dy: float = 0.0
    scroll: float = 0.0
    events: list[tuple[str, str]] = field(default_factory=list)


def _format_action(ev: FrameEvents) -> str:
    dx = round(ev.move_dx)
    dy = round(ev.move_dy)
    scroll = round(ev.scroll)
    has_keys = bool(ev.events)
    if dx == 0 and dy == 0 and scroll == 0 and not has_keys:
        return "NO_OP"
    parts = [f"{dx} {dy} {scroll}"]
    if has_keys:
        markers = [f"{sign}{key}" for sign, key in ev.events]
        parts.append(" ".join(markers))
    return " ; ".join(parts)


# ---------------------------------------------------------------------------
# Keylog parsing
# ---------------------------------------------------------------------------

_UNKNOWN_NAME_RE = re.compile(r"^Unknown\((-?\d+)\)$")


def _resolve_key_name(payload: Any) -> str | None:
    """KeyEvent payload is [internal_code, name]. The recorder serializes
    rdev::Key::Unknown(N) with internal_code = N + 1000 and name = 'Unknown(N)',
    so we recover the raw rdev keycode by parsing the name rather than the
    integer field."""
    if not isinstance(payload, list) or len(payload) < 2:
        return None
    name = str(payload[1])
    if name.startswith("Unknown("):
        m = _UNKNOWN_NAME_RE.match(name)
        if m is None:
            return None
        raw_code = int(m.group(1))
        if raw_code in MACOS_UNKNOWN_NAME_BY_CODE:
            return MACOS_UNKNOWN_NAME_BY_CODE[raw_code]
        return f"KC_{raw_code}"
    return name


def _resolve_button_name(payload: Any) -> str | None:  # noqa: PLR0911 — flat dispatch on payload variants is clearer than nested branches
    """MouseButtonEvent payload is [button, x, y]. button is 'Left'/'Right'/'Middle'
    or {'Other': n}."""
    if not isinstance(payload, list) or len(payload) < 1:
        return None
    button = payload[0]
    if isinstance(button, str):
        if button == "Left":
            return "LMB"
        if button == "Right":
            return "RMB"
        if button == "Middle":
            return "MMB"
        return f"M_{button}"
    if isinstance(button, dict):
        for k, v in button.items():
            return f"M_{k}_{v}"
    return None


@dataclass
class KeylogStats:
    n_events: int = 0
    n_keypress: int = 0
    n_keyrelease: int = 0
    n_mousepress: int = 0
    n_mouserelease: int = 0
    n_mousemove: int = 0
    n_scroll: int = 0
    n_context_changed: int = 0
    n_dangling_release: int = 0
    n_held_at_end: int = 0
    max_simultaneous_keys: int = 0
    context_app_ids: dict[str, int] = field(default_factory=dict)


def _aggregate_events(
    keylog_path: Path,
    n_frames: int,
    target_fps: int,
) -> tuple[list[FrameEvents], KeylogStats]:
    stats = KeylogStats()
    per_frame = [FrameEvents() for _ in range(n_frames)]
    if n_frames == 0 or not keylog_path.exists():
        return per_frame, stats

    raw = keylog_path.read_bytes()
    if not raw:
        return per_frame, stats
    entries = msgpack.unpackb(raw, raw=False)
    if not isinstance(entries, list):
        return per_frame, stats

    held: set[str] = set()
    for entry in entries:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        ts, ev = entry[0], entry[1]
        if not isinstance(ev, list) or len(ev) < 1:
            continue
        ev_type = ev[0]
        payload = ev[1] if len(ev) > 1 else None
        stats.n_events += 1

        try:
            ts_i = int(ts)
        except (TypeError, ValueError):
            continue
        bucket_idx = (ts_i * target_fps) // 1_000_000

        if ev_type == "ContextChanged":
            stats.n_context_changed += 1
            app_id: str | None = None
            if isinstance(payload, dict):
                app_id = str(payload.get("app_id", ""))
            elif isinstance(payload, list) and len(payload) >= 1:
                app_id = str(payload[0])
            if app_id is not None:
                stats.context_app_ids[app_id] = stats.context_app_ids.get(app_id, 0) + 1
            continue

        if bucket_idx < 0 or bucket_idx >= n_frames:
            continue

        if ev_type == "MouseMove":
            stats.n_mousemove += 1
            if isinstance(payload, list) and len(payload) >= 2:
                per_frame[bucket_idx].move_dx += float(payload[0])
                per_frame[bucket_idx].move_dy += float(payload[1])
            continue

        if ev_type == "MouseScroll":
            stats.n_scroll += 1
            if isinstance(payload, list) and len(payload) >= 2:
                dx_s, dy_s = payload[0], payload[1]
                v = dy_s if dy_s != 0 else dx_s
                per_frame[bucket_idx].scroll += float(v)
            continue

        if ev_type in ("KeyPress", "MousePress"):
            if ev_type == "KeyPress":
                stats.n_keypress += 1
                name = _resolve_key_name(payload)
            else:
                stats.n_mousepress += 1
                name = _resolve_button_name(payload)
            if name is None:
                continue
            if name not in held:
                per_frame[bucket_idx].events.append(("+", name))
                held.add(name)
                stats.max_simultaneous_keys = max(stats.max_simultaneous_keys, len(held))
            continue

        if ev_type in ("KeyRelease", "MouseRelease"):
            if ev_type == "KeyRelease":
                stats.n_keyrelease += 1
                name = _resolve_key_name(payload)
            else:
                stats.n_mouserelease += 1
                name = _resolve_button_name(payload)
            if name is None:
                continue
            if name in held:
                per_frame[bucket_idx].events.append(("-", name))
                held.remove(name)
            else:
                stats.n_dangling_release += 1
            continue

    stats.n_held_at_end = len(held)
    return per_frame, stats


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

_RES_RE = re.compile(r"Stream #\d+:\d+.*?Video:.*?(\d{2,5})x(\d{2,5})")


def _probe_resolution(video_path: Path) -> tuple[int, int]:
    if _FFPROBE_BIN:
        out = subprocess.run(
            [
                _FFPROBE_BIN,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        w, h = out.split("x")
        return int(w), int(h)
    proc = subprocess.run(
        [_FFMPEG_BIN, "-hide_banner", "-i", str(video_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    m = _RES_RE.search(proc.stderr or "")
    if m is None:
        raise RuntimeError(f"could not parse resolution from ffmpeg -i for {video_path}")
    return int(m.group(1)), int(m.group(2))


def _extract_frames(
    video_path: Path,
    out_dir: Path,
    target_fps: int,
    target_height: int,
    jpeg_quality: int,
    black_frame_threshold: float = 0.0,
) -> tuple[int, int, int, list[int] | None]:
    """Returns (n_written, out_width, n_raw, kept_indices).

    *n_raw* is the total number of frames extracted from the video before
    any filtering.  *kept_indices* is ``None`` when no black-frame
    filtering is applied (threshold <= 0) — meaning every extracted frame
    was written.  When filtering is active it lists the original
    (pre-filter) frame indices that survived, so callers can align keylog
    buckets.
    """
    in_width, in_height = _probe_resolution(video_path)
    out_width = round(target_height * in_width / in_height)
    out_width += out_width % 2  # ensure even

    cmd = [
        _FFMPEG_BIN,
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={target_fps}:round=up,scale={out_width}:{target_height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={proc.returncode}) for {video_path}: "
            f"{proc.stderr.decode(errors='replace')[:500]}"
        )
    out = proc.stdout
    frame_size = target_height * out_width * 3
    n_raw = len(out) // frame_size
    if n_raw == 0:
        return 0, out_width, 0, None
    frames = np.frombuffer(out, np.uint8).reshape(n_raw, target_height, out_width, 3)

    if black_frame_threshold > 0:
        mean_intensity = frames.mean(axis=(1, 2, 3))
        keep_mask = mean_intensity >= black_frame_threshold
        kept_indices = [int(i) for i in np.where(keep_mask)[0]]
        frames = frames[keep_mask]
    else:
        kept_indices = None

    out_dir.mkdir(parents=True, exist_ok=True)
    for fi in range(len(frames)):
        Image.fromarray(frames[fi]).save(
            out_dir / f"frame_{fi:06d}.jpg",
            format="JPEG",
            quality=jpeg_quality,
        )
    return len(frames), out_width, n_raw, kept_indices


# ---------------------------------------------------------------------------
# Per-segment worker
# ---------------------------------------------------------------------------


def _build_messages(frames_dir: Path, n_frames: int, action_strings: list[str]) -> list[dict]:
    messages: list[dict] = []
    for fi in range(n_frames):
        frame_path = str(frames_dir / f"frame_{fi:06d}.jpg")
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image", "image": frame_path}],
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": action_strings[fi]}],
            }
        )
    return messages


def _process_segment(args: dict) -> dict:
    video_path = Path(args["video_path"])
    output_root = Path(args["output_root"])
    split = args["split"]
    target_fps = args["target_fps"]
    target_height = args["target_height"]
    jpeg_quality = args["jpeg_quality"]
    black_frame_threshold = args.get("black_frame_threshold", 0.0)

    segment_id = video_path.stem
    if video_path.parent.name == "recordings":
        contributor_hash = video_path.parent.parent.name
    else:
        contributor_hash = video_path.parent.name
    segment_dir = output_root / split / segment_id
    frames_dir = segment_dir / "frames"
    summary = {
        "segment_id": segment_id,
        "contributor_hash": contributor_hash,
        "split": split,
        "video_path": str(video_path),
        "n_frames": 0,
        "skip_reason": "",
    }

    try:
        n_frames, out_width, n_raw, kept_indices = _extract_frames(
            video_path, frames_dir, target_fps, target_height, jpeg_quality,
            black_frame_threshold=black_frame_threshold,
        )
    except Exception as e:
        summary["skip_reason"] = f"frame_extraction_failed: {e}"
        return summary

    if n_frames == 0:
        summary["skip_reason"] = "no_frames"
        return summary

    keylog_path = _keylog_path_for(video_path)
    if not keylog_path.exists():
        summary["skip_reason"] = "no_keylog"
        summary["n_frames"] = n_frames
        return summary

    # Aggregate events over the *original* frame count so bucket indices
    # line up with the raw video timeline, then select only the kept frames.
    per_frame_raw, stats = _aggregate_events(keylog_path, n_raw, target_fps)

    if kept_indices is not None:
        per_frame = [per_frame_raw[i] for i in kept_indices]
        n_black_dropped = n_raw - len(kept_indices)
    else:
        per_frame = per_frame_raw
        n_black_dropped = 0

    action_strings = [_format_action(ev) for ev in per_frame]
    messages = _build_messages(frames_dir, n_frames, action_strings)
    n_no_op = sum(1 for s in action_strings if s == "NO_OP")

    meta = {
        "segment_id": segment_id,
        "contributor_hash": contributor_hash,
        "split": split,
        "video_path": str(video_path),
        "keylog_path": str(keylog_path),
        "n_frames": n_frames,
        "n_frames_before_black_filter": n_raw,
        "n_black_dropped": n_black_dropped,
        "kept_indices": kept_indices,
        "frame_height": target_height,
        "frame_width": out_width,
        "target_fps": target_fps,
        "n_no_op": n_no_op,
        "stats": {
            "n_events": stats.n_events,
            "n_keypress": stats.n_keypress,
            "n_keyrelease": stats.n_keyrelease,
            "n_mousepress": stats.n_mousepress,
            "n_mouserelease": stats.n_mouserelease,
            "n_mousemove": stats.n_mousemove,
            "n_scroll": stats.n_scroll,
            "n_context_changed": stats.n_context_changed,
            "n_dangling_release": stats.n_dangling_release,
            "n_held_at_end": stats.n_held_at_end,
            "max_simultaneous_keys": stats.max_simultaneous_keys,
            "context_app_ids": stats.context_app_ids,
        },
        "recorder_emits_context_events": stats.n_context_changed > 0,
    }
    (segment_dir / "meta.json").write_text(json.dumps(meta))

    chat_line = {
        "segment_id": segment_id,
        "contributor_hash": contributor_hash,
        "messages": messages,
    }
    (segment_dir / "chat_line.json").write_text(json.dumps(chat_line))

    summary.update(
        {
            "n_frames": n_frames,
            "n_black_dropped": n_black_dropped,
            "n_no_op": n_no_op,
            "n_context_changed": stats.n_context_changed,
            "n_held_at_end": stats.n_held_at_end,
        }
    )
    return summary


# ---------------------------------------------------------------------------
# chat.jsonl concatenation
# ---------------------------------------------------------------------------


def _concat_chat_jsonl(split_dir: Path) -> int:
    chat_path = split_dir / "chat.jsonl"
    segment_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    n = 0
    with chat_path.open("w") as out_f:
        for seg_dir in segment_dirs:
            line_path = seg_dir / "chat_line.json"
            if not line_path.exists():
                continue
            out_f.write(line_path.read_text().rstrip("\n") + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(_):
    output_dir = Path(FLAGS.output_dir)
    source_path = Path(FLAGS.source_path)

    test_ratio = 1.0 - FLAGS.train_ratio - FLAGS.val_ratio
    assert 0.0 <= test_ratio <= 1.0 + 1e-6, "ratios must be in [0, 1]"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir: {output_dir}")
    print(f"source_path: {source_path}")

    videos = _collect_videos(source_path)
    assert videos, f"No mp4 files found under {source_path}"
    print(f"Found {len(videos)} videos.")

    if FLAGS.max_segments > 0 and len(videos) > FLAGS.max_segments:
        rng = np.random.default_rng(FLAGS.seed)
        rng.shuffle(videos)
        videos = videos[: FLAGS.max_segments]
        print(f"Truncated to max_segments={FLAGS.max_segments}")

    splits = _split_videos(videos, FLAGS.train_ratio, FLAGS.val_ratio, FLAGS.seed)
    for split, vs in splits.items():
        print(f"  {split}: {len(vs)}")

    pool_args: list[dict] = []
    for split, vs in splits.items():
        (output_dir / split).mkdir(parents=True, exist_ok=True)
        for v in vs:
            pool_args.append(
                {
                    "video_path": str(v),
                    "output_root": str(output_dir),
                    "split": split,
                    "target_fps": FLAGS.target_fps,
                    "target_height": FLAGS.target_height,
                    "jpeg_quality": FLAGS.jpeg_quality,
                    "black_frame_threshold": FLAGS.black_frame_threshold,
                }
            )

    n_workers = FLAGS.num_workers if FLAGS.num_workers > 0 else mp.cpu_count()
    n_workers = min(n_workers, max(1, len(pool_args)))
    print(f"Using {n_workers} workers.")

    summaries: list[dict] = []
    with mp.Pool(processes=n_workers) as pool:
        for s in pool.imap_unordered(_process_segment, pool_args):
            summaries.append(s)
            print(
                f"[{len(summaries)}/{len(pool_args)}] "
                f"{s['split']}/{s['segment_id']} "
                f"frames={s.get('n_frames', 0)} "
                f"black_dropped={s.get('n_black_dropped', 0)} "
                f"skip={s['skip_reason']}",
                flush=True,
            )

    for split in ("train", "val", "test"):
        n = _concat_chat_jsonl(output_dir / split)
        print(f"Wrote {n} sessions to {output_dir / split / 'chat.jsonl'}")

    failed = [s for s in summaries if s["skip_reason"]]
    total_frames = sum(s.get("n_frames", 0) for s in summaries)
    total_black_dropped = sum(s.get("n_black_dropped", 0) for s in summaries)

    write_manifest(
        output_dir,
        stage="prepare",
        params={
            "target_fps": FLAGS.target_fps,
            "target_height": FLAGS.target_height,
            "jpeg_quality": FLAGS.jpeg_quality,
            "train_ratio": FLAGS.train_ratio,
            "val_ratio": FLAGS.val_ratio,
            "test_ratio": test_ratio,
            "seed": FLAGS.seed,
            "max_segments": FLAGS.max_segments,
            "black_frame_threshold": FLAGS.black_frame_threshold,
        },
        inputs={"source": str(source_path)},
        stats={
            "n_videos_found": len(videos),
            "n_segments_processed": len(summaries) - len(failed),
            "n_segments_failed": len(failed),
            "total_frames": total_frames,
            "total_black_dropped": total_black_dropped,
            "split_counts": {
                split: sum(1 for s in summaries if s["split"] == split and not s["skip_reason"])
                for split in ("train", "val", "test")
            },
            "failed_segments": [
                {
                    "segment_id": s["segment_id"],
                    "split": s["split"],
                    "skip_reason": s["skip_reason"],
                }
                for s in failed
            ],
        },
    )
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    app.run(main)
