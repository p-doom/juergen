"""One synthetic crowd-cast segment: an uploads tree the stages can be run over.

A generated 20 s / 4 fps clip plus a hand-written raw keylog, laid out exactly
as ``crowd-cast-2026-06-18`` lays out an upload
(``uploads/<version>/<user>/{recordings,keylogs}/``), so stage 00 discovers it,
stage 02 threads it, and stages 03/04 join it by ``segment_id``.

The event stream is not a recording; every event is placed at a master tick
chosen to land on one side of a contract boundary. On a 4 fps master with a
1 fps training view the label windows are

    W0 [0,4)  W1 [4,8)  W2 [8,12)  W3 [12,20)  W4 [20,24)
    W5 [24,28)  W6 [28,32)  W7 [32,36)  W8 [36,40)

(the slot at tick 16 is masked black, so W3 spans two slots; every window past
W8 is NO_OP -- the clip's second half is deliberately inactive, so the default
idle knobs have a run to thin) and the stream
carries one instance of every disposition the dead-zone label policy can reach:

  * a press/release pair inside one window (``LMB``);
  * a pair spanning a WINDOW boundary (``KeyA``: press in W2, release in W3) —
    the split/carry case, whose two halves land in two consecutive assistant
    turns;
  * an autorepeat press of an already-held key (``KeyA`` again in W2), which
    must be deduped rather than emitted as a second press with one release;
  * a release with no press at all (``KeyZ``), the dangling-release case;
  * a pair spanning a DEAD-ZONE boundary in each direction: ``KeyR`` pressed
    visibly and released inside the black span (release clamped back), ``KeyB``
    pressed inside it and released visibly (press clamped forward);
  * a pair entirely inside the black span (``KeyM``), which is dropped whole;
  * a press inside the black span that is never released (``KeyQ``), which is
    dropped rather than clamped forward — clamping it would emit a press with
    no matching release;
  * a press in the last window that is never released (``KeyC``), the
    held-at-end case, which IS emitted (it is a real observed transition);
  * a mouse move inside the black span, and one past the end of video
    coverage, both of which are discarded from labels;
  * a balanced ``ShiftLeft``-enclosed typing run in W6 (``Hi``), which
    the canonical formatter emits as key transitions.

The master frame store is packed from the SAME arrays the mp4 is written from,
through stage 01's own ``pack_master_arrayrecord``, rather than by decoding the
mp4: stage 01 shells out to an ffmpeg binary, which the data-pipeline test venv
does not carry (``imageio-ffmpeg`` is declared in ``data_pipeline/pyproject.toml``
but absent from the gate venv, and there is no ffmpeg on PATH). Feeding the
packer the source arrays is a lossless decode, and it makes the black span
exactly black instead of codec-dependent. The mp4 is still real and is what
stage 00 probes.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import cv2
import msgpack
import numpy as np
from PIL import Image

from pipeline.lib import config
from pipeline.stage_01_master_frames import (
    aggregate_summary,
    pack_master_arrayrecord,
    write_index_jsonl,
    write_summary_and_manifest,
)

VERSION = "v0.0.1-test"
USER_ID = "user_synthetic"
RECORDING_ID = "0000synthetic-0000-0000-0000-00000000000c"
SEGMENT_TAG = "seg0000"
SEGMENT_ID = f"{RECORDING_ID}_{SEGMENT_TAG}"

FRAME_W = 64
FRAME_H = 48
VIDEO_FPS = 4.0
MASTER_FPS = 4.0
N_FRAMES = 80
TRAIN_FPS = 1.0
STRIDE = int(MASTER_FPS / TRAIN_FPS)

#: Half-open master-tick span of the black frames (== video frames, 1:1 here).
BLACK_SPAN = (16, 20)

#: The windows a 1 fps view has over the ACTIVE half, given the masked slot at 16.
#: Every later window is NO_OP.
EVENT_WINDOWS = [
    (0, 4),
    (4, 8),
    (8, 12),
    (12, 20),
    (20, 24),
    (24, 28),
    (28, 32),
    (32, 36),
    (36, 40),
]

#: (seconds on the recorder clock, event type, payload). Times are chosen so
#: ``int(t * MASTER_FPS)`` is the intended tick with no float ambiguity.
EVENTS: list[tuple[float, str, Any]] = [
    (0.10, "MouseMove", [5.0, 0.0]),  # tick 0  -> W0
    (0.50, "MouseMove", [4.0, 3.0]),  # tick 2  -> W0
    (1.10, "MousePress", ["Left"]),  # tick 4  -> W1
    (1.40, "MouseRelease", ["Left"]),  # tick 5  -> W1
    (2.10, "KeyPress", [0, "KeyA"]),  # tick 8  -> W2
    (2.50, "KeyPress", [0, "KeyA"]),  # tick 10 -> autorepeat, deduped
    (2.90, "MouseMove", [2.0, -1.0]),  # tick 11 -> W2
    (3.20, "KeyRelease", [0, "KeyA"]),  # tick 12 -> W3 (spans W2/W3)
    (3.60, "KeyPress", [0, "KeyR"]),  # tick 14 -> W3
    (4.20, "MouseMove", [7.0, 7.0]),  # tick 16 -> black, discarded
    (4.30, "KeyPress", [0, "KeyM"]),  # tick 17 -> black
    (4.40, "KeyRelease", [0, "KeyR"]),  # tick 17 -> black, clamped back
    (4.50, "KeyPress", [0, "KeyB"]),  # tick 18 -> black, clamped forward
    (4.60, "KeyPress", [0, "KeyQ"]),  # tick 18 -> black, never released
    (4.70, "KeyRelease", [0, "KeyM"]),  # tick 18 -> black, pair dropped
    (5.30, "KeyRelease", [0, "KeyB"]),  # tick 21 -> W4
    (5.60, "MouseMove", [-3.0, 2.0]),  # tick 22 -> W4
    (6.10, "KeyRelease", [0, "KeyZ"]),  # tick 24 -> dangling release
    (6.50, "MouseScroll", [0.0, -3.0]),  # tick 26 -> W5
    (7.10, "KeyPress", [0, "ShiftLeft"]),  # tick 28 -> W6, typing run
    (7.15, "KeyPress", [0, "KeyH"]),  # tick 28
    (7.20, "KeyRelease", [0, "KeyH"]),  # tick 28
    (7.25, "KeyRelease", [0, "ShiftLeft"]),  # tick 29
    (7.30, "KeyPress", [0, "KeyI"]),  # tick 29
    (7.35, "KeyRelease", [0, "KeyI"]),  # tick 29
    (8.10, "MouseMove", [1.0, 1.0]),  # tick 32 -> W7
    (9.10, "KeyPress", [0, "KeyC"]),  # tick 36 -> W8, held at end
    (20.50, "MouseMove", [9.0, 9.0]),  # tick 82 -> past coverage
]

#: Text typed during W6.
TYPED_TEXT = "Hi"


def frames() -> list[np.ndarray]:
    """The segment's source frames, RGB uint8. ``BLACK_SPAN`` is exactly zero."""
    out: list[np.ndarray] = []
    black_start, black_end = BLACK_SPAN
    for i in range(N_FRAMES):
        img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        if not black_start <= i < black_end:
            img[:, :] = (160, 90, 40)
            x = (i * 3) % (FRAME_W - 8)
            img[10:20, x : x + 8] = 255
        out.append(img)
    return out


def _write_video(path: Path, source: list[np.ndarray]) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (FRAME_W, FRAME_H)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cv2 could not open an mp4 writer at {path}")
    try:
        for img in source:
            writer.write(img[:, :, ::-1])  # cv2 wants BGR
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"cv2 wrote no mp4 at {path}")


def keylog_entries() -> list[list[Any]]:
    """``EVENTS`` in the on-disk keylog shape: ``[ts_us, [type, payload]]``."""
    return [[round(t * 1e6), [kind, payload]] for t, kind, payload in EVENTS]


def build_uploads_tree(root: Path) -> dict[str, Any]:
    """Write the mp4 + raw keylog under a crowd-cast uploads layout."""
    user_dir = root / "uploads" / VERSION / USER_ID
    rec_dir = user_dir / "recordings"
    key_dir = user_dir / "keylogs"
    rec_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)

    video = rec_dir / f"recording_{RECORDING_ID}_{SEGMENT_TAG}.mp4"
    keylog = key_dir / f"input_{RECORDING_ID}_{SEGMENT_TAG}.msgpack"
    source = frames()
    _write_video(video, source)
    keylog.write_bytes(msgpack.packb(keylog_entries(), use_bin_type=True))
    return {
        "dataset_root": root,
        "video_path": video,
        "keylog_path": keylog,
        "frames": source,
        "segment_id": SEGMENT_ID,
        "recording_id": RECORDING_ID,
    }


def build_master_store(out_dir: Path, clip_row: dict[str, Any], source: list[np.ndarray]) -> Path:
    """Pack a stage-01 frames-master artifact from the clip's source frames.

    Stage 01's own packer, index writer and summary writer; only the ffmpeg
    decode is replaced (see the module docstring). Returns ``out_dir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    segment_frame_dir = out_dir / "frames" / SEGMENT_ID
    segment_frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    for i, img in enumerate(source):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=config.DEFAULT_JPEG_QUALITY)
        path = segment_frame_dir / f"frame_{i:06d}.jpg"
        path.write_bytes(buf.getvalue())
        frame_paths.append(path)

    packed = pack_master_arrayrecord(
        frame_paths,
        segment_frame_dir,
        master_fps=MASTER_FPS,
        video_fps=float(clip_row["video_fps"]),
        video_frame_count=int(clip_row["video_frame_count"]),
    )
    index_row = {
        "segment_id": SEGMENT_ID,
        "recording_id": clip_row["recording_id"],
        "segment_idx": clip_row["segment_idx"],
        "master_fps": MASTER_FPS,
        "target_height": FRAME_H,
        "jpeg_quality": config.DEFAULT_JPEG_QUALITY,
        "video_duration_s": clip_row["video_duration_s"],
        "video_fps": clip_row["video_fps"],
        "status": "ok",
        "num_records": packed["num_records"],
        "shard_path": packed["shard_path"],
        "frame_manifest": packed["manifest_path"],
        "total_jpeg_bytes": packed["total_jpeg_bytes"],
    }
    write_index_jsonl(out_dir / "segment_index.jsonl", [index_row])
    write_summary_and_manifest(
        out_dir,
        aggregate_summary(
            [index_row],
            master_fps=MASTER_FPS,
            target_height=FRAME_H,
            jpeg_quality=config.DEFAULT_JPEG_QUALITY,
            ffmpeg_bin=None,
            source_clips_manifest=None,
        ),
    )
    return out_dir


def write_goals(
    goals_dir: Path, rows: list[dict[str, Any]], *, master_store_id: str, filter_id: str
) -> Path:
    """A minimal stage-03b goals artifact: goals.jsonl + the join manifest."""
    goals_dir.mkdir(parents=True, exist_ok=True)
    with (goals_dir / "goals.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (goals_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "crowdcast_describe_extract_goals",
                "schema_version": 1,
                "goals": "goals.jsonl",
                "method": "describe_extract",
                "input_kind": "frames",
                "master_store_id": master_store_id,
                "filter_id": filter_id,
            },
            indent=2,
        )
        + "\n"
    )
    return goals_dir


def goal_row(goal_id: str, start: int, end: int, instruction: str, **extra: Any) -> dict[str, Any]:
    """One goals.jsonl row satisfying ``pipeline.lib.goals.REQUIRED_GOAL_KEYS``."""
    return {
        "goal_id": goal_id,
        "segment_id": SEGMENT_ID,
        "recording_id": RECORDING_ID,
        "start_master_idx": start,
        "end_master_idx": end,
        "instruction": instruction,
        "instruction_variants": [f"please {instruction}", f"could you {instruction}"],
        "anchor": instruction,
        "grounding": "Synthetic user action.",
        "method": "describe_extract",
        "model": "test-model",
        "prompt_pack_sha": "0" * 16,
        **extra,
    }
