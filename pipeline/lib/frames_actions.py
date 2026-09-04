"""FFmpeg helpers for the Crowd-Cast master image store."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from pipeline.lib.common import ensure_dir


def _jpeg_quality_to_qscale(jpeg_quality: int) -> int:
    quality = max(1, min(100, int(jpeg_quality)))
    return max(2, min(31, round(31 - (quality / 100) * 30)))


def resolve_ffmpeg_bin(value: str | None) -> str:
    if not value:
        raise RuntimeError("ffmpeg was not found; set JUERGEN_ANNOTATION_FFMPEG_BIN")
    path = Path(value).expanduser()
    if path.is_file():
        return str(path)
    resolved = shutil.which(value)
    if resolved:
        return resolved
    raise RuntimeError(f"ffmpeg binary not found: {value}")


def extract_frames_ffmpeg(
    video_path: Path,
    output_dir: Path,
    target_fps: float,
    target_height: int,
    jpeg_quality: int,
    ffmpeg_bin: str,
) -> str:
    ensure_dir(output_dir)
    for old_frame in output_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    video_filter = f"fps=fps={target_fps}:start_time=0:round=near:eof_action=pass,scale=-2:{target_height}"
    command = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        os.environ.get("JUERGEN_ANNOTATION_FFMPEG_THREADS", "4"),
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        video_filter,
        "-q:v",
        str(_jpeg_quality_to_qscale(jpeg_quality)),
        "-start_number",
        "0",
        str(output_dir / "frame_%06d.jpg"),
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"ffmpeg failed for {video_path} with exit code {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return video_filter
