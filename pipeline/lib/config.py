"""Crowd-Cast image and filter contract."""

from __future__ import annotations

import os
import shutil

DEFAULT_TARGET_HEIGHT = 720
DEFAULT_JPEG_QUALITY = 92
DEFAULT_IDLE_MIN_DURATION_S = 4.0
DEFAULT_IDLE_KEEP_HEAD_S = 2.0
DEFAULT_IDLE_KEEP_TAIL_S = 2.0
DEFAULT_IDLE_JUDGMENT_BIN_S = 2.0
DEFAULT_BLACK_LUMA_MAX = 6.0
DEFAULT_BLACK_DARK_FRAC_MIN = 0.999
BLACK_DARK_CUTOFF = 16


def ffmpeg_bin() -> str | None:
    return os.environ.get("JUERGEN_ANNOTATION_FFMPEG_BIN") or shutil.which("ffmpeg")
