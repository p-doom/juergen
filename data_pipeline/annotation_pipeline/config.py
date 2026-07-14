"""Defaults for the annotation pipeline.

The hindsight labeler is configured by env (see labeler.py): LABELER_MODEL
(default Kimi-K2.6), LABELER_BASE_URL ($AZURE_OPENAI_ENDPOINT), LABELER_API_KEY
($AZURE_OPENAI_API_KEY), LABELER_REASONING_EFFORT.
"""

from __future__ import annotations

import os
import shutil

# --- Stage 01: base modalities ---------------------------------------------
DEFAULT_BASE_FPS = 0.5
DEFAULT_TARGET_HEIGHT = 720
# q80 is what we annotated with successfully; Stage 03 feeds the VLM these
# same stored frames (no re-render), so this quality serves both training and
# annotation.
DEFAULT_JPEG_QUALITY = 80
# --- Stage 02: observation view --------------------------------------------
DEFAULT_OBSERVATION_FPS = 0.5
# In each maximal idle run keep the first HEAD and last TAIL observations.
DEFAULT_IDLE_KEEP_HEAD = 1
DEFAULT_IDLE_KEEP_TAIL = 1

# --- Stage 03: VLM annotation ----------------------------------------------
# Frames fed to the labeler come straight from the stage-01 array_record (no
# re-render). At 720p a wide ~150-frame clip can overflow the model's 262K
# context (input + LABELER_MAX_TOKENS completion) — that one clip fails and is
# skipped; the rest run. Set VLM_FRAME_HEIGHT below the stored height to rescue
# it: Stage 03 then downscales the stored frames in memory (still no JPEGs on
# disk). See LABELER_MAX_TOKENS in labeler.py.
DEFAULT_VLM_FRAME_HEIGHT = 720  # height fed to the labeler (<= stored height)


def ffmpeg_bin() -> str | None:
    configured = os.environ.get("JUERGEN_ANNOTATION_FFMPEG_BIN") or os.environ.get("FFMPEG_BIN")
    if configured:
        return configured
    path_bin = shutil.which("ffmpeg")
    if path_bin:
        return path_bin
    try:
        import imageio_ffmpeg  # noqa: PLC0415 - optional dependency

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
