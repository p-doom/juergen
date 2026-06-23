"""Defaults for the annotation pipeline.

The hindsight labeler is configured by env (see labeler.py): LABELER_MODEL
(default gpt-5.5), LABELER_BASE_URL ($AZURE_OPENAI_ENDPOINT), LABELER_API_KEY
($AZURE_OPENAI_API_KEY), LABELER_REASONING_EFFORT.
"""

from __future__ import annotations

import os
import shutil

# --- Stage 01: sampling -----------------------------------------------------
# Base rate (frames + keylog action bins). At 1 fps each kept frame's action
# aggregates ~1 s of input; idle is thinned by the NO_OP head/tail keep below.
DEFAULT_TARGET_FPS = 1
DEFAULT_TARGET_HEIGHT = 720          # training frame height (stage 01)
DEFAULT_JPEG_QUALITY = 95
# In each maximal run of consecutive NO_OP frames keep the first HEAD and the
# last TAIL, drop the middle — so a wait's start AND end (e.g. an agent
# finishing) stay visible without the whole idle stretch.
DEFAULT_NOOP_KEEP_HEAD = 3
DEFAULT_NOOP_KEEP_TAIL = 3

# --- Stage 02: VLM annotation ----------------------------------------------
DEFAULT_VLM_FRAME_HEIGHT = 720       # height of frames rendered for the labeler
DEFAULT_NAME_IMAGE_MAX = 24          # frames per label / verify request

# --- Stage 04/05: token accounting -----------------------------------------
DEFAULT_TRAINEE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_TOKEN_OVERHEAD = 180
BUCKET_LIMITS = {
    "8k": 8_192,
    "16k": 16_384,
    "32k": 32_768,
    "64k": 65_536,
    "128k": 131_072,
    "256k": 262_144,
}

# SFT system prompt: the computer-use agent's action format.
SYSTEM_PROMPT = (
    "You operate a desktop computer. The first user turn shows the initial "
    "screen and the user's goal; subsequent user turns show the current screen. "
    "Reply with the next action toward that goal as `<dx> <dy> <scroll>` "
    "optionally followed by ` ; +KEY -KEY` events, or `NO_OP` if no action."
)


def ffmpeg_bin() -> str | None:
    configured = os.environ.get("JUERGEN_ANNOTATION_FFMPEG_BIN") or os.environ.get("FFMPEG_BIN")
    if configured:
        return configured
    path_bin = shutil.which("ffmpeg")
    if path_bin:
        return path_bin
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional dependency, only for binary discovery
        return None
