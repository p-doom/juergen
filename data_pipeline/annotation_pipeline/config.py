"""Defaults for the annotation pipeline.

The hindsight labeler is configured by env (see labeler.py): LABELER_MODEL
(default Kimi-K2.6), LABELER_BASE_URL ($AZURE_OPENAI_ENDPOINT), LABELER_API_KEY
($AZURE_OPENAI_API_KEY), LABELER_REASONING_EFFORT.
"""

from __future__ import annotations

import os
import shutil

# --- Stage 01: sampling -----------------------------------------------------
# Base rate (frames + keylog action bins). At 0.5 fps we send the VLM one frame
# every 2 s; each kept frame's action bin aggregates ~2 s of input. Idle is
# thinned by the NO_OP head/tail keep below.
DEFAULT_TARGET_FPS = 0.5
DEFAULT_TARGET_HEIGHT = 720          # training frame height (stage 01)
# q80 is what we annotated with successfully; stage 02 now feeds the VLM these
# same stored frames (no re-render), so this quality serves both training and
# annotation.
DEFAULT_JPEG_QUALITY = 80
# In each maximal run of consecutive NO_OP frames keep the first HEAD and the
# last TAIL, drop the middle — so a wait's start AND end (e.g. an agent
# finishing) stay visible without the whole idle stretch. Set both to 0 (via the
# run-time override) to drop NO_OPs entirely.
DEFAULT_NOOP_KEEP_HEAD = 1
DEFAULT_NOOP_KEEP_TAIL = 1

# --- Stage 02: VLM annotation ----------------------------------------------
# Frames fed to the labeler come straight from the stage-01 array_record (no
# re-render). At 720p a wide ~150-frame clip can overflow the model's 262K
# context (input + LABELER_MAX_TOKENS completion) — that one clip fails and is
# skipped; the rest run. Set VLM_FRAME_HEIGHT below the stored height to rescue
# it: stage 02 then downscales the stored frames in memory (still no jpegs on
# disk). See LABELER_MAX_TOKENS in labeler.py.
DEFAULT_VLM_FRAME_HEIGHT = 720       # height fed to the labeler (<= stored height)
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

# SFT system prompt: the plan-aware computer-use format contract. The first
# assistant turn is `<plan prose>\n<first action>` (stage 02b + stage 03);
# every later turn is a pure action. NOTE: hindsight_fold no longer bakes a
# system prompt at assemble time — its canonical ingest recipe injects
# prompts/desktop_action_plan.txt (the single source of truth); prefer that
# file over this constant when they drift.
SYSTEM_PROMPT = (
    "You operate a desktop computer. The first user turn shows the initial screen and the user's "
    "goal. Before acting, briefly state your plan for achieving the goal in one or two sentences, "
    "then give the first action on the same turn. On every later turn, reply with only the next "
    "action toward the goal as `<dx> <dy> <scroll>` optionally followed by ` ; +KEY -KEY` events, "
    "or `NO_OP` if no action."
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
