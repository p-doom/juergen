"""Defaults for the annotation pipeline.

The hindsight labeler is configured by env (see labeler.py): LABELER_MODEL
(default Kimi-K2.6), LABELER_BASE_URL ($AZURE_OPENAI_ENDPOINT), LABELER_API_KEY
($AZURE_OPENAI_API_KEY), LABELER_REASONING_EFFORT.
"""

from __future__ import annotations

import os
import shutil

# Stage 01: sampling.
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
# last TAIL, drop the middle — so a wait's start and end (e.g. an agent
# finishing) stay visible without the whole idle stretch. Set both to 0 (via the
# run-time override) to drop NO_OPs entirely.
DEFAULT_NOOP_KEEP_HEAD = 1
DEFAULT_NOOP_KEEP_TAIL = 1
# High-level NO_OP knob for the sampler (stage 01b), overriding HEAD/TAIL when set:
#   "none" -> keep 0 NO_OP frames (drop every idle frame)
#   "ends" -> keep the first and last frame of each idle run (== HEAD/TAIL 1/1)
#   "all"  -> keep every NO_OP frame (no thinning)
# Unset (None) falls back to DEFAULT_NOOP_KEEP_HEAD/TAIL (legacy interface).
NOOP_MODES = ("none", "ends", "all")

# Stage 03: idle filtering (duration-based, master-fps-agnostic).
# The filter judges idleness in seconds so a 4 fps and a 15 fps master behave
# identically: the interior of any inactive run longer than MIN_DURATION_S is
# dropped, keeping KEEP_HEAD_S / KEEP_TAIL_S at each end (a wait's start and
# end stay visible).
#
# Idleness is judged with the "rounded" predicate by default: a judgment bin is
# active iff its formatted action is non-NO_OP (deltas round to nonzero, or the
# bin carries deduped key events), per 2 s judgment bin (= 1/DEFAULT_TARGET_FPS),
# runs > 4 s (> 2 bins) thinned, 2 s (1 bin) kept at each end. Rounding is what
# makes it fps-dependent: sub-round drift seconds read as idle and are dropped.
# Set IDLE_ACTIVITY "raw" (any nonzero event counts, per master tick) to retain
# them and make the judgment fps-agnostic.
DEFAULT_IDLE_MIN_DURATION_S = 4.0
DEFAULT_IDLE_KEEP_HEAD_S = 2.0
DEFAULT_IDLE_KEEP_TAIL_S = 2.0
DEFAULT_IDLE_JUDGMENT_BIN_S = 1.0 / DEFAULT_TARGET_FPS  # 2 s
IDLE_ACTIVITIES = ("raw", "rounded")
DEFAULT_IDLE_ACTIVITY = "rounded"  # "raw" = fps-agnostic
assert DEFAULT_IDLE_ACTIVITY in IDLE_ACTIVITIES

# Black-frame filtering.
# Detection (per-frame luma metrics) is computed once in stage 01a and written to
# the frame manifest; the drop decision runs in the sampler (01b), so these
# thresholds are tunable without re-decoding. A frame is dropped if its mean luma
# is at/below LUMA_MAX *or* the near-black pixel fraction is at/above DARK_FRAC_MIN.
DEFAULT_DROP_BLACK_FRAMES = True
DEFAULT_BLACK_LUMA_MAX = 6.0         # mean luma (0-255) at/below this -> black
DEFAULT_BLACK_DARK_FRAC_MIN = 0.999  # fraction of pixels below BLACK_DARK_CUTOFF
BLACK_DARK_CUTOFF = 16               # a pixel is "near-black" if its luma < this

# Stage 02: VLM annotation.
# Frames fed to the labeler come straight from the stage-01 array_record (no
# re-render). At 720p a wide ~150-frame clip can overflow the model's 262K
# context (input + LABELER_MAX_TOKENS completion) — that one clip fails and is
# skipped; the rest run. Set VLM_FRAME_HEIGHT below the stored height to rescue
# it: stage 02 then downscales the stored frames in memory (still no jpegs on
# disk). See LABELER_MAX_TOKENS in labeler.py.
DEFAULT_VLM_FRAME_HEIGHT = 720       # height fed to the labeler (<= stored height)
DEFAULT_NAME_IMAGE_MAX = 24          # frames per label / verify request

# Stage 04/05: token accounting.
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
    except Exception:
        return None
