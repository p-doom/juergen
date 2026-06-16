"""Defaults for the v3 Crowd-Cast trajectory pipeline."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
RAW_DATA_ROOT = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom/crowd-cast/"
    "crowd-cast-2026-05-19"
)
# All pipeline output (frame cache + runs/SFT samples) lives next to the raw
# data under processed/, not in the code tree. Override with V3_PROCESSED_ROOT.
PROCESSED_ROOT = Path(
    os.environ.get("V3_PROCESSED_ROOT") or (RAW_DATA_ROOT / "processed")
)

PILOT_VERSION = "0.1.0"
PILOT_USER_ID = "9731b6e46467bf9bca177edaa889ef11d89393e94f2b5bbdbde9789162082d42"
PILOT_RECORDING_ID = "ed57c1ff-7116-4a53-a631-9c1d1f59fc4d"
PILOT_SEGMENT_START = 4
PILOT_SEGMENT_END = 10

DEFAULT_TARGET_FPS = 2
# Stage 01/03/04 training frames at 720p: 540p left dense desktop UI text
# (file tree, terminal, chat) unreadable, so the trainee couldn't learn from it.
# 720p is the source res; downsample later (re-extract, or cap the trainee
# processor's max_pixels at train time) if a smaller context is needed.
DEFAULT_TARGET_HEIGHT = 720
# Stage 02 renders separate (timestamp-overlaid) annotation frames for the VLM,
# also at 720p so the labeler reads the same detail the trainee will.
DEFAULT_VLM_FRAME_HEIGHT = 720
DEFAULT_JPEG_QUALITY = 95
DEFAULT_BLACK_FRAME_THRESHOLD = 5.0
DEFAULT_ACTION_WINDOW_MS = 500
DEFAULT_STAGE01_MAX_NOOP_RUN = 2

# Annotator: Qwen3.6-27B (BF16) served locally via sglang. This is the single
# validated configuration (see README). 40 images/request caps the pass-A
# vision-encoder activation spike that OOM'd at 50 on 720p windows.
DEFAULT_FRAME_IMAGE_MAX = 40
DEFAULT_VLM_MODEL = "Qwen/Qwen3.6-27B"
# Reasoning is disabled: the verification pass (stage 02 pass C) supersedes the
# self-rated confidence that thinking mode improves, and thinking costs ~5x
# wall-clock plus an empty-content failure mode.
DEFAULT_ENABLE_THINKING = False
DEFAULT_VLM_MAX_TOKENS = 4096
DEFAULT_VERIFY_MAX_TOKENS = 1024

# Trainee model whose tokenizer/processor define exact stage-04 token buckets.
# Stage 04 counts in the data_pipeline uv env via the vendored qwen3_encoding.
# No omegalax dependency.
DEFAULT_TRAINEE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

# Stage 02 pass A (segmentation): one sparse-sampled request covers a window so
# the VLM sees whole task arcs. 90s windows give finer, more atomic segments
# than wider ones, with no oversize SFT samples.
DEFAULT_SEGMENT_WINDOW_S = 90.0
DEFAULT_SEGMENT_OVERLAP_S = 30.0
# Stage 02 pass B (instruction naming): fewer, larger frames per candidate.
DEFAULT_NAME_IMAGE_MAX = 24
# Max-width caps are disabled by default so the VLM gets the configured 720p
# frame height. Set these explicitly when endpoint cost/latency matters more.
DEFAULT_VLM_IMAGE_MAX_WIDTH = 0
DEFAULT_SEGMENT_IMAGE_WIDTH = DEFAULT_VLM_IMAGE_MAX_WIDTH
DEFAULT_NAME_IMAGE_WIDTH = DEFAULT_VLM_IMAGE_MAX_WIDTH

# No default tokens-per-image: it depends on the trainee model's vision
# processor. Stage 04 counts exactly with the trainee tokenizer/processor.
DEFAULT_TOKEN_OVERHEAD = 180
BUCKET_LIMITS = {
    "8k": 8_192,
    "16k": 16_384,
    "32k": 32_768,
    "64k": 65_536,
    "128k": 131_072,
    "256k": 262_144,
}


SYSTEM_PROMPT = (
    "You operate a desktop computer. The first user turn shows the initial "
    "screen and the user's goal; subsequent user turns show the current screen. "
    "Reply with the next action toward that goal as `<dx> <dy> <scroll>` "
    "optionally followed by ` ; +KEY -KEY` events, or `NO_OP` if no action."
)


# Local sglang server (see slurm/run_pipeline.sbatch). Override with env.
DEFAULT_VLM_BASE_URL = "http://localhost:8011/v1"


def vlm_base_url() -> str:
    return os.environ.get("V3_VLM_BASE_URL") or DEFAULT_VLM_BASE_URL


def vlm_api_key() -> str:
    # sglang ignores the key; any non-empty string satisfies the OpenAI client.
    return os.environ.get("V3_VLM_API_KEY") or "local-sglang"


def vlm_model() -> str:
    return os.environ.get("V3_VLM_MODEL") or DEFAULT_VLM_MODEL


def ffmpeg_bin() -> str | None:
    configured = os.environ.get("V3_FFMPEG_BIN") or os.environ.get("FFMPEG_BIN")
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
