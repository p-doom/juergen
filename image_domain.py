"""One spelling of which pixels a model was shown.

Two producers write it and one consumer enforces it: `datasets/convert.py` for
the rollout frames its records point at, `pipeline/stage_04_build_conversations`
for the crowd-cast master frames its conversations point at, and
`evals/harness.py`, which refuses to score a checkpoint through a domain other
than the one its dataset named. Comparing those means one string, built here.

The geometry token names the knob that bounded the frame, not the size that came
out. `max_pixels_0` (`ImageBudget` never downscales) and `height_0` (ffmpeg
`scale=null`) both read as "the source's own pixels", but one source is a guest
framebuffer and the other is whatever the crowd-cast screen recording was.
Collapsing them would turn a real train/serve mismatch into a silent match; kept
apart, the worst case is a loud refusal an arm can vouch for in writing.

This module imports nothing, deliberately: `pipeline/` runs in a venv without
`verifiers`, so it cannot reach `agent/history.py`, where `ImageBudget` lives.
"""

from __future__ import annotations

IMAGE_GEOMETRIES = ("max_pixels", "height")


def image_domain(*, media: str, quality: int, geometry: str, extent: int) -> str:
    """`<encoding>_<geometry>_<extent>`, e.g. `jpeg_q80_height_720`.

    `quality` is absent from the PNG form: PNG is lossless, so carrying it would
    refuse a lossless arm against a lossless dataset.
    """
    if geometry not in IMAGE_GEOMETRIES:
        raise ValueError(
            f"image geometry must be one of {IMAGE_GEOMETRIES}, got {geometry!r}"
        )
    encoding = "png" if media == "png" else f"jpeg_q{quality}"
    return f"{encoding}_{geometry}_{extent}"
