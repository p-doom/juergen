"""Image-domain identifiers written into training artifacts.

The desktop observation contract is a fixed identity. Crowd-cast has a separate
generic recording identity because its source is not established as desktop
observations; stage 06 refuses to mix it into desktop SFT records.

The legacy generic geometry token names the knob that bounded a crowd-cast frame,
not the size that came out. It remains only to identify those artifacts; no
generic identity can satisfy the fixed desktop contract.

This module imports nothing, deliberately: `pipeline/` runs in a venv without
`verifiers`, so it cannot reach `agent/history.py`, where `ImageBudget` lives.
"""

from __future__ import annotations

OSWORLD_CURSOR_JPEG_DOMAIN = "osworld_cursor_jpeg_q85_420_1920x1080_v1"

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
