"""One spelling of the image encoding and geometry a dataset contains."""

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
