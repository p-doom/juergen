"""One spelling of the image encoding and geometry a dataset contains."""

from __future__ import annotations

import io

from PIL import Image, JpegImagePlugin


def encode_jpeg_q92(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.convert("RGB").save(
        output,
        format="JPEG",
        quality=92,
        subsampling=2,
        optimize=False,
    )
    return output.getvalue()


_Q92_QUANTIZATION = Image.open(
    io.BytesIO(encode_jpeg_q92(Image.new("RGB", (8, 8))))
).quantization


def validate_jpeg_q92(payload: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(payload))
    image.load()
    if (
        image.format != "JPEG"
        or image.mode != "RGB"
        or JpegImagePlugin.get_sampling(image) != 2
        or image.quantization != _Q92_QUANTIZATION
    ):
        image.close()
        raise ValueError("image is not canonical JPEG q92")
    return image


def jpeg_q92_height_domain(height: int) -> str:
    """The image-domain identifier for a canonical height-bounded JPEG store."""
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError(f"height must be a positive integer, got {height!r}")
    return f"jpeg_q92_height_{height}"
