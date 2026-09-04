"""One spelling of the image encoding and geometry a dataset contains."""

from __future__ import annotations

import io

from PIL import Image, JpegImagePlugin

IMAGE_GEOMETRIES = ("max_pixels", "height")


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
