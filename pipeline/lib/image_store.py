"""Helpers for JPEG-backed ArrayRecord image stores.

The training JSON keeps the usual Qwen structured-content shape, but the image
string can point at an ArrayRecord record instead of a standalone JPEG file:

    {"type": "image", "image": "ar:///abs/path/to/images.array_record#12"}

Each ArrayRecord record is the raw JPEG byte stream for one frame. This mirrors
``data_pipeline/image_store.py`` on the ``main`` branch (Alfred's stage_a store)
so the two pipelines share one URI scheme and the same training consumers.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:  # avoid importing heavy deps at module load
    import numpy as np
    from PIL import Image

ARRAYRECORD_IMAGE_URI_SCHEME = "ar"


def make_arrayrecord_image_uri(shard_path: str | Path, record_index: int) -> str:
    """Return a local-file ArrayRecord image URI for one JPEG record."""
    shard = Path(shard_path).expanduser().resolve()
    if record_index < 0:
        raise ValueError(f"record_index must be non-negative, got {record_index}")
    return f"{ARRAYRECORD_IMAGE_URI_SCHEME}://{shard.as_posix()}#{record_index}"


def parse_arrayrecord_image_uri(uri: str) -> tuple[Path, int]:
    """Parse ``ar:///abs/path/to/shard.array_record#idx``."""
    parsed = urlparse(uri)
    if parsed.scheme != ARRAYRECORD_IMAGE_URI_SCHEME:
        raise ValueError(f"not an ArrayRecord image URI: {uri!r}")
    if parsed.netloc:
        # Reserved for future named stores (ar://store/image_id). We emit local
        # shard paths, which use the empty-netloc ar:/// form.
        raise ValueError(f"unsupported named ArrayRecord image URI: {uri!r}")
    if not parsed.path or not parsed.fragment:
        raise ValueError(f"malformed ArrayRecord image URI: {uri!r}")
    try:
        record_index = int(parsed.fragment)
    except ValueError as e:
        raise ValueError(f"ArrayRecord URI fragment must be an integer: {uri!r}") from e
    if record_index < 0:
        raise ValueError(f"ArrayRecord URI record index must be non-negative: {uri!r}")
    return Path(unquote(parsed.path)), record_index


def is_arrayrecord_image_uri(value: object) -> bool:
    return isinstance(value, str) and value.startswith(f"{ARRAYRECORD_IMAGE_URI_SCHEME}://")


@lru_cache(maxsize=32)
def _reader(shard_path: str):
    """Cache one ArrayRecordReader per shard (frames of a segment read together)."""
    from array_record.python.array_record_module import ArrayRecordReader  # noqa: PLC0415

    return ArrayRecordReader(shard_path)


def read_jpeg_bytes(ref: str | Path) -> bytes:
    """Return raw JPEG bytes for an ``ar://`` URI or a plain file path."""
    if is_arrayrecord_image_uri(str(ref)):
        shard, idx = parse_arrayrecord_image_uri(str(ref))
        return _reader(str(shard)).read([idx])[0]
    return Path(ref).read_bytes()


def read_image_bgr(ref: str | Path) -> np.ndarray:
    """Decode ``ref`` (``ar://`` URI or file path) into an OpenCV BGR array."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if is_arrayrecord_image_uri(str(ref)):
        buf = np.frombuffer(read_jpeg_bytes(ref), dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"could not decode ArrayRecord frame: {ref}")
        return frame
    frame = cv2.imread(str(ref))
    if frame is None:
        raise RuntimeError(f"could not read frame: {ref}")
    return frame


def open_image_pil(ref: str | Path) -> Image.Image:
    """Open ``ref`` (``ar://`` URI or file path) as a PIL image."""
    import io  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    if is_arrayrecord_image_uri(str(ref)):
        return Image.open(io.BytesIO(read_jpeg_bytes(ref)))
    return Image.open(ref)
