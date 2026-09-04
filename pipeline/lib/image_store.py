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
from urllib.parse import unquote, urlparse

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


@lru_cache(maxsize=32)
def _reader(shard_path: str):
    """Cache one ArrayRecordReader per shard (frames of a segment read together)."""
    from array_record.python.array_record_module import ArrayRecordReader

    return ArrayRecordReader(shard_path)


def read_jpeg_bytes(uri: str) -> bytes:
    """Read one JPEG from the canonical ArrayRecord URI domain."""
    shard, index = parse_arrayrecord_image_uri(uri)
    return _reader(str(shard)).read([index])[0]
