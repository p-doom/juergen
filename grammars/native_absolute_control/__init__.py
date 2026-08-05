"""native_absolute_control: bare-line `x y scroll ; EVENTS`, absolute pixels.

The absolute twin of `compact_raw`; the two differ only in whether the leading
integers name a position or an offset.
"""

from .codec import (
    CODEC,
    PAIRED_WITH,
    NativeAbsoluteControlAction,
    NativeAbsoluteControlCodec,
    NativeAbsoluteControlError,
)

__all__ = [
    "CODEC",
    "PAIRED_WITH",
    "NativeAbsoluteControlAction",
    "NativeAbsoluteControlCodec",
    "NativeAbsoluteControlError",
]
