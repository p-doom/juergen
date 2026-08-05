"""diffabs: `dx dy scroll ; +KEY -KEY`, deltas from differencing absolutes."""

from .codec import (
    CODEC,
    X11_BUTTON_CODES,
    DiffabsAction,
    DiffabsCodec,
    DiffabsError,
)

__all__ = [
    "CODEC",
    "DiffabsAction",
    "DiffabsCodec",
    "DiffabsError",
    "X11_BUTTON_CODES",
]
