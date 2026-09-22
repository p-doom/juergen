"""The two action grammars used by the training streams."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ._support import split_control

_MODULES = {
    "deltatype_v2": "grammars.deltatype_v2.codec",
    "ordered_events_v3_relative_1000_grid_v1": (
        "grammars.ordered_events_v3_relative_1000_grid_v1.codec"
    ),
}


def available() -> tuple[str, ...]:
    return tuple(_MODULES)


_CACHE: dict[str, Any] = {}


def load(name: str) -> Any:
    """Load one registered grammar's module-level ``CODEC``."""
    if name in _CACHE:
        return _CACHE[name]
    try:
        module = _MODULES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown grammar {name!r} (available: {list(_MODULES)})"
        ) from exc
    codec = import_module(module).CODEC
    _CACHE[name] = codec
    return codec


def describe(name: str) -> str:
    return load(name).describe()


__all__ = [
    "available",
    "describe",
    "load",
    "split_control",
]
