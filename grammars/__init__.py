"""Action grammars discovered from peer directories."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

_FROM_SUPPORT = (
    "CONTROL_SPEC",
    "CONTROL_TOKEN",
    "Control",
    "NoAction",
    "split_control",
)


def __getattr__(name: str) -> Any:
    if name in _FROM_SUPPORT:
        return getattr(import_module(f"{__name__}._support"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def available() -> tuple[str, ...]:
    root = Path(__file__).parent
    return tuple(
        child.name
        for child in sorted(root.iterdir())
        if child.is_dir()
        and not child.name.startswith(("_", "."))
        and (child / "codec.py").is_file()
    )


_CACHE: dict[str, Any] = {}


def load(name: str) -> Any:
    """Load one peer grammar's module-level ``CODEC``."""
    if name in _CACHE:
        return _CACHE[name]
    names = available()
    if name not in names:
        raise KeyError(f"unknown grammar {name!r} (available: {list(names)})")
    codec = import_module(f"{__name__}.{name}.codec").CODEC
    _CACHE[name] = codec
    return codec


def codecs() -> dict[str, Any]:
    return {name: load(name) for name in available()}


def describe(name: str) -> str:
    return load(name).describe()


__all__ = [
    "CONTROL_SPEC",
    "CONTROL_TOKEN",
    "Control",
    "NoAction",
    "available",
    "codecs",
    "describe",
    "load",
    "split_control",
]
