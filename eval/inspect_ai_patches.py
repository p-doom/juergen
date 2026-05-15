"""In-process monkey-patches for inspect-ai bugs we hit during long-response evals.

Import this module BEFORE any code that imports inspect_ai (i.e. at the top of
``ifeval.py``, ``roundtrip_ifeval.py``, etc.). The patches mutate inspect_ai's
module objects in the current process only, are idempotent (safe to import
multiple times), and survive ``uv sync`` venv rebuilds because they live in
the eval repo's source tree, not inside the installed inspect_ai package.

Two patches:

1. ``inspect_ai._util.rich.format_traceback`` — the rich+pygments rendering
   path is pathologically slow on the tracebacks inspect_ai emits when a
   generation raises (we've seen multi-hour hangs syntax-highlighting a single
   error). Replace with the plain-text variant; the error is still surfaced
   in the inspect log, just without ANSI colors. ``truncate_traceback`` does
   the actual text formatting and is fast.

2. ``inspect_ai.hooks._hooks.get_all_hooks`` — when no third-party hooks are
   registered (typical) and the registry is empty, ``registry_find`` falls
   through to ``ensure_entry_points()`` which re-walks every installed
   package's entry_points.txt on every sample event. With ~200 packages in
   the eval venv and 541xN events per IFEval, this dominates wall-clock.
   Cache the empty list on the first call so subsequent emissions are O(1).

Loud-fail assertions: if inspect_ai's internal layout changes (function
renamed or moved), the assertions trigger at import time so we don't silently
regress to the slow paths.
"""

from __future__ import annotations

import inspect_ai._util.rich as _rich
from inspect_ai.hooks import _hooks

_PATCH_MARKER = "_omegalax_eval_patched"


def _fast_format_traceback(exc_type, exc_value, exc_traceback):
    """Drop-in replacement for ``inspect_ai._util.rich.format_traceback`` that
    skips the rich+pygments render. Returns ``(plain_text, plain_text)`` so
    callers expecting ``(text, ansi)`` still get two values."""
    text, _ = _rich.truncate_traceback(exc_type, exc_value, exc_traceback)
    return text, text


_NO_HOOKS: list = []


def _cached_get_all_hooks():
    """Drop-in replacement for ``inspect_ai.hooks._hooks.get_all_hooks``.

    Returns a shared empty list. This is correct for our setup (no third-party
    ``inspect_ai`` entry-points installed); if you ever add a hooks package
    you'll need to revert this patch."""
    return _NO_HOOKS


def _apply_once() -> None:
    if not getattr(_rich, _PATCH_MARKER, False):
        assert hasattr(_rich, "format_traceback"), (
            "inspect_ai._util.rich.format_traceback is missing — inspect_ai "
            "layout may have changed; re-validate the eval patch module"
        )
        assert hasattr(_rich, "truncate_traceback"), (
            "inspect_ai._util.rich.truncate_traceback is missing — re-validate"
        )
        _rich.format_traceback = _fast_format_traceback
        setattr(_rich, _PATCH_MARKER, True)

    if not getattr(_hooks, _PATCH_MARKER, False):
        assert hasattr(_hooks, "get_all_hooks"), (
            "inspect_ai.hooks._hooks.get_all_hooks is missing — re-validate"
        )
        _hooks.get_all_hooks = _cached_get_all_hooks
        setattr(_hooks, _PATCH_MARKER, True)


_apply_once()
