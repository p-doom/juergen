"""Action grammars: one directory per grammar, discovered at runtime.

A grammar is a peer, not a case in a switch. Each directory holds

* ``codec.py`` — ``parse`` · ``format`` · ``compile`` · ``describe`` ·
  ``stop_sequences``, exported as a module-level ``CODEC`` singleton that
  satisfies ``desktop_env.codec_protocol.Codec``,
* ``vectors/*.json`` — conformance vectors pinning both directions, executed by
  ``grammars/test_vectors.py``.

A grammar contributes NO dispatch table. It ``compile``s to ``Operation``s and
stops there: the Operation vocabulary is closed by physics rather than open per
grammar, so lowering one is a fixed ``if kind ==`` chain inside desktop-env
(``execute/guest_program.py``) over a set no grammar extends. The seven
``handlers.py`` modules that used to live here described a second dispatch engine
that no code in desktop-env ever consumed, and their ``dict[str, Handler]``
annotation named a ``Handler`` with the opposite signature.

``parse`` (eval and RL rollout) and ``format`` (training-target construction)
are members of the same object, because that round-trip is what stops the
trained grammar and the parsed grammar from diverging. ``compile`` is the only
place a coordinate convention is resolved, and it always emits absolute screen
pixels, so nothing downstream carries a coordinate space.
"""

from __future__ import annotations

import sys
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

ENTRY_POINT_GROUP = "juergen.grammars"

_CACHE: dict[str, Any] = {}


def _from_entry_points() -> dict[str, str]:
    return {
        entry.name: entry.value
        for entry in entry_points(group=ENTRY_POINT_GROUP)
    }


def _from_directories() -> dict[str, str]:
    """Fallback for an uninstalled checkout: a peer directory IS a grammar.

    Keeps ``git checkout`` + ``python -c 'import grammars'`` working before the
    package is reinstalled, so adding a grammar never blocks on packaging.
    """
    root = Path(__file__).parent
    found: dict[str, str] = {}
    for child in sorted(root.iterdir()):
        if child.name.startswith(("_", ".")) or not child.is_dir():
            continue
        if (child / "codec.py").is_file():
            found[child.name] = f"{__name__}.{child.name}.codec:CODEC"
    return found


def _targets() -> dict[str, str]:
    targets = _from_directories()
    targets.update(_from_entry_points())  # entry points are authoritative
    return targets


def available() -> tuple[str, ...]:
    """Every registered grammar name."""
    return tuple(sorted(_targets()))


#: Substring of the ImportError raised when the WRONG ``desktop_env`` is
#: installed. See ``_explain_desktop_env``.
_WRONG_DESKTOP_ENV = "desktop_env."


def _explain_desktop_env(exc: ImportError) -> ImportError:
    """Turn a wrong-package import failure into a sentence that says so.

    ``desktop-env`` is TAKEN ON PyPI, by ``xlang-ai/desktop_env`` (OSWorld) —
    same distribution name and same import name, entirely different package. Ours
    has no remote and no index presence, so it is resolved by the
    ``[tool.uv.sources]`` path entry in the root ``pyproject.toml``, and only
    ``uv`` reads that. ``pip install .`` therefore resolves the dependency from
    PyPI, installs OSWorld's package, and this import dies with a bare
    ``No module named 'desktop_env.geometry'``.

    That message names a submodule, which sends the reader looking for a missing
    file. The failure is a wrong package. Nothing in the default message says so,
    and ``available()`` still cheerfully lists all seven grammars beforehand,
    because it reads entry-point metadata and imports nothing — so the first real
    signal is this exception, and it has to be legible.
    """
    installed = sys.modules.get("desktop_env")
    return ImportError(
        "grammars needs this workspace's desktop-env, but the installed "
        f"`desktop_env` ({getattr(installed, '__file__', 'not importable')}) "
        f"has no {exc.name!r}. `desktop-env` on PyPI is xlang-ai/desktop_env "
        "(OSWorld): the same import name, a different package. Install with uv, "
        "which reads the [tool.uv.sources] path entry, or install ours directly "
        "with `uv pip install -e ../desktop-env`."
    )


def load(name: str) -> Any:
    """The codec for one grammar."""
    if name in _CACHE:
        return _CACHE[name]
    targets = _targets()
    try:
        target = targets[name]
    except KeyError:
        raise KeyError(
            f"unknown grammar {name!r} (available: {sorted(targets)})"
        ) from None
    module_path, _, attribute = target.partition(":")
    try:
        module = import_module(module_path)
    except ImportError as exc:
        # Only reinterpret a failure to find part of desktop_env; anything else
        # is this codec's own problem and must not be dressed up as a packaging
        # one.
        if (exc.name or "").startswith(_WRONG_DESKTOP_ENV):
            raise _explain_desktop_env(exc) from exc
        raise
    codec = getattr(module, attribute or "CODEC")
    _CACHE[name] = codec
    return codec


def codecs() -> dict[str, Any]:
    """Every codec, keyed by grammar name."""
    return {name: load(name) for name in available()}


def describe(name: str) -> str:
    """The system prompt for one grammar, derived from its codec's docstrings."""
    return load(name).describe()


__all__ = [
    "ENTRY_POINT_GROUP",
    "available",
    "codecs",
    "describe",
    "load",
]
