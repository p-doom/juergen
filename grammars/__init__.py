"""Action grammars: one directory per grammar, discovered at runtime.

Each directory holds

* ``codec.py`` — ``parse`` · ``format`` · ``compile`` · ``describe`` ·
  ``stop_sequences``, exported as a module-level ``CODEC`` singleton that
  satisfies ``desktop.codec_protocol.Codec``,
* ``vectors/*.json`` — conformance vectors pinning both directions, executed by
  ``grammars/test_vectors.py``.

A grammar contributes no dispatch table. It ``compile``s to ``Operation``s and
stops there: the Operation vocabulary is closed rather than open per grammar, so
lowering one is a fixed ``if kind ==`` chain inside desktop
(``execute/guest_program.py``) over a set no grammar extends. See
``_support.py``.

``parse`` (eval and RL rollout) and ``format`` (training-target construction)
are members of the same object, and the conformance vectors assert the round
trip between them. ``compile`` is the only place a coordinate convention is
resolved, and it always emits absolute screen pixels, so nothing downstream
carries a coordinate space.

No grammar spells termination. Ending an episode dispatches nothing, so it is
not something a grammar can say; it is ``_support.CONTROL_SPEC``, one line read
by ``split_control`` before any codec sees the text and re-exported here for the
episode driver and the dataset builder.
"""

from __future__ import annotations

import sys
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

ENTRY_POINT_GROUP = "juergen.grammars"

#: Re-exported from ``_support`` on first access, never at import. ``_support``
#: needs ``desktop``, and ``import grammars`` must not: ``available()`` lists all
#: seven from entry-point metadata before anything is imported, which is what lets
#: ``_explain_desktop`` report a wrong install as one instead of as a bare
#: ``No module named 'desktop.geometry'``.
_CONTROL_CHANNEL = ("CONTROL_SPEC", "CONTROL_TOKEN", "Control", "split_control")


def __getattr__(name: str) -> Any:
    if name in _CONTROL_CHANNEL:
        return getattr(import_module(f"{__name__}._support"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_CACHE: dict[str, Any] = {}


def _from_entry_points() -> dict[str, str]:
    return {
        entry.name: entry.value
        for entry in entry_points(group=ENTRY_POINT_GROUP)
    }


def _from_directories() -> dict[str, str]:
    """Fallback for an uninstalled checkout: a peer directory is a grammar.

    Keeps ``git checkout`` + ``python -c 'import grammars'`` working before the
    package is reinstalled.
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


#: Substring of the ImportError raised when ``desktop`` is importable but is
#: not the sibling checkout this repo means. See ``_explain_desktop``.
_WRONG_DESKTOP = "desktop."


def _explain_desktop(exc: ImportError) -> ImportError:
    """Turn a wrong-package import failure into a sentence that says so.

    Ours has no index presence, so the dependency is resolved by a
    ``[tool.uv.sources]`` path entry that only ``uv`` reads: ``pip install .``
    reads the index instead and gets PyPI's ``desktop``, a different package
    owning the same import name, and a stale wheel or a shadowing directory on
    ``sys.path`` does the same thing. All of them produce the same bare submodule
    error. That message names a submodule, which sends the reader looking for a
    missing file when the fault is a wrong or missing install — and
    ``available()`` lists all seven grammars first, because it reads entry-point
    metadata and imports nothing.

    Two distinct packages own the names near this one: PyPI's ``desktop`` 0.4.2,
    and ``xlang-ai/desktop_env`` (OSWorld), which ``evals/osworld.py`` imports in
    the same process to score the benchmark.
    """
    installed = sys.modules.get("desktop")
    return ImportError(
        "grammars needs this workspace's desktop, but the importable "
        f"`desktop` ({getattr(installed, '__file__', 'not importable')}) "
        f"has no {exc.name!r}. Ours has no PyPI presence and is resolved "
        "only by the [tool.uv.sources] path entry, which uv reads and pip does "
        "not: install with uv, or install ours directly with "
        "`uv pip install -e ../desktop`. (PyPI's `desktop` 0.4.2 is a different "
        "package with the same import name, and pip installs that one. "
        "xlang-ai/desktop_env — OSWorld — owns the neighbouring name; a leftover "
        "`desktop_env` egg-info or wheel produces exactly this error too.)"
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
        # Only reinterpret a failure to find part of desktop; anything else
        # is this codec's own problem and must not be dressed up as a packaging
        # one.
        if (exc.name or "").startswith(_WRONG_DESKTOP):
            raise _explain_desktop(exc) from exc
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


# Prepended to a codec's ``describe()`` for thinking+action records so training
# matches the thinking eval prompt (and is disambiguated from a tool-call-only
# retention set under an anneal mix). Grammar-independent: it describes the shape
# of the turn, never the action syntax.
#
# It lives here, not in ``datasets/convert.py``, because the dataset builder and
# the eval harness have to assemble the SAME prompt from the same codec and the
# harness compares digests against what the builder wrote. With a copy on each
# side the two drift and the comparison silently stops meaning anything, and the
# harness must not import the dataset tool to avoid that.
THINKING_PREAMBLE = (
    "For each step, first reason in a single <think>...</think> block — your current "
    "sub-goal and what you observe on the screen — then a one-line `Action:` describing "
    "the move, then the action itself.\n\n"
)


def system_prompt(codec: Any, *, thinking: bool) -> str:
    """The grammar's system prompt, in its thinking or plain form."""
    described = codec.describe()
    return (THINKING_PREAMBLE + described) if thinking else described


__all__ = [
    "CONTROL_SPEC",
    "CONTROL_TOKEN",
    "ENTRY_POINT_GROUP",
    "THINKING_PREAMBLE",
    "Control",
    "available",
    "codecs",
    "describe",
    "load",
    "split_control",
    "system_prompt",
]
