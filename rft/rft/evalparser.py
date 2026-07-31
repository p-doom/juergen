"""Import shim: re-export the EXACT parser the evaluation harness uses.

``juergen/eval/action_parser.py`` is a flat module inside the ``eval`` uv
workspace member, not an installed package. Every consumer in this package
imports it through this shim so that:

* there is exactly one import path, and it points at the harness's own parser —
  a round-trip audit through :mod:`rft.roundtrip` is therefore, by construction,
  an audit through the code that scores the eval;
* nobody is tempted to vendor a "compatible" copy. A second parser that drifts
  from the first is how a conversion audit passes while the eval fails.

**Symbol availability is explicit.** Different revisions of ``action_parser.py``
support different grammars: ``parse_deltatype`` / ``format_deltatype`` /
``parse_computer_use_tool_calls`` were added after the ``parse_action`` family. If a
symbol this package needs is absent, the name is bound to a stub that **raises**
naming the missing symbol and the parser file it was looked for in, and
:data:`MISSING_SYMBOLS` records it. There is no fallback implementation: a vendored
parser that silently substitutes for the harness's is precisely the failure this
module prevents.

Set ``JUERGEN_EVAL_DIR`` to point at a specific ``eval`` directory (e.g. a checkout
whose deltatype support is not yet committed).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_ENV_VAR = "JUERGEN_EVAL_DIR"

#: Symbols this package uses. Names in ``_REQUIRED`` must exist or import fails —
#: without them nothing in the package can parse anything. Names in ``_OPTIONAL``
#: gate individual grammars.
_REQUIRED: tuple[str, ...] = (
    "Action",
    "KeyEvent",
    "ComputerUseCall",
    "parse_action",
    "parse_action_tolerant",
    "parse_computer_use_tool_call",
)
_OPTIONAL: tuple[str, ...] = (
    "DeltaTypeAction",
    "parse_deltatype",
    "format_deltatype",
    "parse_computer_use_tool_calls",
)


def _candidate_dirs() -> list[Path]:
    cands: list[Path] = []
    override = os.environ.get(_ENV_VAR)
    if override:
        cands.append(Path(override))
    # rft/rft/evalparser.py -> rft/ -> <repo root>/eval
    repo_root = Path(__file__).resolve().parents[2]
    cands.append(repo_root / "eval")
    return cands


def _load() -> ModuleType:
    tried: list[str] = []
    for d in _candidate_dirs():
        path = d / "action_parser.py"
        tried.append(str(path))
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("juergen_eval_action_parser", path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    raise ImportError(
        "could not locate juergen's eval/action_parser.py (the harness parser). "
        f"Tried: {tried}. Set {_ENV_VAR} to the directory containing it. "
        "Refusing to fall back to a vendored copy."
    )


_mod = _load()

#: Absolute path of the parser actually in use. Printed in every diagnostics block
#: so a report always says which parser produced it.
ACTION_PARSER_PATH: str = str(_mod.__file__)

#: Optional symbols the loaded parser does not provide.
MISSING_SYMBOLS: tuple[str, ...] = ()


def _missing_stub(name: str) -> Any:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise ImportError(
            f"{ACTION_PARSER_PATH} does not define {name!r}. That revision of the "
            "harness parser predates this grammar. Point JUERGEN_EVAL_DIR at an "
            f"eval/ whose action_parser.py provides {name!r}, or land that support "
            "on the branch you are running. This package will NOT substitute its own "
            "parser for the harness's."
        )

    _raise.__name__ = f"missing__{name}"
    _raise.rft_missing_symbol = name  # type: ignore[attr-defined]
    return _raise


_absent: list[str] = []
for _name in _REQUIRED:
    if not hasattr(_mod, _name):
        raise ImportError(
            f"{ACTION_PARSER_PATH} does not define required symbol {_name!r}; "
            "this is not a usable harness parser"
        )
    globals()[_name] = getattr(_mod, _name)
for _name in _OPTIONAL:
    if hasattr(_mod, _name):
        globals()[_name] = getattr(_mod, _name)
    else:
        globals()[_name] = _missing_stub(_name)
        _absent.append(_name)
MISSING_SYMBOLS = tuple(_absent)


def have(name: str) -> bool:
    """Whether the loaded parser really provides ``name``."""
    return name not in MISSING_SYMBOLS


def describe() -> str:
    """One line naming the parser in use and anything it is missing."""
    missing = f"; MISSING: {list(MISSING_SYMBOLS)}" if MISSING_SYMBOLS else ""
    return f"eval parser: {ACTION_PARSER_PATH}{missing}"


__all__ = [
    "ACTION_PARSER_PATH",
    "MISSING_SYMBOLS",
    "Action",
    "ComputerUseCall",
    "DeltaTypeAction",
    "KeyEvent",
    "describe",
    "format_deltatype",
    "have",
    "parse_action",
    "parse_action_tolerant",
    "parse_computer_use_tool_call",
    "parse_computer_use_tool_calls",
    "parse_deltatype",
]
