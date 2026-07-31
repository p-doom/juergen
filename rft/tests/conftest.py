"""Make ``rft`` importable and point the eval-parser shim at this checkout's ``eval/``.

``rft`` is not installed as a package in every venv on this cluster, and
``eval/action_parser.py`` is a flat module rather than an installed package, so tests
resolve both by path. Nothing here vendors or stubs the parser: if the checkout's
``eval/action_parser.py`` lacks a grammar's symbols, the grammar reports itself
unavailable and the tests that need it skip loudly (see ``test_grammars.py``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

RFT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = RFT_DIR.parent
EVAL_DIR = REPO_ROOT / "eval"

if str(RFT_DIR) not in sys.path:
    sys.path.insert(0, str(RFT_DIR))

# Only set the override if the caller has not; an operator pointing
# JUERGEN_EVAL_DIR at a different checkout must win.
if "JUERGEN_EVAL_DIR" not in os.environ and (EVAL_DIR / "action_parser.py").is_file():
    os.environ["JUERGEN_EVAL_DIR"] = str(EVAL_DIR)
