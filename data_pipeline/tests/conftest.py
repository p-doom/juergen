"""Make the source roots importable for the tests.

``pipeline`` lives at the repo root and is intentionally checkout-local, so the
repo root goes on ``sys.path``.

``pipeline.lib.action_format`` renders every label through a grammar codec, and
``grammars`` hard-imports ``desktop``, so the sibling checkout goes on the path
too — exactly as ``juergen/tests/conftest.py`` does. Under a venv that resolves
``desktop`` as the declared dependency it is already importable and this append
is inert.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_DESKTOP = REPO_ROOT.parent / "desktop"
if str(_DESKTOP) not in sys.path:
    sys.path.append(str(_DESKTOP))
