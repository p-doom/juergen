"""Make the source roots importable for the tests.

``pipeline`` lives at the repo root (one package per corpus, e.g.
``pipeline.crowdcast``); ``configs`` lives under ``data_pipeline``.
Neither is an installed package (the stage scripts sys.path-hack their own root
when run directly), so both roots go on ``sys.path``.

``pipeline.crowdcast.lib.action_format`` renders every label through a grammar codec, and
``grammars`` hard-imports ``desktop``, so the sibling checkout goes on the path
too — exactly as ``juergen/tests/conftest.py`` does. Under a venv that resolves
``desktop`` as the declared dependency it is already importable and this append
is inert.
"""

import sys
from pathlib import Path

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DATA_PIPELINE_DIR.parent
for _root in (DATA_PIPELINE_DIR, REPO_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
_DESKTOP = REPO_ROOT.parent / "desktop"
if str(_DESKTOP) not in sys.path:
    sys.path.append(str(_DESKTOP))
