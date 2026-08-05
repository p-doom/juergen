"""Make both source roots importable for the tests.

``pipeline`` now lives at the REPO ROOT (moved out of ``data_pipeline`` in the
data-layer restructure); ``annotation_pipeline`` and ``configs`` still live under
``data_pipeline``. Neither is an installed package (the stage scripts sys.path-hack
their own root when run directly), so both roots go on ``sys.path``.
"""

import sys
from pathlib import Path

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DATA_PIPELINE_DIR.parent
for _root in (DATA_PIPELINE_DIR, REPO_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
