"""Make the repo root importable so tests can import ``realigned_pipeline``,
which is not an installed package (its stage scripts sys.path-hack the same
directory when run directly)."""

import sys
from pathlib import Path

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))
