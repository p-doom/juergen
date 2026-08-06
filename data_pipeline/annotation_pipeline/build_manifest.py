#!/usr/bin/env python3
"""DEPRECATED alias — the implementation moved to ``pipeline``.

``pipeline/stage_00_clip_manifest.py`` is this module, byte-identical apart
from the usage line in its docstring. This alias exists only so recorded
labctl ``submit.sh`` scripts and ``slurm/run_dataset_smoke.sbatch`` keep
resolving ``python3 -m annotation_pipeline.build_manifest``; new callers
should use ``python3 -m pipeline.stage_00_clip_manifest``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``pipeline`` lives at the REPO ROOT, one level above ``data_pipeline``
# (mirrors ``annotation_pipeline/stage_00_realign.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.stage_00_clip_manifest import (  # noqa: E402,F401
    build_row,
    main,
    parse_args,
    probe_video,
)

if __name__ == "__main__":
    main()
