#!/usr/bin/env python3
"""DEPRECATED alias — the implementation moved to ``pipeline``.

``pipeline/stage_02_realign.py`` is this module: the two files were identical
except for the import header (``pipeline.lib.common`` vs
``annotation_pipeline.common``, whose ``ensure_dir`` / ``read_jsonl`` /
``write_json`` are byte-identical, and the REPO_ROOT depth). This alias exists
only so recorded labctl ``submit.sh`` scripts keep resolving
``python -m annotation_pipeline.stage_00_realign``; new callers should use
``python -m pipeline.stage_02_realign``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``pipeline`` lives at the REPO ROOT, one level above ``data_pipeline``.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.stage_02_realign import (  # noqa: E402,F401
    build_keylog_index,
    main,
    parse_args,
    realign_one_recording,
    sibling_segments,
    write_corrected_keylog,
)

if __name__ == "__main__":
    main()
