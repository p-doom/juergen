#!/usr/bin/env python3
"""DEPRECATED alias — the implementation moved to ``pipeline``.

Kept only so recorded labctl ``submit.sh`` scripts and the in-package
``build_sft.py`` driver keep resolving
``python -m annotation_pipeline.stage_04_build_canonical_sft`` /
``from annotation_pipeline.stage_04_build_canonical_sft import build_canonical_sft``.
There is one copy of the logic and it lives at
``pipeline/stage_04_build_canonical_sft.py``; import from there in new code.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``pipeline`` lives at the REPO ROOT, one level above ``data_pipeline``
# (mirrors ``annotation_pipeline/stage_00_realign.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.stage_04_build_canonical_sft import (  # noqa: E402,F401
    VALID_IMAGE_PATH_MODES,
    VALID_SPLIT_GROUPS,
    VALID_TERMINAL_MODES,
    apply_terminal_policy,
    assign_splits,
    build_canonical_sft,
    ensure_empty_dir,
    load_system_prompt,
    main,
    parse_args,
    read_jsonl,
    render_image_path,
    resolve_image_path,
    sanitize_component,
    stage03_output,
    text_content,
    transform_messages,
    write_json,
    write_jsonl,
)

if __name__ == "__main__":
    main()
