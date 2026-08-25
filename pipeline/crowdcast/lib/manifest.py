"""Re-export of the corpus-agnostic manifest helpers.

The manifest is the contract every `pipeline` corpus shares — pmanager polls for
`<output_dir>/manifest.json` to detect completion — so the writer lives at
`pipeline.manifest`, reachable from `pipeline.crowdcast` and `pipeline.finevision`
alike. This module keeps the import path eight crowd-cast callers already use.
"""

from __future__ import annotations

from pipeline.manifest import (
    SCHEMA_VERSION,
    check_artifact_id,
    file_sha256_short,
    make_artifact_id,
    write_manifest,
)

__all__ = [
    "SCHEMA_VERSION",
    "check_artifact_id",
    "file_sha256_short",
    "make_artifact_id",
    "write_manifest",
]
