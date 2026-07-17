"""Shared helper for writing the pipeline ``manifest.json``.

Per pipeline_task() contract (pmanager.configs.schema.pipeline_task), every
pipeline entrypoint MUST write ``<output_dir>/manifest.json`` before exiting
cleanly. pmanager polls for this file to detect dataset completion and
register the dataset in its registry. pmanager does not fabricate one.

The manifest captures: stage name, every config param the entrypoint received,
input fingerprints (paths + key file hashes), output statistics. This is what
makes the dataset reproducible at audit time.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

SCHEMA_VERSION = 1


def write_manifest(
    output_dir: Path,
    *,
    stage: str,
    params: dict,
    inputs: dict,
    stats: dict,
) -> None:
    """Atomic write of ``output_dir/manifest.json``.

    stage   — short name, e.g. "prepare" / "run_length_cap" / "grain_payload" / "chunk_index"
    params  — every config value this stage was invoked with (no secrets)
    inputs  — {"<input_name>": <resolved_path>, ...}
    stats   — stage-specific output statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "params": params,
        "inputs": inputs,
        "stats": stats,
        "built_at": int(time.time()),
        "pmanager_run_id": os.environ.get("PMANAGER_RUN_ID", ""),
        "pmanager_parent_run_id": os.environ.get("PMANAGER_PARENT_RUN_ID", ""),
    }
    tmp = output_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(output_dir / "manifest.json")


def file_sha256_short(path: Path, n: int = 16) -> str:
    """Short SHA-256 of a file. Used for input fingerprints in the manifest."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:n]
