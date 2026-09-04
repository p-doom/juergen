"""Shared helper for writing the pipeline ``manifest.json``.

Per pipeline_task() contract (pmanager.configs.schema.pipeline_task), every
pipeline entrypoint must write ``<output_dir>/manifest.json`` before exiting
cleanly. pmanager polls for this file to detect dataset completion and
register the dataset in its registry. pmanager does not fabricate one.

The manifest captures: stage name, every config param the entrypoint received,
input fingerprints (paths + key file hashes), output statistics.
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


def make_artifact_id(artifact_dir: Path) -> str:
    """Identity of a built artifact: ``<abs dir>::<sha16 of its manifest.json>``.

    Downstream stages record the ids of their inputs (``master_store_id``,
    ``filter_id``) and refuse joins whose recorded id no longer matches the
    artifact on disk (e.g. a master store rebuilt in place)."""
    artifact_dir = Path(artifact_dir).resolve()
    return f"{artifact_dir}::{file_sha256_short(artifact_dir / 'manifest.json')}"


def resolve_chat_artifact(artifact_dir: Path) -> Path:
    artifact_dir = Path(artifact_dir).resolve()
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("chat") != "chat.jsonl":
        raise ValueError(f"invalid chat artifact manifest: {manifest_path}")
    expected = manifest.get("chat_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"chat artifact has no SHA-256: {manifest_path}")
    chat = artifact_dir / "chat.jsonl"
    if not chat.is_file():
        raise FileNotFoundError(f"chat artifact is missing: {chat}")
    observed = file_sha256_short(chat, n=64)
    if observed != expected:
        raise ValueError(
            f"chat digest mismatch for {chat}: expected {expected}, got {observed}"
        )
    return chat


def check_artifact_id(artifact_id: str, *, what: str) -> Path:
    """Verify a recorded artifact id against the artifact currently on disk.

    Returns the artifact directory. Raises if the manifest is gone or its hash
    changed — the artifact was rebuilt and every downstream join is stale."""
    path_s, _, recorded_sha = artifact_id.rpartition("::")
    if not path_s or not recorded_sha:
        raise ValueError(f"malformed {what} artifact id: {artifact_id!r}")
    artifact_dir = Path(path_s)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{what} at {artifact_dir} has no manifest.json (moved or deleted?)"
        )
    current = file_sha256_short(manifest_path, n=len(recorded_sha))
    if current != recorded_sha:
        raise ValueError(
            f"{what} at {artifact_dir} was rebuilt since this artifact was made "
            f"(manifest sha {current} != recorded {recorded_sha}); re-run the consumer stage"
        )
    return artifact_dir
