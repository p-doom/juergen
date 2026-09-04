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
import re
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
    artifact_type = manifest.get("artifact_type")
    expected_contract = {
        "crowdcast_stage_04_conversations": 1,
        "cuagym_stage_04_conversations": 1,
    }
    if manifest.get("schema_version") != expected_contract.get(artifact_type):
        raise ValueError(f"unsupported chat artifact contract: {manifest_path}")
    if artifact_type == "crowdcast_stage_04_conversations":
        _validate_crowdcast_chat_manifest(manifest, manifest_path)
    else:
        _validate_cuagym_chat_manifest(manifest, manifest_path)
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
    rows = []
    for line_number, line in enumerate(
        chat.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            raise ValueError(f"blank chat row at {chat}:{line_number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"chat row must be an object at {chat}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"chat artifact is empty: {chat}")
    if artifact_type == "crowdcast_stage_04_conversations":
        if (
            len(rows) != manifest["n_conversations"]
            or sum(row.get("n_turns", 0) for row in rows) != manifest["n_turns"]
        ):
            raise ValueError(f"Crowd-Cast chat counts mismatch: {chat}")
    else:
        stats = manifest["stats"]
        if (
            len(rows) != stats["records"]
            or len({row.get("task_id") for row in rows}) != stats["rollouts"]
        ):
            raise ValueError(f"CUA-Gym chat counts mismatch: {chat}")
    if artifact_type == "crowdcast_stage_04_conversations":
        master_id = manifest.get("master_store_id")
        if not isinstance(master_id, str):
            raise ValueError(f"Crowd-Cast chat has no master_store_id: {manifest_path}")
        master = _validate_crowdcast_master(
            check_artifact_id(master_id, what="master store")
        )
        expected_domain = f"jpeg_q92_height_{master['target_height']}"
        if manifest["image_domain"] != expected_domain:
            raise ValueError(
                f"Crowd-Cast chat image domain must be {expected_domain}: "
                f"{manifest_path}"
            )
    else:
        inputs = manifest.get("inputs")
        image_store = inputs.get("image_store") if isinstance(inputs, dict) else None
        image_store_id = (
            inputs.get("image_store_id") if isinstance(inputs, dict) else None
        )
        if not isinstance(image_store, str) or not isinstance(image_store_id, str):
            raise ValueError(
                f"CUA-Gym chat has no image-store identity: {manifest_path}"
            )
        store = check_artifact_id(image_store_id, what="CUA-Gym image store")
        if store != Path(image_store).resolve():
            raise ValueError("CUA-Gym chat image-store path/id mismatch")
        from pipeline.cua_gym.stage_01_image_store import validate_image_store

        validate_image_store(store)
    return chat


def _validate_crowdcast_chat_manifest(manifest: dict, path: Path) -> None:
    expected_fields = {
        "action_format",
        "artifact_type",
        "chat",
        "chat_sha256",
        "filter_id",
        "fps",
        "goals_id",
        "grammar",
        "image_domain",
        "master_store_id",
        "n_conversations",
        "n_turns",
        "projection_counts",
        "schema_version",
        "status_counts",
        "stride",
        "system_prompt_sha256",
    }
    if set(manifest) != expected_fields:
        raise ValueError(f"invalid Crowd-Cast chat manifest fields: {path}")
    import grammars

    prompt = grammars.describe("deltatype_v2")
    if (
        manifest.get("action_format") != "canonical"
        or manifest.get("grammar") != "deltatype_v2"
        or manifest.get("system_prompt_sha256")
        != hashlib.sha256(prompt.encode()).hexdigest()
        or not isinstance(manifest.get("image_domain"), str)
        or re.fullmatch(r"jpeg_q92_height_[1-9][0-9]*", manifest["image_domain"])
        is None
        or isinstance(manifest.get("stride"), bool)
        or not isinstance(manifest.get("stride"), int)
        or manifest["stride"] <= 0
        or isinstance(manifest.get("fps"), bool)
        or not isinstance(manifest.get("fps"), (int, float))
        or manifest["fps"] <= 0
        or any(
            isinstance(manifest.get(field), bool)
            or not isinstance(manifest.get(field), int)
            or manifest[field] <= 0
            for field in ("n_conversations", "n_turns")
        )
        or any(
            not isinstance(manifest.get(field), dict)
            or any(
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for key, value in manifest[field].items()
            )
            for field in ("projection_counts", "status_counts")
        )
    ):
        raise ValueError(f"Crowd-Cast chat production contract mismatch: {path}")


def _validate_cuagym_chat_manifest(manifest: dict, path: Path) -> None:
    expected_fields = {
        "artifact_type",
        "chat",
        "chat_sha256",
        "contract",
        "grammar",
        "inputs",
        "schema_version",
        "stats",
    }
    if set(manifest) != expected_fields:
        raise ValueError(f"invalid CUA-Gym chat manifest fields: {path}")
    from pipeline.cua_gym.stage_04_build_conversations import render_contract

    expected_contract = render_contract()
    expected_contract.pop("system_prompt")
    if (
        manifest.get("grammar") != expected_contract["grammar"]
        or manifest.get("contract") != expected_contract
    ):
        raise ValueError(f"CUA-Gym chat production contract mismatch: {path}")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "curated_trajectories",
        "curated_trajectories_id",
        "image_store",
        "image_store_id",
        "source_sha256",
    }:
        raise ValueError(f"invalid CUA-Gym chat inputs: {path}")
    curated_id = inputs["curated_trajectories_id"]
    curated_path = inputs["curated_trajectories"]
    if not isinstance(curated_id, str) or not isinstance(curated_path, str):
        raise TypeError(f"invalid CUA-Gym curated identity: {path}")
    curated = check_artifact_id(curated_id, what="CUA-Gym curated trajectories")
    if curated != Path(curated_path).resolve():
        raise ValueError("CUA-Gym curated trajectory path/id mismatch")
    from pipeline.cua_gym.stage_03_curate_trajectories import resolve_curated_artifact

    _, curated_manifest = resolve_curated_artifact(curated)
    source_sha = inputs.get("source_sha256")
    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 64
        or curated_manifest.get("inputs", {}).get("source_sha256") != source_sha
    ):
        raise ValueError(f"CUA-Gym source digest mismatch: {path}")
    stats = manifest.get("stats")
    if (
        not isinstance(stats, dict)
        or set(stats) != {"records", "rollouts"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in stats.values()
        )
        or stats["records"] != curated_manifest["stats"]["logical_targets"]
        or stats["rollouts"] != curated_manifest["stats"]["retained_rollouts"]
    ):
        raise ValueError(f"invalid CUA-Gym chat counts: {path}")


def _validate_crowdcast_master(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "artifact_type": "juergen_annotation_frames_master",
        "schema_version": 1,
        "segment_index": "segment_index.jsonl",
        "jpeg_quality": 92,
    }
    if {key: manifest.get(key) for key in required} != required:
        raise ValueError(f"invalid Crowd-Cast master artifact: {manifest_path}")
    target_height = manifest.get("target_height")
    if (
        isinstance(target_height, bool)
        or not isinstance(target_height, int)
        or target_height <= 0
    ):
        raise ValueError(f"invalid Crowd-Cast master height: {manifest_path}")
    index_path = root / "segment_index.jsonl"
    if file_sha256_short(index_path, n=64) != manifest.get("segment_index_sha256"):
        raise ValueError(f"Crowd-Cast master index digest mismatch: {index_path}")
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line]
    if not rows or any(row.get("status") != "ok" for row in rows):
        raise ValueError(
            f"Crowd-Cast master index is empty or incomplete: {index_path}"
        )
    from array_record.python.array_record_module import ArrayRecordReader

    for row in rows:
        segment_id = str(row["segment_id"])
        shard = (root / "frames" / segment_id / "images.array_record").resolve()
        frame_manifest = shard.parent / "frame_manifest.jsonl"
        if (
            Path(row["shard_path"]).resolve() != shard
            or Path(row["frame_manifest"]).resolve() != frame_manifest
        ):
            raise ValueError(f"Crowd-Cast master paths mismatch for {segment_id}")
        if file_sha256_short(shard, n=64) != row.get("shard_sha256"):
            raise ValueError(f"Crowd-Cast master shard digest mismatch: {shard}")
        if file_sha256_short(frame_manifest, n=64) != row.get("frame_manifest_sha256"):
            raise ValueError(
                f"Crowd-Cast frame manifest digest mismatch: {frame_manifest}"
            )
        reader = ArrayRecordReader(str(shard))
        if not reader.ok():
            raise ValueError(f"invalid Crowd-Cast master ArrayRecord: {shard}")
        try:
            count = reader.num_records()
        finally:
            reader.close()
        if count <= 0 or count != row.get("num_records"):
            raise ValueError(f"Crowd-Cast master record count mismatch: {shard}")
    return manifest


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
