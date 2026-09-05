"""Shared helper for writing a pipeline artifact's ``manifest.json`` marker."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
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
        _validate_crowdcast_chat_rows(chat, manifest)
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
        _validate_cuagym_chat_rows(chat, manifest, store)
    return chat


def _chat_rows(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"blank chat row at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"chat row must be an object at {path}:{line_number}")
            yield row


def _next_chat_row(rows, path: Path) -> dict:
    try:
        return next(rows)
    except StopIteration as exc:
        raise ValueError(f"chat artifact ended before its source rows: {path}") from exc


def _validate_crowdcast_chat_rows(chat: Path, manifest: dict) -> None:
    import grammars
    from pipeline.lib.views import FilterArtifact
    from pipeline.stage_04_build_conversations import (
        build_segment_conversations,
        resolve_goals,
    )

    filter_dir = check_artifact_id(manifest["filter_id"], what="Crowd-Cast filter")
    goals_dir = check_artifact_id(manifest["goals_id"], what="Crowd-Cast goals")
    artifact = FilterArtifact(filter_dir)
    if artifact.master_store_id != manifest["master_store_id"]:
        raise ValueError("Crowd-Cast chat master/filter identity mismatch")
    goals, goals_id = resolve_goals(artifact, goals_dir)
    if goals_id != manifest["goals_id"]:
        raise ValueError("Crowd-Cast chat goals identity mismatch")
    system_prompt = grammars.describe("deltatype_v2")
    observed = iter(_chat_rows(chat))
    n_conversations = 0
    n_turns = 0
    produced_goal_ids: set[str] = set()
    statuses: Counter[str] = Counter()
    projections: Counter[str] = Counter()
    for index_row in sorted(
        artifact.usable_rows(), key=lambda row: str(row["segment_id"])
    ):
        result = build_segment_conversations(
            {
                "index_row": index_row,
                "filter_segment": artifact.load_segment(str(index_row["segment_id"])),
                "fps": manifest["fps"],
                "goals_by_segment": goals,
                "system_prompt": system_prompt,
            }
        )
        projection = result.get("projection")
        if isinstance(projection, dict) and projection.get(
            "n_projected"
        ) != projection.get("n_goals"):
            raise ValueError("Crowd-Cast sealed chat has an unprojectable goal")
        statuses[result["status"]] += 1
        for expected in sorted(result["rows"], key=lambda row: row["conversation_id"]):
            if _next_chat_row(observed, chat) != expected:
                raise ValueError(
                    "Crowd-Cast chat rows do not match their source artifacts"
                )
            n_conversations += 1
            n_turns += expected["n_turns"]
            produced_goal_ids.add(expected["goal_id"])
        for key, value in result.get("projection", {}).items():
            if isinstance(value, int):
                projections[key] += value
    if next(observed, None) is not None:
        raise ValueError("Crowd-Cast chat has rows absent from its source artifacts")
    expected_goal_ids = {
        goal["goal_id"] for segment_goals in goals.values() for goal in segment_goals
    }
    if (
        n_conversations == 0
        or n_conversations != manifest["n_conversations"]
        or n_turns != manifest["n_turns"]
        or produced_goal_ids != expected_goal_ids
        or n_conversations != len(expected_goal_ids)
    ):
        raise ValueError(f"Crowd-Cast chat counts mismatch: {chat}")
    if manifest["status_counts"] != dict(statuses) or manifest[
        "projection_counts"
    ] != dict(projections):
        raise ValueError("Crowd-Cast chat receipt counts mismatch")


def _validate_cuagym_chat_rows(chat: Path, manifest: dict, image_store: Path) -> None:
    from pipeline.cua_gym.stage_03_curate_trajectories import resolve_curated_artifact
    from pipeline.cua_gym.stage_04_build_conversations import (
        ImageIndex,
        build_episode_records,
        render_contract,
    )

    curated_root = Path(manifest["inputs"]["curated_trajectories"])
    trajectories, _ = resolve_curated_artifact(curated_root)
    images = ImageIndex(image_store)
    contract = render_contract()
    counters: Counter[str] = Counter()
    observed = iter(_chat_rows(chat))
    for line_number, line in enumerate(
        trajectories.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            raise ValueError(f"blank curated row at {trajectories}:{line_number}")
        for expected in build_episode_records(
            json.loads(line), images, contract, counters
        ):
            if _next_chat_row(observed, chat) != expected:
                raise ValueError(
                    "CUA-Gym chat rows do not match their curated/image artifacts"
                )
    if next(observed, None) is not None:
        raise ValueError("CUA-Gym chat has rows absent from its source artifacts")
    if manifest["stats"] != dict(sorted(counters.items())):
        raise ValueError("CUA-Gym chat receipt counts mismatch")


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
        or stats["records"] != curated_manifest["stats"]["executable_targets"]
        or stats["rollouts"] != curated_manifest["stats"]["retained_rollouts"]
    ):
        raise ValueError(f"invalid CUA-Gym chat counts: {path}")


def _validate_crowdcast_master(root: Path) -> dict:
    from pipeline.lib.master_frames import resolve_master_artifact

    return resolve_master_artifact(root)[0]


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
