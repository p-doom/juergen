"""Shared helper for writing a pipeline artifact's ``manifest.json`` marker."""

from __future__ import annotations

import hashlib
import json
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
        master_root = check_artifact_id(master_id, what="master store")
        master, images = _crowdcast_image_inventory(master_root)
        expected_domain = f"jpeg_q92_height_{master['target_height']}"
        if manifest["image_domain"] != expected_domain:
            raise ValueError(
                f"Crowd-Cast chat image domain must be {expected_domain}: "
                f"{manifest_path}"
            )
        filter_root = check_artifact_id(manifest["filter_id"], what="Crowd-Cast filter")
        goals_root = check_artifact_id(manifest["goals_id"], what="Crowd-Cast goals")
        filter_manifest = json.loads((filter_root / "manifest.json").read_text())
        goals_manifest = json.loads((goals_root / "manifest.json").read_text())
        if (
            filter_manifest.get("artifact_type") != "realigned_filter_mask"
            or filter_manifest.get("master_store_id") != master_id
            or goals_manifest.get("artifact_type") != "crowdcast_describe_extract_goals"
            or goals_manifest.get("master_store_id") != master_id
            or goals_manifest.get("filter_id") != manifest["filter_id"]
        ):
            raise ValueError("Crowd-Cast chat provenance mismatch")
        _validate_crowdcast_chat_rows(chat, manifest, images)
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
        _validate_cuagym_chat_rows(chat, manifest, _cuagym_image_inventory(store))
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


def _validate_image(
    value: object,
    images: dict[Path, tuple[int, str]],
    referenced: set[Path],
    chat: Path,
) -> None:
    from pipeline.lib.image_store import parse_arrayrecord_image_uri

    if not isinstance(value, str):
        raise TypeError(f"chat image must be text: {chat}")
    try:
        shard, index = parse_arrayrecord_image_uri(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid chat image URI in {chat}: {value!r}") from exc
    path = shard.resolve()
    receipt = images.get(path)
    if receipt is None or not 0 <= index < receipt[0]:
        raise ValueError(f"chat image is outside its attested store: {value!r}")
    referenced.add(path)


def _validate_message(
    message: object,
    images: dict[Path, tuple[int, str]],
    referenced: set[Path],
    chat: Path,
) -> tuple[str, list[dict]]:
    if not isinstance(message, dict) or set(message) not in (
        {"role", "content"},
        {"role", "content", "loss"},
    ):
        raise ValueError(f"invalid chat message in {chat}")
    role = message.get("role")
    content = message.get("content")
    if (
        role not in {"system", "user", "assistant"}
        or not isinstance(content, list)
        or not content
    ):
        raise ValueError(f"invalid chat message in {chat}")
    if "loss" in message and (role != "assistant" or message["loss"] is not False):
        raise ValueError(f"invalid chat loss mask in {chat}")
    for part in content:
        if not isinstance(part, dict):
            raise TypeError(f"chat content part must be an object: {chat}")
        if set(part) == {"type", "text"} and part.get("type") == "text":
            if not isinstance(part.get("text"), str) or not part["text"]:
                raise ValueError(f"chat text must be non-empty: {chat}")
        elif set(part) == {"type", "image"} and part.get("type") == "image":
            _validate_image(part.get("image"), images, referenced, chat)
        else:
            raise ValueError(f"invalid chat content part in {chat}")
    return role, content


def _validate_referenced_images(
    images: dict[Path, tuple[int, str]], referenced: set[Path]
) -> None:
    for path in referenced:
        expected = images[path][1]
        observed = file_sha256_short(path, n=64)
        if observed != expected:
            raise ValueError(
                f"image shard digest mismatch: expected {expected}, got {observed}: {path}"
            )


def _crowdcast_image_inventory(
    root: Path,
) -> tuple[dict, dict[Path, tuple[int, str]]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("artifact_type") != "juergen_annotation_frames_master"
        or manifest.get("schema_version") != 1
        or manifest.get("jpeg_quality") != 92
        or isinstance(manifest.get("target_height"), bool)
        or not isinstance(manifest.get("target_height"), int)
        or manifest["target_height"] <= 0
        or manifest.get("segment_index") != "segment_index.jsonl"
    ):
        raise ValueError(f"invalid master image contract: {manifest_path}")
    index = root / "segment_index.jsonl"
    expected = manifest.get("segment_index_sha256")
    if not isinstance(expected, str) or file_sha256_short(index, n=64) != expected:
        raise ValueError(f"master image index digest mismatch: {index}")
    images: dict[Path, tuple[int, str]] = {}
    for line_number, line in enumerate(index.read_text().splitlines(), 1):
        row = json.loads(line)
        count = row.get("num_records") if isinstance(row, dict) else None
        shard = row.get("shard_path") if isinstance(row, dict) else None
        digest = row.get("shard_sha256") if isinstance(row, dict) else None
        if (
            not isinstance(shard, str)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or row.get("jpeg_quality") != 92
            or row.get("target_height") != manifest["target_height"]
        ):
            raise ValueError(f"invalid master image row at {index}:{line_number}")
        path = Path(shard).resolve()
        if path in images or not path.is_file():
            raise ValueError(f"invalid master image shard at {index}:{line_number}")
        images[path] = (count, digest)
    if len(images) != manifest.get("n_segments"):
        raise ValueError(f"master image count mismatch: {manifest_path}")
    return manifest, images


def _cuagym_image_inventory(root: Path) -> dict[Path, tuple[int, str]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    required = {
        "artifact_type": "cuagym_stage_01_image_store",
        "schema_version": 1,
        "uri_scheme": "ar:///abs/path/images.array_record#idx",
        "jpeg_quality": 92,
        "width": 1920,
        "height": 1080,
        "image_domain": "osworld_cursor_jpeg_q92_420_1920x1080_v1",
    }
    generation = manifest.get("generation")
    shards = manifest.get("shards")
    if (
        {key: manifest.get(key) for key in required} != required
        or not isinstance(generation, str)
        or Path(generation).name != generation
        or not isinstance(shards, dict)
        or not shards
    ):
        raise ValueError(f"invalid CUA-Gym image-store contract: {manifest_path}")
    images: dict[Path, tuple[int, str]] = {}
    for name, receipt in shards.items():
        count = receipt.get("num_images") if isinstance(receipt, dict) else None
        digest = (
            receipt.get("arrayrecord_sha256") if isinstance(receipt, dict) else None
        )
        path = (root / generation / str(name) / "images.array_record").resolve()
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not path.is_file()
        ):
            raise ValueError(f"invalid CUA-Gym image shard: {name!r}")
        images[path] = (count, digest)
    if manifest.get("num_tars") != len(images) or manifest.get("total_images") != sum(
        item[0] for item in images.values()
    ):
        raise ValueError(f"CUA-Gym image-store count mismatch: {manifest_path}")
    return images


def _validate_crowdcast_chat_rows(
    chat: Path, manifest: dict, images: dict[Path, tuple[int, str]]
) -> None:
    n_conversations = 0
    n_turns = 0
    referenced: set[Path] = set()
    for row in _chat_rows(chat):
        messages = row.get("messages")
        turns = row.get("n_turns")
        if (
            not isinstance(messages, list)
            or isinstance(turns, bool)
            or not isinstance(turns, int)
            or turns <= 0
            or len(messages) != 1 + 2 * turns
            or not isinstance(row.get("recording_id"), str)
            or not row["recording_id"]
        ):
            raise ValueError(f"invalid Crowd-Cast chat row in {chat}")
        roles = [
            _validate_message(message, images, referenced, chat)[0]
            for message in messages
        ]
        expected_roles = [
            "system",
            *[role for _ in range(turns) for role in ("user", "assistant")],
        ]
        if roles != expected_roles:
            raise ValueError(f"invalid Crowd-Cast message order in {chat}")
        if [part["type"] for part in messages[0]["content"]] != ["text"]:
            raise ValueError(f"invalid Crowd-Cast system context in {chat}")
        if sum(part["type"] == "text" for part in messages[1]["content"]) != 1:
            raise ValueError(f"invalid Crowd-Cast instruction context in {chat}")
        if any(
            sum(part["type"] == "image" for part in message["content"]) != 1
            for message in messages
            if message["role"] == "user"
        ):
            raise ValueError(f"Crowd-Cast user message image mismatch in {chat}")
        for message in messages[2::2]:
            if "loss" in message or [part["type"] for part in message["content"]] != [
                "text"
            ]:
                raise ValueError(f"invalid Crowd-Cast target in {chat}")
        n_conversations += 1
        n_turns += turns
    if n_conversations != manifest["n_conversations"] or n_turns != manifest["n_turns"]:
        raise ValueError(f"Crowd-Cast chat counts mismatch: {chat}")
    _validate_referenced_images(images, referenced)


def _validate_cuagym_chat_rows(
    chat: Path,
    manifest: dict,
    images: dict[Path, tuple[int, str]],
) -> None:
    rollouts: set[str] = set()
    records = 0
    referenced: set[Path] = set()
    for row in _chat_rows(chat):
        messages = row.get("messages")
        n_history_turns = row.get("n_history_turns")
        if (
            not isinstance(messages, list)
            or len(messages) < 3
            or not isinstance(row.get("recording_id"), str)
            or not row["recording_id"]
            or isinstance(n_history_turns, bool)
            or not isinstance(n_history_turns, int)
            or n_history_turns < 0
        ):
            raise ValueError(f"invalid CUA-Gym chat row in {chat}")
        roles = [
            _validate_message(message, images, referenced, chat)[0]
            for message in messages
        ]
        if (
            roles[0] != "system"
            or roles[-1] != "assistant"
            or any(
                role != ("user" if index % 2 else "assistant")
                for index, role in enumerate(roles[1:], 1)
            )
        ):
            raise ValueError(f"invalid CUA-Gym message order in {chat}")
        assistants = [message for message in messages if message["role"] == "assistant"]
        history = assistants[:-1]
        target = assistants[-1]
        first_user_text = [
            part["text"] for part in messages[1]["content"] if part["type"] == "text"
        ]
        if (
            [part["type"] for part in messages[0]["content"]] != ["text"]
            or len(first_user_text) != 1
            or len(history) != n_history_turns
            or any(message.get("loss") is not False for message in history)
            or set(target) != {"role", "content"}
            or sum(message.get("loss", True) is True for message in assistants) != 1
            or any(
                sum(part["type"] == "image" for part in message["content"]) != 1
                for message in messages
                if message["role"] == "user"
            )
        ):
            raise ValueError(f"invalid CUA-Gym supervision context in {chat}")
        if any(
            [part["type"] for part in message["content"]] != ["text"]
            for message in assistants
        ):
            raise ValueError(f"invalid CUA-Gym assistant content in {chat}")
        recording_id = row.get("recording_id")
        rollouts.add(recording_id)
        records += 1
    if manifest["stats"] != {"records": records, "rollouts": len(rollouts)}:
        raise ValueError("CUA-Gym chat receipt counts mismatch")
    _validate_referenced_images(images, referenced)


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
    curated_manifest = json.loads((curated / "manifest.json").read_text())
    source_sha = inputs.get("source_sha256")
    curated_stats = curated_manifest.get("stats")
    if (
        curated_manifest.get("artifact_type") != "cuagym_stage_03_curated_trajectories"
        or curated_manifest.get("schema_version") != 1
        or not isinstance(source_sha, str)
        or len(source_sha) != 64
        or curated_manifest.get("inputs", {}).get("source_sha256") != source_sha
        or not isinstance(curated_stats, dict)
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
        or stats["records"] != curated_stats.get("executable_targets")
        or stats["rollouts"] != curated_stats.get("retained_rollouts")
    ):
        raise ValueError(f"invalid CUA-Gym chat counts: {path}")


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
