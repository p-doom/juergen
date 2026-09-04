"""Attest the Omegalax data compiler consumed by the two SFT streams."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from itertools import zip_longest
from pathlib import Path
from typing import Any

_TRACKED_PATHS = (
    "scripts/measure_message_lengths_from_chat.py",
    "scripts/build_sft_records_from_chat.py",
    "omegalax",
    "pyproject.toml",
    "uv.lock",
)
_SNAPSHOT_REVISION = re.compile(r"[0-9a-f]{40}")
_SNAPSHOT_REQUIRED_FILES = {
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
}
_MODEL_WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".h5",
    ".msgpack",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
_LOSS_MASK_PROBE = """
import json
from omegalax.data.qwen3_encoding import message_is_supervised
print(json.dumps([
    message_is_supervised({"role": "assistant", "content": "history", "loss": False}),
    message_is_supervised({"role": "assistant", "content": "target"}),
]))
""".strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attest_processor_snapshot(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if (
        not resolved.is_dir()
        or resolved.parent.name != "snapshots"
        or _SNAPSHOT_REVISION.fullmatch(resolved.name) is None
    ):
        raise ValueError(
            "model snapshot must be an existing Hugging Face snapshots/<40-hex-revision> directory"
        )
    paths = sorted(path for path in resolved.rglob("*") if path.is_file())
    relative = {str(path.relative_to(resolved)): path for path in paths}
    missing = _SNAPSHOT_REQUIRED_FILES - set(relative)
    if missing:
        raise ValueError(
            f"processor snapshot is missing required files: {sorted(missing)}"
        )
    weights = [
        name for name, item in relative.items() if item.suffix in _MODEL_WEIGHT_SUFFIXES
    ]
    if weights:
        raise ValueError(
            f"processor snapshot must not contain model weights: {weights}"
        )
    files = {name: _sha256(item) for name, item in relative.items()}
    return {"path": str(resolved), "revision": resolved.name, "files": files}


def attest_omegalax(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Omegalax project is not a directory: {root}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *_TRACKED_PATHS,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError(f"Omegalax compiler checkout has tracked changes:\n{status}")
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", *_TRACKED_PATHS],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    names = sorted(name.decode() for name in tracked if name)
    required = set(_TRACKED_PATHS) - {"omegalax"}
    if not required.issubset(names) or not any(
        name.startswith("omegalax/") for name in names
    ):
        raise ValueError("Omegalax checkout is missing consumed tracked files")
    files = {name: _sha256(root / name) for name in names}
    environment = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    probe = subprocess.run(
        [
            "uv",
            "run",
            "--locked",
            "--project",
            str(root),
            "python",
            "-c",
            _LOSS_MASK_PROBE,
        ],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if probe != "[false, true]":
        raise ValueError(
            "Omegalax must exclude loss:false assistant messages and supervise targets"
        )
    return {
        "path": str(root),
        "commit": head,
        "tree": tree,
        "files": files,
        "capability": "assistant_loss_false_v1",
    }


def validate_message_lengths(cache: Path, chat: Path) -> int:
    expected_keys: list[tuple[int, int]] = []
    with chat.open(encoding="utf-8") as source:
        conversation_index = 0
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            conversation = json.loads(line)
            messages = conversation.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"chat messages must be a non-empty list at {chat}:{line_number}"
                )
            expected_keys.extend(
                (conversation_index, offset) for offset in range(len(messages))
            )
            conversation_index += 1

    observed_keys: list[tuple[int, int]] = []
    with cache.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"blank cache row at {cache}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {
                "conv_idx",
                "msg_offset",
                "measurement",
            }:
                raise ValueError(f"invalid cache row at {cache}:{line_number}")
            indexes = (row["conv_idx"], row["msg_offset"])
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in indexes
            ):
                raise ValueError(f"invalid cache key at {cache}:{line_number}")
            measurement = row["measurement"]
            fields = {
                "length",
                "vision_tokens",
                "vision_patches",
                "num_images",
                "image_grid_thw",
            }
            if not isinstance(measurement, dict) or set(measurement) != fields:
                raise ValueError(f"invalid measurement at {cache}:{line_number}")
            length = measurement["length"]
            counters = [
                measurement["vision_tokens"],
                measurement["vision_patches"],
                measurement["num_images"],
            ]
            if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
                raise ValueError(f"invalid measured length at {cache}:{line_number}")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counters
            ):
                raise ValueError(f"invalid vision counters at {cache}:{line_number}")
            grid = measurement["image_grid_thw"]
            if not isinstance(grid, list) or len(grid) != measurement["num_images"]:
                raise ValueError(f"invalid image grid at {cache}:{line_number}")
            if any(
                not isinstance(shape, list)
                or len(shape) != 3
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in shape
                )
                for shape in grid
            ):
                raise ValueError(f"invalid image grid at {cache}:{line_number}")
            patches = sum(t * height * width for t, height, width in grid)
            if measurement["vision_patches"] != patches:
                raise ValueError(
                    f"vision patch count mismatch at {cache}:{line_number}"
                )
            if not grid and any(counters):
                raise ValueError(
                    f"empty image grid has nonzero vision counters at {cache}:{line_number}"
                )
            observed_keys.append(indexes)
    if observed_keys != expected_keys:
        mismatch = next(
            (
                (expected, observed)
                for expected, observed in zip_longest(expected_keys, observed_keys)
                if expected != observed
            ),
            None,
        )
        raise ValueError(
            f"message-length cache keys do not match {chat}: "
            f"expected {len(expected_keys)}, got {len(observed_keys)}, "
            f"first mismatch {mismatch}"
        )
    return len(observed_keys)


def validate_record_dataset(
    output_dir: Path,
    *,
    source_chat: Path,
    processor_snapshot: dict[str, Any],
    max_length: int,
    split: str,
    val_fraction: float,
) -> dict[str, Any]:
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_fields = {
        "inline_records",
        "source_chat_path",
        "max_length",
        "overflow_mode",
        "split",
        "val_fraction",
        "profile_metadata",
        "version",
        "num_records",
        "num_shards",
        "shard_paths",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_fields:
        raise ValueError(f"invalid Omegalax metadata contract: {metadata_path}")
    fixed = {
        "inline_records": True,
        "source_chat_path": str(source_chat.resolve()),
        "max_length": max_length,
        "overflow_mode": "split",
        "split": split,
        "val_fraction": val_fraction,
        "profile_metadata": {
            "model_id": processor_snapshot["path"],
            "tokenizer": processor_snapshot["path"],
            "processor": processor_snapshot["path"],
            "preprocessor_config": None,
        },
        "version": 1,
    }
    if {key: metadata.get(key) for key in fixed} != fixed:
        raise ValueError(
            f"Omegalax metadata values do not match Stage06: {metadata_path}"
        )
    counts = (metadata["num_records"], metadata["num_shards"])
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in counts
    ):
        raise ValueError(f"Omegalax metadata counts must be positive: {metadata_path}")
    expected_shards = [
        f"part-{index:05d}.array_record" for index in range(metadata["num_shards"])
    ]
    if metadata["shard_paths"] != expected_shards:
        raise ValueError(
            f"Omegalax metadata shard sequence is invalid: {metadata_path}"
        )
    observed_shards = sorted(path.name for path in output_dir.glob("*.array_record"))
    if observed_shards != expected_shards:
        raise ValueError(f"Omegalax shard set does not match metadata: {output_dir}")

    from array_record.python.array_record_module import ArrayRecordReader

    shards: dict[str, dict[str, Any]] = {}
    total_records = 0
    for name in expected_shards:
        path = output_dir / name
        reader = ArrayRecordReader(str(path))
        if not reader.ok():
            raise ValueError(f"invalid Omegalax ArrayRecord: {path}")
        try:
            num_records = reader.num_records()
        finally:
            reader.close()
        if num_records <= 0:
            raise ValueError(f"empty Omegalax ArrayRecord: {path}")
        total_records += num_records
        shards[name] = {"sha256": _sha256(path), "num_records": num_records}
    if total_records != metadata["num_records"]:
        raise ValueError(
            f"Omegalax record count does not match metadata: {metadata_path}"
        )
    return {
        "metadata_sha256": _sha256(metadata_path),
        "num_records": total_records,
        "shards": shards,
    }
