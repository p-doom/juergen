"""Attest the Omegalax data compiler consumed by the two SFT streams."""

from __future__ import annotations

import contextlib
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
_SNAPSHOT_FILES = {
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
}
_LOSS_MASK_PROBE = """
import json
import sys
from transformers import AutoImageProcessor, AutoTokenizer
from omegalax.data.collator_qwen3 import VLMSFTCollator

snapshot = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
processor = AutoImageProcessor.from_pretrained(
    snapshot, local_files_only=True, use_fast=False
)
collator = VLMSFTCollator(tokenizer, 32, processor)
totals = []
for message in (
    {"role": "assistant", "content": "x", "loss": False},
    {"role": "assistant", "content": "x"},
):
    batch = collator([{"messages": [message]}])
    totals.append(int(batch["loss_mask_BT"].sum()))
print(json.dumps(totals))
""".strip()
_COMPILED_SIDECARS = {
    "sequence_lengths.jsonl",
    "token_stats.json",
    "truncation_stats.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def isolated_subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "UV_NO_SYNC",
        "UV_PROJECT_ENVIRONMENT",
    ):
        environment.pop(name, None)
    environment.update(
        HF_HUB_OFFLINE="1",
        PYTHONNOUSERSITE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    return environment


def omegalax_python(root: Path, *arguments: str) -> list[str]:
    return [
        "uv",
        "run",
        "--offline",
        "--locked",
        "--project",
        str(root.resolve()),
        "python",
        "-I",
        *arguments,
    ]


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
    paths = sorted(path for path in resolved.iterdir() if path.is_file())
    relative = {str(path.relative_to(resolved)): path for path in paths}
    observed = set(relative)
    if observed != _SNAPSHOT_FILES or any(path.is_dir() for path in resolved.iterdir()):
        raise ValueError(
            "processor snapshot files do not match the Qwen3-VL contract: "
            f"missing={sorted(_SNAPSHOT_FILES - observed)}, "
            f"unexpected={sorted(observed - _SNAPSHOT_FILES)}"
        )
    preprocessor = json.loads(relative["preprocessor_config.json"].read_text())
    if not isinstance(preprocessor, dict) or preprocessor.get("merge_size") != 2:
        raise ValueError("processor snapshot must declare merge_size=2")
    files = {name: _sha256(item) for name, item in relative.items()}
    return {
        "path": str(resolved),
        "revision": resolved.name,
        "merge_size": 2,
        "files": files,
    }


def attest_omegalax(root: Path, processor_snapshot: dict[str, Any]) -> dict[str, Any]:
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
        raise ValueError(f"Omegalax compiler checkout has consumed changes:\n{status}")
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
    probe = subprocess.run(
        omegalax_python(
            root,
            "-c",
            _LOSS_MASK_PROBE,
            processor_snapshot["path"],
        ),
        cwd=root,
        env=isolated_subprocess_environment(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if probe != "[0, 2]":
        raise ValueError(
            "Omegalax VLM collator must exclude loss:false history and supervise targets"
        )
    return {
        "path": str(root),
        "commit": head,
        "tree": tree,
        "files": files,
        "capability": "vlm_sft_collator_loss_false_v1",
    }


def validate_message_lengths(cache: Path, chat: Path, *, merge_size: int) -> int:
    if merge_size <= 0:
        raise ValueError("processor merge_size must be positive")
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
            if any(
                height % merge_size or width % merge_size
                for _time, height, width in grid
            ):
                raise ValueError(
                    f"image grid is not divisible by merge_size at {cache}:{line_number}"
                )
            vision_tokens = sum(
                time * (height // merge_size) * (width // merge_size)
                for time, height, width in grid
            )
            if measurement["vision_tokens"] != vision_tokens:
                raise ValueError(
                    f"vision token count mismatch at {cache}:{line_number}"
                )
            if vision_tokens > length:
                raise ValueError(
                    f"vision token count exceeds length at {cache}:{line_number}"
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
    expected_entries = {"metadata.json", *expected_shards, *_COMPILED_SIDECARS}
    entries = {path.name: path for path in output_dir.iterdir()}
    if set(entries) != expected_entries or any(
        not path.is_file() for path in entries.values()
    ):
        raise ValueError(f"Omegalax output set does not match metadata: {output_dir}")
    for name in _COMPILED_SIDECARS:
        sidecar = entries[name]
        if sidecar.stat().st_size == 0:
            raise ValueError(f"empty Omegalax sidecar: {sidecar}")
        sidecar.unlink()

    source_sessions: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    with source_chat.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]
            session_id = f"{source_chat.stem}-{line_number:09d}"
            source_sessions[session_id] = (
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"messages", "session_id"}
                },
                messages,
            )

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
            if num_records <= 0:
                raise ValueError(f"empty Omegalax ArrayRecord: {path}")
            for index in range(num_records):
                try:
                    payload = reader.read()
                    record = json.loads(payload.decode("utf-8"))
                except Exception as exc:
                    raise ValueError(
                        f"cannot read Omegalax record {index} from {path}"
                    ) from exc
                if not isinstance(record, dict) or payload != json.dumps(
                    record, sort_keys=True
                ).encode("utf-8"):
                    raise ValueError(f"noncanonical Omegalax record {index} in {path}")
                session_id = record.get("_omegalax_session_id")
                measured = record.get("_omegalax_measured_length")
                messages = record.get("messages")
                if (
                    not isinstance(session_id, str)
                    or session_id not in source_sessions
                    or isinstance(measured, bool)
                    or not isinstance(measured, int)
                    or not 0 < measured <= max_length
                    or not isinstance(messages, list)
                    or not messages
                    or not any(
                        isinstance(message, dict)
                        and message.get("role") == "assistant"
                        and message.get("loss", True) is True
                        for message in messages
                    )
                ):
                    raise ValueError(f"invalid Omegalax record {index} in {path}")
                expected_metadata, source_messages = source_sessions[session_id]
                observed_metadata = {
                    key: value
                    for key, value in record.items()
                    if key
                    not in {
                        "_omegalax_measured_length",
                        "_omegalax_session_id",
                        "messages",
                    }
                }
                if observed_metadata != expected_metadata or not any(
                    messages == source_messages[start : start + len(messages)]
                    for start in range(len(source_messages) - len(messages) + 1)
                ):
                    raise ValueError(
                        f"Omegalax record {index} does not match source chat"
                    )
            try:
                reader.read()
            except IndexError:
                pass
            else:
                raise ValueError(f"Omegalax ArrayRecord has uncounted records: {path}")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"cannot read Omegalax ArrayRecord: {path}") from exc
        finally:
            with contextlib.suppress(Exception):
                reader.close()
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
