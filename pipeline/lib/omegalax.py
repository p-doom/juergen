"""Attest the Omegalax data compiler consumed by the two SFT streams."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
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
_RECORD_ENCODER_VERIFICATION = """
import json
import sys
from pathlib import Path

from array_record.python.array_record_module import ArrayRecordReader
from transformers import AutoImageProcessor, AutoTokenizer

from omegalax.data.qwen3_encoding import encode_qwen_messages

snapshot = sys.argv[1]
max_length = int(sys.argv[2])
datasets = json.loads(sys.argv[3])
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
processor = AutoImageProcessor.from_pretrained(
    snapshot, local_files_only=True, use_fast=False
)
observed = {}
for split, root_value in datasets.items():
    root = Path(root_value)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    count = 0
    for name in metadata["shard_paths"]:
        reader = ArrayRecordReader(str(root / name))
        try:
            for _ in range(reader.num_records()):
                record = json.loads(reader.read().decode("utf-8"))
                encoded = encode_qwen_messages(
                    record["messages"],
                    tokenizer=tokenizer,
                    image_processor=processor,
                    include_pixels=False,
                )
                length = len(encoded["input_ids"])
                supervised = int(encoded["loss_mask"].sum())
                if (
                    length != record["_omegalax_measured_length"]
                    or length > max_length
                    or supervised <= 0
                ):
                    raise ValueError(
                        f"record encoding mismatch in {root / name}: "
                        f"length={length}, measured="
                        f"{record['_omegalax_measured_length']}, loss={supervised}"
                    )
                count += 1
        finally:
            reader.close()
    observed[split] = count
print(json.dumps(observed, sort_keys=True))
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
    observed = {path.name for path in resolved.iterdir()}
    missing = _SNAPSHOT_FILES - observed
    paths = [resolved / name for name in sorted(_SNAPSHOT_FILES)]
    if missing or any(not path.is_file() for path in paths):
        raise ValueError(
            "processor snapshot files do not match the Qwen3-VL contract: "
            f"missing={sorted(missing)}"
        )
    relative = {str(path.relative_to(resolved)): path for path in paths}
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
    return {
        "path": str(root),
        "commit": head,
        "tree": tree,
        "files": files,
    }


def validate_message_lengths(cache: Path, chat: Path, *, merge_size: int) -> int:
    if merge_size <= 0:
        raise ValueError("processor merge_size must be positive")
    expected = _message_keys(chat)
    expected_count = 0
    observed_count = 0
    with cache.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"blank cache row at {cache}:{line_number}")
            row = json.loads(line)
            indexes = validate_message_length_row(row, cache, line_number, merge_size)
            expected_key = next(expected, None)
            if indexes != expected_key:
                raise ValueError(
                    f"message-length cache keys do not match {chat}: first mismatch "
                    f"({expected_key}, {indexes}) after {observed_count} rows"
                )
            expected_count += 1
            observed_count += 1
    remaining = next(expected, None)
    if remaining is not None:
        expected_count += 1 + sum(1 for _ in expected)
        raise ValueError(
            f"message-length cache keys do not match {chat}: expected "
            f"{expected_count}, got {observed_count}, first mismatch "
            f"({remaining}, None)"
        )
    return observed_count


def validate_message_length_row(row, cache: Path, line_number: int, merge_size: int):
    if not isinstance(row, dict) or set(row) != {
        "conv_idx",
        "msg_offset",
        "measurement",
    }:
        raise ValueError(f"invalid cache row at {cache}:{line_number}")
    key = row["conv_idx"], row["msg_offset"]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in key
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
    if measurement["vision_patches"] != sum(
        t * height * width for t, height, width in grid
    ):
        raise ValueError(f"vision patch count mismatch at {cache}:{line_number}")
    if any(height % merge_size or width % merge_size for _time, height, width in grid):
        raise ValueError(
            f"image grid is not divisible by merge_size at {cache}:{line_number}"
        )
    vision_tokens = sum(
        time * (height // merge_size) * (width // merge_size)
        for time, height, width in grid
    )
    if measurement["vision_tokens"] != vision_tokens:
        raise ValueError(f"vision token count mismatch at {cache}:{line_number}")
    if vision_tokens > length:
        raise ValueError(f"vision token count exceeds length at {cache}:{line_number}")
    if not grid and any(counters):
        raise ValueError(
            f"empty image grid has nonzero vision counters at {cache}:{line_number}"
        )
    return key


def _message_keys(chat: Path):
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
            for offset in range(len(messages)):
                yield conversation_index, offset
            conversation_index += 1


def message_length_map(cache: Path) -> dict[tuple[int, int], int]:
    lengths: dict[tuple[int, int], int] = {}
    with cache.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            key = (row["conv_idx"], row["msg_offset"])
            if key in lengths:
                raise ValueError(f"duplicate message-length cache key: {key}")
            lengths[key] = row["measurement"]["length"]
    return lengths


def require_conversations_fit(cache: Path, chat: Path, *, max_length: int) -> None:
    lengths = message_length_map(cache)
    oversized = []
    conversation_index = 0
    with chat.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            length = sum(
                lengths[(conversation_index, offset)] for offset in range(len(messages))
            )
            if length > max_length:
                oversized.append((line_number, length))
            conversation_index += 1
    if oversized:
        raise ValueError(
            f"source conversations exceed max_length={max_length}: {oversized[:5]}"
        )


def _recording_split(recording_id: str, val_fraction: float) -> str:
    if val_fraction <= 0.0 or not recording_id:
        return "train"
    bucket = int(hashlib.sha1(recording_id.encode()).hexdigest(), 16) % 1000
    return "val" if bucket < round(val_fraction * 1000) else "train"


def _supervised(message: dict[str, Any]) -> bool:
    return message.get("role") == "assistant" and message.get("loss", True) is True


def _expected_records(
    source_chat: Path,
    lengths: dict[tuple[int, int], int],
    *,
    max_length: int,
    split: str,
    val_fraction: float,
):
    conversation_index = 0
    with source_chat.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]
            recording_id = row.get("recording_id")
            if not isinstance(recording_id, str) or not recording_id:
                raise ValueError(
                    f"source chat has no recording_id at {source_chat}:{line_number}"
                )
            if _recording_split(recording_id, val_fraction) != split:
                conversation_index += 1
                continue
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"messages", "session_id"}
            }
            session_id = f"{source_chat.stem}-{line_number:09d}"
            measured_length = sum(
                lengths[(conversation_index, offset)] for offset in range(len(messages))
            )
            if measured_length > max_length:
                raise ValueError(
                    f"source conversation exceeds max_length at "
                    f"{source_chat}:{line_number}"
                )
            if not any(_supervised(message) for message in messages):
                raise ValueError(
                    f"source conversation has no supervised target at "
                    f"{source_chat}:{line_number}"
                )
            yield {
                **metadata,
                "messages": messages,
                "_omegalax_session_id": session_id,
                "_omegalax_measured_length": measured_length,
            }
            conversation_index += 1


def discard_compiler_diagnostics(output_dir: Path) -> None:
    entries = {path.name: path for path in output_dir.iterdir()}
    shards = {name for name in entries if name.endswith(".array_record")}
    expected = {"metadata.json", *shards, *_COMPILED_SIDECARS}
    if (
        set(entries) != expected
        or not shards
        or any(not path.is_file() for path in entries.values())
    ):
        raise ValueError(f"Omegalax output set is invalid: {output_dir}")
    for name in _COMPILED_SIDECARS:
        entries[name].unlink()


def verify_record_encodings(
    omegalax_root: Path,
    processor_snapshot: dict[str, Any],
    datasets: dict[str, Path],
    *,
    max_length: int,
    expected_counts: dict[str, int],
) -> None:
    serialized = json.dumps(
        {name: str(path.resolve()) for name, path in datasets.items()},
        sort_keys=True,
    )
    result = subprocess.run(
        omegalax_python(
            omegalax_root,
            "-c",
            _RECORD_ENCODER_VERIFICATION,
            processor_snapshot["path"],
            str(max_length),
            serialized,
        ),
        cwd=omegalax_root.resolve(),
        env=isolated_subprocess_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        observed = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Omegalax record verification returned invalid output"
        ) from exc
    if observed != expected_counts:
        raise ValueError(
            f"Omegalax record verification count mismatch: "
            f"expected {expected_counts}, got {observed}"
        )


def validate_record_dataset(
    output_dir: Path,
    *,
    source_chat: Path,
    processor_snapshot: dict[str, Any],
    max_length: int,
    split: str,
    val_fraction: float,
    message_lengths: Path,
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
    expected_entries = {"metadata.json", *expected_shards}
    entries = {path.name: path for path in output_dir.iterdir()}
    if set(entries) != expected_entries or any(
        not path.is_file() for path in entries.values()
    ):
        raise ValueError(f"Omegalax output set does not match metadata: {output_dir}")
    lengths = message_length_map(message_lengths)
    expected_records = iter(
        _expected_records(
            source_chat,
            lengths,
            max_length=max_length,
            split=split,
            val_fraction=val_fraction,
        )
    )
    missing = object()

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
                expected = next(expected_records, missing)
                if expected is missing or record != expected:
                    raise ValueError(
                        f"Omegalax record {index} does not match the expected "
                        f"source chunk in {path}"
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
    if next(expected_records, missing) is not missing:
        raise ValueError("Omegalax records omit expected source chunks")
    return {
        "metadata_sha256": _sha256(metadata_path),
        "num_records": total_records,
        "shards": shards,
    }
