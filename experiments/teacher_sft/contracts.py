"""Shared fail-closed contracts for the teacher-SFT pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from experiments.teacher_sft import SCHEMA_VERSION


class ContractError(RuntimeError):
    """An input cannot be proven to satisfy a training-data contract."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"cannot read JSONL {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ContractError(f"JSONL row is not an object at {path}:{line_number}")
        yield row


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    )
    _atomic_write(path, payload)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(path, b"".join(canonical_bytes(dict(row)) + b"\n" for row in rows))


def require_train_split(value: Any, *, context: str) -> None:
    if value != "train":
        raise ContractError(
            f"{context} must declare source_split='train', got {value!r}"
        )


def require_finite_score(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{context} must be a numeric score, not bool")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{context} is not numeric: {value!r}") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ContractError(f"{context} must be finite and in [0,1], got {score!r}")
    return score


def ensure_empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ContractError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def artifact_ref(path: Path, role: str) -> dict[str, str]:
    resolved = path.resolve()
    return {"role": role, "path": str(resolved), "sha256": file_sha256(resolved)}


def load_heldout_denylist(path: Path) -> dict[str, set[str] | str]:
    """Consume opaque heldout identities/hashes without reading heldout tasks."""
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("heldout denylist must be a schema_version=1 object")
    keys = {"task_keys", "source_task_ids", "instruction_sha256", "asset_sha256"}
    result: dict[str, set[str] | str] = {"denylist_sha256": file_sha256(path)}
    for key in keys:
        values = payload.get(key, [])
        if not isinstance(values, list) or any(
            not isinstance(item, str) for item in values
        ):
            raise ContractError(f"heldout denylist {key} must be an array of strings")
        result[key] = set(values)
    return result


def assert_not_heldout(
    *,
    denylist: Mapping[str, set[str] | str],
    task_key: str,
    source_task_id: str,
    instruction: str,
    asset_hashes: Iterable[str] = (),
) -> None:
    instruction_hash = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    checks = (
        (task_key, denylist["task_keys"], "task key"),
        (source_task_id, denylist["source_task_ids"], "source task id"),
        (instruction_hash, denylist["instruction_sha256"], "instruction hash"),
    )
    for value, forbidden, label in checks:
        if value in forbidden:
            raise ContractError(f"heldout leakage: {label} {value!r} is denylisted")
    overlap = set(asset_hashes) & denylist["asset_sha256"]
    if overlap:
        raise ContractError(
            f"heldout leakage: {len(overlap)} asset hash(es) are denylisted"
        )


def verify_declared_hash(path: Path, expected: Any, *, context: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ContractError(f"{context} has no valid declared sha256")
    actual = file_sha256(path)
    if actual != expected:
        raise ContractError(
            f"{context} hash mismatch: declared {expected}, observed {actual}"
        )
