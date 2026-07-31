"""Small deterministic IO helpers; deliberately dependency-free."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


class ContractError(ValueError):
    """A fail-closed Stage-5 contract violation."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def require_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError(f"{field} must be a lowercase SHA-256 hex digest")


def require_nonempty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write_text(path, "".join(canonical_json(dict(row)) + "\n" for row in rows))


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        value = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON object {p}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{p} must contain a JSON object")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        with p.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ContractError(f"{p}:{line_no} must be a JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSONL {p}: {exc}") from exc
    return rows
