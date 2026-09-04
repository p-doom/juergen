"""Shared filesystem and Crowd-Cast keylog helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgpack

UNKNOWN_RE = re.compile(r"^Unknown\((\d+)\)$")
MACOS_UNKNOWN_NAME_BY_CODE: dict[int, str] = {
    10: "ISO_Section",
    62: "ControlRight",
    84: "Keypad2",
    86: "Keypad4",
    88: "Keypad6",
    91: "Keypad8",
    114: "Help",
    115: "Home",
    116: "PageUp",
    117: "ForwardDelete",
    119: "End",
    121: "PageDown",
}


@dataclass
class ActionBin:
    move_dx: float = 0.0
    move_dy: float = 0.0
    scroll: float = 0.0
    events: list[tuple[str, str]] = field(default_factory=list)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as source:
        for line_num, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSONL at {path}:{line_num}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row must be an object at {path}:{line_num}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def resolve_key_name(payload: Any) -> str | None:
    if not isinstance(payload, list) or len(payload) < 2:
        return None
    name = str(payload[1])
    if not name.startswith("Unknown("):
        return name
    match = UNKNOWN_RE.fullmatch(name)
    if match is None:
        return None
    raw_code = int(match.group(1))
    return MACOS_UNKNOWN_NAME_BY_CODE.get(raw_code, f"KC_{raw_code}")


def resolve_button_name(payload: Any) -> str | None:
    if not isinstance(payload, list) or not payload:
        return None
    button = payload[0]
    if isinstance(button, str):
        return {"Left": "LMB", "Right": "RMB", "Middle": "MMB"}.get(
            button, f"M_{button}"
        )
    if isinstance(button, dict) and len(button) == 1:
        key, value = next(iter(button.items()))
        return f"M_{key}_{value}"
    return None


def load_keylog_entries(keylog_path: Path) -> list[Any]:
    if not keylog_path.is_file() or keylog_path.stat().st_size == 0:
        return []
    entries = msgpack.unpackb(keylog_path.read_bytes(), raw=False, strict_map_key=False)
    if not isinstance(entries, list):
        raise TypeError(f"keylog must contain a list: {keylog_path}")
    return entries
