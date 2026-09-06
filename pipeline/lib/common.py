"""Shared filesystem and Crowd-Cast keylog helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgpack
from desktop.execute.keymap import KeymapError, guest_button, guest_key

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

RECORDER_KEY_NAME_ALIASES = {
    "DownArrow": "ArrowDown",
    "LeftArrow": "ArrowLeft",
    "RightArrow": "ArrowRight",
    "UpArrow": "ArrowUp",
    "Dot": "Period",
    "LeftBracket": "BracketLeft",
    "RightBracket": "BracketRight",
}

KEYLOG_ERROR_REASONS = frozenset(
    {
        "empty_keylog",
        "invalid_action_payload",
        "invalid_event",
        "invalid_msgpack",
        "invalid_non_action_payload",
        "unexecutable_action",
        "unsupported_event_type",
    }
)
EVENT_EXCLUSION_REASONS = frozenset(
    {
        "invalid_action_payload",
        "invalid_non_action_payload",
        "unexecutable_action",
        "unsupported_event_type",
    }
)


class KeylogError(ValueError):
    def __init__(self, reason: str, message: str):
        if reason not in KEYLOG_ERROR_REASONS:
            raise ValueError(f"unknown keylog error reason: {reason!r}")
        super().__init__(message)
        self.reason = reason


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
    with path.open() as source:
        for line_num, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row at {path}:{line_num}")
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


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    write_json(temporary, value)
    temporary.replace(path)


def resolve_key_name(payload: Any) -> str | None:
    if (
        not isinstance(payload, list)
        or len(payload) != 2
        or isinstance(payload[0], bool)
        or not isinstance(payload[0], int)
        or payload[0] < 0
        or not isinstance(payload[1], str)
        or not payload[1]
    ):
        return None
    name = payload[1]
    if name.startswith("Unknown("):
        match = UNKNOWN_RE.fullmatch(name)
        if match is None:
            return None
        name = MACOS_UNKNOWN_NAME_BY_CODE.get(int(match.group(1)), "")
    name = RECORDER_KEY_NAME_ALIASES.get(name, name)
    try:
        guest_key(name)
    except KeymapError:
        return None
    return name


def resolve_button_name(payload: Any) -> str | None:
    if not isinstance(payload, list) or len(payload) != 3:
        return None
    button = payload[0]
    if not isinstance(button, str):
        return None
    name = {"Left": "LMB", "Right": "RMB", "Middle": "MMB"}.get(button)
    if name is None:
        return None
    try:
        guest_button(name)
    except KeymapError:
        return None
    return name


def load_keylog_entries(keylog_path: Path) -> list[Any]:
    if not keylog_path.is_file():
        raise FileNotFoundError(f"keylog is missing: {keylog_path}")
    if keylog_path.stat().st_size == 0:
        raise KeylogError("empty_keylog", f"keylog is empty: {keylog_path}")
    try:
        entries = msgpack.unpackb(
            keylog_path.read_bytes(), raw=False, strict_map_key=False
        )
    except (ValueError, msgpack.exceptions.UnpackException) as exc:
        raise KeylogError(
            "invalid_msgpack", f"keylog is not valid msgpack: {keylog_path}"
        ) from exc
    if not isinstance(entries, list):
        raise KeylogError(
            "invalid_event", f"keylog must contain an event list: {keylog_path}"
        )
    if not entries:
        raise KeylogError("empty_keylog", f"keylog is empty: {keylog_path}")
    previous_timestamp = -1
    for index, entry in enumerate(entries):
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or isinstance(entry[0], bool)
            or not isinstance(entry[0], int)
            or entry[0] < 0
            or entry[0] < previous_timestamp
            or not isinstance(entry[1], list)
            or len(entry[1]) != 2
            or not isinstance(entry[1][0], str)
            or not entry[1][0]
        ):
            raise KeylogError(
                "invalid_event", f"invalid keylog event at {keylog_path}:{index}"
            )
        previous_timestamp = entry[0]
    return entries
