"""Shared helpers for trajectory extraction and SFT assembly."""

from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgpack

UNKNOWN_RE = re.compile(r"^Unknown\((-?\d+)\)$")
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


ACTION_EVENT_KINDS = frozenset({"move", "scroll", "press", "release"})
TYPING_KEY_FRAGMENTS = (
    "Key",
    "Return",
    "Backspace",
    "Space",
    "Enter",
    "Digit",
    "Num",
    "Minus",
    "Slash",
    "Period",
    "Comma",
)


@dataclass
class ActionStats:
    n_events: int = 0
    n_mousemove: int = 0
    n_scroll: int = 0
    n_keypress: int = 0
    n_keyrelease: int = 0
    n_mousepress: int = 0
    n_mouserelease: int = 0
    n_dangling_release: int = 0
    n_held_at_end: int = 0
    max_simultaneous_keys: int = 0


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSONL at {path}:{line_num}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def resolve_key_name(payload: Any) -> str | None:
    if not isinstance(payload, list) or len(payload) < 2:
        return None
    name = str(payload[1])
    if name.startswith("Unknown("):
        match = UNKNOWN_RE.match(name)
        if match is None:
            return None
        raw_code = int(match.group(1))
        return MACOS_UNKNOWN_NAME_BY_CODE.get(raw_code, f"KC_{raw_code}")
    return name


def resolve_button_name(payload: Any) -> str | None:
    if not isinstance(payload, list) or len(payload) < 1:
        return None
    button = payload[0]
    if isinstance(button, str):
        return {"Left": "LMB", "Right": "RMB", "Middle": "MMB"}.get(button, f"M_{button}")
    if isinstance(button, dict):
        for key, value in button.items():
            return f"M_{key}_{value}"
    return None


def action_bin_to_dict(action_bin: ActionBin) -> dict[str, Any]:
    return {
        "move_dx": action_bin.move_dx,
        "move_dy": action_bin.move_dy,
        "scroll": action_bin.scroll,
        "events": [[sign, name] for sign, name in action_bin.events],
    }


def action_bin_from_dict(value: dict[str, Any]) -> ActionBin:
    return ActionBin(
        move_dx=float(value["move_dx"]),
        move_dy=float(value["move_dy"]),
        scroll=float(value["scroll"]),
        events=[(str(item[0]), str(item[1])) for item in value["events"]],
    )


def format_action(action_bin: ActionBin) -> str:
    dx = round(action_bin.move_dx)
    dy = round(action_bin.move_dy)
    scroll = round(action_bin.scroll)
    if dx == 0 and dy == 0 and scroll == 0 and not action_bin.events:
        return "NO_OP"
    parts = [f"{dx} {dy} {scroll}"]
    if action_bin.events:
        parts.append(" ".join(f"{sign}{name}" for sign, name in action_bin.events))
    return " ; ".join(parts)


def is_noop_action_bin(action_bin: ActionBin) -> bool:
    return (
        round(action_bin.move_dx) == 0
        and round(action_bin.move_dy) == 0
        and round(action_bin.scroll) == 0
        and not action_bin.events
    )


def load_keylog_entries(keylog_path: Path) -> list[Any]:
    if not keylog_path.exists() or keylog_path.stat().st_size == 0:
        return []
    entries = msgpack.unpackb(keylog_path.read_bytes(), raw=False, strict_map_key=False)
    if not isinstance(entries, list):
        return []
    return entries


def keylog_summary(keylog_path: Path) -> dict[str, Any]:
    entries = load_keylog_entries(keylog_path)
    max_ts = 0
    event_counts: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        with suppress(TypeError, ValueError):
            max_ts = max(max_ts, int(entry[0]))
        ev = entry[1]
        if isinstance(ev, list) and ev:
            event_counts[str(ev[0])] = event_counts.get(str(ev[0]), 0) + 1
    return {
        "keylog_path": str(keylog_path),
        "keylog_exists": keylog_path.exists(),
        "n_keylog_events": len(entries),
        "keylog_duration_s": round(max_ts / 1_000_000, 6) if entries else 0.0,
        "event_counts": event_counts,
    }


def normalize_keylog_events(
    keylog_path: Path,
    *,
    recording_id: str,
    segment_id: str,
    segment_idx: int,
    segment_offset_s: float,
) -> tuple[list[dict[str, Any]], ActionStats]:
    """Normalize a raw keylog without binning or rendering its actions."""

    stats = ActionStats()
    held: set[str] = set()
    normalized: list[dict[str, Any]] = []

    for source_event_idx, entry in enumerate(load_keylog_entries(keylog_path)):
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        timestamp, event = entry[0], entry[1]
        if not isinstance(event, list) or not event:
            continue
        try:
            timestamp_us = int(timestamp)
        except (TypeError, ValueError):
            continue

        event_type = str(event[0])
        payload = event[1] if len(event) > 1 else None
        stats.n_events += 1
        base = {
            "recording_id": recording_id,
            "segment_id": segment_id,
            "segment_idx": segment_idx,
            "source_event_idx": source_event_idx,
            "timestamp_us": timestamp_us,
            "local_time_s": round(timestamp_us / 1_000_000, 6),
            "global_time_s": round(segment_offset_s + timestamp_us / 1_000_000, 6),
            "source_type": event_type,
            "source_payload": payload,
        }

        if event_type == "MouseMove":
            stats.n_mousemove += 1
            if isinstance(payload, list) and len(payload) >= 2:
                normalized.append(
                    {**base, "kind": "move", "dx": float(payload[0]), "dy": float(payload[1])}
                )
        elif event_type == "MouseScroll":
            stats.n_scroll += 1
            if isinstance(payload, list) and len(payload) >= 2:
                normalized.append(
                    {**base, "kind": "scroll", "dx": float(payload[0]), "dy": float(payload[1])}
                )
        elif event_type in ("KeyPress", "MousePress"):
            if event_type == "KeyPress":
                stats.n_keypress += 1
                name = resolve_key_name(payload)
            else:
                stats.n_mousepress += 1
                name = resolve_button_name(payload)
            if name:
                normalized.append({**base, "kind": "press", "key": name})
                held.add(name)
                stats.max_simultaneous_keys = max(stats.max_simultaneous_keys, len(held))
        elif event_type in ("KeyRelease", "MouseRelease"):
            if event_type == "KeyRelease":
                stats.n_keyrelease += 1
                name = resolve_key_name(payload)
            else:
                stats.n_mouserelease += 1
                name = resolve_button_name(payload)
            if name:
                normalized.append({**base, "kind": "release", "key": name})
                if name in held:
                    held.remove(name)
                else:
                    stats.n_dangling_release += 1
        elif event_type == "ContextChanged":
            normalized.append({**base, "kind": "context"})
        else:
            normalized.append({**base, "kind": "unknown"})

    stats.n_held_at_end = len(held)
    normalized.sort(key=lambda item: (int(item["timestamp_us"]), int(item["source_event_idx"])))
    return normalized, stats


def aggregate_event_records(
    events: Iterable[dict[str, Any]], *, held: set[str] | None = None
) -> ActionBin:
    """Project ordered event records into the current aggregate action semantics."""

    action_bin = ActionBin()
    if held is None:
        held = set()
    for event in events:
        kind = event["kind"]
        if kind == "move":
            action_bin.move_dx += float(event["dx"])
            action_bin.move_dy += float(event["dy"])
        elif kind == "scroll":
            dx = float(event["dx"])
            dy = float(event["dy"])
            action_bin.scroll += dy if dy != 0 else dx
        elif kind == "press":
            key = str(event["key"])
            if key not in held:
                action_bin.events.append(("+", key))
                held.add(key)
        elif kind == "release":
            key = str(event["key"])
            if key in held:
                action_bin.events.append(("-", key))
                held.remove(key)
    return action_bin


def bin_event_records(
    events: Iterable[dict[str, Any]], *, n_bins: int, fps: float
) -> list[ActionBin]:
    grouped: list[list[dict[str, Any]]] = [[] for _ in range(n_bins)]
    for event in events:
        if event["kind"] not in ACTION_EVENT_KINDS:
            continue
        bin_idx = int(float(event["local_time_s"]) * fps)
        if 0 <= bin_idx < n_bins:
            grouped[bin_idx].append(event)
    held: set[str] = set()
    return [aggregate_event_records(items, held=held) for items in grouped]


def event_activity(events: Iterable[dict[str, Any]]) -> str:
    action_events = [event for event in events if event["kind"] in ACTION_EVENT_KINDS]
    if not action_events:
        return "idle"
    if any(
        event["kind"] in {"press", "release"}
        and any(fragment in str(event["key"]) for fragment in TYPING_KEY_FRAGMENTS)
        for event in action_events
    ):
        return "type"
    return "other"


def events_have_submission(events: Iterable[dict[str, Any]]) -> bool:
    return any(
        event["kind"] == "press" and ("Return" in str(event["key"]) or "Enter" in str(event["key"]))
        for event in events
    )


def ceil_frames(duration_s: float, target_fps: float) -> int:
    if duration_s <= 0:
        return 0
    return math.ceil(duration_s * target_fps)


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value
