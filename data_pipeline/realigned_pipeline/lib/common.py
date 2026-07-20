"""Shared helpers for trajectory extraction and SFT assembly."""

from __future__ import annotations

import base64
import json
import math
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgpack

from realigned_pipeline.lib.config import SYSTEM_PROMPT

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


def normalize_dashed_argv() -> None:
    """Rewrite ``--foo_bar[=x]`` to ``--foo-bar[=x]`` in sys.argv.

    pmanager configs express entrypoint args as python identifiers (rendered
    ``--foo_bar=value``); the realigned stages declare dashed argparse flags.
    Call this before parse_args() so both spellings work. Only the flag name
    (before the first ``=``) is rewritten; values pass through untouched."""
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg.startswith("--") and "_" in arg.split("=", 1)[0]:
            key, sep, value = arg.partition("=")
            sys.argv[i] = key.replace("_", "-") + sep + value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
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


def system_message() -> dict[str, Any]:
    return {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}


def image_message_content(image_path: str | Path, text: str | None = None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image", "image": str(image_path)}]
    if text:
        content.append({"type": "text", "text": text})
    return content


def assistant_text(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def user_image(image_path: str | Path, text: str | None = None) -> dict[str, Any]:
    return {"role": "user", "content": image_message_content(image_path, text)}


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
        return {"Left": "LMB", "Right": "RMB", "Middle": "MMB"}.get(
            button, f"M_{button}"
        )
    if isinstance(button, dict):
        for key, value in button.items():
            return f"M_{key}_{value}"
    return None


def merge_action_bins(earlier: ActionBin, later: ActionBin) -> ActionBin:
    """Fold an earlier (dropped-frame) bin into the next kept bin, preserving event order."""
    return ActionBin(
        move_dx=earlier.move_dx + later.move_dx,
        move_dy=earlier.move_dy + later.move_dy,
        scroll=earlier.scroll + later.scroll,
        events=earlier.events + later.events,
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
        try:
            max_ts = max(max_ts, int(entry[0]))
        except (TypeError, ValueError):
            pass
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


def aggregate_actions(
    keylog_path: Path, n_bins: int, target_fps: float,
    timemap: Callable[[float], float] | None = None,
) -> tuple[list[ActionBin], ActionStats]:
    """Bin keylog events into ``n_bins`` per-``target_fps`` ActionBins.

    ``timemap`` optionally remaps each event's timestamp (seconds) before bucketing
    — e.g. ``realign_lib.keylog_to_video`` to bin on the realigned video clock
    instead of the raw keylog clock. ``None`` keeps the raw clock (default)."""
    stats = ActionStats()
    bins = [ActionBin() for _ in range(n_bins)]
    held: set[str] = set()

    for entry in load_keylog_entries(keylog_path):
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

        if event_type == "ContextChanged":
            continue
        t_s = timestamp_us / 1_000_000
        if timemap is not None:
            t_s = timemap(t_s)
        bucket_idx = int(t_s * target_fps)
        if bucket_idx < 0 or bucket_idx >= n_bins:
            continue
        action_bin = bins[bucket_idx]

        if event_type == "MouseMove":
            stats.n_mousemove += 1
            if isinstance(payload, list) and len(payload) >= 2:
                action_bin.move_dx += float(payload[0])
                action_bin.move_dy += float(payload[1])
        elif event_type == "MouseScroll":
            stats.n_scroll += 1
            if isinstance(payload, list) and len(payload) >= 2:
                value = payload[1] if payload[1] != 0 else payload[0]
                action_bin.scroll += float(value)
        elif event_type in ("KeyPress", "MousePress"):
            if event_type == "KeyPress":
                stats.n_keypress += 1
                name = resolve_key_name(payload)
            else:
                stats.n_mousepress += 1
                name = resolve_button_name(payload)
            if name and name not in held:
                action_bin.events.append(("+", name))
                held.add(name)
                stats.max_simultaneous_keys = max(stats.max_simultaneous_keys, len(held))
        elif event_type in ("KeyRelease", "MouseRelease"):
            if event_type == "KeyRelease":
                stats.n_keyrelease += 1
                name = resolve_key_name(payload)
            else:
                stats.n_mouserelease += 1
                name = resolve_button_name(payload)
            if name and name in held:
                action_bin.events.append(("-", name))
                held.remove(name)
            elif name:
                stats.n_dangling_release += 1

    stats.n_held_at_end = len(held)
    return bins, stats


def ceil_frames(duration_s: float, target_fps: float) -> int:
    if duration_s <= 0:
        return 0
    return int(math.ceil(duration_s * target_fps))


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


def text_token_estimate(text: str) -> int:
    return max(1, len(text) // 4)
