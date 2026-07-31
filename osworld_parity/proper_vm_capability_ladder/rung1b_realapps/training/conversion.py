from __future__ import annotations

import json
from typing import Any, Iterable

from ...rung1.executor import CompactRawExecutor, NativeAbsoluteExecutor
from ...rung1.transport import RecordingTransport


def native_action_to_compact(
    action: dict[str, Any], cursor: tuple[int, int]
) -> tuple[str, tuple[int, int]]:
    kind = str(action.get("action", "")).lower()
    coordinate = action.get("coordinate")
    target = cursor
    dx = dy = 0
    if coordinate is not None:
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
            raise ValueError("native coordinate must be [x, y]")
        target = (int(round(coordinate[0])), int(round(coordinate[1])))
        dx, dy = target[0] - cursor[0], target[1] - cursor[1]
    prefix = f"{dx} {dy} 0"
    if kind == "mouse_move":
        if coordinate is None:
            raise ValueError("mouse_move requires coordinate")
        return prefix, target
    if kind == "left_click":
        return prefix + "; +LMB -LMB", target
    if kind == "mouse_down":
        button = {"left": "LMB", "middle": "MMB", "right": "RMB"}[str(action.get("button", "left"))]
        return prefix + f"; +{button}", target
    if kind == "mouse_up":
        button = {"left": "LMB", "middle": "MMB", "right": "RMB"}[str(action.get("button", "left"))]
        return prefix + f"; -{button}", target
    if kind == "scroll":
        clicks = int(round(float(action.get("clicks", action.get("pixels", 0)))))
        return f"0 0 {clicks}", cursor
    if kind == "key":
        keys = action.get("keys")
        if isinstance(keys, str):
            keys = [keys]
        if not isinstance(keys, list) or not keys:
            raise ValueError("key action requires keys")
        presses = " ".join("+" + str(key) for key in keys)
        releases = " ".join("-" + str(key) for key in reversed(keys))
        return f"0 0 0; {presses} {releases}", cursor
    if kind == "type":
        text = action.get("text")
        if not isinstance(text, str):
            raise ValueError("type action requires text")
        return f"0 0 0; type({json.dumps(text, ensure_ascii=False)})", cursor
    raise ValueError(f"native teacher action cannot be converted: {kind!r}")


def convert_native_trajectory(
    actions: Iterable[dict[str, Any]], *, initial_cursor: tuple[int, int]
) -> tuple[str, ...]:
    cursor = initial_cursor
    converted: list[str] = []
    for action in actions:
        compact, cursor = native_action_to_compact(action, cursor)
        converted.append(compact)
    return tuple(converted)


def replay_signature(
    actions: Iterable[dict[str, Any]] | Iterable[str],
    *,
    arm: str,
    initial_cursor: tuple[int, int],
) -> dict[str, Any]:
    transport = RecordingTransport(cursor=initial_cursor)
    native = NativeAbsoluteExecutor(transport)
    compact = CompactRawExecutor(transport)
    for action in actions:
        if arm == "native_absolute_control":
            if not isinstance(action, dict):
                raise TypeError("native replay action must be an object")
            native.execute(action)
        else:
            if not isinstance(action, str):
                raise TypeError("compact replay action must be text")
            compact.execute(action)
    return {
        "cursor": list(transport.cursor_position()),
        "scroll_total": transport.audit.scroll_total,
        "typed_texts": list(transport.audit.typed_texts),
        "held_buttons": sorted(transport.audit.held_buttons),
        "held_keys": sorted(transport.audit.held_keys),
    }


def assert_round_trip(
    native_actions: tuple[dict[str, Any], ...], *, initial_cursor: tuple[int, int]
) -> tuple[str, ...]:
    compact = convert_native_trajectory(native_actions, initial_cursor=initial_cursor)
    native_signature = replay_signature(
        native_actions, arm="native_absolute_control", initial_cursor=initial_cursor
    )
    compact_signature = replay_signature(
        compact, arm="compact_raw_phaseb", initial_cursor=initial_cursor
    )
    if native_signature != compact_signature:
        raise ValueError(
            f"native/compact deterministic replay mismatch: {native_signature} != {compact_signature}"
        )
    return compact
