from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..proper_vm_capability_ladder.rung1.transport import HttpVmTransport, Operation


_KEYS = {
    "CTRL": "ControlLeft",
    "CONTROL": "ControlLeft",
    "SHIFT": "ShiftLeft",
    "ALT": "AltLeft",
    "ENTER": "Return",
    "RETURN": "Return",
    "ESC": "Escape",
    "ESCAPE": "Escape",
    "BACKSPACE": "Backspace",
    "TAB": "Tab",
    "SPACE": "Space",
    "DELETE": "Delete",
    "UP": "ArrowUp",
    "DOWN": "ArrowDown",
    "LEFT": "ArrowLeft",
    "RIGHT": "ArrowRight",
}


def _key(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("computer-use key must be a non-empty string")
    raw = value.strip()
    upper = raw.upper()
    if upper in _KEYS:
        return _KEYS[upper]
    if len(raw) == 1 and raw.isalpha():
        return f"Key{raw.upper()}"
    return raw


def _coordinate(arguments: dict[str, Any], screen: tuple[int, int], *, required: bool) -> tuple[int, int] | None:
    raw = arguments.get("coordinate")
    if raw is None:
        if required:
            raise ValueError("coordinate is required")
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("coordinate must be [x, y]")
    try:
        x, y = (int(round(float(value))) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("coordinate values must be numeric") from exc
    return max(0, min(screen[0] - 1, x)), max(0, min(screen[1] - 1, y))


def compile_native_absolute(arguments: dict[str, Any], screen: tuple[int, int]) -> tuple[Operation, ...]:
    if not isinstance(arguments, dict):
        raise TypeError("computer-use arguments must be an object")
    action = str(arguments.get("action", "")).strip().lower()
    rows: list[Operation] = []
    if action == "mouse_move":
        rows.append(Operation("move_to", _coordinate(arguments, screen, required=True)))
    elif action in {"left_click", "right_click", "middle_click", "double_click", "triple_click"}:
        coordinate = _coordinate(arguments, screen, required=False)
        if coordinate is not None:
            rows.append(Operation("move_to", coordinate))
        button = {"left_click": "left", "right_click": "right", "middle_click": "middle", "double_click": "left", "triple_click": "left"}[action]
        clicks = 3 if action == "triple_click" else 2 if action == "double_click" else 1
        for _ in range(clicks):
            rows.extend((Operation("mouse_down", (button,)), Operation("mouse_up", (button,))))
    elif action == "left_click_drag":
        coordinate = _coordinate(arguments, screen, required=True)
        rows.extend((Operation("mouse_down", ("left",)), Operation("move_to", coordinate), Operation("mouse_up", ("left",))))
    elif action == "key":
        keys = arguments.get("keys")
        if isinstance(keys, str):
            keys = [keys]
        if not isinstance(keys, list) or not keys:
            raise ValueError("key action requires a non-empty keys array")
        mapped = tuple(_key(value) for value in keys)
        rows.extend(Operation("key_down", (value,)) for value in mapped)
        rows.extend(Operation("key_up", (value,)) for value in reversed(mapped))
    elif action == "type":
        text = arguments.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("type action requires non-empty text")
        # These development cells deliberately contain printable ASCII only.
        # Terminal readline interprets Ctrl-V as quoted-insert, so the shared
        # editor clipboard primitive is not valid here. One atomic
        # pyautogui.write keeps typing coalesced without changing guest focus.
        try:
            text.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("sign-of-life terminal type must be ASCII") from exc
        rows.append(Operation("ascii_type", (text,)))
    elif action in {"scroll", "hscroll"}:
        try:
            clicks = int(round(float(arguments.get("pixels", 0))))
        except (TypeError, ValueError) as exc:
            raise ValueError("scroll pixels must be numeric") from exc
        if not clicks:
            raise ValueError("zero scroll is not an action")
        rows.append(Operation("scroll", (clicks,)))
    elif action == "wait":
        try:
            seconds = max(0.0, min(10.0, float(arguments.get("time", 1.0))))
        except (TypeError, ValueError) as exc:
            raise ValueError("wait time must be numeric") from exc
        rows.append(Operation("wait", (seconds,)))
    elif action == "terminate":
        return ()
    else:
        raise ValueError(f"unsupported native absolute action: {action!r}")
    return tuple(rows)


def execute_native_absolute(transport: HttpVmTransport, arguments: dict[str, Any]) -> dict[str, Any]:
    operations = compile_native_absolute(arguments, transport.screen_size())
    if not operations:
        return {"terminated": True, "operations": []}
    result = transport.execute_atomic(operations)
    return {"terminated": False, "operations": [asdict(operation) for operation in operations], "receipt": result.as_dict()}
