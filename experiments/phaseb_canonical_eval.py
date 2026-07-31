#!/usr/bin/env python3
"""Shared semantic primitive-plan canonicalizer for Phase-B action grammars."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

_RAW_DIR = Path(__file__).resolve().parent / "phaseb_deltatype_raw_v2"
if str(_RAW_DIR) not in sys.path:
    sys.path.insert(0, str(_RAW_DIR))
from action_v2 import parse_deltatype_v2  # noqa: E402


SW, SH = 1920, 1080
Plan = tuple[tuple[Any, ...], ...]


class CanonicalError(ValueError):
    pass


def _button(value: Any) -> str:
    name = str(value).lower()
    mapping = {
        "lmb": "left", "left": "left",
        "rmb": "right", "right": "right",
        "mmb": "middle", "middle": "middle",
    }
    if name not in mapping:
        raise CanonicalError(f"invalid button: {value!r}")
    return mapping[name]


def _key(value: Any) -> str:
    name = str(value).lower()
    mapping = {
        "return": "enter", "enter": "enter",
        "controlleft": "ctrl", "controlright": "ctrl",
        "ctrl": "ctrl", "control": "ctrl",
        "shiftleft": "shift", "shiftright": "shift", "shift": "shift",
        "altleft": "alt", "altright": "alt", "alt": "alt",
        "metaleft": "meta", "metaright": "meta", "meta": "meta",
        "cmd": "meta", "command": "meta",
        "backspace": "backspace", "delete": "delete", "tab": "tab",
        "esc": "escape", "escape": "escape", "space": "space",
        "home": "home", "end": "end", "pageup": "pageup",
        "pagedown": "pagedown", "arrowup": "up", "up": "up",
        "arrowdown": "down", "down": "down", "arrowleft": "left",
        "left": "left", "arrowright": "right", "right": "right",
    }
    if name in mapping:
        return mapping[name]
    if len(name) == 1 and name.isalpha():
        return name
    if len(name) == 1 and name.isdigit():
        return name
    if name.startswith("key") and len(name) == 4 and name[-1].isalpha():
        return name[-1]
    if name.startswith("digit") and len(name) == 6 and name[-1].isdigit():
        return name[-1]
    if name.startswith("f") and name[1:].isdigit():
        return name.upper()
    raise CanonicalError(f"invalid key: {value!r}")


def canonical_raw(label: str, source_sequence: str) -> Plan:
    action = parse_deltatype_v2(label)
    if action.no_op:
        return (("idle",),)
    if action.terminate:
        return (("terminate", "success"),)
    if action.fail:
        return (("terminate", "failure"),)
    plan: list[tuple[Any, ...]] = []
    if action.dx or action.dy:
        plan.append(("move_px", action.dx, action.dy))
    if action.scroll:
        plan.append(("scroll", action.scroll))
    for kind, value in action.elements:
        if kind == "event":
            transition, raw_name = value
            name = (_button(raw_name) if str(raw_name).upper() in {"LMB", "RMB", "MMB"}
                    else _key(raw_name))
            primitive = "button_down" if transition == "press" else "button_up"
            if name not in {"left", "right", "middle"}:
                primitive = "key_down" if transition == "press" else "key_up"
            plan.append((primitive, name))
        elif kind == "type":
            plan.append(("type", value))
        elif kind == "move":
            if int(value[0]) or int(value[1]):
                plan.append(("move_px", int(value[0]), int(value[1])))
        else:
            raise CanonicalError(f"unknown raw element kind: {kind!r}")
    return tuple(plan)


def _coordinate(call: dict[str, Any]) -> tuple[int, int]:
    value = call.get("coordinate")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CanonicalError(f"move_rel missing two-vector: {call}")
    numbers = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CanonicalError(f"move_rel coordinate is not numeric: {call}")
        number = float(item)
        if not math.isfinite(number) or not -999 <= number <= 999:
            raise CanonicalError(f"move_rel coordinate is non-finite/out of range: {call}")
        numbers.append(number)
    return round(numbers[0] * SW / 1000), round(numbers[1] * SH / 1000)


def _fields(call: dict[str, Any], expected: set[str]) -> None:
    if set(call) != expected:
        raise CanonicalError(
            f"invalid fields for {call.get('action')!r}: {sorted(call)} != {sorted(expected)}"
        )


def canonical_normalized(calls: Iterable[dict[str, Any]]) -> Plan:
    plan: list[tuple[Any, ...]] = []
    for call_value in calls:
        if not isinstance(call_value, dict):
            raise CanonicalError("normalized call is not an object")
        call = dict(call_value)
        action_value = call.get("action")
        if not isinstance(action_value, str) or action_value != action_value.lower():
            raise CanonicalError(f"invalid normalized action name: {action_value!r}")
        action = action_value
        if action == "move_rel":
            _fields(call, {"action", "coordinate"})
            dx, dy = _coordinate(call)
            if dx or dy:
                plan.append(("move_px", dx, dy))
        elif action == "scroll":
            _fields(call, {"action", "pixels"})
            pixels = call.get("pixels")
            if isinstance(pixels, bool) or not isinstance(pixels, int):
                raise CanonicalError(f"scroll pixels must be an integer: {call}")
            plan.append(("scroll", pixels))
        elif action == "type":
            _fields(call, {"action", "text"})
            text = call.get("text")
            if not isinstance(text, str):
                raise CanonicalError(f"type missing text: {call}")
            plan.append(("type", text))
        elif action == "wait":
            _fields(call, {"action", "time"})
            seconds = call.get("time")
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(float(seconds)) or not 0 < float(seconds) <= 60):
                raise CanonicalError(f"wait time must be finite in (0,60]: {call}")
            plan.append(("idle",))
        elif action == "terminate":
            _fields(call, {"action", "status"})
            status = call.get("status")
            if status not in {"success", "failure"}:
                raise CanonicalError(f"invalid terminate status: {call}")
            plan.append(("terminate", status))
        elif action in {"mouse_down", "mouse_up"}:
            _fields(call, {"action", "button"})
            button = _button(call.get("button", "left"))
            if action == "mouse_down":
                plan.append(("button_down", button))
            else:
                plan.append(("button_up", button))
        elif action in {"left_click", "right_click", "middle_click"}:
            _fields(call, {"action"})
            button = action.removesuffix("_click")
            plan.extend((("button_down", button), ("button_up", button)))
        elif action in {"double_click", "triple_click"}:
            _fields(call, {"action"})
            repetitions = 2 if action == "double_click" else 3
            for _ in range(repetitions):
                plan.extend((("button_down", "left"), ("button_up", "left")))
        elif action in {"key", "key_down", "key_up"}:
            _fields(call, {"action", "keys"})
            keys = call.get("keys")
            if (not isinstance(keys, list) or not keys
                    or any(not isinstance(value, str) for value in keys)):
                raise CanonicalError(f"key missing non-empty keys list: {call}")
            names = [_key(value) for value in keys]
            if len(set(names)) != len(names):
                raise CanonicalError(f"duplicate key in chord: {call}")
            if action == "key":
                plan.extend(("key_down", name) for name in names)
                plan.extend(("key_up", name) for name in reversed(names))
            elif action == "key_down":
                plan.extend(("key_down", name) for name in names)
            else:
                plan.extend(("key_up", name) for name in names)
        else:
            raise CanonicalError(f"unsupported normalized action: {call}")
    return tuple(plan)


def net_delta(plan: Plan) -> tuple[int, int]:
    return (
        sum(int(item[1]) for item in plan if item[0] == "move_px"),
        sum(int(item[2]) for item in plan if item[0] == "move_px"),
    )


def plan_hash(plans: Iterable[Plan]) -> str:
    payload = json.dumps(list(plans), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
