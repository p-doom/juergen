"""Deterministic absolute-native -> compact raw-relative action conversion.

One compact line uses the established deltatype grammar. A teacher turn may
contain several lines, which is necessary to preserve causal drag ordering:
mouse-down at the source, movement while held, then mouse-up at the target.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from experiments.teacher_sft.contracts import ContractError

_EVENT_RE = re.compile(r"^([+-])([A-Za-z_][A-Za-z_0-9]*)$")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")
_BUTTONS = {"left": "LMB", "middle": "MMB", "right": "RMB"}
_CLICK_ACTIONS = {
    "left_click": ("LMB", 1),
    "right_click": ("RMB", 1),
    "middle_click": ("MMB", 1),
    "double_click": ("LMB", 2),
    "triple_click": ("LMB", 3),
}
_MOVE_ACTIONS = {"mouse_move", "move", "move_absolute"}
_KEY_ALIASES = {
    "ctrl": "ControlLeft",
    "control": "ControlLeft",
    "shift": "ShiftLeft",
    "alt": "Alt",
    "meta": "MetaLeft",
    "win": "MetaLeft",
    "cmd": "MetaLeft",
    "enter": "Return",
    "return": "Return",
    "esc": "Escape",
    "escape": "Escape",
    "backspace": "Backspace",
    "tab": "Tab",
    "space": "Space",
    "delete": "Delete",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
}


@dataclass(frozen=True)
class Element:
    kind: str  # press | release | type
    value: str


@dataclass(frozen=True)
class CompactAction:
    dx: int = 0
    dy: int = 0
    scroll: int = 0
    elements: tuple[Element, ...] = ()
    control: str | None = None

    def render(self) -> str:
        if self.control:
            return self.control
        if self.dx == self.dy == self.scroll == 0 and not self.elements:
            return "NO_OP"
        result = f"{self.dx} {self.dy} {self.scroll}"
        if self.elements:
            rendered = []
            for element in self.elements:
                if element.kind == "type":
                    rendered.append(
                        f"type({json.dumps(element.value, ensure_ascii=False)})"
                    )
                else:
                    rendered.append(
                        ("+" if element.kind == "press" else "-") + element.value
                    )
            result += " ; " + " ".join(rendered)
        return result


def _round(value: float | Decimal) -> int:
    try:
        numeric = float(value)
        decimal = Decimal(str(value))
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ContractError(f"coordinate/value is not numeric: {value!r}") from exc
    if not math.isfinite(numeric):
        raise ContractError(f"non-finite coordinate: {value!r}")
    return int(decimal.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _point(value: Any, *, context: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ContractError(f"{context} must be [x,y], got {value!r}")
    return _round(value[0]), _round(value[1])


def resolve_absolute_target(
    action: dict[str, Any], screen_size: tuple[int, int]
) -> tuple[int, int] | None:
    coordinate = action.get("coordinate")
    if coordinate is None:
        return None
    x, y = _point(coordinate, context="teacher coordinate")
    width, height = screen_size
    coordinate_space = action.get("coordinate_space")
    if coordinate_space == "absolute_px":
        target = x, y
    elif coordinate_space == "absolute_grid":
        grid = action.get("coordinate_grid")
        if not isinstance(grid, int) or grid <= 1:
            raise ContractError(
                "absolute_grid action requires integer coordinate_grid > 1"
            )
        if not 0 <= x < grid or not 0 <= y < grid:
            raise ContractError(f"teacher grid coordinate outside [0,{grid}): {(x, y)}")
        target = _round(Decimal(x) * width / grid), _round(Decimal(y) * height / grid)
    else:
        raise ContractError(f"unknown/missing coordinate_space: {coordinate_space!r}")
    if not 0 <= target[0] < width or not 0 <= target[1] < height:
        raise ContractError(
            f"absolute target outside viewport: {target} vs {screen_size}"
        )
    return target


def _key_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"invalid key name: {value!r}")
    raw = value.strip()
    lower = raw.lower()
    if lower in _KEY_ALIASES:
        return _KEY_ALIASES[lower]
    if len(raw) == 1 and raw.isalpha():
        return f"Key{raw.upper()}"
    if len(raw) == 1 and raw.isdigit():
        return f"Num{raw}"
    if re.fullmatch(r"[Ff](?:[1-9]|1[0-9]|2[0-4])", raw):
        return raw.upper()
    if not _NAME_RE.fullmatch(raw):
        raise ContractError(f"key is not representable in compact grammar: {raw!r}")
    return raw


@dataclass
class SymbolicState:
    cursor: tuple[int, int]
    screen_size: tuple[int, int]
    held: set[str] = field(default_factory=set)
    typed_text: str = ""
    scroll_total: int = 0

    def apply(self, action: CompactAction) -> None:
        if action.control:
            return
        x = self.cursor[0] + action.dx
        y = self.cursor[1] + action.dy
        width, height = self.screen_size
        if not 0 <= x < width or not 0 <= y < height:
            raise ContractError(
                f"relative action would clip at viewport edge: {(x, y)}"
            )
        self.cursor = x, y
        self.scroll_total += action.scroll
        for element in action.elements:
            if element.kind == "type":
                if any(name in self.held for name in ("LMB", "MMB", "RMB")):
                    raise ContractError(
                        "typing while a mouse button is held is ambiguous"
                    )
                self.typed_text += element.value
            elif element.kind == "press":
                if element.value in self.held:
                    raise ContractError(f"redundant press: {element.value}")
                self.held.add(element.value)
            elif element.kind == "release":
                if element.value not in self.held:
                    raise ContractError(f"dangling release: {element.value}")
                self.held.remove(element.value)
            else:
                raise ContractError(f"unknown compact element: {element.kind}")


def _scan_elements(segment: str) -> tuple[Element, ...]:
    decoder = json.JSONDecoder()
    elements: list[Element] = []
    index = 0
    while index < len(segment):
        if segment[index].isspace():
            index += 1
            continue
        if segment.startswith("type(", index):
            start = index + 5
            while start < len(segment) and segment[start].isspace():
                start += 1
            try:
                value, end = decoder.raw_decode(segment, start)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid type() JSON string: {exc}") from exc
            if not isinstance(value, str):
                raise ContractError("type() payload must be a JSON string")
            while end < len(segment) and segment[end].isspace():
                end += 1
            if end >= len(segment) or segment[end] != ")":
                raise ContractError("type() is missing closing parenthesis")
            elements.append(Element("type", value))
            index = end + 1
            continue
        end = index
        while end < len(segment) and not segment[end].isspace():
            end += 1
        token = segment[index:end]
        match = _EVENT_RE.fullmatch(token)
        if not match:
            raise ContractError(f"malformed compact element: {token!r}")
        elements.append(
            Element("press" if match.group(1) == "+" else "release", match.group(2))
        )
        index = end
    return tuple(elements)


def parse_compact_line(line: str) -> CompactAction:
    line = line.strip()
    if line in {"NO_OP", "TERMINATE", "FAIL"}:
        return CompactAction(control=None if line == "NO_OP" else line)
    mouse, separator, elements = line.partition(";")
    parts = mouse.split()
    if len(parts) != 3:
        raise ContractError(f"compact line requires dx dy scroll: {line!r}")
    try:
        dx, dy, scroll = map(int, parts)
    except ValueError as exc:
        raise ContractError(f"compact mouse fields must be integers: {line!r}") from exc
    parsed = CompactAction(
        dx, dy, scroll, _scan_elements(elements) if separator else ()
    )
    if parsed.render() != line:
        raise ContractError(
            f"non-canonical compact line: {line!r} -> {parsed.render()!r}"
        )
    return parsed


def parse_compact_sequence(text: str) -> tuple[CompactAction, ...]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ContractError("empty compact action sequence")
    return tuple(parse_compact_line(line) for line in lines)


def _movement(
    action: dict[str, Any], before: tuple[int, int], screen: tuple[int, int]
) -> tuple[int, int, tuple[int, int]]:
    target = resolve_absolute_target(action, screen)
    if target is None:
        return 0, 0, before
    return target[0] - before[0], target[1] - before[1], target


def _trace_points(
    trace: dict[str, Any],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int] | None]:
    before = _point(trace.get("cursor_before"), context="trace.cursor_before")
    after = _point(trace.get("cursor_after"), context="trace.cursor_after")
    target_raw = trace.get("resolved_target_px")
    target = (
        _point(target_raw, context="trace.resolved_target_px")
        if target_raw is not None
        else None
    )
    return before, after, target


def _button(action: dict[str, Any]) -> str:
    value = str(action.get("button", "left")).strip().lower()
    try:
        return _BUTTONS[value]
    except KeyError as exc:
        raise ContractError(f"unsupported mouse button: {value!r}") from exc


def convert_native_action(
    action: dict[str, Any],
    trace: dict[str, Any],
    state: SymbolicState,
) -> tuple[CompactAction, ...]:
    """Convert one executed native action, proving it against VM cursor telemetry."""
    if not isinstance(action, dict) or not isinstance(trace, dict):
        raise ContractError("action and trace must be objects")
    kind = str(action.get("action", "")).strip().lower()
    trace_before, trace_after, trace_target = _trace_points(trace)
    if trace_before != state.cursor:
        raise ContractError(
            f"cursor chain break: expected {state.cursor}, trace starts {trace_before}"
        )
    dx, dy, expected_target = _movement(action, trace_before, state.screen_size)
    if action.get("coordinate") is not None:
        if trace_target != expected_target:
            raise ContractError(
                f"resolved-target mismatch: teacher={expected_target}, VM={trace_target}"
            )
        if trace_after != expected_target:
            raise ContractError(
                f"VM cursor did not land on resolved target: {trace_after}"
            )
    else:
        if trace_target is not None:
            raise ContractError("coordinate-less action has a resolved target")
        if trace_after != trace_before:
            raise ContractError(
                f"coordinate-less action moved cursor: {trace_before} -> {trace_after}"
            )

    result: tuple[CompactAction, ...]
    if kind in _MOVE_ACTIONS:
        if action.get("coordinate") is None:
            raise ContractError(f"{kind} requires an absolute coordinate")
        result = (CompactAction(dx, dy),)
    elif kind in _CLICK_ACTIONS:
        button, count = _CLICK_ACTIONS[kind]
        elements = tuple(
            element
            for _ in range(count)
            for element in (Element("press", button), Element("release", button))
        )
        result = (CompactAction(dx, dy, elements=elements),)
    elif kind == "left_click_drag":
        if action.get("coordinate") is None:
            raise ContractError("left_click_drag requires an absolute target")
        result = (
            CompactAction(elements=(Element("press", "LMB"),)),
            CompactAction(dx, dy),
            CompactAction(elements=(Element("release", "LMB"),)),
        )
    elif kind in {"mouse_down", "mouse_up"}:
        element = Element(
            "press" if kind == "mouse_down" else "release", _button(action)
        )
        result = (CompactAction(dx, dy, elements=(element,)),)
    elif kind in {"scroll", "hscroll"}:
        if action.get("coordinate") is not None:
            raise ContractError(
                "coordinate-bearing scroll is not in the compact contract"
            )
        raw = action.get("pixels", action.get("amount"))
        if isinstance(raw, bool):
            raise ContractError("scroll amount cannot be bool")
        scroll = _round(raw)
        if abs(scroll) > 100_000:
            raise ContractError(f"implausible scroll magnitude: {scroll}")
        result = (CompactAction(scroll=scroll),)
    elif kind == "type":
        text = action.get("text")
        if not isinstance(text, str) or not text:
            raise ContractError("type action requires non-empty text")
        if "\x00" in text or len(text) > 16_384:
            raise ContractError("type text is unsafe or exceeds 16384 characters")
        result = (CompactAction(elements=(Element("type", text),)),)
    elif kind in {"key", "key_down", "key_up"}:
        raw_keys = action.get("keys", action.get("key"))
        if isinstance(raw_keys, str):
            raw_keys = [raw_keys]
        if not isinstance(raw_keys, list) or not raw_keys:
            raise ContractError(f"{kind} requires a non-empty keys array")
        keys = tuple(_key_name(key) for key in raw_keys)
        if len(keys) != len(set(keys)):
            raise ContractError(f"duplicate key in chord: {keys}")
        if kind == "key":
            elements = tuple(Element("press", key) for key in keys) + tuple(
                Element("release", key) for key in reversed(keys)
            )
        else:
            elements = tuple(
                Element("press" if kind == "key_down" else "release", key)
                for key in keys
            )
        result = (CompactAction(elements=elements),)
    elif kind == "wait":
        result = (CompactAction(),)
    elif kind == "terminate":
        status = str(action.get("status", "success")).strip().lower()
        result = (
            CompactAction(control="TERMINATE" if status == "success" else "FAIL"),
        )
    else:
        raise ContractError(f"unsupported native action: {kind!r}")

    for compact in result:
        state.apply(compact)
    if state.cursor != trace_after:
        raise ContractError(
            f"symbolic replay cursor {state.cursor} != VM cursor {trace_after}"
        )
    return result
