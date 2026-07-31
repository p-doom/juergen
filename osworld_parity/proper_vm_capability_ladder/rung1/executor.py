from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .transport import InputAudit, Operation, TransportError


class GuiTransport(Protocol):
    audit: InputAudit

    def cursor_position(self) -> tuple[int, int]: ...
    def screen_size(self) -> tuple[int, int]: ...
    def move_to(self, x: int, y: int) -> None: ...
    def mouse_down(self, button: str = "left") -> None: ...
    def mouse_up(self, button: str = "left") -> None: ...
    def scroll(self, clicks: int) -> None: ...
    def key_chord(self, keys: list[str]) -> None: ...
    def coalesced_type(self, text: str) -> None: ...
    def wait(self, seconds: float) -> None: ...


@dataclass(frozen=True)
class DispatchResult:
    adapter: str
    parse_status: str
    executor_dispatch_status: str
    action_class: str
    operations: tuple[Operation, ...]


def _move_relative(transport: GuiTransport, dx: int, dy: int) -> None:
    x, y = transport.cursor_position()
    transport.move_to(x + int(dx), y + int(dy))


class NativeAbsoluteExecutor:
    name = "native_absolute_control"

    def __init__(self, transport: GuiTransport) -> None:
        self.transport = transport

    def execute(self, arguments: dict[str, Any]) -> DispatchResult:
        if not isinstance(arguments, dict):
            raise TypeError("native absolute action must be an object")
        action = str(arguments.get("action", "")).strip().lower()
        before = len(self.transport.audit.operations)

        def move_if_present(required: bool = False) -> None:
            coordinate = arguments.get("coordinate")
            if coordinate is None and not required:
                return
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                raise ValueError(f"{action} requires coordinate [x, y]")
            self.transport.move_to(int(round(coordinate[0])), int(round(coordinate[1])))

        if action == "mouse_move":
            move_if_present(required=True)
            action_class = "mouse_move"
        elif action == "left_click":
            move_if_present()
            self.transport.mouse_down("left")
            self.transport.mouse_up("left")
            action_class = "click"
        elif action == "mouse_down":
            move_if_present()
            self.transport.mouse_down(str(arguments.get("button", "left")))
            action_class = "button_hold"
        elif action == "mouse_up":
            move_if_present()
            self.transport.mouse_up(str(arguments.get("button", "left")))
            action_class = "button_release"
        elif action in {"left_click_drag", "drag_to"}:
            coordinate = arguments.get("coordinate")
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                raise ValueError(f"{action} requires coordinate [x, y]")
            self.transport.mouse_down("left")
            self.transport.move_to(int(round(coordinate[0])), int(round(coordinate[1])))
            self.transport.mouse_up("left")
            action_class = "drag"
        elif action == "scroll":
            clicks = int(round(float(arguments.get("clicks", arguments.get("pixels", 0)))))
            self.transport.scroll(clicks)
            action_class = "scroll"
        elif action == "key":
            keys = arguments.get("keys")
            if isinstance(keys, str):
                keys = [keys]
            if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
                raise ValueError("key action requires string keys")
            self.transport.key_chord(keys)
            action_class = "key_chord"
        elif action == "type":
            text = arguments.get("text")
            if not isinstance(text, str):
                raise ValueError("type action requires string text")
            self.transport.coalesced_type(text)
            action_class = "coalesced_type"
        elif action == "wait":
            self.transport.wait(float(arguments.get("time", 1.0)))
            action_class = "wait"
        else:
            raise ValueError(f"unsupported native absolute action {action!r}")
        return DispatchResult(
            adapter=self.name,
            parse_status="ok",
            executor_dispatch_status="ok",
            action_class=action_class,
            operations=tuple(self.transport.audit.operations[before:]),
        )


@dataclass(frozen=True)
class RawElement:
    kind: str
    value: str
    pressed: bool | None = None


@dataclass(frozen=True)
class CompactRawAction:
    dx: int
    dy: int
    scroll: int
    elements: tuple[RawElement, ...]


def parse_compact_raw(text: str) -> CompactRawAction:
    if not isinstance(text, str):
        raise TypeError("compact raw action must be a string")
    line = text.strip().splitlines()[0] if text.strip() else ""
    if not line:
        raise ValueError("empty compact raw action")
    mouse, separator, tail = line.partition(";")
    tokens = mouse.split()
    if len(tokens) != 3:
        raise ValueError("compact raw action requires dx dy scroll")
    try:
        dx, dy, scroll = (int(token) for token in tokens)
    except ValueError as exc:
        raise ValueError("compact raw mouse values must be integers") from exc
    elements: list[RawElement] = []
    index = 0
    decoder = json.JSONDecoder()
    tail = tail.strip() if separator else ""
    while index < len(tail):
        while index < len(tail) and tail[index].isspace():
            index += 1
        if index >= len(tail):
            break
        if tail.startswith("type(", index):
            start = index + 5
            while start < len(tail) and tail[start].isspace():
                start += 1
            value, end = decoder.raw_decode(tail, start)
            if not isinstance(value, str):
                raise ValueError("type() payload must be a JSON string")
            while end < len(tail) and tail[end].isspace():
                end += 1
            if end >= len(tail) or tail[end] != ")":
                raise ValueError("type() missing closing parenthesis")
            elements.append(RawElement("type", value))
            index = end + 1
            continue
        end = index
        while end < len(tail) and not tail[end].isspace():
            end += 1
        token = tail[index:end]
        if len(token) < 2 or token[0] not in "+-" or not token[1:].replace("_", "").isalnum():
            raise ValueError(f"invalid compact raw event {token!r}")
        elements.append(RawElement("event", token[1:], token[0] == "+"))
        index = end
    return CompactRawAction(dx, dy, scroll, tuple(elements))


class CompactRawExecutor:
    name = "compact_raw_phaseb"

    def __init__(self, transport: GuiTransport) -> None:
        self.transport = transport

    def execute(self, text: str) -> DispatchResult:
        action = parse_compact_raw(text)
        before = len(self.transport.audit.operations)
        if action.dx or action.dy:
            _move_relative(self.transport, action.dx, action.dy)
        if action.scroll:
            self.transport.scroll(action.scroll)
        classes: set[str] = set()
        if action.dx or action.dy:
            classes.add("mouse_move")
        if action.scroll:
            classes.add("scroll")
        for element in action.elements:
            if element.kind == "type":
                self.transport.coalesced_type(element.value)
                classes.add("coalesced_type")
            elif element.value in {"LMB", "RMB", "MMB"}:
                button = {"LMB": "left", "RMB": "right", "MMB": "middle"}[
                    element.value
                ]
                if element.pressed:
                    self.transport.mouse_down(button)
                    classes.add("button_hold")
                else:
                    self.transport.mouse_up(button)
                    classes.add("button_release")
            else:
                # A matched key chord is emitted as ordered presses followed by
                # releases. Buffering the full chord keeps it one atomic guest call.
                if not element.pressed:
                    continue
                releases = {
                    item.value
                    for item in action.elements
                    if item.kind == "event" and item.pressed is False
                }
                if element.value not in releases:
                    raise TransportError(f"raw key press lacks release: {element.value}")
        key_presses = [
            item.value
            for item in action.elements
            if item.kind == "event"
            and item.pressed
            and item.value not in {"LMB", "RMB", "MMB"}
        ]
        if key_presses:
            self.transport.key_chord(key_presses)
            classes.add("key_chord")
        action_class = "+".join(sorted(classes)) if classes else "no_op"
        return DispatchResult(
            adapter=self.name,
            parse_status="ok",
            executor_dispatch_status="ok",
            action_class=action_class,
            operations=tuple(self.transport.audit.operations[before:]),
        )
