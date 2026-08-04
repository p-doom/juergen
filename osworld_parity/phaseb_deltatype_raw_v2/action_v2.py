#!/usr/bin/env python3
"""Versioned raw-deltatype-v2 parser, formatter, and ordered dispatcher."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol


_EVENT_RE = re.compile(r"^([+-])([A-Za-z_][A-Za-z0-9_]*)$")
_MOVE_RE = re.compile(r"MOVE\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


class DeltaTypeV2Error(ValueError):
    pass


@dataclass(frozen=True)
class DeltaTypeV2Action:
    dx: int = 0
    dy: int = 0
    scroll: int = 0
    elements: tuple[tuple[str, Any], ...] = ()
    no_op: bool = False
    terminate: bool = False
    fail: bool = False


def _scan_elements(segment: str) -> tuple[tuple[str, Any], ...]:
    elements: list[tuple[str, Any]] = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(segment):
        if segment[index].isspace():
            index += 1
            continue
        if segment.startswith("MOVE", index):
            match = _MOVE_RE.match(segment, index)
            if match is None:
                raise DeltaTypeV2Error(
                    f"malformed MOVE element: {segment[index:index + 30]!r}"
                )
            end = match.end()
            if end < len(segment) and not segment[end].isspace():
                raise DeltaTypeV2Error(
                    f"malformed MOVE element: {segment[index:index + 30]!r}"
                )
            elements.append(("move", (int(match[1]), int(match[2]))))
            index = end
            continue
        if segment.startswith("type(", index):
            start = index + 5
            while start < len(segment) and segment[start].isspace():
                start += 1
            if start >= len(segment) or segment[start] != '"':
                raise DeltaTypeV2Error("type(...) must wrap a JSON string")
            try:
                text, end = decoder.raw_decode(segment, start)
            except json.JSONDecodeError as exc:
                raise DeltaTypeV2Error(f"bad type() JSON string: {exc}") from exc
            close = end
            while close < len(segment) and segment[close].isspace():
                close += 1
            if close >= len(segment) or segment[close] != ")":
                raise DeltaTypeV2Error("type(...) missing closing ')'")
            elements.append(("type", text))
            index = close + 1
            continue
        end = index
        while end < len(segment) and not segment[end].isspace():
            end += 1
        token = segment[index:end]
        match = _EVENT_RE.fullmatch(token)
        if match is None:
            raise DeltaTypeV2Error(f"malformed deltatype-v2 element: {token!r}")
        elements.append(
            ("event", ("press" if match[1] == "+" else "release", match[2]))
        )
        index = end
    return tuple(elements)


def _validate_ordered_move(action: DeltaTypeV2Action) -> DeltaTypeV2Action:
    moves = [value for kind, value in action.elements if kind == "move"]
    if not moves:
        return action
    expected = (
        ("event", ("press", "LMB")),
        ("move", moves[0]),
        ("event", ("release", "LMB")),
    )
    if len(moves) != 1 or action.scroll != 0 or action.elements != expected:
        raise DeltaTypeV2Error(
            "MOVE is reserved for `initial_dx initial_dy 0 ; "
            "+LMB MOVE(drag_dx,drag_dy) -LMB`"
        )
    return action


def parse_deltatype_v2(text: str) -> DeltaTypeV2Action:
    if not isinstance(text, str):
        raise TypeError(f"parse_deltatype_v2 expects str, got {type(text)!r}")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise DeltaTypeV2Error("empty action text")
    line = lines[-1]
    if line == "NO_OP":
        return DeltaTypeV2Action(no_op=True)
    if line == "TERMINATE":
        return DeltaTypeV2Action(terminate=True)
    if line == "FAIL":
        return DeltaTypeV2Action(fail=True)
    mouse, elements = line.split(";", 1) if ";" in line else (line, "")
    tokens = mouse.strip().split()
    if len(tokens) != 3:
        raise DeltaTypeV2Error(
            f"expected exactly three mouse integers, got {tokens!r}"
        )
    try:
        dx, dy, scroll = (int(token) for token in tokens)
    except ValueError as exc:
        raise DeltaTypeV2Error(
            f"mouse tokens are not integers: {tokens!r}"
        ) from exc
    return _validate_ordered_move(
        DeltaTypeV2Action(dx, dy, scroll, _scan_elements(elements))
    )


def format_deltatype_v2(action: DeltaTypeV2Action) -> str:
    if action.no_op:
        return "NO_OP"
    if action.terminate:
        return "TERMINATE"
    if action.fail:
        return "FAIL"
    rendered: list[str] = []
    for kind, value in action.elements:
        if kind == "move":
            rendered.append(f"MOVE({value[0]},{value[1]})")
        elif kind == "type":
            rendered.append("type(" + json.dumps(value, ensure_ascii=False) + ")")
        elif kind == "event":
            transition, name = value
            rendered.append(("+" if transition == "press" else "-") + name)
        else:
            raise DeltaTypeV2Error(f"unknown element kind: {kind!r}")
    label = f"{action.dx} {action.dy} {action.scroll}"
    return label + (" ; " + " ".join(rendered) if rendered else "")


def ordered_plan(
    action: DeltaTypeV2Action,
    cursor_before: tuple[int, int],
    screen: tuple[int, int] = (1920, 1080),
) -> tuple[tuple[Any, ...], ...]:
    """Return exact abstract dispatch order, clipping every endpoint."""
    if action.no_op or action.terminate or action.fail:
        return ()
    width, height = screen
    cursor = (
        max(0, min(width - 1, cursor_before[0] + action.dx)),
        max(0, min(height - 1, cursor_before[1] + action.dy)),
    )
    commands: list[tuple[Any, ...]] = []
    if cursor != cursor_before:
        commands.append(("moveTo", *cursor))
    if action.scroll:
        commands.append(("scroll", action.scroll))
    for kind, value in action.elements:
        if kind == "event":
            commands.append((value[0], value[1]))
        elif kind == "type":
            commands.append(("type", value))
        elif kind == "move":
            cursor = (
                max(0, min(width - 1, cursor[0] + value[0])),
                max(0, min(height - 1, cursor[1] + value[1])),
            )
            commands.append(("moveTo", *cursor, 0.5))
        else:
            raise DeltaTypeV2Error(f"unknown element kind: {kind!r}")
    return tuple(commands)


class CommandClient(Protocol):
    def execute_ordered(self, command: tuple[Any, ...]) -> None: ...


def dispatch_deltatype_v2(
    client: CommandClient,
    action: DeltaTypeV2Action,
    cursor_before: tuple[int, int],
    screen: tuple[int, int] = (1920, 1080),
) -> tuple[tuple[Any, ...], ...]:
    plan = ordered_plan(action, cursor_before, screen)
    for command in plan:
        client.execute_ordered(command)
    return plan
