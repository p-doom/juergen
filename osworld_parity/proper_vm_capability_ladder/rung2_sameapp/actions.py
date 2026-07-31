from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..rung1.executor import parse_compact_raw


ACTION_SCHEMAS = ("native_absolute_sequence_v1", "compact_raw_phaseb_v1")
OPERATION_KINDS = (
    "click",
    "mouse_down",
    "mouse_move",
    "mouse_up",
    "scroll",
    "key_chord",
    "type",
)


@dataclass(frozen=True)
class SymbolicOperation:
    kind: str
    target: str | None = None
    button: str | None = None
    keys: tuple[str, ...] = ()
    text: str | None = None
    clicks: int = 0

    def __post_init__(self) -> None:
        if self.kind not in OPERATION_KINDS:
            raise ValueError(f"unsupported same-app operation: {self.kind}")


@dataclass(frozen=True)
class ActionTurn:
    semantic_step: int
    operations: tuple[SymbolicOperation, ...]

    def __post_init__(self) -> None:
        if self.semantic_step < 1 or not self.operations:
            raise ValueError("action turns require a semantic step and operations")


@dataclass(frozen=True)
class ScriptedTrajectory:
    fixture_id: str
    near_miss: bool
    turns: tuple[ActionTurn, ...]


def compile_native(
    turn: ActionTurn, geometry: dict[str, tuple[int, int]]
) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    for operation in turn.operations:
        value: dict[str, Any] = {"action": operation.kind}
        if operation.target is not None:
            if operation.target not in geometry:
                raise ValueError(f"missing geometry target: {operation.target}")
            value["coordinate"] = list(geometry[operation.target])
        if operation.button is not None:
            value["button"] = operation.button
        if operation.keys:
            value["keys"] = list(operation.keys)
        if operation.text is not None:
            value["text"] = operation.text
        if operation.kind == "scroll":
            value["clicks"] = operation.clicks
        operations.append(value)
    payload = {
        "schema": "native_absolute_sequence_v1",
        "semantic_step": turn.semantic_step,
        "operations": operations,
    }
    validate_native(payload)
    return payload


def validate_native(payload: dict[str, Any]) -> None:
    if payload.get("schema") != "native_absolute_sequence_v1":
        raise ValueError("native schema mismatch")
    operations = payload.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("native action requires a non-empty operations list")
    for operation in operations:
        if not isinstance(operation, dict) or operation.get("action") not in OPERATION_KINDS:
            raise ValueError("invalid native operation")
        coordinate = operation.get("coordinate")
        if coordinate is not None and (
            not isinstance(coordinate, list)
            or len(coordinate) != 2
            or not all(isinstance(value, int) for value in coordinate)
        ):
            raise ValueError("native coordinate must be [int, int]")
        if operation["action"] == "type" and not isinstance(operation.get("text"), str):
            raise ValueError("native type requires text")
        if operation["action"] == "key_chord" and not operation.get("keys"):
            raise ValueError("native key chord requires keys")


def compile_compact(
    turn: ActionTurn,
    geometry: dict[str, tuple[int, int]],
    cursor: tuple[int, int],
) -> tuple[str, tuple[int, int]]:
    dx = dy = scroll = 0
    tail: list[str] = []
    target_cursor = cursor
    moved = False
    for operation in turn.operations:
        if operation.target is not None:
            if operation.target not in geometry:
                raise ValueError(f"missing geometry target: {operation.target}")
            target = geometry[operation.target]
            if not moved:
                dx, dy = target[0] - cursor[0], target[1] - cursor[1]
                target_cursor = target
                moved = True
            elif target != target_cursor:
                raise ValueError("one compact action turn may move to only one target")
        if operation.kind == "click":
            tail.extend(("+LMB", "-LMB"))
        elif operation.kind == "mouse_down":
            tail.append("+" + _button_token(operation.button))
        elif operation.kind == "mouse_up":
            tail.append("-" + _button_token(operation.button))
        elif operation.kind == "scroll":
            scroll += operation.clicks
        elif operation.kind == "key_chord":
            tail.extend("+" + key for key in operation.keys)
            tail.extend("-" + key for key in reversed(operation.keys))
        elif operation.kind == "type":
            tail.append(f"type({json.dumps(operation.text, ensure_ascii=False)})")
        elif operation.kind == "mouse_move":
            pass
    text = f"{dx} {dy} {scroll}"
    if tail:
        text += "; " + " ".join(tail)
    parse_compact_raw(text)
    return text, target_cursor


def _button_token(button: str | None) -> str:
    try:
        return {"left": "LMB", "right": "RMB", "middle": "MMB"}[button or "left"]
    except KeyError as exc:
        raise ValueError(f"unsupported mouse button: {button!r}") from exc
