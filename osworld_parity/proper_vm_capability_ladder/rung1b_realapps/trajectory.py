from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..rung1.executor import CompactRawExecutor, DispatchResult, NativeAbsoluteExecutor
from .fixtures import Fixture


ARMS = ("native_absolute_control", "compact_raw_phaseb")


@dataclass(frozen=True)
class UiGeometry:
    editor: tuple[int, int] = (960, 540)
    scroll_surface: tuple[int, int] = (960, 540)
    drag_source: tuple[int, int] = (480, 260)
    drag_destination: tuple[int, int] = (480, 360)
    drag_decoy: tuple[int, int] = (480, 310)


@dataclass(frozen=True)
class ScriptedTrajectory:
    arm: str
    actions: tuple[dict[str, Any] | str, ...]
    action_classes: tuple[str, ...]


def _compact_move_click(cursor: tuple[int, int], target: tuple[int, int]) -> str:
    return f"{target[0] - cursor[0]} {target[1] - cursor[1]} 0; +LMB -LMB"


def _compact_move(cursor: tuple[int, int], target: tuple[int, int], suffix: str = "") -> str:
    separator = f"; {suffix}" if suffix else ""
    return f"{target[0] - cursor[0]} {target[1] - cursor[1]} 0{separator}"


def _compact_type(text: str) -> str:
    return f"0 0 0; type({json.dumps(text, ensure_ascii=False)})"


def build_trajectory(
    fixture: Fixture,
    *,
    arm: str,
    cursor: tuple[int, int],
    geometry: UiGeometry = UiGeometry(),
    near_miss: bool = False,
) -> ScriptedTrajectory:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    native = arm == "native_absolute_control"
    if fixture.template == "vscode_focus_type":
        text = str(
            fixture.near_miss["text"] if near_miss else fixture.expected["text"]
        )
        actions: tuple[dict[str, Any] | str, ...]
        if native:
            actions = (
                {"action": "left_click", "coordinate": list(geometry.editor)},
                {"action": "key", "keys": ["ControlLeft", "KeyA"]},
                {"action": "type", "text": text},
                {"action": "key", "keys": ["ControlLeft", "KeyS"]},
            )
        else:
            actions = (
                _compact_move_click(cursor, geometry.editor),
                "0 0 0; +ControlLeft +KeyA -KeyA -ControlLeft",
                _compact_type(text),
                "0 0 0; +ControlLeft +KeyS -KeyS -ControlLeft",
            )
        classes = ("click", "key_chord", "coalesced_type", "key_chord")
    elif fixture.template == "local_document_scroll":
        direction = str(fixture.near_miss["direction"] if near_miss else fixture.params["direction"])
        clicks = -7 if direction == "down" else 7
        if native:
            actions = (
                {"action": "mouse_move", "coordinate": list(geometry.scroll_surface)},
                {"action": "scroll", "clicks": clicks},
            )
        else:
            actions = (
                _compact_move(cursor, geometry.scroll_surface),
                f"0 0 {clicks}",
            )
        classes = ("mouse_move", "scroll")
    else:
        target = geometry.drag_decoy if near_miss else geometry.drag_destination
        if native:
            actions = (
                {"action": "mouse_down", "button": "left", "coordinate": list(geometry.drag_source)},
                {"action": "mouse_move", "coordinate": list(target)},
                {"action": "mouse_up", "button": "left", "coordinate": list(target)},
            )
        else:
            actions = (
                _compact_move(cursor, geometry.drag_source, "+LMB"),
                _compact_move(geometry.drag_source, target),
                "0 0 0; -LMB",
            )
        classes = ("button_hold", "mouse_move", "button_release")
    if len(actions) != fixture.horizon:
        raise AssertionError(f"trajectory/horizon mismatch for {fixture.id}")
    return ScriptedTrajectory(arm, actions, classes)


def execute_trajectory(
    trajectory: ScriptedTrajectory,
    native: NativeAbsoluteExecutor,
    compact: CompactRawExecutor,
) -> list[DispatchResult]:
    results: list[DispatchResult] = []
    for action in trajectory.actions:
        if trajectory.arm == "native_absolute_control":
            if not isinstance(action, dict):
                raise TypeError("native action was not an object")
            result = native.execute(action)
        else:
            if not isinstance(action, str):
                raise TypeError("compact action was not text")
            result = compact.execute(action)
        if result.executor_dispatch_status != "ok":
            raise RuntimeError(f"scripted dispatch failed: {result}")
        results.append(result)
    return results
