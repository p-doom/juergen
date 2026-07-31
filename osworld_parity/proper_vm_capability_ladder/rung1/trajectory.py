from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from .fixtures import Fixture


Arm = Literal["native_absolute_control", "compact_raw_phaseb"]


@dataclass(frozen=True)
class GoldTrajectory:
    arm: Arm
    actions: tuple[dict[str, Any] | str, ...]
    observed_cursor_baseline: tuple[int, int]
    expected_endpoint: tuple[int, int] | None


def _center(geometry: dict[str, Any], name: str) -> tuple[int, int]:
    value = geometry.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"fixture geometry lacks {name!r}")
    return int(value["center_x"]), int(value["center_y"])


def _raw_move(
    start: tuple[int, int], target: tuple[int, int], suffix: str = ""
) -> str:
    dx, dy = target[0] - start[0], target[1] - start[1]
    return f"{dx} {dy} 0" + (f" ; {suffix}" if suffix else "")


def build_trajectory(
    fixture: Fixture,
    state: dict[str, Any],
    *,
    arm: Arm,
    cursor: tuple[int, int],
    near_miss: bool = False,
) -> GoldTrajectory:
    geometry = state.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("fixture geometry missing")
    p = fixture.params
    endpoint: tuple[int, int] | None
    if fixture.template == "click":
        target = _center(geometry, "decoy" if near_miss else "target")
        endpoint = target
        if arm == "native_absolute_control":
            actions: tuple[dict[str, Any] | str, ...] = (
                {"action": "left_click", "coordinate": list(target)},
            )
        else:
            actions = (_raw_move(cursor, target, "+LMB -LMB"),)
    elif fixture.template == "focus_type":
        target = _center(geometry, "target")
        endpoint = target
        text = str(p["target_text"]) + ("x" if near_miss else "")
        if arm == "native_absolute_control":
            actions = (
                {"action": "left_click", "coordinate": list(target)},
                {"action": "key", "keys": ["CTRL", "A"]},
                {"action": "type", "text": text},
            )
        else:
            actions = (
                _raw_move(cursor, target, "+LMB -LMB"),
                "0 0 0 ; +ControlLeft +KeyA -KeyA -ControlLeft",
                "0 0 0 ; type(" + json.dumps(text, ensure_ascii=False) + ")",
            )
    elif fixture.template == "scroll":
        endpoint = None
        clicks = int(p["scroll_clicks"])
        if near_miss:
            clicks = -clicks
        if arm == "native_absolute_control":
            actions = ({"action": "scroll", "clicks": clicks},)
        else:
            actions = (f"0 0 {clicks}",)
    elif fixture.template == "drag":
        rect = geometry.get("target")
        if not isinstance(rect, dict):
            raise ValueError("drag target geometry missing")
        left, right = int(rect["left"]), int(rect["right"])
        width = int(rect["width"])
        y = int(rect["center_y"])
        initial = int(p["initial_value"])
        start = (int(round(left + 8 + (width - 16) * initial / 100)), y)
        endpoint = (right - 2, y) if int(p["target_value"]) == 100 else (left + 2, y)
        if near_miss:
            endpoint = (int(round((start[0] + endpoint[0]) / 2)), y)
        if arm == "native_absolute_control":
            actions = (
                {"action": "mouse_move", "coordinate": list(start)},
                {"action": "left_click_drag", "coordinate": list(endpoint)},
            )
        else:
            actions = (
                _raw_move(cursor, start, "+LMB"),
                _raw_move(start, endpoint),
                "0 0 0 ; -LMB",
            )
    else:
        raise ValueError(f"unsupported fixture template {fixture.template!r}")
    if len(actions) > fixture.horizon:
        raise ValueError(
            f"trajectory exceeds frozen horizon for {fixture.id}: "
            f"{len(actions)} > {fixture.horizon}"
        )
    return GoldTrajectory(
        arm=arm,
        actions=actions,
        observed_cursor_baseline=(int(cursor[0]), int(cursor[1])),
        expected_endpoint=endpoint,
    )
