from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from desktop.geometry import DisplayGeometry

from grammars.ordered_events_v3_relative_1000_grid_v1.codec import (
    CODEC,
    OrderedEventsV3Action,
    Primitive,
    grid_delta,
    pixels_from_grid,
)
from pipeline.cua_gym.key_names import key_name

_BUTTONS = {"left": "LMB", "middle": "MMB", "right": "RMB"}
_CLICKS = {
    "click": ("left", 1),
    "left_click": ("left", 1),
    "right_click": ("right", 1),
    "middle_click": ("middle", 1),
    "double_click": ("left", 2),
    "triple_click": ("left", 3),
}
_NO_OPS = {"wait", "screenshot"}
_UNSUPPORTED = {"call_user", "answer"}
_FIELDS = {
    "terminate": ({"action", "status"}, {"status"}),
    "wait": ({"action", "time"}, {"time"}),
    "screenshot": ({"action"}, set()),
    "mouse_move": ({"action", "coordinate"}, {"coordinate"}),
    "left_click": ({"action", "coordinate"}, set()),
    "click": ({"action", "coordinate"}, set()),
    "right_click": ({"action", "coordinate"}, set()),
    "middle_click": ({"action", "coordinate"}, set()),
    "double_click": ({"action", "coordinate"}, set()),
    "triple_click": ({"action", "coordinate"}, set()),
    "left_click_drag": ({"action", "coordinate"}, {"coordinate"}),
    "left_mouse_down": ({"action", "coordinate"}, set()),
    "left_mouse_up": ({"action", "coordinate"}, set()),
    "key": ({"action", "keys"}, {"keys"}),
    "key_down": ({"action", "keys"}, {"keys"}),
    "key_up": ({"action", "keys"}, {"keys"}),
    "type": ({"action", "text"}, {"text"}),
    "scroll": ({"action", "pixels"}, {"pixels"}),
    "hscroll": ({"action", "pixels"}, {"pixels"}),
    "call_user": ({"action", "text"}, {"text"}),
    "answer": ({"action", "text"}, {"text"}),
}


class UnsupportedSourceAction(ValueError):
    pass


@dataclass(frozen=True)
class Translation:
    action: OrderedEventsV3Action
    target_pixel: tuple[int, int] | None = None

    @property
    def text(self) -> str:
        return CODEC.format(self.action)


def _size(geometry: DisplayGeometry) -> tuple[int, int]:
    width = int(geometry.desktop_width)
    height = int(geometry.desktop_height)
    if width <= 0 or height <= 0:
        raise ValueError(f"display geometry must be positive, got {width}x{height}")
    return width, height


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric, got {value!r}")
    return float(value)


def _cursor(value: object, geometry: DisplayGeometry) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"cursor_before must be a numeric pair, got {value!r}")
    width, height = _size(geometry)
    cursor = (
        round(_number(value[0], field="cursor_before[0]")),
        round(_number(value[1], field="cursor_before[1]")),
    )
    if not (0 <= cursor[0] < width and 0 <= cursor[1] < height):
        raise ValueError(f"cursor_before is outside {width}x{height}: {cursor!r}")
    return cursor


def _coordinate(value: object, geometry: DisplayGeometry) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"coordinate must be a numeric pair, got {value!r}")
    width, height = _size(geometry)
    grid = (
        round(_number(value[0], field="coordinate[0]")),
        round(_number(value[1], field="coordinate[1]")),
    )
    pixel = (
        pixels_from_grid(grid[0], width),
        pixels_from_grid(grid[1], height),
    )
    return (
        min(max(pixel[0], 0), width - 1),
        min(max(pixel[1], 0), height - 1),
    )


def _move(
    target: tuple[int, int] | None,
    cursor: tuple[int, int],
    geometry: DisplayGeometry,
) -> tuple[list[Primitive], tuple[int, int] | None]:
    if target is None:
        return [], None
    width, height = _size(geometry)
    delta = (
        grid_delta(cursor[0], target[0], width),
        grid_delta(cursor[1], target[1], height),
    )
    return (
        [] if delta == (0, 0) else [Primitive("move", dx=delta[0], dy=delta[1])]
    ), target


def _click(button: str, count: int) -> list[Primitive]:
    name = _BUTTONS[button]
    return [
        primitive
        for _ in range(count)
        for primitive in (Primitive("down", name=name), Primitive("up", name=name))
    ]


def _keys(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"keys must be a non-empty list, got {value!r}")
    return [key_name(item) for item in value]


def _typing(text: str) -> list[Primitive]:
    if not text:
        return []
    primitives: list[Primitive] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            primitives.append(Primitive("type", text="".join(run)))
            run.clear()

    for char in text:
        if char == "\n":
            flush()
            primitives.extend(
                (Primitive("down", name="Return"), Primitive("up", name="Return"))
            )
        elif char == "\t":
            flush()
            primitives.extend(
                (Primitive("down", name="Tab"), Primitive("up", name="Tab"))
            )
        elif char != "\r":
            run.append(char)
    flush()
    return primitives


def translate_step(
    arguments: dict[str, Any],
    cursor_before: object,
    geometry: DisplayGeometry,
) -> Translation:
    if not isinstance(arguments, dict):
        raise TypeError(f"computer_use arguments must be an object, got {arguments!r}")
    action = arguments.get("action")
    if not isinstance(action, str):
        raise TypeError(f"computer_use action must be text, got {action!r}")
    if action not in _FIELDS:
        raise ValueError(f"unsupported computer_use action: {action!r}")
    permitted, required = _FIELDS[action]
    extra = set(arguments) - permitted
    missing = required - set(arguments)
    if extra:
        raise ValueError(f"unexpected arguments for {action}: {sorted(extra)}")
    if missing:
        raise ValueError(f"missing arguments for {action}: {sorted(missing)}")
    if action in _UNSUPPORTED:
        raise UnsupportedSourceAction(action)
    cursor = _cursor(cursor_before, geometry)
    coordinate = _coordinate(arguments.get("coordinate"), geometry)
    primitives: list[Primitive] = []
    target = None

    if action == "terminate":
        status = arguments["status"]
        if status not in ("success", "failure"):
            raise ValueError(
                f"terminate status must be success or failure, got {status!r}"
            )
        return Translation(OrderedEventsV3Action(no_op=True, terminate=status))
    if action in _NO_OPS:
        if action == "wait":
            _number(arguments["time"], field="time")
        return Translation(OrderedEventsV3Action(no_op=True))
    if action == "mouse_move":
        if coordinate is None:
            raise ValueError("mouse_move requires coordinate")
        moved, target = _move(coordinate, cursor, geometry)
        primitives.extend(moved)
    elif action in _CLICKS:
        button, count = _CLICKS[action]
        moved, target = _move(coordinate, cursor, geometry)
        primitives.extend(moved)
        primitives.extend(_click(button, count))
    elif action == "left_click_drag":
        if coordinate is None:
            raise ValueError("left_click_drag requires coordinate")
        primitives.append(Primitive("down", name="LMB"))
        moved, target = _move(coordinate, cursor, geometry)
        primitives.extend(moved)
        primitives.append(Primitive("up", name="LMB"))
    elif action in ("left_mouse_down", "left_mouse_up"):
        moved, target = _move(coordinate, cursor, geometry)
        primitives.extend(moved)
        primitives.append(
            Primitive("down" if action == "left_mouse_down" else "up", name="LMB")
        )
    elif action == "key":
        names = _keys(arguments.get("keys"))
        primitives.extend(Primitive("down", name=name) for name in names)
        primitives.extend(Primitive("up", name=name) for name in reversed(names))
    elif action in ("key_down", "key_up"):
        primitives.extend(
            Primitive("down" if action == "key_down" else "up", name=name)
            for name in _keys(arguments.get("keys"))
        )
    elif action == "type":
        text = arguments["text"]
        if not isinstance(text, str):
            raise ValueError(f"type requires text, got {text!r}")
        primitives.extend(_typing(text))
    elif action in ("scroll", "hscroll"):
        amount = round(_number(arguments["pixels"], field="pixels"))
        if amount:
            primitives.append(
                Primitive(
                    "scroll",
                    dx=amount if action == "hscroll" else 0,
                    dy=amount if action == "scroll" else 0,
                )
            )
    return Translation(
        OrderedEventsV3Action(
            primitives=tuple(primitives),
            no_op=not primitives,
        ),
        target_pixel=target,
    )
