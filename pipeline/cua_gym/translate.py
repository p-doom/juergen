from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from desktop.geometry import DisplayGeometry

from grammars.ordered_events_v3_relative_1000_grid_v1.codec import (
    CODEC,
    OrderedEventsV3Action,
    Primitive,
    grid_from_pixels,
    pixels_from_grid,
)
from pipeline.cua_gym.key_names import key_name

_BUTTONS = {"left": "LMB", "middle": "MMB", "right": "RMB"}
_CLICKS = {
    "left_click": ("left", 1),
    "right_click": ("right", 1),
    "middle_click": ("middle", 1),
    "double_click": ("left", 2),
    "triple_click": ("left", 3),
}
_NO_OPS = {"wait", "screenshot"}
_UNSUPPORTED = {"call_user", "answer"}


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
        grid_from_pixels(target[0], width) - grid_from_pixels(cursor[0], width),
        grid_from_pixels(target[1], height) - grid_from_pixels(cursor[1], height),
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
    if action in _UNSUPPORTED:
        raise UnsupportedSourceAction(str(action))
    if not isinstance(action, str):
        raise TypeError(f"computer_use action must be text, got {action!r}")
    cursor = _cursor(cursor_before, geometry)
    coordinate = _coordinate(arguments.get("coordinate"), geometry)
    primitives: list[Primitive] = []
    target = None

    if action == "terminate":
        status = arguments.get("status", "success")
        if status not in ("success", "failure"):
            raise ValueError(
                f"terminate status must be success or failure, got {status!r}"
            )
        return Translation(OrderedEventsV3Action(no_op=True, terminate=status))
    if action in _NO_OPS:
        return Translation(OrderedEventsV3Action(no_op=True))
    if action == "mouse_move":
        if coordinate is None:
            raise ValueError("mouse_move requires coordinate")
        moved, target = _move(coordinate, cursor, geometry)
        primitives.extend(moved)
    elif action in _CLICKS or action == "click":
        button, count = _CLICKS.get(action, ("left", 1))
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
        text = arguments.get("text")
        if not isinstance(text, str):
            raise ValueError(f"type requires text, got {text!r}")
        primitives.extend(_typing(text))
    elif action in ("scroll", "hscroll"):
        amount = round(_number(arguments.get("pixels", 0), field="pixels"))
        if amount:
            primitives.append(
                Primitive(
                    "scroll",
                    dx=amount if action == "hscroll" else 0,
                    dy=amount if action == "scroll" else 0,
                )
            )
    else:
        raise ValueError(f"unsupported computer_use action: {action!r}")

    return Translation(
        OrderedEventsV3Action(
            primitives=tuple(primitives),
            no_op=not primitives,
        ),
        target_pixel=target,
    )


def rewrite_assistant(source: str, action: OrderedEventsV3Action) -> str:
    if not isinstance(source, str):
        raise TypeError("assistant_raw must be text")
    prefix, separator, _ = source.partition("<tool_call>")
    if not separator or "</think>" not in prefix:
        raise ValueError("assistant_raw must contain reasoning followed by a tool call")
    if prefix.lstrip().startswith("<think>"):
        raise ValueError("assistant_raw unexpectedly contains an opening <think> tag")
    return f"<think>{prefix.rstrip()}\n\n{CODEC.format(action)}"
