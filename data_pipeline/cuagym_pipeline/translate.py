from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cuagym_pipeline.key_names import UnmappableKeyError, pyautogui_to_rdev
from cuagym_pipeline.oev3_render import (
    NO_OP,
    TERMINATE,
    join_primitives,
    render_down,
    render_move,
    render_scroll,
    render_type,
    render_up,
)

GRID_SIZE = 1000
_BUTTON_NAMES = {"left": "LMB", "middle": "MMB", "right": "RMB"}
_CLICK_COUNTS = {
    "left_click": ("left", 1),
    "right_click": ("right", 1),
    "middle_click": ("middle", 1),
    "double_click": ("left", 2),
    "triple_click": ("left", 3),
}
_NOOP_ACTIONS = frozenset({"wait", "screenshot"})
_DROP_ACTIONS = frozenset({"call_user", "answer"})


class DropStepError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def px_to_norm(xy: tuple[float, float], screen: tuple[int, int]) -> tuple[int, int]:
    return (
        round(xy[0] / screen[0] * GRID_SIZE),
        round(xy[1] / screen[1] * GRID_SIZE),
    )


def norm_to_px(xy: tuple[int, int], screen: tuple[int, int]) -> tuple[int, int]:
    return (
        round(xy[0] / GRID_SIZE * screen[0]),
        round(xy[1] / GRID_SIZE * screen[1]),
    )


@dataclass
class StepTranslation:
    line: str
    move_delta: tuple[int, int] | None = None
    target_norm: tuple[int, int] | None = None
    dropped_reason: str | None = None
    stats: dict[str, int] = field(default_factory=dict)


def _coordinate(value: Any, screen: tuple[int, int]) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise DropStepError(f"malformed coordinate: {value!r}")
    x, y = value
    if not all(isinstance(item, (int, float)) for item in (x, y)):
        raise DropStepError(f"non-numeric coordinate: {value!r}")
    px = norm_to_px((round(x), round(y)), screen)
    clamped = (
        min(max(px[0], 0), screen[0] - 1),
        min(max(px[1], 0), screen[1] - 1),
    )
    return px_to_norm(clamped, screen)


def _move(
    target_norm: tuple[int, int] | None, cursor_norm: tuple[int, int]
) -> tuple[list[str], tuple[int, int] | None]:
    if target_norm is None:
        return [], None
    delta = (target_norm[0] - cursor_norm[0], target_norm[1] - cursor_norm[1])
    return ([] if delta == (0, 0) else [render_move(*delta)]), delta


def _click(button: str, count: int) -> list[str]:
    name = _BUTTON_NAMES[button]
    return [primitive for _ in range(count) for primitive in (render_down(name), render_up(name))]


def _type(text: str) -> list[str]:
    primitives: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            primitives.append(render_type("".join(run)))
            run.clear()

    for char in text:
        if char == "\n":
            flush()
            primitives.extend((render_down("Return"), render_up("Return")))
        elif char == "\t":
            flush()
            primitives.extend((render_down("Tab"), render_up("Tab")))
        elif char != "\r":
            run.append(char)
    flush()
    return primitives


def _key_transitions(keys: Any) -> list[str]:
    if not isinstance(keys, list) or not keys:
        raise DropStepError(f"malformed keys: {keys!r}")
    names = [pyautogui_to_rdev(key) for key in keys]
    return [
        *(render_down(name) for name in names),
        *(render_up(name) for name in reversed(names)),
    ]


def translate_step(
    args: dict[str, Any],
    cursor_before_px: tuple[float, float],
    screen: tuple[int, int],
) -> StepTranslation:
    action = args.get("action")
    if action in _DROP_ACTIONS:
        return StepTranslation(line="", dropped_reason=str(action))
    cursor_norm = px_to_norm(cursor_before_px, screen)
    coordinate = _coordinate(args.get("coordinate"), screen)
    primitives: list[str] = []
    move_delta = None
    target_norm = None

    try:
        if action == "terminate":
            return StepTranslation(line=TERMINATE, stats={"terminate": 1})
        if action in _NOOP_ACTIONS:
            return StepTranslation(line=NO_OP, stats={str(action): 1})
        if action == "mouse_move":
            if coordinate is None:
                raise DropStepError("mouse_move without coordinate")
            moved, move_delta = _move(coordinate, cursor_norm)
            target_norm = coordinate
            primitives.extend(moved)
        elif action in _CLICK_COUNTS or action == "click":
            button, count = _CLICK_COUNTS.get(str(action), ("left", 1))
            moved, move_delta = _move(coordinate, cursor_norm)
            target_norm = coordinate
            primitives.extend(moved)
            primitives.extend(_click(button, count))
        elif action == "left_click_drag":
            if coordinate is None:
                raise DropStepError("left_click_drag without coordinate")
            primitives.append(render_down("LMB"))
            moved, move_delta = _move(coordinate, cursor_norm)
            target_norm = coordinate
            primitives.extend(moved)
            primitives.append(render_up("LMB"))
        elif action in ("left_mouse_down", "left_mouse_up"):
            moved, move_delta = _move(coordinate, cursor_norm)
            target_norm = coordinate
            primitives.extend(moved)
            primitives.append(
                render_down("LMB") if action == "left_mouse_down" else render_up("LMB")
            )
        elif action == "key":
            primitives.extend(_key_transitions(args.get("keys")))
        elif action in ("key_down", "key_up"):
            keys = args.get("keys")
            if not isinstance(keys, list) or not keys:
                raise DropStepError(f"{action} without keys")
            names = [pyautogui_to_rdev(key) for key in keys]
            render = render_down if action == "key_down" else render_up
            primitives.extend(render(name) for name in names)
        elif action == "type":
            text = args.get("text")
            if not isinstance(text, str):
                raise DropStepError(f"type without text: {args!r}")
            primitives.extend(_type(text))
        elif action in ("scroll", "hscroll"):
            amount = round(args.get("pixels") or 0)
            if amount:
                primitives.append(
                    render_scroll(0, amount) if action == "scroll" else render_scroll(amount, 0)
                )
        else:
            raise DropStepError(f"unknown action: {action!r}")
    except UnmappableKeyError as exc:
        raise DropStepError(str(exc)) from exc

    return StepTranslation(
        line=join_primitives(primitives),
        move_delta=move_delta,
        target_norm=target_norm,
        stats={str(action): 1},
    )


def rewrite_assistant(assistant_raw: str, action_line: str) -> str:
    reasoning, separator, _ = assistant_raw.partition("<tool_call>")
    if not separator:
        raise DropStepError("no <tool_call> in assistant_raw")
    if "</think>" not in reasoning:
        raise DropStepError("no </think> in assistant_raw")
    return f"<think>{reasoning.rstrip()}\n\n{action_line}"
