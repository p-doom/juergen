from __future__ import annotations

import re
from dataclasses import dataclass, field

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

GRID = 1000

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


class DropStep(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def px_to_norm(xy: tuple[float, float], screen: tuple[int, int]) -> tuple[int, int]:
    return (
        round(xy[0] / screen[0] * GRID),
        round(xy[1] / screen[1] * GRID),
    )


def norm_to_px(xy: tuple[int, int], screen: tuple[int, int]) -> tuple[int, int]:
    return (
        round(xy[0] / GRID * screen[0]),
        round(xy[1] / GRID * screen[1]),
    )


@dataclass
class StepTranslation:
    line: str
    move_delta: tuple[int, int] | None = None
    target_norm: tuple[int, int] | None = None
    dropped_reason: str | None = None
    stats: dict[str, int] = field(default_factory=dict)


def _coerce_coordinate(value, screen: tuple[int, int]) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise DropStep(f"malformed coordinate: {value!r}")
    x, y = value
    if not all(isinstance(v, (int, float)) for v in (x, y)):
        raise DropStep(f"non-numeric coordinate: {value!r}")
    px = norm_to_px((round(x), round(y)), screen)
    clamped = (
        min(max(px[0], 0), screen[0] - 1),
        min(max(px[1], 0), screen[1] - 1),
    )
    return px_to_norm(clamped, screen)


def _move_primitives(
    target_norm: tuple[int, int] | None, cursor_norm: tuple[int, int]
) -> tuple[list[str], tuple[int, int] | None]:
    if target_norm is None:
        return [], None
    dx = target_norm[0] - cursor_norm[0]
    dy = target_norm[1] - cursor_norm[1]
    if dx == 0 and dy == 0:
        return [], (0, 0)
    return [render_move(dx, dy)], (dx, dy)


def _click_primitives(button: str, count: int) -> list[str]:
    name = _BUTTON_NAMES[button]
    prims: list[str] = []
    for _ in range(count):
        prims.append(render_down(name))
        prims.append(render_up(name))
    return prims


def _type_primitives(text: str) -> list[str]:
    prims: list[str] = []
    run: list[str] = []

    def flush():
        if run:
            prims.append(render_type("".join(run)))
            run.clear()

    for ch in text:
        if ch == "\n":
            flush()
            prims.append(render_down("Return"))
            prims.append(render_up("Return"))
        elif ch == "\t":
            flush()
            prims.append(render_down("Tab"))
            prims.append(render_up("Tab"))
        elif ch == "\r":
            continue
        else:
            run.append(ch)
    flush()
    return prims


def _key_transition_primitives(keys: list) -> list[str]:
    if not isinstance(keys, list) or not keys:
        raise DropStep(f"malformed keys: {keys!r}")
    names = [pyautogui_to_rdev(k) for k in keys]
    prims = [render_down(n) for n in names]
    prims.extend(render_up(n) for n in reversed(names))
    return prims


def translate_step(
    args: dict,
    cursor_before_px: tuple[float, float],
    screen: tuple[int, int],
) -> StepTranslation:
    action = args.get("action")
    if action in _DROP_ACTIONS:
        return StepTranslation(line="", dropped_reason=action)
    cursor_norm = px_to_norm(cursor_before_px, screen)
    coord = _coerce_coordinate(args.get("coordinate"), screen)
    prims: list[str] = []
    move_delta = None
    target_norm = None

    try:
        if action == "terminate":
            return StepTranslation(line=TERMINATE, stats={"terminate": 1})
        if action in _NOOP_ACTIONS:
            return StepTranslation(line=NO_OP, stats={action: 1})
        if action == "mouse_move":
            if coord is None:
                raise DropStep("mouse_move without coordinate")
            moved, move_delta = _move_primitives(coord, cursor_norm)
            target_norm = coord
            prims.extend(moved)
            if not prims:
                return StepTranslation(
                    line=NO_OP,
                    move_delta=move_delta,
                    target_norm=target_norm,
                    stats={"mouse_move_zero": 1},
                )
        elif action in _CLICK_COUNTS or action == "click":
            button, count = _CLICK_COUNTS.get(action, ("left", 1))
            moved, move_delta = _move_primitives(coord, cursor_norm)
            target_norm = coord
            prims.extend(moved)
            prims.extend(_click_primitives(button, count))
        elif action == "left_click_drag":
            if coord is None:
                raise DropStep("left_click_drag without coordinate")
            prims.append(render_down("LMB"))
            moved, move_delta = _move_primitives(coord, cursor_norm)
            target_norm = coord
            prims.extend(moved)
            prims.append(render_up("LMB"))
        elif action == "left_mouse_down":
            moved, move_delta = _move_primitives(coord, cursor_norm)
            target_norm = coord
            prims.extend(moved)
            prims.append(render_down("LMB"))
        elif action == "left_mouse_up":
            moved, move_delta = _move_primitives(coord, cursor_norm)
            target_norm = coord
            prims.extend(moved)
            prims.append(render_up("LMB"))
        elif action == "key":
            prims.extend(_key_transition_primitives(args.get("keys")))
        elif action == "key_down":
            names = [pyautogui_to_rdev(k) for k in args.get("keys") or []]
            if not names:
                raise DropStep("key_down without keys")
            prims.extend(render_down(n) for n in names)
        elif action == "key_up":
            names = [pyautogui_to_rdev(k) for k in args.get("keys") or []]
            if not names:
                raise DropStep("key_up without keys")
            prims.extend(render_up(n) for n in names)
        elif action == "type":
            text = args.get("text")
            if not isinstance(text, str):
                raise DropStep(f"type without text: {args!r}")
            prims.extend(_type_primitives(text))
            if not prims:
                return StepTranslation(line=NO_OP, stats={"type_empty": 1})
        elif action == "scroll":
            pixels = round(args.get("pixels") or 0)
            if pixels == 0:
                return StepTranslation(line=NO_OP, stats={"scroll_zero": 1})
            prims.append(render_scroll(0, pixels))
        elif action == "hscroll":
            pixels = round(args.get("pixels") or 0)
            if pixels == 0:
                return StepTranslation(line=NO_OP, stats={"scroll_zero": 1})
            prims.append(render_scroll(pixels, 0))
        else:
            raise DropStep(f"unknown action: {action!r}")
    except UnmappableKeyError as exc:
        raise DropStep(str(exc)) from exc

    return StepTranslation(
        line=join_primitives(prims),
        move_delta=move_delta,
        target_norm=target_norm,
        stats={action: 1},
    )


_THINK_CLOSE = "</think>"


def rewrite_assistant(assistant_raw: str, action_line: str) -> str:
    head, sep, _ = assistant_raw.partition("<tool_call>")
    if not sep:
        raise DropStep("no <tool_call> in assistant_raw")
    if _THINK_CLOSE not in head:
        raise DropStep("no </think> in assistant_raw")
    return f"<think>{head.rstrip()}\n{action_line}"


def reconstruct_target_px(
    cursor_before_px: tuple[float, float],
    move_delta: tuple[int, int],
    screen: tuple[int, int],
) -> tuple[int, int]:
    cursor_norm = px_to_norm(cursor_before_px, screen)
    return norm_to_px(
        (cursor_norm[0] + move_delta[0], cursor_norm[1] + move_delta[1]), screen
    )
