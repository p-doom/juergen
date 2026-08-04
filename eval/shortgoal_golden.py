"""Golden oracle policies: one ``GoldenStep`` per assistant turn, in pixels.

A ``GoldenStep`` is the ordered list of pixel-space primitives that make up ONE
assistant reply — one ``ordered_events_v4`` line — so the turn boundaries of the
training data are semantic by construction: move+click is one step, type+Enter is
one step, a drag is ``down`` / ``move`` / ``up`` in one step, and a wait is a
whole-line ``no_op``.

Policies think in ABSOLUTE screen pixels (``{"kind": "move", "to_xy": [x, y]}``);
the recorder is what grid-snaps those targets (``shortgoal_grammar.snap_point_px``)
and renders them as either a rel ``move(dx,dy)`` delta from the live cursor or an
abs ``move_to(x,y)``. That keeps the two arms byte-identical apart from the move
token, and keeps every policy pure: the returned steps depend only on the task
params and the context (cursor, screen, geometry), never on wall-clock or a
default random state.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

GoldenStep = list[dict[str, Any]]

STEP_KINDS = ("move", "down", "up", "type", "scroll", "no_op")
MAX_STEPS = 8
MOUSE_NAMES = ("LMB", "MMB", "RMB")
COMMIT_KEY = "Return"


@dataclass(frozen=True)
class GoldenCtx:
    """What a policy may look at besides the task: live cursor, screen, geometry."""

    cursor_xy: tuple[int, int]
    screen_wh: tuple[int, int]
    geometry: dict[str, Any] = field(default_factory=dict)


def move(x: int, y: int) -> dict[str, Any]:
    """A pointer move to absolute pixel ``(x, y)``."""
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (x, y)):
        raise ValueError(f"move target must be int pixels, got {(x, y)!r}")
    return {"kind": "move", "to_xy": [int(x), int(y)]}


def down(name: str) -> dict[str, Any]:
    """A key or mouse-button press, rdev name."""
    return _named("down", name)


def up(name: str) -> dict[str, Any]:
    """A key or mouse-button release, rdev name."""
    return _named("up", name)


def type_text(text: str) -> dict[str, Any]:
    """A run of typed characters (never a newline: Return is down/up)."""
    if not isinstance(text, str) or not text:
        raise ValueError(f"type text must be a nonempty string, got {text!r}")
    if any(ord(c) < 32 or ord(c) == 127 for c in text):
        raise ValueError(f"control character in type text: {text!r}")
    return {"kind": "type", "text": text}


def scroll(notches: int) -> dict[str, Any]:
    """A wheel scroll of ``notches`` (positive scrolls up)."""
    if not isinstance(notches, int) or isinstance(notches, bool) or notches == 0:
        raise ValueError(f"scroll notches must be a nonzero int, got {notches!r}")
    return {"kind": "scroll", "notches": int(notches)}


def no_op() -> dict[str, Any]:
    """The whole-line wait primitive."""
    return {"kind": "no_op"}


def _named(kind: str, name: str) -> dict[str, Any]:
    if not isinstance(name, str) or not name or any(c in name for c in " (),;"):
        raise ValueError(f"{kind}() needs a bare input name, got {name!r}")
    return {"kind": kind, "name": name}


def _xy(value: Any, what: str) -> tuple[int, int]:
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        raise ValueError(f"{what} must be a pixel pair, got {value!r}")
    x, y = value
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (x, y)):
        raise ValueError(f"{what} must be int pixels, got {value!r}")
    return int(x), int(y)


def click_step(xy: Any, *, name: str = "LMB", count: int = 1) -> GoldenStep:
    """Move to ``xy`` and click there — one turn."""
    if not isinstance(count, int) or not 1 <= count <= 3:
        raise ValueError(f"click count must be 1..3, got {count!r}")
    if name not in MOUSE_NAMES:
        raise ValueError(f"click needs a mouse button, got {name!r}")
    step: GoldenStep = [move(*_xy(xy, "click target"))]
    for _ in range(count):
        step.extend((down(name), up(name)))
    return step


def combo_step(names: Sequence[str]) -> GoldenStep:
    """Press ``names`` in order and release them in reverse — one turn."""
    keys = list(names)
    if not keys:
        raise ValueError("combo needs at least one key")
    return [down(key) for key in keys] + [up(key) for key in reversed(keys)]


def type_enter_step(text: str) -> GoldenStep:
    """Type ``text`` and press Return — one turn."""
    return [type_text(text), down(COMMIT_KEY), up(COMMIT_KEY)]


def commit_step() -> GoldenStep:
    """Press Return to confirm a fixture interaction — one turn."""
    return combo_step((COMMIT_KEY,))


def drag_step(from_xy: Any, to_xy: Any, *, name: str = "LMB") -> GoldenStep:
    """Press at ``from_xy``, move to ``to_xy``, release — one turn."""
    if name not in MOUSE_NAMES:
        raise ValueError(f"drag needs a mouse button, got {name!r}")
    return [
        move(*_xy(from_xy, "drag start")),
        down(name),
        move(*_xy(to_xy, "drag end")),
        up(name),
    ]


def _p_term_type_enter(task: Any, ctx: GoldenCtx) -> list[GoldenStep]:
    return [type_enter_step(task.params["command"])]


def _p_term_two_commands(task: Any, ctx: GoldenCtx) -> list[GoldenStep]:
    commands = task.params["commands"]
    if len(commands) != 2:
        raise ValueError(f"term_two_commands needs 2 commands, got {commands!r}")
    return [type_enter_step(command) for command in commands]


def _p_term_launch_editor(task: Any, ctx: GoldenCtx) -> list[GoldenStep]:
    waits = task.params["waits"]
    if not isinstance(waits, int) or not 1 <= waits <= 2:
        raise ValueError(f"launch-editor waits must be 1 or 2, got {waits!r}")
    return [type_enter_step(task.params["command"])] + [[no_op()] for _ in range(waits)]


def _p_gedit_write_save(task: Any, ctx: GoldenCtx) -> list[GoldenStep]:
    steps = [[type_text(task.params["sentence"])]]
    steps.extend(combo_step(combo) for combo in task.params["combos"])
    return steps


def _p_key_combos(task: Any, ctx: GoldenCtx) -> list[GoldenStep]:
    combos = task.params["combos"]
    if not combos:
        raise ValueError(f"{task.task_id} has no combos to press")
    return [combo_step(combo) for combo in combos]


def _p_fx_click_commit(task: Any, ctx: GoldenCtx) -> list[GoldenStep]:
    return [click_step(task.params["target_xy"]), commit_step()]


def _p_fx_double_click_commit(task: Any, ctx: GoldenCtx) -> list[GoldenStep]:
    return [click_step(task.params["target_xy"], count=2), commit_step()]


def _p_fx_right_click_commit(task: Any, ctx: GoldenCtx) -> list[GoldenStep]:
    return [click_step(task.params["target_xy"], name="RMB"), commit_step()]


def _p_fx_drag_slider(task: Any, ctx: GoldenCtx) -> list[GoldenStep]:
    return [drag_step(task.params["handle_xy"], task.params["target_xy"]), commit_step()]


def _p_fx_scroll_commit(task: Any, ctx: GoldenCtx) -> list[GoldenStep]:
    return [
        [move(*_xy(task.params["pane_xy"], "scroll pane centre"))],
        [scroll(task.params["notches"])],
        commit_step(),
    ]


def _p_fx_scroll_find_click(task: Any, ctx: GoldenCtx) -> list[GoldenStep]:
    bursts = task.params["bursts"]
    if not bursts:
        raise ValueError(f"{task.task_id} has no scroll bursts")
    steps: list[GoldenStep] = [[move(*_xy(task.params["pane_xy"], "list centre"))]]
    steps.extend([scroll(notches)] for notches in bursts)
    steps.append(click_step(task.params["target_xy"]))
    return steps


def _p_fx_two_clicks(task: Any, ctx: GoldenCtx) -> list[GoldenStep]:
    return [click_step(task.params["first_xy"]), click_step(task.params["second_xy"])]


def _p_web_click(task: Any, ctx: GoldenCtx) -> list[GoldenStep]:
    return [click_step(task.params["target_xy"])]


def _p_web_click_type(task: Any, ctx: GoldenCtx) -> list[GoldenStep]:
    return [click_step(task.params["input_xy"]), type_enter_step(task.params["text"])]


def _p_web_scroll_click(task: Any, ctx: GoldenCtx) -> list[GoldenStep]:
    return [[scroll(task.params["notches"])], click_step(task.params["target_xy"])]


POLICIES: dict[str, Callable[[Any, GoldenCtx], list[GoldenStep]]] = {
    "p_term_type_enter": _p_term_type_enter,
    "p_term_two_commands": _p_term_two_commands,
    "p_term_launch_editor": _p_term_launch_editor,
    "p_gedit_write_save": _p_gedit_write_save,
    "p_key_combos": _p_key_combos,
    "p_fx_click_commit": _p_fx_click_commit,
    "p_fx_double_click_commit": _p_fx_double_click_commit,
    "p_fx_right_click_commit": _p_fx_right_click_commit,
    "p_fx_drag_slider": _p_fx_drag_slider,
    "p_fx_scroll_commit": _p_fx_scroll_commit,
    "p_fx_scroll_find_click": _p_fx_scroll_find_click,
    "p_fx_two_clicks": _p_fx_two_clicks,
    "p_web_click": _p_web_click,
    "p_web_click_type": _p_web_click_type,
    "p_web_scroll_click": _p_web_scroll_click,
}


def validate_step(step: Any) -> GoldenStep:
    """Check one turn: known kinds, ``no_op`` alone, no dangling press."""
    if not (isinstance(step, list) and step):
        raise ValueError(f"a golden step must be a nonempty list, got {step!r}")
    kinds = [prim.get("kind") for prim in step]
    if any(kind not in STEP_KINDS for kind in kinds):
        raise ValueError(f"unknown primitive kind in step: {kinds!r}")
    if "no_op" in kinds and len(step) != 1:
        raise ValueError(f"no_op is a whole line, got {kinds!r}")
    held: list[str] = []
    for prim in step:
        if prim["kind"] == "down":
            if prim["name"] in held:
                raise ValueError(f"{prim['name']} pressed twice in one step")
            held.append(prim["name"])
        elif prim["kind"] == "up":
            if prim["name"] not in held:
                raise ValueError(f"{prim['name']} released without a press in the same step")
            held.remove(prim["name"])
    if held:
        raise ValueError(f"step ends with {held!r} still held")
    return step


def validate_steps(steps: Any) -> list[GoldenStep]:
    """Check a whole golden trajectory: 1..8 turns, each a valid turn."""
    if not (isinstance(steps, list) and steps):
        raise ValueError(f"a golden trajectory needs >=1 step, got {steps!r}")
    if len(steps) > MAX_STEPS:
        raise ValueError(f"golden trajectory of {len(steps)} steps exceeds {MAX_STEPS}")
    for step in steps:
        validate_step(step)
    return steps


def golden_steps(task: Any, ctx: GoldenCtx) -> list[GoldenStep]:
    """The ordered golden turns for ``task`` — pure given its params and ``ctx``."""
    if not isinstance(ctx, GoldenCtx):
        raise TypeError(f"golden policies need a GoldenCtx, got {type(ctx)!r}")
    screen_w, screen_h = _xy(ctx.screen_wh, "screen size")
    if screen_w <= 0 or screen_h <= 0:
        raise ValueError(f"screen size must be positive, got {ctx.screen_wh!r}")
    policy = POLICIES.get(task.policy_id)
    if policy is None:
        raise KeyError(f"unknown golden policy: {task.policy_id!r}")
    steps = validate_steps(policy(task, ctx))
    for x, y in move_targets(steps):
        if not (0 <= x < screen_w and 0 <= y < screen_h):
            raise ValueError(f"{task.task_id} moves to ({x},{y}), off a {screen_w}x{screen_h} screen")
    return steps


def move_targets(steps: Iterable[GoldenStep]) -> list[tuple[int, int]]:
    """Every absolute pixel target the trajectory moves to, in order."""
    return [
        _xy(prim["to_xy"], "move target")
        for step in steps
        for prim in step
        if prim["kind"] == "move"
    ]
