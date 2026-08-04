"""Binding contract for the short-goal ``ordered_events_v4`` action formats.

Both arms share the ``ordered_events_v3`` line grammar (``"; "``-joined
primitives, whole-line ``NO_OP``, rdev ``down(NAME)``/``up(NAME)``,
quote-escaped ``type("...")``) with two changes: ``scroll`` takes a single
wheel-notch int (no hscroll), and the mouse primitive is the ONLY difference
between the arms — ``ordered_events_v4_rel`` emits ``move(dx,dy)`` as per-axis
thousandths of the screen, ``ordered_events_v4_abs`` emits ``move_to(x,y)`` on
the 0-1000 grid.

This module renders the wire text; ``action_parser.parse_ordered_v4_action``
parses exactly the same grammar. render -> parse -> render is byte-identical
(test_shortgoal_grammar.py), which is what lets the builder validate every
training line and the closed-loop evaluator dispatch model output through the
same contract.

``IMAGE_PLACEHOLDER``, ``K_IMAGES``, ``KEEP_IMAGES`` and ``FRAME_JPEG_QUALITY``
are the keep-text context constants that both ``shortgoal_build`` and the
runtime window read, so a training record and the closed-loop prompt for the
same decision carry the same turns AND the same frame bytes.

``THOUGHT_MAX_CHARS`` is the capture-only budget for the per-step first-person
thought an agent driver may record alongside a turn (``shortgoal_agent_record``
writes it, ``shortgoal_build`` validates it and renders nothing): the rungs are
no-think, so the field exists purely for a later thinking-render ablation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace

from action_parser import OrderedAction, OrderedPrimitive

ARM_REL = "ordered_events_v4_rel"
ARM_ABS = "ordered_events_v4_abs"
ARMS = (ARM_REL, ARM_ABS)

TERMINATE_LINE = "TERMINATE"
NO_OP_LINE = "NO_OP"

IMAGE_PLACEHOLDER = "<Image collapsed>"
K_IMAGES = 6
KEEP_IMAGES = 3
FRAME_JPEG_QUALITY = 90
THOUGHT_MAX_CHARS = 500

GRID = 1000

PROMPT_IDS = {
    ARM_REL: "shortgoal_oev4_rel_v1",
    ARM_ABS: "shortgoal_oev4_abs_v1",
}

ORDERED_EVENTS_V4_GRAMMAR = r'''
line       = "NO_OP" / primitive *("; " primitive)
primitive  = move / move_to / scroll / down / up / type
move       = "move(" int "," int ")"          ; ordered_events_v4_rel ONLY:
                                              ; per-axis thousandths of the
                                              ; screen (1000 == full width for
                                              ; dx, full height for dy), ints
                                              ; in [-1000,1000]; move(0,0) is
                                              ; never rendered and is rejected
move_to    = "move_to(" uint "," uint ")"     ; ordered_events_v4_abs ONLY:
                                              ; 0-1000 screen grid, always
                                              ; emitted before a click
scroll     = "scroll(" int ")"                ; wheel notches, nonzero,
                                              ; positive scrolls up; no hscroll
down       = "down(" NAME ")"                 ; key/button press, as in v3
up         = "up(" NAME ")"                   ; key/button release, as in v3
type       = "type(" DQUOTE chars DQUOTE ")"  ; run of >=1 typed characters
int        = ["-"] 1*DIGIT
uint       = 1*DIGIT
NAME       = 1*name-char                      ; name-char = any char except
                                              ; whitespace "(" ")" "," ";"
chars      = 1*char                           ; char = escape / plain
escape     = "\" ("\" / DQUOTE)               ; \\ -> backslash, \" -> quote
plain      = any printable US-keyboard character except "\" and DQUOTE
                                              ; Return/Tab are down/up, never
                                              ; typed, so chars has no
                                              ; newline/tab

Whole-line replies: "NO_OP" means "wait, take another settled screenshot" and
"TERMINATE" (the goal is done) is the ENTIRE assistant reply of the final turn
— neither ever shares a line with primitives.

Denormalization to VM pixels (denorm_v4, mirroring freeroll's move-delta
scaling): rel dx_px = round(dx * sw / 1000), dy_px = round(dy * sh / 1000);
abs x_px = clamp(round(x * sw / 1000), 0, sw - 1) and likewise for y. Golden
targets are grid-snapped (snap_point_px) before dispatch, so every recorded
pixel is exactly representable on the 0-1000 grid.
'''

_NAME_RE = re.compile(r"^[^\s(),;]+$")


def _check_arm(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown ordered_events_v4 arm: {arm!r}")
    return arm


def _check_screen(screen_px: int) -> int:
    if not isinstance(screen_px, int) or screen_px <= 0:
        raise ValueError(f"screen extent must be a positive int, got {screen_px!r}")
    return screen_px


def _check_int(value: object, *, lo: int, hi: int, what: str) -> int:
    if not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f"{what} must be an int in [{lo},{hi}], got {value!r}")
    return value


def _escape_type_text(text: str) -> str:
    """Escape a ``type("...")`` payload: only ``\\`` and ``"`` (v3 rule)."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _grid_to_px(grid: object, screen_px: int) -> int:
    """A 0-1000 grid coordinate as a pixel of a ``screen_px``-wide axis."""
    _check_screen(screen_px)
    g = _check_int(grid, lo=0, hi=GRID, what="grid coordinate")
    return min(screen_px - 1, max(0, round(g * screen_px / GRID)))


def render_primitive(prim: OrderedPrimitive, arm: str) -> str:
    """One v4 primitive as exact wire text; raises on anything unrenderable."""
    _check_arm(arm)
    if prim.kind == "move":
        if arm != ARM_REL:
            raise ValueError(f"move() belongs to {ARM_REL}, not {arm}")
        dx = _check_int(prim.dx, lo=-GRID, hi=GRID, what="move dx")
        dy = _check_int(prim.dy, lo=-GRID, hi=GRID, what="move dy")
        if dx == 0 and dy == 0:
            raise ValueError("move(0,0) is never rendered")
        return f"move({dx},{dy})"
    if prim.kind == "move_to":
        if arm != ARM_ABS:
            raise ValueError(f"move_to() belongs to {ARM_ABS}, not {arm}")
        x = _check_int(prim.x, lo=0, hi=GRID, what="move_to x")
        y = _check_int(prim.y, lo=0, hi=GRID, what="move_to y")
        return f"move_to({x},{y})"
    if prim.kind == "scroll":
        if prim.dx not in (None, 0):
            raise ValueError(f"ordered_events_v4 has no hscroll: {prim.dx!r}")
        if not isinstance(prim.dy, int) or prim.dy == 0:
            raise ValueError(f"scroll notches must be a nonzero int, got {prim.dy!r}")
        return f"scroll({prim.dy})"
    if prim.kind in ("down", "up"):
        if not isinstance(prim.name, str) or not _NAME_RE.match(prim.name):
            raise ValueError(f"invalid input name for {prim.kind}(): {prim.name!r}")
        return f"{prim.kind}({prim.name})"
    if prim.kind == "type":
        if not isinstance(prim.text, str) or not prim.text:
            raise ValueError(f"type() needs >=1 character, got {prim.text!r}")
        if any(ord(c) < 32 or ord(c) == 127 for c in prim.text):
            raise ValueError(f"control character in type() payload: {prim.text!r}")
        return f'type("{_escape_type_text(prim.text)}")'
    raise ValueError(f"unrenderable ordered_events_v4 primitive: {prim.kind!r}")


def render_line(prims: Sequence[OrderedPrimitive], arm: str) -> str:
    """The complete assistant action line for ``prims`` (``NO_OP`` when empty)."""
    _check_arm(arm)
    if not prims:
        return NO_OP_LINE
    return "; ".join(render_primitive(p, arm) for p in prims)


def norm_delta(d_px: int, screen_px: int) -> int:
    """A pixel delta as per-axis thousandths of the screen (rel arm)."""
    _check_screen(screen_px)
    if not isinstance(d_px, int):
        raise ValueError(f"pixel delta must be an int, got {d_px!r}")
    grid = round(d_px * GRID / screen_px)
    if not -GRID <= grid <= GRID:
        raise ValueError(f"delta {d_px} exceeds one {screen_px}px screen")
    return grid


def norm_point(p_px: int, screen_px: int) -> int:
    """A pixel coordinate as a 0-1000 grid coordinate (abs arm)."""
    _check_screen(screen_px)
    if not isinstance(p_px, int) or not 0 <= p_px < screen_px:
        raise ValueError(f"point {p_px!r} is not a pixel of a {screen_px}px axis")
    return min(GRID, round(p_px * GRID / screen_px))


def snap_point_px(p_px: int, screen_px: int) -> int:
    """The pixel closest to ``p_px``'s own grid point.

    Golden targets pass through this before dispatch so the recorded pixel is
    exactly representable: grid -> pixel -> grid is then the identity."""
    return _grid_to_px(norm_point(p_px, screen_px), screen_px)


def denorm_v4(action: OrderedAction, screen_size: tuple[int, int]) -> OrderedAction:
    """A parsed v4 action in grid space -> the same action in pixel space.

    rel ``move`` deltas scale by (sw/1000, sh/1000) exactly like freeroll's
    ``_scale_ordered_moves``; abs ``move_to`` grid coordinates become the pixel
    of that grid point; scroll notches, key events and typed text pass
    through untouched."""
    if not isinstance(action, OrderedAction):
        raise TypeError(f"denorm_v4 expects OrderedAction, got {type(action)!r}")
    sw, sh = screen_size
    _check_screen(sw)
    _check_screen(sh)
    out: list[OrderedPrimitive] = []
    for p in action.primitives:
        if p.kind == "move":
            out.append(replace(
                p,
                dx=round(_check_int(p.dx, lo=-GRID, hi=GRID, what="move dx") * sw / GRID),
                dy=round(_check_int(p.dy, lo=-GRID, hi=GRID, what="move dy") * sh / GRID),
            ))
        elif p.kind == "move_to":
            out.append(replace(p, x=_grid_to_px(p.x, sw), y=_grid_to_px(p.y, sh)))
        else:
            out.append(p)
    return OrderedAction(primitives=tuple(out), no_op=action.no_op)
