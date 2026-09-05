"""Ordered desktop events with relative per-axis thousandth coordinates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from desktop.geometry import DisplayGeometry
from desktop.ir import Operation

from .. import _support

#: Rendered names must be unambiguous inside the mini-program syntax. Wider than
#: the bare-token family's ``_support.EVENT_NAME_RE`` because a name here sits
#: inside parentheses instead of after a ``+``/``-`` sign. ``Primitive`` enforces
#: it on construction, so a label emitter cannot write a name ``_scan`` rejects.
NAME_RE = re.compile(r"[^\s(),;]+")
_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_PAIR_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_ARG_RE = re.compile(r"\(\s*([^\s(),;]+)\s*\)")

NO_OP = "NO_OP"
GRID = 1000


class OrderedEventsV3Error(ValueError):
    """Malformed ordered-events-v3 action text."""


def pixels_from_grid(value: int, dimension: int) -> int:
    if dimension <= 0:
        raise OrderedEventsV3Error(
            f"display dimension must be positive, got {dimension}"
        )
    return round(value / GRID * dimension)


def grid_from_pixels(value: int, dimension: int) -> int:
    if dimension <= 0:
        raise OrderedEventsV3Error(
            f"display dimension must be positive, got {dimension}"
        )
    return round(value / dimension * GRID)


def grid_delta(cursor: int, target: int, dimension: int) -> int:
    """Encode the nearest compiled landing, including display-edge clamping."""
    if dimension <= 0:
        raise OrderedEventsV3Error(
            f"display dimension must be positive, got {dimension}"
        )
    if not 0 <= cursor < dimension or not 0 <= target < dimension:
        raise OrderedEventsV3Error(
            f"cursor and target must be inside a {dimension}-pixel axis"
        )
    center = grid_from_pixels(target - cursor, dimension)

    def landing(unit: int) -> int:
        return min(max(cursor + pixels_from_grid(unit, dimension), 0), dimension - 1)

    return min(
        range(center - 2, center + 3),
        key=lambda unit: (abs(landing(unit) - target), abs(unit - center), abs(unit)),
    )


def escape(text: str) -> str:
    """Encode a ``type()`` payload. Only ``\\`` and ``"`` are escaped."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def unescape(body: str) -> str:
    """Decode a ``type()`` payload, rejecting every escape except ``\\`` and ``"``."""
    out: list[str] = []
    index = 0
    while index < len(body):
        char = body[index]
        if char == "\\":
            if index + 1 >= len(body):
                raise OrderedEventsV3Error("trailing backslash in type() payload")
            following = body[index + 1]
            if following not in ("\\", '"'):
                raise OrderedEventsV3Error(
                    f'type() accepts only the escapes \\\\ and \\" , got '
                    f"'\\{following}'. Press Return as down(Return); up(Return)."
                )
            out.append(following)
            index += 2
            continue
        if char in "\n\r\t":
            raise OrderedEventsV3Error(
                "type() payload cannot contain a control character; press Return "
                "as down(Return); up(Return)"
            )
        out.append(char)
        index += 1
    return "".join(out)


@dataclass(frozen=True)
class Primitive:
    """One ordered primitive: ``move`` / ``scroll`` / ``down`` / ``up`` / ``type``."""

    kind: str
    dx: int = 0
    dy: int = 0
    name: str = ""
    text: str = ""

    def __post_init__(self) -> None:
        if self.kind in ("move", "scroll"):
            if (
                type(self.dx) is not int
                or type(self.dy) is not int
                or self.name != ""
                or self.text != ""
            ):
                raise OrderedEventsV3Error(
                    f"{self.kind} requires exactly two integer deltas"
                )
            return
        if self.kind in ("down", "up"):
            if not (isinstance(self.name, str) and NAME_RE.fullmatch(self.name)):
                raise OrderedEventsV3Error(
                    f"{self.name!r} is not a name {self.kind}() can spell "
                    f"(must match {NAME_RE.pattern})"
                )
            if (
                type(self.dx) is not int
                or self.dx != 0
                or type(self.dy) is not int
                or self.dy != 0
                or self.text != ""
            ):
                raise OrderedEventsV3Error(f"{self.kind} requires only a name")
            return
        if self.kind == "type":
            if not isinstance(self.text, str) or not self.text:
                raise OrderedEventsV3Error("type() payload cannot be empty")
            if (
                type(self.dx) is not int
                or self.dx != 0
                or type(self.dy) is not int
                or self.dy != 0
                or self.name != ""
            ):
                raise OrderedEventsV3Error("type requires only text")
            if any(char in self.text for char in "\n\r\t"):
                raise OrderedEventsV3Error(
                    "type() payload cannot contain a control character; use key transitions"
                )
            return
        raise OrderedEventsV3Error(f"unknown primitive kind: {self.kind!r}")

    def render(self) -> str:
        if self.kind in ("move", "scroll"):
            return f"{self.kind}({self.dx},{self.dy})"
        if self.kind in ("down", "up"):
            return f"{self.kind}({self.name})"
        if self.kind == "type":
            return f'type("{escape(self.text)}")'
        raise OrderedEventsV3Error(f"unknown primitive kind: {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        if self.kind in ("move", "scroll"):
            return {"kind": self.kind, "dx": self.dx, "dy": self.dy}
        if self.kind in ("down", "up"):
            return {"kind": self.kind, "name": self.name}
        return {"kind": "type", "text": self.text}


def primitive_from_dict(value: dict[str, Any]) -> Primitive:
    if not isinstance(value, dict):
        raise OrderedEventsV3Error("primitive must be an object")
    if "kind" not in value:
        raise OrderedEventsV3Error("primitive is missing kind")
    kind = value["kind"]
    if kind in ("move", "scroll"):
        if set(value) != {"kind", "dx", "dy"}:
            raise OrderedEventsV3Error(f"invalid {kind} primitive fields")
        return Primitive(kind, dx=value["dx"], dy=value["dy"])
    if kind in ("down", "up"):
        if set(value) != {"kind", "name"}:
            raise OrderedEventsV3Error(f"invalid {kind} primitive fields")
        return Primitive(kind, name=value["name"])
    if kind == "type":
        if set(value) != {"kind", "text"}:
            raise OrderedEventsV3Error("invalid type primitive fields")
        return Primitive("type", text=value["text"])
    raise OrderedEventsV3Error(f"unknown primitive kind: {kind!r}")


@dataclass(frozen=True)
class OrderedEventsV3Action:
    """An ordered primitive program, or NO_OP."""

    primitives: tuple[Primitive, ...] = ()
    no_op: bool = False
    #: The episode-control status this turn declares, from ``_support.CONTROL_SPEC``.
    #: ``parse`` never sees a control line.
    terminate: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.primitives, tuple) or any(
            not isinstance(primitive, Primitive) for primitive in self.primitives
        ):
            raise OrderedEventsV3Error("primitives must be a tuple of Primitive values")
        if type(self.no_op) is not bool:
            raise OrderedEventsV3Error("no_op must be a boolean")
        if self.no_op != (not self.primitives):
            raise OrderedEventsV3Error("no_op must exactly identify an empty program")
        _support.terminate_status(self.terminate, error=OrderedEventsV3Error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitives": [item.to_dict() for item in self.primitives],
            "no_op": self.no_op,
            "terminate": self.terminate,
        }


def action_from_dict(value: dict[str, Any]) -> OrderedEventsV3Action:
    if not isinstance(value, dict) or set(value) != {
        "primitives",
        "no_op",
        "terminate",
    }:
        raise OrderedEventsV3Error(
            "action must contain exactly primitives, no_op and terminate"
        )
    primitives = value["primitives"]
    if not isinstance(primitives, list):
        raise OrderedEventsV3Error("primitives must be a list")
    return OrderedEventsV3Action(
        primitives=tuple(primitive_from_dict(item) for item in primitives),
        no_op=value["no_op"],
        terminate=_support.terminate_status(
            value["terminate"], error=OrderedEventsV3Error
        ),
    )


class OrderedEventsV3Codec:
    """You operate a desktop computer with a mouse and keyboard.

    Each user turn shows the current screen, with the cursor visible as a small
    arrow. Reply with exactly ONE action line per turn; only the final non-empty
    line is read, so a `<think></think>` block may precede it.

    An action line is one or more primitives separated by `; ` and applied
    strictly left to right, so movement and presses interleave exactly as
    performed: `move(12,-4); down(LMB); move(3,0); up(LMB)` moves onto a target,
    presses, drags slightly and releases, all in one turn. Anything pressed with
    `down(...)` STAYS HELD until its `up(...)`, which may be a later turn.
    """

    name = "ordered_events_v3_relative_1000_grid_v1"

    #: Empty by design: a `<think>` block legally precedes the action line, so
    #: no newline or token sequence ends the turn early.
    stop_sequences: tuple[str, ...] = ()

    @_support.production("move(dx,dy)")
    def _move(self) -> None:
        """Move by (dx, dy) thousandths of each screen axis RELATIVE to the
        cursor: dx > 0 RIGHT, dx < 0 LEFT; dy > 0 DOWN, dy < 0 UP.
        """

    @_support.production("scroll(dx,dy)")
    def _scroll(self) -> None:
        """Scroll in wheel ticks: dy > 0 scrolls up, dy < 0 down; dx is
        horizontal.
        """

    @_support.production("down(EV)")
    def _down(self) -> None:
        """Press and HOLD key/button EV. Mouse buttons are LMB (left), RMB
        (right), MMB (middle). Keyboard keys use rdev names: KeyA, Return,
        Escape, Tab, Space, Backspace, ShiftLeft, ControlLeft, Alt, MetaLeft,
        ArrowUp, ArrowDown, ArrowLeft, ArrowRight, and so on.
        """

    @_support.production("up(EV)")
    def _up(self) -> None:
        """Release key/button EV. A left click is
        `move(dx,dy); down(LMB); up(LMB)`; a chord presses in order and releases
        in reverse: `down(ControlLeft); down(KeyC); up(KeyC); up(ControlLeft)`.
        """

    @_support.production('type("TEXT")')
    def _type(self) -> None:
        """Type TEXT as one burst. Inside the quotes only two escapes exist:
        `\\\\` for a backslash and `\\"` for a quote. TEXT cannot contain a
        newline — press Return as `down(Return); up(Return)`.
        """

    @_support.production("NO_OP")
    def _no_op(self) -> None:
        """Do nothing / wait for the screen to settle."""

    def notes(self) -> None:
        """Recipes (the classic desktop actions in this format)
          move onto a target:  move(dx,dy)
          right click:         move(dx,dy); down(RMB); up(RMB)
          double click:        down(LMB); up(LMB); down(LMB); up(LMB)
          click-and-drag:      down(LMB); move(dx,dy); up(LMB)
          scroll down / up:    scroll(0,-3)   /   scroll(0,3)

        Emit one action line per turn, and nothing else except the control line
        below — no JSON, no tool calls, no other commentary.
        """

    def describe(self) -> str:
        return _support.render_spec(self)

    @property
    def digest(self) -> str:
        return _support.spec_digest(self.describe())

    def parse(self, text: str) -> OrderedEventsV3Action:
        line = _support.final_line(text)
        if line == NO_OP:
            return OrderedEventsV3Action(no_op=True)
        return OrderedEventsV3Action(primitives=self._scan(line))

    def format(self, action: OrderedEventsV3Action) -> str:
        body = (
            NO_OP
            if action.no_op
            else "; ".join(item.render() for item in action.primitives)
        )
        return _support.with_control(body, action.terminate, error=OrderedEventsV3Error)

    def compile(
        self,
        text: str,
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
    ) -> tuple[Operation, ...]:
        return self.compile_action(self.parse(text), geometry, cursor)

    def compile_action(
        self,
        action: OrderedEventsV3Action,
        geometry: DisplayGeometry,
        cursor: tuple[int, int],
    ) -> tuple[Operation, ...]:
        """Scale each relative move exactly once, then fold it onto ``cursor``."""
        if action.no_op:
            return ()
        width, height = _support.screen_size(geometry)
        operations: list[Operation] = []
        here = _support.clamp(cursor, geometry)
        for item in action.primitives:
            if item.kind == "move":
                target = _support.clamp(
                    (
                        here[0] + pixels_from_grid(item.dx, width),
                        here[1] + pixels_from_grid(item.dy, height),
                    ),
                    geometry,
                )
                if target != here:
                    operations.append(_support.move_to(target))
                here = target
            elif item.kind == "scroll":
                if item.dx or item.dy:
                    operations.append(_support.scroll(item.dx, item.dy))
            elif item.kind in ("down", "up"):
                pressed = item.kind == "down"
                button = _support.BUTTONS.get(item.name)
                if button is not None:
                    operations.append(
                        _support.mouse_down(button)
                        if pressed
                        else _support.mouse_up(button)
                    )
                else:
                    operations.append(
                        _support.key_down(item.name)
                        if pressed
                        else _support.key_up(item.name)
                    )
            elif item.kind == "type":
                operations.append(
                    _support.lower_typing(item.text, error=OrderedEventsV3Error)
                )
            else:
                raise OrderedEventsV3Error(f"unknown primitive kind: {item.kind!r}")
        return tuple(operations)

    def _scan(self, line: str) -> tuple[Primitive, ...]:
        primitives: list[Primitive] = []
        index = 0
        while True:
            while index < len(line) and line[index].isspace():
                index += 1
            if index >= len(line):
                break
            call = _CALL_RE.match(line, index)
            if call is None:
                # Nothing recognised yet means the line is not an action line of
                # this grammar at all, which a terminating turn is allowed.
                error = OrderedEventsV3Error if primitives else _support.NoAction
                raise error(
                    f"expected a primitive call, got {line[index : index + 24]!r}"
                )
            kind = call[1]
            open_paren = call.end() - 1
            if kind in ("move", "scroll"):
                pair = _PAIR_RE.match(line, open_paren)
                if pair is None:
                    raise OrderedEventsV3Error(f"{kind}() takes two integers")
                primitives.append(Primitive(kind, dx=int(pair[1]), dy=int(pair[2])))
                index = pair.end()
            elif kind in ("down", "up"):
                argument = _ARG_RE.match(line, open_paren)
                if argument is None or not NAME_RE.fullmatch(argument[1]):
                    raise OrderedEventsV3Error(f"{kind}() takes one key or button name")
                primitives.append(Primitive(kind, name=argument[1]))
                index = argument.end()
            elif kind == "type":
                payload, index = self._scan_type(line, open_paren)
                primitives.append(Primitive("type", text=payload))
            else:
                raise OrderedEventsV3Error(f"unknown primitive {kind!r}")
            while index < len(line) and line[index].isspace():
                index += 1
            if index >= len(line):
                break
            if line[index] != ";":
                raise OrderedEventsV3Error(
                    f"primitives are separated by '; ', got {line[index : index + 12]!r}"
                )
            index += 1
        if not primitives:
            raise OrderedEventsV3Error("empty action line")
        return tuple(primitives)

    def _scan_type(self, line: str, open_paren: int) -> tuple[str, int]:
        index = open_paren + 1
        while index < len(line) and line[index].isspace():
            index += 1
        if index >= len(line) or line[index] != '"':
            raise OrderedEventsV3Error('type() must wrap a "quoted" payload')
        index += 1
        body: list[str] = []
        while index < len(line):
            char = line[index]
            if char == "\\":
                body.append(char)
                index += 1
                if index >= len(line):
                    raise OrderedEventsV3Error("trailing backslash in type() payload")
                body.append(line[index])
                index += 1
                continue
            if char == '"':
                break
            body.append(char)
            index += 1
        else:
            raise OrderedEventsV3Error("type() payload is not closed")
        index += 1  # closing quote
        while index < len(line) and line[index].isspace():
            index += 1
        if index >= len(line) or line[index] != ")":
            raise OrderedEventsV3Error("type() missing closing ')'")
        return unescape("".join(body)), index + 1


CODEC = OrderedEventsV3Codec()
