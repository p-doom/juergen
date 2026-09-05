"""Primitives shared by the two retained grammar codecs.

Coordinates reach this module as resolved screen pixels. Deltatype owns raw
relative deltas; ordered-events owns normalized relative deltas.

The Operation vocabulary the codecs emit (all coordinates are absolute screen
pixels, already clamped to the display):

===================  =========================  ==================================
kind                 args                       meaning
===================  =========================  ==================================
``move_to``          ``(x: int, y: int)``       instant absolute move
``glide_to``         ``(x, y, seconds: float)`` timed absolute move (drag stroke)
``mouse_down``       ``(button: str)``          ``left`` / ``right`` / ``middle``
``mouse_up``         ``(button: str)``          as above
``scroll``           ``(dx: int, dy: int)``     wheel ticks, +dy up, +dx right
``key_down``         ``(name: str)``            rdev key name
``key_up``           ``(name: str)``            rdev key name
``coalesced_type``   ``(text: str)``            one atomic burst of literal text
===================  =========================  ==================================
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import re
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from desktop.geometry import DisplayGeometry
from desktop.ir import Operation

# The single point of contact with DisplayGeometry. Its field names are the
# verbatim Harbor ones — ``desktop_width`` / ``desktop_height`` for the display
# and ``window_*`` for the window inside it — so this function is where that
# naming stops and the codecs' plain (width, height) begins.


def screen_size(geometry: DisplayGeometry) -> tuple[int, int]:
    """Display size in physical screen pixels."""
    size = int(geometry.desktop_width), int(geometry.desktop_height)
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"display geometry must be positive, got {size[0]}x{size[1]}")
    return size


def clamp(point: tuple[int, int], geometry: DisplayGeometry) -> tuple[int, int]:
    """Clamp an absolute pixel onto the display."""
    width, height = screen_size(geometry)
    x, y = point
    return (max(0, min(width - 1, int(x))), max(0, min(height - 1, int(y))))


def move_to(point: tuple[int, int]) -> Operation:
    return Operation("move_to", (int(point[0]), int(point[1])))


def glide_to(point: tuple[int, int], seconds: float) -> Operation:
    return Operation("glide_to", (int(point[0]), int(point[1]), float(seconds)))


def mouse_down(button: str) -> Operation:
    return Operation("mouse_down", (str(button),))


def mouse_up(button: str) -> Operation:
    return Operation("mouse_up", (str(button),))


def scroll(dx: int, dy: int) -> Operation:
    return Operation("scroll", (int(dx), int(dy)))


def key_down(name: str) -> Operation:
    return Operation("key_down", (str(name),))


def key_up(name: str) -> Operation:
    return Operation("key_up", (str(name),))


def coalesced_type(text: str) -> Operation:
    return Operation("coalesced_type", (str(text),))


_ORDER = itertools.count()


@dataclass(frozen=True)
class Production:
    """One rendered grammar production: its surface syntax and its spec text."""

    member: str
    syntax: str
    doc: str
    order: int


def production(syntax: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a codec member as a grammar production rendered into the prompt.

    The decorated member's docstring is the only specification of that
    production.
    """

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        function.__grammar_production__ = (syntax, next(_ORDER))  # type: ignore[attr-defined]
        return function

    return decorate


def productions(codec: Any) -> tuple[Production, ...]:
    """Every production of ``codec``, in definition order, subclass wins."""
    found: dict[str, Production] = {}
    for cls in reversed(type(codec).__mro__):
        for name, member in vars(cls).items():
            marker = getattr(member, "__grammar_production__", None)
            if marker is None:
                continue
            syntax, order = marker
            found[name] = Production(
                member=name,
                syntax=syntax,
                doc=inspect.cleandoc(member.__doc__ or ""),
                order=order,
            )
    return tuple(sorted(found.values(), key=lambda item: item.order))


def render_spec(codec: Any) -> str:
    """Render the codec's system prompt from its own docstrings.

    Preamble = the codec class docstring. Body = one block per ``@production``.
    Epilogue = the ``notes`` docstring. All three are mandatory; a missing one
    raises rather than being omitted from the prompt. ``CONTROL_SPEC`` closes it:
    the episode-control channel is not a production of any grammar, so it is
    appended here rather than declared by both codecs.
    """
    preamble = inspect.getdoc(type(codec))
    if not preamble:
        raise ValueError(f"{type(codec).__name__} must have a class docstring")
    items = productions(codec)
    if not items:
        raise ValueError(f"{type(codec).__name__} must declare a production")
    body: list[str] = []
    for item in items:
        if not item.syntax.strip():
            raise ValueError(f"{type(codec).__name__}.{item.member} has empty syntax")
        if not item.doc:
            raise ValueError(
                f"{type(codec).__name__}.{item.member} must have a docstring"
            )
        body.append("  " + item.syntax)
        body.append(textwrap.indent(item.doc, "      "))
    notes = inspect.getdoc(codec.notes)
    if not notes:
        raise ValueError(f"{type(codec).__name__}.notes must have a docstring")
    blocks = [
        preamble,
        "\n".join(body),
        notes,
        CONTROL_SPEC,
    ]
    return "\n\n".join(blocks) + "\n"


def spec_digest(text: str) -> str:
    """sha256 of a rendered spec."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NoAction(ValueError):
    """The text holds no action of this grammar, as opposed to a broken one.

    Codecs raise this before recognizing an action and their own error after
    recognizing a malformed one.
    """


def final_line(text: str) -> str:
    """The last non-empty line. Reasoning before the action line is legal.

    Both retained grammars pick their action this way.
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text)!r}")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise NoAction("empty action text")
    return lines[-1]


# the bare-token family:  dx dy scroll [ ; elements ]

BUTTONS = {"LMB": "left", "RMB": "right", "MMB": "middle"}

#: The key/button names a bare-token tail can spell. ``Element`` enforces it on
#: construction, so a label emitter cannot write a name this family's parser then
#: calls malformed -- the pipeline's ``format_action`` did exactly that, spelling
#: an unmapped keycode ``KC_-1``.
EVENT_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_EVENT_RE = re.compile(rf"^([+-])({EVENT_NAME_RE.pattern})$")
_MOVE_RE = re.compile(r"MOVE\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")

#: A head token this family would try to read as a coordinate. ``1.5`` is a
#: rejected coordinate, ``Done.`` is not a coordinate at all -- see ``NoAction``.
_MOUSE_HEAD_RE = re.compile(r"[-+.]?\d")


@dataclass(frozen=True)
class Element:
    """One ordered element of a bare-token action's tail.

    ``kind`` is ``event`` (``name`` + ``pressed``), ``type`` (``text``), or
    ``move`` (``delta``, a second relative delta used by drag forms).
    """

    kind: str
    name: str = ""
    pressed: bool | None = None
    text: str = ""
    delta: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.kind == "event":
            if not (isinstance(self.name, str) and EVENT_NAME_RE.fullmatch(self.name)):
                raise ValueError(
                    f"{self.name!r} is not a name a bare-token action can spell "
                    f"(must match {EVENT_NAME_RE.pattern})"
                )
            if (
                type(self.pressed) is not bool
                or self.text != ""
                or self.delta is not None
            ):
                raise ValueError("event elements require only name and pressed")
            return
        if self.kind == "type":
            if (
                not isinstance(self.text, str)
                or self.name != ""
                or self.pressed is not None
                or self.delta is not None
            ):
                raise ValueError("type elements require only text")
            return
        if self.kind == "move":
            if (
                not isinstance(self.delta, tuple)
                or len(self.delta) != 2
                or any(type(value) is not int for value in self.delta)
                or self.name != ""
                or self.pressed is not None
                or self.text != ""
            ):
                raise ValueError("move elements require exactly two integer deltas")
            return
        raise ValueError(f"unknown element kind: {self.kind!r}")

    def render(self) -> str:
        if self.kind == "event":
            return ("+" if self.pressed else "-") + self.name
        if self.kind == "type":
            return "type(" + json.dumps(self.text, ensure_ascii=False) + ")"
        if self.kind == "move":
            assert self.delta is not None
            return f"MOVE({self.delta[0]},{self.delta[1]})"
        raise ValueError(f"unknown element kind: {self.kind!r}")


def parse_mouse_triple(
    segment: str, *, error: type[Exception] = ValueError
) -> tuple[int, int, int]:
    """``"<a> <b> <scroll>"`` -> three ints. The codec names the first two."""
    tokens = segment.strip().split()
    complaint = f"expected exactly three mouse integers, got {tokens!r}"
    if tokens and _MOUSE_HEAD_RE.match(tokens[0]) is None:
        raise NoAction(complaint)
    if len(tokens) != 3:
        raise error(complaint)
    try:
        first, second, third = (int(token) for token in tokens)
    except ValueError as exc:
        raise error(f"mouse tokens are not integers: {tokens!r}") from exc
    return first, second, third


def scan_elements(
    segment: str,
    *,
    allow_type: bool = True,
    allow_move: bool = False,
    error: type[Exception] = ValueError,
) -> tuple[Element, ...]:
    """Scan the post-``;`` tail into ordered elements.

    ``type("…")`` wraps a JSON string, so its payload may contain spaces, ``;``,
    ``+`` and escaped quotes; everything else is whitespace separated.
    """
    elements: list[Element] = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(segment):
        if segment[index].isspace():
            index += 1
            continue
        if allow_move and segment.startswith("MOVE", index):
            match = _MOVE_RE.match(segment, index)
            if match is None or (
                match.end() < len(segment) and not segment[match.end()].isspace()
            ):
                raise error(f"malformed MOVE element: {segment[index : index + 30]!r}")
            elements.append(Element("move", delta=(int(match[1]), int(match[2]))))
            index = match.end()
            continue
        if allow_type and segment.startswith("type(", index):
            start = index + len("type(")
            while start < len(segment) and segment[start].isspace():
                start += 1
            if start >= len(segment) or segment[start] != '"':
                raise error("type(...) must wrap a JSON string")
            try:
                value, end = decoder.raw_decode(segment, start)
            except json.JSONDecodeError as exc:
                raise error(f"bad type() JSON string: {exc}") from exc
            if not isinstance(value, str):
                raise error("type() payload must be a JSON string")
            while end < len(segment) and segment[end].isspace():
                end += 1
            if end >= len(segment) or segment[end] != ")":
                raise error("type(...) missing closing ')'")
            elements.append(Element("type", text=value))
            index = end + 1
            continue
        end = index
        while end < len(segment) and not segment[end].isspace():
            end += 1
        token = segment[index:end]
        match = _EVENT_RE.fullmatch(token)
        if match is None:
            raise error(f"malformed element: {token!r}")
        elements.append(Element("event", name=match[2], pressed=match[1] == "+"))
        index = end
    return tuple(elements)


def render_elements(elements: Sequence[Element]) -> str:
    """Canonical tail: ``" ; "`` then space-separated elements, or ``""``."""
    if not elements:
        return ""
    return " ; " + " ".join(element.render() for element in elements)


def lower_typing(text: str, *, error: type[Exception] = ValueError) -> Operation:
    """One coalesced type. Rejects an embedded newline.

    The executor cannot type a newline inside a burst, and a double-escaped
    ``\\n`` types two literal characters instead of pressing Return. Return is an
    event, never a character.
    """
    if "\n" in text or "\r" in text:
        raise error("type() cannot embed a newline; press Return as an event instead")
    return coalesced_type(text)


def lower_transitions(
    elements: Sequence[Element], *, error: type[Exception] = ValueError
) -> list[Operation]:
    """Lower ``event`` and ``type`` elements in order. ``move`` is the codec's."""
    operations: list[Operation] = []
    for element in elements:
        if element.kind == "type":
            operations.append(lower_typing(element.text, error=error))
        elif element.kind == "event":
            button = BUTTONS.get(element.name)
            if button is not None:
                operations.append(
                    mouse_down(button) if element.pressed else mouse_up(button)
                )
            else:
                operations.append(
                    key_down(element.name) if element.pressed else key_up(element.name)
                )
        else:
            raise error(f"cannot lower element kind: {element.kind!r}")
    return operations


def terminate_status(
    value: object, *, error: type[Exception] = ValueError
) -> str | None:
    """Validate a ``terminate`` status. ``None``, success or failure.

    Three states, one spelling — see ``CONTROL_SPEC`` — and no case folding. The
    bool forms this used to accept (``False`` meaning None, ``True`` meaning
    success) were removed: ``True`` meaning success is how a terminate whose
    status was lost lands as a claimed success.
    """
    if value is None:
        return None
    if value in ("success", "failure"):
        return str(value)
    raise error(f"terminate must be None, 'success' or 'failure', got {value!r}")


# The episode-control channel.
#
# Ending an episode dispatches nothing: it is a message to the episode driver, not
# something the computer does. A grammar's ``compile`` emits ``Operation``s, so a
# control token inside one lowers to zero operations — a category error, and every
# grammar reinvented it. One spelling lives here, rendered into both prompts and
# read back once before either codec sees the text.

CONTROL_TOKEN = "TERMINATE"

_CONTROL_LINE_RE = re.compile(rf"(?:^|\n){CONTROL_TOKEN}: (success|failure)\n?\Z")

#: Appended to every grammar's prompt. It names no coordinate, no key and no
#: action, which is what makes it grammar-independent.
CONTROL_SPEC = f"""Ending the episode
  {CONTROL_TOKEN}: success
      The goal has been achieved. Stop.
  {CONTROL_TOKEN}: failure
      The goal cannot be achieved. Stop.
  Either is a line of its own and must be the LAST line of your reply. An action
  on the lines before it is performed first, and the episode then ends. These two
  lines are the only way to stop, and they have no other spelling."""


@dataclass(frozen=True)
class Control:
    """The episode-control decision read off one completion.

    ``body`` is the completion with the control removed, and is the only text a
    codec is given — so nothing on the far side of a termination can be parsed or
    dispatched. ``ignored`` remains zero because the control line must be last.
    """

    status: str | None
    body: str
    ignored: int = 0


def render_control(status: str, *, error: type[Exception] = ValueError) -> str:
    """The one control line. The status is validated, so it cannot be lost."""
    return f"{CONTROL_TOKEN}: {terminate_status(status, error=error)}"


def with_control(
    body: str, status: str | None, *, error: type[Exception] = ValueError
) -> str:
    """An action's text plus its control line, which always comes last.

    An empty ``body`` yields the control line alone.
    """
    if status is None:
        return body
    line = render_control(status, error=error)
    return f"{body}\n{line}" if body else line


def split_control(text: str) -> Control:
    """Remove one exact final ``TERMINATE: success|failure`` control line."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text)!r}")
    if CONTROL_TOKEN not in text:
        return Control(None, text)
    if (
        text.count(CONTROL_TOKEN) != 1
        or (match := _CONTROL_LINE_RE.search(text)) is None
    ):
        raise ValueError(
            "TERMINATE must occur exactly once as the literal final line "
            "'TERMINATE: success' or 'TERMINATE: failure'"
        )
    return Control(match[1], text[: match.start()])
