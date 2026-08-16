"""Primitives shared by the grammar codecs.

Adding a grammar never edits this file. A new codec either reuses these helpers
or ignores them. There is no table, registry, or enum here that a new grammar
has to be added to — the only registration is the ``juergen.grammars`` entry
point in ``pyproject.toml``.

Nothing here knows a coordinate convention. Every helper that touches a
coordinate is handed an already-resolved absolute screen pixel. Each codec owns
its own convention (raw relative delta, normalized relative delta, absolute,
diff-of-absolute, …) and resolves it in ``compile`` before anything shared sees
it. There is no coordinate-space enum.

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
``wait``             ``(seconds: float)``       idle
===================  =========================  ==================================
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import re
import textwrap
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from desktop.geometry import DisplayGeometry
from desktop.ir import Operation, scroll_deltas


# The single point of contact with DisplayGeometry. Its field names are the
# verbatim Harbor ones — ``desktop_width`` / ``desktop_height`` for the display
# and ``window_*`` for the window inside it — so this function is where that
# naming stops and the codecs' plain (width, height) begins.


def screen_size(geometry: DisplayGeometry) -> tuple[int, int]:
    """Display size in physical screen pixels."""
    return int(geometry.desktop_width), int(geometry.desktop_height)


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


def wait(seconds: float) -> Operation:
    return Operation("wait", (float(seconds),))


# Prompt derivation follows BrowserGym's core/action pattern: the docstring is
# the spec.

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
    raises rather than being omitted from the prompt.
    """
    body: list[str] = []
    for item in productions(codec):
        body.append("  " + item.syntax)
        body.append(textwrap.indent(item.doc, "      "))
    blocks = [
        inspect.cleandoc(type(codec).__doc__),
        "\n".join(body),
        inspect.cleandoc(codec.notes.__doc__),
    ]
    return "\n\n".join(blocks) + "\n"


def render_tool_prompt(codec: Any, *, properties: dict[str, Any]) -> str:
    """Render a ``computer_use`` tool prompt from the codec's own docstrings.

    Used by the tool-call grammars, whose prompt is a function signature rather
    than a grammar listing. The ``<tools>`` envelope is preserved verbatim (an
    off-the-shelf model recognises it), but every word inside it comes from a
    docstring: the class docstring is the preamble, ``tool_description`` is the
    function description, each ``@production`` is one action's description and
    one entry of the action enum, and ``notes`` is the epilogue.
    """
    actions = productions(codec)
    described = inspect.cleandoc(codec.tool_description.__doc__)
    schema = {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": described,
            "parameters": {
                "properties": {
                    "action": {
                        "description": "The action to perform. The available "
                        "actions are:\n"
                        + "\n".join(f"* `{item.syntax}`: {item.doc}" for item in actions),
                        "enum": [item.syntax for item in actions],
                        "type": "string",
                    },
                    **properties,
                },
                "required": ["action"],
                "type": "object",
            },
        },
    }
    epilogue = inspect.cleandoc(codec.notes.__doc__)
    return (
        "\n".join(
            [
                inspect.cleandoc(type(codec).__doc__),
                "",
                "# Tools",
                "",
                "You may call one or more functions to assist with the user query.",
                "",
                "You are provided with function signatures within "
                "<tools></tools> XML tags:",
                "<tools>",
                json.dumps(schema, ensure_ascii=False),
                "</tools>",
                "",
                epilogue,
            ]
        )
        + "\n"
    )


def spec_digest(text: str) -> str:
    """sha256 of a rendered spec. Reported, never enforced."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def drift_report(codec: Any, *, producer: dict[str, str]) -> dict[str, Any]:
    """Describe how the live prompt relates to the recorded producer prompt.

    Never raises: the digest is data, not a gate. A blocking digest check was
    tried and made in-place editing of a grammar expensive enough that forking a
    worktree was cheaper.
    """
    observed = spec_digest(codec.describe())
    recorded = producer.get("prompt_sha256")
    return {
        "grammar": codec.name,
        "prompt_sha256": observed,
        "producer": dict(producer),
        # None when the grammar recorded no producer digest to compare against,
        # so "unknown" is never reported as "differs".
        "matches_producer": None if recorded is None else observed == recorded,
    }


_ACTION_MARKER_RE = re.compile(r"(?im)^\s*action\s*:\s*(.+?)\s*$")


def final_line(text: str, *, error: type[Exception] = ValueError) -> str:
    """The last non-empty line. Reasoning before the action line is legal."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text)!r}")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise error("empty action text")
    return lines[-1]


def action_line(text: str, *, error: type[Exception] = ValueError) -> str:
    """The last ``Action: <body>`` marker if present, else the last non-empty line."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text)!r}")
    markers = list(_ACTION_MARKER_RE.finditer(text))
    if markers:
        body = markers[-1].group(1).strip()
        if body:
            return body
    return final_line(text, error=error)


# the bare-token family:  dx dy scroll [ ; elements ]

BUTTONS = {"LMB": "left", "RMB": "right", "MMB": "middle"}

_EVENT_RE = re.compile(r"^([+-])([A-Za-z_][A-Za-z0-9_]*)$")
_MOVE_RE = re.compile(r"MOVE\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


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

    def render(self) -> str:
        if self.kind == "event":
            return ("+" if self.pressed else "-") + self.name
        if self.kind == "type":
            return "type(" + json.dumps(self.text, ensure_ascii=False) + ")"
        if self.kind == "move":
            assert self.delta is not None
            return f"MOVE({self.delta[0]},{self.delta[1]})"
        raise ValueError(f"unknown element kind: {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "event":
            return {"kind": "event", "name": self.name, "pressed": self.pressed}
        if self.kind == "type":
            return {"kind": "type", "text": self.text}
        if self.kind == "move":
            assert self.delta is not None
            return {"kind": "move", "delta": list(self.delta)}
        raise ValueError(f"unknown element kind: {self.kind!r}")


def element_from_dict(value: dict[str, Any]) -> Element:
    kind = value["kind"]
    if kind == "event":
        return Element("event", name=value["name"], pressed=bool(value["pressed"]))
    if kind == "type":
        return Element("type", text=value["text"])
    if kind == "move":
        dx, dy = value["delta"]
        return Element("move", delta=(int(dx), int(dy)))
    raise ValueError(f"unknown element kind: {kind!r}")


def parse_mouse_triple(
    segment: str, *, error: type[Exception] = ValueError
) -> tuple[int, int, int]:
    """``"<a> <b> <scroll>"`` -> three ints. The codec names the first two."""
    tokens = segment.strip().split()
    if len(tokens) != 3:
        raise error(f"expected exactly three mouse integers, got {tokens!r}")
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
                raise error(f"malformed MOVE element: {segment[index:index + 30]!r}")
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
        raise error(
            "type() cannot embed a newline; press Return as an event instead"
        )
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
        elif element.kind != "move":
            raise error(f"unknown element kind: {element.kind!r}")
    return operations


# lifting: Operations -> an action, the training-label direction
#
# ``compile_action`` read backwards. The lift lives on the codec so a converter
# holding absolute Operations — a recorded trajectory, a scripted oracle, a
# teacher rollout — need not know what a coordinate means or how a grammar
# spells termination; everything reusable lives here.
#
# Where a grammar cannot express what the Operations say, the lift raises. A
# ``glide_to`` inside a held button is a drag, and the converters this replaces
# degraded it into a stationary ``+LMB -LMB``, discarding the stroke.


@dataclass(frozen=True)
class Group:
    """One high-level step recovered from the Operation stream."""

    kind: str
    target: tuple[int, int] | None = None
    seconds: float = 0.0
    button: str = ""
    repeats: int = 0
    keys: tuple[str, ...] = ()
    text: str = ""
    dx: int = 0
    dy: int = 0


def _button_of(operation: Operation) -> str:
    return str(tuple(operation.args)[0])


def group_operations(
    operations: Sequence[Operation],
    *,
    geometry: DisplayGeometry,
    cursor: tuple[int, int],
    error: type[Exception] = ValueError,
) -> tuple[Group, ...]:
    """Absolute Operations -> high-level groups, in order.

    Adjacent same-button click pairs collapse into one ``click`` with a repeat
    count, a ``key_down`` run followed by exactly its reverse collapses into one
    ``chord``, and a ``glide_to`` stays a ``stroke`` so a drag survives the trip.
    Grammars that want individual transitions expand the groups again; for the
    transition spellings the collapse and the expansion are exact inverses.

    Every canonical Operation kind is accepted here. A kind that falls through to
    "unknown Operation kind" makes a whole recorded trajectory unliftable in every
    grammar at once, so these are handled explicitly:

    * ``click(button)`` — desktop's executor synthesises this itself
      (``guest_program.lower_guest_operations``), so it appears in any stream
      that has been through that lowering. It is one press/release pair, and a
      pair spelled this way coalesces with pairs spelled as
      ``mouse_down``/``mouse_up``, so ``click, click`` is a double click.
    * ``drag(x0, y0, x1, y1)`` — decomposed into ``move``, ``button_down(left)``,
      ``stroke``, ``button_up(left)``. That is the same group shape a codec
      already recognises as a drag, so ``deltatype_v2`` reaches its ``MOVE``
      form, ``ordered_events_v3`` and ``move_rel`` interleave, and
      ``native_absolute`` reaches ``left_click_drag``, while the three grammars
      with no stroke primitive raise. A zero-extent drag keeps its press and
      release, which is why ``ir.drag`` exists as its own kind.
    * ``ascii_type(text)`` — becomes the same ``type`` group as
      ``coalesced_type``. This is the one flattening in this function: no grammar
      here has two typing primitives, so per-keystroke ASCII and one clipboard
      burst cannot be told apart downstream of the lift. The mechanism changes
      (``pyautogui.write`` becomes a paste) and the recompiled stream therefore
      differs from the input; the vectors pin that as a documented-lossy case.
    """
    ops = list(operations)
    groups: list[Group] = []
    index = 0

    def click_pair(at: int) -> str | None:
        """The button of a click pair starting at ``at``, in either spelling."""
        if at < len(ops) and ops[at].kind == "click":
            return _button_of(ops[at])
        if (
            at + 1 < len(ops)
            and ops[at].kind == "mouse_down"
            and ops[at + 1].kind == "mouse_up"
            and _button_of(ops[at]) == _button_of(ops[at + 1])
        ):
            return _button_of(ops[at])
        return None

    def pair_width(at: int) -> int:
        return 1 if ops[at].kind == "click" else 2

    while index < len(ops):
        operation = ops[index]
        kind = operation.kind
        args = tuple(operation.args)
        if kind == "move_to":
            groups.append(Group("move", target=clamp((args[0], args[1]), geometry)))
            index += 1
        elif kind == "glide_to":
            seconds = float(args[2]) if len(args) > 2 else 0.0
            groups.append(
                Group(
                    "stroke",
                    target=clamp((args[0], args[1]), geometry),
                    seconds=seconds,
                )
            )
            index += 1
        elif kind == "drag":
            if len(args) < 4:
                raise error(f"drag needs (x0, y0, x1, y1), got {args!r}")
            start = clamp((args[0], args[1]), geometry)
            end = clamp((args[2], args[3]), geometry)
            groups.append(Group("move", target=start))
            groups.append(Group("button_down", button="left"))
            groups.append(Group("stroke", target=end, seconds=0.0))
            groups.append(Group("button_up", button="left"))
            index += 1
        elif kind in ("mouse_down", "click"):
            button = click_pair(index)
            if button is None:
                groups.append(Group("button_down", button=_button_of(operation)))
                index += 1
                continue
            repeats = 0
            scan = index
            while click_pair(scan) == button:
                repeats += 1
                scan += pair_width(scan)
            groups.append(Group("click", button=button, repeats=repeats))
            index = scan
        elif kind == "mouse_up":
            groups.append(Group("button_up", button=_button_of(operation)))
            index += 1
        elif kind == "key_down":
            downs: list[str] = []
            scan = index
            while scan < len(ops) and ops[scan].kind == "key_down":
                downs.append(str(tuple(ops[scan].args)[0]))
                scan += 1
            ups: list[str] = []
            after = scan
            while after < len(ops) and ops[after].kind == "key_up":
                ups.append(str(tuple(ops[after].args)[0]))
                after += 1
            if ups and ups == list(reversed(downs)):
                groups.append(Group("chord", keys=tuple(downs)))
                index = after
            else:
                groups.append(Group("key_down", keys=tuple(downs)))
                index = scan
        elif kind == "key_up":
            ups = []
            scan = index
            while scan < len(ops) and ops[scan].kind == "key_up":
                ups.append(str(tuple(ops[scan].args)[0]))
                scan += 1
            groups.append(Group("key_up", keys=tuple(ups)))
            index = scan
        elif kind in ("coalesced_type", "ascii_type"):
            # The one flattening: see this function's docstring.
            groups.append(Group("type", text=str(args[0])))
            index += 1
        elif kind == "scroll":
            # Both arities, disambiguated by desktop's own scroll_deltas --
            # the one function ir.py declares as the only place that decides.
            dx, dy = scroll_deltas(args)
            groups.append(Group("scroll", dx=int(dx), dy=int(dy)))
            index += 1
        elif kind == "wait":
            groups.append(Group("wait", seconds=float(args[0])))
            index += 1
        else:
            raise error(f"cannot lift unknown Operation kind: {kind!r}")
    return tuple(groups)


@dataclass(frozen=True)
class BareTokenPlan:
    """The head and tail of a bare-token action recovered from Operations."""

    target: tuple[int, int] | None
    scroll: int
    elements: tuple[Element, ...]
    idle: bool


def bare_token_plan(
    groups: Sequence[Group],
    *,
    cursor: tuple[int, int],
    allow_type: bool,
    allow_stroke: bool,
    allow_hscroll: bool = False,
    error: type[Exception] = ValueError,
) -> BareTokenPlan:
    """Groups -> ``[one move][one scroll][ordered transitions]``.

    That fixed shape is what every bare-token grammar can say, so anything the
    stream asks for outside it — a second move, a scroll after a transition, a
    stroke in a grammar with no stroke primitive — raises rather than being
    reordered or dropped.
    """
    target: tuple[int, int] | None = None
    scroll = 0
    elements: list[Element] = []
    here = cursor
    scrolled = False
    for group in groups:
        kind = group.kind
        if kind == "move":
            if elements or scrolled:
                raise error(
                    "a move after a scroll or a transition cannot be expressed: "
                    "the mouse move is applied before the tail"
                )
            if target is not None:
                raise error(
                    "two separate moves in one action cannot be expressed; "
                    "ordered_events_v3 can interleave moves"
                )
            target = group.target
            here = group.target or here
        elif kind == "scroll":
            if group.dx and not allow_hscroll:
                raise error("horizontal scroll cannot be expressed in this grammar")
            if scrolled:
                raise error("two scrolls in one action cannot be expressed")
            if elements:
                raise error(
                    "a scroll after a transition cannot be expressed: the scroll "
                    "is applied before the tail"
                )
            scroll = group.dy
            scrolled = True
        elif kind == "stroke":
            if not allow_stroke:
                raise error(
                    "a drag stroke cannot be expressed in this grammar: it has no "
                    "MOVE primitive, and degrading it to a stationary click would "
                    "discard the stroke"
                )
            assert group.target is not None
            elements.append(
                Element(
                    "move", delta=(group.target[0] - here[0], group.target[1] - here[1])
                )
            )
            here = group.target
        elif kind == "click":
            for _ in range(group.repeats):
                elements.append(_button_element(group.button, True, error=error))
                elements.append(_button_element(group.button, False, error=error))
        elif kind in ("button_down", "button_up"):
            elements.append(
                _button_element(group.button, kind == "button_down", error=error)
            )
        elif kind == "chord":
            elements.extend(Element("event", name=key, pressed=True) for key in group.keys)
            elements.extend(
                Element("event", name=key, pressed=False)
                for key in reversed(group.keys)
            )
        elif kind in ("key_down", "key_up"):
            pressed = kind == "key_down"
            elements.extend(
                Element("event", name=key, pressed=pressed) for key in group.keys
            )
        elif kind == "type":
            if not allow_type:
                raise error(
                    "a coalesced type() cannot be expressed in this grammar; it "
                    "spells literal text as key transitions"
                )
            if "\n" in group.text or "\r" in group.text:
                raise error(
                    "type() cannot embed a newline; press Return as an event"
                )
            elements.append(Element("type", text=group.text))
        elif kind == "wait":
            if len(groups) != 1:
                raise error(
                    "a timed wait alongside other operations cannot be expressed; "
                    "this grammar only has an idle action"
                )
            return BareTokenPlan(None, 0, (), idle=True)
        else:  # pragma: no cover - group_operations fixes the set
            raise error(f"cannot lift group kind: {kind!r}")
    idle = target is None and not scroll and not elements
    return BareTokenPlan(target, scroll, tuple(elements), idle=idle)


def _button_element(
    button: str, pressed: bool, *, error: type[Exception] = ValueError
) -> Element:
    for token, name in BUTTONS.items():
        if name == button:
            return Element("event", name=token, pressed=pressed)
    raise error(f"unsupported mouse button: {button!r}")


def click_action_name(button: str, repeats: int) -> str | None:
    """The computer_use action for N adjacent press/release pairs, or None.

    None means the schema has no single action for it (a middle-button
    double-click, say) and the caller must emit the transitions individually.
    """
    if repeats == 1:
        return {
            "left": "left_click",
            "right": "right_click",
            "middle": "middle_click",
        }.get(button)
    if button == "left" and repeats == 2:
        return "double_click"
    if button == "left" and repeats == 3:
        return "triple_click"
    return None


def terminate_status(
    value: object, *, error: type[Exception] = ValueError
) -> str | None:
    """Validate the lift's ``terminate`` argument. ``None``, success or failure.

    Each grammar then spells it its own way — a flag, a distinct FAIL token, a
    ``terminate`` call, or not at all. Three states, three spellings, no case
    folding. The bool forms this used to accept (``False`` meaning None, ``True``
    meaning success) were removed: ``True`` meaning success is how a terminate
    whose status was lost lands as a claimed success.
    """
    if value is None:
        return None
    if value in ("success", "failure"):
        return str(value)
    raise error(f"terminate must be None, 'success' or 'failure', got {value!r}")


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\s*```$")


def iter_tool_calls(text: str) -> Iterator[dict[str, Any]]:
    """Yield every ``computer_use`` arguments dict, in emission order.

    Tagged ``<tool_call>`` blocks win. If the completion has none, a bare JSON
    object (optionally inside a ``` fence) or a JSON array of objects is
    accepted, which is how the RL rollout path sees vLLM-parsed output.
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text)!r}")
    payloads: list[Any] = []
    tagged = False
    for match in _TOOL_CALL_RE.finditer(text):
        tagged = True
        try:
            payloads.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    if not tagged:
        stripped = _FENCE_CLOSE_RE.sub("", _FENCE_OPEN_RE.sub("", text.strip()))
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            payloads.append(loaded)
        elif isinstance(loaded, list):
            payloads.extend(item for item in loaded if isinstance(item, dict))
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        if payload.get("name") not in (None, "computer_use"):
            continue
        arguments = payload.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if isinstance(arguments, dict) and isinstance(arguments.get("action"), str):
            yield arguments


def render_tool_calls(calls: Sequence[dict[str, Any]]) -> str:
    """Canonical emission: one ``<tool_call>`` block per call."""
    return "\n".join(
        "<tool_call>\n"
        + json.dumps({"name": "computer_use", "arguments": call}, ensure_ascii=False)
        + "\n</tool_call>"
        for call in calls
    )


# pyautogui-style key words -> rdev names, so ``key_down``/``key_up`` args are
# one vocabulary across every grammar. Retained from the sign-of-life executor,
# including its fallback: an unrecognised name passes through unchanged.
_KEY_ALIASES = {
    "CTRL": "ControlLeft",
    "CONTROL": "ControlLeft",
    "SHIFT": "ShiftLeft",
    "ALT": "AltLeft",
    "OPTION": "AltLeft",
    "META": "MetaLeft",
    "SUPER": "MetaLeft",
    "WIN": "MetaLeft",
    "CMD": "MetaLeft",
    "ENTER": "Return",
    "RETURN": "Return",
    "ESC": "Escape",
    "ESCAPE": "Escape",
    "BACKSPACE": "Backspace",
    "TAB": "Tab",
    "SPACE": "Space",
    "DELETE": "Delete",
    "DEL": "Delete",
    "INSERT": "Insert",
    "HOME": "Home",
    "END": "End",
    "PAGEUP": "PageUp",
    "PAGEDOWN": "PageDown",
    "UP": "ArrowUp",
    "DOWN": "ArrowDown",
    "LEFT": "ArrowLeft",
    "RIGHT": "ArrowRight",
}


def normalize_key(value: object, *, error: type[Exception] = ValueError) -> str:
    """A tool-call key word -> its rdev name.

    Only the tool-call grammars call this. The bare-token grammars already
    speak rdev, and normalising them would silently rewrite e.g. ``Alt`` into
    ``AltLeft``.
    """
    if not isinstance(value, str) or not value.strip():
        raise error("key must be a non-empty string")
    raw = value.strip()
    alias = _KEY_ALIASES.get(raw.upper())
    if alias is not None:
        return alias
    if len(raw) == 1 and raw.isalpha():
        return f"Key{raw.upper()}"
    return raw


# No handler tables live here, and none should be added. There is no dispatch
# engine in desktop for a grammar to contribute a ``dict[str, Handler]`` to. A
# shared ``match`` over grammar-specific action names would be wrong, because
# that set is open per grammar; a codec's job ends at ``compile``, and the
# Operation vocabulary on the far side is closed: a pointer moves, a button
# transitions, a wheel turns, text arrives. Lowering it is a fixed
# ``if kind ==`` chain in ``desktop.execute.guest_program`` over something no
# grammar extends.


#: The preamble of both bare-line paired-eval arms, ``compact_raw`` and
#: ``native_absolute_control``, held in one string and assigned to both codec
#: classes' ``__doc__``. A constant rather than two docstrings because the arms
#: had drifted: the same sentence wrapped at a different column in each, so the
#: two prompts tokenised differently.
MATCHED_ARM_PREAMBLE = """You operate a real desktop VM with a mouse and keyboard.

First write one short sentence describing the action, without numbers. Then
output exactly one bare action line. Only the final non-empty line is read as
the action; do not use JSON or tool calls.

Complete exactly the next semantic step shown in the user context.
"""

#: The productions the two arms share verbatim, likewise held once. Only the
#: mouse-triple productions differ between the arms.
MATCHED_ARM_PRESS = """Press NAME. Mouse buttons are LMB, RMB, MMB. Keyboard keys use rdev
names: Return, Tab, Backspace, Escape, ControlLeft, ShiftLeft, KeyA, ...
"""

MATCHED_ARM_RELEASE = """Release NAME. A chord presses in order and releases in reverse."""

MATCHED_ARM_TYPE = """Type TEXT as one coalesced burst. TEXT is a JSON string. It must NOT
contain a newline — press Return as an event (`+Return -Return`).
"""

MATCHED_ARM_NOTES = """Emit exactly one action line per turn."""

#: Which member of a paired arm takes which shared string. Applied by
#: ``apply_matched_arm_prose`` after each arm's class body.
_MATCHED_ARM_PROSE = {
    "_press": MATCHED_ARM_PRESS,
    "_release": MATCHED_ARM_RELEASE,
    "_type": MATCHED_ARM_TYPE,
    "notes": MATCHED_ARM_NOTES,
}


def apply_matched_arm_prose(codec_class: type) -> type:
    """Install the shared paired-arm prose onto one arm's class.

    Called by ``compact_raw`` and ``native_absolute_control`` and by nothing
    else, so the shared half of the two prompts has exactly one source.
    """
    codec_class.__doc__ = MATCHED_ARM_PREAMBLE
    for member, doc in _MATCHED_ARM_PROSE.items():
        function = getattr(codec_class, member, None)
        if function is None:
            raise AttributeError(
                f"{codec_class.__name__} is a matched arm but has no {member!r}"
            )
        function.__doc__ = doc
    return codec_class
