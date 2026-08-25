"""Parser for the BC action token format emitted by Qwen3-VL-2B BC checkpoints.

Action grammar (from data_pipeline/stage_a_prepare.py):

    "NO_OP"                              all-zero, no key transitions
    "<dx> <dy> <scroll>"                 mouse-only
    "<dx> <dy> <scroll> ; +K1 -K2"       mouse + key transitions

Event tokens are ``+<name>`` for press / ``-<name>`` for release.
Mouse buttons appear as ``LMB`` (left), ``RMB`` (right), ``MMB`` (middle).
Keyboard events use rdev key names (e.g. ``KeyA``, ``ShiftLeft``,
``ControlLeft``, ``Return``).

The parser is intentionally lenient on whitespace and tolerates a trailing
newline / EOS token. It raises on truly malformed input so callers can
count parse errors as a separate failure mode.

Two further formats are parsed here, one function per format — freeroll picks
one by ``--action_format`` and never mixes them:

  * ``parse_computer_use_tool_call`` — Qwen3-VL native ``<tool_call>`` JSON.
  * ``parse_ordered_action`` — the ordered_events_v2/v3 mini-program
    (``move(4,-1); down(LMB); up(LMB)``), inverse of
    ``pipeline/crowdcast/lib/action_format.py``. Unlike the
    aggregate grammar above it is order-preserving, so ``move -> click ->
    move`` survives in a single turn; see the section at the bottom of this
    module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


# Mouse button names → X11 button code (matches XTest convention).
_MOUSE_BUTTON_CODES = {
    "LMB": 1,  # left
    "MMB": 2,  # middle
    "RMB": 3,  # right
}


@dataclass(frozen=True)
class KeyEvent:
    """A single key/button transition emitted by the model.

    ``kind`` is one of ``"press"``, ``"release"``.
    ``what`` is the unmodified token (e.g., ``"LMB"``, ``"KeyA"``).
    ``mouse_button`` is non-None for mouse-button events.
    """
    kind: str  # "press" | "release"
    what: str
    mouse_button: int | None  # 1/2/3 if a mouse button, else None


@dataclass(frozen=True)
class Action:
    """Parsed action: mouse delta + scroll + ordered key/button events.

    ``no_op`` is True iff the action was the literal ``"NO_OP"`` token. In
    that case dx, dy, scroll are all 0 and events is empty.
    """
    dx: int
    dy: int
    scroll: int
    events: tuple[KeyEvent, ...]
    no_op: bool

    @property
    def has_left_click_press(self) -> bool:
        """True iff any event is a press of the left mouse button."""
        return any(e.kind == "press" and e.what == "LMB" for e in self.events)

    @property
    def has_left_click_release(self) -> bool:
        return any(e.kind == "release" and e.what == "LMB" for e in self.events)


@dataclass(frozen=True)
class ComputerUseCall:
    """A parsed `<tool_call>` for the computer_use function."""

    name: str
    arguments: dict


_EVENT_RE = re.compile(r"^([+-])([A-Za-z_][A-Za-z_0-9]*)$")


def parse_action(text: str) -> Action:
    """Parse a single action token.

    Args:
        text: the assistant's response. Trailing whitespace is stripped.
              Extra trailing tokens after the action (newline, EOS) are
              tolerated; everything after the first ``\\n`` is ignored.

    Returns:
        An ``Action`` value object.

    Raises:
        ValueError if the text doesn't match the action grammar at all.
    """
    if not isinstance(text, str):
        raise TypeError(f"parse_action expects str, got {type(text)!r}")
    # Strip whitespace; cut to first newline so trailing chatter is ignored.
    text = text.strip()
    if "\n" in text:
        text = text.split("\n", 1)[0].strip()
    if not text:
        raise ValueError("empty action text")

    if text == "NO_OP":
        return Action(dx=0, dy=0, scroll=0, events=(), no_op=True)

    # Split off the key-events segment by ``;``.
    if ";" in text:
        mouse_part, key_part = text.split(";", 1)
    else:
        mouse_part, key_part = text, ""

    # Parse the mouse triplet.
    mouse_tokens = mouse_part.strip().split()
    if len(mouse_tokens) != 3:
        raise ValueError(
            f"expected 3 mouse tokens (dx dy scroll), got "
            f"{len(mouse_tokens)}: {mouse_part!r}"
        )
    try:
        dx, dy, scroll = (int(t) for t in mouse_tokens)
    except ValueError as e:
        raise ValueError(
            f"mouse tokens not int-parseable: {mouse_tokens!r}"
        ) from e

    # Parse the event segment.
    events: list[KeyEvent] = []
    if key_part.strip():
        for tok in key_part.strip().split():
            m = _EVENT_RE.match(tok)
            if not m:
                raise ValueError(f"malformed event token: {tok!r}")
            sign, name = m.group(1), m.group(2)
            kind = "press" if sign == "+" else "release"
            mouse_button = _MOUSE_BUTTON_CODES.get(name)
            events.append(KeyEvent(kind=kind, what=name, mouse_button=mouse_button))

    return Action(dx=dx, dy=dy, scroll=scroll, events=tuple(events), no_op=False)


# Matches a line shaped like ``Action: <body>`` (case-insensitive).
# Used by ``parse_action_tolerant`` to locate the action line when the
# model emits CoT prose with an explicit ``Action:`` marker (the
# convention used by ``cot_directions_v1`` and similar prompts).
_ACTION_MARKER_RE = re.compile(
    r"(?im)^\s*action\s*:\s*(.+?)\s*$"
)


def parse_action_tolerant(text: str) -> Action:
    """Like ``parse_action`` but tolerates prose preceding the action.

    Strategy:
      1. Try strict ``parse_action`` on the full text. BC-clean output
         takes this path with no behavioral change.
      2. If that fails, look for the last ``Action: <body>`` marker
         (case-insensitive, line-anchored) and strict-parse the body.
      3. If no marker is present, strict-parse the last non-blank line.

    The action body is ALWAYS routed through strict ``parse_action`` —
    we never grab digits from the middle of arbitrary prose. This
    catches the cot_directions_v1 pattern
        "Reasoning: ...\\nAction: 200 -100 0 ; +LMB -LMB"
    and the simpler "prose then action on last line" pattern
        "The target is to the right.\\n100 0 0 ; +LMB"
    but rejects responses where the action is buried mid-sentence —
    those are genuine format failures and should be counted as such.
    """
    if not isinstance(text, str):
        raise TypeError(f"parse_action_tolerant expects str, got {type(text)!r}")
    # Fast path: strict parser succeeds on the full text → return.
    try:
        return parse_action(text)
    except (ValueError, TypeError):
        pass
    # Look for an explicit ``Action:`` marker. Last match wins (model's
    # final answer is at the end of its reasoning, not in worked examples
    # the prompt may have echoed back).
    marker_matches = list(_ACTION_MARKER_RE.finditer(text))
    if marker_matches:
        return parse_action(marker_matches[-1].group(1))
    # Fall back to the last non-blank line.
    lines = [ln for ln in (s.strip() for s in text.splitlines()) if ln]
    if not lines:
        raise ValueError(f"empty response in tolerant parse of {text!r}")
    return parse_action(lines[-1])


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)


def _json_candidates(text: str) -> list[str]:
    """Return possible JSON payloads for a computer-use tool call."""
    candidates = [m.group(1).strip() for m in _TOOL_CALL_RE.finditer(text)]
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    if stripped.startswith("{"):
        candidates.append(stripped)
    return candidates


def parse_computer_use_tool_call(text: str) -> ComputerUseCall:
    """Parse the OpenAI computer-use example `<tool_call>` output.

    Expected shape:
        <tool_call>
        {"name": "computer_use", "arguments": {"action": "..."}}
        </tool_call>

    The last valid `computer_use` call wins, matching the tolerant action
    parser's behavior of preferring the model's final answer if it echoed
    examples before the actual action.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"parse_computer_use_tool_call expects str, got {type(text)!r}"
        )
    parsed: ComputerUseCall | None = None
    errors: list[str] = []
    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as e:
            errors.append(str(e))
            continue
        if not isinstance(payload, dict):
            errors.append("tool call payload is not an object")
            continue
        name = payload.get("name")
        arguments = payload.get("arguments")
        if name != "computer_use":
            errors.append(f"unexpected tool call name: {name!r}")
            continue
        if not isinstance(arguments, dict):
            errors.append("computer_use arguments is not an object")
            continue
        action = arguments.get("action")
        if not isinstance(action, str) or not action:
            errors.append("computer_use arguments.action missing or not a string")
            continue
        parsed = ComputerUseCall(name=name, arguments=arguments)
    if parsed is None:
        suffix = f": {'; '.join(errors)}" if errors else ""
        raise ValueError(f"no valid computer_use tool call found{suffix}")
    return parsed


# --------------------------------------------------------------------------
# ordered_events_v2 / v3 (the ORDERED mini-program format)
# --------------------------------------------------------------------------
# Inverse of pipeline/crowdcast/lib/action_format.py's
# OrderedFormatter (v2) / OrderedTypingFormatter (v3), implementing
# ORDERED_EVENTS_V3_GRAMMAR verbatim:
#
#     line       = "NO_OP" / primitive *("; " primitive)
#     primitive  = move / scroll / down / up / type
#     move       = "move(" int "," int ")"
#     scroll     = "scroll(" int "," int ")"
#     down       = "down(" NAME ")"
#     up         = "up(" NAME ")"
#     type       = "type(" DQUOTE chars DQUOTE ")"
#
# v2 is the v3 grammar minus ``type``, so ONE parser serves both: a v2-trained
# checkpoint simply never emits ``type(...)``. TERMINATE is not part of the
# grammar -- stage 04 OVERWRITES the final turn's action with it, so it always
# arrives alone and freeroll._is_terminate intercepts it before parsing.
#
# The primitive list is order-significant: ``move -> click -> move`` in one
# turn is the whole point of the format, and the aggregate ``Action`` (single
# dx/dy/scroll + a flat event tuple) cannot represent it. Hence a separate
# value type rather than a lossy projection onto ``Action``.

_ORDERED_VECTOR_RE = re.compile(r"(move|scroll)\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
# NAME is "any char except whitespace ( ) , ;" per the grammar.
_ORDERED_INPUT_RE = re.compile(r"(down|up)\(\s*([^\s(),;]+)\s*\)")
# chars = 1*(escape / plain); escape = "\" ("\" / DQUOTE). The payload may
# contain "; ", so a line is NEVER safely split on the separator -- it must be
# scanned left to right, which is what _parse_ordered_primitives does.
_ORDERED_TYPE_RE = re.compile(r'type\("((?:\\.|[^"\\])*)"\)')

_ORDERED_NO_OP = "NO_OP"


def _unescape_typed_text(payload: str) -> str:
    """Inverse of action_format._escape_typed_text (``\\\\`` and ``\\"`` only).

    A backslash before any other character is not an escape in this grammar,
    so it is kept literally (both characters pass through) rather than being
    silently swallowed.
    """
    out: list[str] = []
    i, n = 0, len(payload)
    while i < n:
        ch = payload[i]
        if ch == "\\" and i + 1 < n and payload[i + 1] in ('\\', '"'):
            out.append(payload[i + 1])
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


@dataclass(frozen=True)
class OrderedPrimitive:
    """One primitive of an ordered mini-program.

    Mirrors action_format.ActionPrimitive (the render side) but lives here so
    eval/ stays independent of the data_pipeline package, and adds
    ``mouse_button`` so dispatch does not re-derive it.

    ``kind`` is one of ``"move"``, ``"scroll"``, ``"down"``, ``"up"``,
    ``"type"``. ``dx``/``dy`` are set for move/scroll, ``input_name`` for
    down/up, ``text`` (unescaped) for type.
    """

    kind: str
    dx: int | None = None
    dy: int | None = None
    input_name: str | None = None
    text: str | None = None
    mouse_button: int | None = None  # 1/2/3 for LMB/MMB/RMB on down/up

    def render(self) -> str:
        """Re-render this primitive in the wire format (round-trip helper)."""
        if self.kind in ("move", "scroll"):
            return f"{self.kind}({self.dx},{self.dy})"
        if self.kind == "type":
            escaped = self.text.replace("\\", "\\\\").replace('"', '\\"')
            return f'type("{escaped}")'
        return f"{self.kind}({self.input_name})"

    def as_key_event(self) -> KeyEvent:
        """The equivalent aggregate-format KeyEvent (down/up primitives only)."""
        if self.kind not in ("down", "up"):
            raise ValueError(f"{self.kind!r} primitive is not a key/button event")
        return KeyEvent(
            kind="press" if self.kind == "down" else "release",
            what=self.input_name,
            mouse_button=self.mouse_button,
        )


@dataclass(frozen=True)
class OrderedAction:
    """A parsed ordered mini-program: primitives in the order performed."""

    primitives: tuple[OrderedPrimitive, ...]
    no_op: bool

    @property
    def key_events(self) -> tuple[KeyEvent, ...]:
        """The down/up primitives as aggregate-format events, order preserved.

        Lets consumers that only care about key/button transitions (e.g.
        left-click detection) treat an ordered action like an aggregate one.
        Motion and typing are dropped, so this is NOT a lossless projection.
        """
        return tuple(
            p.as_key_event() for p in self.primitives if p.kind in ("down", "up")
        )

    @property
    def has_left_click_press(self) -> bool:
        return any(
            p.kind == "down" and p.input_name == "LMB" for p in self.primitives
        )

    @property
    def has_left_click_release(self) -> bool:
        return any(
            p.kind == "up" and p.input_name == "LMB" for p in self.primitives
        )

    def render(self) -> str:
        """Re-render the whole line (exact inverse of parse for valid input)."""
        if self.no_op:
            return _ORDERED_NO_OP
        return "; ".join(p.render() for p in self.primitives)


def _parse_ordered_primitives(body: str) -> list[OrderedPrimitive]:
    """Scan ``body`` left to right into primitives, or raise ValueError.

    Scanning (rather than splitting on ``"; "``) is required: a
    ``type("a; b")`` payload legally contains the separator.
    """
    primitives: list[OrderedPrimitive] = []
    pos, n = 0, len(body)
    while True:
        while pos < n and body[pos].isspace():
            pos += 1
        if pos >= n:
            break

        if m := _ORDERED_VECTOR_RE.match(body, pos):
            kind, dx, dy = m.group(1), int(m.group(2)), int(m.group(3))
            primitives.append(OrderedPrimitive(kind=kind, dx=dx, dy=dy))
        elif m := _ORDERED_INPUT_RE.match(body, pos):
            kind, name = m.group(1), m.group(2)
            primitives.append(OrderedPrimitive(
                kind=kind,
                input_name=name,
                mouse_button=_MOUSE_BUTTON_CODES.get(name),
            ))
        elif m := _ORDERED_TYPE_RE.match(body, pos):
            text = _unescape_typed_text(m.group(1))
            if not text:
                raise ValueError('type("") payload is empty')
            primitives.append(OrderedPrimitive(kind="type", text=text))
        else:
            raise ValueError(
                f"not an ordered primitive at offset {pos}: {body[pos:pos + 40]!r}"
            )

        pos = m.end()
        while pos < n and body[pos].isspace():
            pos += 1
        if pos >= n:
            break
        if body[pos] != ";":
            raise ValueError(
                f"expected ';' between primitives at offset {pos}: "
                f"{body[pos:pos + 40]!r}"
            )
        pos += 1
        # A trailing ';' with nothing after it is malformed, not an empty
        # primitive -- the loop top breaks out and we catch it here.
        if not body[pos:].strip():
            raise ValueError("trailing ';' with no primitive after it")

    if not primitives:
        raise ValueError("no primitives in ordered action")
    return primitives


def parse_ordered_action(text: str) -> OrderedAction:
    """Parse one ordered_events_v2/v3 action line.

    Args:
        text: the assistant's response. Leading/trailing whitespace is
              stripped and everything after the first newline is ignored
              (trailing chatter / EOS), matching ``parse_action``.

    Returns:
        An ``OrderedAction``. ``NO_OP`` yields ``no_op=True`` and no
        primitives.

    Raises:
        ValueError if the line does not match the ordered grammar.
    """
    if not isinstance(text, str):
        raise TypeError(f"parse_ordered_action expects str, got {type(text)!r}")
    text = text.strip()
    if "\n" in text:
        text = text.split("\n", 1)[0].strip()
    if not text:
        raise ValueError("empty action text")
    if text == _ORDERED_NO_OP:
        return OrderedAction(primitives=(), no_op=True)
    return OrderedAction(
        primitives=tuple(_parse_ordered_primitives(text)), no_op=False
    )


def parse_ordered_action_tolerant(text: str) -> OrderedAction:
    """Like ``parse_ordered_action`` but tolerates prose before the action.

    Same three-step strategy as ``parse_action_tolerant`` (strict on the full
    text, then the last ``Action:`` marker, then the last non-blank line), and
    the same guarantee: the candidate is ALWAYS routed through the strict
    parser, so primitives are never scraped out of the middle of prose. This
    matters for the thinking-SFT prompts, whose replies are reasoning followed
    by the action line.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"parse_ordered_action_tolerant expects str, got {type(text)!r}"
        )
    try:
        return parse_ordered_action(text)
    except (ValueError, TypeError):
        pass
    if marker_matches := list(_ACTION_MARKER_RE.finditer(text)):
        return parse_ordered_action(marker_matches[-1].group(1))
    lines = [ln for ln in (s.strip() for s in text.splitlines()) if ln]
    if not lines:
        raise ValueError(f"empty response in tolerant parse of {text!r}")
    return parse_ordered_action(lines[-1])
