"""Parsers for the action token formats emitted by BC/SFT checkpoints.

Canonical (aggregate) grammar (from data_pipeline/stage_a_prepare.py):

    "NO_OP"                              all-zero, no key transitions
    "<dx> <dy> <scroll>"                 mouse-only
    "<dx> <dy> <scroll> ; +K1 -K2"       mouse + key transitions

Event tokens are ``+<name>`` for press / ``-<name>`` for release.
Mouse buttons appear as ``LMB`` (left), ``RMB`` (right), ``MMB`` (middle).
Keyboard events use rdev key names (e.g. ``KeyA``, ``ShiftLeft``,
``ControlLeft``, ``Return``).

Ordered grammar (``ordered_events_v3``, superset of ``ordered_events_v2``;
see ORDERED_EVENTS_V3_GRAMMAR in
data_pipeline/realigned_pipeline/lib/action_format.py):

    "NO_OP"
    primitive("; " primitive)*   with primitive one of
        move(dx,dy) | scroll(dx,dy) | down(NAME) | up(NAME) | type("text")

Inside ``type("...")`` the only escapes are ``\\\\`` (backslash) and ``\\"``
(double quote); the payload may contain '; ', parentheses and commas, so
splitting into primitives must be quote-aware (see
``_split_ordered_primitives``).

Short-goal ordered grammar (``ordered_events_v4_rel`` / ``ordered_events_v4_abs``;
binding contract: ``ORDERED_EVENTS_V4_GRAMMAR`` in eval/shortgoal_grammar.py) —
the v3 line grammar with a single-int ``scroll(<notches>)`` and one
arm-divergent mouse primitive, ``move(dx,dy)`` in per-axis thousandths of the
screen (rel) or ``move_to(x,y)`` on the 0-1000 grid (abs); see
``parse_ordered_v4_action``.

Native tool-call grammar (``computer_use_rel_v1``; binding contract:
data_pipeline/realigned_pipeline/system_prompts/cua_v4_thinking.txt):

    [<think>...</think>] 1*( "<tool_call>" json "</tool_call>" )

with each json body ``{"name": "computer_use", "arguments": {...}}`` and the
arguments validated against the cua_v4 action enum + per-action argument
schema (see ``parse_computer_use_action``). Blocks execute strictly in
order; the parser is whitespace-tolerant inside/between blocks but rejects
any non-whitespace content outside them.

All parsers are intentionally lenient on whitespace and tolerate a trailing
newline / EOS token. They raise on truly malformed input so callers can
count parse errors as a separate failure mode.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class OrderedPrimitive:
    """One primitive of an ordered/native action program.

    ``ordered_events_v3`` lines use kinds ``"move"``, ``"scroll"``,
    ``"down"``, ``"up"``, ``"type"``. ``dx``/``dy`` are set for
    move/scroll, ``name`` (and ``mouse_button`` for LMB/RMB/MMB) for
    down/up, and ``text`` — the UNESCAPED typed characters — for type.

    ``computer_use_rel_v1`` tool calls reuse ``move`` (mouse_move_rel — so
    resolution scaling of cursor deltas stays a kind=="move" rewrite),
    ``scroll`` (scroll -> dy=pixels, hscroll -> dx=pixels; positive up /
    right, matching pyautogui) and ``type``, and add:

    - ``"click"``        ``name`` in left/right/middle, ``count`` 1/2/3
                         (left/right/middle_click, double_click, triple_click)
    - ``"button_down"``/``"button_up"``  ``name`` in left/right/middle
    - ``"key_combo"``    ``keys``: pyautogui names pressed in order,
                         released in reverse (the `key` action)
    - ``"key_down"``/``"key_up"``        ``name``: one pyautogui key name
    - ``"wait"``         no fields; NEVER dispatched (behaves like NO_OP)
    - ``"terminate"``    ``status`` in success/failure; NEVER dispatched
                         (the rollout loop's stop condition)

    ``ordered_events_v4_abs`` adds ``"move_to"`` with ``x``/``y`` — a 0-1000
    grid point as parsed, VM pixels after ``shortgoal_grammar.denorm_v4``.
    Every other kind is shared, so ``x``/``y`` are None everywhere else.
    """
    kind: str
    dx: int | None = None
    dy: int | None = None
    name: str | None = None
    mouse_button: int | None = None  # 1/2/3 if a mouse button, else None
    text: str | None = None
    keys: tuple[str, ...] | None = None  # key_combo only
    count: int | None = None  # click only: 1/2/3
    status: str | None = None  # terminate only: "success" | "failure"
    x: int | None = None
    y: int | None = None


@dataclass(frozen=True)
class OrderedAction:
    """Parsed ordered action: primitives in emission order (empty for NO_OP)."""
    primitives: tuple[OrderedPrimitive, ...]
    no_op: bool

    @property
    def has_left_click_press(self) -> bool:
        return any(
            (p.kind == "down" and p.name == "LMB")
            or (p.kind in ("click", "button_down") and p.name == "left")
            for p in self.primitives
        )


# Grammar: interior whitespace is not emitted, but we stay lenient around
# the comma (matching the canonical parser's whitespace tolerance).
_ORDERED_DELTA_RE = re.compile(r"^(move|scroll)\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$")
# NAME per grammar: 1+ chars, none of whitespace ( ) , ; — same charset as
# the pipeline's _INPUT_NAME_RE. A double quote can never reach here: the
# quote-aware splitter already rejected it as an unterminated string.
_ORDERED_KEY_RE = re.compile(r"^(down|up)\(([^\s(),;]+)\)$")


def _split_ordered_primitives(line: str) -> list[str]:
    """Split one action line on ``;`` while respecting ``type("...")`` strings.

    A character-level scan: inside a double-quoted string a backslash
    consumes the following character (escape validity is checked later by
    ``_parse_type_primitive``), so an escaped ``\\"`` never closes the
    string and ``; `` inside a typed payload never splits.
    """
    parts: list[str] = []
    buf: list[str] = []
    in_string = False
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if in_string:
            if c == "\\":
                if i + 1 >= n:
                    raise ValueError(
                        f"dangling escape at end of type() string: {line!r}"
                    )
                buf.append(c)
                buf.append(line[i + 1])
                i += 2
                continue
            buf.append(c)
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            buf.append(c)
            i += 1
            continue
        if c == ";":
            parts.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    if in_string:
        raise ValueError(f"unterminated type() string: {line!r}")
    parts.append("".join(buf).strip())
    return parts


def _parse_type_primitive(prim: str) -> str:
    """Unescape the payload of a ``type("...")`` primitive.

    Only ``\\\\`` and ``\\"`` are legal escapes (ORDERED_EVENTS_V3_GRAMMAR);
    anything else — including a truncated trailing backslash — is malformed.
    """
    body = prim[len("type("):]
    if not body.startswith('"'):
        raise ValueError(f"type() payload must be double-quoted: {prim!r}")
    chars: list[str] = []
    i = 1
    closed_at = None
    while i < len(body):
        c = body[i]
        if c == "\\":
            nxt = body[i + 1] if i + 1 < len(body) else None
            if nxt not in ("\\", '"'):
                raise ValueError(
                    f"invalid escape in type() (only \\\\ and \\\" are legal): {prim!r}"
                )
            chars.append(nxt)
            i += 2
            continue
        if c == '"':
            closed_at = i
            break
        chars.append(c)
        i += 1
    if closed_at is None:
        raise ValueError(f"unterminated type() string: {prim!r}")
    if body[closed_at + 1:] != ")":
        raise ValueError(f"trailing characters after type() close quote: {prim!r}")
    if not chars:
        raise ValueError(f"empty type() payload (grammar requires >=1 char): {prim!r}")
    return "".join(chars)


def _parse_ordered_primitive(prim: str) -> OrderedPrimitive:
    if m := _ORDERED_DELTA_RE.match(prim):
        return OrderedPrimitive(kind=m.group(1), dx=int(m.group(2)), dy=int(m.group(3)))
    if m := _ORDERED_KEY_RE.match(prim):
        name = m.group(2)
        return OrderedPrimitive(
            kind=m.group(1), name=name, mouse_button=_MOUSE_BUTTON_CODES.get(name)
        )
    if prim.startswith("type("):
        return OrderedPrimitive(kind="type", text=_parse_type_primitive(prim))
    raise ValueError(f"malformed ordered primitive: {prim!r}")


def parse_ordered_action(text: str) -> OrderedAction:
    """Parse one ``ordered_events_v3`` (or v2 — a strict subset) action line.

    Args:
        text: the assistant's response. Trailing whitespace is stripped;
              everything after the first ``\\n`` is ignored (safe: the
              type() payload grammar admits no newline, so a legal action
              always fits on the first line).

    Returns:
        An ``OrderedAction`` value object.

    Raises:
        ValueError if the text doesn't match the grammar — including a
        truncated trailing primitive (``down(LM``, ``type("hel``), which
        must never be partially dispatched.
    """
    if not isinstance(text, str):
        raise TypeError(f"parse_ordered_action expects str, got {type(text)!r}")
    text = text.strip()
    if "\n" in text:
        text = text.split("\n", 1)[0].strip()
    if not text:
        raise ValueError("empty action text")

    if text == "NO_OP":
        return OrderedAction(primitives=(), no_op=True)

    parts = _split_ordered_primitives(text)
    if any(not p for p in parts):
        raise ValueError(f"empty primitive in ordered action: {text!r}")
    return OrderedAction(
        primitives=tuple(_parse_ordered_primitive(p) for p in parts),
        no_op=False,
    )


def parse_ordered_action_tolerant(text: str) -> OrderedAction:
    """Like ``parse_ordered_action`` but tolerates prose preceding the action.

    Mirrors ``parse_action_tolerant``: strict parse of the full text first
    (which already cuts to the first line), then strict parse of the last
    non-blank line. The action body is ALWAYS routed through the strict
    parser — malformed primitives are never partially accepted.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"parse_ordered_action_tolerant expects str, got {type(text)!r}"
        )
    try:
        return parse_ordered_action(text)
    except ValueError:
        pass
    lines = [ln for ln in (s.strip() for s in text.splitlines()) if ln]
    if not lines:
        raise ValueError(f"empty response in tolerant parse of {text!r}")
    return parse_ordered_action(lines[-1])


ORDERED_V4_ARM_REL = "ordered_events_v4_rel"
ORDERED_V4_ARM_ABS = "ordered_events_v4_abs"
ORDERED_V4_ARMS = frozenset({ORDERED_V4_ARM_REL, ORDERED_V4_ARM_ABS})
ORDERED_V4_SCALE = 1000

_ORDERED_V4_MOVE_RE = re.compile(r"^move\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$")
_ORDERED_V4_MOVE_TO_RE = re.compile(r"^move_to\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
_ORDERED_V4_SCROLL_RE = re.compile(r"^scroll\(\s*(-?\d+)\s*\)$")


def _parse_ordered_v4_primitive(prim: str, *, arm: str) -> OrderedPrimitive:
    if m := _ORDERED_V4_MOVE_RE.match(prim):
        if arm != ORDERED_V4_ARM_REL:
            raise ValueError(f"move() is only legal in {ORDERED_V4_ARM_REL}: {prim!r}")
        dx, dy = int(m.group(1)), int(m.group(2))
        if abs(dx) > ORDERED_V4_SCALE or abs(dy) > ORDERED_V4_SCALE:
            raise ValueError(f"move() delta outside +-{ORDERED_V4_SCALE}: {prim!r}")
        if dx == 0 and dy == 0:
            raise ValueError(f"move(0,0) is never emitted: {prim!r}")
        return OrderedPrimitive(kind="move", dx=dx, dy=dy)
    if m := _ORDERED_V4_MOVE_TO_RE.match(prim):
        if arm != ORDERED_V4_ARM_ABS:
            raise ValueError(f"move_to() is only legal in {ORDERED_V4_ARM_ABS}: {prim!r}")
        x, y = int(m.group(1)), int(m.group(2))
        if x > ORDERED_V4_SCALE or y > ORDERED_V4_SCALE:
            raise ValueError(f"move_to() outside the 0-{ORDERED_V4_SCALE} grid: {prim!r}")
        return OrderedPrimitive(kind="move_to", x=x, y=y)
    if m := _ORDERED_V4_SCROLL_RE.match(prim):
        notches = int(m.group(1))
        if notches == 0:
            raise ValueError(f"scroll(0) is never emitted: {prim!r}")
        return OrderedPrimitive(kind="scroll", dx=0, dy=notches)
    if m := _ORDERED_KEY_RE.match(prim):
        name = m.group(2)
        return OrderedPrimitive(
            kind=m.group(1), name=name, mouse_button=_MOUSE_BUTTON_CODES.get(name)
        )
    if prim.startswith("type("):
        return OrderedPrimitive(kind="type", text=_parse_type_primitive(prim))
    raise ValueError(f"malformed ordered_events_v4 primitive: {prim!r}")


def parse_ordered_v4_action(text: str, *, arm: str) -> OrderedAction:
    """Parse one ``ordered_events_v4`` action line for ``arm``.

    Args:
        text: the assistant's response. Trailing whitespace is stripped;
              everything after the first ``\\n`` is ignored (as in v3 — the
              type() payload grammar admits no newline).
        arm: ``ordered_events_v4_rel`` or ``ordered_events_v4_abs``. The other
             arm's mouse primitive is a parse error, never a silent accept.

    Returns:
        An ``OrderedAction`` whose primitives are ``move`` (normalized deltas
        in dx/dy), ``move_to`` (grid coordinates in x/y), ``scroll`` (dx=0,
        dy=notches), ``down``, ``up``, ``type``.

    Raises:
        ValueError on anything outside the grammar: the wrong arm's mouse
        primitive, out-of-range coordinates, ``scroll(0)``, ``move(0,0)``,
        two-argument v3 ``scroll(dx,dy)``, an empty line, trailing garbage, a
        truncated primitive, or ``TERMINATE`` (a whole-line reply the caller
        recognises before parsing, exactly as with v3).
    """
    if not isinstance(text, str):
        raise TypeError(f"parse_ordered_v4_action expects str, got {type(text)!r}")
    if arm not in ORDERED_V4_ARMS:
        raise ValueError(f"unknown ordered_events_v4 arm: {arm!r}")
    text = text.strip()
    if "\n" in text:
        text = text.split("\n", 1)[0].strip()
    if not text:
        raise ValueError("empty action text")

    if text == "NO_OP":
        return OrderedAction(primitives=(), no_op=True)

    parts = _split_ordered_primitives(text)
    if any(not p for p in parts):
        raise ValueError(f"empty primitive in ordered action: {text!r}")
    return OrderedAction(
        primitives=tuple(_parse_ordered_v4_primitive(p, arm=arm) for p in parts),
        no_op=False,
    )


def parse_ordered_v4_action_tolerant(text: str, *, arm: str) -> OrderedAction:
    """Like ``parse_ordered_v4_action`` but tolerates prose preceding the line.

    Mirrors ``parse_ordered_action_tolerant``: strict parse of the full text
    first (which already cuts to the first line), then strict parse of the last
    non-blank line. The body is ALWAYS routed through the strict parser.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"parse_ordered_v4_action_tolerant expects str, got {type(text)!r}"
        )
    if arm not in ORDERED_V4_ARMS:
        raise ValueError(f"unknown ordered_events_v4 arm: {arm!r}")
    try:
        return parse_ordered_v4_action(text, arm=arm)
    except ValueError:
        pass
    lines = [ln for ln in (s.strip() for s in text.splitlines()) if ln]
    if not lines:
        raise ValueError(f"empty response in tolerant parse of {text!r}")
    return parse_ordered_v4_action(lines[-1], arm=arm)


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
# computer_use_rel_v1 — Qwen-native tool calls with a RELATIVE mouse.
# Binding contract: system_prompts/cua_v4_thinking.txt (action enum +
# per-action argument schema + emission format).
# --------------------------------------------------------------------------

_CUA_V4_TOOL_CALL_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL
)

_CUA_V4_BUTTONS = ("left", "right", "middle")
_CUA_V4_STATUSES = ("success", "failure")

# action -> required argument names. Per the contract every listed argument
# is "Required only by action=X", so required == allowed: anything else
# (besides "action" itself) is an unknown argument and rejected.
_CUA_V4_REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    "key": ("keys",),
    "type": ("text",),
    "mouse_move_rel": ("delta",),
    "left_click": (),
    "right_click": (),
    "middle_click": (),
    "double_click": (),
    "triple_click": (),
    "button_down": ("button",),
    "button_up": ("button",),
    "key_down": ("key",),
    "key_up": ("key",),
    "scroll": ("pixels",),
    "hscroll": ("pixels",),
    "wait": ("time",),
    "terminate": ("status",),
}

# click-family action -> (pyautogui button, click count).
_CUA_V4_CLICKS = {
    "left_click": ("left", 1),
    "right_click": ("right", 1),
    "middle_click": ("middle", 1),
    "double_click": ("left", 2),
    "triple_click": ("left", 3),
}


def _strip_leading_think(text: str) -> str:
    """Drop a leading ``<think>...</think>`` block (mirrors freeroll's
    ``_strip_think``): opener present -> cut through the FIRST closer;
    unterminated opener -> the whole reply is thought (return "");
    dangling closer with no opener -> cut through it; else unchanged."""
    s = text.lstrip()
    if s.startswith("<think>"):
        end = s.find("</think>")
        if end == -1:
            return ""
        return s[end + len("</think>"):]
    end = s.find("</think>")
    if end != -1 and "<think>" not in s[:end]:
        return s[end + len("</think>"):]
    return text


def _cua_v4_number(arguments: dict, key: str, *, where: str) -> float:
    v = arguments[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where}: {key!r} must be a number, got {v!r}")
    return float(v)


def _cua_v4_primitive(arguments: dict, *, where: str) -> OrderedPrimitive:
    """Validate one computer_use arguments object against the cua_v4 schema
    and convert it to an ``OrderedPrimitive``."""
    action = arguments.get("action")
    if not isinstance(action, str):
        raise ValueError(f"{where}: 'action' missing or not a string: {action!r}")
    required = _CUA_V4_REQUIRED_ARGS.get(action)
    if required is None:
        raise ValueError(
            f"{where}: unknown action {action!r} "
            f"(known: {sorted(_CUA_V4_REQUIRED_ARGS)})"
        )
    extra = set(arguments) - {"action"} - set(required)
    if extra:
        raise ValueError(
            f"{where}: unknown argument(s) {sorted(extra)} for action {action!r}"
        )
    missing = set(required) - set(arguments)
    if missing:
        raise ValueError(
            f"{where}: missing required argument(s) {sorted(missing)} "
            f"for action {action!r}"
        )

    if action == "mouse_move_rel":
        delta = arguments["delta"]
        if not isinstance(delta, (list, tuple)) or len(delta) != 2:
            raise ValueError(f"{where}: delta must be [dx, dy], got {delta!r}")
        for v in delta:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(
                    f"{where}: delta values must be numbers, got {delta!r}"
                )
        return OrderedPrimitive(
            kind="move", dx=round(delta[0]), dy=round(delta[1])
        )
    if action == "key":
        keys = arguments["keys"]
        if not isinstance(keys, list) or not keys:
            raise ValueError(
                f"{where}: keys must be a non-empty array of strings, got {keys!r}"
            )
        for k in keys:
            if not isinstance(k, str) or not k.strip():
                raise ValueError(
                    f"{where}: keys entries must be non-empty strings, got {k!r}"
                )
        return OrderedPrimitive(kind="key_combo", keys=tuple(keys))
    if action == "type":
        text = arguments["text"]
        if not isinstance(text, str):
            raise ValueError(f"{where}: text must be a string, got {text!r}")
        return OrderedPrimitive(kind="type", text=text)
    if action in _CUA_V4_CLICKS:
        button, count = _CUA_V4_CLICKS[action]
        return OrderedPrimitive(kind="click", name=button, count=count)
    if action in ("button_down", "button_up"):
        button = arguments["button"]
        if button not in _CUA_V4_BUTTONS:
            raise ValueError(
                f"{where}: button must be one of {_CUA_V4_BUTTONS}, got {button!r}"
            )
        return OrderedPrimitive(kind=action, name=button)
    if action in ("key_down", "key_up"):
        key = arguments["key"]
        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                f"{where}: key must be a non-empty string, got {key!r}"
            )
        return OrderedPrimitive(kind=action, name=key)
    if action == "scroll":
        pixels = _cua_v4_number(arguments, "pixels", where=where)
        return OrderedPrimitive(kind="scroll", dx=0, dy=round(pixels))
    if action == "hscroll":
        pixels = _cua_v4_number(arguments, "pixels", where=where)
        return OrderedPrimitive(kind="scroll", dx=round(pixels), dy=0)
    if action == "wait":
        _cua_v4_number(arguments, "time", where=where)  # validated, ignored
        return OrderedPrimitive(kind="wait")
    if action == "terminate":
        status = arguments["status"]
        if status not in _CUA_V4_STATUSES:
            raise ValueError(
                f"{where}: status must be one of {_CUA_V4_STATUSES}, got {status!r}"
            )
        return OrderedPrimitive(kind="terminate", status=status)
    raise AssertionError(f"unhandled cua_v4 action {action!r}")  # pragma: no cover


def _parse_cua_v4_block(body: str) -> OrderedPrimitive:
    """Parse one ``<tool_call>`` body (raw JSON text) into a primitive."""
    where = f"tool call {body.strip()!r}"
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"{where}: invalid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise ValueError(f"{where}: payload is not a JSON object")
    unknown_top = set(payload) - {"name", "arguments"}
    if unknown_top:
        raise ValueError(f"{where}: unknown top-level key(s) {sorted(unknown_top)}")
    if payload.get("name") != "computer_use":
        raise ValueError(f"{where}: name must be 'computer_use', got {payload.get('name')!r}")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError(f"{where}: arguments must be a JSON object, got {arguments!r}")
    return _cua_v4_primitive(arguments, where=where)


def parse_computer_use_action(text: str) -> OrderedAction:
    """Parse a ``computer_use_rel_v1`` reply: 1+ ``<tool_call>`` blocks.

    An optional leading ``<think>...</think>`` block is stripped first.
    After that the reply must consist EXCLUSIVELY of ``<tool_call>`` blocks
    (whitespace-tolerant inside and between blocks); zero blocks or any
    non-whitespace content outside the blocks raises a ValueError naming
    the offending fragment. Every block must be valid — a malformed block
    anywhere rejects the whole reply, so a truncated tail is never
    partially dispatched.

    Returns an ``OrderedAction`` whose primitives are the tool calls in
    emission order (see ``OrderedPrimitive`` for the kind mapping), so
    dispatch shares ``dispatch_ordered_action`` with the ordered formats.
    """
    if not isinstance(text, str):
        raise TypeError(f"parse_computer_use_action expects str, got {type(text)!r}")
    s = _strip_leading_think(text)
    matches = list(_CUA_V4_TOOL_CALL_RE.finditer(s))
    if not matches:
        raise ValueError(f"no <tool_call> blocks found in {s.strip()!r}")
    prev_end = 0
    for m in matches:
        fragment = s[prev_end:m.start()].strip()
        if fragment:
            raise ValueError(
                f"unexpected content outside <tool_call> blocks: {fragment!r}"
            )
        prev_end = m.end()
    tail = s[prev_end:].strip()
    if tail:
        raise ValueError(f"unexpected content outside <tool_call> blocks: {tail!r}")
    return OrderedAction(
        primitives=tuple(_parse_cua_v4_block(m.group(1)) for m in matches),
        no_op=False,
    )


def parse_computer_use_action_tolerant(text: str) -> OrderedAction:
    """Like ``parse_computer_use_action`` but tolerates prose around blocks.

    Mirrors the other tolerant parsers: strict parse first; on failure,
    extract whatever ``<tool_call>`` blocks exist and parse those, ignoring
    surrounding chatter. Every block is still routed through the strict
    block parser — a malformed block anywhere rejects the whole reply
    (never partially dispatched), and zero blocks is a hard error.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"parse_computer_use_action_tolerant expects str, got {type(text)!r}"
        )
    try:
        return parse_computer_use_action(text)
    except ValueError:
        pass
    s = _strip_leading_think(text)
    matches = list(_CUA_V4_TOOL_CALL_RE.finditer(s))
    if not matches:
        raise ValueError(
            f"no <tool_call> blocks in tolerant parse of {s.strip()!r}"
        )
    return OrderedAction(
        primitives=tuple(_parse_cua_v4_block(m.group(1)) for m in matches),
        no_op=False,
    )


# --------------------------------------------------------------------------
# computer_use_rel_step_v1 — finite screen-relative steps, atomic input.
# Binding contract: action_specs/computer_use_rel_step_v1.json and
# system_prompts/cua_rel_step_v1_thinking.txt.
# --------------------------------------------------------------------------

_REL_STEP_SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "data_pipeline/realigned_pipeline/action_specs/computer_use_rel_step_v1.json"
)
_REL_STEP_SPEC = json.loads(_REL_STEP_SPEC_PATH.read_text())
if _REL_STEP_SPEC.get("format") != "computer_use_rel_step_v1":  # pragma: no cover
    raise RuntimeError(f"wrong relative-step spec at {_REL_STEP_SPEC_PATH}")

_REL_STEP_VALID_DELTAS = frozenset(
    (int(scale) * int(direction[0]), int(scale) * int(direction[1]))
    for scale in _REL_STEP_SPEC["movement_scales"]
    for direction in _REL_STEP_SPEC["directions"]
)
_REL_STEP_SCROLL_STEPS = frozenset(int(v) for v in _REL_STEP_SPEC["scroll_steps"])
_REL_STEP_MAX_CALLS = int(_REL_STEP_SPEC["max_tool_calls"])
_REL_STEP_MAX_TYPE_CHARS = int(_REL_STEP_SPEC["typing_max_chars"])
_REL_STEP_REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    "mouse_move_rel": ("delta",),
    "left_click": (),
    "right_click": (),
    "middle_click": (),
    "double_click": (),
    "triple_click": (),
    "button_down": ("button",),
    "button_up": ("button",),
    "key": ("keys",),
    "type": ("text",),
    "scroll": ("steps",),
    "hscroll": ("steps",),
    "wait": (),
    "terminate": ("status",),
}


def _rel_step_primitive(arguments: dict, *, where: str) -> OrderedPrimitive:
    """Validate one rel-step arguments object without coercion or repair."""
    action = arguments.get("action")
    if not isinstance(action, str) or action not in _REL_STEP_REQUIRED_ARGS:
        raise ValueError(f"{where}: unknown or missing action {action!r}")
    required = set(_REL_STEP_REQUIRED_ARGS[action])
    extra = set(arguments) - {"action"} - required
    missing = required - set(arguments)
    if extra:
        raise ValueError(f"{where}: unknown argument(s) {sorted(extra)} for {action!r}")
    if missing:
        raise ValueError(f"{where}: missing argument(s) {sorted(missing)} for {action!r}")

    if action == "mouse_move_rel":
        delta = arguments["delta"]
        if (
            not isinstance(delta, list)
            or len(delta) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in delta)
            or tuple(delta) not in _REL_STEP_VALID_DELTAS
        ):
            raise ValueError(
                f"{where}: delta must be an exact fixed relative step, got {delta!r}"
            )
        return OrderedPrimitive(kind="move", dx=delta[0], dy=delta[1])
    if action == "key":
        keys = arguments["keys"]
        if (
            not isinstance(keys, list)
            or not keys
            or any(not isinstance(k, str) or not k.strip() for k in keys)
        ):
            raise ValueError(f"{where}: keys must be a non-empty string array")
        return OrderedPrimitive(kind="key_combo", keys=tuple(keys))
    if action == "type":
        value = arguments["text"]
        if not isinstance(value, str) or not value or len(value) > _REL_STEP_MAX_TYPE_CHARS:
            raise ValueError(
                f"{where}: text must contain 1..{_REL_STEP_MAX_TYPE_CHARS} characters"
            )
        return OrderedPrimitive(kind="type", text=value)
    if action in _CUA_V4_CLICKS:
        button, count = _CUA_V4_CLICKS[action]
        return OrderedPrimitive(kind="click", name=button, count=count)
    if action in ("button_down", "button_up"):
        button = arguments["button"]
        if button not in _CUA_V4_BUTTONS:
            raise ValueError(f"{where}: invalid button {button!r}")
        return OrderedPrimitive(kind=action, name=button)
    if action in ("scroll", "hscroll"):
        steps = arguments["steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or steps not in _REL_STEP_SCROLL_STEPS:
            raise ValueError(
                f"{where}: steps must be one of {sorted(_REL_STEP_SCROLL_STEPS)}, got {steps!r}"
            )
        return OrderedPrimitive(
            kind="scroll",
            dx=steps if action == "hscroll" else 0,
            dy=steps if action == "scroll" else 0,
        )
    if action == "wait":
        return OrderedPrimitive(kind="wait")
    if action == "terminate":
        if arguments["status"] != "success":
            raise ValueError(f"{where}: terminate status must be 'success'")
        return OrderedPrimitive(kind="terminate", status="success")
    raise AssertionError(f"unhandled relative-step action {action!r}")  # pragma: no cover


def _parse_rel_step_block(body: str) -> OrderedPrimitive:
    where = f"tool call {body.strip()!r}"
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"{where}: invalid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise ValueError(f"{where}: payload must be an object")
    if set(payload) != {"name", "arguments"}:
        raise ValueError(f"{where}: top-level keys must be exactly name and arguments")
    if payload["name"] != _REL_STEP_SPEC["tool_name"]:
        raise ValueError(f"{where}: name must be {_REL_STEP_SPEC['tool_name']!r}")
    arguments = payload["arguments"]
    if not isinstance(arguments, dict):
        raise ValueError(f"{where}: arguments must be an object")
    return _rel_step_primitive(arguments, where=where)


def _validate_rel_step_program(primitives: tuple[OrderedPrimitive, ...]) -> None:
    """Enforce response-level atomicity before any primitive is dispatched."""
    if len(primitives) > _REL_STEP_MAX_CALLS:
        raise ValueError(
            f"too many tool calls: {len(primitives)} > {_REL_STEP_MAX_CALLS}"
        )
    terminations = [i for i, p in enumerate(primitives) if p.kind == "terminate"]
    if terminations:
        if len(primitives) != 1:
            raise ValueError("terminate must be the only tool call in a reply")
        return

    state_calls = [i for i, p in enumerate(primitives) if p.kind in ("button_down", "button_up")]
    move_calls = [i for i, p in enumerate(primitives) if p.kind == "move"]
    if state_calls:
        if len(state_calls) != 2:
            raise ValueError("drag must contain exactly one button_down and one button_up")
        down_i, up_i = state_calls
        down, up = primitives[down_i], primitives[up_i]
        if (
            down.kind != "button_down"
            or up.kind != "button_up"
            or down.name != up.name
            or down_i != 0
            or up_i != len(primitives) - 1
            or not move_calls
            or any(not (down_i < i < up_i) for i in move_calls)
            or any(p.kind not in ("button_down", "move", "button_up") for p in primitives)
        ):
            raise ValueError(
                "drag must be button_down, one or more fixed moves, button_up for the same button"
            )
        return

    if len(move_calls) > 1:
        raise ValueError("normal replies may contain at most one mouse move")
    if move_calls and any(p.kind != "move" for p in primitives):
        raise ValueError("a normal mouse move must be the only action in its reply")


def parse_computer_use_rel_step_action(text: str) -> OrderedAction:
    """Strict, transactional parser for ``computer_use_rel_step_v1``.

    An optional leading think block is accepted. Everything after it must be
    valid tool-call blocks, and the whole program is validated before the
    returned action can reach the executor. There is deliberately no tolerant
    parser for this safety-critical format.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"parse_computer_use_rel_step_action expects str, got {type(text)!r}"
        )
    s = _strip_leading_think(text)
    matches = list(_CUA_V4_TOOL_CALL_RE.finditer(s))
    if not matches:
        raise ValueError(f"no <tool_call> blocks found in {s.strip()!r}")
    prev_end = 0
    for match in matches:
        fragment = s[prev_end:match.start()].strip()
        if fragment:
            raise ValueError(f"unexpected content outside tool calls: {fragment!r}")
        prev_end = match.end()
    tail = s[prev_end:].strip()
    if tail:
        raise ValueError(f"unexpected content outside tool calls: {tail!r}")
    primitives = tuple(_parse_rel_step_block(m.group(1)) for m in matches)
    _validate_rel_step_program(primitives)
    return OrderedAction(primitives=primitives, no_op=all(p.kind == "wait" for p in primitives))
