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
"""

from __future__ import annotations

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
