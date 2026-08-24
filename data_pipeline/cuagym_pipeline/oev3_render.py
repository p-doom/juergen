"""Render ``ordered_events_v3`` action lines programmatically.

Ported from data_pipeline/realigned_pipeline/lib/action_format.py (grammar
constant + type()-escaping) on the shortgoal branch; the inverse parsers are
``parse_ordered_action`` / ``parse_ordered_action_tolerant`` in
eval/action_parser.py. This module only BUILDS grammar-conformant lines —
it deliberately carries none of the keylog-folding formatter.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

NO_OP = "NO_OP"
TERMINATE = "TERMINATE"

_INPUT_NAME_RE = re.compile(r"^[^\s(),;]+$")

ORDERED_EVENTS_V3_GRAMMAR = r'''
line       = "NO_OP" / primitive *("; " primitive)
primitive  = move / scroll / down / up / type
move       = "move(" int "," int ")"         ; accumulated motor-tick mouse delta
scroll     = "scroll(" int "," int ")"       ; accumulated motor-tick scroll delta
down       = "down(" NAME ")"                ; key/button press
up         = "up(" NAME ")"                  ; key/button release
type       = "type(" DQUOTE chars DQUOTE ")" ; run of >=1 typed characters
int        = ["-"] 1*DIGIT                   ; move(0,0)/scroll(0,0) never emitted
NAME       = 1*name-char                     ; name-char = any char except
                                             ; whitespace "(" ")" "," ";"
chars      = 1*char                          ; char = escape / plain
escape     = "\" ("\" / DQUOTE)              ; \\ -> backslash, \" -> double quote
plain      = any printable US-keyboard character except "\" and DQUOTE:
             letters, digits, space, `~!@#$%^&*()-_=+[]{}|;:'<>,./?
                                             ; Return/Tab are down/up, never
                                             ; typed, so chars has no newline/tab
'''


def _escape_typed_text(text: str) -> str:
    """Escape for the ``type("...")`` payload: only ``\\`` and ``"``."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def render_move(dx: int, dy: int) -> str:
    return f"move({int(dx)},{int(dy)})"


def render_scroll(dx: int, dy: int) -> str:
    return f"scroll({int(dx)},{int(dy)})"


def _render_key(kind: str, name: str) -> str:
    if not _INPUT_NAME_RE.match(name):
        raise ValueError(f"invalid input NAME for {kind}(): {name!r}")
    return f"{kind}({name})"


def render_down(name: str) -> str:
    return _render_key("down", name)


def render_up(name: str) -> str:
    return _render_key("up", name)


def render_type(text: str) -> str:
    if not text:
        raise ValueError("empty type() payload (grammar requires >=1 char)")
    if "\n" in text or "\r" in text or "\t" in text:
        raise ValueError(
            f"type() payload admits no newline/tab (use down/up primitives): {text!r}"
        )
    return f'type("{_escape_typed_text(text)}")'


def join_primitives(primitives: Sequence[str]) -> str:
    if not primitives:
        return NO_OP
    return "; ".join(primitives)
