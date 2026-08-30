from __future__ import annotations

import re
from collections.abc import Sequence

NO_OP = "NO_OP"
TERMINATE = "TERMINATE"
_NAME_RE = re.compile(r"^[^\s(),;]+$")


def render_move(dx: int, dy: int) -> str:
    return f"move({int(dx)},{int(dy)})"


def render_scroll(dx: int, dy: int) -> str:
    return f"scroll({int(dx)},{int(dy)})"


def _render_key(kind: str, name: str) -> str:
    if not _NAME_RE.match(name):
        raise ValueError(f"invalid input NAME for {kind}(): {name!r}")
    return f"{kind}({name})"


def render_down(name: str) -> str:
    return _render_key("down", name)


def render_up(name: str) -> str:
    return _render_key("up", name)


def render_type(text: str) -> str:
    if not text:
        raise ValueError("empty type() payload")
    if any(char in text for char in "\n\r\t"):
        raise ValueError("type() payload cannot contain newline, carriage return, or tab")
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'type("{escaped}")'


def join_primitives(primitives: Sequence[str]) -> str:
    return "; ".join(primitives) if primitives else NO_OP
