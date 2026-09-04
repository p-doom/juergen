from __future__ import annotations

import re

_NAME_RE = re.compile(r"[^\s(),;]+")
_FUNCTION_KEY_RE = re.compile(r"f([1-9]|1[0-9]|2[0-4])", re.IGNORECASE)

_ALIASES = {
    "enter": "Return",
    "return": "Return",
    "esc": "Escape",
    "escape": "Escape",
    "backspace": "Backspace",
    "tab": "Tab",
    "space": "Space",
    " ": "Space",
    "shift": "ShiftLeft",
    "shiftleft": "ShiftLeft",
    "shiftright": "ShiftRight",
    "ctrl": "ControlLeft",
    "control": "ControlLeft",
    "ctrlleft": "ControlLeft",
    "ctrlright": "ControlRight",
    "alt": "Alt",
    "altleft": "Alt",
    "option": "Alt",
    "altgr": "AltGr",
    "altright": "AltGr",
    "win": "MetaLeft",
    "winleft": "MetaLeft",
    "winright": "MetaRight",
    "super": "MetaLeft",
    "meta": "MetaLeft",
    "cmd": "MetaLeft",
    "command": "MetaLeft",
    "windows": "MetaLeft",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "pageup": "PageUp",
    "page_up": "PageUp",
    "pagedown": "PageDown",
    "page_down": "PageDown",
    "home": "Home",
    "end": "End",
    "delete": "Delete",
    "del": "Delete",
    "insert": "Insert",
    ",": "Comma",
    ".": "Period",
    "/": "Slash",
    "\\": "Backslash",
    ";": "Semicolon",
    "'": "Quote",
    "-": "Minus",
    "=": "Equal",
    "`": "Backquote",
    "[": "BracketLeft",
    "]": "BracketRight",
}


def key_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"key name must be non-empty text, got {value!r}")
    lowered = value.lower()
    if lowered in _ALIASES:
        return _ALIASES[lowered]
    if len(lowered) == 1 and lowered.isascii() and lowered.isalpha():
        return f"Key{lowered.upper()}"
    if len(lowered) == 1 and lowered.isascii() and lowered.isdigit():
        return f"Num{lowered}"
    match = _FUNCTION_KEY_RE.fullmatch(lowered)
    if match:
        return f"F{match.group(1)}"
    if _NAME_RE.fullmatch(value):
        return value
    raise ValueError(f"key name violates the action grammar: {value!r}")
