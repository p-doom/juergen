from __future__ import annotations

import re

_NAME_RE = re.compile(r"^[^\s(),;]+$")
_PYAUTOGUI_TO_RDEV = {
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
_F_KEY_RE = re.compile(r"^f([1-9]|1[0-9]|2[0-4])$")


class UnmappableKeyError(ValueError):
    pass


def pyautogui_to_rdev(name: str) -> str:
    key = str(name).strip()
    if not key:
        raise UnmappableKeyError(f"empty key name: {name!r}")
    lowered = key.lower()
    if lowered in _PYAUTOGUI_TO_RDEV:
        return _PYAUTOGUI_TO_RDEV[lowered]
    if len(lowered) == 1 and lowered.isalpha() and lowered.isascii():
        return f"Key{lowered.upper()}"
    if len(lowered) == 1 and lowered.isdigit():
        return f"Num{lowered}"
    match = _F_KEY_RE.match(lowered)
    if match:
        return f"F{match.group(1)}"
    if _NAME_RE.match(key):
        return key
    raise UnmappableKeyError(f"key name violates NAME grammar: {name!r}")
