from __future__ import annotations

import re

_FUNCTION_KEY_RE = re.compile(r"f([1-9]|1[0-9]|2[0-4])", re.IGNORECASE)

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
    "comma": "Comma",
    ".": "Period",
    "/": "Slash",
    "\\": "Backslash",
    ";": "Semicolon",
    "'": "Quote",
    "-": "Minus",
    "=": "Equal",
    "`": "Backquote",
    "grave": "Backquote",
    "[": "BracketLeft",
    "]": "BracketRight",
    "mod": "ControlLeft",
    "print": "PrintScreen",
}


def key_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"key name must be non-empty text, got {value!r}")
    lowered = value.lower()
    if lowered in _PYAUTOGUI_TO_RDEV:
        return _PYAUTOGUI_TO_RDEV[lowered]
    if len(lowered) == 1 and lowered.isascii() and lowered.isalpha():
        return f"Key{lowered.upper()}"
    if len(lowered) == 1 and lowered.isascii() and lowered.isdigit():
        return f"Num{lowered}"
    match = _FUNCTION_KEY_RE.fullmatch(lowered)
    if match:
        return f"F{match.group(1)}"
    raise ValueError(f"unsupported pyautogui key name: {value!r}")
