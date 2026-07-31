from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class TransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class Operation:
    kind: str
    args: tuple[Any, ...]


@dataclass
class InputAudit:
    operations: list[Operation] = field(default_factory=list)
    held_buttons: set[str] = field(default_factory=set)
    held_keys: set[str] = field(default_factory=set)
    scroll_total: int = 0
    typed_texts: list[str] = field(default_factory=list)


def compile_unicode_coalesced_type(text: str) -> str:
    """Compile exact Unicode text to one guest process / one clipboard paste.

    This compiler is the sole production-semantics typing path used by both
    action adapters.  ``pyautogui.write`` is deliberately forbidden because it
    is not Unicode-safe on the pinned Ubuntu guest.
    """
    if not isinstance(text, str):
        raise TypeError("coalesced type text must be a string")
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return (
        "import base64, pyperclip; "
        f"_r1a_text=base64.b64decode({encoded!r}).decode('utf-8'); "
        "pyperclip.copy(_r1a_text); pyautogui.hotkey('ctrl', 'v')"
    )


_KEYS = {
    "ControlLeft": "ctrlleft",
    "ControlRight": "ctrlright",
    "ShiftLeft": "shiftleft",
    "ShiftRight": "shiftright",
    "Alt": "alt",
    "AltGr": "altright",
    "MetaLeft": "winleft",
    "MetaRight": "winright",
    "Return": "enter",
    "Escape": "esc",
    "Backspace": "backspace",
    "Delete": "delete",
    "Tab": "tab",
    "Space": "space",
    "ArrowUp": "up",
    "ArrowDown": "down",
    "ArrowLeft": "left",
    "ArrowRight": "right",
}


def pyautogui_key(name: str) -> str:
    if name in _KEYS:
        return _KEYS[name]
    if name.startswith("Key") and len(name) == 4 and name[-1].isalpha():
        return name[-1].lower()
    if len(name) == 1:
        return name.lower()
    return name.lower()


class HttpVmTransport:
    _PREFIX = "import pyautogui; pyautogui.FAILSAFE=False; pyautogui.PAUSE=0; "

    def __init__(self, base_url: str, *, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.audit = InputAudit()

    def _request_json(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.load(response)
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise TransportError(f"VM request {method} {path} failed: {exc}") from exc

    def execute_argv(self, argv: list[str]) -> dict[str, Any]:
        result = self._request_json(
            "POST", "/execute", {"command": argv, "shell": False}
        )
        if not isinstance(result, dict):
            raise TransportError("VM /execute returned a non-object")
        if result.get("status") != "success" or result.get("returncode") != 0:
            raise TransportError(
                f"guest command failed: status={result.get('status')!r} "
                f"rc={result.get('returncode')!r} stderr={result.get('error')!r}"
            )
        return result

    def execute_pyautogui(self, code: str) -> None:
        self.execute_argv(["python", "-c", self._PREFIX + code])

    def cursor_position(self) -> tuple[int, int]:
        value = self._request_json("GET", "/cursor_position")
        if not isinstance(value, list) or len(value) != 2:
            raise TransportError(f"invalid cursor position: {value!r}")
        return int(value[0]), int(value[1])

    def screen_size(self) -> tuple[int, int]:
        value = self._request_json("POST", "/screen_size", {})
        if not isinstance(value, dict):
            raise TransportError(f"invalid screen size: {value!r}")
        return int(value["width"]), int(value["height"])

    def move_to(self, x: int, y: int) -> None:
        width, height = self.screen_size()
        x = max(0, min(width - 1, int(x)))
        y = max(0, min(height - 1, int(y)))
        self.execute_pyautogui(f"pyautogui.moveTo({x}, {y})")
        self.audit.operations.append(Operation("move_to", (x, y)))

    def mouse_down(self, button: str = "left") -> None:
        if button in self.audit.held_buttons:
            raise TransportError(f"button already held: {button}")
        self.execute_pyautogui(f"pyautogui.mouseDown(button={button!r})")
        self.audit.held_buttons.add(button)
        self.audit.operations.append(Operation("mouse_down", (button,)))

    def mouse_up(self, button: str = "left") -> None:
        if button not in self.audit.held_buttons:
            raise TransportError(f"button not held: {button}")
        self.execute_pyautogui(f"pyautogui.mouseUp(button={button!r})")
        self.audit.held_buttons.remove(button)
        self.audit.operations.append(Operation("mouse_up", (button,)))

    def scroll(self, clicks: int) -> None:
        self.execute_pyautogui(f"pyautogui.scroll({int(clicks)})")
        self.audit.scroll_total += int(clicks)
        self.audit.operations.append(Operation("scroll", (int(clicks),)))

    def key_chord(self, keys: list[str]) -> None:
        if not keys:
            raise TransportError("empty key chord")
        mapped = [pyautogui_key(key) for key in keys]
        presses = "; ".join(f"pyautogui.keyDown({key!r})" for key in mapped)
        releases = "; ".join(
            f"pyautogui.keyUp({key!r})" for key in reversed(mapped)
        )
        self.execute_pyautogui(presses + "; " + releases)
        self.audit.operations.append(Operation("key_chord", tuple(keys)))

    def coalesced_type(self, text: str) -> None:
        self.execute_pyautogui(compile_unicode_coalesced_type(text))
        self.audit.typed_texts.append(text)
        self.audit.operations.append(Operation("coalesced_type", (text,)))

    def wait(self, seconds: float) -> None:
        seconds = max(0.0, min(10.0, float(seconds)))
        time.sleep(seconds)
        self.audit.operations.append(Operation("wait", (seconds,)))


class RecordingTransport:
    """Deterministic transport used to unit-test adapter state transitions."""

    def __init__(
        self, *, cursor: tuple[int, int] = (50, 50), screen: tuple[int, int] = (1920, 1080)
    ) -> None:
        self._cursor = cursor
        self._screen = screen
        self.audit = InputAudit()

    def cursor_position(self) -> tuple[int, int]:
        return self._cursor

    def screen_size(self) -> tuple[int, int]:
        return self._screen

    def move_to(self, x: int, y: int) -> None:
        x = max(0, min(self._screen[0] - 1, int(x)))
        y = max(0, min(self._screen[1] - 1, int(y)))
        self._cursor = (x, y)
        self.audit.operations.append(Operation("move_to", (x, y)))

    def mouse_down(self, button: str = "left") -> None:
        if button in self.audit.held_buttons:
            raise TransportError(f"button already held: {button}")
        self.audit.held_buttons.add(button)
        self.audit.operations.append(Operation("mouse_down", (button,)))

    def mouse_up(self, button: str = "left") -> None:
        if button not in self.audit.held_buttons:
            raise TransportError(f"button not held: {button}")
        self.audit.held_buttons.remove(button)
        self.audit.operations.append(Operation("mouse_up", (button,)))

    def scroll(self, clicks: int) -> None:
        self.audit.scroll_total += int(clicks)
        self.audit.operations.append(Operation("scroll", (int(clicks),)))

    def key_chord(self, keys: list[str]) -> None:
        if not keys:
            raise TransportError("empty key chord")
        self.audit.operations.append(Operation("key_chord", tuple(keys)))

    def coalesced_type(self, text: str) -> None:
        self.audit.typed_texts.append(text)
        self.audit.operations.append(Operation("coalesced_type", (text,)))

    def wait(self, seconds: float) -> None:
        self.audit.operations.append(Operation("wait", (float(seconds),)))
