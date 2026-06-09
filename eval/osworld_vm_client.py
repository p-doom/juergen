"""OSWorld in-VM Flask agent client + BC action translation.

The in-VM agent (OSWorld's ``desktop_env/server/main.py``) exposes a
Flask app on port 5000 with these endpoints we care about:

  GET  /screenshot         -> raw PNG bytes (full desktop framebuffer)
  GET  /cursor_position    -> {"x": int, "y": int}
  GET  /screen_size        -> {"width": int, "height": int}
  POST /execute            -> {"command": ["python", "-c", "<code>"], "shell": false}
                              runs via subprocess. The server does NOT eval strings —
                              wrap pyautogui calls as a python -c invocation.

There is NO native delta-dispatch endpoint. To drive a BC model that
emits ``dx dy scroll [; +EV -EV]`` deltas, we:
  1. Query ``/cursor_position`` to learn where the OS thinks the
     cursor is.
  2. Compute the absolute target ``(cx + dx, cy + dy)``, clipped to
     screen bounds.
  3. POST ``pyautogui.moveTo(tx, ty)`` wrapped as a python -c call to ``/execute``.
  4. For each event in the action, POST a matching pyautogui call:
     LMB/RMB/MMB -> mouseDown/mouseUp, keys -> keyDown/keyUp (with
     rdev-name → pyautogui-name mapping).
  5. For scroll, POST ``pyautogui.scroll(n)``.

This loses delta-magnitude semantics (the OS sees a teleport, not a
swept motion) but preserves the model's emission format. Good enough
for free rollouts where we're watching behavior, not measuring
grounding precision against deltas-as-such.
"""

from __future__ import annotations

import io
import logging
import time
import urllib.parse
from dataclasses import dataclass

import requests
from PIL import Image

from action_parser import Action, KeyEvent

_LOGGER = logging.getLogger(__name__)


# rdev → pyautogui key name. Only the common keys; unmapped names are
# passed through verbatim (pyautogui accepts many lowercased X11 names).
_RDEV_TO_PYAUTOGUI = {
    "Return": "enter",
    "Escape": "esc",
    "Backspace": "backspace",
    "Tab": "tab",
    "Space": "space",
    "ShiftLeft": "shiftleft",
    "ShiftRight": "shiftright",
    "ControlLeft": "ctrlleft",
    "ControlRight": "ctrlright",
    "Alt": "alt",
    "AltGr": "altright",
    "MetaLeft": "winleft",
    "MetaRight": "winright",
    "ArrowUp": "up",
    "ArrowDown": "down",
    "ArrowLeft": "left",
    "ArrowRight": "right",
    "PageUp": "pageup",
    "PageDown": "pagedown",
    "Home": "home",
    "End": "end",
    "Delete": "delete",
    "Insert": "insert",
}

_MOUSE_BUTTON_NAMES = {1: "left", 2: "middle", 3: "right"}


def _rdev_to_pyautogui(name: str) -> str:
    if name in _RDEV_TO_PYAUTOGUI:
        return _RDEV_TO_PYAUTOGUI[name]
    # rdev "KeyA"-style → "a"
    if name.startswith("Key") and len(name) == 4 and name[3].isalpha():
        return name[3].lower()
    # rdev "Num0".."Num9" → "0".."9"
    if name.startswith("Num") and len(name) == 4 and name[3].isdigit():
        return name[3]
    return name.lower()


@dataclass
class StepResult:
    """One step's dispatch outcome — for trajectory logging."""
    cursor_before: tuple[int, int]
    cursor_after: tuple[int, int]
    intended_target: tuple[int, int]  # post-clip absolute
    delta: tuple[int, int]
    scroll: int
    events_dispatched: list[str]  # pyautogui calls executed
    parse_ok: bool
    action_text: str


class OSWorldClient:
    """Thin synchronous client over the in-VM Flask agent."""

    def __init__(self, base_url: str, *, request_timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = request_timeout_s
        self._sess = requests.Session()

    # ------------------------------------------------------------- ready
    def wait_ready(self, *, timeout_s: float = 180.0, poll_s: float = 2.0) -> None:
        """Poll /screenshot until it returns 200 — the canonical
        readiness check used by OSWorld's provider."""
        url = f"{self.base_url}/screenshot"
        start = time.time()
        last_log = 0.0
        while time.time() - start < timeout_s:
            try:
                r = self._sess.get(url, timeout=5.0)
                if r.status_code == 200 and r.content:
                    _LOGGER.info("VM ready after %.1fs", time.time() - start)
                    return
            except requests.RequestException:
                pass
            elapsed = time.time() - start
            if elapsed - last_log >= 15:
                _LOGGER.info("waiting for VM /screenshot... %.0fs", elapsed)
                last_log = elapsed
            time.sleep(poll_s)
        raise TimeoutError(
            f"VM agent at {self.base_url} not ready after {timeout_s}s"
        )

    # ------------------------------------------------------------- query
    def screenshot(self) -> Image.Image:
        r = self._sess.get(f"{self.base_url}/screenshot", timeout=self.timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")

    def cursor_position(self) -> tuple[int, int]:
        r = self._sess.get(f"{self.base_url}/cursor_position", timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        # Flask handler returns jsonify(pos.x, pos.y) → [x, y] list
        return int(d[0]), int(d[1])

    def screen_size(self) -> tuple[int, int]:
        r = self._sess.post(f"{self.base_url}/screen_size", timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return int(d["width"]), int(d["height"])

    # ---------------------------------------------------------- dispatch
    _PYAUTOGUI_PREFIX = (
        "import pyautogui; import time; pyautogui.FAILSAFE = False; pyautogui.PAUSE = 0; "
    )

    def execute(self, command: str) -> None:
        """Run a pyautogui expression in the VM via /execute.

        The Flask server runs subprocess, not eval, so we wrap the expression
        as ``python -c "<prefix>; <command>"``.
        """
        full_code = self._PYAUTOGUI_PREFIX + command
        r = self._sess.post(
            f"{self.base_url}/execute",
            json={"command": ["python", "-c", full_code], "shell": False},
            timeout=self.timeout,
        )
        r.raise_for_status()

    def dispatch_action(self, action: Action) -> StepResult:
        """Apply a parsed BC ``Action`` to the VM.

        Mouse delta -> absolute moveTo (clipped to screen).
        Events     -> pyautogui mouseDown/mouseUp / keyDown/keyUp.
        Scroll     -> pyautogui.scroll(n).
        """
        cursor_before = self.cursor_position()
        sw, sh = self.screen_size()
        cx, cy = cursor_before
        executed: list[str] = []

        if action.no_op:
            return StepResult(
                cursor_before=cursor_before,
                cursor_after=cursor_before,
                intended_target=cursor_before,
                delta=(0, 0),
                scroll=0,
                events_dispatched=[],
                parse_ok=True,
                action_text="NO_OP",
            )

        # Cursor motion: clip to screen.
        tx = max(0, min(sw - 1, cx + action.dx))
        ty = max(0, min(sh - 1, cy + action.dy))
        if (tx, ty) != (cx, cy):
            cmd = f"pyautogui.moveTo({tx}, {ty})"
            self.execute(cmd)
            executed.append(cmd)

        # Scroll (after motion, before button events — pyautogui scrolls
        # at current cursor position).
        if action.scroll != 0:
            cmd = f"pyautogui.scroll({int(action.scroll)})"
            self.execute(cmd)
            executed.append(cmd)

        # Events (mouse + keyboard).
        for ev in action.events:
            cmd = _event_to_pyautogui(ev)
            if cmd is None:
                _LOGGER.debug("skipping unmapped event %r", ev)
                continue
            self.execute(cmd)
            executed.append(cmd)

        cursor_after = self.cursor_position()
        return StepResult(
            cursor_before=cursor_before,
            cursor_after=cursor_after,
            intended_target=(tx, ty),
            delta=(action.dx, action.dy),
            scroll=action.scroll,
            events_dispatched=executed,
            parse_ok=True,
            action_text="",  # caller fills the raw text
        )


def _event_to_pyautogui(ev: KeyEvent) -> str | None:
    """Render one parsed key/button event as a pyautogui call."""
    if ev.mouse_button is not None:
        btn = _MOUSE_BUTTON_NAMES.get(ev.mouse_button)
        if btn is None:
            return None
        op = "mouseDown" if ev.kind == "press" else "mouseUp"
        return f"pyautogui.{op}(button='{btn}')"
    # Keyboard.
    key = _rdev_to_pyautogui(ev.what)
    op = "keyDown" if ev.kind == "press" else "keyUp"
    return f"pyautogui.{op}({key!r})"
