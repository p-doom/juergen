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
import re
import time
from dataclasses import dataclass

import requests
from PIL import Image

from action_parser import Action, DeltaTypeAction, KeyEvent

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
    "Comma": ",",
    "Period": ".",
    "Slash": "/",
    "Backslash": "\\",
    "Semicolon": ";",
    "Quote": "'",
    "Minus": "-",
    "Equal": "=",
    "Backquote": "`",
    "BracketLeft": "[",
    "BracketRight": "]",
}

_MOUSE_BUTTON_NAMES = {1: "left", 2: "middle", 3: "right"}

_COMPUTER_USE_KEY_ALIASES = {
    "CTRL": "ctrl",
    "CONTROL": "ctrl",
    "SHIFT": "shift",
    "ALT": "alt",
    "OPTION": "alt",
    "CMD": "command",
    "COMMAND": "command",
    "META": "win",
    "SUPER": "win",
    "WIN": "win",
    "WINDOWS": "win",
    "ENTER": "enter",
    "RETURN": "enter",
    "ESC": "esc",
    "ESCAPE": "esc",
    "BACKSPACE": "backspace",
    "DELETE": "delete",
    "DEL": "delete",
    "TAB": "tab",
    "SPACE": "space",
    "UP": "up",
    "DOWN": "down",
    "LEFT": "left",
    "RIGHT": "right",
    "PAGEUP": "pageup",
    "PAGE_UP": "pageup",
    "PAGEDOWN": "pagedown",
    "PAGE_DOWN": "pagedown",
    "HOME": "home",
    "END": "end",
}


def _rdev_to_pyautogui(name: str) -> str:
    if name in _RDEV_TO_PYAUTOGUI:
        return _RDEV_TO_PYAUTOGUI[name]
    # rdev "KeyA"-style → "a"
    if name.startswith("Key") and len(name) == 4 and name[3].isalpha():
        return name[3].lower()
    # rdev "Num0".."Num9" → "0".."9"
    if name.startswith("Num") and len(name) == 4 and name[3].isdigit():
        return name[3]
    # DOM-style "Digit0".."Digit9" appears in some generated traces/prompts.
    if name.startswith("Digit") and len(name) == 6 and name[5].isdigit():
        return name[5]
    return name.lower()


def _computer_use_key_to_pyautogui(name: str) -> str:
    """Map common computer-use key spellings to pyautogui key names."""
    if not isinstance(name, str):
        raise TypeError(f"key must be a string, got {type(name)!r}")
    stripped = name.strip()
    upper = stripped.upper()
    if upper in _COMPUTER_USE_KEY_ALIASES:
        return _COMPUTER_USE_KEY_ALIASES[upper]
    if m := re.match(r"^F([1-9]|1[0-9]|2[0-4])$", upper):
        return f"f{m.group(1)}"
    if len(stripped) == 1:
        return stripped.lower()
    return _rdev_to_pyautogui(stripped)


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
        raise TimeoutError(f"VM agent at {self.base_url} not ready after {timeout_s}s")

    # ------------------------------------------------------------- query
    def screenshot(self) -> Image.Image:
        r = self._sess.get(f"{self.base_url}/screenshot", timeout=self.timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")

    def screenshot_settled(
        self,
        *,
        min_delay_s: float = 0.0,
        stability_timeout_s: float = 0.0,
        poll_s: float = 0.1,
    ) -> Image.Image:
        """Capture a screenshot after the desktop has stopped repainting.

        ``dispatch_action`` returns as soon as the pyautogui events are *sent*;
        the app may not have handled them and repainted the framebuffer yet, so
        a screenshot taken immediately can miss the action's visible effect. The
        model then sees an unchanged frame and re-emits the same action. This
        method waits for the UI to settle before grabbing the frame:

        - Sleep ``min_delay_s`` first (a fixed post-action settle).
        - If ``stability_timeout_s > 0``, poll the framebuffer every ``poll_s``
          until two consecutive captures are pixel-identical (repainting has
          stopped) or the timeout elapses; return the last frame.

        With both delays zero this is exactly ``screenshot()``. Note: a
        constantly-animating element (blinking caret, clock) can prevent
        stability, in which case it waits the full ``stability_timeout_s``.
        """
        if min_delay_s > 0:
            time.sleep(min_delay_s)
        if stability_timeout_s <= 0:
            return self.screenshot()

        prev = self.screenshot()
        deadline = time.time() + stability_timeout_s
        while time.time() < deadline:
            time.sleep(poll_s)
            cur = self.screenshot()
            if cur.tobytes() == prev.tobytes():
                return cur
            prev = cur
        return prev

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

    def dispatch_deltatype(self, action: DeltaTypeAction, *, rel_coord_grid: int = 0) -> StepResult:
        """Execute one compact relative line, including coalesced type()."""
        cursor_before = self.cursor_position()
        sw, sh = self.screen_size()
        if action.no_op or action.terminate or action.fail:
            return StepResult(
                cursor_before=cursor_before,
                cursor_after=cursor_before,
                intended_target=cursor_before,
                delta=(0, 0),
                scroll=0,
                events_dispatched=[],
                parse_ok=True,
                action_text="",
            )
        dx = round(action.dx * sw / rel_coord_grid) if rel_coord_grid else action.dx
        dy = round(action.dy * sh / rel_coord_grid) if rel_coord_grid else action.dy
        target = cursor_before[0] + dx, cursor_before[1] + dy
        if not 0 <= target[0] < sw or not 0 <= target[1] < sh:
            raise ValueError(f"relative action would clip at viewport edge: {target}")
        executed = []
        if target != cursor_before:
            command = f"pyautogui.moveTo({target[0]}, {target[1]})"
            self.execute(command)
            executed.append(command)
        if action.scroll:
            command = f"pyautogui.scroll({action.scroll})"
            self.execute(command)
            executed.append(command)
        for kind, element in action.elements:
            command = (
                _event_to_pyautogui(element)
                if kind == "event"
                else f"pyautogui.write({element!r}, interval=0)"
            )
            if command is not None:
                self.execute(command)
                executed.append(command)
        cursor_after = self.cursor_position()
        return StepResult(
            cursor_before=cursor_before,
            cursor_after=cursor_after,
            intended_target=target,
            delta=(dx, dy),
            scroll=action.scroll,
            events_dispatched=executed,
            parse_ok=True,
            action_text="",
        )

    def dispatch_computer_use(self, arguments: dict) -> StepResult:
        """Apply one OpenAI computer-use style tool call to the VM."""
        if not isinstance(arguments, dict):
            raise TypeError(f"computer_use arguments must be dict, got {type(arguments)!r}")

        action = str(arguments.get("action", "")).strip().lower()
        cursor_before = self.cursor_position()
        sw, sh = self.screen_size()
        executed: list[str] = []
        target = cursor_before
        scroll = 0

        def coord_from_args(*, required: bool) -> tuple[int, int] | None:
            raw = arguments.get("coordinate")
            if raw is None:
                if required:
                    raise ValueError(f"computer_use action {action!r} requires coordinate")
                return None
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ValueError(f"coordinate must be [x, y], got {raw!r}")
            try:
                x = round(float(raw[0]))
                y = round(float(raw[1]))
            except (TypeError, ValueError) as e:
                raise ValueError(f"coordinate values must be numeric, got {raw!r}") from e
            return max(0, min(sw - 1, x)), max(0, min(sh - 1, y))

        def move_to(pos: tuple[int, int]) -> None:
            nonlocal target
            target = pos
            if pos != self.cursor_position():
                cmd = f"pyautogui.moveTo({pos[0]}, {pos[1]})"
                self.execute(cmd)
                executed.append(cmd)

        if action == "mouse_move":
            pos = coord_from_args(required=True)
            assert pos is not None
            move_to(pos)
        elif action in {
            "left_click",
            "right_click",
            "middle_click",
            "double_click",
            "triple_click",
        }:
            pos = coord_from_args(required=False)
            if pos is not None:
                move_to(pos)
            button = {
                "left_click": "left",
                "right_click": "right",
                "middle_click": "middle",
                "double_click": "left",
                "triple_click": "left",
            }[action]
            clicks = 2 if action in {"double_click", "triple_click"} else 1
            cmd = f"pyautogui.click(clicks={clicks}, interval=0.05, button={button!r})"
            self.execute(cmd)
            executed.append(cmd)
        elif action == "left_click_drag":
            pos = coord_from_args(required=True)
            assert pos is not None
            target = pos
            cmd = f"pyautogui.dragTo({pos[0]}, {pos[1]}, duration=0.2, button='left')"
            self.execute(cmd)
            executed.append(cmd)
        elif action in {"scroll", "hscroll"}:
            try:
                scroll = round(float(arguments.get("pixels", 0)))
            except (TypeError, ValueError) as e:
                raise ValueError(f"pixels must be numeric, got {arguments.get('pixels')!r}") from e
            if scroll:
                cmd = f"pyautogui.scroll({scroll})"
                self.execute(cmd)
                executed.append(cmd)
        elif action == "key":
            keys = arguments.get("keys")
            if isinstance(keys, str):
                keys = [keys]
            if not isinstance(keys, list) or not keys:
                raise ValueError(f"keys must be a non-empty array, got {keys!r}")
            py_keys = [_computer_use_key_to_pyautogui(k) for k in keys]
            for key in py_keys:
                cmd = f"pyautogui.keyDown({key!r})"
                self.execute(cmd)
                executed.append(cmd)
            for key in reversed(py_keys):
                cmd = f"pyautogui.keyUp({key!r})"
                self.execute(cmd)
                executed.append(cmd)
        elif action == "type":
            text = str(arguments.get("text", ""))
            if text:
                cmd = f"pyautogui.write({text!r}, interval=0)"
                self.execute(cmd)
                executed.append(cmd)
        elif action == "wait":
            try:
                wait_s = float(arguments.get("time", 1.0))
            except (TypeError, ValueError) as e:
                raise ValueError(f"time must be numeric, got {arguments.get('time')!r}") from e
            wait_s = max(0.0, min(10.0, wait_s))
            time.sleep(wait_s)
            executed.append(f"time.sleep({wait_s})")
        elif action in {"answer", "terminate"}:
            # The freeroll loop consumes terminate as a stop condition. `answer`
            # has no desktop side effect, so preserve it as a logged no-op.
            pass
        else:
            raise ValueError(f"unsupported computer_use action: {action!r}")

        cursor_after = self.cursor_position()
        if target == cursor_before and cursor_after != cursor_before:
            target = cursor_after
        return StepResult(
            cursor_before=cursor_before,
            cursor_after=cursor_after,
            intended_target=target,
            delta=(target[0] - cursor_before[0], target[1] - cursor_before[1]),
            scroll=scroll,
            events_dispatched=executed,
            parse_ok=True,
            action_text="",
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
