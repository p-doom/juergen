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
import urllib.parse
from dataclasses import dataclass

import requests
from PIL import Image

from action_parser import Action, KeyEvent, OrderedAction

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


# computer_use_rel_v1 (cua_v4_thinking) key remaps. Per the contract, key
# names arrive ALREADY as lowercase pyautogui names — the ONLY remap needed
# is "command": pyautogui's X11 backend (_pyautogui_x11.keyboardMapping)
# initialises every KEY_NAMES entry to None and only fills the X11-relevant
# ones; 'command' (a macOS name) stays None, so keyDown('command') on the
# Linux VM is a SILENT no-op. 'win'/'winleft' map to Super_L — we use
# "winleft" to match the existing rdev convention (MetaLeft -> "winleft" in
# _RDEV_TO_PYAUTOGUI above). This table must agree with the canonical table
# the data pipeline's formatter exports from
# data_pipeline/realigned_pipeline/lib/action_format.py — asserted by
# test_freeroll_helpers.KeyTableConsistencyTests via a path-based load.
_CUA_V4_KEY_TO_PYAUTOGUI = {
    "command": "winleft",
}


def _cua_v4_key_to_pyautogui(name: str) -> str:
    """computer_use_rel_v1 key name -> pyautogui name (identity but for
    the documented remaps; names arrive already lowercase per contract)."""
    return _CUA_V4_KEY_TO_PYAUTOGUI.get(name, name)


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
        raise TimeoutError(
            f"VM agent at {self.base_url} not ready after {timeout_s}s"
        )

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

    def dispatch_ordered_action(self, action: OrderedAction) -> StepResult:
        """Apply a parsed ordered/native action program to the VM.

        Primitives execute strictly left to right, matching the training
        semantics (movement, presses, and typing interleave exactly as
        emitted).

        ``ordered_events_v3`` kinds:

        - ``move(dx,dy)``   -> absolute ``moveTo`` (the cursor position is
          tracked locally across primitives, so a click after a move lands
          where the move left the cursor; each hop is clipped to screen).
        - ``scroll(dx,dy)`` -> ``pyautogui.scroll(dy)`` (dy > 0 scrolls up,
          matching pyautogui's sign) and ``pyautogui.hscroll(dx)``.
        - ``down/up(NAME)`` -> mouseDown/mouseUp / keyDown/keyUp via the
          same rdev->pyautogui mapping as the canonical format.
        - ``type("text")``  -> one ``pyautogui.write`` call (the same
          typing path ``dispatch_computer_use`` uses); no cursor effect.

        ``computer_use_rel_v1`` kinds (mouse_move_rel arrives as ``move``,
        scroll/hscroll as ``scroll``, type as ``type`` — shared with above):

        - ``click``       -> ``pyautogui.click(clicks=N, button=...)`` at the
          tracked cursor (no coordinate: clicks land wherever the cursor is).
        - ``button_down``/``button_up`` -> mouseDown/mouseUp(button=...).
        - ``key_combo``   -> keyDown each key in order, keyUp in reverse
          (pyautogui.hotkey semantics); names pass through
          ``_cua_v4_key_to_pyautogui`` ("command" -> "winleft" on the VM).
        - ``key_down``/``key_up``       -> keyDown/keyUp (same key remap).
        - ``wait``        -> NOT dispatched: behaves like NO_OP (the rollout
          loop's screenshot settle logic already waits for the UI; the
          contract's ``time`` argument is validated at parse and ignored).
        - ``terminate``   -> never reaches dispatch (the rollout loop stops
          on it before dispatching); raises if it does.
        """
        cursor_before = self.cursor_position()
        sw, sh = self.screen_size()
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

        tx, ty = cursor_before
        total_dx = total_dy = 0
        scroll_v = 0
        for p in action.primitives:
            if p.kind == "move":
                total_dx += p.dx
                total_dy += p.dy
                ntx = max(0, min(sw - 1, tx + p.dx))
                nty = max(0, min(sh - 1, ty + p.dy))
                if (ntx, nty) != (tx, ty):
                    cmd = f"pyautogui.moveTo({ntx}, {nty})"
                    self.execute(cmd)
                    executed.append(cmd)
                tx, ty = ntx, nty
            elif p.kind == "scroll":
                if p.dy:
                    cmd = f"pyautogui.scroll({int(p.dy)})"
                    self.execute(cmd)
                    executed.append(cmd)
                if p.dx:
                    cmd = f"pyautogui.hscroll({int(p.dx)})"
                    self.execute(cmd)
                    executed.append(cmd)
                scroll_v += p.dy
            elif p.kind in ("down", "up"):
                ev = KeyEvent(
                    kind="press" if p.kind == "down" else "release",
                    what=p.name,
                    mouse_button=p.mouse_button,
                )
                cmd = _event_to_pyautogui(ev)
                if cmd is None:
                    _LOGGER.debug("skipping unmapped event %r", p)
                    continue
                self.execute(cmd)
                executed.append(cmd)
            elif p.kind == "type":
                if p.text:
                    cmd = _type_write_command(p.text)
                    self.execute(cmd)
                    executed.append(cmd)
            elif p.kind == "click":
                cmd = (
                    f"pyautogui.click(clicks={p.count}, interval=0.05, "
                    f"button={p.name!r})"
                )
                self.execute(cmd)
                executed.append(cmd)
            elif p.kind in ("button_down", "button_up"):
                op = "mouseDown" if p.kind == "button_down" else "mouseUp"
                cmd = f"pyautogui.{op}(button={p.name!r})"
                self.execute(cmd)
                executed.append(cmd)
            elif p.kind == "key_combo":
                py_keys = [_cua_v4_key_to_pyautogui(k) for k in p.keys]
                for key in py_keys:
                    cmd = f"pyautogui.keyDown({key!r})"
                    self.execute(cmd)
                    executed.append(cmd)
                for key in reversed(py_keys):
                    cmd = f"pyautogui.keyUp({key!r})"
                    self.execute(cmd)
                    executed.append(cmd)
            elif p.kind in ("key_down", "key_up"):
                op = "keyDown" if p.kind == "key_down" else "keyUp"
                cmd = f"pyautogui.{op}({_cua_v4_key_to_pyautogui(p.name)!r})"
                self.execute(cmd)
                executed.append(cmd)
            elif p.kind == "wait":
                # NO dispatch: behaves like NO_OP. The caller's settle logic
                # already waits for the screen before the next screenshot.
                pass
            elif p.kind == "terminate":
                raise ValueError(
                    "terminate is a rollout stop condition, never dispatched"
                )
            else:
                raise ValueError(f"unknown ordered primitive kind: {p.kind!r}")

        cursor_after = self.cursor_position()
        return StepResult(
            cursor_before=cursor_before,
            cursor_after=cursor_after,
            intended_target=(tx, ty),
            delta=(total_dx, total_dy),
            scroll=scroll_v,
            events_dispatched=executed,
            parse_ok=True,
            action_text="",  # caller fills the raw text
        )

    def dispatch_computer_use(self, arguments: dict) -> StepResult:
        """Apply one OpenAI computer-use style tool call to the VM."""
        if not isinstance(arguments, dict):
            raise TypeError(
                f"computer_use arguments must be dict, got {type(arguments)!r}"
            )

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
                    raise ValueError(
                        f"computer_use action {action!r} requires coordinate"
                    )
                return None
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ValueError(f"coordinate must be [x, y], got {raw!r}")
            try:
                x = int(round(float(raw[0])))
                y = int(round(float(raw[1])))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"coordinate values must be numeric, got {raw!r}"
                ) from e
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
            cmd = (
                f"pyautogui.click(clicks={clicks}, interval=0.05, "
                f"button={button!r})"
            )
            self.execute(cmd)
            executed.append(cmd)
        elif action == "left_click_drag":
            pos = coord_from_args(required=True)
            assert pos is not None
            target = pos
            cmd = (
                f"pyautogui.dragTo({pos[0]}, {pos[1]}, duration=0.2, "
                "button='left')"
            )
            self.execute(cmd)
            executed.append(cmd)
        elif action in {"scroll", "hscroll"}:
            try:
                scroll = int(round(float(arguments.get("pixels", 0))))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"pixels must be numeric, got {arguments.get('pixels')!r}"
                ) from e
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
                cmd = _type_write_command(text)
                self.execute(cmd)
                executed.append(cmd)
        elif action == "wait":
            try:
                wait_s = float(arguments.get("time", 1.0))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"time must be numeric, got {arguments.get('time')!r}"
                ) from e
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


def _type_write_command(text: str) -> str:
    """Render a typed string as one ``pyautogui.write`` call.

    ``repr`` produces a valid Python string literal (quotes/backslashes
    escaped), and ``/execute`` passes the code as an argv element — no
    shell quoting layer — so arbitrary printable payloads survive intact.
    """
    return f"pyautogui.write({text!r}, interval=0)"


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
