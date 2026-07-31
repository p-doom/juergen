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


BUTTON_MASKS = {"left": 1 << 8, "middle": 1 << 9, "right": 1 << 10}
ALL_POINTER_BUTTON_MASK = sum(BUTTON_MASKS.values())
ATOMIC_RESULT_PREFIX = "RUNG1A_ATOMIC_RESULT="


@dataclass(frozen=True)
class AtomicExecutionResult:
    ok: bool
    cursor: tuple[int, int]
    pointer_button_mask: int
    observed_pointer_button_mask: int
    expected_pointer_button_mask: int
    guest_process_count: int
    cleanup_attempted: bool
    error: str | None
    operations: tuple[Operation, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "cursor": list(self.cursor),
            "pointer_button_mask": self.pointer_button_mask,
            "observed_pointer_button_mask": self.observed_pointer_button_mask,
            "expected_pointer_button_mask": self.expected_pointer_button_mask,
            "guest_process_count": self.guest_process_count,
            "cleanup_attempted": self.cleanup_attempted,
            "error": self.error,
            "operations": [
                {"kind": item.kind, "args": list(item.args)}
                for item in self.operations
            ],
        }


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
        "import base64, time, tkinter as _r1a_tk; "
        f"_r1a_text=base64.b64decode({encoded!r}).decode('utf-8'); "
        "_r1a_root=_r1a_tk.Tk(); _r1a_root.withdraw(); "
        "_r1a_root.clipboard_clear(); _r1a_root.clipboard_append(_r1a_text); "
        "_r1a_root.update(); pyautogui.hotkey('ctrl', 'v'); "
        "time.sleep(0.25); _r1a_root.update(); _r1a_root.destroy()"
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


def expected_atomic_input_state(
    operations: tuple[Operation, ...],
    *,
    initial_buttons: set[str],
    initial_keys: set[str],
) -> tuple[set[str], set[str]]:
    buttons = set(initial_buttons)
    keys = set(initial_keys)
    for operation in operations:
        if operation.kind == "mouse_down":
            button = str(operation.args[0])
            if button in buttons:
                raise TransportError(f"button already held: {button}")
            buttons.add(button)
        elif operation.kind == "mouse_up":
            button = str(operation.args[0])
            if button not in buttons:
                raise TransportError(f"button not held: {button}")
            buttons.remove(button)
        elif operation.kind == "key_down":
            key = str(operation.args[0])
            if key in keys:
                raise TransportError(f"key already held: {key}")
            keys.add(key)
        elif operation.kind == "key_up":
            key = str(operation.args[0])
            if key not in keys:
                raise TransportError(f"key not held: {key}")
            keys.remove(key)
    return buttons, keys


def pointer_mask_for_buttons(buttons: set[str]) -> int:
    unknown = buttons - set(BUTTON_MASKS)
    if unknown:
        raise TransportError(f"unsupported pointer buttons: {sorted(unknown)}")
    return sum(BUTTON_MASKS[button] for button in buttons)


def compile_atomic_guest_program(
    operations: tuple[Operation, ...],
    *,
    initial_buttons: set[str],
    initial_keys: set[str],
) -> tuple[str, int]:
    """Compile one compact action to exactly one ordered guest process."""
    final_buttons, _ = expected_atomic_input_state(
        operations,
        initial_buttons=initial_buttons,
        initial_keys=initial_keys,
    )
    expected_mask = pointer_mask_for_buttons(final_buttons)
    lines = [
        "import json, traceback, pyautogui",
        "pyautogui.FAILSAFE=False",
        "pyautogui.PAUSE=0",
        "_r1a_trace=[]",
        f"_r1a_expected_mask={expected_mask}",
        f"_r1a_touched_buttons=set({sorted(initial_buttons)!r})",
        f"_r1a_touched_keys=set({[pyautogui_key(k) for k in sorted(initial_keys)]!r})",
        "_r1a_error=None",
        "_r1a_cleanup=False",
        "_r1a_observed_mask=-1",
        "def _r1a_pointer_state():",
        "    _backend=pyautogui.platformModule",
        "    _backend._display.sync()",
        "    _pointer=_backend._display.screen().root.query_pointer()",
        f"    return int(_pointer.root_x),int(_pointer.root_y),int(_pointer.mask)&{ALL_POINTER_BUTTON_MASK}",
        "try:",
    ]
    indent = "    "
    for index, operation in enumerate(operations):
        kind, args = operation.kind, operation.args
        lines.append(f"{indent}# RUNG1A_ATOMIC_STEP_{index}:{kind}")
        if kind == "move_relative":
            dx, dy = int(args[0]), int(args[1])
            lines.extend(
                [
                    f"{indent}_x,_y=pyautogui.position()",
                    f"{indent}_w,_h=pyautogui.size()",
                    f"{indent}_tx=max(0,min(int(_w)-1,int(_x)+({dx})))",
                    f"{indent}_ty=max(0,min(int(_h)-1,int(_y)+({dy})))",
                    f"{indent}pyautogui.moveTo(_tx,_ty)",
                    f"{indent}_r1a_trace.append({{'kind':'move_to','args':[_tx,_ty]}})",
                ]
            )
        elif kind == "scroll":
            clicks = int(args[0])
            lines.extend(
                [
                    f"{indent}pyautogui.scroll({clicks})",
                    f"{indent}_r1a_trace.append({{'kind':'scroll','args':[{clicks}]}})",
                ]
            )
        elif kind in {"mouse_down", "mouse_up"}:
            button = str(args[0])
            method = "mouseDown" if kind == "mouse_down" else "mouseUp"
            lines.extend(
                [
                    f"{indent}_r1a_touched_buttons.add({button!r})",
                    f"{indent}pyautogui.{method}(button={button!r})",
                    f"{indent}_r1a_trace.append({{'kind':{kind!r},'args':[{button!r}]}})",
                ]
            )
        elif kind in {"key_down", "key_up"}:
            key = str(args[0])
            mapped = pyautogui_key(key)
            method = "keyDown" if kind == "key_down" else "keyUp"
            lines.extend(
                [
                    f"{indent}_r1a_touched_keys.add({mapped!r})",
                    f"{indent}pyautogui.{method}({mapped!r})",
                    f"{indent}_r1a_trace.append({{'kind':{kind!r},'args':[{key!r}]}})",
                ]
            )
        elif kind == "coalesced_type":
            text = str(args[0])
            lines.extend(
                [
                    f"{indent}{compile_unicode_coalesced_type(text)}",
                    f"{indent}_r1a_trace.append({{'kind':'coalesced_type','args':[{text!r}]}})",
                ]
            )
        elif kind == "raise_for_test":
            lines.append(f"{indent}raise RuntimeError({str(args[0])!r})")
        else:
            raise TransportError(f"unsupported atomic operation: {kind}")
    lines.extend(
        [
            f"{indent}_cx,_cy,_r1a_observed_mask=_r1a_pointer_state()",
            f"{indent}if _r1a_observed_mask != _r1a_expected_mask:",
            f"{indent}    raise RuntimeError(f'pointer button mask {{_r1a_observed_mask}} != expected {{_r1a_expected_mask}}')",
            "except BaseException as _exc:",
            "    _r1a_error=''.join(traceback.format_exception_only(type(_exc),_exc)).strip()",
            "    _r1a_cleanup=True",
            "    for _key in sorted(_r1a_touched_keys,reverse=True):",
            "        try: pyautogui.keyUp(_key)",
            "        except BaseException: pass",
            "    for _button in ('left','middle','right'):",
            "        try: pyautogui.mouseUp(button=_button)",
            "        except BaseException: pass",
            "_r1a_cx,_r1a_cy,_r1a_final_mask=_r1a_pointer_state()",
            "_r1a_payload={'ok':_r1a_error is None,'cursor':[_r1a_cx,_r1a_cy],",
            " '_r1a_schema':1,'pointer_button_mask':_r1a_final_mask,",
            " 'observed_pointer_button_mask':_r1a_observed_mask,",
            " 'expected_pointer_button_mask':_r1a_expected_mask,",
            " 'guest_process_count':1,'cleanup_attempted':_r1a_cleanup,",
            " 'error':_r1a_error,'operations':_r1a_trace}",
            f"print({ATOMIC_RESULT_PREFIX!r}+json.dumps(_r1a_payload,separators=(',',':'),ensure_ascii=False))",
        ]
    )
    return "\n".join(lines), expected_mask


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

    def execute_compact_atomic(
        self, operations: tuple[Operation, ...]
    ) -> AtomicExecutionResult:
        program, expected_mask = compile_atomic_guest_program(
            operations,
            initial_buttons=set(self.audit.held_buttons),
            initial_keys=set(self.audit.held_keys),
        )
        _, expected_keys = expected_atomic_input_state(
            operations,
            initial_buttons=set(self.audit.held_buttons),
            initial_keys=set(self.audit.held_keys),
        )
        result = self.execute_argv(["python", "-c", program])
        output = result.get("output")
        if not isinstance(output, str):
            raise TransportError("atomic guest action returned no stdout")
        lines = [line for line in output.splitlines() if line.startswith(ATOMIC_RESULT_PREFIX)]
        if len(lines) != 1:
            raise TransportError(f"atomic guest result marker count was {len(lines)}")
        try:
            payload = json.loads(lines[0][len(ATOMIC_RESULT_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise TransportError("atomic guest action returned invalid JSON") from exc
        if not isinstance(payload, dict) or payload.get("_r1a_schema") != 1:
            raise TransportError("atomic guest action returned an invalid schema")
        cursor = payload.get("cursor")
        raw_operations = payload.get("operations")
        if (
            not isinstance(cursor, list)
            or len(cursor) != 2
            or not isinstance(raw_operations, list)
        ):
            raise TransportError("atomic guest action returned invalid state")
        traced: list[Operation] = []
        for item in raw_operations:
            if not isinstance(item, dict) or not isinstance(item.get("args"), list):
                raise TransportError("atomic guest action returned invalid trace")
            traced.append(Operation(str(item.get("kind", "")), tuple(item["args"])))
        pointer_mask = int(payload.get("pointer_button_mask", -1))
        reported_expected = int(payload.get("expected_pointer_button_mask", -1))
        if reported_expected != expected_mask or pointer_mask < 0:
            raise TransportError("atomic guest action mask contract mismatch")
        atomic_result = AtomicExecutionResult(
            ok=bool(payload.get("ok")),
            cursor=(int(cursor[0]), int(cursor[1])),
            pointer_button_mask=pointer_mask,
            observed_pointer_button_mask=int(
                payload.get("observed_pointer_button_mask", -1)
            ),
            expected_pointer_button_mask=reported_expected,
            guest_process_count=int(payload.get("guest_process_count", 0)),
            cleanup_attempted=bool(payload.get("cleanup_attempted")),
            error=None if payload.get("error") is None else str(payload["error"]),
            operations=tuple(traced),
        )
        self.audit.operations.extend(traced)
        self.audit.held_buttons = {
            button
            for button, mask in BUTTON_MASKS.items()
            if pointer_mask & mask
        }
        self.audit.held_keys = expected_keys if atomic_result.ok else set()
        for operation in traced:
            if operation.kind == "scroll":
                self.audit.scroll_total += int(operation.args[0])
            elif operation.kind == "coalesced_type":
                self.audit.typed_texts.append(str(operation.args[0]))
        return atomic_result

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
        self.atomic_invocations = 0

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

    def execute_compact_atomic(
        self, operations: tuple[Operation, ...]
    ) -> AtomicExecutionResult:
        self.atomic_invocations += 1
        before = len(self.audit.operations)
        initial_buttons = set(self.audit.held_buttons)
        initial_keys = set(self.audit.held_keys)
        final_buttons, final_keys = expected_atomic_input_state(
            operations,
            initial_buttons=initial_buttons,
            initial_keys=initial_keys,
        )
        expected_mask = pointer_mask_for_buttons(final_buttons)
        observed_mask = -1
        cleanup_attempted = False
        error: str | None = None
        try:
            for operation in operations:
                kind, args = operation.kind, operation.args
                if kind == "move_relative":
                    self.move_to(
                        self._cursor[0] + int(args[0]),
                        self._cursor[1] + int(args[1]),
                    )
                elif kind == "scroll":
                    self.scroll(int(args[0]))
                elif kind == "mouse_down":
                    self.mouse_down(str(args[0]))
                elif kind == "mouse_up":
                    self.mouse_up(str(args[0]))
                elif kind == "key_down":
                    key = str(args[0])
                    if key in self.audit.held_keys:
                        raise TransportError(f"key already held: {key}")
                    self.audit.held_keys.add(key)
                    self.audit.operations.append(Operation("key_down", (key,)))
                elif kind == "key_up":
                    key = str(args[0])
                    if key not in self.audit.held_keys:
                        raise TransportError(f"key not held: {key}")
                    self.audit.held_keys.remove(key)
                    self.audit.operations.append(Operation("key_up", (key,)))
                elif kind == "coalesced_type":
                    self.coalesced_type(str(args[0]))
                elif kind == "raise_for_test":
                    raise RuntimeError(str(args[0]))
                else:
                    raise TransportError(f"unsupported atomic operation: {kind}")
            observed_mask = pointer_mask_for_buttons(self.audit.held_buttons)
            if observed_mask != expected_mask:
                raise TransportError(
                    f"pointer button mask {observed_mask} != expected {expected_mask}"
                )
            self.audit.held_keys = final_keys
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            cleanup_attempted = True
            for key in sorted(self.audit.held_keys, reverse=True):
                self.audit.operations.append(Operation("key_up", (key,)))
            self.audit.held_keys.clear()
            for button in sorted(self.audit.held_buttons):
                try:
                    self.mouse_up(button)
                except BaseException:
                    self.audit.held_buttons.discard(button)
            self.audit.held_buttons.clear()
        final_mask = pointer_mask_for_buttons(self.audit.held_buttons)
        return AtomicExecutionResult(
            ok=error is None,
            cursor=self._cursor,
            pointer_button_mask=final_mask,
            observed_pointer_button_mask=observed_mask,
            expected_pointer_button_mask=expected_mask,
            guest_process_count=1,
            cleanup_attempted=cleanup_attempted,
            error=error,
            operations=tuple(self.audit.operations[before:]),
        )
