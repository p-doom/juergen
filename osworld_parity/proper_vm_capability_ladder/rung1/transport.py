from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class TransportError(RuntimeError):
    def __init__(self, message: str, *, evidence: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        if evidence is not None or not hasattr(self, "evidence"):
            self.evidence = evidence


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

# Coalesced-typing clipboard timings, named so that the CPU-contention
# hypothesis for a lost paste is expressed in code rather than in magic
# numbers.  The guest clipboard owner must outlive the paste by enough for the
# target application to request the selection contents.
CLIPBOARD_PASTE_DELAY_MS = 150
CLIPBOARD_OWNER_LIFETIME_MS = 750
ATOMIC_RESULT_PREFIX = "RUNG1A_ATOMIC_RESULT="
PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND = "pyautogui_release_motion"
DIRECT_XTEST_CLICK_BACKEND = "direct_xtest_no_release_motion"
CLICK_BACKENDS = frozenset(
    {PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND, DIRECT_XTEST_CLICK_BACKEND}
)
CLICK_DWELL_S = 0.05
PASSIVE_X_OBSERVER_LIMITATION = (
    "not installed: a same-process XRecord/XI2 observer requires a second X "
    "connection and concurrent event consumption, which is not demonstrably "
    "non-perturbing for this timing experiment"
)


@dataclass(frozen=True)
class AtomicExecutionResult:
    ok: bool
    cursor: tuple[int, int]
    cursor_before: tuple[int, int]
    cursor_after: tuple[int, int]
    pointer_button_mask: int
    observed_pointer_button_mask: int
    expected_pointer_button_mask: int
    guest_process_count: int
    guest_returncode: int
    raw_result_marker: str
    cleanup_attempted: bool
    error: str | None
    failure_kind: str | None
    operations: tuple[Operation, ...]
    semantic_operations: tuple[Operation, ...]
    lowered_operations: tuple[Operation, ...]
    backend_primitives: tuple[dict[str, Any], ...] = ()
    x_event_sync_evidence: tuple[dict[str, Any], ...] = ()
    click_backend: str = PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND
    x_injection_evidence: tuple[dict[str, Any], ...] = ()
    x_injection_timestamps: tuple[dict[str, Any], ...] = ()
    passive_x_observer: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "cursor": list(self.cursor),
            "cursor_before": list(self.cursor_before),
            "cursor_after": list(self.cursor_after),
            "pointer_button_mask": self.pointer_button_mask,
            "observed_pointer_button_mask": self.observed_pointer_button_mask,
            "expected_pointer_button_mask": self.expected_pointer_button_mask,
            "guest_process_count": self.guest_process_count,
            "guest_returncode": self.guest_returncode,
            "raw_result_marker": self.raw_result_marker,
            "cleanup_attempted": self.cleanup_attempted,
            "error": self.error,
            "failure_kind": self.failure_kind,
            "operations": [
                {"kind": item.kind, "args": list(item.args)}
                for item in self.operations
            ],
            "semantic_operations": [
                {"kind": item.kind, "args": list(item.args)}
                for item in self.semantic_operations
            ],
            "lowered_operations": [
                {"kind": item.kind, "args": list(item.args)}
                for item in self.lowered_operations
            ],
            "backend_primitives": list(self.backend_primitives),
            "x_event_sync_evidence": list(self.x_event_sync_evidence),
            "click_backend": self.click_backend,
            "x_injection_evidence": list(self.x_injection_evidence),
            "x_injection_timestamps": list(self.x_injection_timestamps),
            "passive_x_observer": dict(self.passive_x_observer),
        }


def compile_unicode_coalesced_type(text: str) -> str:
    """Compile exact Unicode text to one guest process / one clipboard paste.

    This compiler is the sole production-semantics typing path used by both
    action adapters.  ``pyautogui.write`` is deliberately forbidden because it
    is not Unicode-safe on the pinned Ubuntu guest.

    Clipboard-backend history on the pinned image, oldest to newest:

    1. ``pyperclip`` imports but raises at runtime; the guest has no xclip,
       xsel or wl-copy backend.
    2. Tk owns the X11 selection but only while the interpreter pumps its event
       loop, and Tk's own ``clipboard_clear`` collapsed the editor selection in
       VS Code/LibreOffice.
    3. GTK (this version) owns the selection from a real GLib main loop, proves
       the round trip before pasting, and re-asserts select-all immediately
       before its single paste because taking clipboard ownership drops the
       target widget's selection on the pinned image.

    Focus/type trajectories already emit their own Ctrl-A; the re-assertion
    inside the clipboard owner is deliberately redundant with it so the paste
    replaces rather than appends even when ownership stole the selection.
    """
    if not isinstance(text, str):
        raise TypeError("coalesced type text must be a string")
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    program = f"""
import base64,gi
gi.require_version('Gtk','3.0')
from gi.repository import Gtk,Gdk,GLib
value=base64.b64decode({encoded!r}).decode('utf-8')
clipboard=Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
clipboard.set_text(value,-1)
if clipboard.wait_for_text()!=value:
 raise RuntimeError('clipboard round-trip failed')
_r1a_pasted=[]
def paste():
 pyautogui.hotkey('ctrl','a')
 pyautogui.hotkey('ctrl','v')
 _r1a_pasted.append(True)
 return False
GLib.timeout_add({CLIPBOARD_PASTE_DELAY_MS},paste)
GLib.timeout_add({CLIPBOARD_OWNER_LIFETIME_MS},Gtk.main_quit)
Gtk.main()
if not _r1a_pasted:
 raise RuntimeError('clipboard owner expired before the paste callback ran')
""".strip()
    return f"exec({program!r})"


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
            if button not in BUTTON_MASKS:
                raise TransportError(f"unsupported pointer button: {button}")
            if button in buttons:
                raise TransportError(f"button already held: {button}")
            buttons.add(button)
        elif operation.kind == "mouse_up":
            button = str(operation.args[0])
            if button not in BUTTON_MASKS:
                raise TransportError(f"unsupported pointer button: {button}")
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


def lower_guest_operations(
    operations: tuple[Operation, ...],
) -> tuple[Operation, ...]:
    """Lower only an adjacent same-button press/release to one click primitive.

    The input remains the canonical semantic event stream.  In particular, a
    move, key event, type, or any other operation between the transitions
    prevents coalescing, which keeps drag/hold trajectories explicit.
    """
    lowered: list[Operation] = []
    index = 0
    while index < len(operations):
        operation = operations[index]
        if operation.kind == "mouse_down" and index + 1 < len(operations):
            following = operations[index + 1]
            if (
                following.kind == "mouse_up"
                and following.args == operation.args
            ):
                lowered.append(Operation("click", operation.args))
                index += 2
                continue
        lowered.append(operation)
        index += 1
    return tuple(lowered)


def compile_atomic_guest_program(
    operations: tuple[Operation, ...],
    *,
    initial_buttons: set[str],
    initial_keys: set[str],
    click_backend: str = PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
) -> tuple[str, int]:
    """Compile one adapter action to exactly one ordered guest process."""
    if click_backend not in CLICK_BACKENDS:
        raise TransportError(f"unsupported click backend: {click_backend}")
    final_buttons, _ = expected_atomic_input_state(
        operations,
        initial_buttons=initial_buttons,
        initial_keys=initial_keys,
    )
    expected_mask = pointer_mask_for_buttons(final_buttons)
    guest_operations = lower_guest_operations(operations)
    semantic_payload = [
        {"kind": operation.kind, "args": list(operation.args)}
        for operation in operations
    ]
    lowered_payload = [
        {"kind": operation.kind, "args": list(operation.args)}
        for operation in guest_operations
    ]
    lines = [
        "import json, sys, traceback, time as _r1a_time, pyautogui",
        "pyautogui.FAILSAFE=False",
        "pyautogui.PAUSE=0",
        "_r1a_trace=[]",
        f"_r1a_expected_mask={expected_mask}",
        f"_r1a_expected_initial_mask={pointer_mask_for_buttons(initial_buttons)}",
        f"_r1a_touched_buttons=set({sorted(initial_buttons)!r})",
        f"_r1a_touched_keys=set({[pyautogui_key(k) for k in sorted(initial_keys)]!r})",
        "_r1a_error=None",
        "_r1a_failure_kind=None",
        "_r1a_cleanup=False",
        "_r1a_observed_mask=-1",
        "_r1a_cursor_before=[-1,-1]",
        "_r1a_cursor_after=[-1,-1]",
        f"_r1a_semantic_operations={semantic_payload!r}",
        f"_r1a_lowered_operations={lowered_payload!r}",
        f"_r1a_click_backend={click_backend!r}",
        "_r1a_backend_primitives=[]",
        "_r1a_x_event_sync=[]",
        "_r1a_x_injections=[]",
        "_r1a_click_timings=[]",
        "_r1a_x_injection_sequence=0",
        "_r1a_x_phase='outside_click'",
        "_r1a_passive_x_observer={'installed':False,'observer_process_count':0,'additional_x_connection_count':0,'assessment':'omitted_not_demonstrably_non_perturbing','limitation':"
        f"{PASSIVE_X_OBSERVER_LIMITATION!r}" "}",
        "def _r1a_sync_after_x_event(_event):",
        "    _backend=pyautogui.platformModule",
        "    _display=getattr(_backend,'_display',None)",
        "    _flush=getattr(_display,'flush',None)",
        "    _sync=getattr(_display,'sync',None)",
        "    _supported=callable(_flush) and callable(_sync)",
        "    _started=_r1a_time.monotonic_ns()",
        "    if not _supported:",
        "        _completed=_r1a_time.monotonic_ns()",
        "        _r1a_x_event_sync.append({'event':_event,'backend':getattr(_backend,'__name__',type(_backend).__name__),'flush':False,'sync':False,'started_guest_monotonic_ns':_started,'completed_guest_monotonic_ns':_completed,'duration_ns':_completed-_started})",
        "        raise RuntimeError('X11 flush/sync unavailable after '+_event)",
        "    _flush()",
        "    _sync()",
        "    _completed=_r1a_time.monotonic_ns()",
        "    _r1a_x_event_sync.append({'event':_event,'backend':getattr(_backend,'__name__',type(_backend).__name__),'flush':True,'sync':True,'started_guest_monotonic_ns':_started,'completed_guest_monotonic_ns':_completed,'duration_ns':_completed-_started})",
        "def _r1a_click(_button):",
        "    global _r1a_x_injection_sequence,_r1a_x_phase",
        "    _backend=pyautogui.platformModule",
        "    _display=getattr(_backend,'_display',None)",
        "    _down=getattr(_backend,'_mouseDown',None)",
        "    _up=getattr(_backend,'_mouseUp',None)",
        "    _move=getattr(_backend,'_moveTo',None)",
        "    _fake_input=getattr(_backend,'fake_input',None)",
        "    _x11=getattr(_backend,'X',None)",
        "    _button_map=getattr(_backend,'BUTTON_NAME_MAPPING',None)",
        "    _hooked=callable(_down) and callable(_up) and callable(_move) and callable(_fake_input) and _x11 is not None and isinstance(_button_map,dict) and callable(getattr(_display,'flush',None)) and callable(getattr(_display,'sync',None))",
        "    _release_motion=_r1a_click_backend=="
        f"{PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND!r}",
        "    _primitive={'kind':'click','button':_button,'call':'pyautogui.click(clicks=1, interval=0.05)','click_backend':_r1a_click_backend,'x11_per_event_sync_hooked':_hooked,'dwell_ms':50,'release_side_motion_notify':_release_motion,'injection_attempt_count':1,'retry_count':0,'ordering':['mouse_down','flush','sync','dwell','mouse_up','flush','sync']}",
        "    _r1a_backend_primitives.append(_primitive)",
        "    if not _hooked:",
        "        raise RuntimeError('X11 click primitive hooks unavailable')",
        "    _event_names={int(_x11.MotionNotify):'motion_notify',int(_x11.ButtonPress):'button_press',int(_x11.ButtonRelease):'button_release'}",
        "    def _r1a_traced_fake_input(*_args,**_kwargs):",
        "        global _r1a_x_injection_sequence",
        "        _event_type=int(_args[1] if len(_args)>1 else _kwargs.get('event_type',-1))",
        "        _detail=int(_args[2] if len(_args)>2 else _kwargs.get('detail',0))",
        "        _started=_r1a_time.monotonic_ns()",
        "        _result=_fake_input(*_args,**_kwargs)",
        "        _completed=_r1a_time.monotonic_ns()",
        "        _r1a_x_injection_sequence+=1",
        "        _r1a_x_injections.append({'sequence':_r1a_x_injection_sequence,'phase':_r1a_x_phase,'event':_event_names.get(_event_type,'event_'+str(_event_type)),'event_type':_event_type,'detail':_detail,'x':_kwargs.get('x'),'y':_kwargs.get('y'),'started_guest_monotonic_ns':_started,'completed_guest_monotonic_ns':_completed,'duration_ns':_completed-_started})",
        "        return _result",
        "    def _r1a_direct_down(*_args,**_kwargs):",
        "        _x,_y,_raw_button=_args[:3]",
        # Preserve the current backend's press-side XTest stream exactly.  The
        # experimental delta is only the release-side MotionNotify below.
        "        _move(_x,_y)",
        "        _backend.fake_input(_display,_x11.ButtonPress,_button_map[_raw_button])",
        "        _display.sync()",
        "    def _r1a_direct_up(*_args,**_kwargs):",
        "        _raw_button=_args[2]",
        "        _backend.fake_input(_display,_x11.ButtonRelease,_button_map[_raw_button])",
        "        _display.sync()",
        "    _timing={'click_backend':_r1a_click_backend,'backend_identity':getattr(_backend,'__name__',type(_backend).__name__),'release_side_motion_notify':_release_motion,'clock':'time.monotonic_ns','dwell_requested_ns':50000000,'x_injection_start_sequence':_r1a_x_injection_sequence}",
        "    def _r1a_down(*_args,**_kwargs):",
        "        global _r1a_x_phase",
        "        _prior_phase=_r1a_x_phase",
        "        _r1a_x_phase='press'",
        "        _timing['press_call_before_guest_monotonic_ns']=_r1a_time.monotonic_ns()",
        "        try:",
        "            _result=(_down if _release_motion else _r1a_direct_down)(*_args,**_kwargs)",
        "        finally:",
        "            _timing['press_call_after_guest_monotonic_ns']=_r1a_time.monotonic_ns()",
        "            _r1a_x_phase=_prior_phase",
        "        _r1a_sync_after_x_event('mouse_down')",
        "        _timing['press_sync_completed_guest_monotonic_ns']=_r1a_time.monotonic_ns()",
        # pyautogui.click's interval is between repeated clicks, not between
        # press and release.  Chromium intermittently observed only pointerdown
        # when the two XTest events were adjacent even though X11 reported a
        # released final mask.  A bounded dwell after the synced press keeps
        # the fixed click primitive while making browser receipt causal.
        "        _timing['dwell_started_guest_monotonic_ns']=_r1a_time.monotonic_ns()",
        f"        _r1a_time.sleep({CLICK_DWELL_S!r})",
        "        _timing['dwell_completed_guest_monotonic_ns']=_r1a_time.monotonic_ns()",
        "        _timing['dwell_duration_ns']=_timing['dwell_completed_guest_monotonic_ns']-_timing['dwell_started_guest_monotonic_ns']",
        "        _r1a_trace.append({'kind':'mouse_down','args':[_button]})",
        "        return _result",
        "    def _r1a_up(*_args,**_kwargs):",
        "        global _r1a_x_phase",
        "        _prior_phase=_r1a_x_phase",
        "        _r1a_x_phase='release'",
        "        _timing['release_call_before_guest_monotonic_ns']=_r1a_time.monotonic_ns()",
        "        try:",
        "            _result=(_up if _release_motion else _r1a_direct_up)(*_args,**_kwargs)",
        "        finally:",
        "            _timing['release_call_after_guest_monotonic_ns']=_r1a_time.monotonic_ns()",
        "            _r1a_x_phase=_prior_phase",
        "        _r1a_sync_after_x_event('mouse_up')",
        "        _timing['release_sync_completed_guest_monotonic_ns']=_r1a_time.monotonic_ns()",
        "        _r1a_trace.append({'kind':'mouse_up','args':[_button]})",
        "        return _result",
        "    _backend.fake_input=_r1a_traced_fake_input",
        "    _backend._mouseDown=_r1a_down",
        "    _backend._mouseUp=_r1a_up",
        "    _timing['click_started_guest_monotonic_ns']=_r1a_time.monotonic_ns()",
        "    try:",
        "        pyautogui.click(clicks=1,interval=0.05,button=_button)",
        "    finally:",
        "        _timing['click_completed_guest_monotonic_ns']=_r1a_time.monotonic_ns()",
        "        _timing['press_xtest_sequence']=[_item['event'] for _item in _r1a_x_injections if _item['sequence']>_timing['x_injection_start_sequence'] and _item['phase']=='press']",
        "        _timing['release_xtest_sequence']=[_item['event'] for _item in _r1a_x_injections if _item['sequence']>_timing['x_injection_start_sequence'] and _item['phase']=='release']",
        "        _primitive['press_xtest_sequence']=list(_timing['press_xtest_sequence'])",
        "        _primitive['release_xtest_sequence']=list(_timing['release_xtest_sequence'])",
        "        _r1a_click_timings.append(_timing)",
        "        _backend._mouseDown=_down",
        "        _backend._mouseUp=_up",
        "        _backend.fake_input=_fake_input",
        "def _r1a_pointer_state():",
        "    _backend=pyautogui.platformModule",
        "    _backend._display.sync()",
        "    _pointer=_backend._display.screen().root.query_pointer()",
        f"    return int(_pointer.root_x),int(_pointer.root_y),int(_pointer.mask)&{ALL_POINTER_BUTTON_MASK}",
        "try:",
        "    _r1a_bx,_r1a_by,_r1a_initial_mask=_r1a_pointer_state()",
        "    _r1a_cursor_before=[_r1a_bx,_r1a_by]",
        "    if _r1a_initial_mask != _r1a_expected_initial_mask:",
        "        _r1a_failure_kind='verification'",
        "        raise RuntimeError(f'initial pointer button mask {_r1a_initial_mask} != expected {_r1a_expected_initial_mask}')",
    ]
    indent = "    "
    for index, operation in enumerate(guest_operations):
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
                    f"{indent}_r1a_backend_primitives.append({{'kind':'move_to','call':'pyautogui.moveTo','requested_delta':[{dx},{dy}],'cursor_before':[int(_x),int(_y)],'cursor_after':[_tx,_ty],'actual_delta':[_tx-int(_x),_ty-int(_y)],'clamped':(_tx-int(_x),_ty-int(_y))!=({dx},{dy})}})",
                    f"{indent}_r1a_trace.append({{'kind':'move_to','args':[_tx,_ty]}})",
                ]
            )
        elif kind == "move_to":
            x, y = int(args[0]), int(args[1])
            lines.extend(
                [
                    f"{indent}_x,_y=pyautogui.position()",
                    f"{indent}_w,_h=pyautogui.size()",
                    f"{indent}_tx=max(0,min(int(_w)-1,{x}))",
                    f"{indent}_ty=max(0,min(int(_h)-1,{y}))",
                    f"{indent}pyautogui.moveTo(_tx,_ty)",
                    f"{indent}_r1a_backend_primitives.append({{'kind':'move_to','call':'pyautogui.moveTo','requested_position':[{x},{y}],'cursor_before':[int(_x),int(_y)],'cursor_after':[_tx,_ty],'clamped':(_tx,_ty)!=({x},{y})}})",
                    f"{indent}_r1a_trace.append({{'kind':'move_to','args':[_tx,_ty]}})",
                ]
            )
        elif kind == "scroll":
            clicks = int(args[0])
            lines.extend(
                [
                    f"{indent}pyautogui.scroll({clicks})",
                    f"{indent}_r1a_backend_primitives.append({{'kind':'scroll','call':'pyautogui.scroll','clicks':{clicks}}})",
                    f"{indent}_r1a_trace.append({{'kind':'scroll','args':[{clicks}]}})",
                ]
            )
        elif kind == "click":
            button = str(args[0])
            lines.extend(
                [
                    f"{indent}_r1a_touched_buttons.add({button!r})",
                    f"{indent}_r1a_click({button!r})",
                ]
            )
        elif kind in {"mouse_down", "mouse_up"}:
            button = str(args[0])
            method = "mouseDown" if kind == "mouse_down" else "mouseUp"
            lines.extend(
                [
                    f"{indent}_r1a_touched_buttons.add({button!r})",
                    f"{indent}pyautogui.{method}(button={button!r})",
                    f"{indent}_r1a_sync_after_x_event({kind!r})",
                    f"{indent}_r1a_backend_primitives.append({{'kind':{kind!r},'button':{button!r},'call':'pyautogui.{method}'}})",
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
                    f"{indent}_r1a_backend_primitives.append({{'kind':{kind!r},'key':{key!r},'mapped_key':{mapped!r},'call':'pyautogui.{method}'}})",
                    f"{indent}_r1a_trace.append({{'kind':{kind!r},'args':[{key!r}]}})",
                ]
            )
        elif kind == "coalesced_type":
            text = str(args[0])
            lines.extend(
                [
                    f"{indent}{compile_unicode_coalesced_type(text)}",
                    f"{indent}_r1a_backend_primitives.append({{'kind':'coalesced_type','call':'gtk_clipboard_ctrl_v','utf8_bytes':{len(text.encode('utf-8'))}}})",
                    f"{indent}_r1a_trace.append({{'kind':'coalesced_type','args':[{text!r}]}})",
                ]
            )
        elif kind == "wait":
            seconds = max(0.0, min(10.0, float(args[0])))
            lines.extend(
                [
                    f"{indent}_r1a_time.sleep({seconds!r})",
                    f"{indent}_r1a_backend_primitives.append({{'kind':'wait','call':'time.sleep','seconds':{seconds!r}}})",
                    f"{indent}_r1a_trace.append({{'kind':'wait','args':[{seconds!r}]}})",
                ]
            )
        elif kind == "raise_for_test":
            lines.extend(
                [
                    f"{indent}_r1a_failure_kind='injected'",
                    f"{indent}raise RuntimeError({str(args[0])!r})",
                ]
            )
        else:
            raise TransportError(f"unsupported atomic operation: {kind}")
    lines.extend(
        [
            f"{indent}_cx,_cy,_r1a_observed_mask=_r1a_pointer_state()",
            f"{indent}_r1a_cursor_after=[_cx,_cy]",
            f"{indent}if _r1a_observed_mask != _r1a_expected_mask:",
            f"{indent}    _r1a_failure_kind='verification'",
            f"{indent}    raise RuntimeError(f'pointer button mask {{_r1a_observed_mask}} != expected {{_r1a_expected_mask}}')",
            "except BaseException as _exc:",
            "    _r1a_error=''.join(traceback.format_exception_only(type(_exc),_exc)).strip()",
            "    if _r1a_failure_kind is None: _r1a_failure_kind='infrastructure'",
            "    _r1a_cleanup=True",
            "    for _key in sorted(_r1a_touched_keys,reverse=True):",
            "        try: pyautogui.keyUp(_key)",
            "        except BaseException: pass",
            "    for _button in ('left','middle','right'):",
            "        try: pyautogui.mouseUp(button=_button)",
            "        except BaseException: pass",
            "_r1a_cx,_r1a_cy,_r1a_final_mask=_r1a_pointer_state()",
            "_r1a_cursor_after=[_r1a_cx,_r1a_cy]",
            "_r1a_payload={'ok':_r1a_error is None,'cursor':[_r1a_cx,_r1a_cy],",
            " 'cursor_before':_r1a_cursor_before,'cursor_after':_r1a_cursor_after,",
            " '_r1a_schema':1,'pointer_button_mask':_r1a_final_mask,",
            " 'observed_pointer_button_mask':_r1a_observed_mask,",
            " 'expected_pointer_button_mask':_r1a_expected_mask,",
            " 'guest_process_count':1,'cleanup_attempted':_r1a_cleanup,",
            " 'error':_r1a_error,'failure_kind':_r1a_failure_kind,",
            " 'operations':_r1a_trace,'semantic_operations':_r1a_semantic_operations,",
            " 'lowered_operations':_r1a_lowered_operations,",
            " 'backend_primitives':_r1a_backend_primitives,",
            " 'x_event_sync_evidence':_r1a_x_event_sync,",
            " 'click_backend':_r1a_click_backend,",
            " 'x_injection_evidence':_r1a_x_injections,",
            " 'x_injection_timestamps':_r1a_click_timings,",
            " 'passive_x_observer':_r1a_passive_x_observer}",
            f"print({ATOMIC_RESULT_PREFIX!r}+json.dumps(_r1a_payload,separators=(',',':'),ensure_ascii=False))",
            "if _r1a_error is not None: sys.exit(1)",
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

    def execute_argv(
        self, argv: list[str], *, check: bool = True
    ) -> dict[str, Any]:
        result = self._request_json(
            "POST", "/execute", {"command": argv, "shell": False}
        )
        if not isinstance(result, dict):
            raise TransportError("VM /execute returned a non-object")
        if check and (
            result.get("status") != "success" or result.get("returncode") != 0
        ):
            raise TransportError(
                f"guest command failed: status={result.get('status')!r} "
                f"rc={result.get('returncode')!r} stderr={result.get('error')!r}"
            )
        return result

    def execute_pyautogui(self, code: str) -> None:
        self.execute_argv(["python", "-c", self._PREFIX + code])

    def execute_atomic(
        self,
        operations: tuple[Operation, ...],
        *,
        click_backend: str = PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
    ) -> AtomicExecutionResult:
        program, expected_mask = compile_atomic_guest_program(
            operations,
            initial_buttons=set(self.audit.held_buttons),
            initial_keys=set(self.audit.held_keys),
            click_backend=click_backend,
        )
        _, expected_keys = expected_atomic_input_state(
            operations,
            initial_buttons=set(self.audit.held_buttons),
            initial_keys=set(self.audit.held_keys),
        )
        result = self.execute_argv(["python", "-c", program], check=False)
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
        cursor_before = payload.get("cursor_before")
        cursor_after = payload.get("cursor_after")
        raw_operations = payload.get("operations")
        raw_semantic = payload.get("semantic_operations")
        raw_lowered = payload.get("lowered_operations")
        raw_primitives = payload.get("backend_primitives", [])
        raw_sync_evidence = payload.get("x_event_sync_evidence", [])
        raw_x_injections = payload.get("x_injection_evidence", [])
        raw_click_timings = payload.get("x_injection_timestamps", [])
        passive_x_observer = payload.get("passive_x_observer")
        if (
            not isinstance(cursor, list)
            or len(cursor) != 2
            or not isinstance(cursor_before, list)
            or len(cursor_before) != 2
            or not isinstance(cursor_after, list)
            or len(cursor_after) != 2
            or not isinstance(raw_operations, list)
            or not isinstance(raw_semantic, list)
            or not isinstance(raw_lowered, list)
            or not isinstance(raw_primitives, list)
            or not isinstance(raw_sync_evidence, list)
            or not isinstance(raw_x_injections, list)
            or not isinstance(raw_click_timings, list)
            or not isinstance(passive_x_observer, dict)
            or not all(isinstance(item, dict) for item in raw_primitives)
            or not all(isinstance(item, dict) for item in raw_sync_evidence)
            or not all(isinstance(item, dict) for item in raw_x_injections)
            or not all(isinstance(item, dict) for item in raw_click_timings)
        ):
            raise TransportError("atomic guest action returned invalid state")
        if payload.get("click_backend") != click_backend:
            raise TransportError("atomic guest action click backend drifted")
        if passive_x_observer != {
            "installed": False,
            "observer_process_count": 0,
            "additional_x_connection_count": 0,
            "assessment": "omitted_not_demonstrably_non_perturbing",
            "limitation": PASSIVE_X_OBSERVER_LIMITATION,
        }:
            raise TransportError("atomic guest action passive X observer evidence drifted")

        def validate_guest_timestamps(
            records: list[dict[str, Any]], *, fields: tuple[str, str]
        ) -> None:
            started_field, completed_field = fields
            for record in records:
                started = record.get(started_field)
                completed = record.get(completed_field)
                duration = record.get("duration_ns")
                if (
                    not isinstance(started, int)
                    or isinstance(started, bool)
                    or not isinstance(completed, int)
                    or isinstance(completed, bool)
                    or completed < started
                    or not isinstance(duration, int)
                    or isinstance(duration, bool)
                    or duration != completed - started
                ):
                    raise TransportError(
                        "atomic guest action returned invalid monotonic timestamp evidence"
                    )

        validate_guest_timestamps(
            raw_sync_evidence,
            fields=(
                "started_guest_monotonic_ns",
                "completed_guest_monotonic_ns",
            ),
        )
        validate_guest_timestamps(
            raw_x_injections,
            fields=(
                "started_guest_monotonic_ns",
                "completed_guest_monotonic_ns",
            ),
        )
        injection_sequences = [item.get("sequence") for item in raw_x_injections]
        if (
            not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in injection_sequences
            )
            or injection_sequences != list(range(1, len(raw_x_injections) + 1))
        ):
            raise TransportError("atomic guest action X injection sequence drifted")
        x_event_names = {4: "button_press", 5: "button_release", 6: "motion_notify"}
        for item in raw_x_injections:
            event_type = item.get("event_type")
            detail = item.get("detail")
            x = item.get("x")
            y = item.get("y")
            if (
                item.get("phase") not in {"outside_click", "press", "release"}
                or not isinstance(event_type, int)
                or isinstance(event_type, bool)
                or event_type not in x_event_names
                or item.get("event") != x_event_names[event_type]
                or not isinstance(detail, int)
                or isinstance(detail, bool)
                or ((x is None) != (y is None))
                or (
                    x is not None
                    and (
                        not isinstance(x, int)
                        or isinstance(x, bool)
                        or not isinstance(y, int)
                        or isinstance(y, bool)
                    )
                )
            ):
                raise TransportError("atomic guest action X injection identity drifted")
        expected_click_count = sum(
            operation.kind == "click" for operation in lower_guest_operations(operations)
        )
        expected_press_sequence = ["motion_notify", "button_press"]
        expected_release_sequence = (
            ["motion_notify", "button_release"]
            if click_backend == PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND
            else ["button_release"]
        )
        click_primitives = [
            primitive
            for primitive in raw_primitives
            if primitive.get("kind") == "click"
        ]
        if len(click_primitives) != expected_click_count:
            raise TransportError("atomic guest action click primitive count drifted")
        for primitive in click_primitives:
            if (
                primitive.get("click_backend") != click_backend
                or primitive.get("release_side_motion_notify")
                != (click_backend == PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND)
                or primitive.get("injection_attempt_count") != 1
                or primitive.get("retry_count") != 0
                or primitive.get("dwell_ms") != int(CLICK_DWELL_S * 1000)
                or primitive.get("press_xtest_sequence") != expected_press_sequence
                or primitive.get("release_xtest_sequence")
                != expected_release_sequence
            ):
                raise TransportError("atomic guest action click primitive drifted")
        if len(raw_click_timings) != expected_click_count:
            raise TransportError("atomic guest action click timing count drifted")
        covered_controlled_sequences: set[int] = set()
        for timing, primitive in zip(raw_click_timings, click_primitives):
            required_timestamps = (
                "click_started_guest_monotonic_ns",
                "press_call_before_guest_monotonic_ns",
                "press_call_after_guest_monotonic_ns",
                "press_sync_completed_guest_monotonic_ns",
                "dwell_started_guest_monotonic_ns",
                "dwell_completed_guest_monotonic_ns",
                "release_call_before_guest_monotonic_ns",
                "release_call_after_guest_monotonic_ns",
                "release_sync_completed_guest_monotonic_ns",
                "click_completed_guest_monotonic_ns",
            )
            values = [timing.get(field) for field in required_timestamps]
            if (
                timing.get("click_backend") != click_backend
                or not isinstance(timing.get("backend_identity"), str)
                or timing.get("release_side_motion_notify")
                != (click_backend == PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND)
                or timing.get("clock") != "time.monotonic_ns"
                or timing.get("dwell_requested_ns") != int(CLICK_DWELL_S * 1e9)
                or timing.get("press_xtest_sequence") != expected_press_sequence
                or timing.get("release_xtest_sequence")
                != expected_release_sequence
                or any(not isinstance(value, int) for value in values)
                or values != sorted(values)
                or timing.get("dwell_duration_ns")
                != timing.get("dwell_completed_guest_monotonic_ns")
                - timing.get("dwell_started_guest_monotonic_ns")
            ):
                raise TransportError("atomic guest action click timing evidence drifted")
            start_sequence = timing.get("x_injection_start_sequence")
            button_detail = {"left": 1, "middle": 2, "right": 3}.get(
                primitive.get("button")
            )
            expected_injections = [
                ("press", "motion_notify", 6, 0),
                ("press", "button_press", 4, button_detail),
                *(
                    [("release", "motion_notify", 6, 0)]
                    if click_backend == PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND
                    else []
                ),
                ("release", "button_release", 5, button_detail),
            ]
            if (
                not isinstance(start_sequence, int)
                or isinstance(start_sequence, bool)
                or button_detail is None
            ):
                raise TransportError("atomic guest action click X sequence origin drifted")
            controlled = [
                item
                for item in raw_x_injections
                if start_sequence
                < item["sequence"]
                <= start_sequence + len(expected_injections)
            ]
            if [item["sequence"] for item in controlled] != list(
                range(start_sequence + 1, start_sequence + 1 + len(expected_injections))
            ):
                raise TransportError("atomic guest action controlled X sequence drifted")
            for item, expected in zip(controlled, expected_injections):
                if (
                    item.get("phase"),
                    item.get("event"),
                    item.get("event_type"),
                    item.get("detail"),
                ) != expected:
                    raise TransportError("atomic guest action controlled X identity drifted")
                covered_controlled_sequences.add(item["sequence"])
            controlled_clock_values = [
                value
                for item in controlled
                for value in (
                    item["started_guest_monotonic_ns"],
                    item["completed_guest_monotonic_ns"],
                )
            ]
            if controlled_clock_values != sorted(controlled_clock_values):
                raise TransportError("atomic guest action controlled X clock drifted")
            press_motion = controlled[0]
            press_coordinates = (press_motion.get("x"), press_motion.get("y"))
            if not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in press_coordinates
            ):
                raise TransportError("atomic guest action press coordinates drifted")
            for item in controlled:
                if item["event"] in {"button_press", "button_release"} and (
                    item.get("x") is not None or item.get("y") is not None
                ):
                    raise TransportError("atomic guest action button coordinates drifted")
            release_motions = [
                item
                for item in controlled
                if item["phase"] == "release" and item["event"] == "motion_notify"
            ]
            if release_motions and (
                release_motions[0].get("x"), release_motions[0].get("y")
            ) != press_coordinates:
                raise TransportError(
                    "atomic guest action release motion coordinates drifted"
                )
            if not (
                timing["press_call_before_guest_monotonic_ns"]
                <= controlled[0]["started_guest_monotonic_ns"]
                <= controlled[1]["completed_guest_monotonic_ns"]
                <= timing["press_call_after_guest_monotonic_ns"]
                <= timing["press_sync_completed_guest_monotonic_ns"]
                <= timing["dwell_started_guest_monotonic_ns"]
                <= timing["dwell_completed_guest_monotonic_ns"]
                <= timing["release_call_before_guest_monotonic_ns"]
                <= controlled[2]["started_guest_monotonic_ns"]
                <= controlled[-1]["completed_guest_monotonic_ns"]
                <= timing["release_call_after_guest_monotonic_ns"]
                <= timing["release_sync_completed_guest_monotonic_ns"]
            ):
                raise TransportError("atomic guest action X timing order drifted")
        all_controlled_sequences = {
            item["sequence"]
            for item in raw_x_injections
            if item.get("phase") in {"press", "release"}
        }
        if covered_controlled_sequences != all_controlled_sequences:
            raise TransportError("atomic guest action unbound controlled X event")

        def parse_operations(raw: list[Any], label: str) -> tuple[Operation, ...]:
            parsed: list[Operation] = []
            for item in raw:
                if not isinstance(item, dict) or not isinstance(item.get("args"), list):
                    raise TransportError(
                        f"atomic guest action returned invalid {label} trace"
                    )
                parsed.append(Operation(str(item.get("kind", "")), tuple(item["args"])))
            return tuple(parsed)

        traced = parse_operations(raw_operations, "executed")
        semantic = parse_operations(raw_semantic, "semantic")
        lowered = parse_operations(raw_lowered, "lowered")
        if semantic != operations or lowered != lower_guest_operations(operations):
            raise TransportError("atomic guest action operation streams drifted")
        pointer_mask = int(payload.get("pointer_button_mask", -1))
        reported_expected = int(payload.get("expected_pointer_button_mask", -1))
        if reported_expected != expected_mask or pointer_mask < 0:
            raise TransportError("atomic guest action mask contract mismatch")
        ok = bool(payload.get("ok"))
        returncode = int(result.get("returncode", -1))
        if ok != (returncode == 0):
            raise TransportError(
                "atomic guest action success/returncode contract mismatch"
            )
        if ok and result.get("status") != "success":
            raise TransportError("atomic guest action returned an invalid status")
        failure_kind = payload.get("failure_kind")
        if failure_kind not in {None, "verification", "infrastructure", "injected"}:
            raise TransportError("atomic guest action returned invalid failure kind")
        if ok != (failure_kind is None):
            raise TransportError("atomic guest action failure classification mismatch")
        if ok != (payload.get("error") is None):
            raise TransportError("atomic guest action error contract mismatch")
        if [int(cursor[0]), int(cursor[1])] != [
            int(cursor_after[0]),
            int(cursor_after[1]),
        ]:
            raise TransportError("atomic guest cursor alias/readback mismatch")
        guest_process_count = int(payload.get("guest_process_count", 0))
        if guest_process_count != 1:
            raise TransportError("atomic action did not use exactly one guest process")
        atomic_result = AtomicExecutionResult(
            ok=ok,
            cursor=(int(cursor[0]), int(cursor[1])),
            cursor_before=(int(cursor_before[0]), int(cursor_before[1])),
            cursor_after=(int(cursor_after[0]), int(cursor_after[1])),
            pointer_button_mask=pointer_mask,
            observed_pointer_button_mask=int(
                payload.get("observed_pointer_button_mask", -1)
            ),
            expected_pointer_button_mask=reported_expected,
            guest_process_count=guest_process_count,
            guest_returncode=returncode,
            raw_result_marker=lines[0],
            cleanup_attempted=bool(payload.get("cleanup_attempted")),
            error=None if payload.get("error") is None else str(payload["error"]),
            failure_kind=None if failure_kind is None else str(failure_kind),
            operations=traced,
            semantic_operations=semantic,
            lowered_operations=lowered,
            backend_primitives=tuple(dict(item) for item in raw_primitives),
            x_event_sync_evidence=tuple(dict(item) for item in raw_sync_evidence),
            click_backend=click_backend,
            x_injection_evidence=tuple(dict(item) for item in raw_x_injections),
            x_injection_timestamps=tuple(dict(item) for item in raw_click_timings),
            passive_x_observer=dict(passive_x_observer),
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
        self.atomic_inputs: list[tuple[Operation, ...]] = []

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

    def execute_atomic(
        self, operations: tuple[Operation, ...]
    ) -> AtomicExecutionResult:
        self.atomic_invocations += 1
        self.atomic_inputs.append(operations)
        before = len(self.audit.operations)
        cursor_before = self._cursor
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
        failure_kind: str | None = None
        lowered = lower_guest_operations(operations)
        backend_primitives: list[dict[str, Any]] = []
        sync_evidence: list[dict[str, Any]] = []
        try:
            for operation in lowered:
                kind, args = operation.kind, operation.args
                if kind == "move_relative":
                    requested = (int(args[0]), int(args[1]))
                    move_before = self._cursor
                    self.move_to(
                        self._cursor[0] + requested[0],
                        self._cursor[1] + requested[1],
                    )
                    actual = (
                        self._cursor[0] - move_before[0],
                        self._cursor[1] - move_before[1],
                    )
                    backend_primitives.append(
                        {
                            "kind": "move_to",
                            "call": "recording.move_to",
                            "requested_delta": list(requested),
                            "cursor_before": list(move_before),
                            "cursor_after": list(self._cursor),
                            "actual_delta": list(actual),
                            "clamped": actual != requested,
                        }
                    )
                elif kind == "move_to":
                    requested = (int(args[0]), int(args[1]))
                    move_before = self._cursor
                    self.move_to(*requested)
                    backend_primitives.append(
                        {
                            "kind": "move_to",
                            "call": "recording.move_to",
                            "requested_position": list(requested),
                            "cursor_before": list(move_before),
                            "cursor_after": list(self._cursor),
                            "clamped": self._cursor != requested,
                        }
                    )
                elif kind == "scroll":
                    self.scroll(int(args[0]))
                    backend_primitives.append(
                        {
                            "kind": "scroll",
                            "call": "recording.scroll",
                            "clicks": int(args[0]),
                        }
                    )
                elif kind == "mouse_down":
                    self.mouse_down(str(args[0]))
                    backend_primitives.append(
                        {"kind": kind, "button": str(args[0]), "call": "recording.mouse_down"}
                    )
                    sync_evidence.append(
                        {"event": kind, "backend": "recording_x11", "flush": True, "sync": True}
                    )
                elif kind == "mouse_up":
                    self.mouse_up(str(args[0]))
                    backend_primitives.append(
                        {"kind": kind, "button": str(args[0]), "call": "recording.mouse_up"}
                    )
                    sync_evidence.append(
                        {"event": kind, "backend": "recording_x11", "flush": True, "sync": True}
                    )
                elif kind == "click":
                    self.mouse_down(str(args[0]))
                    sync_evidence.append(
                        {"event": "mouse_down", "backend": "recording_x11", "flush": True, "sync": True}
                    )
                    self.mouse_up(str(args[0]))
                    sync_evidence.append(
                        {"event": "mouse_up", "backend": "recording_x11", "flush": True, "sync": True}
                    )
                    backend_primitives.append(
                        {
                            "kind": "click",
                            "button": str(args[0]),
                            "call": "pyautogui.click(clicks=1, interval=0.05)",
                            "x11_per_event_sync_hooked": True,
                            "dwell_ms": 50,
                            "ordering": [
                                "mouse_down",
                                "flush",
                                "sync",
                                "dwell",
                                "mouse_up",
                                "flush",
                                "sync",
                            ],
                        }
                    )
                elif kind == "key_down":
                    key = str(args[0])
                    if key in self.audit.held_keys:
                        raise TransportError(f"key already held: {key}")
                    self.audit.held_keys.add(key)
                    self.audit.operations.append(Operation("key_down", (key,)))
                    backend_primitives.append(
                        {"kind": kind, "key": key, "call": "recording.key_down"}
                    )
                elif kind == "key_up":
                    key = str(args[0])
                    if key not in self.audit.held_keys:
                        raise TransportError(f"key not held: {key}")
                    self.audit.held_keys.remove(key)
                    self.audit.operations.append(Operation("key_up", (key,)))
                    backend_primitives.append(
                        {"kind": kind, "key": key, "call": "recording.key_up"}
                    )
                elif kind == "coalesced_type":
                    self.coalesced_type(str(args[0]))
                    backend_primitives.append(
                        {"kind": kind, "call": "recording.coalesced_type"}
                    )
                elif kind == "wait":
                    self.wait(float(args[0]))
                    backend_primitives.append(
                        {
                            "kind": kind,
                            "call": "recording.wait",
                            "seconds": float(args[0]),
                        }
                    )
                elif kind == "raise_for_test":
                    failure_kind = "injected"
                    raise RuntimeError(str(args[0]))
                else:
                    raise TransportError(f"unsupported atomic operation: {kind}")
            observed_mask = pointer_mask_for_buttons(self.audit.held_buttons)
            if observed_mask != expected_mask:
                failure_kind = "verification"
                raise TransportError(
                    f"pointer button mask {observed_mask} != expected {expected_mask}"
                )
            self.audit.held_keys = final_keys
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if failure_kind is None:
                failure_kind = "infrastructure"
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
            cursor_before=cursor_before,
            cursor_after=self._cursor,
            pointer_button_mask=final_mask,
            observed_pointer_button_mask=observed_mask,
            expected_pointer_button_mask=expected_mask,
            guest_process_count=1,
            guest_returncode=0 if error is None else 1,
            raw_result_marker=(
                f"{ATOMIC_RESULT_PREFIX}<recording:{'ok' if error is None else 'failed'}>"
            ),
            cleanup_attempted=cleanup_attempted,
            error=error,
            failure_kind=failure_kind,
            operations=tuple(self.audit.operations[before:]),
            semantic_operations=operations,
            lowered_operations=lowered,
            backend_primitives=tuple(backend_primitives),
            x_event_sync_evidence=tuple(sync_evidence),
        )
