from __future__ import annotations

import json

import pytest

from osworld_parity.proper_vm_capability_ladder.rung1.executor import (
    CompactRawExecutor,
    NativeAbsoluteExecutor,
)
from osworld_parity.proper_vm_capability_ladder.rung1.fixtures import load_manifest
from osworld_parity.proper_vm_capability_ladder.rung1.trajectory import GoldTrajectory
from osworld_parity.proper_vm_capability_ladder.rung1.selfcheck import (
    SelfcheckError,
    _assert_dispatch_journal,
    _execute,
)
from osworld_parity.proper_vm_capability_ladder.rung1.transport import (
    ATOMIC_RESULT_PREFIX,
    HttpVmTransport,
    Operation,
    RecordingTransport,
    TransportError,
    compile_atomic_guest_program,
    compile_unicode_coalesced_type,
)


def _kinds(transport: RecordingTransport) -> list[str]:
    return [operation.kind for operation in transport.audit.operations]


class StatefulClickTransport(RecordingTransport):
    """Tiny GUI-state model: a click requires down/up on the same hitbox."""

    def __init__(self, *, cursor: tuple[int, int]) -> None:
        super().__init__(cursor=cursor)
        self.target = (290, 390, 310, 410)
        self.checked = False
        self._down_hit = ""

    def _hit(self) -> str:
        x, y = self.cursor_position()
        left, top, right, bottom = self.target
        return "target" if left <= x < right and top <= y < bottom else ""

    def mouse_down(self, button: str = "left") -> None:
        super().mouse_down(button)
        if button == "left":
            self._down_hit = self._hit()

    def mouse_up(self, button: str = "left") -> None:
        up_hit = self._hit()
        super().mouse_up(button)
        if button == "left" and self._down_hit == up_hit == "target":
            self.checked = not self.checked
        self._down_hit = ""


class SingleProcessHttpTransport(HttpVmTransport):
    def __init__(self) -> None:
        super().__init__("http://not-used.invalid")
        self.execute_calls: list[list[str]] = []

    def execute_argv(self, argv: list[str]) -> dict:
        self.execute_calls.append(argv)
        payload = {
            "_r1a_schema": 1,
            "ok": True,
            "cursor": [300, 400],
            "pointer_button_mask": 0,
            "observed_pointer_button_mask": 0,
            "expected_pointer_button_mask": 0,
            "guest_process_count": 1,
            "cleanup_attempted": False,
            "error": None,
            "operations": [
                {"kind": "move_to", "args": [300, 400]},
                {"kind": "mouse_down", "args": ["left"]},
                {"kind": "mouse_up", "args": ["left"]},
            ],
        }
        return {
            "status": "success",
            "returncode": 0,
            "output": ATOMIC_RESULT_PREFIX + json.dumps(payload),
        }

def test_click_adapters_match_cursor_and_button_transitions() -> None:
    native_transport = RecordingTransport(cursor=(10, 20))
    NativeAbsoluteExecutor(native_transport).execute(
        {"action": "left_click", "coordinate": [300, 400]}
    )
    raw_transport = RecordingTransport(cursor=(10, 20))
    CompactRawExecutor(raw_transport).execute("290 380 0 ; +LMB -LMB")
    assert native_transport.cursor_position() == raw_transport.cursor_position() == (300, 400)
    assert _kinds(native_transport) == _kinds(raw_transport) == [
        "move_to",
        "mouse_down",
        "mouse_up",
    ]
    assert not native_transport.audit.held_buttons
    assert not raw_transport.audit.held_buttons


def test_compact_click_is_one_guest_process_with_compiled_order() -> None:
    transport = SingleProcessHttpTransport()
    result = CompactRawExecutor(transport).execute("290 380 0 ; +LMB -LMB")
    assert len(transport.execute_calls) == 1
    argv = transport.execute_calls[0]
    assert argv[:2] == ["python", "-c"]
    program = argv[2]
    compile(program, "<rung1a-atomic>", "exec")
    assert program.index("RUNG1A_ATOMIC_STEP_0:move_relative") < program.index(
        "RUNG1A_ATOMIC_STEP_1:mouse_down"
    ) < program.index("RUNG1A_ATOMIC_STEP_2:mouse_up")
    assert result.executor_dispatch_status == "ok"
    assert result.atomic_state is not None
    assert result.atomic_state["guest_process_count"] == 1
    assert result.atomic_state["pointer_button_mask"] == 0


def test_atomic_exception_releases_preexisting_and_new_buttons() -> None:
    transport = RecordingTransport()
    transport.mouse_down("right")
    result = transport.execute_compact_atomic(
        (
            Operation("mouse_down", ("left",)),
            Operation("raise_for_test", ("injected failure",)),
        )
    )
    assert result.ok is False
    assert result.cleanup_attempted is True
    assert result.pointer_button_mask == 0
    assert transport.audit.held_buttons == set()
    assert "injected failure" in (result.error or "")

    program, _ = compile_atomic_guest_program(
        (
            Operation("mouse_down", ("left",)),
            Operation("raise_for_test", ("injected failure",)),
        ),
        initial_buttons=set(),
        initial_keys=set(),
    )
    assert "except BaseException" in program
    assert "pyautogui.mouseUp(button=_button)" in program


def test_atomic_compiler_rejects_impossible_held_button_transitions() -> None:
    with pytest.raises(TransportError, match="button not held"):
        compile_atomic_guest_program(
            (Operation("mouse_up", ("left",)),),
            initial_buttons=set(),
            initial_keys=set(),
        )
    with pytest.raises(TransportError, match="button already held"):
        compile_atomic_guest_program(
            (Operation("mouse_down", ("left",)),),
            initial_buttons={"left"},
            initial_keys=set(),
        )


def test_stateful_raw_click_requires_observed_baseline_and_endpoint() -> None:
    observed = (10, 20)
    exact = StatefulClickTransport(cursor=observed)
    trajectory = GoldTrajectory(
        arm="compact_raw_phaseb",
        actions=("290 380 0 ; +LMB -LMB",),
        observed_cursor_baseline=observed,
        expected_endpoint=(300, 400),
    )
    _, journal = _execute("compact_raw_phaseb", exact, trajectory)
    assert journal["baseline_matches"] is True
    assert journal["endpoint_matches"] is True
    assert journal["atomic_guest_process_count"] == 1
    assert journal["final_pointer_button_mask"] == 0
    assert journal["actions"][0]["guest_cursor_after"] == [300, 400]
    assert exact.checked is True

    # The same raw delta is a genuine negative when it was compiled against an
    # assumed/stale baseline rather than the cursor observed at dispatch.
    stale = StatefulClickTransport(cursor=(100, 100))
    CompactRawExecutor(stale).execute("290 380 0 ; +LMB -LMB")
    assert stale.cursor_position() == (390, 480)
    assert stale.checked is False

    # The production selfcheck now catches that mismatch before dispatch.
    guarded = StatefulClickTransport(cursor=(100, 100))
    records, stale_journal = _execute("compact_raw_phaseb", guarded, trajectory)
    assert records == []
    assert stale_journal["dispatch_status"] == "blocked_baseline_drift"
    assert stale_journal["baseline_matches"] is False
    assert stale_journal["endpoint_matches"] is False
    assert guarded.cursor_position() == (100, 100)
    assert guarded.checked is False


def test_pre_oracle_mask_requires_release_except_declared_hold_probe() -> None:
    fixture = next(
        item
        for item in load_manifest().select(split="development")
        if item.template == "click"
    )
    transport = RecordingTransport(cursor=(300, 400))
    trajectory = GoldTrajectory(
        arm="compact_raw_phaseb",
        actions=("0 0 0 ; +LMB",),
        observed_cursor_baseline=(300, 400),
        expected_endpoint=(300, 400),
    )
    _, journal = _execute("compact_raw_phaseb", transport, trajectory)
    assert journal["final_pointer_button_mask"] == 1 << 8
    with pytest.raises(SelfcheckError, match="pointer button mask"):
        _assert_dispatch_journal(fixture, "compact_raw_phaseb", "gold", journal)
    _assert_dispatch_journal(
        fixture,
        "compact_raw_phaseb",
        "held-button injection",
        journal,
        required_pointer_button_mask=1 << 8,
    )


def test_stateful_click_rejects_button_down_before_move() -> None:
    transport = StatefulClickTransport(cursor=(10, 20))
    raw = CompactRawExecutor(transport)
    raw.execute("0 0 0 ; +LMB")
    raw.execute("290 380 0")
    raw.execute("0 0 0 ; -LMB")
    assert transport.cursor_position() == (300, 400)
    assert transport.checked is False


def test_drag_adapters_match_hold_move_release_state() -> None:
    native_transport = RecordingTransport(cursor=(50, 50))
    native = NativeAbsoluteExecutor(native_transport)
    native.execute({"action": "mouse_move", "coordinate": [200, 300]})
    native.execute({"action": "left_click_drag", "coordinate": [700, 300]})
    raw_transport = RecordingTransport(cursor=(50, 50))
    raw = CompactRawExecutor(raw_transport)
    raw.execute("150 250 0 ; +LMB")
    assert raw_transport.audit.held_buttons == {"left"}
    raw.execute("500 0 0")
    raw.execute("0 0 0 ; -LMB")
    assert native_transport.cursor_position() == raw_transport.cursor_position() == (700, 300)
    assert _kinds(native_transport) == _kinds(raw_transport) == [
        "move_to",
        "mouse_down",
        "move_to",
        "mouse_up",
    ]
    assert not native_transport.audit.held_buttons
    assert not raw_transport.audit.held_buttons


def test_unicode_coalesced_type_is_one_shared_operation() -> None:
    text = "東京 Grüße λ ✓"
    native_transport = RecordingTransport()
    NativeAbsoluteExecutor(native_transport).execute({"action": "type", "text": text})
    raw_transport = RecordingTransport()
    CompactRawExecutor(raw_transport).execute('0 0 0 ; type("東京 Grüße λ ✓")')
    assert native_transport.audit.typed_texts == raw_transport.audit.typed_texts == [text]
    assert _kinds(native_transport) == _kinds(raw_transport) == ["coalesced_type"]
    compiled = compile_unicode_coalesced_type(text)
    assert "pyperclip.copy" in compiled
    assert "pyautogui.hotkey('ctrl', 'v')" in compiled
    assert "pyautogui.write" not in compiled
    atomic_program, _ = compile_atomic_guest_program(
        (Operation("coalesced_type", (text,)),),
        initial_buttons=set(),
        initial_keys=set(),
    )
    assert compiled in atomic_program


def test_signed_scroll_state_is_preserved_by_both_adapters() -> None:
    for clicks in (-7, 6):
        native_transport = RecordingTransport()
        NativeAbsoluteExecutor(native_transport).execute(
            {"action": "scroll", "clicks": clicks}
        )
        raw_transport = RecordingTransport()
        CompactRawExecutor(raw_transport).execute(f"0 0 {clicks}")
        assert native_transport.audit.scroll_total == clicks
        assert raw_transport.audit.scroll_total == clicks


def test_reset_model_recreates_cursor_button_type_and_scroll_state() -> None:
    dirty = RecordingTransport(cursor=(50, 50))
    dirty.move_to(900, 700)
    dirty.mouse_down("left")
    dirty.scroll(-8)
    dirty.coalesced_type("leak")
    assert dirty.audit.held_buttons == {"left"}
    clean = RecordingTransport(cursor=(50, 50))
    assert clean.cursor_position() == (50, 50)
    assert clean.audit.held_buttons == set()
    assert clean.audit.scroll_total == 0
    assert clean.audit.typed_texts == []
