from __future__ import annotations

from osworld_parity.proper_vm_capability_ladder.rung1.executor import (
    CompactRawExecutor,
    NativeAbsoluteExecutor,
)
from osworld_parity.proper_vm_capability_ladder.rung1.transport import (
    RecordingTransport,
    compile_unicode_coalesced_type,
)


def _kinds(transport: RecordingTransport) -> list[str]:
    return [operation.kind for operation in transport.audit.operations]


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
