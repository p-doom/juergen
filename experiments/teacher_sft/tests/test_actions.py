from __future__ import annotations

import pytest

from experiments.teacher_sft.actions import (
    SymbolicState,
    convert_native_action,
    parse_compact_sequence,
)
from experiments.teacher_sft.contracts import ContractError


def trace(before, after, target=None):
    return {
        "cursor_before": list(before),
        "cursor_after": list(after),
        "resolved_target_px": list(target) if target is not None else None,
    }


def test_click_drag_scroll_and_coalesced_typing_are_causal() -> None:
    state = SymbolicState(cursor=(100, 50), screen_size=(200, 100))
    click = {
        "action": "left_click",
        "coordinate": [750, 500],
        "coordinate_space": "absolute_grid",
        "coordinate_grid": 1000,
    }
    converted = convert_native_action(
        click, trace((100, 50), (150, 50), (150, 50)), state
    )
    assert [action.render() for action in converted] == ["50 0 0 ; +LMB -LMB"]

    drag = {
        "action": "left_click_drag",
        "coordinate": [20, 80],
        "coordinate_space": "absolute_px",
    }
    converted = convert_native_action(drag, trace((150, 50), (20, 80), (20, 80)), state)
    assert [action.render() for action in converted] == [
        "0 0 0 ; +LMB",
        "-130 30 0",
        "0 0 0 ; -LMB",
    ]

    typing = {"action": "type", "text": 'hello; +world "quoted"'}
    converted = convert_native_action(typing, trace((20, 80), (20, 80)), state)
    label = converted[0].render()
    assert label == '0 0 0 ; type("hello; +world \\"quoted\\"")'
    assert (
        parse_compact_sequence(label)[0].elements[0].value == 'hello; +world "quoted"'
    )

    scroll = convert_native_action(
        {"action": "scroll", "pixels": -7}, trace((20, 80), (20, 80)), state
    )
    assert scroll[0].render() == "0 0 -7"


def test_cursor_telemetry_mismatch_fails_closed() -> None:
    state = SymbolicState(cursor=(10, 10), screen_size=(100, 100))
    action = {
        "action": "mouse_move",
        "coordinate": [20, 20],
        "coordinate_space": "absolute_px",
    }
    with pytest.raises(ContractError, match="resolved-target mismatch"):
        convert_native_action(action, trace((10, 10), (21, 20), (21, 20)), state)


def test_unbalanced_release_and_noncanonical_lines_fail() -> None:
    state = SymbolicState(cursor=(0, 0), screen_size=(100, 100))
    with pytest.raises(ContractError, match="dangling release"):
        state.apply(parse_compact_sequence("0 0 0 ; -LMB")[0])
    with pytest.raises(ContractError, match="non-canonical"):
        parse_compact_sequence("0 0 0;+LMB -LMB")
