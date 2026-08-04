from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[2]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from phaseb_canonical_eval import CanonicalError, canonical_normalized, canonical_raw
from phaseb_oracle_eval import normalized_calls


def block(arguments: str) -> str:
    return '<tool_call>{"name":"computer_use","arguments":' + arguments + "}</tool_call>"


def test_dual_channels_are_deduplicated_only_when_semantically_equal() -> None:
    text = block('{"action":"left_click"}')
    structured = [{"function": {"name": "computer_use", "arguments": '{"action":"left_click"}'}}]
    assert normalized_calls(text, structured) == [{"action": "left_click"}]
    disagree = [{"function": {"name": "computer_use", "arguments": '{"action":"right_click"}'}}]
    with pytest.raises(CanonicalError):
        normalized_calls(text, disagree)


def test_malformed_or_noncomputer_block_is_fail_loud() -> None:
    valid = block('{"action":"left_click"}')
    with pytest.raises(CanonicalError):
        normalized_calls("<tool_call>{bad}</tool_call>" + valid)
    with pytest.raises(CanonicalError):
        normalized_calls('<tool_call>{"name":"other","arguments":{}}</tool_call>')
    with pytest.raises(CanonicalError):
        normalized_calls("<tool_call>" + valid)


@pytest.mark.parametrize("coordinate", [[True, 0], [1000, 0], [float("inf"), 0], [float("nan"), 0]])
def test_invalid_coordinates_are_rejected(coordinate: list[object]) -> None:
    with pytest.raises(CanonicalError):
        canonical_normalized([{"action": "move_rel", "coordinate": coordinate}])


def test_fractional_coordinate_is_valid_but_click_coordinate_is_forbidden() -> None:
    assert canonical_normalized([{"action": "move_rel", "coordinate": [1.5, -2.5]}])
    with pytest.raises(CanonicalError):
        canonical_normalized([{"action": "left_click", "coordinate": [1, 2]}])


@pytest.mark.parametrize("pixels", [True, 1.5])
def test_scroll_requires_an_integer(pixels: object) -> None:
    with pytest.raises(CanonicalError):
        canonical_normalized([{"action": "scroll", "pixels": pixels}])


def test_button_and_key_transitions_are_stateless_across_oracle_turns() -> None:
    assert canonical_normalized([{"action": "mouse_up", "button": "left"}]) == (
        ("button_up", "left"),
    )
    with pytest.raises(CanonicalError):
        canonical_normalized([{"action": "mouse_down", "button": "side"}])
    assert canonical_normalized([{"action": "mouse_down", "button": "left"}]) == (
        ("button_down", "left"),
    )
    assert canonical_normalized([
        {"action": "key_down", "keys": ["ctrl"]},
        {"action": "key_up", "keys": ["ctrl"]},
    ]) == (("key_down", "ctrl"), ("key_up", "ctrl"))
    assert canonical_normalized([{"action": "key_up", "keys": ["ctrl"]}]) == (
        ("key_up", "ctrl"),
    )
    assert canonical_normalized([{"action": "key_down", "keys": ["ctrl"]}]) == (
        ("key_down", "ctrl"),
    )


def test_unknown_keys_and_terminate_status_are_rejected() -> None:
    with pytest.raises(CanonicalError):
        canonical_normalized([{"action": "key", "keys": ["not-a-key"]}])
    with pytest.raises(CanonicalError):
        canonical_normalized([{"action": "terminate", "status": "done"}])
    with pytest.raises(CanonicalError):
        canonical_raw("0 0 0 ; +MadeUp -MadeUp", "key")
