import pytest

from pipeline.stage_04_build_conversations import build_messages


def test_build_messages_marks_the_context_selected_by_the_conversation_schema():
    turns = [("frame-1.png", "NO_OP"), ("frame-2.png", "move(1,0)")]

    messages, carry, split_unit_ends = build_messages(
        turns, instruction=None, system_prompt="grammar"
    )
    assert carry == [0]
    assert split_unit_ends == [1, 3, 5]
    assert messages[0]["role"] == "system"

    messages, carry, split_unit_ends = build_messages(
        turns, instruction="open settings", system_prompt="grammar"
    )
    assert carry == [0, 1]
    assert split_unit_ends == [1, 3, 5]
    assert messages[1]["content"][0] == {"type": "text", "text": "open settings"}


def test_build_messages_rejects_a_row_without_observation_action_units():
    with pytest.raises(ValueError, match="invalid Crowd-Cast conversation roles"):
        build_messages([], instruction=None, system_prompt="grammar")
