from pipeline.stage_04_build_conversations import build_messages


def test_build_messages_marks_the_context_selected_by_the_conversation_schema():
    turns = [("frame.png", "NO_OP")]

    messages, carry = build_messages(turns, instruction=None, system_prompt="grammar")
    assert carry == [0]
    assert messages[0]["role"] == "system"

    messages, carry = build_messages(
        turns, instruction="open settings", system_prompt="grammar"
    )
    assert carry == [0, 1]
    assert messages[1]["content"][0] == {"type": "text", "text": "open settings"}
