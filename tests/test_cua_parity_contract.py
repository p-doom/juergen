from cua_parity_contract import (
    MAX_COMPLETED_TURNS,
    PREVIOUS_ACTIONS_MAX_CHARS,
    SYSTEM_PROMPT,
    previous_actions,
    render_history,
)
from grammars.ordered_events_v3_relative_1000_grid_v1.codec import CODEC


def _steps(count: int) -> list[dict[str, object]]:
    return [
        {
            "step": index,
            "image": {"type": "image", "image": f"image-{index}"},
            "assistant": f"assistant-{index}",
            "action_text": f"action-{index}",
        }
        for index in range(count)
    ]


def test_renderer_uses_exact_prompt_and_four_completed_turns():
    messages = render_history(instruction="Do the task", steps=_steps(7), target_index=6)

    assert messages[0] == {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
    }
    assert SYSTEM_PROMPT == CODEC.describe()
    assert len([message for message in messages if message["role"] == "assistant"]) == (
        MAX_COMPLETED_TURNS
    )
    assert messages[1]["content"][0]["image"] == "image-2"
    instruction = messages[1]["content"][1]["text"]
    assert "Instruction: Do the task" in instruction
    assert "Step 0: action-0" in instruction
    assert "Step 1: action-1" in instruction
    assert len(previous_actions([(index, "x" * 80) for index in range(4)])) <= (
        PREVIOUS_ACTIONS_MAX_CHARS
    )


def test_renderer_rejects_incomplete_completed_turn():
    steps = _steps(2)
    steps[0]["assistant"] = ""

    try:
        render_history(instruction="Do the task", steps=steps, target_index=1)
    except TypeError as exc:
        assert "non-empty assistant" in str(exc)
    else:
        raise AssertionError("empty assistant text was accepted")
