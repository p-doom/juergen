"""Conversation rendering shared by CUA-Gym training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from grammars.ordered_events_v3_relative_1000_grid_v1.codec import CODEC

MAX_COMPLETED_TURNS = 4
PREVIOUS_ACTIONS_MAX_CHARS = 160
OBSERVATION_CONTRACT = "osworld_cursor_jpeg_q92_420_1920x1080_v1"
OBSERVATION_SIZE = (1920, 1080)
JPEG_QUALITY = 92
JPEG_SUBSAMPLING = 2
SYSTEM_PROMPT = CODEC.describe()


def _text(value: str) -> dict[str, str]:
    return {"type": "text", "text": value}


def previous_actions(evicted: Sequence[tuple[int, str]]) -> str:
    if not evicted:
        return "None"
    value = "\n".join(
        f"Step {step}: {action.replace(chr(10), ' | ')}" for step, action in evicted
    )
    if len(value) <= PREVIOUS_ACTIONS_MAX_CHARS:
        return value
    marker = "…[earlier actions omitted]\n"
    return marker + value[-(PREVIOUS_ACTIONS_MAX_CHARS - len(marker)) :]


def instruction_text(instruction: str, evicted: Sequence[tuple[int, str]]) -> str:
    return (
        "Please generate the next move according to the UI screenshot, instruction "
        "and previous actions.\n\n"
        f"Instruction: {instruction}\n\nPrevious actions:\n{previous_actions(evicted)}"
    )


def render_history(
    *,
    instruction: str,
    steps: Sequence[Mapping[str, Any]],
    target_index: int,
) -> list[dict[str, Any]]:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    if not 0 <= target_index < len(steps):
        raise ValueError("target_index must identify one history step")
    window_start = max(0, target_index - MAX_COMPLETED_TURNS)
    evicted: list[tuple[int, str]] = []
    for step in steps[:window_start]:
        step_id = step.get("step")
        action = step.get("action_text")
        if type(step_id) is not int or not isinstance(action, str) or not action:
            raise TypeError(
                "completed history steps require integer step and non-empty action_text"
            )
        evicted.append((step_id, action))

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [_text(SYSTEM_PROMPT)]}
    ]
    for index in range(window_start, target_index + 1):
        step = steps[index]
        image = step.get("image")
        if not isinstance(image, Mapping) or not image:
            raise TypeError("history step image must be a content part")
        content = [dict(image)]
        if index == window_start:
            content.append(_text(instruction_text(instruction, evicted)))
        messages.append({"role": "user", "content": content})
        if index < target_index:
            assistant = step.get("assistant")
            if not isinstance(assistant, str) or not assistant:
                raise TypeError(
                    "completed history steps require non-empty assistant text"
                )
            messages.append({"role": "assistant", "content": [_text(assistant)]})
    return messages
