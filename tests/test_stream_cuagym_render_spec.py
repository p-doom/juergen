from __future__ import annotations

from desktop.vm.observation import OBSERVATION_CONTRACT as DESKTOP_OBSERVATION_CONTRACT
from desktop.vm.observation import OBSERVATION_SIZE as DESKTOP_OBSERVATION_SIZE

from harness_render import (
    TrainingTurn,
    render_sft_records,
)
from stream_cuagym_qwen35 import SPEC_SHA256, metadata, renderer


def test_candidate_spec_pins_the_traced_render_contract():
    bound = renderer()
    turns = [
        TrainingTurn(
            image=f"frame-{index}",
            assistant=f"<think>reason {index}</think>\n\nmove({index},0)",
            action=f"move({index},0)",
        )
        for index in range(6)
    ]
    records = render_sft_records(bound, instruction="Do the task", turns=turns)

    assert bound.spec.max_completed_turns == 4
    assert bound.spec.sha256 == SPEC_SHA256
    assert metadata()["jpeg_quality"] == 92
    assert (metadata()["width"], metadata()["height"]) == (1920, 1080)
    assert bound.spec.observation_contract == DESKTOP_OBSERVATION_CONTRACT
    assert (metadata()["width"], metadata()["height"]) == DESKTOP_OBSERVATION_SIZE
    assert [_image_count(record) for record in records] == [1, 2, 3, 4, 5, 5]
    assert "Step 1: move(0,0)" in records[5][1]["content"][1]["text"]
    assert all(
        "<think>" not in part["text"]
        for message in records[5][:-1]
        if message["role"] == "assistant"
        for part in message["content"]
    )
    assert records[5][-1]["content"][0]["text"] == turns[5].assistant


def _image_count(messages):
    return sum(
        part.get("type") == "image"
        for message in messages
        for part in message.get("content", [])
        if isinstance(part, dict)
    )
