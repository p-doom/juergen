from __future__ import annotations

from pathlib import Path

from harness_render import (
    HarnessRenderer,
    HarnessRenderSpec,
    TrainingTurn,
    render_sft_records,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "harness_render_specs" / "stream-cuagym-qwen35-render-q92.json"
PROMPT_PATH = (
    ROOT
    / "data_pipeline"
    / "realigned_pipeline"
    / "system_prompts"
    / "cua_v3_cuagym.txt"
)
SPEC_SHA256 = "4d5479adbf5cf85db2112cb841a70baeb6b4b1e61568fc7aa38bc29340d58818"
ACTION_CONTRACT = "ordered_events_v3_relative_1000_grid_v1"
OBSERVATION_CONTRACT = "jpeg_q92_rgb_420_1920x1080_v1"


def test_candidate_spec_pins_the_traced_render_contract():
    prompt = PROMPT_PATH.read_text().strip()
    spec = HarnessRenderSpec.load(SPEC_PATH, expected_sha256=SPEC_SHA256)
    renderer = HarnessRenderer(
        spec,
        spec_sha256=SPEC_SHA256,
        system_prompt=prompt,
        action_contract=ACTION_CONTRACT,
        observation_contract=OBSERVATION_CONTRACT,
    )
    turns = [
        TrainingTurn(
            image=f"frame-{index}",
            assistant=f"<think>reason {index}</think>\n\nmove({index},0)",
        )
        for index in range(6)
    ]
    records = render_sft_records(renderer, instruction="Do the task", turns=turns)

    assert spec.max_completed_turns == 4
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
