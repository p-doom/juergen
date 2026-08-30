from __future__ import annotations

import copy
import hashlib
import json

import pytest

from harness_render import (
    HarnessRenderer,
    HarnessRenderSpec,
    TrainingTurn,
    render_sft_records,
)

SYSTEM_PROMPT = "Synthetic harness prompt"
INSTRUCTION_TEMPLATE = (
    "Instruction: {instruction}\n\nPrevious actions:\n{previous_actions}"
)
ACTION_CONTRACT = "synthetic_action_v1"
OBSERVATION_CONTRACT = "synthetic_image_v1"


def _spec(tmp_path):
    spec = HarnessRenderSpec(
        schema_version=1,
        spec_id="synthetic-render-v1",
        max_completed_turns=4,
        max_previous_action_chars=160,
        system_prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        action_contract=ACTION_CONTRACT,
        observation_contract=OBSERVATION_CONTRACT,
        instruction_template=INSTRUCTION_TEMPLATE,
    )
    path = tmp_path / "render.json"
    path.write_bytes(spec.canonical_bytes())
    return HarnessRenderSpec.load(path, expected_sha256=spec.sha256), spec.sha256


def _renderer(spec, digest, **overrides):
    bindings = {
        "spec_sha256": digest,
        "system_prompt": SYSTEM_PROMPT,
        "action_contract": ACTION_CONTRACT,
        "observation_contract": OBSERVATION_CONTRACT,
    }
    bindings.update(overrides)
    return HarnessRenderer(spec, **bindings)


def _turns():
    return [
        TrainingTurn(
            image=f"frame-{index}",
            assistant=(
                f"<think>reason {index}</think>\n\n"
                f"Action: synthetic {index}\nmove({index},0)"
            ),
            action=f"move({index},0)",
        )
        for index in range(7)
    ]


def _prompt_bytes(messages):
    normalized = copy.deepcopy(messages)
    for message in normalized:
        message.pop("loss", None)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _image_count(messages):
    return sum(
        part.get("type") == "image"
        for message in messages
        for part in message.get("content", [])
        if isinstance(part, dict)
    )


def test_online_and_offline_prompts_are_byte_equivalent(tmp_path):
    spec, digest = _spec(tmp_path)
    turns = _turns()
    online_renderer = _renderer(spec, digest)
    online_renderer.start(turns[0].image)
    online = []
    for index, turn in enumerate(turns):
        online.append(online_renderer.render_prompt(instruction="Do the task"))
        if index + 1 < len(turns):
            online_renderer.complete(
                assistant=turn.assistant,
                action=turn.action,
                next_image=turns[index + 1].image,
            )

    offline = render_sft_records(
        _renderer(spec, digest),
        instruction="Do the task",
        turns=turns,
    )
    assert [_prompt_bytes(messages) for messages in online] == [
        _prompt_bytes(messages[:-1]) for messages in offline
    ]
    assert [_image_count(messages) for messages in online] == [1, 2, 3, 4, 5, 5, 5]

    prompt_text = online[5][1]["content"][1]["text"]
    assert "Step 1: move(0,0)" in prompt_text
    assert "Step 2:" not in prompt_text
    assert all(
        "<think>" not in part["text"]
        for message in online[5]
        if message["role"] == "assistant"
        for part in message["content"]
    )
    assert offline[5][-1]["content"][0]["text"] == turns[5].assistant


def test_spec_and_runtime_mismatches_fail_before_rendering(tmp_path):
    spec, digest = _spec(tmp_path)
    path = tmp_path / "render.json"
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"synthetic-render-v1", b"synthetic-render-v2"))
    with pytest.raises(ValueError, match="render spec digest mismatch"):
        HarnessRenderSpec.load(path, expected_sha256=digest)

    with pytest.raises(ValueError, match="system prompt digest mismatch"):
        _renderer(spec, digest, system_prompt="wrong")
    with pytest.raises(ValueError, match="action contract mismatch"):
        _renderer(spec, digest, action_contract="wrong")
    with pytest.raises(ValueError, match="observation contract mismatch"):
        _renderer(spec, digest, observation_contract="wrong")


def test_spec_shape_and_history_invariants_fail_fast(tmp_path):
    spec, digest = _spec(tmp_path)
    data = json.loads(spec.canonical_bytes())
    data["compatibility_mode"] = True
    raw = (json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n").encode()
    with pytest.raises(ValueError, match="unexpected=.*compatibility_mode"):
        HarnessRenderSpec.from_bytes(
            raw,
            expected_sha256=hashlib.sha256(raw).hexdigest(),
        )

    renderer = _renderer(spec, digest)
    with pytest.raises(RuntimeError, match="before start"):
        renderer.render_prompt(instruction="Do the task")
    renderer.start("frame-0")
    with pytest.raises(ValueError, match="unterminated <think>"):
        renderer.complete(
            assistant="<think>unterminated",
            action="NO_OP",
            next_image="frame-1",
        )
    with pytest.raises(ValueError, match="does not match"):
        renderer.complete(
            assistant="<think>done</think>\nNO_OP",
            action="move(1,0)",
            next_image="frame-1",
        )
