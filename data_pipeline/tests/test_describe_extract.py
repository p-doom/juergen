from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.annotation.lib.driver import run_driver
from pipeline.annotation.lib.labeler import Labeler, LabelerConfig
from pipeline.annotation.lib.prompts import PromptPack
from pipeline.annotation.methods.describe_extract.annotator import clean_goals


def _goal() -> dict:
    return {
        "instruction": "send the report",
        "anchor": "Send",
        "grounding": "The user submits the report.",
        "start_frame": 2,
        "end_frame": 7,
    }


def test_describe_extract_goal_contract_is_strict_and_clamped():
    assert clean_goals({"goals": [_goal()]}, 3, 6, 5) == [
        {
            "instruction": "send the report",
            "anchor": "Send",
            "grounding": "The user submits the report.",
            "start_frame": 3,
            "end_frame": 5,
        }
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"anchor": ""}, "anchor"),
        ({"grounding": ""}, "grounding"),
        ({"start_frame": "bad"}, "frame bounds"),
        ({"start_frame": 2.0}, "frame bounds"),
        ({"start_frame": True}, "frame bounds"),
        ({"start_frame": 8, "end_frame": 7}, "precedes"),
    ],
)
def test_describe_extract_rejects_malformed_goal_fields(change, message):
    goal = _goal() | change
    with pytest.raises(ValueError, match=message):
        clean_goals({"goals": [goal]}, 0, 10, 10)


def test_prompt_render_requires_every_field(tmp_path: Path):
    path = tmp_path / "prompts.yaml"
    path.write_text(
        "system: system\n"
        "describe_prose: 'frames=${count}'\n"
        "extract_system: extract-system\n"
        "extract: extract\n"
    )
    prompts = PromptPack(path)
    assert prompts.render("describe_prose", count=3) == "frames=3"
    with pytest.raises(KeyError, match="count"):
        prompts.render("describe_prose")


def test_annotation_driver_propagates_worker_failure(tmp_path: Path):
    def fail(item):
        raise RuntimeError(f"broken {item}")

    with pytest.raises(RuntimeError, match="broken"):
        run_driver(
            ["first", "second"],
            item_id=str,
            est_tokens=lambda _: 1,
            run_item=fail,
            progress_path=tmp_path / "progress.jsonl",
            target_tpm=1000,
            max_workers=2,
        )


def _labeler(tmp_path: Path, **response_values) -> Labeler:
    values = {
        "model": "test-model",
        "content": "result",
        "finish_reason": "stop",
        "usage": {"total_tokens": 7},
        **response_values,
    }
    message = SimpleNamespace(content=values["content"], reasoning_content="reason")
    choice = SimpleNamespace(message=message, finish_reason=values["finish_reason"])
    usage = values["usage"]
    response = SimpleNamespace(
        model=values["model"],
        choices=[choice],
        usage=(SimpleNamespace(model_dump=lambda: usage) if usage is not None else None),
    )
    instance = Labeler.__new__(Labeler)
    instance.config = LabelerConfig(
        model="test-model",
        base_url="https://labeler.example/v1",
        api_key="secret",
        transient_retries=0,
    )
    instance._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )
    return instance


def _call(labeler: Labeler, cache: Path):
    return labeler.call_full(
        "system",
        "user",
        images=["data:image/jpeg;base64,AA=="],
        image_labels=["frame 0"],
        cache_path=cache,
    )


def test_labeler_attests_provider_model_finish_usage_and_cache(tmp_path: Path):
    cache = tmp_path / "call.txt"
    result = _call(_labeler(tmp_path), cache)
    assert result.model == "test-model"
    assert result.finish_reason == "stop"
    assert result.usage["total_tokens"] == 7
    payload = cache.read_text()
    assert '"response_sha256"' in payload

    changed = _labeler(tmp_path)
    changed.config = LabelerConfig(
        model="test-model",
        base_url="https://other.example/v1",
        api_key="secret",
        transient_retries=0,
    )
    with pytest.raises(ValueError, match="base_url mismatch"):
        _call(changed, cache)


@pytest.mark.parametrize("field", ["content", "reasoning"])
def test_labeler_cache_seals_response_content(tmp_path: Path, field: str):
    cache = tmp_path / "call.json"
    labeler = _labeler(tmp_path)
    _call(labeler, cache)
    payload = json.loads(cache.read_text())
    payload["response"][field] += "tampered"
    cache.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="response digest mismatch"):
        _call(labeler, cache)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"model": "other-model"}, "model mismatch"),
        ({"finish_reason": "length"}, "finish_reason"),
        ({"finish_reason": "tool_calls"}, "finish_reason"),
        ({"usage": None}, "structured usage"),
        ({"usage": {"total_tokens": 0}}, "total_tokens"),
    ],
)
def test_labeler_rejects_unaccounted_provider_responses(tmp_path: Path, change: dict, message: str):
    with pytest.raises((TypeError, ValueError), match=message):
        _call(_labeler(tmp_path, **change), tmp_path / "call.txt")
