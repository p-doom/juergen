from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.annotation.lib.driver import run_driver
from pipeline.annotation.lib.prompts import PromptPack
from pipeline.annotation.methods.describe_extract.annotator import clean_goals


def _goal() -> dict:
    return {
        "instruction": "send the report",
        "instruction_variants": ["send it", "please send the report"],
        "anchor": "Send",
        "grounding": "The user submits the report.",
        "start_frame": 2,
        "end_frame": 7,
    }


def test_describe_extract_goal_contract_is_strict_and_clamped():
    assert clean_goals({"goals": [_goal()]}, 3, 6, 5) == [
        {
            "instruction": "send the report",
            "instruction_variants": ["send it", "please send the report"],
            "anchor": "Send",
            "grounding": "The user submits the report.",
            "start_frame": 3,
            "end_frame": 5,
        }
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"instruction_variants": ["one"]}, "exactly two"),
        ({"instruction_variants": ["send the report", "other"]}, "distinct"),
        ({"anchor": ""}, "anchor"),
        ({"grounding": ""}, "grounding"),
        ({"start_frame": "bad"}, "frame bounds"),
        ({"start_frame": 8, "end_frame": 7}, "precedes"),
    ],
)
def test_describe_extract_rejects_malformed_goal_fields(change, message):
    goal = _goal() | change
    with pytest.raises(ValueError, match=message):
        clean_goals({"goals": [goal]}, 0, 10, 10)


def test_prompt_render_requires_every_field(tmp_path: Path):
    path = tmp_path / "prompts.yaml"
    path.write_text("prompt: 'frames=${count}'\n")
    prompts = PromptPack(path)
    assert prompts.render("prompt", count=3) == "frames=3"
    with pytest.raises(KeyError, match="count"):
        prompts.render("prompt")


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
