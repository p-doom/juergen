from __future__ import annotations

import json

import pytest
from PIL import Image

from stage5_rft.relative_mouse_records import (
    _task_split,
    build_pure_relative_mouse_records,
)
from stage5_rft.util import ContractError, read_jsonl


def _tool(dx: int, dy: int) -> str:
    return (
        '<tool_call>\n{"name": "computer_use", "arguments": '
        f'{{"action": "move_rel", "coordinate": [{dx}, {dy}]}}}}\n</tool_call>'
    )


def _row(task_id: int, background: str, *, assistant: str | None = None) -> dict:
    raw = assistant or _tool(50, 0)
    return {
        "env": "movebox",
        "task": {
            "kind": "train",
            "idx": task_id,
            "background_path": background,
            "box": [190, 90, 260, 170],
            "cursor_start": [100, 100],
        },
        "accepted": True,
        "reward": 1.0,
        "steps": 1,
        "traj": [{"cursor": [100, 100], "delta": [50, 0], "assistant": raw}],
    }


def _opposite_split_ids() -> tuple[int, int]:
    by_split = {}
    for task_id in range(100):
        split = _task_split(task_id=str(task_id), salt="test", val_fraction=0.5)
        by_split.setdefault(split, task_id)
    return by_split["train"], by_split["val"]


def test_pure_relative_mouse_builder_replays_and_preserves_actions(tmp_path):
    backgrounds = tmp_path / "backgrounds"
    backgrounds.mkdir()
    background = backgrounds / "train.png"
    Image.new("RGB", (320, 180), "white").save(background)
    train_id, val_id = _opposite_split_ids()
    rollouts = tmp_path / "rollouts.jsonl"
    rows = [_row(train_id, str(background)), _row(val_id, str(background))]
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows))

    output = tmp_path / "dataset"
    manifest = build_pure_relative_mouse_records(
        rollout_glob=str(rollouts),
        output_dir=output,
        approved_background_root=backgrounds,
        val_fraction=0.5,
        split_salt="test",
    )
    assert manifest["method"] == "pure_rejection_sft"
    assert manifest["synthetic_actions_added"] == 0
    assert manifest["synthetic_terminate_added"] is False
    assert manifest["contains_official_heldout"] is False
    assert manifest["contains_real_vm_eval"] is False
    assert manifest["contains_crowd_cast"] is False
    train = read_jsonl(output / "_normalized/train/chat.jsonl")
    val = read_jsonl(output / "_normalized/val/chat.jsonl")
    assert {row["task_id"] for row in train}.isdisjoint({row["task_id"] for row in val})
    targets = [row["messages"][-1]["content"][0]["text"] for row in train + val]
    assert targets == [_tool(50, 0), _tool(50, 0)]
    assert all("terminate" not in target for target in targets)


def test_pure_relative_mouse_builder_fails_closed_on_action_drift(tmp_path):
    backgrounds = tmp_path / "backgrounds"
    backgrounds.mkdir()
    background = backgrounds / "train.png"
    Image.new("RGB", (320, 180), "white").save(background)
    train_id, val_id = _opposite_split_ids()
    rows = [
        _row(train_id, str(background), assistant=_tool(51, 0)),
        _row(val_id, str(background)),
    ]
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ContractError, match="malformed or unverifiable"):
        build_pure_relative_mouse_records(
            rollout_glob=str(rollouts),
            output_dir=tmp_path / "dataset",
            approved_background_root=backgrounds,
            val_fraction=0.5,
            split_salt="test",
        )


def test_pure_relative_mouse_builder_preserves_no_op_exactly(tmp_path):
    backgrounds = tmp_path / "backgrounds"
    backgrounds.mkdir()
    background = backgrounds / "train.png"
    Image.new("RGB", (320, 180), "white").save(background)
    train_id, val_id = _opposite_split_ids()
    wait = (
        '<tool_call>\n{"name": "computer_use", "arguments": '
        '{"action": "wait", "time": 1}}\n</tool_call>'
    )
    rows = []
    for task_id in (train_id, val_id):
        row = _row(task_id, str(background))
        row["steps"] = 2
        row["traj"] = [
            {"cursor": [100, 100], "delta": None, "assistant": wait},
            {"cursor": [100, 100], "delta": [50, 0], "assistant": _tool(50, 0)},
        ]
        rows.append(row)
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "dataset"
    manifest = build_pure_relative_mouse_records(
        rollout_glob=str(rollouts),
        output_dir=output,
        approved_background_root=backgrounds,
        val_fraction=0.5,
        split_salt="test",
    )
    assert manifest["no_op_actions_dropped"] == 0
    assert manifest["no_op_actions_preserved"] == 2
    records = read_jsonl(output / "_normalized/train/chat.jsonl") + read_jsonl(
        output / "_normalized/val/chat.jsonl"
    )
    assert [row["messages"][-1]["content"][0]["text"] for row in records].count(wait) == 2


def test_pure_relative_mouse_builder_preserves_multi_tool_turn_exactly(tmp_path):
    backgrounds = tmp_path / "backgrounds"
    backgrounds.mkdir()
    background = backgrounds / "train.png"
    Image.new("RGB", (320, 180), "white").save(background)
    train_id, val_id = _opposite_split_ids()
    multi = _tool(50, 0) + "\n" + (
        '<tool_call>\n{"name": "computer_use", "arguments": '
        '{"action": "left_click"}}\n</tool_call>'
    )
    rollouts = tmp_path / "rollouts.jsonl"
    rows = [
        _row(train_id, str(background), assistant=multi),
        _row(val_id, str(background), assistant=multi),
    ]
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "dataset"
    build_pure_relative_mouse_records(
        rollout_glob=str(rollouts),
        output_dir=output,
        approved_background_root=backgrounds,
        val_fraction=0.5,
        split_salt="test",
    )
    records = read_jsonl(output / "_normalized/train/chat.jsonl") + read_jsonl(
        output / "_normalized/val/chat.jsonl"
    )
    assert [row["messages"][-1]["content"][0]["text"] for row in records] == [multi, multi]
