from __future__ import annotations

import json
from pathlib import Path

from experiments.teacher_sft.contracts import (
    file_sha256,
    read_json,
    write_json,
    write_jsonl,
)
from experiments.teacher_sft.conversion import convert_accepted
from experiments.teacher_sft.rejection import reject_rollouts
from experiments.teacher_sft.sft import build_sft
from experiments.teacher_sft.task_sources import build_task_manifest, load_task_rows


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def observation(path: Path, cursor: tuple[int, int]) -> dict:
    return {
        "image_path": str(path.resolve()),
        "image_sha256": file_sha256(path),
        "cursor": list(cursor),
        "screen_size": [100, 100],
    }


def test_cpu_supply_chain_smoke(tmp_path: Path) -> None:
    task_root = tmp_path / "source" / "app"
    dump(
        task_root / "task-1.json", {"instruction": "Complete the train-only GUI task."}
    )
    dump(tmp_path / "train.json", {"app": ["task-1"]})
    dump(
        tmp_path / "source_spec.json",
        {
            "schema_version": 1,
            "validation_fraction": 0,
            "sources": [
                {
                    "kind": "osworld",
                    "source_split": "train",
                    "source_revision": "fixture-rev",
                    "task_index": str(tmp_path / "train.json"),
                    "task_root": str(tmp_path / "source"),
                }
            ],
        },
    )
    deny = {
        "schema_version": 1,
        "task_keys": [],
        "source_task_ids": [],
        "instruction_sha256": [],
        "asset_sha256": [],
    }
    dump(tmp_path / "deny.json", deny)
    build_task_manifest(
        tmp_path / "source_spec.json", tmp_path / "deny.json", tmp_path / "tasks"
    )
    task = load_task_rows(tmp_path / "tasks")[0]

    images = []
    for index in range(6):
        path = tmp_path / "images" / f"{index}.png"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"cpu-smoke-png" + bytes([index]))
        images.append(path)
    specs = [
        (
            (10, 10),
            (20, 20),
            {
                "action": "left_click",
                "coordinate": [20, 20],
                "coordinate_space": "absolute_px",
            },
            [20, 20],
        ),
        ((20, 20), (20, 20), {"action": "type", "text": "hello world"}, None),
        (
            (20, 20),
            (30, 35),
            {
                "action": "left_click_drag",
                "coordinate": [30, 35],
                "coordinate_space": "absolute_px",
            },
            [30, 35],
        ),
        ((30, 35), (30, 35), {"action": "scroll", "pixels": -4}, None),
        ((30, 35), (30, 35), {"action": "terminate", "status": "success"}, None),
    ]
    steps = []
    for index, (before, after, action, target) in enumerate(specs):
        steps.append(
            {
                "step_index": index,
                "observation_before": observation(images[index], before),
                "teacher_response": json.dumps(action),
                "actions": [action],
                "execution_traces": [
                    {
                        "cursor_before": list(before),
                        "cursor_after": list(after),
                        "resolved_target_px": target,
                    }
                ],
                "observation_after": observation(images[index + 1], after),
            }
        )
    rollout = {
        "schema_version": 1,
        "rollout_id": "fixture-001",
        "task": task,
        "teacher": {
            "model_id": "scripted-cpu-teacher",
            "model_revision": "fixture",
            "action_space": "native_absolute",
        },
        "steps": steps,
        "result": {
            "reward": 1.0,
            "success": True,
            "environment_success": True,
            "termination": "success",
            "parse_errors": 0,
            "error": None,
        },
    }
    rollout_path = tmp_path / "rollouts" / "fixture-001" / "rollout.json"
    write_json(rollout_path, rollout)
    index_path = tmp_path / "rollouts" / "index.jsonl"
    write_jsonl(
        index_path,
        [
            {
                "rollout_id": "fixture-001",
                "task_key": task["task_key"],
                "path": str(rollout_path.resolve()),
                "sha256": file_sha256(rollout_path),
            }
        ],
    )
    write_json(
        tmp_path / "rollouts" / "manifest.json",
        {
            "artifact_type": "fixture",
            "construction_scope": "train_only",
            "task_manifest_sha256": file_sha256(tmp_path / "tasks" / "manifest.json"),
            "index_sha256": file_sha256(index_path),
        },
    )

    reject_rollouts(tmp_path / "tasks", tmp_path / "rollouts", tmp_path / "rejection")
    convert_accepted(tmp_path / "rejection", tmp_path / "converted")
    manifest = build_sft(
        tmp_path / "converted", tmp_path / "deny.json", tmp_path / "sft"
    )

    converted = json.loads((tmp_path / "converted" / "converted.jsonl").read_text())
    assert converted["steps"][2]["compact_action"].splitlines() == [
        "0 0 0 ; +LMB",
        "10 15 0",
        "0 0 0 ; -LMB",
    ]
    assert manifest["construction_scope"] == "train_only"
    assert manifest["split_files"]["train"]["n_rows"] == 1
    chat = json.loads(
        (tmp_path / "sft" / "_normalized" / "train" / "chat.jsonl").read_text()
    )
    visible = json.dumps(chat["messages"])
    assert "reward" not in visible and "source_revision" not in visible
    assert read_json(tmp_path / "converted" / "manifest.json")[
        "symbolic_replay_verified"
    ]
