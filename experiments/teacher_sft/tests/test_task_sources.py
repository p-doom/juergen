from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.teacher_sft.contracts import ContractError
from experiments.teacher_sft.task_sources import build_task_manifest, load_task_rows


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def denylist(path: Path, **updates) -> None:
    value = {
        "schema_version": 1,
        "task_keys": [],
        "source_task_ids": [],
        "instruction_sha256": [],
        "asset_sha256": [],
    }
    value.update(updates)
    dump(path, value)


def test_osworld_and_cuagym_sources_share_one_train_manifest(tmp_path: Path) -> None:
    osw_root = tmp_path / "osworld"
    dump(
        osw_root / "writer" / "osw-1.json", {"instruction": "Write a train-only note."}
    )
    dump(tmp_path / "osworld_train.json", {"writer": ["osw-1"]})
    cua = tmp_path / "cua" / "cua-1"
    dump(cua / "task.json", {"task_instruction": "Create a train-only mock post."})
    (cua / "reward.py").write_text("print('REWARD: 1.0')\n")
    (cua / "initial_setup.py").write_text("print('setup')\n")
    (tmp_path / "cua_train.jsonl").write_text(
        json.dumps(
            {
                "id": "cua-1",
                "instruction": "Create a train-only mock post.",
                "split": "train",
                "app_type": "notion_mock",
                "platform": "web",
                "setup_files": ["initial_setup.py"],
            }
        )
        + "\n"
    )
    spec = {
        "schema_version": 1,
        "validation_fraction": 0.25,
        "split_seed": "test",
        "sources": [
            {
                "kind": "osworld",
                "source_split": "train",
                "source_revision": "osw-rev",
                "task_index": str(tmp_path / "osworld_train.json"),
                "task_root": str(osw_root),
            },
            {
                "kind": "cua_gym",
                "source_split": "train",
                "source_revision": "cua-rev",
                "task_index": str(tmp_path / "cua_train.jsonl"),
                "bundle_root": str(tmp_path / "cua"),
            },
        ],
    }
    dump(tmp_path / "sources.json", spec)
    denylist(tmp_path / "deny.json")
    manifest = build_task_manifest(
        tmp_path / "sources.json", tmp_path / "deny.json", tmp_path / "output"
    )
    rows = load_task_rows(tmp_path / "output")
    assert manifest["construction_scope"] == "train_only"
    assert {row["source"] for row in rows} == {"osworld", "cua_gym"}
    assert all(row["source_split"] == "train" for row in rows)


def test_eval_named_index_and_denylisted_task_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    dump(root / "app" / "id.json", {"instruction": "Do a training action."})
    dump(tmp_path / "test_all.json", {"app": ["id"]})
    spec = {
        "schema_version": 1,
        "sources": [
            {
                "kind": "osworld",
                "source_split": "train",
                "source_revision": "rev",
                "task_index": str(tmp_path / "test_all.json"),
                "task_root": str(root),
            }
        ],
    }
    dump(tmp_path / "sources.json", spec)
    denylist(tmp_path / "deny.json", task_keys=["osworld:id"])
    with pytest.raises(ContractError, match="heldout/eval-scoped"):
        build_task_manifest(
            tmp_path / "sources.json", tmp_path / "deny.json", tmp_path / "out"
        )
