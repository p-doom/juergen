from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.collector import collect


ROOT = Path(__file__).resolve().parents[3]
RECIPES = ROOT / "labctl" / "recipes"


def test_cpu_collection_recipe_forbids_gpu_and_collects_train_only() -> None:
    path = RECIPES / "rung2_sameapp_teacher_collect_cpu_kvm.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    assert raw["resources"]["gpus"] == 0
    command = " ".join(raw["command"])
    assert "--split=train" in command
    assert "CUDA_VISIBLE_DEVICES" in command
    assert "sealed_eval" not in command


def test_build_collection_is_deterministic_and_train_only(tmp_path: Path) -> None:
    first = collect(mode="build", split="train", output=tmp_path / "a")
    second = collect(mode="build", split="train", output=tmp_path / "b")
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["gpu_used"] is False
    assert first["training_ready"] is False
    rows = [json.loads(line) for line in (tmp_path / "a" / "teacher_trajectories.jsonl").read_text().splitlines()]
    assert rows and all(row["split"] == "train" for row in rows)
    assert all(row["sealed_eval_material"] is False for row in rows)
    assert all(row["training_ready"] is False for row in rows)
    with pytest.raises(ValueError, match="train-only"):
        collect(mode="build", split="development", output=tmp_path / "dev")
