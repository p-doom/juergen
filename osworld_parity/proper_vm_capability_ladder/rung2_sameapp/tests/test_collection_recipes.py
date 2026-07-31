from __future__ import annotations

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


def test_legacy_collection_cannot_bypass_hardened_replay(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="legacy rung2_sameapp collector is disabled"):
        collect(mode="build", split="train", output=tmp_path / "a")
