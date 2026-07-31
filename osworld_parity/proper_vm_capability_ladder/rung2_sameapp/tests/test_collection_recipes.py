from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.collector import collect


ROOT = Path(__file__).resolve().parents[3]
RECIPES = ROOT / "labctl" / "recipes"


@pytest.mark.parametrize(
    "recipe_name",
    (
        "rung2_sameapp_replay_build_cpu.toml",
        "rung2_sameapp_teacher_collect_cpu_kvm.toml",
    ),
)
def test_cpu_recipes_use_hardened_vm_replay_with_declared_inputs(
    recipe_name: str,
) -> None:
    path = RECIPES / recipe_name
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    assert raw["resources"]["gpus"] == 0
    command = " ".join(raw["command"])
    assert "rung2_sameapp.replay" in command
    assert "--mode=vm" in command
    assert "--split=development" in command
    assert "--task-setup-validation" in command
    assert (
        "--expected-provider-sha256 "
        "76a8f44fab16c6dd38a4378a270e38758ba8d31885f244baedb95d8178f588d7"
        in command
    )
    assert "task_setup_validation.json" in command
    assert command.index('"$@"') < command.index("--mode=vm")
    assert "task_setup_validation" in raw["inputs"]
    assert raw["inputs"]["task_setup_validation"]["type"] == "artifact"
    assert set(raw["inputs"]) == {"task_setup_validation", "vm", "qemu", "provider"}
    assert "CUDA_VISIBLE_DEVICES" in command
    assert "rung2_sameapp.collector" not in command
    assert "--mode=build" not in command
    assert "--split=train" not in command
    assert "sealed_eval" not in command
    assert raw["outputs"]["result"]["marker"] == "replay.json"


def test_legacy_collection_cannot_bypass_hardened_replay(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="legacy rung2_sameapp collector is disabled"):
        collect(mode="build", split="train", output=tmp_path / "a")
