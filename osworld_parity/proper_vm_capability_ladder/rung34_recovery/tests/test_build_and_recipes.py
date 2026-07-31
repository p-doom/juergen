import hashlib
import json
import tomllib
from pathlib import Path

import pytest

from osworld_parity.proper_vm_capability_ladder.rung34_recovery.build import (
    build_contract_artifacts,
)
from osworld_parity.proper_vm_capability_ladder.rung34_recovery.gates import (
    EarlierGateError,
    REQUIRED_GATES,
    require_earlier_gate_evidence,
)
from osworld_parity.proper_vm_capability_ladder.rung34_recovery.validate_dataset import (
    validate_dataset,
)


RECIPE_DIR = Path(__file__).parents[3] / "labctl" / "recipes"
RECIPES = (
    "rung34_recovery_contract_build_cpu.toml",
    "rung34_recovery_vm_replay_cpu_kvm.toml",
    "rung34_recovery_rollout_validate_cpu.toml",
)


def test_build_artifacts_are_cpu_only_and_have_no_hidden_exports(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result = build_contract_artifacts(tmp_path)
    assert result["gpu_count"] == 0
    assert result["models_run"] == 0
    assert result["sealed_evaluation_opened"] == 0
    assert result["trainer_only_values_exported"] is False
    payload = "\n".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.glob("*.jsonl")
    )
    assert '"hidden_state"' not in payload
    assert '"reward"' not in payload
    assert '"oracle"' not in payload


def test_rollout_dataset_validator_is_schema_only(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    source = tmp_path / "source"
    build_contract_artifacts(source)
    result = validate_dataset(
        source / "on_policy_schema_contract.jsonl", tmp_path / "validated"
    )
    assert result["record_count"] == 2
    assert result["models_run"] == 0


@pytest.mark.parametrize("recipe_name", RECIPES)
def test_labctl_recipes_allocate_no_gpu_or_model_runtime(recipe_name):
    recipe = tomllib.loads((RECIPE_DIR / recipe_name).read_text(encoding="utf-8"))
    assert recipe["resources"]["gpus"] == 0
    assert recipe["env"]["CUDA_VISIBLE_DEVICES"] == ""
    command = " ".join(recipe["command"]).lower()
    assert "torchrun" not in command
    assert "sglang" not in command
    assert "vllm" not in command
    assert "transformers" not in command


def test_vm_recipe_is_explicitly_blocked_on_earlier_gate_artifact():
    recipe = tomllib.loads(
        (RECIPE_DIR / "rung34_recovery_vm_replay_cpu_kvm.toml").read_text(
            encoding="utf-8"
        )
    )
    assert "earlier_gates" in recipe["inputs"]
    assert recipe["args"]["gate_evidence"].endswith("/result.json")


def test_gate_evidence_requires_all_prior_passes_and_commitments(tmp_path):
    path = tmp_path / "gates.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_commit": "48a54e8585eb9d6abff31e2ba6ea857c946a7d3d",
                "gates": {
                    gate: {
                        "status": "passed",
                        "artifact_sha256": hashlib.sha256(gate.encode()).hexdigest(),
                    }
                    for gate in REQUIRED_GATES
                },
            }
        ),
        encoding="utf-8",
    )
    assert require_earlier_gate_evidence(path)["schema_version"] == 1
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["gates"]["roadmap_3_2"]["status"] = "pending"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(EarlierGateError, match="has not passed"):
        require_earlier_gate_evidence(path)
