from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RECIPES = ROOT / "labctl" / "recipes"
PIPELINE = ROOT / "labctl" / "pipelines" / "executor_certification.toml"
DIAGNOSTIC_PIPELINE = (
    ROOT / "labctl" / "pipelines" / "executor_instrumented_diagnostic.toml"
)
DIAGNOSTIC_RECIPE = (
    ROOT / "labctl" / "recipes" / "executor_diag_click_instrumented_cpu_kvm.toml"
)
NAMES = (
    "executor_cert_build_cpu.toml",
    "executor_cert_click_preflight_cpu_kvm.toml",
    "executor_cert_failure_probe_cpu_kvm.toml",
    "executor_cert_click_full_0_cpu_kvm.toml",
    "executor_cert_click_full_1_cpu_kvm.toml",
    "executor_cert_click_full_2_cpu_kvm.toml",
    "executor_cert_click_full_3_cpu_kvm.toml",
    "executor_cert_rung1a_dev_cpu_kvm.toml",
    "executor_cert_rung1b_dev_cpu_kvm.toml",
    "executor_cert_sameapp_dev_cpu_kvm.toml",
    "executor_cert_aggregate_cpu.toml",
)


def test_certification_recipes_are_cpu_only_clean_integration_runs() -> None:
    for name in NAMES:
        path = RECIPES / name
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        text = path.read_text(encoding="utf-8")
        assert raw["repo"] == "juergen_gui_executor_integration"
        assert raw["resources"]["gpus"] == 0
        assert raw["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert "--no-requeue" in raw["resources"]["sbatch_extra"]
        assert "--ntasks=1" in raw["resources"]["sbatch_extra"]
        assert "--ntasks-per-node=1" in raw["resources"]["sbatch_extra"]
        assert "heldout" not in text.lower()
        assert "sealed_eval" not in text.lower()
        assert "teacher_collect" not in text.lower()
        assert "--array" not in text
        assert "{inputs." not in text
        assert "{run.id}" in raw["outputs"]["result"]["alias"]


def test_every_vm_recipe_requires_slurm_isolation_and_build_dependency() -> None:
    vm_names = [
        name
        for name in NAMES
        if name not in {"executor_cert_build_cpu.toml", "executor_cert_aggregate_cpu.toml"}
    ]
    for name in vm_names:
        path = RECIPES / name
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        text = path.read_text(encoding="utf-8")
        assert raw["inputs"]["build"] == {
            "type": "stage",
            "stage": "build",
            "role": "result",
        }
        assert "SLURM_NTASKS" in text
        assert "vm_metadata.json" in text
        assert "ARTIFACT_INDEX.json" in text
        assert ".inputs[]|select(.role==$role)|.resolved_path" in text
        assert ".inputs." not in text
        assert "qemu-system-x86_64-wrapped" in text
        assert "qemu-system-x86_64" in text
        assert "ld-linux-x86-64.so.2" in text
        assert "Ubuntu.qcow2" in text


def test_pipeline_is_the_exact_fail_closed_dag() -> None:
    pipeline = tomllib.loads(PIPELINE.read_text(encoding="utf-8"))
    stages = pipeline["stages"]
    assert set(stages) == {
        "build",
        "preflight",
        "failure_probe",
        "click_full_0",
        "click_full_1",
        "click_full_2",
        "click_full_3",
        "rung1a",
        "rung1b",
        "sameapp",
        "aggregate",
    }
    preflight = tomllib.loads((RECIPES / stages["preflight"]["recipe"].split("/")[-1]).read_text())
    failure = tomllib.loads((RECIPES / stages["failure_probe"]["recipe"].split("/")[-1]).read_text())
    assert preflight["inputs"]["build"]["stage"] == "build"
    assert failure["inputs"]["preflight"]["stage"] == "preflight"
    for stage in ("click_full_0", "click_full_1", "click_full_2", "click_full_3", "rung1a", "rung1b", "sameapp"):
        raw = tomllib.loads((RECIPES / stages[stage]["recipe"].split("/")[-1]).read_text())
        assert raw["inputs"]["failure_probe"]["stage"] == "failure_probe"
    aggregate = tomllib.loads((RECIPES / "executor_cert_aggregate_cpu.toml").read_text())
    assert set(aggregate["inputs"]) == {
        "build",
        "preflight",
        "failure_probe",
        "click_full_0",
        "click_full_1",
        "click_full_2",
        "click_full_3",
        "rung1a",
        "rung1b",
        "sameapp",
    }
    assert aggregate["outputs"]["result"]["marker"] == "EXECUTOR_READY.json"


def test_full_click_recipes_use_four_registered_core_shards() -> None:
    for shard in range(4):
        raw = tomllib.loads(
            (RECIPES / f"executor_cert_click_full_{shard}_cpu_kvm.toml").read_text()
        )
        command = " ".join(raw["command"])
        assert f"--shard-index={shard}" in command
        assert "--suite=certification" in command
        assert raw["outputs"]["result"]["marker"] == (
            f"transport_certification_shard_{shard}.json"
        )


def test_instrumented_diagnostic_is_a_separate_fail_closed_identity() -> None:
    pipeline = tomllib.loads(DIAGNOSTIC_PIPELINE.read_text(encoding="utf-8"))
    assert pipeline == {
        "name": "proper_vm_executor_instrumented_diagnostic_v1",
        "stages": {
            "build": {"recipe": "../recipes/executor_cert_build_cpu.toml"},
            "instrumented_diagnostic": {
                "recipe": "../recipes/executor_diag_click_instrumented_cpu_kvm.toml"
            },
        },
    }
    raw = tomllib.loads(DIAGNOSTIC_RECIPE.read_text(encoding="utf-8"))
    text = DIAGNOSTIC_RECIPE.read_text(encoding="utf-8")
    assert raw["name"] == "proper_vm_executor_click_instrumented_diagnostic_cpu_kvm_v1"
    assert raw["repo"] == "juergen_gui_executor_integration"
    assert raw["resources"]["gpus"] == 0
    assert raw["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert raw["inputs"]["build"] == {
        "type": "stage",
        "stage": "build",
        "role": "result",
    }
    assert raw["outputs"]["result"]["marker"] == "ARTIFACT_INDEX.json"
    assert "diagnostic_rc=$?" in text
    assert '--result="$terminal_name=$terminal_path"' in text
    assert "transport_diagnostic_progress.json" in text
    assert "vm_metadata.json" in text
    assert 'exit "$diagnostic_rc"' in text
    assert "retry" not in " ".join(raw["command"]).lower()
    assert "heldout" not in text.lower()
    assert "sealed_eval" not in text.lower()
