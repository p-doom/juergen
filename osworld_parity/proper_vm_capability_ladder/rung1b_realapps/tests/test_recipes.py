from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[3] / "labctl" / "recipes"
RECIPES = (
    "rung1b_realapps_build_selfcheck_cpu.toml",
    "rung1b_realapps_vm_selfcheck_cpu_kvm.toml",
    "rung1b_training_data_build_cpu.toml",
    "rung1b_teacher_collect_cpu_kvm.toml",
)


def test_all_rung1b_recipes_are_cpu_only_and_parse():
    for name in RECIPES:
        recipe = tomllib.loads((ROOT / name).read_text(encoding="utf-8"))
        assert recipe["resources"]["gpus"] == 0
        assert "evaluation" not in str(recipe.get("args", {}).get("split", ""))


def test_kvm_recipes_pin_provider_and_refuse_gpu_visibility():
    for name in ("rung1b_realapps_vm_selfcheck_cpu_kvm.toml", "rung1b_teacher_collect_cpu_kvm.toml"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "/dev/kvm" in text
        assert "GPU allocation is forbidden" in text
        assert "gpus = 0" in text
    selfcheck = tomllib.loads(
        (ROOT / "rung1b_realapps_vm_selfcheck_cpu_kvm.toml").read_text(encoding="utf-8")
    )
    assert selfcheck["args"]["expected_provider_sha256"] == (
        "76a8f44fab16c6dd38a4378a270e38758ba8d31885f244baedb95d8178f588d7"
    )
