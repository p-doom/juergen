from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from experiments.relative_factorial.build_relative import ARMS, BuildError, build
from experiments.relative_factorial.effects import CELLS, EffectError, _load_cell, calculate


ROOT = Path("/fast/project/HFMI_SynergyUnit/p-doom_shared/franz")
SOURCE = ROOT / "audit_operand/r3data_2k"
AUDIT = ROOT / "audit_operand"
EVAL_SCENES = AUDIT / "runs/rung2_offshelf/px/scenes.jsonl"
EXPERIMENT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("relative_factorial") / "out"
    report = build(source_root=SOURCE, out_root=out, audit_dir=AUDIT, eval_scenes=EVAL_SCENES)
    return out, report


def test_full_build_invariants(built):
    out, report = built
    assert report["converter_regression_groups"] == {"passing": 7, "total": 7}
    assert all(counts == {"train": 2000, "val": 200} for counts in report["counts"].values())
    assert report["geometry_leak"] == {"train_vs_seed0_eval": 0, "val_vs_seed0_eval": 0}
    for key in (
        "exact_record_image_order_matching",
        "assistant_outside_action_identity",
        "gold_relative_action_parse_and_land",
        "prompt_equality_to_rung2_eval",
        "prose_grammar_identity",
        "action_span_identity_across_preamble_twins",
    ):
        assert report[key]["passing"] == report[key]["total"]
    assert report["preamble_digit_leak"] == {"leaking": 0, "total": 2200}
    assert (out / "build_manifest.json").is_file()
    assert (out / "invariant_report.json").is_file()
    for arm in ARMS:
        assert len((out / arm / "_normalized/train/chat.jsonl").read_text().splitlines()) == 2000
        assert len((out / arm / "_normalized/val/chat.jsonl").read_text().splitlines()) == 200


def test_builder_refuses_overwrite(built):
    out, _ = built
    with pytest.raises(BuildError, match="refusing to overwrite"):
        build(source_root=SOURCE, out_root=out, audit_dir=AUDIT, eval_scenes=EVAL_SCENES)


def test_prompt_and_action_examples_are_relative(built):
    out, _ = built
    tool = json.loads((out / "reltool_pre/_normalized/train/chat.jsonl").read_text().splitlines()[0])
    raw = json.loads((out / "relraw_pre/_normalized/train/chat.jsonl").read_text().splitlines()[0])
    assert "Mouse movement is RELATIVE" in tool["messages"][0]["content"][0]["text"]
    assert '"action": "move_rel"' in tool["messages"][-1]["content"][0]["text"]
    assert "RELATIVE mouse move" in raw["messages"][0]["content"][0]["text"]
    assert raw["messages"][-1]["content"][0]["text"].splitlines()[-1].endswith("0 ; +LMB -LMB")
    assert tool["messages"][-1]["content"][0]["text"].splitlines()[0] == raw["messages"][-1]["content"][0]["text"].splitlines()[0]


def test_factorial_effect_sign_and_scale():
    # y = intercept + sum(beta_term * product(code)); the reported factorial
    # effect is 2*beta for every term under the documented +/-1 contrast.
    beta = {
        (0,): 0.10,
        (1,): 0.20,
        (2,): 0.30,
        (0, 1): 0.04,
        (0, 2): -0.05,
        (1, 2): 0.06,
        (0, 1, 2): -0.07,
    }
    values = {}
    for cell, spec in CELLS.items():
        codes = spec[:3]
        value = 0.5
        for axes, coefficient in beta.items():
            product = 1
            for axis in axes:
                product *= codes[axis]
            value += coefficient * product
        values[cell] = value
    effects = calculate(values)["effects"]
    assert effects["relativity"]["effect"] == pytest.approx(0.20)
    assert effects["grammar"]["effect"] == pytest.approx(0.40)
    assert effects["preamble"]["effect"] == pytest.approx(0.60)
    assert effects["relativity×grammar"]["effect"] == pytest.approx(0.08)
    assert effects["relativity×preamble"]["effect"] == pytest.approx(-0.10)
    assert effects["grammar×preamble"]["effect"] == pytest.approx(0.12)
    assert effects["relativity×grammar×preamble"]["effect"] == pytest.approx(-0.14)


def test_effect_loader_requires_matched_greedy_manifests(tmp_path):
    for index, (cell, spec) in enumerate(CELLS.items()):
        rel, grammar_code, pre, grammar_name = spec[:4]
        directory = tmp_path / cell
        directory.mkdir()
        manifest = {
            "relativity": "relative" if rel == 1 else "absolute",
            "grammar_wrapper": "tool" if grammar_code == 1 else "bare",
            "grammar_name": grammar_name,
            "preamble": pre == 1,
            "sampling": {"temperature": 0.0, "k": 1},
        }
        report = {"summary": {f"{grammar_name}/all": {"in_box": index / 10}}}
        (directory / "eval_manifest.json").write_text(json.dumps(manifest))
        (directory / "report.json").write_text(json.dumps(report))
        with pytest.raises(EffectError, match="missing report.json/rows.jsonl/eval_manifest.json"):
            _load_cell(cell, directory, "in_box")


def test_labctl_training_and_eval_contracts():
    script = (EXPERIMENT / "train_export.sh").read_text()
    exporter = (EXPERIMENT / "export_checkpoint.sh").read_text()
    for required in (
        "--lora_rank=32",
        "--lora_alpha=32",
        "--max_length=4096",
        "--num_steps=750",
        "--num_loss_tiles=8",
        "--val_steps=15",
        'CKPT="$ORBAX/000750"',
        "--checkpoint_path=\"$CKPT\"",
    ):
        assert required in script

    recipes = EXPERIMENT / "labctl/recipes"
    build_recipe_text = (recipes / "build_tokenize.toml").read_text()
    assert '"${A[prime_rl_repo]}/.venv/bin/python" "$EXP/build_relative.py"' in build_recipe_text
    train_names = ["reltool_act", "relraw_act", "reltool_pre", "relraw_pre"]
    for name in train_names:
        recipe = tomllib.loads((recipes / f"train_{name}.toml").read_text())
        assert recipe["repo"] == "juergen_rft"
        assert recipe["resources"] == {
            "gpus": 1,
            "cpus": 12,
            "mem": "150GB",
            "time": "08:00:00",
            "qos": "low",
            "account": "hfmi_synergyunit",
            "partition": "standard",
            "sbatch_extra": [
                "--signal=USR1@600", "--requeue", "--ntasks=1",
                "--ntasks-per-node=1", "--exclude=hai001",
                "--nodelist=hai003,hai004,hai007",
            ],
        }
        assert recipe["args"]["arm"] == name

    assert 'bash "$SCRIPT_DIR/export_checkpoint.sh"' in script
    assert "JAX_PLATFORMS=cpu srun --ntasks=1 --nodes=1 uv run" in exporter
    assert 'python scripts/export_to_hf.py' in exporter
    assert '"export_ran_inside_srun":True' in exporter

    export_names = ["abstool_act", "absraw_act", "abstool_pre", "absraw_pre"]
    for name in export_names:
        recipe = tomllib.loads((recipes / f"export_{name}.toml").read_text())
        assert recipe["repo"] == "juergen_rft"
        assert "gpus" not in recipe["resources"]
        assert recipe["resources"]["time"] == "08:00:00"
        assert recipe["resources"]["qos"] == "low"
        assert "--requeue" in recipe["resources"]["sbatch_extra"]
        assert recipe["inputs"]["orbax"]["path"].endswith(f"/{name}_r32_2k/000750")
        assert recipe["outputs"]["model"]["marker"] == "export_manifest.json"

    canonical_eval_names = [
        "reltool_act", "relraw_act", "reltool_pre", "relraw_pre",
        "abstool_act", "absraw_act", "abstool_pre", "absraw_pre",
    ]
    eval_files = [recipes / f"eval_{name}.toml" for name in canonical_eval_names]
    assert len(eval_files) == 8
    for path in eval_files:
        recipe = tomllib.loads(path.read_text())
        assert recipe["repo"] == "juergen_rft"
        assert recipe["resources"]["gpus"] == 1
        assert recipe["resources"]["time"] == "08:00:00"
        assert recipe["resources"]["qos"] == "low"
        assert "--requeue" in recipe["resources"]["sbatch_extra"]
        assert recipe["args"]["preamble"] in {"true", "false"}
        stem = path.stem.removeprefix("eval_")
        expected = {
            "reltool_act": ("train_reltool_act", "move_rel", "false"),
            "relraw_act": ("train_relraw_act", "deltatype_raw", "false"),
            "reltool_pre": ("train_reltool_pre", "move_rel", "true"),
            "relraw_pre": ("train_relraw_pre", "deltatype_raw", "true"),
            "abstool_act": ("export_abstool_act", "absolute_toolcall", "false"),
            "absraw_act": ("export_absraw_act", "absolute_raw", "false"),
            "abstool_pre": ("export_abstool_pre", "absolute_toolcall", "true"),
            "absraw_pre": ("export_absraw_pre", "absolute_raw", "true"),
        }[stem]
        assert recipe["inputs"]["model"]["stage"] == expected[0]
        assert recipe["args"]["grammar"] == expected[1]
        assert recipe["args"]["preamble"] == expected[2]
        if path.name.startswith(("eval_absraw", "eval_abstool")):
            assert recipe["inputs"]["model"]["type"] == "stage"
            assert recipe["inputs"]["model"]["stage"] == path.stem.replace("eval_", "export_")
            assert recipe["args"]["model_path"] == "{inputs.model.path}/hf"

    pipeline = tomllib.loads(
        (EXPERIMENT / "labctl/pipelines/full_factorial.toml").read_text()
    )
    assert len(pipeline["stages"]) == 18
