from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[2]
PINNED_COMMON = {
    "phaseb_oracle_eval.py": "13ac5d8731d375d833359e750ca0608bf869449177df3047f1955413b915d7c6",
    "phaseb_canonical_eval.py": "08bad72ba5b63b2a1f36c1622535cb4022d5856b95d86fb11e91f93a8ffaacc2",
    "phaseb_relative/relative_eval.py": "006bddca38ad7304adbe50c9956b03fe2feb3ae8bcb34e89412825842f38000f",
    "phaseb_deltatype_raw_v2/action_v2.py": "1ded3d5a7e51da71cf3082049fbdd404971ebf72a95d93f333ebb3ee3075ccb7",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recipe(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_final_evals_pin_one_common_scorer_bundle_and_drain_vllm_tree():
    wrappers = {
        "normalized": EXPERIMENTS / "phaseb_normalized_v2/eval.sh",
        "raw": EXPERIMENTS / "phaseb_deltatype_raw_v2/eval.sh",
    }
    recipes = {
        "normalized": EXPERIMENTS / "phaseb_normalized_v2/labctl/recipes/eval_A_to_A_r256_s900_v1.toml",
        "raw": EXPERIMENTS / "phaseb_deltatype_raw_v2/labctl/recipes/eval_A_to_B_r256_s900_v1.toml",
    }
    for arm, wrapper in wrappers.items():
        source = wrapper.read_text()
        full_eval = source.rindex('"$PY" "${A[experiment_dir]}/../phaseb_oracle_eval.py"')
        shutdown = source.index("shutdown_vllm\n", full_eval)
        assert "setsid --wait uv run --no-sync vllm serve" in source
        assert 'kill -TERM -- "-$pid"' in source
        assert 'kill -KILL -- "-$pid"' in source
        assert 'kill -0 -- "-$pid"' in source
        assert shutdown > full_eval

        command = " ".join(recipe(recipes[arm])["command"])
        assert digest(wrapper) in command
        for relative, expected in PINNED_COMMON.items():
            assert digest(EXPERIMENTS / relative) == expected
            assert expected in command


def test_final_exports_pin_exact_omegalax_state_and_are_marker_last():
    arms = (
        ("phaseb_normalized_v2", "export_A_to_A_r256_s900_v1.toml"),
        ("phaseb_deltatype_raw_v2", "export_resume_A_to_B_r256_s900_v1.toml"),
    )
    for directory, filename in arms:
        script = EXPERIMENTS / directory / "export.sh"
        command = " ".join(recipe(EXPERIMENTS / directory / "labctl/recipes" / filename)["command"])
        assert digest(script) in command
        assert "b3f32c002998a1134c78845847a53ca9cc17fb10" in command
        assert "329d339f7211042e21c57346d7181374fca0593c939f770960f6f6741e793401" in command
        source = script.read_text()
        assert source.index("scripts/export_to_hf.py") < source.index("manifest.write_text")


def test_shared_scorer_seals_identical_inputs_and_estimand():
    source = (EXPERIMENTS / "phaseb_oracle_eval.py").read_text()
    assert "EXPECTED_ROWS = 233" in source
    assert "EXPECTED_COORD = 178" in source
    assert 'temperature=0.0' in source
    assert 'max_tokens=256' in source
    assert '"estimand": "oracle_history_single_turn_greedy_generation"' in source
    assert "sealed raw-v2 held-out file changed" in source
    assert "sealed normalized-v2 held-out file changed" in source
    assert "canonical gold seal changed" in source
