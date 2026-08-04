"""Phase-A curriculum: overlap-gate contract and labctl recipe invariants.

The two overlap-gate tests are ported verbatim from the originating suite
(`experiments/synthetic_multistep/tests/test_synthetic_multistep.py` at
juergen-rft 860bb66) — they are the only tests there whose dependency closure
lies inside this branch. The remaining tests pin the lineage this branch claims
(r256/alpha256, 750 steps, lr 1e-4, the two warm-start artifacts) and the recipe
hygiene rules, directly against the committed recipe TOMLs.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

OSWORLD_PARITY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OSWORLD_PARITY / "synthetic_multistep"))

from build_curriculum_stage2 import _collision_types, _overlaps  # noqa: E402


def load_recipes(*subdirs: str) -> dict[str, dict]:
    recipes: dict[str, dict] = {}
    for subdir in subdirs:
        for path in sorted((OSWORLD_PARITY / subdir / "labctl/recipes").glob("*.toml")):
            recipes[path.name] = tomllib.loads(path.read_text())
    return recipes


# --- ported verbatim -------------------------------------------------------


def test_curriculum_overlap_gate_checks_every_frozen_geometry_key():
    left = {
        "bbox": {(1, 2, 3, 4)},
        "center": {(2, 3)},
        "cursor_bbox": {((9, 8), (1, 2, 3, 4))},
        "image_sha256": {"abc"},
    }
    right = {key: set(values) for key, values in left.items()}
    assert _overlaps(left, right) == {
        "bbox": 1, "center": 1, "cursor_bbox": 1, "image_sha256": 1,
    }
    assert _overlaps(left, {key: set() for key in left}) == {
        "bbox": 0, "center": 0, "cursor_bbox": 0, "image_sha256": 0,
    }


def test_curriculum_generator_forced_collision_is_rejected_by_all_keys():
    cursor = (9, 8)
    bbox = (1, 2, 151, 152)
    forbidden = {
        "bbox": {bbox},
        "center": {(76, 77)},
        "cursor_bbox": {(cursor, bbox)},
        "image_sha256": {"forced-image-hash"},
    }
    assert _collision_types(
        cursor=cursor, bbox=bbox, image_sha256="forced-image-hash",
        forbidden=forbidden,
    ) == {"bbox", "center", "cursor_bbox", "image_sha256"}


# --- lineage contract ------------------------------------------------------


def test_phase_a_stage2_trains_750_steps_at_lr_1e_4_from_the_pinned_warm_start():
    script = (OSWORLD_PARITY / "synthetic_multistep/curriculum_train_export.sh").read_text()
    assert "--num_steps=750" in script
    assert "--learning_rate=1e-4" in script
    assert "--lora_rank=256 --lora_alpha=256" in script
    train = load_recipes("synthetic_multistep")["train_curriculum_A_to_B_r256_pinned.toml"]
    # The tool-call warm-up that produced this checkpoint is a resolved
    # experiment and is not in this branch; the checkpoint artifact is the
    # pipeline's entry point and must stay pinned by run id.
    assert train["inputs"]["source_model"]["artifact"] == (
        "relative_factorial_reltool_pre_r256_s750_capacity_v1_"
        "run_019fb4beda3472b289ae60fc612c1cea"
    )


def test_phase_b_continues_from_the_phase_a_stage2_endpoint():
    recipes = load_recipes("phaseb_deltatype_raw_v2")
    warm_start = (
        "synthetic_multistep_curriculum_A_to_B_raw_pre_r256_s750_recovered_v1_"
        "run_019fb56fb2f471118f1a9ed683def8b0"
    )
    for filename in (
        "tokenize_authorize_production_r256_v1.toml",
        "train_production_A_to_B_r256_s900_v1.toml",
    ):
        assert recipes[filename]["inputs"]["source_model"]["artifact"] == warm_start


def test_no_recipe_declares_a_personal_path():
    for path in sorted(OSWORLD_PARITY.rglob("labctl/recipes/*.toml")):
        text = path.read_text()
        for personal in ("/home/franz", "/fast/home/"):
            assert personal not in text, f"{path.name}: {personal}"


def test_every_recipe_is_unique_bounded_and_never_self_submits():
    recipes = load_recipes("synthetic_multistep", "phaseb_deltatype_raw_v2")
    assert recipes, "no recipes found"
    aliases = [
        recipe["outputs"][next(iter(recipe["outputs"]))]["alias"]
        for recipe in recipes.values()
    ]
    assert len(aliases) == len(set(aliases))
    for filename, recipe in recipes.items():
        command = " ".join(recipe["command"])
        assert "sbatch" not in command, filename
        assert "labctl submit" not in command, filename
        hours, minutes, seconds = (
            int(part) for part in recipe["resources"]["time"].split(":")
        )
        assert 0 < hours * 3600 + minutes * 60 + seconds <= 8 * 3600, filename


def test_recipes_execute_the_immutable_snapshot_not_the_live_worktree():
    # `.provenance.repo_path` reads the mutable worktree; an edit mid-run is what
    # truncated Phase-A stage 2 (job 135464) and forced a re-export.
    recipes = load_recipes("synthetic_multistep", "phaseb_deltatype_raw_v2")
    for filename, recipe in recipes.items():
        command = " ".join(recipe["command"])
        assert ".provenance.repo_path" not in command, filename
        assert ".source_path" in command, filename


def test_recipes_reference_only_paths_that_exist_in_this_branch():
    recipes = load_recipes("synthetic_multistep", "phaseb_deltatype_raw_v2")
    repo = OSWORLD_PARITY.parent
    for filename, recipe in recipes.items():
        command = " ".join(recipe["command"])
        for token in command.split():
            if "$REPO/osworld_parity/" in token:
                relative = token.split("$REPO/", 1)[1].strip('"\'\\')
                assert (repo / relative).exists(), f"{filename}: {relative}"
