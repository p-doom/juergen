from __future__ import annotations

import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from experiments.synthetic_multistep.contract import (
    Contract,
    DEFAULT_AUDIT_DIR,
    EXPECTED_ACTION,
    SEMANTICS,
    load_frozen,
    load_jsonl,
    request_seed,
    serialize_action,
    strict_schema_ok,
    unit_range_ok,
    verify_frozen_sources,
)
from experiments.synthetic_multistep.evaluate import run_episode, validate_episode_artifact
from experiments.synthetic_multistep.metrics import summarize
from experiments.synthetic_multistep.compare import paired_uncertainty
from experiments.synthetic_multistep.production_compare import miss_diagnostics
from experiments.synthetic_multistep.build_curriculum_stage2 import (
    _collision_types,
    _overlaps,
)
from experiments.synthetic_multistep.curriculum_launch_gate import _recipe_diff
from experiments.synthetic_multistep.teacher_forced import _find_subsequence
from experiments.synthetic_multistep.build_typing_factorial import render as render_typing
from experiments.synthetic_multistep.typing_evaluate import decode as decode_typing
from experiments.synthetic_multistep.typing_evaluate import schema_ok as typing_schema_ok

HERE = Path(__file__).resolve().parents[1]
SLURM_RUNTIME_PY = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/"
    "reinforcement-learning/prime-rl/.venv/bin/python"
)


def test_typing_execution_and_format_schema_are_strict():
    event = lambda kind, what: SimpleNamespace(kind=kind, what=what, mouse_button=None)
    coalesced = SimpleNamespace(
        no_op=False, terminate=False, fail=False,
        elements=(("type", "abc 2"),),
    )
    perkey = SimpleNamespace(
        no_op=False, terminate=False, fail=False,
        elements=(("event", event("press", "KeyA")),
                  ("event", event("release", "KeyA")),
                  ("event", event("press", "Space")),
                  ("event", event("release", "Space")),
                  ("event", event("press", "Num2")),
                  ("event", event("release", "Num2"))),
    )
    assert typing_schema_ok(coalesced, "coalesced")
    assert decode_typing(coalesced.elements) == "abc 2"
    assert typing_schema_ok(perkey, "perkey")
    assert decode_typing(perkey.elements) == "a 2"
    perkey.elements = perkey.elements[:-1]
    assert not typing_schema_ok(perkey, "perkey")


def test_frozen_sources_and_step1_byte_geometry_identity():
    actual = verify_frozen_sources()
    assert actual["rung2_scene.py"] == load_frozen()["sources"]["rung2_scene.py"]
    scene = load_jsonl(
        DEFAULT_AUDIT_DIR / "runs/rung2_offshelf/px/scenes.jsonl"
    )[0]
    contract = Contract(verify=False)
    assert contract.screen == (1920, 1080)
    assert contract.render_png(scene["bbox"], tuple(scene["cursor"])) == Path(
        scene["image_path"]
    ).read_bytes()


@pytest.mark.parametrize("semantic", SEMANTICS)
def test_coordinate_units_oracle_parse_and_cursor_update(semantic):
    contract = Contract(verify=False)
    cursor = (321, 876)
    target = (1601, 121)
    coord = contract.ideal_coord(semantic, cursor, target)
    text = serialize_action(semantic, coord)
    move = contract.parse(semantic, text)
    assert move.parse_ok and move.coord == coord
    assert move.action == EXPECTED_ACTION[semantic]
    assert strict_schema_ok(semantic, text, move.coord)
    assert unit_range_ok(semantic, move.coord)
    landing = contract.apply_coord(semantic, cursor, coord)
    # Both normalized semantics must resolve to exactly the same quantized target.
    assert landing == contract.apply_coord("absolute_toolcall", cursor, contract.to_norm(target))
    assert abs(landing[0] - target[0]) <= 1
    assert abs(landing[1] - target[1]) <= 1


def test_semantic_independent_seeds_and_matched_serialization():
    assert request_seed("e", 0, 2, 3) == request_seed("e", 0, 2, 3)
    absolute = serialize_action("absolute_toolcall", (700, 200))
    relative = serialize_action("move_rel", (-70, 20))
    # The wrapper/key order/whitespace are identical; only the two semantic values differ.
    assert absolute.replace("left_click", "ACTION").replace("700, 200", "COORD") == (
        relative.replace("move_rel", "ACTION").replace("-70, 20", "COORD")
    )
    assert strict_schema_ok("absolute_toolcall", absolute, (700, 200))
    assert not strict_schema_ok("move_rel", absolute, (700, 200))


def test_raw_pixel_oracle_schema_preamble_and_endpoint():
    contract = Contract(verify=False)
    cursor = (321, 876)
    target = (1601, 121)
    coord = contract.ideal_coord("deltatype_raw", cursor, target)
    text = serialize_action("deltatype_raw", coord, prose="trained prose retained")
    move = contract.parse("deltatype_raw", text)
    assert move.parse_ok and move.coord == coord and move.action == "delta"
    assert strict_schema_ok("deltatype_raw", text, move.coord)
    assert unit_range_ok("deltatype_raw", move.coord)
    assert contract.apply_coord("deltatype_raw", cursor, coord) == target
    prompt = contract.user_text(
        "deltatype_raw", cursor, target, target_index=0, target_count=4,
        preamble=True,
    )
    base = contract.rung2.build_user_text(
        contract.rung2.GRAMMARS["deltatype_raw"],
        {"cursor": list(cursor), "target_center": list(target)},
        False,
        True,
    )
    old = (
        "This is a SINGLE-STEP targeting task. The screenshot is the FINAL state -- "
        "do NOT wait and do NOT terminate."
    )
    new = (
        "This is a MULTI-STEP targeting task. After a correct action a new green box "
        "appears; after a miss the same box remains. Keep acting until every box is hit.\n"
        "Current target: 1 of 4."
    )
    assert prompt == base.replace(old, new)
    assert contract.system_prompt("deltatype_raw") == contract.rung2.GRAMMARS[
        "deltatype_raw"
    ]["system"]


def test_raw_pixel_closed_loop_recovery_uses_actual_cursor(tmp_path):
    episode_root = tmp_path / "episodes_raw"
    episode_root.mkdir()
    contract = Contract(verify=False)
    cursor = (200, 200)
    bbox = [900, 400, 1050, 550]
    target = (975, 475)
    step1 = contract.render_png(bbox, cursor)
    (episode_root / "step1.png").write_bytes(step1)
    spec = {
        "episode_id": "raw", "episode_index": 0, "kind": "long",
        "initial_cursor": list(cursor), "step1_image": "step1.png",
        "step1_png_sha256": __import__("hashlib").sha256(step1).hexdigest(),
        "targets": [{"target_index": 0, "bbox": bbox, "target_center": list(target)}],
    }
    first = (300, 50)
    after_first = contract.apply_coord("deltatype_raw", cursor, first)
    second = contract.ideal_coord("deltatype_raw", after_first, target)
    outputs = [
        serialize_action("deltatype_raw", first, prose="first complete output"),
        serialize_action("deltatype_raw", second, prose="second complete output"),
    ]
    row = run_episode(
        _FakeClient(outputs), model="fake", semantic="deltatype_raw", contract=contract,
        episode_root=episode_root, spec=spec, max_attempts=3, history_turns=3,
        max_tokens=192, preamble=True,
    )
    assert row["completed"] and row["preamble"]
    assert row["steps"][1]["cursor_before"] == list(after_first)
    assert row["steps"][1]["cursor_after"] == list(target)
    assert all(step["schema_ok"] and step["unit_range_ok"] for step in row["steps"])


def test_full_episode_build_guards_and_teacher_forced_prefixes(tmp_path):
    root = tmp_path / "episodes"
    # Match the compiled labctl/Slurm wrapper contract exactly.  The system
    # python on compute nodes does not include Pillow, and labctl expands
    # templates only through [args], not inside the command string.
    subprocess.run(
        [
            "bash", str(HERE / "build_stage.sh"),
            f"--runtime_python={SLURM_RUNTIME_PY}",
            f"--audit_dir={DEFAULT_AUDIT_DIR}", f"--out={root}",
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    manifest = json.loads((root / "build_manifest.json").read_text())
    validated, specs = validate_episode_artifact(root)
    assert validated == manifest
    assert len(specs) == 80
    assert manifest["oracle_hits"] == {"absolute_toolcall": 320, "move_rel": 320}
    assert set(manifest["oracle_rate"].values()) == {1.0}
    assert manifest["step1_identity"] == {
        "checked": 80, "byte_equal": 80, "geometry_equal": 80
    }
    assert manifest["generated_oracle_geometry"]["train_overlap"] == 0
    assert manifest["generated_oracle_geometry"]["val_overlap"] == 0
    for split in ("heldout_train", "heldout_val"):
        assert all(manifest["leak_report"][kind][split] == 0
                   for kind in ("scene_id", "bbox", "center", "geometry"))

    oracle = {
        semantic: load_jsonl(root / f"oracle_{semantic}.jsonl") for semantic in SEMANTICS
    }
    for absolute, relative in zip(
        oracle["absolute_toolcall"], oracle["move_rel"], strict=True
    ):
        for a_turn, r_turn in zip(absolute["turns"], relative["turns"], strict=True):
            assert a_turn["image_sha256"] == r_turn["image_sha256"]
            assert a_turn["cursor_before"] == r_turn["cursor_before"]
            assert a_turn["landing"] == r_turn["landing"]
            assert a_turn["hit"] and r_turn["hit"]


def _step(*, attempt, before, after, hit=False, parse=True, schema=True,
          coord=(1, 0), oscillation=False):
    return {
        "target_index": 0,
        "attempt": attempt,
        "distance_before": before,
        "distance_after": after,
        "progress_px": before - after,
        "hit": hit,
        "parse_ok": parse,
        "schema_ok": schema,
        "coord": list(coord) if coord is not None else None,
        "unit_range_ok": True,
        "regression": after > before,
        "oscillation": oscillation,
    }


def test_metric_definitions_recovery_auc_progress_and_schema():
    rows = [
        {"target_count": 1, "completed": True, "steps": [
            _step(attempt=1, before=100, after=50),
            _step(attempt=2, before=50, after=0, hit=True),
        ]},
        {"target_count": 1, "completed": False, "steps": [
            _step(attempt=1, before=100, after=120),
            _step(attempt=2, before=120, after=100, oscillation=True),
            _step(attempt=3, before=100, after=100, parse=False, schema=False, coord=None),
        ]},
    ]
    metrics = summarize(rows, max_attempts=3)
    assert metrics["target_reach_cdf_by_attempt"] == {"1": 0.0, "2": 0.5, "3": 0.5}
    assert metrics["episode_completion_rate"] == 0.5
    assert metrics["first_miss_recovery_rate"] == 0.5
    # First episode's first miss recovers; its only miss event is recovered.
    assert metrics["miss_event_recovery_rate"] == pytest.approx(1 / 4)
    assert metrics["progress_rate"] == 3 / 4
    assert metrics["regression_rate"] == 1 / 4
    assert metrics["parse_rate"] == 4 / 5
    assert metrics["strict_schema_rate"] == 4 / 5
    assert metrics["oscillation_rate"] == 1 / 5
    assert math.isfinite(metrics["normalized_distance_auc"])


def test_production_miss_diagnostics_separate_geometry_and_contract_failures():
    geometry_miss = _step(attempt=1, before=100, after=20)
    recovery = _step(attempt=2, before=20, after=0, hit=True)
    parse_miss = _step(
        attempt=1, before=100, after=100, parse=False, schema=False, coord=None
    )
    unit_miss = _step(attempt=1, before=100, after=50)
    unit_miss["unit_range_ok"] = False
    rows = [
        {"target_count": 1, "steps": [geometry_miss, recovery]},
        {"target_count": 1, "steps": [parse_miss]},
        {"target_count": 1, "steps": [unit_miss]},
    ]
    diagnostics = miss_diagnostics(rows)
    assert diagnostics == {
        "step_misses": 3,
        "parse_failures": 1,
        "strict_schema_failures": 1,
        "coordinate_unit_violations": 1,
        "geometry_only_misses": 1,
        "first_attempt_missed_targets": 3,
        "first_misses_recovered_by_attempt_2": 1,
        "first_misses_recovered_by_attempt_3": 1,
        "unrecovered_targets": 2,
    }


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


def test_curriculum_recipe_gate_allows_only_source_and_identity_fields():
    base = {
        "name": "A", "inputs": {"source_model": {"artifact": "source-a"}},
        "outputs": {"model": {"alias": "model-a"}}, "resources": {"time": "03:00:00"},
    }
    matched = {
        "name": "B", "inputs": {"source_model": {"artifact": "source-b"}},
        "outputs": {"model": {"alias": "model-b"}}, "resources": {"time": "03:00:00"},
    }
    assert _recipe_diff(base, matched) == []
    matched["resources"]["time"] = "02:59:00"
    assert _recipe_diff(base, matched) == ["resources.time"]


def test_teacher_forced_span_alignment_is_unique_and_fail_closed():
    assert _find_subsequence([9, 1, 2, 3, 8], [1, 2, 3]) == (1, 4)
    with pytest.raises(Exception, match="unique token span"):
        _find_subsequence([1, 2, 1, 2], [1, 2])


def test_typing_split_visual_nonces_are_byte_disjoint(tmp_path):
    train = tmp_path / "train.png"
    validation = tmp_path / "validation.png"
    render_typing(train, 0)
    render_typing(validation, 2000)
    assert train.read_bytes() != validation.read_bytes()


class _FakeCompletions:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def create(self, **_kwargs):
        text = next(self.outputs)
        message = SimpleNamespace(content=text, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(completion_tokens=10),
        )


class _FakeClient:
    def __init__(self, outputs):
        self.chat = SimpleNamespace(completions=_FakeCompletions(outputs))


def test_true_loop_uses_emitted_cursor_state_and_recovers(tmp_path):
    episode_root = tmp_path / "episodes"
    episode_root.mkdir()
    contract = Contract(verify=False)
    cursor = (200, 200)
    bbox = [900, 400, 1050, 550]
    target = (975, 475)
    step1 = contract.render_png(bbox, cursor)
    (episode_root / "step1.png").write_bytes(step1)
    spec = {
        "episode_id": "fake",
        "episode_index": 0,
        "kind": "long",
        "initial_cursor": list(cursor),
        "step1_image": "step1.png",
        "step1_png_sha256": __import__("hashlib").sha256(step1).hexdigest(),
        "targets": [{"target_index": 0, "bbox": bbox, "target_center": list(target)}],
    }
    # Deliberate first miss, then a relative oracle computed from the actual new cursor.
    first = (250, 100)
    after_first = contract.apply_coord("move_rel", cursor, first)
    second = contract.ideal_coord("move_rel", after_first, target)
    outputs = [serialize_action("move_rel", first), serialize_action("move_rel", second)]
    row = run_episode(
        _FakeClient(outputs), model="fake", semantic="move_rel", contract=contract,
        episode_root=episode_root, spec=spec, max_attempts=3, history_turns=3,
        max_tokens=192,
    )
    assert row["completed"] and row["reached_targets"] == 1
    assert len(row["steps"]) == 2
    assert row["steps"][1]["cursor_before"] == list(after_first)
    assert row["steps"][0]["observation_sha256"] != row["steps"][1]["observation_sha256"]
    assert row["steps"][1]["hit"]
    metrics = summarize([row], max_attempts=3)
    assert metrics["first_miss_recovery_rate"] == 1.0


def test_paired_uncertainty_clusters_episodes_and_targets():
    def row(episode, semantic, first_hit):
        steps = [_step(attempt=1, before=100, after=0 if first_hit else 20,
                       hit=first_hit, oscillation=False)]
        steps[0].update({"bbox": [10, 10, 20, 20], "sampling_seed": 7})
        if not first_hit:
            recovery = _step(attempt=2, before=20, after=0, hit=True)
            recovery.update({"bbox": [10, 10, 20, 20], "sampling_seed": 8})
            steps.append(recovery)
        return {"episode_id": episode, "episode_index": int(episode[1:]),
                "kind": "long", "semantic": semantic, "k": 0,
                "initial_cursor": [0, 0], "target_count": 1,
                "completed": True, "steps": steps}
    absolute = [row(f"e{i}", "absolute_toolcall", True) for i in range(80)]
    relative = [row(f"e{i}", "move_rel", i >= 8) for i in range(80)]
    report = paired_uncertainty(absolute, relative, n_boot=1000, seed=3)
    metric = report["metrics"]["first_attempt_reach_rate"]
    assert metric["absolute"] == 1.0 and metric["relative"] == 0.9
    assert metric["relative_minus_absolute"] == pytest.approx(-0.1)
    assert report["mcnemar"]["first_attempt_reach"]["absolute_only"] == 8
    assert report["mcnemar"]["reach_by_attempt_2"]["discordant"] == 0


def test_labctl_recipes_are_unique_bounded_and_do_not_submit():
    recipe_dir = HERE / "labctl/recipes"
    recipes = {path.name: tomllib.loads(path.read_text()) for path in recipe_dir.glob("*.toml")}
    expected_core = {
        "build_episodes.toml", "eval_absolute_primary.toml",
        "eval_move_rel_primary.toml", "compare_primary.toml",
        "compare_primary_pinned_019fb4f6.toml",
        "eval_move_rel_capacity_r64_pinned.toml",
        "eval_move_rel_capacity_r256_pinned.toml",
        "compare_capacity_r64_pinned.toml",
        "compare_capacity_r256_pinned.toml",
        "capacity_curve_pinned.toml",
        "eval_production_move_rel_r256_pinned.toml",
        "eval_production_deltatype_raw_r256_pinned.toml",
        "compare_production_movement_r256_pinned.toml",
        "build_curriculum_stage2_pinned.toml",
        "train_curriculum_A_to_B_r256_pinned.toml",
        "train_curriculum_B_to_B_r256_pinned.toml",
        "train_curriculum_A_to_B_r256_lr5e5_prepared.toml",
        "train_curriculum_B_to_B_r256_lr5e5_prepared.toml",
        "export_recovery_A_to_B_pinned.toml",
        "export_recovery_B_to_B_pinned.toml",
        "export_recovery_A_to_B_lr5e5_pinned.toml",
        "export_recovery_B_to_B_lr5e5_pinned.toml",
        "eval_teacher_forced_A_to_B_pinned.toml",
        "eval_teacher_forced_B_to_B_pinned.toml",
        "eval_multistep_A_to_B_pinned.toml",
        "eval_multistep_B_to_B_pinned.toml",
        "eval_teacher_forced_A_to_B_lr5e5_pinned.toml",
        "eval_teacher_forced_B_to_B_lr5e5_pinned.toml",
        "eval_multistep_A_to_B_lr5e5_pinned.toml",
        "eval_multistep_B_to_B_lr5e5_pinned.toml",
        "compare_teacher_forced_original_pinned.toml",
        "compare_curriculum_original_pinned.toml",
        "compare_teacher_forced_lr5e5_pinned.toml",
        "compare_curriculum_lr5e5_pinned.toml",
        "cleanup_curriculum_original_orbax_pinned.toml",
        "cleanup_curriculum_lr5e5_orbax_pinned.toml",
        "cleanup_typing_untrusted_orbax_pinned.toml",
        "cleanup_retired_mihir_videocua_pinned.toml",
            "cleanup_retired_nativerel_videocua_pinned.toml",
            "proper_vm_stage2_live_smoke_prepared.toml",
            "proper_vm_stage2_absolute_prepared.toml",
            "proper_vm_stage2_normalized_prepared.toml",
            "proper_vm_stage2_raw_a_to_b_prepared.toml",
        "build_typing_factorial_pinned.toml",
        "train_typing_A_coalesced_pinned.toml",
        "train_typing_B_coalesced_pinned.toml",
        "train_typing_A_perkey_pinned.toml",
        "train_typing_B_perkey_pinned.toml",
        "resume_typing_A_coalesced_pinned.toml",
        "resume_typing_B_coalesced_pinned.toml",
        "resume_typing_A_perkey_pinned.toml",
        "resume_typing_B_perkey_pinned.toml",
        "resume_typing_A_perkey_hai008_pinned.toml",
        "resume_typing_B_coalesced_hai008_pinned.toml",
        "resume_typing_B_perkey_hai008_pinned.toml",
        "resume_typing_A_coalesced_tp2_gate_pinned.toml",
        "resume_typing_A_coalesced_tp2_exact_pinned.toml",
        "resume_typing_A_perkey_tp2_exact_pinned.toml",
        "resume_typing_B_coalesced_tp2_exact_pinned.toml",
        "resume_typing_B_perkey_tp2_exact_pinned.toml",
        "typing_tp2_offline_nnx_restore_gate_pinned.toml",
        "eval_typing_A_coalesced_pinned.toml",
        "eval_typing_B_coalesced_pinned.toml",
        "eval_typing_A_perkey_pinned.toml",
        "eval_typing_B_perkey_pinned.toml",
    }
    assert expected_core <= set(recipes)
    aliases = [recipe["outputs"][next(iter(recipe["outputs"]))]["alias"]
               for recipe in recipes.values()]
    assert len(aliases) == len(set(aliases))
    assert all("{run.id}" in alias for alias in aliases)
    for filename, recipe in recipes.items():
        hours, minutes, seconds = (int(x) for x in recipe["resources"]["time"].split(":"))
        limit = (3 * 3600 if filename.startswith(("train_curriculum_", "train_typing_", "resume_typing_"))
                 else 3600)
        assert hours * 3600 + minutes * 60 + seconds <= limit
        command = " ".join(recipe["command"])
        assert "sbatch" not in command and "labctl submit" not in command
        if filename.startswith("resume_typing_"):
            assert recipe["env"]["XLA_PYTHON_CLIENT_ALLOCATOR"] == "cuda_async"
            assert recipe["env"]["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.95"
    typing_resume = (HERE / "typing_factorial_resume_export.sh").read_text()
    assert 'SOURCE_CKPT250="$SOURCE_ORBAX/000250"; ORBAX="$OUT/orbax"' in typing_resume
    assert 'cp -a --reflink=auto --sparse=always "$SOURCE_ORBAX/." "$CLONE_TMP/"' in typing_resume
    assert "byte_compared_every_file':True" in typing_resume
    assert "'sealed_parent_orbax_unchanged':True" in typing_resume
    tp2_gate = recipes["resume_typing_A_coalesced_tp2_gate_pinned.toml"]
    assert tp2_gate["resources"]["gpus"] == 2
    assert tp2_gate["args"]["tp_size"] == "2"
    assert "recovered_tp2_gate" in tp2_gate["outputs"]["model"]["alias"]
    assert '--gpus-per-task="$TP_SIZE"' in typing_resume
    tp2_wrapper = (HERE / "typing_tp2_train_entrypoint.py").read_text()
    assert "expected_local_device_count" in tp2_wrapper
    assert "device_preflight_pass" in tp2_wrapper
    assert "_JAX_DISTRIBUTED_INITIALIZE(local_device_ids=[0, 1]" in tp2_wrapper
    assert "ocp.ArrayRestoreArgs" in tp2_wrapper
    assert 'mesh_shapes != [(2, 1, 1)]' in tp2_wrapper
    assert 'source_mesh_shapes != [(1, 1, 1)]' in tp2_wrapper
    assert "_checkpoint_dtype_tp2_target" in tp2_wrapper
    assert "PartitionSpec(None)" in tp2_wrapper
    assert 'jax.ShapeDtypeStruct((2,), jnp.uint32' in tp2_wrapper
    assert "all_train_state_leaves_bitwise_equal_to_cpu_source_restore" in tp2_wrapper
    assert "fresh_optimizer_dtype_canonicalization_applied" in tp2_wrapper
    assert "got.astype" not in tp2_wrapper
    for filename in (
        "resume_typing_A_coalesced_tp2_exact_pinned.toml",
        "resume_typing_A_perkey_tp2_exact_pinned.toml",
        "resume_typing_B_coalesced_tp2_exact_pinned.toml",
        "resume_typing_B_perkey_tp2_exact_pinned.toml",
    ):
        exact = recipes[filename]
        assert exact["resources"]["gpus"] == 2
        assert exact["args"]["tp_size"] == "2"
        assert "--nodelist=hai004" in exact["resources"]["sbatch_extra"]
        assert "--deadline=2026-07-31T09:00:00" in exact["resources"]["sbatch_extra"]
    for filename in (
        "export_recovery_A_to_B_pinned.toml",
        "export_recovery_B_to_B_pinned.toml",
    ):
        command = " ".join(recipes[filename]["command"])
        assert ".source_path" in command
        assert ".provenance.repo_path" not in command
    recovery = (HERE / "curriculum_export_recovery.sh").read_text()
    assert '--model_id="$BASE_MODEL"' in recovery
    teacher = (HERE / "teacher_forced.py").read_text()
    assert "device_map=" not in teacher
    assert '.to(torch.device("cuda:0"))' in teacher
    curriculum_evals = [
        recipe for filename, recipe in recipes.items()
        if filename.startswith(("eval_teacher_forced_", "eval_multistep_"))
    ]
    for recipe in curriculum_evals:
        command = " ".join(recipe["command"])
        assert ".source_path" in command
        assert ".provenance.repo_path" not in command
    for filename, recipe in recipes.items():
        if filename.startswith("eval_multistep_"):
            assert "--experiment_dir=$REPO/experiments/synthetic_multistep" in " ".join(
                recipe["command"]
            )
    build_command = " ".join(recipes["build_episodes.toml"]["command"])
    assert "{inputs.runtime.path}" not in build_command
    assert 'bash "$REPO/experiments/synthetic_multistep/build_stage.sh" "$@"' in build_command
    assert recipes["build_episodes.toml"]["args"]["runtime_python"] == (
        "{inputs.runtime.path}/.venv/bin/python"
    )
    assert recipes["build_episodes.toml"]["inputs"]["runtime"]["path"].endswith(
        "/reinforcement-learning/prime-rl"
    )
    frozen = load_frozen()["primary_checkpoints"]
    assert recipes["eval_absolute_primary.toml"]["inputs"]["model"]["artifact"] == frozen[
        "absolute_toolcall"
    ]
    assert recipes["eval_move_rel_primary.toml"]["inputs"]["model"]["artifact"] == frozen[
        "move_rel"
    ]
    pinned = recipes["compare_primary_pinned_019fb4f6.toml"]["inputs"]
    assert pinned["absolute"]["artifact"].startswith(
        "synthetic_multistep_phasea_absolute_primary_v1_run_019fb4f6"
    )
    assert pinned["relative"]["artifact"].startswith(
        "synthetic_multistep_phasea_move_rel_r32_primary_v1_run_019fb4f6"
    )
    capacity = load_frozen()["capacity_sensitivity"]
    for rank in (64, 256):
        recipe = recipes[f"eval_move_rel_capacity_r{rank}_pinned.toml"]
        assert recipe["inputs"]["model"]["artifact"] == capacity[
            "candidate_checkpoints"
        ][str(rank)]
        assert recipe["inputs"]["episodes"]["artifact"] == capacity["episode_artifact"]
        assert recipe["args"]["comparison_label"] == "capacity_sensitivity"
        assert recipe["resources"]["gpus"] == 1
        assert "--exclude=hai001,hai008" in recipe["resources"]["sbatch_extra"]
    production = load_frozen()["production_movement_bridge"]
    production_recipes = {
        "move_rel": recipes["eval_production_move_rel_r256_pinned.toml"],
        "deltatype_raw": recipes["eval_production_deltatype_raw_r256_pinned.toml"],
    }
    for semantic, recipe in production_recipes.items():
        assert recipe["inputs"]["model"]["artifact"] == production["checkpoints"][semantic]
        assert recipe["inputs"]["episodes"]["artifact"] == production["episode_artifact"]
        assert recipe["args"]["preamble"] == "true"
        assert recipe["args"]["comparison_label"] == "production_movement_bridge"
