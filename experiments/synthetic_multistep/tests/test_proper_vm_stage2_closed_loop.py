from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "proper_vm_stage2_closed_loop" / "closed_loop_contract.py"
SPEC = importlib.util.spec_from_file_location("roadmap_stage2_closed_loop_contract", CONTRACT_PATH)
assert SPEC and SPEC.loader
closed = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = closed
SPEC.loader.exec_module(closed)

DYNAMIC_PATH = ROOT / "proper_vm_stage2_closed_loop" / "dynamic_guest_app.py"
DYNAMIC_SPEC = importlib.util.spec_from_file_location(
    "roadmap_stage2_dynamic_guest_app", DYNAMIC_PATH
)
assert DYNAMIC_SPEC and DYNAMIC_SPEC.loader
dynamic = importlib.util.module_from_spec(DYNAMIC_SPEC)
sys.modules[DYNAMIC_SPEC.name] = dynamic
DYNAMIC_SPEC.loader.exec_module(dynamic)

RUNNER_PATH = ROOT / "proper_vm_stage2_closed_loop" / "runner.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("roadmap_stage2_runner", RUNNER_PATH)
assert RUNNER_SPEC and RUNNER_SPEC.loader
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)

AGGREGATE_PATH = ROOT / "proper_vm_stage2_closed_loop" / "aggregate.py"
AGGREGATE_SPEC = importlib.util.spec_from_file_location(
    "roadmap_stage2_closed_loop_aggregate", AGGREGATE_PATH
)
assert AGGREGATE_SPEC and AGGREGATE_SPEC.loader
stage2_aggregate = importlib.util.module_from_spec(AGGREGATE_SPEC)
sys.modules[AGGREGATE_SPEC.name] = stage2_aggregate
AGGREGATE_SPEC.loader.exec_module(stage2_aggregate)

CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "roadmap_stage2_synthetic_contract", ROOT / "contract.py"
)
assert CONTRACT_SPEC and CONTRACT_SPEC.loader
synthetic_contract = importlib.util.module_from_spec(CONTRACT_SPEC)
sys.modules[CONTRACT_SPEC.name] = synthetic_contract
CONTRACT_SPEC.loader.exec_module(synthetic_contract)


TARGETS = ((100, 100, 250, 250), (500, 300, 650, 450), (900, 500, 1050, 650), (1300, 700, 1450, 850))


def evidence(endpoint, hit, *, valid=True):
    return closed.AttemptEvidence(
        raw_output="output",
        parse_ok=valid,
        schema_ok=valid,
        unit_range_ok=valid,
        dispatched=valid,
        endpoint=endpoint,
        actual_cursor_after=endpoint if valid else None,
        guest_hit=hit if valid else None,
    )


def test_miss_changes_actual_cursor_and_rendered_cursor_pixels_but_not_target():
    contract = synthetic_contract.Contract()
    state = closed.initial_state("ep", (20, 20))
    before = closed.reference_png(contract, state, TARGETS)
    transition = closed.advance(state, evidence((400, 400), False), TARGETS)
    after = closed.reference_png(contract, transition.after, TARGETS)
    assert transition.after.target_index == 0
    assert transition.after.cursor == (400, 400)
    assert transition.after.attempts_on_target == 1
    assert transition.render_changed is True
    assert before != after


def test_hit_alone_advances_target_and_retains_cursor_in_next_pixels():
    contract = synthetic_contract.Contract()
    state = closed.initial_state("ep", (20, 20))
    transition = closed.advance(state, evidence((175, 175), True), TARGETS)
    assert transition.target_advanced is True
    assert transition.after.target_index == 1
    assert transition.after.cursor == (175, 175)
    expected = contract.render_png(TARGETS[1], (175, 175))
    assert closed.reference_png(contract, transition.after, TARGETS) == expected


def test_invalid_outputs_consume_retry_without_dispatch_or_pixel_change():
    contract = synthetic_contract.Contract()
    state = closed.initial_state("ep", (20, 20))
    before = closed.reference_png(contract, state, TARGETS)
    transition = closed.advance(state, evidence(None, None, valid=False), TARGETS)
    assert transition.after.cursor == state.cursor
    assert transition.after.target_index == state.target_index
    assert transition.after.attempts_on_target == 1
    assert transition.render_changed is False
    assert closed.reference_png(contract, transition.after, TARGETS) == before


def test_third_miss_terminates_without_exposing_later_target():
    state = closed.initial_state("ep", (20, 20))
    for endpoint in ((300, 300), (350, 350), (400, 400)):
        transition = closed.advance(state, evidence(endpoint, False), TARGETS)
        state = transition.after
    assert state.terminated is True and state.success is False
    assert state.target_index == 0
    assert transition.terminal_reason == "retry_budget_exhausted"
    with pytest.raises(closed.TransitionError):
        closed.advance(state, evidence((175, 175), True), TARGETS)


def test_all_four_verified_hits_complete_episode_with_attempt_curve():
    state = closed.initial_state("ep", (20, 20))
    for target_index, endpoint in enumerate(((175, 175), (575, 375), (975, 575), (1375, 775))):
        transition = closed.advance(state, evidence(endpoint, True), TARGETS)
        state = transition.after
        assert len(state.target_hit_attempts) == target_index + 1
    assert state.terminated is True and state.success is True
    assert state.target_hit_attempts == (1, 1, 1, 1)
    assert transition.terminal_reason == "all_targets_reached"


def test_guest_geometry_or_cursor_disagreement_invalidates_transition():
    state = closed.initial_state("ep", (20, 20))
    bad_hit = evidence((400, 400), True)
    with pytest.raises(closed.TransitionError, match="geometry"):
        closed.advance(state, bad_hit, TARGETS)
    bad_cursor = closed.AttemptEvidence("x", True, True, True, True, (175, 175), (176, 175), True)
    with pytest.raises(closed.TransitionError, match="physical cursor"):
        closed.advance(state, bad_cursor, TARGETS)


def test_seed_slots_are_arm_independent_and_condition_separated():
    seed = closed.request_seed("multi_step_closed_loop", "ep", 2, 3)
    assert seed == closed.request_seed("multi_step_closed_loop", "ep", 2, 3)
    assert seed != closed.request_seed("multi_step_closed_loop", "ep", 2, 2)
    assert seed != closed.request_seed("single_step_sentinel", "ep", 2, 3)


def test_design_separates_finite_parity_from_underpowered_inference():
    protocol = json.loads(
        (ROOT / "proper_vm_stage2_closed_loop" / "PROTOCOL_DRAFT.json").read_text()
    )
    assert protocol["launch_authorized"] is False
    assert protocol["conditions"]["single_step_sentinel"]["inferential_role"] == "separate diagnostic; never pooled with the multi-step primary"
    primary = protocol["primary_endpoint"]
    assert "three unoffset" in primary["finite_benchmark_gate"]
    assert "fails with one" in primary["inferential_support_gate"]
    assert "finite parity but inferentially unresolved" in primary["joint_reporting_rule"]


def test_guest_release_advances_only_on_hit():
    index, hit, complete = dynamic.release_transition(0, TARGETS, (400, 400))
    assert (index, hit, complete) == (0, False, False)
    index, hit, complete = dynamic.release_transition(0, TARGETS, (175, 175))
    assert (index, hit, complete) == (1, True, False)
    index, hit, complete = dynamic.release_transition(3, TARGETS, (1375, 775))
    assert (index, hit, complete) == (3, True, True)


def test_dynamic_render_command_is_monotonic_target_and_hash_checked(tmp_path):
    image = tmp_path / "scene.png"
    image.write_bytes(b"dynamic-pixels")
    digest = synthetic_contract.sha256_bytes(image.read_bytes())
    command = {
        "episode_revision": "revision",
        "sequence": 1,
        "target_index": 0,
        "bbox": list(TARGETS[0]),
        "cursor": [400, 400],
        "image_sha256": digest,
    }
    assert dynamic.validate_render_command(
        command,
        episode_revision="revision",
        previous_sequence=0,
        target_index=0,
        targets=TARGETS,
        image_path=image,
    ) == (1, (400, 400), digest)
    command["target_index"] = 1
    with pytest.raises(dynamic.GuestContractError, match="target"):
        dynamic.validate_render_command(
            command,
            episode_revision="revision",
            previous_sequence=0,
            target_index=0,
            targets=TARGETS,
            image_path=image,
        )


def test_runner_is_launch_disabled_by_frozen_draft():
    protocol = runner.load_protocol(runner.PROTOCOL_PATH, require_authorized=False)
    runner.validate_protocol(protocol)
    assert protocol["launch_authorized"] is False
    with pytest.raises(runner.GateError, match="not authorized"):
        runner.load_protocol(runner.PROTOCOL_PATH, require_authorized=True)


def test_dynamic_smoke_is_small_no_model_and_x_client_bounded():
    source = (
        ROOT / "proper_vm_stage2_closed_loop" / "live_dynamic_smoke.py"
    ).read_text()
    assert '"model_server_started": False' in source
    assert '"model_checkpoint_loaded": False' in source
    assert '"model_generated_history": False' in source
    assert '"full_stage2_arms_authorized": False' in source
    assert "_assert_no_guest_processes(controller, GUEST_SOURCE)" in source
    assert "count > baseline_x_clients + X_CLIENT_SLACK" in source
    recipe = tomllib.loads(
        (
            ROOT
            / "labctl/recipes/proper_vm_roadmap_stage2_dynamic_smoke_prepared.toml"
        ).read_text()
    )
    resources = recipe["resources"]
    assert "gpus" not in resources
    assert resources["time"] == "00:30:00"
    assert "--deadline=2026-07-31T09:40:00" in resources["sbatch_extra"]


def test_dynamic_frame_wait_preserves_exact_hash_gate(monkeypatch):
    class Controller:
        def __init__(self):
            self.frames = iter(("stale", "expected"))

        def get_screenshot(self):
            return next(self.frames)

    monkeypatch.setattr(runner, "rgb_sha256", lambda value: value)
    assert runner._wait_for_exact_rgb(
        Controller(), "expected", label="test", timeout_s=1.0
    ) == "expected"

    class StaleController:
        def get_screenshot(self):
            return "stale"

    with pytest.raises(runner.GateError, match="pixel mismatch"):
        runner._wait_for_exact_rgb(
            StaleController(), "expected", label="test", timeout_s=0.01
        )


def test_resume_accepts_only_atomic_complete_units():
    valid = {
        "schema_version": 1,
        "artifact_type": "synthetic_proper_vm_roadmap_stage2_complete_unit",
        "status": "complete",
        "arm": "normalized_relative",
        "protocol_sha256": "protocol",
        "condition": "multi_step_closed_loop",
        "rows": [{"attempt": 1}],
        "summary": {"success": True},
    }
    assert runner._basic_unit_trusted(valid, "normalized_relative", "protocol")
    for key, replacement in (
        ("status", "partial"),
        ("protocol_sha256", "other"),
        ("rows", []),
        ("summary", None),
    ):
        corrupt = dict(valid)
        corrupt[key] = replacement
        assert not runner._basic_unit_trusted(corrupt, "normalized_relative", "protocol")


def test_resume_policy_and_worst_case_runtime_are_frozen():
    protocol = json.loads(runner.PROTOCOL_PATH.read_text())
    resume = protocol["partial_resumability"]
    assert resume["mid_episode_resume_allowed"] is False
    assert resume["atomic_units"] == [
        "one complete single-step sentinel cell",
        "one complete multi-step episode",
    ]
    upper = protocol["resource_upper_bound_draft"]
    assert upper["maximum_requests_per_arm"] == 1280
    assert upper["total_model_requests_all_arms"] == 3840
    assert upper["estimated_worst_case_minutes_per_arm"] == 165
    assert upper["prepared_hard_wall_minutes_per_arm"] == 180


def test_bounded_vm_chunks_are_disjoint_complete_and_request_bounded(tmp_path):
    chunks = runner._chunk_ranges()
    assert len(chunks) == 16
    assert chunks == tuple((start, start + 5) for start in range(0, 80, 5))
    all_paths = []
    for start, end in chunks:
        paths = runner._chunk_unit_paths(tmp_path, start, end)
        assert len(paths) == 25
        assert (end - start) * runner.REQUEST_SLOTS_PER_EPISODE == 80
        all_paths.extend(paths)
    assert len(all_paths) == 400
    assert len(set(all_paths)) == 400


def test_one_cell_model_kvm_preflight_is_exact_and_full_launch_stays_disabled():
    protocol = json.loads(runner.PROTOCOL_PATH.read_text())
    gate = protocol["model_kvm_one_cell_preflight"]
    assert gate["status"] == "pass_authorization_consumed"
    assert gate["launch_authorized"] is False
    assert gate["cell"] == {"episode_index": 0, "target_index": 0}
    assert gate["requests_per_arm"] == 1
    assert gate["independent_arms"] == list(runner.ARM_NAMES)
    assert gate["full_arm_launch_authorized"] is False
    assert protocol["launch_authorized"] is False

    recipe_dir = ROOT / "labctl" / "recipes"
    expected = {
        "absolute": ("absolute_matched_control", "absolute_toolcall"),
        "normalized": ("normalized_relative", "move_rel"),
        "raw": ("raw_relative", "deltatype_raw"),
    }
    for label, (arm, grammar) in expected.items():
        recipe = tomllib.loads(
            (
                recipe_dir
                / f"proper_vm_roadmap_stage2_{label}_one_cell_preflight.toml"
            ).read_text()
        )
        assert recipe["args"]["mode"] == "one_cell_preflight"
        assert recipe["args"]["arm"] == arm
        assert recipe["args"]["grammar"] == grammar
        assert recipe["outputs"]["result"]["marker"] == "preflight_manifest.json"
        assert recipe["resources"]["gpus"] == 1
        assert recipe["resources"]["time"] == "00:10:00"
        assert "--deadline=2026-07-31T09:45:00" in recipe["resources"][
            "sbatch_extra"
        ]


def test_chunk0_authorization_is_hash_pinned_and_results_are_consumed():
    protocol = json.loads(runner.PROTOCOL_PATH.read_text())
    authorization = json.loads(runner.CHUNK0_AUTHORIZATION_PATH.read_text())
    gate = protocol["chunk0_closed_loop_pilot"]
    assert gate["status"] == "pass_authorization_consumed"
    assert gate["launch_authorized"] is False
    assert gate["chunk_index"] == 0
    assert gate["chunks_1_15_launch_authorized"] is False
    assert gate["full_arm_launch_authorized"] is False
    assert authorization["episode_start_inclusive"] == 0
    assert authorization["episode_end_exclusive"] == 5
    assert authorization["atomic_units_per_arm"] == 25
    assert authorization["maximum_request_slots_per_arm"] == 80
    assert set(authorization["preflight_manifests"]) == set(runner.ARM_NAMES)
    assert set(gate["results"]) == set(runner.ARM_NAMES)
    with pytest.raises(runner.GateError, match="authorization drift"):
        runner._validate_chunk0_authorization(protocol)
    for arm, record in gate["results"].items():
        manifest_path = Path(record["manifest"])
        assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == record[
            "manifest_sha256"
        ]
        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] == "complete"
        assert manifest["arm"] == arm
        assert manifest["chunk_index"] == 0
        assert manifest["atomic_units"] == 25
        assert manifest["chunks_1_15_launched"] is False
        assert manifest["full_arm_launch_authorized"] is False
    assert protocol["launch_authorized"] is False

    expected = {
        "absolute": ("absolute_matched_control", "absolute_toolcall"),
        "normalized": ("normalized_relative", "move_rel"),
        "raw": ("raw_relative", "deltatype_raw"),
    }
    recipe_dir = ROOT / "labctl" / "recipes"
    for label, (arm, grammar) in expected.items():
        recipe = tomllib.loads(
            (
                recipe_dir
                / f"proper_vm_roadmap_stage2_{label}_chunk0_pilot.toml"
            ).read_text()
        )
        assert recipe["args"]["mode"] == "chunk0_pilot"
        assert recipe["args"]["arm"] == arm
        assert recipe["args"]["grammar"] == grammar
        assert recipe["outputs"]["result"]["marker"] == "chunk0_pilot_manifest.json"
        assert recipe["resources"]["time"] == "00:20:00"
        assert "--deadline=2026-07-31T09:58:00" in recipe["resources"][
            "sbatch_extra"
        ]


def test_one_cell_manifest_hash_dependency_and_qemu_port_probe_scope():
    assert runner.hashlib.sha256(b"one-cell").hexdigest() == hashlib.sha256(
        b"one-cell"
    ).hexdigest()
    live_smoke = (
        ROOT / "proper_vm_stage2" / "live_smoke.py"
    ).read_text(encoding="utf-8")
    assert 'probe.bind(("0.0.0.0", port))' in live_smoke


def test_prepared_arm_recipes_are_launch_disabled_and_exact():
    recipe_dir = ROOT / "labctl" / "recipes"
    expected = {
        "proper_vm_roadmap_stage2_absolute_prepared.toml": (
            "absolute_matched_control",
            "absolute_toolcall",
        ),
        "proper_vm_roadmap_stage2_normalized_prepared.toml": (
            "normalized_relative",
            "move_rel",
        ),
        "proper_vm_roadmap_stage2_raw_prepared.toml": (
            "raw_relative",
            "deltatype_raw",
        ),
    }
    for name, (arm, grammar) in expected.items():
        recipe = tomllib.loads((recipe_dir / name).read_text())
        assert recipe["args"]["arm"] == arm
        assert recipe["args"]["grammar"] == grammar
        assert recipe["resources"]["gpus"] == 1
        assert recipe["resources"]["time"] == "03:00:00"
        assert "prepared_not_authorized" in recipe["name"]
        assert "--no-requeue" in recipe["resources"]["sbatch_extra"]


def test_stage2_wrapper_preflights_before_gpu_or_model_server():
    text = (ROOT / "proper_vm_stage2_closed_loop" / "run_arm_stage.sh").read_text()
    preflight = text.index("--preflight-only")
    assert preflight < text.index("nvidia-smi")
    assert preflight < text.index("vllm serve")
    assert "proper_vm_stage2_closed_loop.runner" in text
    assert 'HOST_PY="${A[host_python]}/bin/python"' in text
    assert 'MODE="${A[mode]:-full}"' in text
    assert '--launch-scope "$MODE"' in text
    assert '"${runner[@]}" --base-url "$base_url" --one-cell-preflight' in text


def test_episode_finite_parity_and_inferential_support_are_separate():
    absolute = {i: {"summary": {"success": True}} for i in range(80)}
    three_harms = {
        i: {"summary": {"success": i >= 3}} for i in range(80)
    }
    result = stage2_aggregate._contrast(absolute, three_harms)
    assert result["treatment_minus_absolute"] == -3 / 80
    assert result["finite_benchmark_noninferior"] is True
    assert result["inferential_support"] is False
    assert result["conclusion"] == "finite parity but inferentially unresolved"
    four_harms = {
        i: {"summary": {"success": i >= 4}} for i in range(80)
    }
    assert stage2_aggregate._contrast(absolute, four_harms)[
        "finite_benchmark_noninferior"
    ] is False
    assert stage2_aggregate._contrast(absolute, absolute)["inferential_support"] is True


def test_aggregator_replays_dynamic_pixels_parse_and_exact_button_counts():
    contract = synthetic_contract.Contract()
    cell = type(
        "Cell",
        (),
        {
            "episode_id": "ep",
            "episode_index": 0,
            "target_index": 0,
            "cursor": (20, 20),
            "bbox": TARGETS[0],
        },
    )()
    episode_id = "ep:t00"
    endpoint = (175, 175)
    png = closed.reference_png(contract, closed.initial_state(episode_id, cell.cursor), [cell.bbox])
    row = {
        "condition": "single_step_sentinel",
        "episode_id": episode_id,
        "episode_index": 0,
        "target_index": 0,
        "attempt": 1,
        "request_seed": closed.request_seed("single_step_sentinel", episode_id, 0, 1),
        "raw_output": "155 155 0 ; +LMB -LMB",
        "tool_calls": [],
        "completion_tokens": 8,
        "parse_ok": True,
        "schema_ok": True,
        "unit_range_ok": True,
        "dispatched": True,
        "coord": [155, 155],
        "endpoint": list(endpoint),
        "cursor_before": list(cell.cursor),
        "cursor_after": list(endpoint),
        "active_bbox": list(cell.bbox),
        "observation_rgb_sha256": stage2_aggregate.rgb_sha256(png),
        "render_revision_before": "initial",
        "guest_hit": True,
        "target_advanced": True,
        "attempts_on_target_after": 0,
        "terminated": True,
        "success": True,
        "terminal_reason": "all_targets_reached",
        "guest_state": {
            "down": False,
            "target_index": 0,
            "completed": True,
            "button_presses": 1,
            "button_releases": 1,
            "render_revision": "initial",
            "rendered_cursor": list(cell.cursor),
            "image_sha256": hashlib.sha256(png).hexdigest(),
            "last_release_position": list(endpoint),
            "last_hit": True,
        },
        "next_observation_rgb_sha256": None,
        "next_render_revision": None,
    }
    unit = {
        "schema_version": 1,
        "artifact_type": "synthetic_proper_vm_roadmap_stage2_complete_unit",
        "status": "complete",
        "condition": "single_step_sentinel",
        "episode_id": episode_id,
        "episode_index": 0,
        "arm": "raw_relative",
        "protocol_sha256": "protocol",
        "sentinel_target_index": 0,
        "rows": [row],
        "summary": {
            "success": True,
            "terminated": True,
            "attempts_total": 1,
            "target_hit_attempts": [1],
            "targets_reached": 1,
            "final_target_index": 0,
            "final_cursor": list(endpoint),
        },
    }
    stage2_aggregate._replay_unit(
        unit,
        arm="raw_relative",
        protocol_hash="protocol",
        contract=contract,
        episode_cells=[cell],
    )
    row["guest_state"]["button_releases"] = 0
    with pytest.raises(runner.GateError, match="button-count"):
        stage2_aggregate._replay_unit(
            unit,
            arm="raw_relative",
            protocol_hash="protocol",
            contract=contract,
            episode_cells=[cell],
        )
