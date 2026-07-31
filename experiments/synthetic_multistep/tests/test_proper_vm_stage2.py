from __future__ import annotations

import importlib.util
import io
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "proper_vm_stage2" / "gate.py"
SPEC = importlib.util.spec_from_file_location("proper_vm_stage2_gate", MODULE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)

AGGREGATE_PATH = ROOT / "proper_vm_stage2" / "aggregate.py"
AGGREGATE_SPEC = importlib.util.spec_from_file_location(
    "proper_vm_stage2_aggregate", AGGREGATE_PATH
)
assert AGGREGATE_SPEC and AGGREGATE_SPEC.loader
aggregate = importlib.util.module_from_spec(AGGREGATE_SPEC)
sys.modules[AGGREGATE_SPEC.name] = aggregate
AGGREGATE_SPEC.loader.exec_module(aggregate)

RUN_ARM_PATH = ROOT / "proper_vm_stage2" / "run_arm.py"
sys.path.insert(0, str(RUN_ARM_PATH.parent))
sys.modules.setdefault("gate", gate)
RUN_ARM_SPEC = importlib.util.spec_from_file_location(
    "proper_vm_stage2_run_arm", RUN_ARM_PATH
)
assert RUN_ARM_SPEC and RUN_ARM_SPEC.loader
run_arm = importlib.util.module_from_spec(RUN_ARM_SPEC)
sys.modules[RUN_ARM_SPEC.name] = run_arm
RUN_ARM_SPEC.loader.exec_module(run_arm)


def test_full_cpu_gate_passes_for_scope_corrected_recovery_protocol():
    result = gate.run_selftest()
    assert result["status"] == "pass"
    assert result["launch_authorized"] is True
    assert result["scope_classification"]["roadmap_stage"] == "1.5"
    assert result["scope_classification"]["is_user_roadmap_stage_2"] is False
    assert result["scope_classification"]["is_free_running_multi_step_closed_loop"] is False
    assert result["cells"] == 320
    assert result["episodes"] == 80
    assert result["actuation_replays_tested"] == 1920


@pytest.mark.parametrize("semantic", ["absolute_toolcall", "move_rel", "deltatype_raw"])
def test_drag_plan_holds_before_movement_and_releases(semantic):
    plan = gate.actuation_plan(semantic, "drag", (100, 200), (400, 500))
    assert plan[0] == ("mouseDown", "left")
    assert plan[-1] == ("mouseUp", "left")
    assert plan[1][0] == ("moveRel" if semantic == "move_rel" else "moveTo")
    assert float(plan[1][-1]) > 0


def test_click_plan_moves_before_click():
    for semantic in ("absolute_toolcall", "move_rel", "deltatype_raw"):
        plan = gate.actuation_plan(semantic, "click", (10, 20), (100, 200))
        assert plan[0][0] in {"moveTo", "moveRel"}
        assert plan[-1][0] in {"click", "mouseUp"}


def test_paired_gate_rejects_unpaired_ids():
    with pytest.raises(gate.GateError):
        gate.paired_noninferiority({"a": True}, {"b": True})


def test_paired_gate_fails_at_first_prespecified_harm_boundary():
    n = 320
    maximum = gate.maximum_confirmatory_harms(n, 0.05, 0.05)
    absolute = {str(i): True for i in range(n)}
    treatment = {str(i): i >= maximum + 1 for i in range(n)}
    result = gate.paired_noninferiority(absolute, treatment)
    assert result["finite_benchmark_noninferior"] is True
    assert result["conservative_confirmatory_noninferior"] is False
    assert result["pass"] is False


def test_fake_desktop_rejects_invalid_event_order():
    desktop = gate.FakeDesktop((0, 0), (10, 10, 20, 20))
    with pytest.raises(gate.GateError):
        desktop.execute(("mouseUp", "left"))


def test_launch_requires_explicit_authorized_protocol_state(tmp_path):
    authorized = gate.load_protocol(require_launch_authorized=True)
    prepared = json.loads(json.dumps(authorized))
    prepared["status"] = "prepared_not_launched"
    prepared["launch_gate"]["authorized"] = False
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(prepared))
    assert gate.load_protocol(path)["status"] == "prepared_not_launched"
    with pytest.raises(gate.GateError, match="not explicitly authorized"):
        gate.load_protocol(path, require_launch_authorized=True)
    with pytest.raises(gate.GateError, match="must not authorize"):
        gate.load_protocol()
    assert authorized["status"] == "authorized_ready"


def test_stage_wrapper_checks_authorization_before_gpu_touch():
    text = (ROOT / "proper_vm_stage2" / "run_arm_stage.sh").read_text()
    preflight = text.index("--preflight-only")
    assert preflight < text.index("nvidia-smi")
    assert preflight < text.index("vllm serve")
    imports = text.index('"gymnasium", "openai", "PIL", "desktop_env.desktop_env"')
    assert imports < text.index("nvidia-smi")
    assert 'HOST_PY="${A[host_python]}/bin/python"' in text
    assert '"$HOST_PY" "$REPO/experiments/synthetic_multistep/proper_vm_stage2/run_arm.py"' in text
    assert '"$PY" "$REPO/experiments/relative_factorial/readiness.py"' in text


def test_prepared_arm_recipes_are_exact_and_nonrecursive():
    recipe_dir = ROOT / "labctl" / "recipes"
    expected = {
        "proper_vm_stage2_absolute_prepared.toml": (
            "absolute_phase_a",
            "absolute_toolcall",
        ),
        "proper_vm_stage2_normalized_prepared.toml": (
            "normalized_phase_a",
            "move_rel",
        ),
        "proper_vm_stage2_raw_a_to_b_prepared.toml": (
            "raw_a_to_b",
            "deltatype_raw",
        ),
    }
    for name, (arm, grammar) in expected.items():
        recipe = tomllib.loads((recipe_dir / name).read_text())
        required_resources = {
            "gpus": 1,
            "cpus": 32,
            "mem": "128GB",
            "time": "01:00:00",
        }
        assert all(recipe["resources"].get(key) == value for key, value in required_resources.items())
        assert recipe["args"]["arm"] == arm
        assert recipe["args"]["grammar"] == grammar
        assert recipe["args"]["live_smoke_manifest"].endswith("/live_smoke_manifest.json")
        assert "--no-requeue" in recipe["resources"]["sbatch_extra"]
        assert "--deadline=2026-07-31T09:45:00" in recipe["resources"]["sbatch_extra"]
        command = " ".join(recipe["command"])
        assert "sbatch" not in command and "labctl run" not in command


def test_fixed_four_chunk_fallback_recipes_are_disjoint_and_fresh_vm_jobs():
    recipe_dir = ROOT / "labctl" / "recipes"
    arms = {
        "absolute": ("absolute_phase_a", "absolute_toolcall"),
        "normalized": ("normalized_phase_a", "move_rel"),
        "raw_a_to_b": ("raw_a_to_b", "deltatype_raw"),
    }
    for stem, (arm, grammar) in arms.items():
        observed = []
        for chunk_index in range(4):
            path = recipe_dir / f"proper_vm_stage2_{stem}_chunk{chunk_index}_recovery.toml"
            recipe = tomllib.loads(path.read_text())
            start, stop = chunk_index * 80, (chunk_index + 1) * 80
            assert recipe["resources"]["gpus"] == 1
            assert "--no-requeue" in recipe["resources"]["sbatch_extra"]
            assert recipe["outputs"]["result"]["marker"] == "chunk_manifest.json"
            assert recipe["args"]["arm"] == arm
            assert recipe["args"]["grammar"] == grammar
            assert recipe["args"]["chunk_index"] == str(chunk_index)
            assert recipe["args"]["chunk_start"] == str(start)
            assert recipe["args"]["chunk_stop"] == str(stop)
            observed.extend(range(start, stop))
        assert observed == list(range(320))


def test_chunk_assembly_is_fail_closed_on_partial_overlap_and_order():
    text = (ROOT / "proper_vm_stage2" / "assemble_chunks.py").read_text()
    assert 'rows.partial.jsonl' in text
    assert "overlapping cells" in text
    assert "exactly cover the frozen 320-cell order" in text
    assert "zip(args.chunks, CHUNK_BOUNDS, strict=True)" in text


def _solid_png(rgb):
    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", (4, 3), rgb).save(output, format="PNG")
    return output.getvalue()


def test_exact_screenshot_readiness_poll_preserves_pixel_gate():
    expected = _solid_png((1, 2, 3))
    stale = _solid_png((9, 8, 7))

    class Controller:
        def __init__(self):
            self.screenshots = iter((stale, stale, expected))

        def get_screenshot(self):
            return next(self.screenshots)

    screenshot, attempts, hashes = run_arm._wait_for_exact_screenshot(
        Controller(), expected, timeout_s=1.0, poll_s=0.0
    )
    assert screenshot == expected
    assert attempts == 3
    assert hashes[-1] == gate.rgb_sha256(expected)
    assert hashes[0] == gate.rgb_sha256(stale)


def test_exact_screenshot_readiness_poll_fails_closed_with_hash_evidence():
    expected = _solid_png((1, 2, 3))
    stale = _solid_png((9, 8, 7))

    class Controller:
        def get_screenshot(self):
            return stale

    with pytest.raises(run_arm.GateError) as caught:
        run_arm._wait_for_exact_screenshot(
            Controller(), expected, timeout_s=0.0, poll_s=0.0
        )
    message = str(caught.value)
    assert f"expected={gate.rgb_sha256(expected)}" in message
    assert "attempts=1" in message
    assert gate.rgb_sha256(stale) in message


def test_aggregator_replay_validation_fails_closed():
    cell = SimpleNamespace(cell_id="cell", cursor=(10, 20))
    endpoint = (100, 200)
    plan = gate.actuation_plan("move_rel", "drag", cell.cursor, endpoint)
    replay = {
        "success": True,
        "cursor_after": list(endpoint),
        "plan": [list(command) for command in plan],
        "state": {
            "drag_success": True,
            "down": False,
            "button_presses": 1,
            "button_releases": 1,
        },
    }
    aggregate._validate_replay(
        replay,
        cell=cell,
        semantic="move_rel",
        operation="drag",
        endpoint=endpoint,
        expected_hit=True,
    )
    replay["state"]["button_releases"] = 0
    with pytest.raises(aggregate.GateError, match="guest-state mismatch"):
        aggregate._validate_replay(
            replay,
            cell=cell,
            semantic="move_rel",
            operation="drag",
            endpoint=endpoint,
            expected_hit=True,
        )
