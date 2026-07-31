from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_canonical_click_and_drag_are_common_pixel_equivalent():
    build = load("build_relative")
    _, osw = build.load_modules(
        Path("/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/audit_operand"),
        Path("/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/scripts"),
    )
    cursor = [200, 300]
    for args, target, expected_tail in (
        ({"action": "left_click", "coordinate": [700, 500]}, [1344, 540], ["move_rel", "left_click"]),
        ({"action": "left_click_drag", "coordinate": [700, 500]}, [1344, 540],
         ["mouse_down", "move_rel", "mouse_up"]),
    ):
        text = build.render_relative_action(osw, args, cursor, target)
        calls = [dict(call.arguments) for call in osw.parse_computer_use_tool_calls(text)]
        assert [call["action"] for call in calls] == expected_tail
        relative = build.relative_landing(osw, text, cursor)
        absolute = build.normalized_abs_landing(args)
        assert absolute is not None
        assert max(abs(a - b) for a, b in zip(relative, absolute, strict=True)) <= 1


def test_relative_evaluator_teacher_roundtrip():
    evaluate = load("relative_eval")
    record = {
        "sample_id": "fixture",
        "phaseb_relative_audit": [{
            "cursor_before_px": [200, 300],
            "absolute_landing_px": [1344, 540],
            "relative_landing_px": [1344, 540],
        }],
        "messages": [{"role": "assistant", "content": [{"type": "text", "text":
            "Reasoning stays here.\n<tool_call>\n"
            '{"name":"computer_use","arguments":{"action":"move_rel","coordinate":[596,222]}}'
            "\n</tool_call>\n<tool_call>\n"
            '{"name":"computer_use","arguments":{"action":"left_click"}}'
            "\n</tool_call>"}]}],
    }
    gold = evaluate.teacher(record)
    row = evaluate.score(record, record["messages"][-1]["content"][0]["text"])
    assert gold["landing"] == (1344, 540)
    assert row["parse_ok"] and row["action_match"] and row["err_px"] == 0


def test_vision_budget_regression_catches_failed_five_image_record():
    budget = load("preflight_vision_budget")
    failed_record = [{
        "session_id": "observed-five-image-record",
        "num_images": 5,
        "vision_tokens": 10200,
        "vision_patches": 40800,
    }]
    with pytest.raises(ValueError, match=r"real_images=5.*real_patches=40800"):
        budget.check_budget(failed_record, max_images=2, max_patches=16000)
    budget.check_budget(failed_record, max_images=29, max_patches=64000)


def test_full_relative_dataset_fits_proven_phaseb_vision_budget():
    budget = load("preflight_vision_budget")
    dataset = Path(
        "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/franz.srambical/"
        "phaseb_relative_twins_v1_run_019fb4c0e52e77e286e84d7ea9319f94"
    )
    report = budget.preflight(dataset, "prose_keep", max_images=29, max_patches=64000)
    assert report["records_scanned"] == 2616
    assert report["observed"] == {"max_images": 5, "max_patches": 40800}


def test_visionfixed_export_and_eval_recipes_are_cpu_gpu_separated():
    recipes = ROOT / "labctl" / "recipes"
    export = tomllib.loads(
        (recipes / "export_prose_keep_visionfixed_v2.toml").read_text()
    )
    evaluation = tomllib.loads(
        (recipes / "eval_prose_keep_visionfixed_v2.toml").read_text()
    )

    assert "gpus" not in export["resources"]
    assert export["inputs"]["source"]["artifact"].endswith(
        "run_019fb4cbe728789086b8a63931c9979e"
    )
    assert export["outputs"]["model"]["marker"] == "export_manifest.json"
    assert evaluation["resources"]["gpus"] == 1
    assert evaluation["inputs"]["model"]["artifact"] == (
        export["outputs"]["model"]["alias"]
    )
    assert evaluation["outputs"]["result"]["marker"] == "eval_manifest.json"
