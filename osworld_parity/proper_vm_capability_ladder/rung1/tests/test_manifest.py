from __future__ import annotations

import copy
import json

import pytest

from osworld_parity.proper_vm_capability_ladder.rung1.fixtures import (
    COORDINATE_CONTRACT,
    HORIZONS,
    TEMPLATES,
    load_manifest,
)
from osworld_parity.proper_vm_capability_ladder.rung1.selfcheck import (
    _checkpoint_progress,
    _reset_component_diff,
    _reset_component_report,
    _validate_manifest_bounds,
    main,
)


def test_frozen_fixture_counts_and_horizons() -> None:
    manifest = load_manifest()
    assert len(manifest.fixtures) == 40
    assert len(manifest.select(split="development")) == 8
    assert len(manifest.select(split="evaluation")) == 32
    for template in TEMPLATES:
        cells = [item for item in manifest.fixtures if item.template == template]
        assert len(cells) == 10
        assert sum(item.split == "evaluation" for item in cells) == 8
        assert {item.horizon for item in cells} == {HORIZONS[template]}
        assert {item.coordinate_contract for item in cells} == {COORDINATE_CONTRACT}


def test_manifest_has_no_official_task_material() -> None:
    manifest = load_manifest()
    serialized = json.dumps(
        [item.unsigned_payload() for item in manifest.fixtures]
    ).lower()
    for forbidden in (
        "evaluation_examples",
        "task_config",
        "heldout",
        "osworld_train",
        "osworld_eval",
    ):
        assert forbidden not in serialized
    assert "00000000-0000-0000-0000-000000000000" not in serialized


def test_every_fixture_hash_is_sealed() -> None:
    for fixture in load_manifest().fixtures:
        fixture.verify_hash()


def test_labctl_rendered_underscore_argument_is_accepted(tmp_path) -> None:
    assert (
        main(
            [
                "--mode=build",
                f"--output={tmp_path}",
                "--expected_provider_sha256=unused-in-build-mode",
            ]
        )
        == 0
    )
    assert (tmp_path / "selfcheck.json").is_file()


def test_every_manifest_row_fits_a_measured_smaller_viewport() -> None:
    manifest = load_manifest()
    report = _validate_manifest_bounds(
        manifest,
        {
            "screen_x": 0,
            "screen_y": 0,
            "screen_width": 1280,
            "screen_height": 720,
            "outer_width": 1280,
            "outer_height": 720,
            "inner_width": 1280,
            "inner_height": 640,
            "chrome_top": 80,
        },
        (1280, 720),
    )
    assert len(report["rows"]) == 40
    assert set(report["rows"]) == {fixture.id for fixture in manifest.fixtures}
    for audit in report["placement_collision_audit"].values():
        assert audit["unique_origin_count"] >= 8
        assert audit["max_origin_multiplicity"] <= 2
    click_row = report["rows"]["r1a-click-dev-1102"]
    assert click_row["design_origin"] == [1320, 650]
    assert len(click_row["transformed_viewport_origin"]) == 2


def test_active_dispatch_journal_is_durable_before_oracle(tmp_path) -> None:
    cell = {
        "fixture_id": "r1a-click-dev-1101",
        "arm": "compact_raw_phaseb",
        "gold_cursor_journal": {
            "planned_observed_baseline": [1730, 973],
            "dispatch_cursor_before": [1730, 973],
            "expected_endpoint": [365, 345],
            "final_cursor": [365, 345],
        },
        "gold_state_before_oracle": {"current": {"checked": False}},
    }
    _checkpoint_progress(
        tmp_path,
        cells=[],
        expected_cell_count=16,
        active_cell=cell,
        stage="gold_state_recorded_before_oracle",
    )
    payload = json.loads((tmp_path / "progress.json").read_text())
    assert payload["completed_cell_count"] == 0
    assert payload["active_cell"]["arm"] == "compact_raw_phaseb"
    assert payload["active_cell"]["journal_stage"] == (
        "gold_state_recorded_before_oracle"
    )
    assert payload["active_cell"]["gold_cursor_journal"]["final_cursor"] == [
        365,
        345,
    ]


@pytest.mark.parametrize(
    ("component", "mutation"),
    [
        ("fixture", "fixture"),
        ("logical_state", "logical"),
        ("cursor_button", "cursor"),
        ("cursor_button", "button"),
        ("window_geometry", "window"),
        ("dom_geometry", "dom"),
    ],
)
def test_reset_component_diff_reports_exact_changed_component(
    component: str, mutation: str
) -> None:
    state_a = {
        "fixture_id": "r1a-click-dev-1101",
        "fixture_sha256": "fixture-sha",
        "ready": True,
        "current": {"checked": False, "decoy_checked": False},
        "geometry": {
            "window": {"inner_width": 1850, "inner_height": 966},
            "target": {"left": 351, "top": 331, "right": 379, "bottom": 359},
            "decoy": {"left": 351, "top": 387, "right": 364, "bottom": 400},
        },
    }
    state_b = copy.deepcopy(state_a)
    cursor_a = (1728, 972)
    cursor_b = cursor_a
    buttons_a = buttons_b = 0
    if mutation == "fixture":
        state_b["fixture_sha256"] = "other-fixture-sha"
    elif mutation == "logical":
        state_b["current"]["checked"] = True
    elif mutation == "cursor":
        cursor_b = (1729, 972)
    elif mutation == "button":
        buttons_b = 1
    elif mutation == "window":
        state_b["geometry"]["window"]["inner_width"] = 1849
    elif mutation == "dom":
        state_b["geometry"]["target"]["left"] = 350
    reset_a = _reset_component_report(
        state_a, cursor=cursor_a, pointer_buttons=buttons_a
    )
    reset_b = _reset_component_report(
        state_b, cursor=cursor_b, pointer_buttons=buttons_b
    )
    diff = _reset_component_diff(reset_a, reset_b)
    assert diff["all_equal"] is False
    assert diff["differing_components"] == [component]
    assert diff["components"][component]["equal"] is False
    assert (
        diff["components"][component]["reset_a_sha256"]
        != diff["components"][component]["reset_b_sha256"]
    )
    for name, detail in diff["components"].items():
        if name != component:
            assert detail["equal"] is True
            assert detail["reset_a_sha256"] == detail["reset_b_sha256"]


def test_reset_snapshots_and_component_diff_are_checkpointed_before_failure(
    tmp_path,
) -> None:
    state_a = {
        "fixture_id": "r1a-click-dev-1101",
        "fixture_sha256": "fixture-sha",
        "ready": True,
        "current": {"checked": False, "decoy_checked": False},
        "geometry": {
            "window": {"inner_width": 1850},
            "target": {"left": 351},
        },
    }
    state_b = copy.deepcopy(state_a)
    state_b["geometry"]["window"]["inner_width"] = 1849
    reset_a = _reset_component_report(state_a, cursor=(10, 20), pointer_buttons=0)
    reset_b = _reset_component_report(state_b, cursor=(10, 20), pointer_buttons=0)
    cell = {
        "reset_a_snapshot": {"browser_state": state_a, "cursor": [10, 20]},
        "reset_b_snapshot": {"browser_state": state_b, "cursor": [10, 20]},
        "reset_a_components": reset_a,
        "reset_b_components": reset_b,
        "reset_component_diff": _reset_component_diff(reset_a, reset_b),
    }
    _checkpoint_progress(
        tmp_path,
        cells=[],
        expected_cell_count=16,
        active_cell=cell,
        stage="reset_comparison_recorded",
    )
    persisted = json.loads((tmp_path / "progress.json").read_text())["active_cell"]
    assert persisted["journal_stage"] == "reset_comparison_recorded"
    assert persisted["reset_a_snapshot"]["browser_state"] == state_a
    assert persisted["reset_b_snapshot"]["browser_state"] == state_b
    assert persisted["reset_component_diff"]["differing_components"] == [
        "window_geometry"
    ]
