from __future__ import annotations

import json

from osworld_parity.proper_vm_capability_ladder.rung1.fixtures import (
    COORDINATE_CONTRACT,
    HORIZONS,
    TEMPLATES,
    load_manifest,
)
from osworld_parity.proper_vm_capability_ladder.rung1.selfcheck import (
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
