from __future__ import annotations

import json

from osworld_parity.proper_vm_capability_ladder.rung1.fixtures import (
    HORIZONS,
    TEMPLATES,
    load_manifest,
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
