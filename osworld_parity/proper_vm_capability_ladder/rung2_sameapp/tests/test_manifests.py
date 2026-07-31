from __future__ import annotations

import pytest

from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.fixtures import (
    APPS,
    HORIZONS,
    ManifestError,
    assert_collectable_split,
    load_all_manifests,
)


def test_manifests_are_sealed_and_pairwise_disjoint() -> None:
    manifests = load_all_manifests()
    ids: list[str] = []
    seeds: list[int] = []
    for split, manifest in manifests.items():
        assert manifest.sealed is (split == "sealed_eval")
        assert {fixture.app for fixture in manifest.fixtures} == set(APPS)
        assert {fixture.horizon for fixture in manifest.fixtures} == {
            HORIZONS[app] for app in APPS
        }
        ids.extend(fixture.id for fixture in manifest.fixtures)
        seeds.extend(fixture.parameter_seed for fixture in manifest.fixtures)
    assert len(ids) == len(set(ids))
    assert len(seeds) == len(set(seeds))


def test_sealed_eval_cannot_enter_replay_or_collection() -> None:
    with pytest.raises(ManifestError, match="eval-only"):
        assert_collectable_split("sealed_eval")
