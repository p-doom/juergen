from __future__ import annotations

import json
from pathlib import Path

from osworld_parity.proper_vm_capability_ladder.rung1.executor import parse_compact_raw
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.actions import (
    compile_compact,
    compile_native,
    validate_native,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.fixtures import (
    load_all_manifests,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.trajectory import (
    build_trajectory,
)


def _dummy_geometry():
    return {
        "editor": (820, 520), "cell": (640, 420), "source": (500, 300),
        "destination": (500, 420), "decoy": (500, 360), "moved": (500, 300),
        "nav": (260, 140), "decoy_nav": (460, 140), "toggle": (340, 760),
        "decoy_toggle": (340, 820), "scroll_surface": (960, 540),
    }


def test_gold_and_near_miss_compile_to_both_action_schemas() -> None:
    manifests = load_all_manifests()
    for manifest in (manifests["train"], manifests["development"]):
        for fixture in manifest.fixtures:
            for near_miss in (False, True):
                trajectory = build_trajectory(fixture, near_miss=near_miss)
                assert len(trajectory.turns) <= fixture.horizon
                assert len({turn.semantic_step for turn in trajectory.turns}) == fixture.semantic_steps
                geometry = _dummy_geometry()
                cursor = (960, 540)
                for turn in trajectory.turns:
                    native = compile_native(turn, geometry)
                    validate_native(native)
                    compact, cursor = compile_compact(turn, geometry, cursor)
                    parse_compact_raw(compact)


def test_checked_in_schema_names_match_compilers() -> None:
    path = Path(__file__).resolve().parents[1] / "action_schemas.json"
    schemas = json.loads(path.read_text(encoding="utf-8"))
    assert schemas["schema_version"] == 1
    assert set(schemas) >= {"native_absolute_sequence_v1", "compact_raw_phaseb_v1"}
