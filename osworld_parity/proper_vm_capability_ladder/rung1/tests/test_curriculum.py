from __future__ import annotations

import copy
import json
from pathlib import Path

from PIL import Image

from osworld_parity.proper_vm_capability_ladder.rung1.curriculum import (
    FORMATS,
    build_curriculum,
    iter_cells,
    load_curriculum_spec,
    make_scene,
    oracle_accepts,
    plan_trajectory,
    render_state,
    validate_seed_and_parameter_isolation,
    validate_trajectory,
)


def test_curriculum_spec_counts_and_seed_isolation() -> None:
    spec = load_curriculum_spec()
    scenes = [
        make_scene(seed, capability, split)
        for split in ("train", "validation")
        for seed, capability in iter_cells(spec, split)
    ]
    assert len([scene for scene in scenes if scene["split"] == "train"]) == 1664
    assert len([scene for scene in scenes if scene["split"] == "validation"]) == 208
    assert validate_seed_and_parameter_isolation(spec, scenes) == {
        "sealed_fixture_seed_overlap": 0,
        "sealed_fixture_parameter_fingerprint_overlap": 0,
        "train_validation_scene_fingerprint_overlap": 0,
    }


def test_all_capabilities_round_trip_in_both_formats() -> None:
    capabilities = (
        "click",
        "focus_type",
        "scroll",
        "drag",
        "composition_2",
        "composition_3",
        "composition_4",
    )
    for offset, capability in enumerate(capabilities):
        scene = make_scene(510000 + offset, capability, "validation")
        for arm in FORMATS:
            result = validate_trajectory(scene, arm)
            assert result["round_trip_count"] == result["turn_count"]
            assert result["final_pointer_mask"] == 0
            final = plan_trajectory(scene, arm).final_state
            assert oracle_accepts(scene, final)


def test_format_twins_share_initial_pixels_and_raw_drag_stays_explicit(
    tmp_path: Path,
) -> None:
    scene = make_scene(520001, "composition_4", "validation")
    native = plan_trajectory(scene, "native_absolute_control")
    compact = plan_trajectory(scene, "compact_raw_phaseb")
    native_path = tmp_path / "native.png"
    compact_path = tmp_path / "compact.png"
    native_hash = render_state(scene, native.turns[0].state, native_path)
    compact_hash = render_state(scene, compact.turns[0].state, compact_path)
    assert native_hash == compact_hash
    assert Image.open(native_path).size == (1000, 700)

    raw_actions = [turn.action for turn in compact.turns]
    down = next(index for index, action in enumerate(raw_actions) if action.endswith("+LMB"))
    assert "; -LMB" in raw_actions[down + 2]
    assert "+LMB -LMB" not in raw_actions[down]


def test_curriculum_builder_publishes_only_after_invariants(tmp_path: Path) -> None:
    spec = copy.deepcopy(load_curriculum_spec())
    for row in spec["matrix"]:
        row["train"] = int(row["capability"] == "composition_4")
        row["validation"] = int(row["capability"] == "focus_type")
    spec["splits"]["train"]["records_per_format"] = 1
    spec["splits"]["validation"]["records_per_format"] = 1
    output = tmp_path / "curriculum"
    report = build_curriculum(output, spec_override=spec)
    assert report["status"] == "pass"
    assert (output / "build_manifest.json").is_file()
    assert report["format_twin_identity"] == {"passing": 2, "total": 2}
    for arm in FORMATS:
        for split in ("train", "val"):
            path = output / arm / "_normalized" / split / "chat.jsonl"
            row = json.loads(path.read_text(encoding="utf-8"))
            images = [
                item["image"]
                for message in row["messages"]
                for item in message["content"]
                if item["type"] == "image"
            ]
            assert images and all(Path(image).is_file() for image in images)
