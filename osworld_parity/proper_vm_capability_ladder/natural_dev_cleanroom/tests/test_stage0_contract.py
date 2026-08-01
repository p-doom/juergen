from __future__ import annotations

import inspect
import subprocess
import tomllib
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.oracle import (
    initial_state,
    scripted_state,
)
from osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.qualify import (
    qualify_static as qualify_auxiliary_static,
)
from osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.schema import (
    Corpus,
    load_corpus,
)
from osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.smoke_schema import (
    load_smoke,
)
from osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.stage0_generate_inventory import (
    build,
)
from osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.stage0_actions import (
    MAX_MULTI_EMITTED_EVENTS,
    MAX_MULTI_PRIMITIVES,
    compile_multi_compact,
    compile_multi_native,
    compile_visible_app_switch_compact,
    compile_visible_app_switch_native,
    record_program_counts,
)
from osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.stage0_loader import (
    ANCHOR_APPS,
    MODES,
    RECORD_ELIGIBILITY,
    load_stage0_inventory,
    sha256_value,
)
from osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.stage0_oracle import (
    evaluate_composed,
    evaluate_composed_in_fresh_process,
)
from osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.stage0_qualify import (
    _passive_rebind_active_geometry,
    _probe_source,
    _switch_rebind_receipt,
    _verify_switch_rebind_receipt,
    _vm_record,
    _vm_repetition,
)


RECIPE_ROOT = Path(__file__).parents[3] / "labctl" / "recipes"


def test_stage0_inventory_is_generator_clean_balanced_and_disjoint() -> None:
    inventory = load_stage0_inventory()
    generated = build()
    assert generated["manifest_payload_sha256"] == inventory.manifest_payload_sha256
    assert generated["tasks"] == [task.to_dict() for task in inventory.tasks]
    assert len(inventory.tasks) == 40
    assert {
        (task.anchor_app, task.mode) for task in inventory.tasks
    } == {(app, mode) for app in ANCHOR_APPS for mode in MODES}
    assert all(
        sum(task.anchor_app == app and task.mode == mode for task in inventory.tasks)
        == 4
        for app in ANCHOR_APPS
        for mode in MODES
    )
    assert all(task.eligibility == RECORD_ELIGIBILITY for task in inventory.tasks)
    old_ids = {task.id for task in load_corpus().tasks} | {
        task.id for task in load_smoke().tasks
    }
    old_seeds = {task.parameter_seed for task in load_corpus().tasks} | {
        task.parameter_seed for task in load_smoke().tasks
    }
    new_ids = {task.id for task in inventory.tasks} | {
        source.id for task in inventory.tasks for source in task.component_tasks
    }
    new_seeds = {task.parameter_seed for task in inventory.tasks} | {
        source.parameter_seed
        for task in inventory.tasks
        for source in task.component_tasks
    }
    assert not old_ids & new_ids
    assert not old_seeds & new_seeds
    source_instructions = [
        source.instruction
        for task in inventory.tasks
        for source in task.component_tasks
    ]
    assert len(source_instructions) == 60
    assert len(set(source_instructions)) == 60
    old_corpus = load_corpus()
    auxiliary = qualify_auxiliary_static(
        Corpus(
            tasks=(old_corpus.tasks[0],),
            manifest_payload_sha256=old_corpus.manifest_payload_sha256,
            provenance=old_corpus.provenance,
            eligibility=old_corpus.eligibility,
        )
    )
    assert auxiliary["inventory_role"] == "auxiliary_development_only"
    assert auxiliary["eligibility"]["stage0"] is False


def test_multi_records_cover_true_ordered_cross_app_composition() -> None:
    inventory = load_stage0_inventory()
    for anchor in ANCHOR_APPS:
        records = [
            task
            for task in inventory.tasks
            if task.anchor_app == anchor and task.mode == "multi"
        ]
        assert {task.component_tasks[1].app for task in records} == set(ANCHOR_APPS) - {
            anchor
        }
        for record in records:
            assert record.bridge == "multi_app"
            assert record.semantic_steps == 2
            assert len(record.component_tasks) == 2
            assert record.component_tasks[0].app == anchor
            assert record.component_tasks[1].app != anchor
            assert "Alt+Tab" in record.instruction
            assert [item["semantic_steps"] for item in record.ordered_components] == [
                1,
                1,
            ]
            counts = record_program_counts(record.component_tasks)
            expected_action_turns = 4 if "chrome" in {
                source.app for source in record.component_tasks
            } else 3
            assert counts["primitive_actions"] == expected_action_turns
            assert counts["primitive_actions"] <= MAX_MULTI_PRIMITIVES
            assert counts["emitted_events"] <= MAX_MULTI_EMITTED_EVENTS
            assert record.program_budget == {
                "primitive_actions": counts["primitive_actions"],
                "primitive_action_ceiling": MAX_MULTI_PRIMITIVES,
                "emitted_events": counts["emitted_events"],
                "emitted_event_ceiling": MAX_MULTI_EMITTED_EVENTS,
                "visible_app_switch_included": True,
            }
            for source in record.component_tasks:
                geometry = (
                    {
                        "nav": (100, 100),
                        "toggle": (100, 500),
                        "decoy_nav": (200, 100),
                        "decoy_toggle": (200, 500),
                    }
                    if source.app == "chrome"
                    else {}
                )
                native = compile_multi_native(source, geometry)
                compact = compile_multi_compact(source, geometry, (50, 50))
                near_native = compile_multi_native(source, geometry, near_miss=True)
                near_compact = compile_multi_compact(
                    source, geometry, (50, 50), near_miss=True
                )
                assert native and compact
                assert near_native and near_compact
                assert sum(len(turn["operations"]) for turn in native) == source.horizon
                assert source.semantic_steps == 1
                assert source.horizon <= 3
            assert counts["primitive_actions"] == (
                1
                + sum(
                    len(compile_multi_native(source, {
                        "nav": (100, 100),
                        "toggle": (100, 500),
                        "decoy_nav": (200, 100),
                        "decoy_toggle": (200, 500),
                    } if source.app == "chrome" else {}))
                    for source in record.component_tasks
                )
            )


def test_fresh_composed_oracle_rejects_each_component_near_miss() -> None:
    record = next(
        task
        for task in load_stage0_inventory().tasks
        if task.anchor_app == "writer" and task.mode == "multi"
    )
    reset = [initial_state(task) for task in record.component_tasks]
    assert not evaluate_composed_in_fresh_process(record, reset).MOUSE_SOLVED
    gold = [scripted_state(task, near_miss=False) for task in record.component_tasks]
    assert evaluate_composed_in_fresh_process(record, gold).MOUSE_SOLVED
    for index, task in enumerate(record.component_tasks):
        states = list(gold)
        states[index] = scripted_state(task, near_miss=True)
        result = evaluate_composed(record, states)
        assert result.oracle_status == "ok"
        assert not result.MOUSE_SOLVED


def test_stage0_runtime_uses_one_visible_alt_tab_and_no_retry_loop() -> None:
    assert compile_visible_app_switch_native()["operations"] == [
        {"action": "key_chord", "keys": ["AltLeft", "Tab"]}
    ]
    assert compile_visible_app_switch_compact() == (
        "0 0 0; +AltLeft +Tab -Tab -AltLeft"
    )
    source = inspect.getsource(_vm_repetition)
    receipt_source = inspect.getsource(_switch_rebind_receipt)
    assert '{"action": "key", "keys": ["AltLeft", "Tab"]}' in receipt_source
    assert '"policy_visible": True' in receipt_source
    assert "for attempt" not in source
    assert "target_token not in after" in source
    assert "_passive_rebind_active_geometry" in source
    assert "_switch_rebind_receipt" in source
    assert '"fresh_passive_post_switch_probe"' in inspect.getsource(
        _switch_rebind_receipt
    )
    passive_source = inspect.getsource(_passive_rebind_active_geometry)
    assert "wmctrl" not in passive_source
    assert "probe_geometry" not in passive_source
    assert "near_miss_order" in source
    record_source = inspect.getsource(_vm_record)
    assert "near_miss_trials" in record_source
    assert 'row.get("near_miss_exact") is True' in record_source
    assert 'row.get("trial_state_exact") is True' in record_source
    assert "for attempt" not in record_source


def test_switch_rebind_receipt_is_exactly_bound_and_independently_sealed() -> None:
    record = next(
        task for task in load_stage0_inventory().tasks if task.mode == "multi"
    )
    task = record.component_tasks[1]
    window = {"window_id": "0x20", "window_line": f"0x20 {task.params.get('file_name', task.id)}"}
    frame = {
        **window,
        "x": 0,
        "y": 0,
        "width": 1200,
        "height": 700,
        "window_class": "fixture",
    }
    events = (
        SimpleNamespace(kind="key_down", args=("AltLeft",)),
        SimpleNamespace(kind="key_down", args=("Tab",)),
        SimpleNamespace(kind="key_up", args=("Tab",)),
        SimpleNamespace(kind="key_up", args=("AltLeft",)),
    )
    switch = SimpleNamespace(
        operations=events,
        adapter="native_absolute_control",
        action_class="key_chord",
        executor_dispatch_status="ok",
        atomic_state={"ok": True},
    )
    receipt = _switch_rebind_receipt(
        record,
        task,
        compile_visible_app_switch_native(),
        switch,
        {"window_id": "0x10", "window_line": "anchor"},
        window,
        {"editor": (600, 350)},
        {
            "probe": "active_window_geometry_read_only_v1",
            "activation_commands": 0,
            "active_before": frame,
            "active_after": frame,
        },
    )
    unsigned = dict(receipt)
    seal = unsigned.pop("switch_rebind_receipt_sha256")
    assert seal == sha256_value(unsigned)
    assert receipt["task_id"] == task.id
    assert receipt["fixture_sha256"] == task.fixture_sha256
    assert receipt["record_semantic_step"] == 2
    assert receipt["app"] == task.app
    assert receipt["arm"] == "native_absolute_control"
    assert receipt["exact_dispatched_event_sha256"] == sha256_value(
        receipt["exact_dispatched_events"]
    )
    _verify_switch_rebind_receipt(receipt, record, task)


def test_short_files_probe_observes_exact_near_miss_path(tmp_path: Path) -> None:
    task = next(
        source
        for record in load_stage0_inventory().tasks
        for source in record.component_tasks
        if record.mode == "multi" and source.app == "files"
    )
    root = tmp_path / task.id
    root.mkdir()
    (root / str(task.near_miss["final_name"])).write_text(
        str(task.params["content"]), encoding="utf-8"
    )

    class LocalTransport:
        _r2_guest_root = PurePosixPath(str(tmp_path))

        def execute_argv(self, argv: list[str]) -> dict[str, str]:
            completed = subprocess.run(
                argv, check=True, capture_output=True, text=True
            )
            return {"output": completed.stdout}

    assert _probe_source(LocalTransport(), task, task.as_fixture()) == scripted_state(
        task, near_miss=True
    )


def test_stage0_recipe_is_cpu_only_five_shard_repeatability_path() -> None:
    with (RECIPE_ROOT / "natural_dev_cleanroom_stage0_cpu_kvm.toml").open(
        "rb"
    ) as handle:
        recipe = tomllib.load(handle)
    assert recipe["resources"]["gpus"] == 0
    assert recipe["sweep"] == {
        "arg": "shard-index",
        "start": 0,
        "end": 4,
        "throttle": 5,
    }
    assert recipe["args"]["shard-index"] == "0"
    assert all("_" not in key for key in recipe["args"])
    assert "hai001,hai002,hai005" in " ".join(
        recipe["resources"]["sbatch_extra"]
    )
