from __future__ import annotations

from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.actions import (
    ACTION_SCHEMAS,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.manifests import (
    load_materialized_curriculum,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.program import (
    build_program,
    compile_program,
)


def _tasks():
    return [
        task
        for manifest in load_materialized_curriculum().values()
        for task in manifest.tasks
    ]


def _live_bindings(task):
    geometry = {
        name: (200 + index * 70, 200 + index * 40)
        for index, name in enumerate(task.geometry_contract["required_targets"])
    }
    return geometry, (50, 50)


def test_gold_and_near_miss_programs_compile_for_both_transports() -> None:
    for task in _tasks():
        for near_miss in (False, True):
            program = build_program(task, near_miss=near_miss)
            assert len(program.turns) <= max(task.budgets["primitive_actions"].values())
            geometry, initial_cursor = _live_bindings(task)
            for action_schema in ACTION_SCHEMAS:
                compiled = compile_program(
                    task,
                    action_schema,
                    geometry=geometry,
                    initial_cursor=initial_cursor,
                    near_miss=near_miss,
                )
                assert len(compiled) == len(program.turns)


def test_compact_program_preserves_signed_scroll_drag_and_unicode_type() -> None:
    tasks = _tasks()
    chrome = [task for task in tasks if task.app == "chrome"]
    assert {
        int(
            compile_program(
                task,
                "compact_raw_phaseb_v1",
                geometry=_live_bindings(task)[0],
                initial_cursor=_live_bindings(task)[1],
            )[1].split()[2]
        )
        for task in chrome
    } == {-6, 6}

    files = next(task for task in tasks if task.app == "files")
    file_actions = compile_program(
        files,
        "compact_raw_phaseb_v1",
        geometry=_live_bindings(files)[0],
        initial_cursor=_live_bindings(files)[1],
    )
    press = next(index for index, action in enumerate(file_actions) if "+LMB" in action and "-LMB" not in action)
    release = next(index for index, action in enumerate(file_actions) if "-LMB" in action and "+LMB" not in action)
    assert release - press == 2

    vscode = next(task for task in tasks if task.app == "vscode")
    assert any(
        "Zürich μ" in action
        for action in compile_program(
            vscode,
            "compact_raw_phaseb_v1",
            geometry=_live_bindings(vscode)[0],
            initial_cursor=_live_bindings(vscode)[1],
        )
    )


def test_programs_finish_with_zero_held_inputs() -> None:
    # build_program validates the cross-turn hold/release state and raises if any
    # mouse input remains held. Both gold and near-miss must satisfy it.
    for task in _tasks():
        build_program(task, near_miss=False)
        build_program(task, near_miss=True)
