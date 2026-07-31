from __future__ import annotations

from dataclasses import replace

import pytest

from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.actions import (
    ACTION_SCHEMAS,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.manifests import (
    load_materialized_curriculum,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.oracle import (
    initial_state,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.program import (
    build_program,
    compile_program,
    compile_semantic_step,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.runtime import (
    RuntimeProbe,
    RuntimeProbeError,
    bind_repeated_runtime_probes,
    refresh_binding_after_step,
)


def _tasks():
    return [
        task
        for manifest in load_materialized_curriculum().values()
        for task in manifest.tasks
    ]


def _probe(task, *, initial_cursor=(50, 50)) -> RuntimeProbe:
    state = initial_state(task)
    state.pop("held_inputs")
    geometry = {
        name: (200 + index * 70, 200 + index * 40)
        for index, name in enumerate(task.geometry_contract["required_targets"])
    }
    return RuntimeProbe(
        state=state,
        geometry=geometry,
        initial_cursor=initial_cursor,
        screen_size=(1400, 900),
        geometry_probe_version=task.geometry_contract["probe_version"],
        state_probe_version=task.geometry_contract["state_probe_version"],
    )


def _binding(task):
    probe = _probe(task)
    binding = bind_repeated_runtime_probes(task, (probe, probe))
    if task.app == "chrome":
        binding = refresh_binding_after_step(
            task,
            binding,
            completed_step=2,
            probe=replace(
                probe,
                state={"post_scroll": True},
                initial_cursor=probe.geometry["scroll_surface"],
            ),
        )
    return binding


def _actions(compiled):
    return [action for segment in compiled.segments for action in segment.actions]


def test_gold_and_near_miss_programs_compile_for_both_transports() -> None:
    for task in _tasks():
        for near_miss in (False, True):
            program = build_program(task, near_miss=near_miss)
            assert len(program.turns) <= max(
                task.budget_contract["primitive_action_caps"].values()
            )
            for action_schema in ACTION_SCHEMAS:
                compiled = compile_program(
                    task, action_schema, binding=_binding(task), near_miss=near_miss
                )
                assert compiled.resolved_primitive_actions == len(program.turns)
                assert compiled.resolved_budget_sha256
                assert all(segment.binding_sha256 for segment in compiled.segments)


def test_compact_program_preserves_signed_scroll_drag_and_unicode_type() -> None:
    tasks = _tasks()
    chrome = [task for task in tasks if task.app == "chrome"]
    assert {
        int(_actions(compile_program(task, "compact_raw_phaseb_v1", binding=_binding(task)))[1].split()[2])
        for task in chrome
    } == {-6, 6}

    files = next(task for task in tasks if task.app == "files")
    file_actions = _actions(
        compile_program(files, "compact_raw_phaseb_v1", binding=_binding(files))
    )
    press = next(index for index, action in enumerate(file_actions) if "+LMB" in action and "-LMB" not in action)
    release = next(index for index, action in enumerate(file_actions) if "-LMB" in action and "+LMB" not in action)
    assert release - press == 2

    vscode = next(task for task in tasks if task.app == "vscode")
    assert any(
        "Zürich μ" in action
        for action in _actions(
            compile_program(vscode, "compact_raw_phaseb_v1", binding=_binding(vscode))
        )
    )


def test_compact_cursor_already_at_target_omits_move_and_resolves_lower_budget() -> None:
    task = next(task for task in _tasks() if task.app == "writer")
    moving_probe = _probe(task)
    target = moving_probe.geometry["editor"]
    stationary_probe = replace(moving_probe, initial_cursor=target)
    moving = compile_semantic_step(
        task,
        "compact_raw_phaseb_v1",
        binding=bind_repeated_runtime_probes(task, (moving_probe, moving_probe)),
        semantic_step_index=1,
    )
    stationary = compile_semantic_step(
        task,
        "compact_raw_phaseb_v1",
        binding=bind_repeated_runtime_probes(task, (stationary_probe, stationary_probe)),
        semantic_step_index=1,
    )
    assert stationary.actions[0].startswith("0 0 0;")
    assert stationary.resolved_primitive_events == moving.resolved_primitive_events - 1
    assert stationary.resolved_budget_sha256 != moving.resolved_budget_sha256


def test_chrome_later_target_requires_and_uses_post_scroll_probe() -> None:
    task = next(task for task in _tasks() if task.app == "chrome")
    initial = _probe(task)
    binding = bind_repeated_runtime_probes(task, (initial, initial))
    with pytest.raises(RuntimeProbeError, match="post-step-2 refresh"):
        compile_semantic_step(
            task,
            "compact_raw_phaseb_v1",
            binding=binding,
            semantic_step_index=3,
        )
    changed_geometry = dict(initial.geometry)
    changed_geometry["toggle"] = (930, 310)
    refreshed_cursor = (610, 580)
    refreshed = replace(
        initial,
        state={"post_scroll": True},
        geometry=changed_geometry,
        initial_cursor=refreshed_cursor,
    )
    binding = refresh_binding_after_step(
        task, binding, completed_step=2, probe=refreshed
    )
    compact = compile_semantic_step(
        task,
        "compact_raw_phaseb_v1",
        binding=binding,
        semantic_step_index=3,
    )
    assert compact.actions[0].startswith("320 -270 0;")
    native = compile_semantic_step(
        task,
        "native_absolute_sequence_v1",
        binding=binding,
        semantic_step_index=3,
    )
    assert native.actions[0]["operations"][0]["coordinate"] == [930, 310]


def test_programs_finish_with_zero_held_inputs() -> None:
    for task in _tasks():
        build_program(task, near_miss=False)
        build_program(task, near_miss=True)
