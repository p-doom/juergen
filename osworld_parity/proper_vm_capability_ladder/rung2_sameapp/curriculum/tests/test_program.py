from __future__ import annotations

from dataclasses import replace
import time

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
    compile_semantic_step,
    record_executed_segment,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.runtime import (
    RuntimeEvidenceLedger,
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
    ledger = RuntimeEvidenceLedger(setup_commit="a" * 40, reset_provider="test")
    values = []
    for index in range(2):
        current = replace(probe, state=dict(probe.state), geometry=dict(probe.geometry))
        started = time.monotonic_ns()
        current = ledger.issue_reset_probe(
            task,
            current,
            reset_started_monotonic_ns=started,
            probe_completed_monotonic_ns=time.monotonic_ns(),
            transport_endpoint=f"test://reset/{index}",
        )
        values.append(current)
    return bind_repeated_runtime_probes(task, tuple(values), ledger=ledger), ledger


def _compile_all(task, action_schema, *, near_miss=False):
    binding, ledger = _binding(task)
    segments = []
    for step in range(1, task.semantic_step_count + 1):
        segment = compile_semantic_step(
            task,
            action_schema,
            binding=binding,
            semantic_step_index=step,
            near_miss=near_miss,
        )
        segments.append(segment)
        if task.app == "chrome" and step == 2:
            started = time.monotonic_ns()
            receipt = record_executed_segment(
                segment,
                tuple(
                    ({"executor_dispatch_status": "ok", "atomic_state": {"ok": True}},)
                    for _ in segment.actions
                ),
                execution_started_monotonic_ns=started,
                execution_completed_monotonic_ns=time.monotonic_ns(),
            )
            state = dict(binding.initial_probe.state)
            delta = int(task.params["minimum_scroll_delta"])
            state["scroll_y"] += -delta if task.params["scroll_direction"] == "up" else delta
            action_started = receipt.execution_started_monotonic_ns
            action_completed = receipt.execution_completed_monotonic_ns
            probe_started = time.monotonic_ns()
            refreshed = replace(
                binding.initial_probe,
                state=state,
                geometry=dict(binding.initial_probe.geometry),
                initial_cursor=binding.initial_probe.geometry["scroll_surface"],
                reset_cycle_evidence=None,
            )
            refreshed = ledger.issue_refresh_probe(
                task,
                binding,
                refreshed,
                completed_step=2,
                executed_segment_sha256=receipt.executed_receipt_sha256,
                action_started_monotonic_ns=action_started,
                action_completed_monotonic_ns=action_completed,
                probe_started_monotonic_ns=probe_started,
                probe_completed_monotonic_ns=time.monotonic_ns(),
            )
            binding = refresh_binding_after_step(
                task,
                binding,
                completed_step=2,
                probe=refreshed,
                executed_segment_sha256=receipt.executed_receipt_sha256,
                ledger=ledger,
            )
    return segments


def _actions(segments):
    return [action for segment in segments for action in segment.actions]


def test_gold_and_near_miss_programs_compile_for_both_transports() -> None:
    for task in _tasks():
        for near_miss in (False, True):
            program = build_program(task, near_miss=near_miss)
            assert len(program.turns) <= max(
                task.budget_contract["primitive_action_caps"].values()
            )
            for action_schema in ACTION_SCHEMAS:
                compiled = _compile_all(task, action_schema, near_miss=near_miss)
                assert sum(item.resolved_primitive_actions for item in compiled) == len(program.turns)
                assert all(segment.resolved_budget_sha256 for segment in compiled)
                assert all(segment.binding_sha256 for segment in compiled)


def test_compact_program_preserves_signed_scroll_drag_and_unicode_type() -> None:
    tasks = _tasks()
    chrome = [task for task in tasks if task.app == "chrome"]
    assert {
        int(_actions(_compile_all(task, "compact_raw_phaseb_v1"))[1].split()[2])
        for task in chrome
    } == {-6, 6}

    files = next(task for task in tasks if task.app == "files")
    file_actions = _actions(
        _compile_all(files, "compact_raw_phaseb_v1")
    )
    press = next(index for index, action in enumerate(file_actions) if "+LMB" in action and "-LMB" not in action)
    release = next(index for index, action in enumerate(file_actions) if "-LMB" in action and "+LMB" not in action)
    assert release - press == 2

    vscode = next(task for task in tasks if task.app == "vscode")
    assert any(
        "Zürich μ" in action
        for action in _actions(
            _compile_all(vscode, "compact_raw_phaseb_v1")
        )
    )


def test_compact_cursor_already_at_target_omits_move_and_resolves_lower_budget() -> None:
    task = next(task for task in _tasks() if task.app == "writer")
    moving_probe = _probe(task)
    target = moving_probe.geometry["editor"]
    stationary_probe = replace(moving_probe, initial_cursor=target)
    moving_binding, _ = _binding(task)
    # Replace the stationary binding's live probes through a new attributed cycle.
    stationary_probe = replace(moving_probe, initial_cursor=target)
    ledger = RuntimeEvidenceLedger(setup_commit="a" * 40, reset_provider="test")
    stationary_values = []
    for index in range(2):
        started = time.monotonic_ns()
        value = ledger.issue_reset_probe(
            task,
            replace(stationary_probe, state=dict(stationary_probe.state), geometry=dict(stationary_probe.geometry)),
            reset_started_monotonic_ns=started,
            probe_completed_monotonic_ns=time.monotonic_ns(),
            transport_endpoint=f"test://stationary/{index}",
        )
        stationary_values.append(value)
    stationary_binding = bind_repeated_runtime_probes(task, tuple(stationary_values), ledger=ledger)
    moving = compile_semantic_step(
        task,
        "compact_raw_phaseb_v1",
        binding=moving_binding,
        semantic_step_index=1,
    )
    stationary = compile_semantic_step(
        task,
        "compact_raw_phaseb_v1",
        binding=stationary_binding,
        semantic_step_index=1,
    )
    assert stationary.actions[0].startswith("0 0 0;")
    assert stationary.resolved_primitive_events == moving.resolved_primitive_events - 1
    assert stationary.resolved_budget_sha256 != moving.resolved_budget_sha256


def test_chrome_later_target_requires_and_uses_post_scroll_probe() -> None:
    task = next(task for task in _tasks() if task.app == "chrome")
    initial = _probe(task)
    binding, ledger = _binding(task)
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
    refreshed_state = dict(binding.initial_probe.state)
    delta = int(task.params["minimum_scroll_delta"])
    refreshed_state["scroll_y"] += (
        -delta if task.params["scroll_direction"] == "up" else delta
    )
    refreshed = replace(
        initial,
        state=refreshed_state,
        geometry=changed_geometry,
        initial_cursor=refreshed_cursor,
    )
    scroll_segment = compile_semantic_step(
        task,
        "compact_raw_phaseb_v1",
        binding=binding,
        semantic_step_index=2,
    )
    started = time.monotonic_ns()
    receipt = record_executed_segment(
        scroll_segment,
        (({"executor_dispatch_status": "ok", "atomic_state": {"ok": True}},),),
        execution_started_monotonic_ns=started,
        execution_completed_monotonic_ns=time.monotonic_ns(),
    )
    refreshed = replace(refreshed, reset_cycle_evidence=None)
    probe_started = time.monotonic_ns()
    refreshed = ledger.issue_refresh_probe(
        task,
        binding,
        refreshed,
        completed_step=2,
        executed_segment_sha256=receipt.executed_receipt_sha256,
        action_started_monotonic_ns=receipt.execution_started_monotonic_ns,
        action_completed_monotonic_ns=receipt.execution_completed_monotonic_ns,
        probe_started_monotonic_ns=probe_started,
        probe_completed_monotonic_ns=time.monotonic_ns(),
    )
    binding = refresh_binding_after_step(
        task,
        binding,
        completed_step=2,
        probe=refreshed,
        executed_segment_sha256=receipt.executed_receipt_sha256,
        ledger=ledger,
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
