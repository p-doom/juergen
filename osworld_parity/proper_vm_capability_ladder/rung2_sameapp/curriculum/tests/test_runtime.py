from __future__ import annotations

import os

import pytest

from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.manifests import (
    load_manifest,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.oracle import (
    as_sameapp_fixture,
    evaluate_in_fresh_process,
    initial_state,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.runtime import (
    RuntimeProbe,
    RuntimeProbeError,
    resolved_cursor_history,
    validate_repeated_runtime_probes,
    validate_runtime_probe,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.vm import _chrome_html


def _probe(task) -> RuntimeProbe:
    state = initial_state(task)
    state.pop("held_inputs")
    geometry = {
        name: (220 + index * 80, 180 + index * 60)
        for index, name in enumerate(task.geometry_contract["required_targets"])
    }
    return RuntimeProbe(
        state=state,
        geometry=geometry,
        initial_cursor=(40, 50),
        screen_size=(1400, 900),
        geometry_probe_version=task.geometry_contract["probe_version"],
        state_probe_version=task.geometry_contract["state_probe_version"],
    )


def test_chrome_up_setup_establishes_and_proves_nonzero_initial_scroll() -> None:
    task = next(
        task
        for task in load_manifest("development").tasks
        if task.app == "chrome"
    )
    assert task.params["scroll_direction"] == "up"
    assert task.params["initial_scroll_y"] == 720
    html = _chrome_html(as_sameapp_fixture(task))
    assert "scrollTo(0,720)" in html
    assert "#settings{margin-top:400px}" in html
    validate_runtime_probe(task, _probe(task))
    wrong = _probe(task)
    wrong.state["scroll_y"] = 0
    with pytest.raises(RuntimeProbeError, match="setup-state mismatch"):
        validate_runtime_probe(task, wrong)


def test_geometry_drift_across_exact_resets_is_rejected() -> None:
    task = load_manifest("development").tasks[0]
    first = _probe(task)
    second = _probe(task)
    name = next(iter(second.geometry))
    second.geometry[name] = (second.geometry[name][0] + 1, second.geometry[name][1])
    with pytest.raises(RuntimeProbeError, match="geometry drift"):
        validate_repeated_runtime_probes(task, first, second)


def test_runtime_resolves_cursor_refs_and_fresh_semantic_oracle() -> None:
    task = load_manifest("development").tasks[0]
    probe = _probe(task)
    history = resolved_cursor_history(task, probe)
    first = history[0]
    state = {
        "task_id": task.task_id,
        "fixture_sha256": task.fixture_sha256,
        "held_inputs": [],
        "initial_cursor": list(probe.initial_cursor),
        "geometry": {name: list(point) for name, point in probe.geometry.items()},
        "cursor": list(first["cursor_after"]),
        "app_state": probe.state,
    }
    result = evaluate_in_fresh_process(
        task,
        state,
        expected_step_index=first["step_id"],
        expected_target_ref=first["target_ref"],
    )
    assert result.oracle_pid != os.getpid()
    assert result.oracle_status == "ok"
    assert result.MOUSE_SOLVED is True
    assert result.semantic_step_index == first["step_id"]
    assert result.matched_target_ref == first["target_ref"]
    assert result.semantic_state_sha256
    wrong = evaluate_in_fresh_process(
        task,
        state,
        expected_step_index=first["step_id"],
        expected_target_ref="writer.unregistered_target",
    )
    assert wrong.oracle_status == "error"
    assert wrong.MOUSE_SOLVED is False
