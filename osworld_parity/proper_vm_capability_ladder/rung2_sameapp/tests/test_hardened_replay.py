from __future__ import annotations

import inspect
import hashlib
import time
from dataclasses import replace

import pytest

from osworld_parity.proper_vm_capability_ladder.rung2_sameapp import replay
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.manifests import (
    load_manifest,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.oracle import (
    initial_state,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.program import (
    aggregate_executed_segments,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.runtime import (
    RuntimeEvidenceLedger,
    RuntimeProbe,
    RuntimeProbeError,
    bind_repeated_runtime_probes,
)


class _Audit:
    held_buttons: set[str] = set()
    held_keys: set[str] = set()


class _Transport:
    audit = _Audit()


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


def _binding(task):
    ledger = RuntimeEvidenceLedger(setup_commit="b" * 40, reset_provider="test")
    base = _probe(task)
    values = []
    for index in range(2):
        started = time.monotonic_ns()
        values.append(
            ledger.issue_reset_probe(
                task,
                replace(base, state=dict(base.state), geometry=dict(base.geometry)),
                reset_started_monotonic_ns=started,
                probe_completed_monotonic_ns=time.monotonic_ns(),
                transport_endpoint=f"test://generation/{index}",
            )
        )
    return bind_repeated_runtime_probes(task, tuple(values), ledger=ledger), ledger


def _ok_dispatch(*_args, **_kwargs):
    return ({"executor_dispatch_status": "ok", "atomic_state": {"ok": True}},)


def test_production_replay_has_no_direct_symbolic_compiler_bypass() -> None:
    source = inspect.getsource(replay)
    assert "compile_native" not in source
    assert "compile_compact" not in source
    assert "build_trajectory" not in source
    assert "scripted_state" not in source
    assert "_export_guest_artifact" in source
    assert "verify_fixture_contract" in source
    assert "_load_setup_dependency" in source
    with pytest.raises(RuntimeError, match="live VM bindings"):
        replay.run_build_replay("development")


def test_production_setup_dependency_requires_pinned_raw_sha_and_artifact_id(
    tmp_path, monkeypatch
) -> None:
    manifest = load_manifest("development")
    path = tmp_path / "task_setup_validation.json"
    path.write_bytes(b"immutable setup evidence")
    raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        replay,
        "load_task_setup_validation",
        lambda candidate, consumed_manifest: {
            "artifact_id": "setup-artifact-1",
            "setup_commit": "b" * 40,
            "vm_snapshot_id": "osworld_ready",
        },
    )
    result = replay._load_setup_dependency(
        manifest,
        path=path,
        expected_artifact_id="setup-artifact-1",
        expected_raw_sha256=raw_sha,
    )
    assert result["artifact_id"] == "setup-artifact-1"
    with pytest.raises(ValueError, match="raw SHA mismatch"):
        replay._load_setup_dependency(
            manifest,
            path=path,
            expected_artifact_id="setup-artifact-1",
            expected_raw_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="artifact ID mismatch"):
        replay._load_setup_dependency(
            manifest,
            path=path,
            expected_artifact_id="wrong",
            expected_raw_sha256=raw_sha,
        )


def test_chrome_production_path_keeps_a_a_b_receipts_and_signed_refresh(
    monkeypatch,
) -> None:
    task = next(task for task in load_manifest("development").tasks if task.app == "chrome")
    binding, ledger = _binding(task)
    initial_binding_sha = binding.binding_sha256
    changed_state = dict(binding.initial_probe.state)
    minimum = int(task.params["minimum_scroll_delta"])
    changed_state["scroll_y"] += (
        -minimum if task.params["scroll_direction"] == "up" else minimum
    )
    changed_geometry = dict(binding.initial_probe.geometry)
    changed_geometry["toggle"] = (990, 310)
    refreshed = replace(
        binding.initial_probe,
        state=changed_state,
        geometry=changed_geometry,
        initial_cursor=(610, 580),
        reset_cycle_evidence=None,
    )
    monkeypatch.setattr(replay, "_dispatch_compiled_action", _ok_dispatch)
    monkeypatch.setattr(replay, "probe_runtime", lambda *_args, **_kwargs: refreshed)
    monkeypatch.setattr(replay.time, "sleep", lambda _seconds: None)
    journal, aggregate = replay._execute_bound_trajectory(
        _Transport(),
        task,
        binding,
        ledger,
        action_schema="compact_raw_phaseb_v1",
        near_miss=False,
    )
    assert [row["executed_receipt"]["binding_revision"] for row in journal] == [1, 1, 2]
    reset_cycles = journal[0]["binding_receipt"]["reset_cycles"]
    assert len({row["reset_id"] for row in reset_cycles}) == 2
    assert len({row["generation_id"] for row in reset_cycles}) == 2
    assert all(row["probe_sha256"] for row in reset_cycles)
    assert journal[0]["binding_receipt"]["initial_geometry_sha256"]
    assert [row["executed_receipt"]["binding_sha256"] for row in journal[:2]] == [
        initial_binding_sha,
        initial_binding_sha,
    ]
    assert journal[2]["executed_receipt"]["binding_sha256"] != initial_binding_sha
    transition = journal[1]["post_scroll_refresh"]
    assert transition["completed_step"] == 2
    assert transition["observed_scroll_delta"] == -minimum
    assert transition["reset_generation_id"] == binding.initial_probe.reset_cycle_evidence.generation_id
    assert transition["executed_segment_sha256"] == journal[1]["executed_receipt"]["executed_receipt_sha256"]
    binding_transition = journal[1]["refreshed_binding_receipt"]["refresh_transitions"][0]
    assert binding_transition["pre_binding_revision"] == 1
    assert binding_transition["post_binding_revision"] == 2
    assert binding_transition["pre_binding_sha256"] == initial_binding_sha
    assert binding_transition["post_binding_sha256"] == journal[2]["binding_sha256"]
    assert binding_transition["transition_receipt_sha256"]
    assert [item.executed_receipt_sha256 for item in aggregate.segments] == [
        row["executed_receipt"]["executed_receipt_sha256"] for row in journal
    ]
    assert aggregate.resolved_primitive_events <= task.budget_contract[
        "primitive_event_caps"
    ]["compact_raw_phaseb_v1"]

    newest = aggregate.segments[-1]
    regenerated_first = replace(
        aggregate.segments[0],
        binding_revision=newest.binding_revision,
        binding_sha256=newest.binding_sha256,
    )
    with pytest.raises(ValueError, match="Chrome binding transition mismatch"):
        aggregate_executed_segments(
            task,
            "compact_raw_phaseb_v1",
            segments=(regenerated_first,) + aggregate.segments[1:],
        )


def test_chrome_production_path_rejects_stale_pre_scroll_probe(monkeypatch) -> None:
    task = next(task for task in load_manifest("development").tasks if task.app == "chrome")
    binding, ledger = _binding(task)
    stale = replace(
        binding.initial_probe,
        state=dict(binding.initial_probe.state),
        geometry=dict(binding.initial_probe.geometry),
        reset_cycle_evidence=None,
    )
    monkeypatch.setattr(replay, "_dispatch_compiled_action", _ok_dispatch)
    monkeypatch.setattr(replay, "probe_runtime", lambda *_args, **_kwargs: stale)
    monkeypatch.setattr(replay.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeProbeError, match="signed scroll delta"):
        replay._execute_bound_trajectory(
            _Transport(),
            task,
            binding,
            ledger,
            action_schema="compact_raw_phaseb_v1",
            near_miss=False,
        )


def test_production_path_rejects_executor_failure_instead_of_receipting(monkeypatch) -> None:
    task = next(task for task in load_manifest("development").tasks if task.app == "writer")
    binding, ledger = _binding(task)
    monkeypatch.setattr(
        replay,
        "_dispatch_compiled_action",
        lambda *_args, **_kwargs: (
            {"executor_dispatch_status": "error", "atomic_state": {"ok": False}},
        ),
    )
    with pytest.raises(ValueError, match="dispatch did not complete"):
        replay._execute_bound_trajectory(
            _Transport(),
            task,
            binding,
            ledger,
            action_schema="native_absolute_sequence_v1",
            near_miss=False,
        )
