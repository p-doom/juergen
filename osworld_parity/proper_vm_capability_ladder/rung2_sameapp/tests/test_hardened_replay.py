from __future__ import annotations

import inspect
import hashlib
import time
import uuid
from dataclasses import replace

import pytest

from osworld_parity.proper_vm_capability_ladder.rung1.transport import RecordingTransport
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp import replay
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.manifests import (
    load_manifest,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.oracle import (
    initial_state,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.program import (
    aggregate_executed_segments,
    compile_semantic_step,
    record_executed_segment,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.runtime import (
    RuntimeProbe,
    RuntimeProbeError,
    bind_repeated_runtime_probes,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.tests.evidence import (
    attribute_new_observation,
    make_ledger,
)


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
    ledger, attestor = make_ledger()
    base = _probe(task)
    values = []
    for index in range(2):
        values.append(
            attribute_new_observation(
                task,
                ledger,
                attestor,
                replace(base, state=dict(base.state), geometry=dict(base.geometry)),
                endpoint=f"test://generation/{index}",
            )
        )
    return bind_repeated_runtime_probes(task, tuple(values), ledger=ledger), ledger


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


@pytest.mark.parametrize("failed", ("reset_rejected", "near_miss_rejected", "gold_passed"))
def test_production_rejects_any_oracle_error_contract(failed: str) -> None:
    task = load_manifest("development").tasks[0]
    contract = {
        "reset_rejected": True,
        "near_miss_rejected": True,
        "gold_passed": True,
        "reset_reproducible": True,
        "fresh_process_final_oracle": True,
        "zero_held_inputs": True,
    }
    contract[failed] = False
    with pytest.raises(RuntimeError, match="fixture contract failed"):
        replay._require_fixture_contract(task, "native_absolute_control", contract)


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
        observation_id=uuid.uuid4().hex,
        observed_monotonic_ns=time.monotonic_ns(),
    )
    monkeypatch.setattr(replay, "probe_runtime", lambda *_args, **_kwargs: refreshed)
    monkeypatch.setattr(replay.time, "sleep", lambda _seconds: None)
    journal, aggregate = replay._execute_bound_trajectory(
        RecordingTransport(cursor=(40, 50), screen=(1400, 900)),
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
        observation_id=uuid.uuid4().hex,
        observed_monotonic_ns=time.monotonic_ns(),
    )
    monkeypatch.setattr(replay, "probe_runtime", lambda *_args, **_kwargs: stale)
    monkeypatch.setattr(replay.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeProbeError, match="signed scroll delta"):
        replay._execute_bound_trajectory(
            RecordingTransport(cursor=(40, 50), screen=(1400, 900)),
            task,
            binding,
            ledger,
            action_schema="compact_raw_phaseb_v1",
            near_miss=False,
        )


def test_production_path_rejects_executor_failure_instead_of_receipting(monkeypatch) -> None:
    task = next(task for task in load_manifest("development").tasks if task.app == "writer")
    binding, ledger = _binding(task)

    def failed_dispatch(_transport, _schema, action):
        return tuple(
            replay._seal_dispatch_result(
                {
                    "adapter": "native_absolute_control",
                    "parse_status": "ok",
                    "executor_dispatch_status": "error",
                    "action_class": "click",
                    "operations": (),
                    "atomic_state": {"ok": False, "operations": ()},
                },
                compiled_payload=operation,
                compiled_operation_index=index,
                cursor_before=(40, 50),
                cursor_after=(40, 50),
            )
            for index, operation in enumerate(action["operations"])
        )

    monkeypatch.setattr(
        replay,
        "_dispatch_compiled_action",
        failed_dispatch,
    )
    with pytest.raises(ValueError, match="dispatch did not complete"):
        replay._execute_bound_trajectory(
            RecordingTransport(cursor=(40, 50), screen=(1400, 900)),
            task,
            binding,
            ledger,
            action_schema="native_absolute_sequence_v1",
            near_miss=False,
        )


def test_native_multi_operation_action_requires_exact_dispatch_coverage() -> None:
    task = next(task for task in load_manifest("development").tasks if task.app == "writer")
    binding, ledger = _binding(task)
    segment = compile_semantic_step(
        task,
        "native_absolute_sequence_v1",
        binding=binding,
        semantic_step_index=1,
    )
    transport = RecordingTransport(cursor=(40, 50), screen=(1400, 900))
    complete = replay._dispatch_compiled_action(
        transport, "native_absolute_sequence_v1", segment.actions[0]
    )
    assert len(segment.actions[0]["operations"]) == 3
    assert len(complete) == 3
    started = time.monotonic_ns()
    with pytest.raises(ValueError, match="cardinality does not cover"):
        record_executed_segment(
            segment,
            (complete[:1],),
            execution_started_monotonic_ns=started,
            execution_completed_monotonic_ns=time.monotonic_ns(),
        )
    receipt = record_executed_segment(
        segment,
        (complete,),
        execution_started_monotonic_ns=started,
        execution_completed_monotonic_ns=time.monotonic_ns(),
    )
    forged_segment = replace(
        segment,
        actions=({"operations": segment.actions[0]["operations"][:1]},),
    )
    with pytest.raises(RuntimeProbeError, match="declared semantic trajectory"):
        ledger.record_executed_segment(
            task,
            binding,
            forged_segment,
            (complete[:1],),
            receipt,
            near_miss=False,
        )


def test_chrome_refresh_rejects_opaque_or_unrecorded_receipt() -> None:
    task = next(task for task in load_manifest("development").tasks if task.app == "chrome")
    binding, ledger = _binding(task)
    state = dict(binding.initial_probe.state)
    state["scroll_y"] -= int(task.params["minimum_scroll_delta"])
    probe = replace(
        binding.initial_probe,
        state=state,
        reset_cycle_evidence=None,
        observation_id=uuid.uuid4().hex,
        observed_monotonic_ns=time.monotonic_ns(),
    )
    started = time.monotonic_ns()
    completed = time.monotonic_ns()
    probe_started = time.monotonic_ns()
    with pytest.raises(RuntimeProbeError, match="requires ExecutedSegmentReceipt"):
        ledger.issue_refresh_probe(
            task,
            binding,
            probe,
            completed_step=2,
            executed_segment="f" * 64,
            action_started_monotonic_ns=started,
            action_completed_monotonic_ns=completed,
            probe_started_monotonic_ns=probe_started,
            probe_completed_monotonic_ns=time.monotonic_ns(),
        )
    segment = compile_semantic_step(
        task,
        "compact_raw_phaseb_v1",
        binding=binding,
        semantic_step_index=2,
    )
    transport = RecordingTransport(cursor=(40, 50), screen=(1400, 900))
    action_started = time.monotonic_ns()
    dispatches = tuple(
        replay._dispatch_compiled_action(
            transport, "compact_raw_phaseb_v1", action
        )
        for action in segment.actions
    )
    action_completed = time.monotonic_ns()
    receipt = record_executed_segment(
        segment,
        dispatches,
        execution_started_monotonic_ns=action_started,
        execution_completed_monotonic_ns=action_completed,
    )
    second_probe_start = time.monotonic_ns()
    with pytest.raises(RuntimeProbeError, match="not ledger-recorded"):
        ledger.issue_refresh_probe(
            task,
            binding,
            probe,
            completed_step=2,
            executed_segment=receipt,
            action_started_monotonic_ns=action_started,
            action_completed_monotonic_ns=action_completed,
            probe_started_monotonic_ns=second_probe_start,
            probe_completed_monotonic_ns=time.monotonic_ns(),
        )
