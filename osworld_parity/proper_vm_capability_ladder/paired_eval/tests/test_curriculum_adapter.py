from __future__ import annotations

from dataclasses import asdict, make_dataclass
from types import SimpleNamespace
import time
from typing import Any

import pytest

from ..curriculum_adapter import (
    CURRICULUM_SUITE,
    adapt_curriculum_manifest,
    load_curriculum_evaluation_manifest,
)
from ..manifest import ManifestError
from ..receipts import validate_binding_receipt, validate_executed_segment
from ..verifier import FreshProcessTaskVerifier
from .helpers import evaluation_manifest, ready_marker, sealed_file, task_manifest, task_row


def _validated_curriculum_task():
    row = task_row()
    row.pop("suite")
    row["natural_multistep"] = True
    for step in row["semantic_steps"]:
        step.pop("value_ref", None)
        step["arguments"] = {}
    # Re-seal after adding exact curriculum-v1 fields.
    import hashlib

    from ..contracts import canonical_json

    unsigned = dict(row)
    unsigned.pop("fixture_sha256")
    row["fixture_sha256"] = hashlib.sha256(canonical_json(unsigned)).hexdigest()
    task_type = make_dataclass(
        "ValidatedCurriculumTask",
        [(key, Any) for key in row],
        frozen=True,
        namespace={"verify": lambda self: None},
    )
    return task_type(**row)


def test_exact_curriculum_adapter_joins_loader_validated_tasks(tmp_path) -> None:
    _, ready_sha = ready_marker(tmp_path / "EXECUTOR_READY.json")
    _, task_seal = task_manifest()
    evaluation_path, _ = sealed_file(
        tmp_path / "evaluation.json",
        evaluation_manifest(task_seal, ready_sha, attempts=8),
    )
    curriculum = SimpleNamespace(
        split="development",
        tasks=(_validated_curriculum_task(),),
        manifest_payload_sha256=task_seal,
    )
    manifest = adapt_curriculum_manifest(
        evaluation_path=evaluation_path,
        curriculum=curriculum,
    )
    assert manifest.task_suite == CURRICULUM_SUITE
    assert manifest.tasks[0].semantic_step_count == 2


def test_curriculum_loader_rejects_non_development_path_before_import(tmp_path) -> None:
    with pytest.raises(ManifestError, match="only explicit development.json"):
        load_curriculum_evaluation_manifest(
            tmp_path / "evaluation.json",
            tmp_path / "sealed_eval.json",
        )


def test_adapter_against_installed_semantic_curriculum_v1(tmp_path) -> None:
    curriculum_module = pytest.importorskip(
        "osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.manifests"
    )
    curriculum = curriculum_module.load_manifest("development")
    _, ready_sha = ready_marker(tmp_path / "EXECUTOR_READY.json")
    evaluation_path, _ = sealed_file(
        tmp_path / "evaluation.json",
        evaluation_manifest(
            curriculum.manifest_payload_sha256,
            ready_sha,
            attempts=8,
        ),
    )
    manifest = adapt_curriculum_manifest(
        evaluation_path=evaluation_path,
        curriculum=curriculum,
    )
    assert len(manifest.tasks) == 5
    assert {task.app for task in manifest.tasks} == {
        "writer",
        "calc",
        "files",
        "chrome",
        "vscode",
    }
    oracle_module = pytest.importorskip(
        "osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.oracle"
    )
    source_task = curriculum.tasks[0]
    task = manifest.task(source_task.task_id)
    final_state = oracle_module.scripted_state(source_task, near_miss=False)
    final = FreshProcessTaskVerifier().verify(
        task=task,
        state=final_state,
        expected_step_index=None,
        expected_target_ref=None,
        timeout_seconds=30,
    )
    assert final.status == "ok"
    assert final.task_solved is True

    geometry = {
        target: [100 + index, 200 + index]
        for index, target in enumerate(source_task.geometry_contract["required_targets"])
    }
    milestone = source_task.gold_cursor_history[0]
    resolved = (
        [50, 60]
        if milestone.cursor_after_ref == "runtime.initial_cursor"
        else geometry[milestone.cursor_after_ref.removeprefix("geometry.")]
    )
    semantic_state = {
        "task_id": source_task.task_id,
        "fixture_sha256": source_task.fixture_sha256,
        "held_inputs": [],
        "geometry": geometry,
        "initial_cursor": [50, 60],
        "cursor": resolved,
    }
    semantic = FreshProcessTaskVerifier().verify(
        task=task,
        state=semantic_state,
        expected_step_index=1,
        expected_target_ref=milestone.target_ref,
        timeout_seconds=30,
    )
    assert semantic.status == "ok"
    assert semantic.matched_target_ref == milestone.target_ref


def test_approved_curriculum_receipts_match_paired_validator() -> None:
    pytest.importorskip(
        "osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.program"
    )
    from osworld_parity.proper_vm_capability_ladder.rung1.transport import (
        RecordingTransport,
    )
    from osworld_parity.proper_vm_capability_ladder.rung2_sameapp import replay
    from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.manifests import (
        load_manifest,
    )
    from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.program import (
        compile_semantic_step,
        record_executed_segment,
    )
    from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.tests.test_hardened_replay import (
        _binding,
    )

    task = next(task for task in load_manifest("development").tasks if task.app == "writer")
    binding, _ = _binding(task)
    binding_receipt = binding.receipt()
    validate_binding_receipt(
        binding_receipt,
        task_id=task.task_id,
        fixture_sha256=task.fixture_sha256,
        snapshot_id="osworld_ready",
        setup_commit=binding.reset_probes[0].reset_cycle_evidence.setup_commit,
        require_fresh=True,
    )
    segment = compile_semantic_step(
        task,
        "native_absolute_sequence_v1",
        binding=binding,
        semantic_step_index=1,
    )
    transport = RecordingTransport(
        cursor=segment.expected_cursor_before,
        screen=(1400, 900),
    )
    started = time.monotonic_ns()
    dispatches = tuple(
        replay._dispatch_compiled_action(
            transport, "native_absolute_sequence_v1", action
        )
        for action in segment.actions
    )
    completed = time.monotonic_ns()
    executed = record_executed_segment(
        segment,
        dispatches,
        execution_started_monotonic_ns=started,
        execution_completed_monotonic_ns=completed,
    )
    validate_executed_segment(
        compiled_segment=asdict(segment),
        dispatches=dispatches,
        executed_receipt=asdict(executed),
        binding_receipt=binding_receipt,
        task_id=task.task_id,
        fixture_sha256=task.fixture_sha256,
        action_schema="native_absolute_sequence_v1",
        expected_semantic_step=1,
        expected_cursor_before=segment.expected_cursor_before,
    )
