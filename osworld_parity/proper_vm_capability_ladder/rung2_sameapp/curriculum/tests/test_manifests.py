from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.actions import (
    ACTION_SCHEMAS,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.manifests import (
    ROOT,
    load_family_commitments,
    load_manifest,
    load_materialized_curriculum,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.schema import (
    APPS,
    EXCLUSIONS,
    PRIMARY_CAPABILITIES,
    CurriculumSchemaError,
)


def test_only_train_and_development_are_materialized() -> None:
    assert {path.name for path in ROOT.glob("*.json")} == {
        "train.json",
        "development.json",
    }
    with pytest.raises(CurriculumSchemaError, match="not materialized"):
        load_manifest("sealed_eval", root=Path("/path/that/must/not/be/read"))


def test_family_registry_has_commitments_but_zero_sealed_inputs() -> None:
    registry = load_family_commitments()
    assert registry["held_inputs_present"] is False
    assert registry["split_design"]["development_claim"] == "within_family_only"
    for family in registry["families"]:
        assert set(family["within_family_split_commitments"]) == {
            "train",
            "development",
        }
        assert family["eligible_for_future_family_holdout"] is False
    for commitment in registry["future_family_splits"].values():
        assert commitment["materialized"] is False
        assert commitment["inputs_present"] is False
        assert "disjoint" in commitment["family_id_assignment"]
        assert "seed" not in commitment


def test_materialized_tasks_cover_phase_b_and_edge_labels() -> None:
    manifests = load_materialized_curriculum()
    tasks = [task for manifest in manifests.values() for task in manifest.tasks]
    assert len(tasks) == 10
    assert all(task.natural_multistep is True for task in tasks)
    assert all(2 <= task.semantic_step_count <= 4 for task in tasks)
    assert all(tuple(task.exclusions) == EXCLUSIONS for task in tasks)
    assert all(task.verifier["fresh_process"] is True for task in tasks)
    assert all(task.verifier["entrypoint"] == "main" for task in tasks)
    assert all(task.verifier["state_extractor_entrypoint"] == "extract_state" for task in tasks)
    assert all(task.snapshot["id"] == "osworld_ready" for task in tasks)
    assert all(task.geometry_contract["source"] == "live_probe" for task in tasks)
    assert all(task.initial_cursor["source"] == "live_probe" for task in tasks)
    assert all("geometry" not in task.params for task in tasks)
    assert all(
        set(task.budgets["primitive_actions"]) == set(ACTION_SCHEMAS)
        and set(task.budgets["primitive_events"]) == set(ACTION_SCHEMAS)
        for task in tasks
    )
    assert all(task.fixture_sha256 for task in tasks)
    assert {task.app for task in tasks} == set(APPS)
    capabilities = {
        capability
        for task in tasks
        for capability in task.coverage["primary_capabilities"]
    }
    assert capabilities == set(PRIMARY_CAPABILITIES)
    assert {
        sign for task in tasks for sign in task.coverage["signed_vertical_scroll"]
    } == {"up", "down"}
    assert {case for task in tasks for case in task.coverage["edge_cases"]} == {
        "unicode",
        "file_drag",
        "ctrl_s",
    }


def test_task_schema_has_action_independent_correlatable_history() -> None:
    schema_path = Path(__file__).parents[1] / "semantic_task.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "action_schema" not in schema["properties"]
    for task in load_manifest("development").tasks:
        task.verify()
        row = asdict(task)
        assert "action_schema" not in row
        assert len(task.gold_cursor_history) == task.semantic_step_count
        for step, milestone in zip(
            task.semantic_steps, task.gold_cursor_history, strict=True
        ):
            assert (step.step_id, step.target_ref) == (
                milestone.step_id,
                milestone.target_ref,
            )
            assert milestone.cursor_before_ref
            assert milestone.cursor_after_ref
