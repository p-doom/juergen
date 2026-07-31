from __future__ import annotations

import json

import pytest

from osworld_parity.mixed_action_short_vm.manifest import (
    DEVELOPMENT_MANIFEST,
    SEALED_EVALUATION_MANIFEST,
    TRAIN_MANIFEST,
    ManifestError,
    SEQUENCES,
    load_manifest,
    materialize_tasks,
)


def test_train_development_and_sealed_metadata_are_deterministic_and_disjoint() -> None:
    train_manifest = load_manifest(TRAIN_MANIFEST)
    development_manifest = load_manifest(DEVELOPMENT_MANIFEST)
    sealed = load_manifest(SEALED_EVALUATION_MANIFEST)

    train_a = materialize_tasks(train_manifest)
    train_b = materialize_tasks(load_manifest(TRAIN_MANIFEST))
    development = materialize_tasks(development_manifest)
    assert train_a == train_b
    assert len(train_a) == 12
    assert len(development) == 6
    assert len(sealed.cells) == 8

    assert {task.task_id for task in train_a}.isdisjoint(
        task.task_id for task in development
    )
    assert {task.parameter_seed for task in train_a}.isdisjoint(
        task.parameter_seed for task in development
    )
    assert {task.task_sha256 for task in train_a}.isdisjoint(
        task.task_sha256 for task in development
    )
    assert all(2 <= task.semantic_step_count <= 4 for task in (*train_a, *development))
    assert set().union(*(set(task.steps) for task in train_a)) == {
        "focus",
        "coalesced_type",
        "scroll",
        "click",
        "drag",
    }


def test_sealed_evaluation_is_commitment_metadata_only_and_fail_closed() -> None:
    text = SEALED_EVALUATION_MANIFEST.read_text(encoding="utf-8")
    raw = json.loads(text)
    assert raw["materialized"] is False
    assert raw["sealing"]["status"] == "reserved_not_materialized"
    assert all("seed" not in cell for cell in raw["cells"])
    assert "target_text" not in text
    assert "instruction" not in text
    sealed = load_manifest(SEALED_EVALUATION_MANIFEST)
    with pytest.raises(ManifestError, match="cannot be materialized"):
        materialize_tasks(sealed)


def test_sequence_inventory_is_exactly_two_to_four_semantic_steps() -> None:
    assert SEQUENCES
    assert {len(sequence) for sequence in SEQUENCES.values()} == {2, 3, 4}
    assert all(2 <= len(sequence) <= 4 for sequence in SEQUENCES.values())
