import json

import pytest

from osworld_parity.proper_vm_capability_ladder.rung34_recovery.spec import (
    SealedEvaluationError,
    load_recovery_tasks,
    load_sealed_commitment,
)


def test_train_development_and_sealed_namespaces_are_disjoint():
    train = load_recovery_tasks("train")
    development = load_recovery_tasks("development")
    sealed = load_sealed_commitment()
    assert len(train) == 18
    assert len(development) == 6
    assert sealed["task_count"] == 12
    assert all(task.id.startswith("r34-train-") for task in train)
    assert all(task.id.startswith("r34-development-") for task in development)
    assert sealed["namespace"] == "r34-sealed-"
    assert {task.id for task in train}.isdisjoint(task.id for task in development)
    train_cells = {(task.fixture.template, task.fixture.parameter_seed) for task in train}
    dev_cells = {(task.fixture.template, task.fixture.parameter_seed) for task in development}
    assert train_cells.isdisjoint(dev_cells)


def test_sealed_evaluation_fails_before_any_path_read(monkeypatch):
    def forbidden_read(*args, **kwargs):
        raise AssertionError("a path was read before sealed-eval rejection")

    monkeypatch.setattr("pathlib.Path.read_text", forbidden_read)
    with pytest.raises(SealedEvaluationError):
        load_recovery_tasks("evaluation_sealed")


def test_sealed_commitment_contains_no_rows_or_seeds():
    sealed = load_sealed_commitment()
    serialized = json.dumps(sealed, sort_keys=True)
    assert "tasks" not in sealed
    assert "parameter_seed" not in serialized
    assert sealed["content_policy"] == "opaque_commitment_only_no_task_rows"


@pytest.mark.parametrize("split", ["train", "development"])
def test_every_recovery_task_has_exactly_two_extra_policy_steps(split):
    for task in load_recovery_tasks(split):
        assert task.recovery_horizon == task.base_horizon + 2
        assert task.fixture.split == split
