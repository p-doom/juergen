import json
from dataclasses import asdict

import pytest

from osworld_parity.proper_vm_capability_ladder.rung34_recovery.actions import ARMS
from osworld_parity.proper_vm_capability_ladder.rung34_recovery.env import (
    DeterministicRecoveryBackend,
    RecoveryTrainingEnv,
)
from osworld_parity.proper_vm_capability_ladder.rung34_recovery.rollouts import (
    RolloutSchemaError,
    public_on_policy_record,
    scripted_recovery_records,
    validate_rollout_record,
)
from osworld_parity.proper_vm_capability_ladder.rung34_recovery.spec import (
    load_recovery_tasks,
)


@pytest.mark.parametrize("split", ["train", "development"])
@pytest.mark.parametrize("arm", ARMS)
def test_scripted_recovery_records_export_no_trainer_only_values(split, arm):
    rows = scripted_recovery_records(split, arm)
    assert rows
    serialized = json.dumps(rows, sort_keys=True)
    for forbidden in ('"reward"', '"hidden_state"', '"oracle"', '"expected"', '"near_miss"'):
        assert forbidden not in serialized
    assert all(row["trainer_only_values_exported"] is False for row in rows)
    assert all(
        any(event["outcome_label"] == "injected_perturbation" for event in row["events"])
        for row in rows
    )


def test_on_policy_record_preserves_all_three_diagnostic_labels():
    env = RecoveryTrainingEnv(
        DeterministicRecoveryBackend(), split="development", arm="compact_raw_phaseb"
    )
    env.reset(task_index=0)
    env.step("TEST_EXECUTOR_FAILURE")
    env.step("0 0 0")
    env.step("TEST_GOLD")
    task = load_recovery_tasks("development")[0]
    row = public_on_policy_record(
        task,
        arm="compact_raw_phaseb",
        events=(asdict(event) for event in env.public_events()),
    )
    labels = {event["outcome_label"] for event in row["events"]}
    assert {
        "injected_perturbation",
        "natural_ineffective_action",
        "executor_failure",
        "effective_recovery_action",
    } <= labels


def test_rollout_validation_rejects_nested_trainer_leak():
    row = scripted_recovery_records("development", ARMS[0])[0]
    row["events"][0]["action"] = {"debug": {"hidden_state": {"secret": 1}}}
    with pytest.raises(RolloutSchemaError, match="trainer-only field leak"):
        validate_rollout_record(row)


def test_rollout_validation_rejects_sealed_split():
    row = scripted_recovery_records("development", ARMS[0])[0]
    row["split"] = "evaluation_sealed"
    with pytest.raises(RolloutSchemaError, match="sealed/unknown"):
        validate_rollout_record(row)
