from dataclasses import replace

import pytest

from osworld_parity.proper_vm_capability_ladder.rung34_recovery.env import (
    DeterministicRecoveryBackend,
    InjectionDispatchError,
    RecoveryTrainingEnv,
)


def test_policy_observation_excludes_oracle_reward_and_perturbation_label():
    env = RecoveryTrainingEnv(
        DeterministicRecoveryBackend(), split="development", arm="compact_raw_phaseb"
    )
    observation, info = env.reset(task_index=0)
    assert set(observation.as_model_input()) == {"instruction", "screenshot_png"}
    assert not {"reward", "oracle", "hidden_state", "expected", "near_miss"} & set(info)
    assert env.public_events()
    assert all(event.outcome_label == "injected_perturbation" for event in env.public_events())


def test_outcome_labels_separate_natural_ineffective_executor_failure_and_effective():
    env = RecoveryTrainingEnv(
        DeterministicRecoveryBackend(), split="development", arm="compact_raw_phaseb"
    )
    env.reset(task_index=0)
    _, reward, terminated, truncated, info = env.step("0 0 0")
    assert reward == 0.0 and not terminated and not truncated
    assert info["outcome_label"] == "natural_ineffective_action"
    _, reward, terminated, truncated, info = env.step("TEST_EXECUTOR_FAILURE")
    assert reward == 0.0 and not terminated and not truncated
    assert info["outcome_label"] == "executor_failure"
    _, reward, terminated, truncated, info = env.step("TEST_GOLD")
    assert reward == 1.0 and terminated and not truncated
    assert info["outcome_label"] == "effective_recovery_action"


def test_policy_horizon_excludes_controller_injection():
    env = RecoveryTrainingEnv(
        DeterministicRecoveryBackend(), split="development", arm="compact_raw_phaseb"
    )
    _, info = env.reset(task_index=2)  # scroll base horizon=2, recovery horizon=4
    assert info["recovery_horizon"] == 4
    for step in range(1, 5):
        _, _, terminated, truncated, step_info = env.step("0 0 0")
        assert step_info["step"] == step
        assert not terminated
        assert truncated is (step == 4)


def test_reset_is_deterministic_after_controlled_injection():
    env = RecoveryTrainingEnv(
        DeterministicRecoveryBackend(), split="development", arm="native_absolute_control"
    )
    first_observation, _ = env.reset(task_index=4)
    first_hash = env.trainer_hidden_state_sha256()
    first_events = env.public_events()
    env.step({"test": "gold"})
    second_observation, _ = env.reset(task_index=4)
    assert env.trainer_hidden_state_sha256() == first_hash
    assert env.public_events() == first_events
    assert second_observation.screenshot_png == first_observation.screenshot_png


class FailingInjectionBackend(DeterministicRecoveryBackend):
    def dispatch(self, task, arm, action):
        snapshot = super().dispatch(task, arm, action)
        return replace(snapshot, executor_dispatch_status="error")


def test_failed_injection_is_executor_failure_not_injected_perturbation():
    env = RecoveryTrainingEnv(
        FailingInjectionBackend(), split="development", arm="compact_raw_phaseb"
    )
    with pytest.raises(InjectionDispatchError):
        env.reset(task_index=0)
    assert env.public_events()[-1].outcome_label == "executor_failure"
