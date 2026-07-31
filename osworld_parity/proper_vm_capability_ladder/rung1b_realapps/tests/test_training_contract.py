import pytest

from osworld_parity.proper_vm_capability_ladder.rung1b_realapps.training.collector import (
    TeacherCandidate,
    rejection_sample,
)
from osworld_parity.proper_vm_capability_ladder.rung1b_realapps.training.conversion import (
    assert_round_trip,
    convert_native_trajectory,
    replay_signature,
)
from osworld_parity.proper_vm_capability_ladder.rung1b_realapps.training.demonstrations import (
    scripted_gold_records,
)
from osworld_parity.proper_vm_capability_ladder.rung1b_realapps.training.env import (
    DeterministicTestBackend,
    Rung1bTrainingEnv,
)
from osworld_parity.proper_vm_capability_ladder.rung1b_realapps.training.splits import (
    SealedEvaluationError,
    load_split_manifest,
    materialize_tasks,
    proposed_train_extension,
)


def test_split_cells_are_disjoint_and_eval_is_sealed():
    manifest = load_split_manifest()
    sets = [
        {(cell.template, cell.seed) for cell in manifest.splits[name]}
        for name in ("train", "development", "evaluation_sealed")
    ]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    with pytest.raises(SealedEvaluationError):
        materialize_tasks("evaluation_sealed")


def test_train_materialization_is_deterministic_and_scalable():
    first = materialize_tasks("train")
    second = materialize_tasks("train")
    assert first == second
    assert len(first) == 18
    extension = proposed_train_extension("files_drag", first_seed=50001, count=100)
    assert len(extension) == 100


@pytest.mark.parametrize("split", ["train", "development"])
def test_demonstration_export_has_no_hidden_reward_or_oracle_state(split):
    rows = scripted_gold_records(split)
    assert rows
    assert all(row["hidden_reward_in_record"] is False for row in rows)
    assert all(row["oracle_state_in_record"] is False for row in rows)
    assert all("expected" not in row and "near_miss" not in row for row in rows)


def test_policy_observation_cannot_see_oracle_or_expected_state():
    env = Rung1bTrainingEnv(
        DeterministicTestBackend(), split="train", arm="compact_raw_phaseb"
    )
    observation, info = env.reset(task_index=0)
    payload = observation.as_model_input()
    assert set(payload) == {"instruction", "screenshot_png"}
    serialized_keys = set(payload) | set(info)
    assert not {"expected", "near_miss", "hidden_state", "oracle", "reward"} & serialized_keys
    _, reward, terminated, truncated, step_info = env.step("TEST_GOLD")
    assert reward == 1.0 and terminated and not truncated
    assert not {"expected", "hidden_state", "oracle"} & set(step_info)


def test_environment_truncates_at_frozen_horizon():
    env = Rung1bTrainingEnv(
        DeterministicTestBackend(), split="development", arm="compact_raw_phaseb"
    )
    _, info = env.reset(task_index=2)  # scroll horizon two
    assert info["horizon"] == 2
    _, reward, terminated, truncated, _ = env.step("0 0 0")
    assert reward == 0 and not terminated and not truncated
    _, reward, terminated, truncated, _ = env.step("0 0 0")
    assert reward == 0 and not terminated and truncated


def test_native_to_compact_round_trip_and_replay():
    native = (
        {"action": "mouse_move", "coordinate": [200, 300]},
        {"action": "left_click", "coordinate": [220, 320]},
        {"action": "scroll", "clicks": -7},
        {"action": "key", "keys": ["ControlLeft", "KeyA"]},
        {"action": "type", "text": "Grüße 東京 🧭"},
    )
    compact = assert_round_trip(native, initial_cursor=(50, 60))
    assert compact == convert_native_trajectory(native, initial_cursor=(50, 60))
    assert replay_signature(native, arm="native_absolute_control", initial_cursor=(50, 60)) == replay_signature(
        compact, arm="compact_raw_phaseb", initial_cursor=(50, 60)
    )


def test_teacher_rejection_sampling_accepts_only_hidden_success():
    actions = ({"action": "scroll", "clicks": -7},)

    def produce(attempt):
        return TeacherCandidate(
            actions,
            (50, 60),
            ("a" * 64,),
            1.0 if attempt == 1 else 0.0,
            attempt == 1,
        )

    candidate, attempts = rejection_sample(produce, max_attempts=2)
    assert attempts == 2 and candidate.accepted_reward == 1.0
