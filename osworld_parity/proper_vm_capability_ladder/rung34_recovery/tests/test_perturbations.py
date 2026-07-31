import pytest

from osworld_parity.proper_vm_capability_ladder.rung1b_realapps.training.conversion import (
    replay_signature,
)
from osworld_parity.proper_vm_capability_ladder.rung34_recovery.actions import (
    ARMS,
    build_perturbation,
    build_recovery_demonstration,
)
from osworld_parity.proper_vm_capability_ladder.rung34_recovery.spec import (
    load_recovery_tasks,
)


def test_controlled_perturbations_cover_all_required_failure_modes():
    tasks = load_recovery_tasks("development")
    assert {task.perturbation for task in tasks} == {
        "wrong_focus",
        "opposite_scroll",
        "wrong_file_drag",
    }


@pytest.mark.parametrize("task_index", range(6))
def test_native_and_compact_perturbation_replay_is_matched(task_index):
    task = load_recovery_tasks("development")[task_index]
    initial_cursor = (73, 91)
    native = build_perturbation(
        task, arm=ARMS[0], initial_cursor=initial_cursor
    )
    compact = build_perturbation(
        task, arm=ARMS[1], initial_cursor=initial_cursor
    )
    assert replay_signature(
        native.actions, arm=ARMS[0], initial_cursor=initial_cursor
    ) == replay_signature(
        compact.actions, arm=ARMS[1], initial_cursor=initial_cursor
    )


@pytest.mark.parametrize("arm", ARMS)
def test_scripted_recovery_fits_horizon_and_corrects_opposite_scroll(arm):
    for task in load_recovery_tasks("development"):
        demo = build_recovery_demonstration(
            task, arm=arm, initial_cursor=(73, 91)
        )
        assert len(demo.policy_actions) <= task.recovery_horizon
        expected = task.base_horizon + (1 if task.perturbation == "opposite_scroll" else 0)
        assert len(demo.policy_actions) == expected
        assert demo.perturbation.arm == arm
