from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..rung1b_realapps.training.conversion import (
    assert_round_trip,
    convert_native_trajectory,
)
from ..rung1b_realapps.trajectory import UiGeometry, build_trajectory
from .spec import RecoveryTask


ARMS = ("native_absolute_control", "compact_raw_phaseb")


@dataclass(frozen=True)
class RecoveryGeometry:
    base: UiGeometry = UiGeometry()
    wrong_focus: tuple[int, int] = (120, 72)
    benign_target: tuple[int, int] = (1650, 80)
    wrong_file_source: tuple[int, int] = (480, 210)


@dataclass(frozen=True)
class ActionScript:
    arm: str
    actions: tuple[dict[str, Any] | str, ...]


@dataclass(frozen=True)
class RecoveryDemonstration:
    task: RecoveryTask
    arm: str
    initial_cursor: tuple[int, int]
    perturbation: ActionScript
    policy_actions: tuple[dict[str, Any] | str, ...]


def _native_perturbation(
    task: RecoveryTask, geometry: RecoveryGeometry
) -> tuple[dict[str, Any], ...]:
    if task.perturbation == "wrong_focus":
        return ({"action": "left_click", "coordinate": list(geometry.wrong_focus)},)
    if task.perturbation == "opposite_scroll":
        desired = str(task.fixture.params["direction"])
        opposite_clicks = 7 if desired == "down" else -7
        return (
            {"action": "mouse_move", "coordinate": list(geometry.base.scroll_surface)},
            {"action": "scroll", "clicks": opposite_clicks},
        )
    if task.perturbation == "wrong_file_drag":
        return (
            {
                "action": "mouse_down",
                "button": "left",
                "coordinate": list(geometry.wrong_file_source),
            },
            {"action": "mouse_move", "coordinate": list(geometry.base.drag_destination)},
            {
                "action": "mouse_up",
                "button": "left",
                "coordinate": list(geometry.base.drag_destination),
            },
        )
    if task.perturbation == "benign_wrong_click":
        return ({"action": "left_click", "coordinate": list(geometry.benign_target)},)
    raise ValueError(f"unknown controlled perturbation: {task.perturbation}")


def build_perturbation(
    task: RecoveryTask,
    *,
    arm: str,
    initial_cursor: tuple[int, int],
    geometry: RecoveryGeometry = RecoveryGeometry(),
) -> ActionScript:
    if arm not in ARMS:
        raise ValueError(f"unknown recovery arm: {arm}")
    native = _native_perturbation(task, geometry)
    compact = assert_round_trip(native, initial_cursor=initial_cursor)
    actions: tuple[dict[str, Any] | str, ...] = native if arm == ARMS[0] else compact
    return ActionScript(arm, actions)


def build_recovery_demonstration(
    task: RecoveryTask,
    *,
    arm: str,
    initial_cursor: tuple[int, int],
    geometry: RecoveryGeometry = RecoveryGeometry(),
) -> RecoveryDemonstration:
    perturbation = build_perturbation(
        task, arm=arm, initial_cursor=initial_cursor, geometry=geometry
    )
    native_injection = _native_perturbation(task, geometry)
    cursor_after_injection = initial_cursor
    # Reuse the audited converter to compute the exact cursor after controller actions.
    compact_injection = convert_native_trajectory(
        native_injection, initial_cursor=initial_cursor
    )
    from ..rung1b_realapps.training.conversion import replay_signature

    signature = replay_signature(
        compact_injection,
        arm="compact_raw_phaseb",
        initial_cursor=initial_cursor,
    )
    cursor_after_injection = tuple(int(value) for value in signature["cursor"])
    policy_actions = build_recovery_policy_actions(
        task,
        arm=arm,
        cursor_after_injection=cursor_after_injection,
        geometry=geometry,
    )
    return RecoveryDemonstration(
        task, arm, initial_cursor, perturbation, policy_actions
    )


def build_recovery_policy_actions(
    task: RecoveryTask,
    *,
    arm: str,
    cursor_after_injection: tuple[int, int],
    geometry: RecoveryGeometry = RecoveryGeometry(),
) -> tuple[dict[str, Any] | str, ...]:
    base = build_trajectory(
        task.fixture,
        arm=arm,
        cursor=cursor_after_injection,
        geometry=geometry.base,
    )
    policy_actions = base.actions
    # An opposite scroll may actually move the document. One extra signed
    # scroll cancels the controller displacement before the audited base
    # trajectory makes progress; it is one of the two explicit recovery steps.
    if task.perturbation == "opposite_scroll":
        desired_clicks = -7 if task.fixture.params["direction"] == "down" else 7
        correction: dict[str, Any] | str = (
            {"action": "scroll", "clicks": desired_clicks}
            if arm == "native_absolute_control"
            else f"0 0 {desired_clicks}"
        )
        policy_actions = (correction,) + policy_actions
    if len(base.actions) != task.base_horizon:
        raise AssertionError("base trajectory/horizon drift")
    if len(policy_actions) > task.recovery_horizon:
        raise AssertionError("scripted recovery exceeds horizon+2")
    return policy_actions
