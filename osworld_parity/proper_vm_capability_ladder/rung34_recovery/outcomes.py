from __future__ import annotations

from enum import Enum


class ActionOrigin(str, Enum):
    CONTROLLER_INJECTION = "controller_injection"
    ON_POLICY = "on_policy"
    SCRIPTED_TEACHER = "scripted_teacher"


class OutcomeLabel(str, Enum):
    INJECTED_PERTURBATION = "injected_perturbation"
    NATURAL_INEFFECTIVE_ACTION = "natural_ineffective_action"
    EXECUTOR_FAILURE = "executor_failure"
    EFFECTIVE_RECOVERY_ACTION = "effective_recovery_action"
    SCRIPTED_RECOVERY_ACTION = "scripted_recovery_action"


def classify_outcome(
    *,
    origin: ActionOrigin,
    executor_dispatch_status: str,
    hidden_state_changed: bool,
) -> OutcomeLabel:
    # Dispatch failure wins: an intended injection that never reached the guest
    # must never be mislabeled as a controlled perturbation.
    if executor_dispatch_status != "ok":
        return OutcomeLabel.EXECUTOR_FAILURE
    if origin is ActionOrigin.CONTROLLER_INJECTION:
        return OutcomeLabel.INJECTED_PERTURBATION
    if origin is ActionOrigin.SCRIPTED_TEACHER:
        return OutcomeLabel.SCRIPTED_RECOVERY_ACTION
    if not hidden_state_changed:
        return OutcomeLabel.NATURAL_INEFFECTIVE_ACTION
    return OutcomeLabel.EFFECTIVE_RECOVERY_ACTION
