from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from ..rung1b_realapps.fixtures import sha256_value
from ..rung1b_realapps.states import gold_state, reset_state
from .actions import ARMS, RecoveryGeometry, build_perturbation
from .oracle import TrainerOnlyOracle
from .outcomes import ActionOrigin, OutcomeLabel, classify_outcome
from .spec import RecoveryTask, load_recovery_tasks


@dataclass(frozen=True)
class PolicyObservation:
    instruction: str
    screenshot_png: bytes

    def as_model_input(self) -> dict[str, Any]:
        return {"instruction": self.instruction, "screenshot_png": self.screenshot_png}


@dataclass(frozen=True)
class BackendSnapshot:
    screenshot_png: bytes
    hidden_state: dict[str, Any]
    geometry: RecoveryGeometry
    cursor: tuple[int, int]
    executor_dispatch_status: str = "ok"


@dataclass(frozen=True)
class PublicActionEvent:
    sequence_index: int
    policy_step: int | None
    origin: str
    action: dict[str, Any] | str
    executor_dispatch_status: str
    outcome_label: str
    screenshot_sha256: str


class RecoveryBackend(Protocol):
    def reset(self, task: RecoveryTask) -> BackendSnapshot: ...

    def dispatch(
        self, task: RecoveryTask, arm: str, action: dict[str, Any] | str
    ) -> BackendSnapshot: ...


class InjectionDispatchError(RuntimeError):
    pass


def _screenshot_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class RecoveryTrainingEnv:
    """Gym-like recovery env with controller injection and trainer-only reward."""

    def __init__(
        self,
        backend: RecoveryBackend,
        *,
        split: str = "train",
        arm: str = "compact_raw_phaseb",
        oracle: TrainerOnlyOracle | None = None,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown recovery arm: {arm}")
        self.backend = backend
        self.tasks = load_recovery_tasks(split)
        self.split = split
        self.arm = arm
        self._oracle = oracle or TrainerOnlyOracle()
        self._task: RecoveryTask | None = None
        self._hidden_state: dict[str, Any] | None = None
        self._steps = 0
        self._done = True
        self._events: list[PublicActionEvent] = []

    def _event(
        self,
        *,
        origin: ActionOrigin,
        action: dict[str, Any] | str,
        snapshot: BackendSnapshot,
        before_state: dict[str, Any],
        policy_step: int | None,
    ) -> PublicActionEvent:
        changed = sha256_value(before_state) != sha256_value(snapshot.hidden_state)
        label = classify_outcome(
            origin=origin,
            executor_dispatch_status=snapshot.executor_dispatch_status,
            hidden_state_changed=changed,
        )
        return PublicActionEvent(
            len(self._events),
            policy_step,
            origin.value,
            action,
            snapshot.executor_dispatch_status,
            label.value,
            _screenshot_sha(snapshot.screenshot_png),
        )

    def reset(self, *, task_index: int = 0) -> tuple[PolicyObservation, dict[str, Any]]:
        task = self.tasks[task_index % len(self.tasks)]
        snapshot = self.backend.reset(task)
        self._task = task
        self._hidden_state = snapshot.hidden_state
        self._steps = 0
        self._done = False
        self._events = []
        script = build_perturbation(
            task,
            arm=self.arm,
            initial_cursor=snapshot.cursor,
            geometry=snapshot.geometry,
        )
        for action in script.actions:
            before = self._hidden_state
            snapshot = self.backend.dispatch(task, self.arm, action)
            event = self._event(
                origin=ActionOrigin.CONTROLLER_INJECTION,
                action=action,
                snapshot=snapshot,
                before_state=before,
                policy_step=None,
            )
            self._events.append(event)
            self._hidden_state = snapshot.hidden_state
            if event.outcome_label == OutcomeLabel.EXECUTOR_FAILURE.value:
                self._done = True
                raise InjectionDispatchError(
                    f"controlled perturbation dispatch failed for {task.id}"
                )
        observation = PolicyObservation(task.instruction, snapshot.screenshot_png)
        return observation, {
            "task_id": task.id,
            "task_sha256": task.task_sha256,
            "split": task.split,
            "arm": self.arm,
            "base_horizon": task.base_horizon,
            "recovery_horizon": task.recovery_horizon,
            "controlled_perturbation_applied": True,
        }

    def step(
        self, action: dict[str, Any] | str
    ) -> tuple[PolicyObservation, float, bool, bool, dict[str, Any]]:
        if self._done or self._task is None or self._hidden_state is None:
            raise RuntimeError("recovery environment requires reset")
        before = self._hidden_state
        snapshot = self.backend.dispatch(self._task, self.arm, action)
        self._steps += 1
        event = self._event(
            origin=ActionOrigin.ON_POLICY,
            action=action,
            snapshot=snapshot,
            before_state=before,
            policy_step=self._steps,
        )
        self._events.append(event)
        self._hidden_state = snapshot.hidden_state
        evaluation = self._oracle.evaluate(self._task, snapshot.hidden_state)
        terminated = evaluation.solved
        truncated = self._steps >= self._task.recovery_horizon and not terminated
        self._done = terminated or truncated
        observation = PolicyObservation(self._task.instruction, snapshot.screenshot_png)
        info = {
            "task_id": self._task.id,
            "step": self._steps,
            "recovery_horizon": self._task.recovery_horizon,
            "arm": self.arm,
            "executor_dispatch_status": snapshot.executor_dispatch_status,
            "outcome_label": event.outcome_label,
        }
        return observation, evaluation.reward, terminated, truncated, info

    def public_events(self) -> tuple[PublicActionEvent, ...]:
        return tuple(self._events)

    def trainer_hidden_state_sha256(self) -> str:
        """Trainer/reset-test hook. This value must never enter rollout records."""
        if self._hidden_state is None:
            raise RuntimeError("recovery environment requires reset")
        return sha256_value(self._hidden_state)


class DeterministicRecoveryBackend:
    """CPU-only test backend with deterministic reset and failure injection."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.task: RecoveryTask | None = None
        self.counter = 0
        self.dispatches_since_reset = 0

    def reset(self, task: RecoveryTask) -> BackendSnapshot:
        self.task = task
        self.state = reset_state(task.fixture)
        self.counter += 1
        self.dispatches_since_reset = 0
        return BackendSnapshot(
            b"\x89PNG\r\n\x1a\nrecovery-reset",
            dict(self.state),
            RecoveryGeometry(),
            (73, 91),
        )

    def dispatch(
        self, task: RecoveryTask, arm: str, action: dict[str, Any] | str
    ) -> BackendSnapshot:
        self.dispatches_since_reset += 1
        status = "ok"
        if action == "TEST_EXECUTOR_FAILURE" or action == {"test": "executor_failure"}:
            status = "error"
        elif action == "TEST_GOLD" or action == {"test": "gold"}:
            self.state = gold_state(task.fixture)
        screenshot = (
            b"\x89PNG\r\n\x1a\nrecovery-"
            + str(self.dispatches_since_reset).encode("ascii")
        )
        return BackendSnapshot(
            screenshot,
            dict(self.state),
            RecoveryGeometry(),
            (73, 91),
            status,
        )
