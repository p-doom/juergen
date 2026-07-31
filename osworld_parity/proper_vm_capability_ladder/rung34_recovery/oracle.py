from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..rung1b_realapps.oracle import evaluate_state
from .spec import RecoveryTask


@dataclass(frozen=True)
class TrainerEvaluation:
    reward: float
    solved: bool
    oracle_status: str
    reason: str


class TrainerOnlyOracle:
    """Hidden evaluator; instances belong to the environment backend/trainer."""

    def evaluate(self, task: RecoveryTask, hidden_state: dict[str, Any]) -> TrainerEvaluation:
        result = evaluate_state(task.fixture, hidden_state)
        if result.oracle_status != "ok":
            raise RuntimeError(f"trainer-only recovery oracle failed: {result.reason}")
        return TrainerEvaluation(
            1.0 if result.MOUSE_SOLVED else 0.0,
            bool(result.MOUSE_SOLVED),
            result.oracle_status,
            result.reason,
        )
