"""State oracles over terminal evidence recorded while the desktop lease is live."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import verifiers.v1 as vf

from evals.tasks import (
    FULL_SUCCESS_THRESHOLD,
    RESULT_KEY,
    DesktopTaskData,
    valid_result,
)

__all__ = [
    "OSWorldEvaluateOracle",
    "OracleOutcome",
    "StateOracle",
    "final_probe",
]


@dataclass(frozen=True)
class OracleOutcome:
    """A postcondition verdict.

    `status` is `"ok"` or `"error"`: an oracle that could not read its evidence is
    not a failed task, and collapsing the two is how a broken probe turns into a
    silent 0/4. `evidence` carries every clause separately so a partial pass is
    diagnosable without re-running the VM.
    """

    task_id: str
    status: str
    success: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def final_probe(trace: vf.Trace) -> dict[str, Any] | None:
    result = trace.info.get(RESULT_KEY) or {}
    probe = result.get("final_probe")
    return probe if isinstance(probe, dict) else None


class StateOracle:
    """Mixin: success is decided from recorded realized state, never from actions.

    A subclass implements `evaluate_state`. The judgement stays in the task
    (verifiers' single judgement authority); the extraction stays in the family's
    `Preparer.probe`.
    """

    def evaluate_state(
        self, task: DesktopTaskData, state: dict[str, Any]
    ) -> OracleOutcome:  # pragma: no cover - subclass hook
        raise NotImplementedError

    @vf.reward
    async def postcondition(self, trace: vf.Trace) -> float:
        """1.0 iff every required terminal postcondition is recorded."""
        task = trace.task.data
        assert isinstance(task, DesktopTaskData)
        probe = final_probe(trace)
        if probe is None:
            raise RuntimeError(
                "state oracle has no recorded terminal guest evidence — "
                "infrastructure-invalid, not a task failure"
            )
        outcome = self.evaluate_state(task, probe)
        trace.info.setdefault("oracle", {})["postcondition"] = outcome.as_dict()
        if outcome.status != "ok":
            raise RuntimeError(f"state oracle could not evaluate: {outcome.reason}")
        return 1.0 if outcome.success else 0.0

    @vf.metric
    async def postcondition_recorded(self, trace: vf.Trace) -> dict[str, float]:
        """The same verdict as a metric for existing result consumers."""
        task = trace.task.data
        assert isinstance(task, DesktopTaskData)
        probe = final_probe(trace)
        if probe is None:
            return {"postcondition_recorded": 0.0, "postcondition_evidence_missing": 1.0}
        outcome = self.evaluate_state(task, probe)
        return {
            "postcondition_recorded": 1.0 if outcome.success else 0.0,
            "postcondition_oracle_error": 0.0 if outcome.status == "ok" else 1.0,
        }


class OSWorldEvaluateOracle:
    """OSWorld benchmark tasks: the reward is `DesktopEnv.evaluate()`.

    No shaping and no curriculum, so a lift here is a lift on the benchmark. A
    missing, non-finite or out-of-range score raises rather than returning 0, so
    infrastructure failure is never trained as task failure. `launch` records the
    score before releasing the desktop, so reward evaluation is trace-only.
    """

    full_success_threshold: float = FULL_SUCCESS_THRESHOLD

    @vf.reward
    async def task_success(self, trace: vf.Trace) -> float:
        result = valid_result(trace, "OSWorld")
        raw = result.get("task_reward")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RuntimeError("OSWorld task reward is missing or non-numeric")
        score = float(raw)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError(f"OSWorld task reward is invalid: {score!r}")
        return score

    @vf.metric
    async def full_success(self, trace: vf.Trace) -> dict[str, float]:
        result = trace.info.get(RESULT_KEY) or {}
        raw = result.get("task_reward")
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return {"full_success": 0.0, "task_reward_missing": 1.0}
        return {
            "full_success": 1.0 if float(raw) >= self.full_success_threshold else 0.0,
            "task_reward": float(raw),
        }


class NoOracle:
    """freeroll: a qualitative probe with no realized-state postcondition.

    Reward-free: there is no validated number to report, only the metrics below.
    """

    @vf.metric
    async def freeroll(self, trace: vf.Trace) -> dict[str, float]:
        result = trace.info.get(RESULT_KEY) or {}
        return {
            "n_steps": float(result.get("steps", 0)),
            "parse_errors": float(result.get("parse_errors", 0)),
            "clicked": 1.0 if result.get("outcome") == "click" else 0.0,
            "terminated": 1.0 if result.get("control_terminate") else 0.0,
        }


class PairedArmDivergence:
    """Cross-arm comparison as a `@vf.group_reward`.

    Two arms are two rollouts of one task: `score_group` receives
    `{"task", "traces"}` and runs after every rollout in the group finishes
    (`episode.py:47-49`). It reports the first turn at which the arms diverged.
    """

    @vf.group_reward(weight=0.0)
    async def arm_agreement(self, traces: list[vf.Trace]) -> list[float]:
        """1.0 for every trace in a group whose arms agree turn-for-turn.

        Weight 0: a diagnostic recorded per trace, not a training signal. Turning it
        on would reward two grammars for being identical.
        """
        divergence = _first_divergence(traces)
        agree = 0.0 if divergence is not None else 1.0
        for trace in traces:
            trace.info.setdefault("paired", {})["first_divergence"] = divergence
        return [agree] * len(traces)


def _steps(trace: vf.Trace) -> list[dict[str, Any]]:
    result = trace.info.get(RESULT_KEY) or {}
    steps = result.get("steps_detail")
    return steps if isinstance(steps, list) else []


def _first_divergence(traces: list[vf.Trace]) -> dict[str, Any] | None:
    """The first turn index at which any two arms differ on a comparable field."""
    if len(traces) < 2:
        return None
    series = [_steps(trace) for trace in traces]
    fields = ("frame_sha256", "control", "parse_ok", "cursor_after")
    for index in range(max(len(s) for s in series)):
        present = [s for s in series if index < len(s)]
        if len(present) != len(series):
            return {
                "turn_index": index,
                "field": "trajectory_length",
                "values": [len(s) for s in series],
            }
        for name in fields:
            values = [s[index].get(name) for s in present]
            if len({repr(value) for value in values}) != 1:
                return {"turn_index": index, "field": name, "values": values}
    return None
