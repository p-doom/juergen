"""State oracles as rewards that read real VM state.

The objection this module answers — *"verifiers rewards can only see the message
trace, so a state oracle cannot live there"* — is false.
`Task.score(self, trace, runtime)` (`v1/task.py:269-313`) builds
`available = {"task": self.data, "trace": trace, "runtime": runtime}` and
`decorators.invoke` (`v1/decorators.py:28-30`) passes only the keys whose *names*
appear in the callee's signature. So a reward asks for what it needs by declaring
a parameter, and `_requires_runtime` (`task.py:70-73`) partitions the signals:

    runtime declared without a default  ->  runtime-dependent, skipped offline
    runtime absent (or defaulted)       ->  trace-only, always scored

Note there is no `@vf.reward(runtime=...)` kwarg; the decorator takes `weight` and
`priority` only. Runtime is requested by parameter name.

So `_state_check(task, transport)` becomes `postcondition(self, trace, runtime)`
and keeps reading the guest. Two probe sources, both real:

  * **live** — the rollout's desktop lease is still open (the harness extends it
    past `launch` by `scoring_grace_s` exactly for this), so the oracle re-reads
    the guest at scoring time;
  * **recorded** — the grace window closed, so the oracle uses the final probe the
    harness took inside the episode, which is the same read-only extraction one
    settle-interval earlier.

Which one was used is recorded as a metric, because a run whose oracles all fell
back to `recorded` is a run whose grace window is mistuned.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import verifiers.v1 as vf

from agent.desktop import lease_for_trace
from evals.tasks import RESULT_KEY, DesktopTaskData, in_bbox, preparer_for

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "OSWorldEvaluateOracle",
    "OracleOutcome",
    "ReachOracle",
    "StateOracle",
    "final_probe",
    "probe_now",
]


@dataclass(frozen=True)
class OracleOutcome:
    """A postcondition verdict.

    `status` is `"ok"` or `"error"`: an oracle that could not read its evidence is
    NOT a failed task, and collapsing the two is how a broken probe turns into a
    silent 0/4. `evidence` carries every clause separately so a partial pass is
    diagnosable without re-running the VM.
    """

    task_id: str
    status: str
    success: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    source: str = "recorded"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def final_probe(trace: vf.Trace) -> dict[str, Any] | None:
    result = trace.info.get(RESULT_KEY) or {}
    probe = result.get("final_probe")
    return probe if isinstance(probe, dict) else None


def probe_now(trace: vf.Trace, task: DesktopTaskData) -> tuple[dict[str, Any] | None, str]:
    """Read guest state at scoring time, live if the lease is still open."""
    lease = lease_for_trace(trace.id)
    if lease is not None:
        try:
            return preparer_for(task.kind).probe(lease.session, task), "live"
        except Exception as exc:  # noqa: BLE001 - fall back rather than lose the score
            _LOGGER.warning("live probe failed for %s: %r", trace.id, exc)
    return final_probe(trace), "recorded"


# --------------------------------------------------------------------------- #
# mixins
# --------------------------------------------------------------------------- #


class StateOracle:
    """Mixin: success is decided from realized guest state, never from the trace.

    A subclass implements `evaluate_state`. That keeps the *judgement* in the task
    (verifiers' single judgement authority) while the *extraction* stays in the
    family's `Preparer.probe`.
    """

    def evaluate_state(
        self, task: DesktopTaskData, state: dict[str, Any]
    ) -> OracleOutcome:  # pragma: no cover - subclass hook
        raise NotImplementedError

    @vf.reward
    async def postcondition(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """1.0 iff every required realized VM postcondition is observed.

        Declares `runtime`, so `_requires_runtime` is True and offline replay skips
        it rather than scoring a VM it cannot reach as a failure. The recorded
        counterpart below is the offline-safe view of the same verdict.
        """
        del runtime
        task = trace.task.data
        assert isinstance(task, DesktopTaskData)
        probe, source = probe_now(trace, task)
        if probe is None:
            raise RuntimeError(
                "state oracle has no guest evidence (neither live nor recorded) — "
                "infrastructure-invalid, not a task failure"
            )
        outcome = self.evaluate_state(task, probe)
        trace.info.setdefault("oracle", {})["postcondition"] = {
            **outcome.as_dict(),
            "source": source,
        }
        if outcome.status != "ok":
            raise RuntimeError(f"state oracle could not evaluate: {outcome.reason}")
        return 1.0 if outcome.success else 0.0

    @vf.metric
    async def postcondition_recorded(self, trace: vf.Trace) -> dict[str, float]:
        """The same verdict from the in-episode probe alone — always available.

        Offline re-scoring of `traces.jsonl` gets the number here; the reward above
        is the authoritative, runtime-backed one.
        """
        task = trace.task.data
        assert isinstance(task, DesktopTaskData)
        probe = final_probe(trace)
        if probe is None:
            return {"postcondition_recorded": 0.0, "postcondition_evidence_missing": 1.0}
        outcome = self.evaluate_state(task, probe)
        live = 1.0 if (trace.info.get("oracle", {}).get("postcondition", {}).get("source") == "live") else 0.0
        return {
            "postcondition_recorded": 1.0 if outcome.success else 0.0,
            "postcondition_oracle_error": 0.0 if outcome.status == "ok" else 1.0,
            "postcondition_probe_live": live,
        }


class ReachOracle:
    """Grounding: did the cursor enter the labelled bbox at any frame?

    The headline number is reach-at-any-frame, not final-frame containment — a
    cursor that passes through the target and overshoots has still demonstrated the
    grounding. `reach_frame` (the first hit, or -1) is the ordering statistic and
    the shaped term is a bounded function of the closest approach, so a miss still
    carries gradient.
    """

    shaping_weight: float = 0.3
    shaping_scale: float = 400.0

    @vf.reward
    async def reach(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        del runtime
        task = trace.task.data
        assert isinstance(task, DesktopTaskData)
        result = trace.info.get(RESULT_KEY)
        if not isinstance(result, dict):
            raise RuntimeError("grounding rollout published no result")
        if int(result.get("reach_frame", -1)) >= 0:
            return 1.0
        probe, _ = probe_now(trace, task)
        cursor = tuple((probe or {}).get("cursor") or (-1, -1))
        bbox = task.bbox
        return 1.0 if (bbox and cursor != (-1, -1) and in_bbox(cursor, bbox)) else 0.0

    @vf.reward
    async def shaped_progress(self, trace: vf.Trace) -> float:
        result = trace.info.get(RESULT_KEY) or {}
        if int(result.get("reach_frame", -1)) >= 0:
            return 0.0
        best = float(result.get("best_distance", -1.0))
        if best < 0:
            return 0.0
        return self.shaping_weight * math.exp(-best / max(self.shaping_scale, 1.0))

    @vf.metric
    async def reach_frame(self, trace: vf.Trace) -> dict[str, float]:
        result = trace.info.get(RESULT_KEY) or {}
        frame = int(result.get("reach_frame", -1))
        return {
            "reach_frame": float(frame),
            "reached": 1.0 if frame >= 0 else 0.0,
            "best_distance_px": float(result.get("best_distance", -1.0)),
        }


class OSWorldEvaluateOracle:
    """OSWorld benchmark tasks: the reward IS `DesktopEnv.evaluate()`.

    No shaping and no curriculum, so a lift here is a lift on the benchmark. A
    missing, non-finite or out-of-range score raises rather than returning 0 —
    infrastructure failure must never be trained as task failure. It needs a live
    guest, hence `runtime`.
    """

    full_success_threshold: float = 0.999

    @vf.reward
    async def task_success(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        del runtime
        result = trace.info.get(RESULT_KEY)
        if not isinstance(result, dict):
            raise RuntimeError("OSWorld task result is missing (infrastructure-invalid)")
        if result.get("validity") != "valid":
            raise RuntimeError(
                f"OSWorld task result is infrastructure-invalid: {result.get('infra_error')}"
            )
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

    Deliberately reward-free. The old runner reported `stop_reason` and a
    `click` boolean and nothing else, and dressing that up as a reward would
    manufacture a number nobody validated.
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


# --------------------------------------------------------------------------- #
# paired arms
# --------------------------------------------------------------------------- #


class PairedArmDivergence:
    """Cross-arm comparison as a `@vf.group_reward`.

    `paired_eval/runner.py` ran two grammars over one task inside one process and
    asserted they started from an identical live cursor, geometry and reset
    signature, then reported the first turn where they diverged. Two arms are two
    rollouts of one task under verifiers, so the comparison belongs here:
    `score_group` receives `{"task", "traces"}` and runs after every rollout in the
    group finishes (`episode.py:47-49`).

    Note what is *not* preserved: the receipt chain that made the old comparison
    auditable across process boundaries (binding receipts, compiled-segment hashes,
    executed-segment receipts, successor validation). Those existed because an
    untrusted runtime process produced the evidence; one in-process harness and one
    codec removes the untrusted producer.
    """

    @vf.group_reward(weight=0.0)
    async def arm_agreement(self, traces: list[vf.Trace]) -> list[float]:
        """1.0 for every trace in a group whose arms agree turn-for-turn.

        Weight 0 by design: this is a diagnostic recorded per trace, not a training
        signal. Turning it on would reward two grammars for being identical, which
        is the opposite of what an A/B is for.
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
