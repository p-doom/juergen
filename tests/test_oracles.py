"""Item 16 — oracles as runtime-declaring rewards.

There is **no `@vf.reward(runtime=...)` kwarg**: the decorator takes `weight` and
`priority` only. Runtime is injected by *declaring a `runtime` parameter with no
default* — `_requires_runtime` (`v1/task.py:70-73`) reads
`signature(fn).parameters["runtime"].default is Parameter.empty`, and `Task.score`
drops runtime-requiring signals when it has no runtime.

So each oracle must (a) actually receive a live runtime, and (b) have a trace-only
`@vf.metric` twin that returns the same verdict on a replayed trace.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest
import verifiers.v1 as vf
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.task import _requires_runtime

import agent.desktop as dsk
from evals.oracles import (
    NoOracle,
    OracleOutcome,
    OSWorldEvaluateOracle,
    PairedArmDivergence,
    ReachOracle,
    StateOracle,
    final_probe,
    probe_now,
)
from evals.signoflife.taskset import SignOfLifeTask
from evals.tasks import RESULT_KEY, DesktopTask
from juergen_doubles import FakeSession, make_task_data, make_trace


class _Runtime:
    """A `vf.Runtime` stand-in — the harness only needs its presence."""

    id = "runtime-1"


def _probe_ok(task_id: str = "cell") -> dict:
    return {
        "schema_version": 1,
        "task_id": task_id,
        "capture_file_exists": True,
        "captured_text": "hello",
    }


class Gate(SignOfLifeTask):
    pass


def _gate(*, expected=None, probe=None, name="terminal_exact_text", **result):
    data = make_task_data(
        name=name, kind="terminal_exact_text", expected=expected or {"text": "hello"}
    )
    payload = {"final_probe": probe, "steps_detail": [], **result}
    return data, make_trace(data, episode=payload)


# =========================================================================== #
# the runtime-injection contract
# =========================================================================== #


def test_there_is_no_runtime_kwarg_on_the_reward_decorator() -> None:
    parameters = inspect.signature(vf.reward).parameters
    assert set(parameters) == {"func", "weight", "priority"}, sorted(parameters)


@pytest.mark.parametrize(
    "cls,name",
    [
        (StateOracle, "postcondition"),
        (ReachOracle, "reach"),
        (OSWorldEvaluateOracle, "task_success"),
    ],
)
def test_each_state_reading_reward_declares_runtime_with_no_default(cls, name) -> None:
    fn = getattr(cls, name)
    parameter = inspect.signature(fn).parameters.get("runtime")
    assert parameter is not None, f"{name} must declare `runtime` to receive it"
    assert parameter.default is inspect.Parameter.empty
    assert _requires_runtime(fn), f"{name} must be skipped when there is no runtime"


@pytest.mark.parametrize(
    "cls,name",
    [
        (StateOracle, "postcondition_recorded"),
        (ReachOracle, "reach_frame"),
        (ReachOracle, "shaped_progress"),
        (OSWorldEvaluateOracle, "full_success"),
        (NoOracle, "freeroll"),
    ],
)
def test_each_trace_only_signal_does_not_require_a_runtime(cls, name) -> None:
    fn = getattr(cls, name)
    assert not _requires_runtime(fn), f"{name} must score offline"
    assert "runtime" not in inspect.signature(fn).parameters


def test_a_reward_receives_a_live_runtime_when_one_is_supplied() -> None:
    seen = {}

    class Probe(DesktopTask):
        @vf.reward
        async def needs_runtime(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
            seen["runtime"] = runtime
            return 1.0

    data = make_task_data()
    trace = make_trace(data, episode={})
    runtime = _Runtime()
    asyncio.run(Probe(data).score(trace, runtime))
    assert seen["runtime"] is runtime, "runtime is injected by parameter name"
    assert trace.rewards["needs_runtime"] == 1.0


def test_a_runtime_declaring_reward_is_skipped_offline_rather_than_scored_zero() -> None:
    """Scoring a VM you cannot reach as a failure is worse than no data."""
    data, trace = _gate(probe=_probe_ok("terminal_exact_text"), name="terminal_exact_text")
    asyncio.run(Gate(data).score(trace, None))
    assert "postcondition" not in trace.rewards, "the runtime-backed reward is skipped"
    assert trace.metrics["postcondition_recorded"] == 1.0, "the twin still scores"


# =========================================================================== #
# StateOracle: live vs recorded, and the twin agreement
# =========================================================================== #


def test_the_oracle_reads_the_live_guest_when_the_lease_is_still_open(monkeypatch) -> None:
    data, trace = _gate(probe=None, name="terminal_exact_text")

    class Lease:
        session = FakeSession()

    monkeypatch.setattr(
        dsk.REGISTRY, "get", lambda trace_id: Lease() if trace_id == trace.id else None
    )
    monkeypatch.setattr(
        "evals.oracles.preparer_for",
        lambda kind: type("P", (), {"probe": staticmethod(lambda s, t: _probe_ok(t.name))})(),
    )
    asyncio.run(Gate(data).score(trace, _Runtime()))
    assert trace.rewards["postcondition"] == 1.0
    assert trace.info["oracle"]["postcondition"]["source"] == "live"


def test_the_oracle_falls_back_to_the_recorded_probe_once_the_grace_window_closes() -> None:
    data, trace = _gate(probe=_probe_ok("terminal_exact_text"), name="terminal_exact_text")
    asyncio.run(Gate(data).score(trace, _Runtime()))
    assert trace.rewards["postcondition"] == 1.0
    assert trace.info["oracle"]["postcondition"]["source"] == "recorded"
    assert trace.metrics["postcondition_probe_live"] == 0.0


def test_a_failing_live_probe_falls_back_rather_than_losing_the_score(monkeypatch) -> None:
    data, trace = _gate(probe=_probe_ok("terminal_exact_text"), name="terminal_exact_text")

    class Lease:
        session = FakeSession()

    monkeypatch.setattr(dsk.REGISTRY, "get", lambda trace_id: Lease())

    def angry(kind):
        class P:
            @staticmethod
            def probe(session, task):
                raise RuntimeError("guest went away")

        return P()

    monkeypatch.setattr("evals.oracles.preparer_for", angry)
    asyncio.run(Gate(data).score(trace, _Runtime()))
    assert trace.rewards["postcondition"] == 1.0
    assert trace.info["oracle"]["postcondition"]["source"] == "recorded"


def test_no_evidence_at_all_raises_rather_than_scoring_a_failure() -> None:
    """Infrastructure-invalid is not a task failure."""
    data, trace = _gate(probe=None, name="terminal_exact_text")
    with pytest.raises(Exception) as excinfo:
        asyncio.run(Gate(data).score(trace, _Runtime()))
    assert "no guest evidence" in str(excinfo.value)


def test_an_unreadable_probe_raises_instead_of_reporting_zero() -> None:
    data, trace = _gate(probe={"schema_version": 99}, name="terminal_exact_text")
    with pytest.raises(Exception) as excinfo:
        asyncio.run(Gate(data).score(trace, _Runtime()))
    assert "could not evaluate" in str(excinfo.value)


@pytest.mark.parametrize("passing", [True, False])
def test_the_reward_and_its_trace_only_twin_agree_on_a_replayed_trace(passing: bool) -> None:
    """★ Item 16's second half: the twin must return the same verdict."""
    probe = _probe_ok("terminal_exact_text")
    if not passing:
        probe = {**probe, "captured_text": "something else"}
    data, trace = _gate(probe=probe, name="terminal_exact_text")
    asyncio.run(Gate(data).score(trace, _Runtime()))
    live_verdict = trace.rewards["postcondition"]
    replay = make_trace(data, episode={"final_probe": probe, "steps_detail": []})
    asyncio.run(Gate(data).score(replay, None))
    assert replay.metrics["postcondition_recorded"] == live_verdict
    assert live_verdict == (1.0 if passing else 0.0)
    assert replay.metrics["postcondition_oracle_error"] == 0.0


def test_the_twin_reports_missing_evidence_without_raising() -> None:
    data, trace = _gate(probe=None, name="terminal_exact_text")
    asyncio.run(Gate(data).score(trace, None))
    assert trace.metrics["postcondition_recorded"] == 0.0
    assert trace.metrics["postcondition_evidence_missing"] == 1.0


def test_final_probe_and_probe_now_tolerate_a_missing_result() -> None:
    trace = make_trace()
    assert final_probe(trace) is None
    assert final_probe(make_trace(episode={"final_probe": "not a dict"})) is None
    data = make_task_data(kind="none")
    probe, source = probe_now(make_trace(data), data)
    assert probe is None and source == "recorded"


def test_the_base_state_oracle_hook_is_abstract() -> None:
    class Bare(StateOracle):
        pass

    with pytest.raises(NotImplementedError):
        Bare().evaluate_state(make_task_data(), {})


def test_the_outcome_dataclass_round_trips_and_defaults_to_recorded() -> None:
    outcome = OracleOutcome(task_id="t", status="ok", success=True, reason="r")
    assert outcome.source == "recorded"
    assert outcome.as_dict() == {
        "task_id": "t",
        "status": "ok",
        "success": True,
        "reason": "r",
        "evidence": {},
        "source": "recorded",
    }


# =========================================================================== #
# ReachOracle
# =========================================================================== #


class Reach(ReachOracle, DesktopTask):
    pass


def _reach(**result):
    data = make_task_data(kind="grounding", bbox=(10, 10, 50, 50))
    return data, make_trace(data, episode={"steps_detail": [], **result})


def test_reach_is_reach_at_any_frame_not_final_frame_containment() -> None:
    """A cursor that passes through and overshoots has still demonstrated grounding."""
    data, trace = _reach(reach_frame=3, best_distance=0.0)
    asyncio.run(Reach(data).score(trace, _Runtime()))
    assert trace.rewards["reach"] == 1.0
    assert trace.metrics["reached"] == 1.0 and trace.metrics["reach_frame"] == 3.0


def test_a_miss_falls_back_to_the_scoring_time_cursor() -> None:
    data, trace = _reach(reach_frame=-1, best_distance=80.0, final_probe={"cursor": [20, 20]})
    asyncio.run(Reach(data).score(trace, _Runtime()))
    assert trace.rewards["reach"] == 1.0, "a cursor inside the bbox at scoring time counts"


def test_a_genuine_miss_scores_zero_with_a_shaped_remainder() -> None:
    data, trace = _reach(reach_frame=-1, best_distance=80.0, final_probe={"cursor": [900, 900]})
    asyncio.run(Reach(data).score(trace, _Runtime()))
    assert trace.rewards["reach"] == 0.0
    assert 0.0 < trace.rewards["shaped_progress"] < 0.3, "a miss still carries gradient"


def test_the_shaped_term_is_zero_once_the_target_is_reached() -> None:
    data, trace = _reach(reach_frame=1, best_distance=0.0)
    asyncio.run(Reach(data).score(trace, _Runtime()))
    assert trace.rewards["shaped_progress"] == 0.0


def test_the_shaped_term_is_zero_when_distance_is_the_unset_sentinel() -> None:
    data, trace = _reach(reach_frame=-1, best_distance=-1.0)
    asyncio.run(Reach(data).score(trace, _Runtime()))
    assert trace.rewards["shaped_progress"] == 0.0, "-1.0 means undefined, not adjacent"


def test_the_shaped_term_decreases_with_distance() -> None:
    scores = []
    for distance in (10.0, 100.0, 400.0, 2000.0):
        data, trace = _reach(reach_frame=-1, best_distance=distance, final_probe={"cursor": [999, 999]})
        asyncio.run(Reach(data).score(trace, _Runtime()))
        scores.append(trace.rewards["shaped_progress"])
    assert scores == sorted(scores, reverse=True), scores
    assert all(0.0 <= s <= 0.3 for s in scores)


def test_a_missing_result_raises_rather_than_training_a_zero() -> None:
    data = make_task_data(kind="grounding", bbox=(10, 10, 50, 50))
    with pytest.raises(Exception, match="published no result"):
        asyncio.run(Reach(data).score(make_trace(data), _Runtime()))


# =========================================================================== #
# OSWorldEvaluateOracle
# =========================================================================== #


class OSWorld(OSWorldEvaluateOracle, DesktopTask):
    pass


def _osworld(**result):
    data = make_task_data(kind="osworld")
    return data, make_trace(data, episode={"validity": "valid", **result})


def test_the_osworld_reward_is_the_evaluate_score_verbatim() -> None:
    data, trace = _osworld(task_reward=0.5)
    asyncio.run(OSWorld(data).score(trace, _Runtime()))
    assert trace.rewards["task_success"] == 0.5, "no shaping, no curriculum"
    assert trace.metrics["full_success"] == 0.0
    assert trace.metrics["task_reward"] == 0.5


def test_full_success_needs_essentially_one() -> None:
    data, trace = _osworld(task_reward=1.0)
    asyncio.run(OSWorld(data).score(trace, _Runtime()))
    assert trace.metrics["full_success"] == 1.0


@pytest.mark.parametrize(
    "payload,message",
    [
        ({}, "missing or non-numeric"),
        ({"task_reward": None}, "missing or non-numeric"),
        ({"task_reward": True}, "missing or non-numeric"),
        ({"task_reward": "1.0"}, "missing or non-numeric"),
        ({"task_reward": float("nan")}, "invalid"),
        ({"task_reward": 1.5}, "invalid"),
        ({"task_reward": -0.1}, "invalid"),
    ],
)
def test_an_invalid_osworld_score_raises_rather_than_returning_zero(payload, message) -> None:
    data, trace = _osworld(**payload)
    with pytest.raises(Exception, match=message):
        asyncio.run(OSWorld(data).score(trace, _Runtime()))


def test_an_infra_invalid_osworld_rollout_raises() -> None:
    data = make_task_data(kind="osworld")
    trace = make_trace(data, episode={"validity": "infra_invalid", "infra_error": {"stage": "x"}})
    with pytest.raises(Exception, match="infrastructure-invalid"):
        asyncio.run(OSWorld(data).score(trace, _Runtime()))


def test_a_missing_osworld_result_raises() -> None:
    data = make_task_data(kind="osworld")
    with pytest.raises(Exception, match="result is missing"):
        asyncio.run(OSWorld(data).score(make_trace(data), _Runtime()))


def test_the_osworld_metric_reports_a_missing_reward_without_raising() -> None:
    data, trace = _osworld()
    asyncio.run(OSWorld(data).score(trace, None))
    assert trace.metrics["task_reward_missing"] == 1.0
    assert trace.metrics["full_success"] == 0.0


# =========================================================================== #
# NoOracle / PairedArmDivergence
# =========================================================================== #


def test_the_freeroll_probe_is_deliberately_reward_free() -> None:
    class Free(NoOracle, DesktopTask):
        pass

    data = make_task_data(kind="none")
    trace = make_trace(data, episode={"steps": 4, "outcome": "click", "parse_errors": 1})
    asyncio.run(Free(data).score(trace, _Runtime()))
    assert trace.rewards == {}, "dressing a stop_reason up as a reward manufactures a number"
    assert trace.metrics["clicked"] == 1.0 and trace.metrics["n_steps"] == 4.0


def test_the_paired_group_reward_has_weight_zero() -> None:
    """Rewarding two grammars for being identical is the opposite of an A/B."""
    fn = PairedArmDivergence.arm_agreement
    assert getattr(fn, "_vf_weight", 1.0) == 0.0
    assert getattr(fn, "group_reward", False) is True


def test_arm_agreement_finds_the_first_divergent_turn() -> None:
    class Paired(PairedArmDivergence, DesktopTask):
        pass

    data = make_task_data()
    a = make_trace(
        data,
        episode={
            "steps_detail": [
                {"frame_sha256": "x", "control": None, "parse_ok": True, "cursor_after": [1, 1]},
                {"frame_sha256": "y", "control": None, "parse_ok": True, "cursor_after": [2, 2]},
            ]
        },
    )
    b = make_trace(
        data,
        episode={
            "steps_detail": [
                {"frame_sha256": "x", "control": None, "parse_ok": True, "cursor_after": [1, 1]},
                {"frame_sha256": "z", "control": None, "parse_ok": True, "cursor_after": [2, 2]},
            ]
        },
    )
    asyncio.run(Paired(data).score_group([a, b]))
    divergence = a.info["paired"]["first_divergence"]
    assert divergence["turn_index"] == 1 and divergence["field"] == "frame_sha256"
    assert a.rewards.get("arm_agreement", 0.0) == 0.0


def test_arm_agreement_reports_agreement_and_length_mismatch() -> None:
    class Paired(PairedArmDivergence, DesktopTask):
        pass

    data = make_task_data()
    step = {"frame_sha256": "x", "control": None, "parse_ok": True, "cursor_after": [1, 1]}
    same = [make_trace(data, episode={"steps_detail": [dict(step)]}) for _ in range(2)]
    asyncio.run(Paired(data).score_group(same))
    assert same[0].info["paired"]["first_divergence"] is None
    short = make_trace(data, episode={"steps_detail": [dict(step)]})
    long = make_trace(data, episode={"steps_detail": [dict(step), dict(step)]})
    asyncio.run(Paired(data).score_group([short, long]))
    assert short.info["paired"]["first_divergence"]["field"] == "trajectory_length"


def test_a_single_trace_group_has_no_divergence() -> None:
    class Paired(PairedArmDivergence, DesktopTask):
        pass

    data = make_task_data()
    one = make_trace(data, episode={"steps_detail": []})
    asyncio.run(Paired(data).score_group([one]))
    assert one.info["paired"]["first_divergence"] is None


# =========================================================================== #
# the gate task's signal set
# =========================================================================== #


def test_the_gate_task_has_exactly_one_reward_and_it_needs_a_runtime() -> None:
    task = Gate(make_task_data(kind="terminal_exact_text", expected={"text": "x"}))
    rewards = discover_decorated(task, "reward")
    assert [fn.__name__ for fn in rewards] == ["postcondition"]
    assert all(_requires_runtime(fn) for fn in rewards)


def test_no_gate_metric_requires_a_runtime_so_offline_rescoring_gets_them_all() -> None:
    task = Gate(make_task_data(kind="terminal_exact_text", expected={"text": "x"}))
    for fn in discover_decorated(task, "metric"):
        assert not _requires_runtime(fn), fn.__name__
