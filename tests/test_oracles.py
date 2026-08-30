"""Trace-only rewards over terminal evidence captured inside `launch`."""

from __future__ import annotations

import asyncio
import inspect

import pytest
import verifiers.v1 as vf
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.task import _requires_runtime

from evals.oracles import (
    NoOracle,
    OracleOutcome,
    OSWorldEvaluateOracle,
    PairedArmDivergence,
    StateOracle,
    final_probe,
)
from evals.signoflife.taskset import SignOfLifeTask
from evals.tasks import DesktopTask
from juergen_doubles import make_task_data, make_trace


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


def test_there_is_no_runtime_kwarg_on_the_reward_decorator() -> None:
    parameters = inspect.signature(vf.reward).parameters
    assert set(parameters) == {"func", "weight", "priority"}, sorted(parameters)


@pytest.mark.parametrize(
    "cls,name",
    [
        (StateOracle, "postcondition"),
        (OSWorldEvaluateOracle, "task_success"),
        (StateOracle, "postcondition_recorded"),
        (OSWorldEvaluateOracle, "full_success"),
        (NoOracle, "freeroll"),
    ],
)
def test_each_trace_only_signal_does_not_require_a_runtime(cls, name) -> None:
    fn = getattr(cls, name)
    assert not _requires_runtime(fn), f"{name} must score offline"
    assert "runtime" not in inspect.signature(fn).parameters


def test_the_state_reward_scores_offline_from_the_recorded_terminal_probe() -> None:
    data, trace = _gate(probe=_probe_ok("terminal_exact_text"), name="terminal_exact_text")
    asyncio.run(Gate(data).score(trace, None))
    assert trace.rewards["postcondition"] == 1.0


def test_no_evidence_at_all_raises_rather_than_scoring_a_failure() -> None:
    """Infrastructure-invalid is not a task failure."""
    data, trace = _gate(probe=None, name="terminal_exact_text")
    with pytest.raises(Exception) as excinfo:
        asyncio.run(Gate(data).score(trace, None))
    assert "no recorded terminal guest evidence" in str(excinfo.value)


def test_an_unreadable_probe_raises_instead_of_reporting_zero() -> None:
    data, trace = _gate(probe={"schema_version": 99}, name="terminal_exact_text")
    with pytest.raises(Exception) as excinfo:
        asyncio.run(Gate(data).score(trace, None))
    assert "could not evaluate" in str(excinfo.value)


@pytest.mark.parametrize("passing", [True, False])
def test_the_reward_and_its_trace_only_twin_agree_on_a_replayed_trace(passing: bool) -> None:
    """The twin must return the same verdict."""
    probe = _probe_ok("terminal_exact_text")
    if not passing:
        probe = {**probe, "captured_text": "something else"}
    data, trace = _gate(probe=probe, name="terminal_exact_text")
    asyncio.run(Gate(data).score(trace, None))
    verdict = trace.rewards["postcondition"]
    replay = make_trace(data, episode={"final_probe": probe, "steps_detail": []})
    asyncio.run(Gate(data).score(replay, None))
    assert replay.metrics["postcondition_recorded"] == verdict
    assert verdict == (1.0 if passing else 0.0)
    assert replay.metrics["postcondition_oracle_error"] == 0.0


def test_offline_scoring_keeps_missing_evidence_infrastructure_invalid() -> None:
    data, trace = _gate(probe=None, name="terminal_exact_text")
    with pytest.raises(Exception, match="no recorded terminal guest evidence"):
        asyncio.run(Gate(data).score(trace, None))
    assert trace.metrics["postcondition_recorded"] == 0.0
    assert trace.metrics["postcondition_evidence_missing"] == 1.0


def test_final_probe_tolerates_a_missing_result() -> None:
    trace = make_trace()
    assert final_probe(trace) is None
    assert final_probe(make_trace(episode={"final_probe": "not a dict"})) is None


def test_the_base_state_oracle_hook_is_abstract() -> None:
    class Bare(StateOracle):
        pass

    with pytest.raises(NotImplementedError):
        Bare().evaluate_state(make_task_data(), {})


def test_the_outcome_dataclass_round_trips() -> None:
    outcome = OracleOutcome(task_id="t", status="ok", success=True, reason="r")
    assert outcome.as_dict() == {
        "task_id": "t",
        "status": "ok",
        "success": True,
        "reason": "r",
        "evidence": {},
    }


class OSWorld(OSWorldEvaluateOracle, DesktopTask):
    pass


def _osworld(**result):
    data = make_task_data(kind="osworld")
    return data, make_trace(data, episode={"validity": "valid", **result})


def test_the_osworld_reward_is_the_evaluate_score_verbatim() -> None:
    data, trace = _osworld(task_reward=0.5)
    asyncio.run(OSWorld(data).score(trace, None))
    assert trace.rewards["task_success"] == 0.5, "no shaping, no curriculum"
    assert trace.metrics["full_success"] == 0.0
    assert trace.metrics["task_reward"] == 0.5


def test_full_success_needs_exactly_one() -> None:
    data, trace = _osworld(task_reward=1.0)
    asyncio.run(OSWorld(data).score(trace, None))
    assert trace.metrics["full_success"] == 1.0

    data, trace = _osworld(task_reward=0.999999)
    asyncio.run(OSWorld(data).score(trace, None))
    assert trace.metrics["full_success"] == 0.0


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
        asyncio.run(OSWorld(data).score(trace, None))


def test_an_infra_invalid_osworld_rollout_raises() -> None:
    data = make_task_data(kind="osworld")
    trace = make_trace(data, episode={"validity": "infra_invalid", "infra_error": {"stage": "x"}})
    with pytest.raises(Exception, match="infrastructure-invalid"):
        asyncio.run(OSWorld(data).score(trace, None))


def test_a_missing_osworld_result_raises() -> None:
    data = make_task_data(kind="osworld")
    with pytest.raises(Exception, match="published no result"):
        asyncio.run(OSWorld(data).score(make_trace(data), None))


def test_offline_osworld_scoring_refuses_a_missing_reward() -> None:
    data, trace = _osworld()
    with pytest.raises(Exception, match="missing or non-numeric"):
        asyncio.run(OSWorld(data).score(trace, None))
    assert trace.metrics["task_reward_missing"] == 1.0
    assert trace.metrics["full_success"] == 0.0


def test_the_freeroll_probe_is_deliberately_reward_free() -> None:
    class Free(NoOracle, DesktopTask):
        pass

    data = make_task_data(kind="none")
    trace = make_trace(data, episode={"steps": 4, "outcome": "click", "parse_errors": 1})
    asyncio.run(Free(data).score(trace, None))
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


def test_the_gate_task_has_exactly_one_trace_only_reward() -> None:
    task = Gate(make_task_data(kind="terminal_exact_text", expected={"text": "x"}))
    rewards = discover_decorated(task, "reward")
    assert [fn.__name__ for fn in rewards] == ["postcondition"]
    assert all(not _requires_runtime(fn) for fn in rewards)


def test_no_gate_metric_requires_a_runtime_so_offline_rescoring_gets_them_all() -> None:
    task = Gate(make_task_data(kind="terminal_exact_text", expected={"text": "x"}))
    for fn in discover_decorated(task, "metric"):
        assert not _requires_runtime(fn), fn.__name__
