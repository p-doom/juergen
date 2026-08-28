"""The episode driver.

Every behaviour is exercised through a real `DesktopHarness.launch` against a fake
pool injected by `pool_target`, which is what `pool_target` exists for. No VM, no
GPU, no network.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from pathlib import Path

import pytest
import verifiers.v1 as vf

import agent.desktop as dsk
from evals.harness import (
    ArtifactConfig,
    BudgetConfig,
    DesktopHarness,
    DesktopHarnessConfig,
    DesktopPoolConfig,
    HistoryConfig,
    ImageBudgetConfig,
    ScriptedConfig,
    SettleConfig,
    _TRAJECTORY,
    _assert_frame_set,
    _Budget,
    _is_left_click,
    _screenshot,
    _to_thread,
)
from agent.agent import load_codec
from desktop.geometry import DisplayGeometry
from evals.tasks import (
    PREPARERS,
    RESULT_KEY,
    DesktopState,
    DesktopTaskData,
    register_preparer,
)
from juergen_doubles import (
    FakeSession,
    load_convert,
    make_ctx,
    make_task_data,
    make_trace,
    jpeg,
)


@pytest.fixture(autouse=True)
def _no_pool_leak():
    yield
    dsk.close_all_pools()


class ScriptablePreparer:
    """A `Preparer` whose probe returns a caller-supplied sequence."""

    kind = "harness_test"

    def __init__(self) -> None:
        self.probes: list[dict] = []
        self.prepared = 0
        self.plan: list = []
        self.rendered: list[str] = []

    def prepare(self, session, task):
        self.prepared += 1
        return {"prepared": "harness_test"}

    def probe(self, session, task):
        if self.probes:
            return self.probes.pop(0) if len(self.probes) > 1 else self.probes[0]
        return {"postcondition_status": "ok", "postcondition_success": False}

    def script_plan(self, task, *, negative):
        return list(self.plan)

    def render_step(self, session, task, *, codec, intent):
        self.rendered.append(intent)
        return intent


@pytest.fixture
def preparer():
    instance = ScriptablePreparer()
    register_preparer(instance)
    try:
        yield instance
    finally:
        PREPARERS.pop("harness_test", None)


def _config(tmp_path: Path, **kwargs) -> DesktopHarnessConfig:
    base = dict(
        id="test_harness",
        codec="deltatype_v2",
        history=HistoryConfig(name="interleaved_frames", n_history_frames=8),
        images=ImageBudgetConfig(max_images=4),
        settle=SettleConfig(min_delay_s=0.0, per_kind={}),
        artifacts=ArtifactConfig(
            output_dir=str(tmp_path), save_frames=True, save_prompts=True, write_gif=False
        ),
        pool=DesktopPoolConfig(
            key=f"test-{tmp_path.name}",
            max_node_slots=2,
            slot_dir=str(tmp_path / "slots"),
            pool_target="juergen_harness_pool:Pool",
            hide_gpu_during_boot=False,
            scoring_grace_s=0.0,
        ),
        require_unsolved_start=True,
    )
    base.update(kwargs)
    return DesktopHarnessConfig(**base)


def _run(config, task_data, *, replies=None, session=None):
    import juergen_harness_pool

    juergen_harness_pool.Pool.session = session or FakeSession()
    harness = DesktopHarness(config)
    trace = vf.Trace(
        task=vf.TraceTask(type="DesktopTask", data=task_data), state=DesktopState()
    )
    ctx = make_ctx(replies=list(replies or []))
    asyncio.run(harness.launch(ctx, trace, None, "", "", {}))
    return trace, trace.info[RESULT_KEY], ctx


def _task(**kwargs) -> DesktopTaskData:
    return make_task_data(kind="harness_test", **kwargs)


def test_a_full_episode_publishes_one_result_shape(tmp_path, preparer) -> None:
    trace, result, _ = _run(_config(tmp_path), _task(max_steps=2), replies=["0 0 0 ;", "0 0 0 ;"])
    assert result["schema_version"] == 1 and result["validity"] == "valid"
    assert result["codec"] == "deltatype_v2"
    assert result["history_policy"] == "interleaved_frames"
    assert result["outcome"] == "max_steps"
    assert result["success"] is False
    assert result["steps"] == 2 and len(result["steps_detail"]) == 2
    assert result["host"] and "slurm_job_id" in result
    assert trace.is_completed and trace.stop_condition == "max_steps"


def test_the_episode_stops_the_moment_the_postcondition_is_reached(tmp_path, preparer) -> None:
    preparer.probes = [
        {"postcondition_status": "ok", "postcondition_success": False},
        {"postcondition_status": "ok", "postcondition_success": True},
    ]
    _, result, _ = _run(_config(tmp_path), _task(max_steps=6), replies=["0 0 0 ;"] * 6)
    assert result["outcome"] == "postcondition_reached"
    assert result["success"] is True and result["steps"] == 1


def test_a_model_terminate_without_the_postcondition_is_recorded_as_such(tmp_path, preparer) -> None:
    _, result, _ = _run(
        _config(tmp_path), _task(max_steps=4), replies=["TERMINATE: success"]
    )
    assert result["outcome"] == "model_terminate_without_postcondition"
    assert result["control_terminate"] == "terminate" and result["terminate_step"] == 1
    assert result["success"] is False


def test_a_terminating_turn_dispatches_its_work_before_the_episode_stops(
    tmp_path, preparer
) -> None:
    """`[move_rel, left_click, terminate]` used to dispatch nothing at all, so the
    TERMINATE audit read it as a bare premature terminate.
    """
    calls = [
        {"action": "move_rel", "coordinate": [50, 50]},
        {"action": "left_click"},
    ]
    reply = "\n".join(
        [
            *(
                "<tool_call>\n"
                + json.dumps({"name": "computer_use", "arguments": call})
                + "\n</tool_call>"
                for call in calls
            ),
            "TERMINATE: success",
        ]
    )
    session = FakeSession()
    _, result, _ = _run(
        _config(tmp_path, codec="move_rel"),
        _task(max_steps=4),
        replies=[reply],
        session=session,
    )
    assert result["control_terminate"] == "terminate" and result["terminate_step"] == 1
    assert result["outcome"] == "model_terminate_without_postcondition"
    (dispatched,) = session.operations_log
    kinds = [op.kind for op in dispatched]
    assert kinds == ["move_to", "mouse_down", "mouse_up"], kinds
    assert result["ignored_after_terminate"] == 0


def test_calls_after_a_vendor_terminate_are_counted_and_never_dispatched(
    tmp_path, preparer
) -> None:
    """The count alone would pass whether or not the clicks reached the guest, and
    dispatch is unconditional, so the guest is asserted on too: the control channel
    cuts the body at the terminate, so nothing after it is even parsed.
    """
    reply = "\n".join(
        "<tool_call>\n"
        + json.dumps({"name": "computer_use", "arguments": call})
        + "\n</tool_call>"
        for call in (
            {"action": "terminate", "status": "success"},
            {"action": "left_click"},
            {"action": "left_click"},
        )
    )
    session = FakeSession()
    _, result, _ = _run(
        _config(tmp_path, codec="native_absolute"),
        _task(max_steps=4),
        replies=[reply],
        session=session,
    )
    assert result["ignored_after_terminate"] == 2
    assert [op.kind for batch in session.operations_log for op in batch] == [], (
        "nothing runs once the episode has ended"
    )


def test_a_self_declared_fail_is_recorded_as_fail_not_terminate(tmp_path, preparer) -> None:
    _, result, _ = _run(
        _config(tmp_path), _task(max_steps=4), replies=["TERMINATE: failure"]
    )
    assert result["control_terminate"] == "fail"
    assert result["outcome"] == "model_fail_without_postcondition"


def test_a_parse_error_is_counted_and_the_episode_continues(tmp_path, preparer) -> None:
    _, result, _ = _run(
        _config(tmp_path), _task(max_steps=3), replies=["not an action", "0 0 0 ;", "0 0 0 ;"]
    )
    assert result["parse_errors"] == 1
    assert result["steps"] == 3, "a parse error is a scored outcome, not a stop"
    assert result["steps_detail"][0]["parse_ok"] is False


_REJECTED = "Your previous action was rejected and did not run."


def _wire_messages(ctx, turn: int) -> list[dict]:
    """The messages the given model turn actually sent, 1-based."""
    return ctx.client.calls[turn - 1]["body"]["messages"]


def test_by_default_a_rejected_action_is_indistinguishable_from_an_inert_one(
    tmp_path, preparer
) -> None:
    """What every recorded arm did: the turn after a parse error carries the frame and
    nothing else, so the model cannot tell a refused action from one that ran and
    changed nothing."""
    _, result, ctx = _run(
        _config(tmp_path), _task(max_steps=2), replies=["not an action", "0 0 0 ;"]
    )
    assert result["parse_errors"] == 1
    assert [part["type"] for part in _wire_messages(ctx, 2)[-1]["content"]] == ["image_url"]
    assert "parse_error_notice" not in result


def test_a_configured_notice_reaches_the_turn_after_the_rejected_one(
    tmp_path, preparer
) -> None:
    """The arm the default exists to be compared against: same episode, one text part
    added to the observation that followed the refusal, and to no other turn."""
    _, result, ctx = _run(
        _config(tmp_path, parse_error_notice=_REJECTED),
        _task(max_steps=3),
        replies=["not an action", "0 0 0 ;", "0 0 0 ;"],
    )
    assert result["parse_errors"] == 1
    assert _wire_messages(ctx, 2)[-1]["content"][-1] == {"type": "text", "text": _REJECTED}
    assert [part["type"] for part in _wire_messages(ctx, 1)[-1]["content"]] == [
        "text",
        "image_url",
    ], "nothing precedes the first turn"
    assert [part["type"] for part in _wire_messages(ctx, 3)[-1]["content"]] == [
        "image_url"
    ], "the turn after a parseable action gets no notice"
    assert not [
        message
        for message in _wire_messages(ctx, 3)
        if message["role"] == "assistant" and _REJECTED in (message["content"] or "")
    ], "a notice in the assistant channel would be trained on as the model's own words"
    assert result["parse_error_notice"] == _REJECTED


def test_a_token_capped_turn_is_a_truncation_not_a_parse_error(tmp_path, preparer) -> None:
    """`max_tokens` is a harness knob, so a turn cut off at the cap is a measurement
    that did not happen. Scored as a parse error it becomes model behaviour, and a
    truncation that happens to parse dispatches a fragment of the real action.
    """
    session = FakeSession()
    _, result, _ = _run(
        _config(tmp_path),
        _task(max_steps=4),
        replies=[("thinking about it\n10 10 0 ; +LM", "length"), "0 0 0 ;"],
        session=session,
    )
    assert result["outcome"] == "truncated_action"
    assert result["parse_errors"] == 0, "the model did not emit an unparseable action"
    assert result["steps_detail"][-1]["truncated"] is True
    assert session.operations_log == [], "a truncated action must not be dispatched"


def test_the_guest_receipt_is_published_beside_the_round_trip_cursor(
    tmp_path, preparer
) -> None:
    """`execute_atomic`'s return was bound and dropped on every dispatching turn.

    It is the guest's own account of the action — whether it believes it happened,
    and the cursor observed inside the VM around it. The published
    `cursor_before`/`cursor_after` are two separate host round-trips instead, taken
    before the turn and after the settle, so a click the guest reports as FAILED
    still gets a plausible-looking cursor pair. Both are published: they are
    measured differently, `datasets/convert.py` and `rl/grounding/taskset.py` both
    read the round-trip pair, and a disagreement is a finding rather than an error.
    """
    session = FakeSession(cursor=(140, 90))
    _, result, _ = _run(
        _config(tmp_path), _task(max_steps=1), replies=["0 0 0 ; +LMB -LMB"], session=session
    )
    step = result["steps_detail"][-1]
    receipt = step["guest_receipt"]
    assert receipt is not None, "a dispatching turn must carry the guest's verdict"
    assert receipt["ok"] is True and receipt["failure_kind"] is None
    assert receipt["cursor_before"] == step["cursor_before"], (
        "the guest and the host round-trip disagree about where the cursor was"
    )
    assert receipt["cursor_after"] == step["cursor_after"]

    rows = [
        json.loads(line)
        for line in (tmp_path / "cell" / _TRAJECTORY).read_text().splitlines()
    ]
    assert rows[-1]["info"]["guest_receipt"] == receipt, "it reaches the trajectory too"


def test_a_turn_that_dispatches_nothing_has_no_guest_receipt(tmp_path, preparer) -> None:
    """`NO_OP` compiles to zero operations, so `execute_atomic` is never called.

    `None` here means "nothing was dispatched", which is what makes a `False` `ok`
    on a dispatching turn readable as a real guest failure.
    """
    _, result, _ = _run(_config(tmp_path), _task(max_steps=1), replies=["NO_OP"])
    assert result["steps_detail"][-1]["guest_receipt"] is None


def test_a_receipt_that_carries_no_verdict_is_refused() -> None:
    """The transport contract, not a shape to tolerate.

    A session returning something without `ok` publishes a turn in which a failed
    action is indistinguishable from a successful one, which is the state this
    field exists to end.
    """
    from evals.harness import _receipt_record

    with pytest.raises(TypeError, match="not an execution receipt"):
        _receipt_record({"dispatched": 2})


def test_a_truncated_turn_is_never_written_to_the_trajectory(tmp_path, preparer) -> None:
    """It has no post-action frame, so it has no row — and `datasets/convert.py`
    relies on that rather than filtering it.

    The episode loop records the turn in `steps_detail` and breaks WITHOUT
    appending a frame, and `_trajectory_rows` emits `steps_detail[: n_frames - 1]`,
    which is exactly the entry it excludes. Were a row ever written, the reader
    would have to reject it: the turn compiled to an EMPTY operation stream, which
    every grammar lifts to a legitimate idle action, so it would enter a dataset as
    an unterminated `<think>` block labelled NO_OP.
    """
    _, result, _ = _run(
        _config(tmp_path),
        _task(max_steps=4),
        replies=[("thinking about it\n10 10 0 ; +LM", "length")],
    )
    assert result["steps_detail"][-1]["truncated"] is True, "the turn was recorded"

    run_dir = tmp_path / "cell"
    rows = [
        json.loads(line) for line in (run_dir / _TRAJECTORY).read_text().splitlines()
    ]
    assert [row["step_num"] for row in rows] == [0], "only the reset survives"
    assert not [row for row in rows if row["info"].get("truncated")]
    frames = sorted(p.name for p in (run_dir / "steps").glob("*.jpg"))
    assert frames == ["step_000.jpg"], "the truncated turn produced no frame"


def test_a_whole_turn_is_not_flagged_as_truncated(tmp_path, preparer) -> None:
    _, result, _ = _run(_config(tmp_path), _task(max_steps=2), replies=["0 0 0 ;"] * 2)
    assert result["outcome"] == "max_steps"
    assert [step["truncated"] for step in result["steps_detail"]] == [False, False]


def test_stop_on_click_turns_a_free_rollout_into_a_single_decision_probe(tmp_path, preparer) -> None:
    config = _config(tmp_path, stop_on_click=True)
    _, result, _ = _run(config, _task(max_steps=8), replies=["0 0 0 ; +LMB -LMB"] * 8)
    assert result["outcome"] == "click" and result["steps"] == 1
    off = _config(tmp_path, stop_on_click=False)
    _, result, _ = _run(off, _task(max_steps=3), replies=["0 0 0 ; +LMB -LMB"] * 3)
    assert result["outcome"] == "max_steps" and result["steps"] == 3


def test_is_left_click_reads_the_compiled_operations_not_the_grammar(tmp_path) -> None:
    from agent.agent import Decision, EffectiveSampling

    sampling = EffectiveSampling("m", None, None, None, (), "harness_default", ())

    def decision(ops):
        return Decision(1, "t", None, tuple(ops), None, None, sampling)

    assert _is_left_click(decision([{"kind": "mouse_down", "args": ("left",)}]))
    assert _is_left_click(decision([{"kind": "mouse_down", "args": (1,)}]))
    assert not _is_left_click(decision([{"kind": "mouse_down", "args": ("right",)}]))
    assert not _is_left_click(decision([{"kind": "mouse_up", "args": ("left",)}]))
    assert not _is_left_click(decision([]))


def test_reach_frame_records_the_first_in_bbox_step_and_the_closest_approach(tmp_path, preparer) -> None:
    preparer.probes = [
        {"postcondition_status": "ok", "in_bbox": False, "cursor": [0, 0]},
        {"postcondition_status": "ok", "in_bbox": True, "cursor": [20, 20]},
    ]
    session = FakeSession(cursor=(20, 20))
    _, result, _ = _run(
        _config(tmp_path, require_unsolved_start=False),
        _task(max_steps=3, bbox=(10, 10, 50, 50)),
        replies=["0 0 0 ;"] * 3,
        session=session,
    )
    assert result["reach_frame"] == 1, "the FIRST hit, not the last"
    assert result["best_distance"] == 0.0


def test_best_distance_keeps_the_minimum_over_the_rollout(tmp_path, preparer) -> None:
    session = FakeSession(cursor=(200, 200))
    _, result, _ = _run(
        _config(tmp_path, require_unsolved_start=False),
        _task(max_steps=2, bbox=(10, 10, 50, 50)),
        replies=["0 0 0 ;"] * 2,
        session=session,
    )
    assert result["best_distance"] > 0
    assert result["reach_frame"] == -1


def test_a_cell_that_starts_solved_is_refused(tmp_path, preparer) -> None:
    """A gate that starts solved measures nothing."""
    preparer.probes = [{"postcondition_status": "ok", "postcondition_success": True}]
    _, result, _ = _run(_config(tmp_path), _task(max_steps=2), replies=["0 0 0 ;"])
    assert result["validity"] == "infra_invalid"
    assert result["success"] is None, "None, not False — prime-rl must drop the rollout"
    assert "unsolved state" in result["infra_error"]["message"]


def test_an_unreadable_initial_state_is_refused(tmp_path, preparer) -> None:
    preparer.probes = [{"postcondition_status": "error", "postcondition_success": False}]
    _, result, _ = _run(_config(tmp_path), _task(max_steps=2), replies=["0 0 0 ;"])
    assert result["validity"] == "infra_invalid"
    assert "unreadable initial state" in result["infra_error"]["message"]


def test_the_precondition_can_be_switched_off_for_grounding(tmp_path, preparer) -> None:
    preparer.probes = [{"postcondition_status": "ok", "postcondition_success": True}]
    config = _config(tmp_path, require_unsolved_start=False)
    _, result, _ = _run(config, _task(max_steps=1), replies=["0 0 0 ;"])
    assert result["validity"] == "valid"


def test_success_is_none_not_false_on_infrastructure_failure(tmp_path, preparer) -> None:
    class Broken(FakeSession):
        def execute_atomic(self, operations):
            raise ConnectionError("transport died")

    _, result, _ = _run(
        _config(tmp_path), _task(max_steps=2), replies=["0 0 0 ; +LMB -LMB"], session=Broken()
    )
    assert result["outcome"] == "executor_error"
    assert result["validity"] == "infra_invalid" and result["success"] is None
    assert result["executor_errors"] == 1


def test_a_bad_action_is_a_scored_outcome_not_an_infra_failure(tmp_path, preparer) -> None:
    class Picky(FakeSession):
        def execute_atomic(self, operations):
            raise ValueError("that coordinate is off-screen")

    _, result, _ = _run(
        _config(tmp_path), _task(max_steps=2), replies=["0 0 0 ; +LMB -LMB"] * 2, session=Picky()
    )
    assert result["validity"] == "valid", "a TypeError/ValueError is the SUT misbehaving"
    assert result["action_errors"] == 2 and result["executor_errors"] == 0
    assert result["steps_detail"][0]["action_error"]["type"] == "ValueError"


def test_only_a_held_state_refusal_is_a_scored_executor_error(tmp_path, preparer) -> None:
    from desktop.execute.guest_program import ExecutionError, HeldStateError

    class Unbalanced(FakeSession):
        def execute_atomic(self, operations):
            raise HeldStateError("key not held: shiftleft")

    _, refused, _ = _run(
        _config(tmp_path / "refused"),
        _task(max_steps=1),
        replies=["0 0 0 ; +LMB -LMB"],
        session=Unbalanced(),
    )
    assert refused["validity"] == "valid" and refused["success"] is False
    assert refused["action_errors"] == 1 and refused["executor_errors"] == 0
    assert refused["steps_detail"][0]["action_error"]["type"] == "HeldStateError"

    class GuestGone(FakeSession):
        def execute_atomic(self, operations):
            raise ExecutionError("guest request failed")

    _, failed, _ = _run(
        _config(tmp_path / "failed"),
        _task(max_steps=1),
        replies=["0 0 0 ; +LMB -LMB"],
        session=GuestGone(),
    )
    assert failed["validity"] == "infra_invalid" and failed["success"] is None
    assert failed["action_errors"] == 0 and failed["executor_errors"] == 1


def test_a_model_call_failure_is_infrastructure(tmp_path, preparer) -> None:
    class Angry:
        async def get_response(self, *args, **kwargs):
            raise TimeoutError("endpoint gone")

    harness = DesktopHarness(_config(tmp_path))
    import juergen_harness_pool

    juergen_harness_pool.Pool.session = FakeSession()
    trace = vf.Trace(task=vf.TraceTask(type="T", data=_task(max_steps=2)), state=DesktopState())
    ctx = vf.ModelContext(model="m", client=Angry(), sampling=vf.Sampling())
    asyncio.run(harness.launch(ctx, trace, None, "", "", {}))
    result = trace.info[RESULT_KEY]
    assert result["outcome"] == "model_error" and result["validity"] == "infra_invalid"
    assert result["infra_error"]["stage"] == "model"


def _launch(config, task_data, client) -> tuple[vf.Trace, dict]:
    import juergen_harness_pool

    juergen_harness_pool.Pool.session = FakeSession()
    trace = vf.Trace(
        task=vf.TraceTask(type="DesktopTask", data=task_data), state=DesktopState()
    )
    ctx = vf.ModelContext(model="m", client=client(trace), sampling=vf.Sampling())
    asyncio.run(DesktopHarness(config).launch(ctx, trace, None, "", "", {}))
    return trace, trace.info[RESULT_KEY]


def test_a_framework_stop_that_refuses_the_call_is_not_an_infra_failure(
    tmp_path, preparer
) -> None:
    """The orchestrator halts a rollout by stamping `stop_condition` and answering the
    next call with a 400 (`interception/server.py:400-406`). Reading only the transport
    failure marks a healthy rollout `infra_invalid` with `success=None`, so prime-rl
    drops it and N deflates by exactly the rollouts the batch terminated.
    """

    class Refusing:
        def __init__(self, trace) -> None:
            self.trace = trace
            self.turns = 0

        async def get_response(self, *args, **kwargs):
            self.turns += 1
            if self.turns > 1:
                self.trace.stop("max_turns")
                raise RuntimeError("rollout stopped: max_turns")
            return type(
                "R",
                (),
                {
                    "message": type("M", (), {"content": "0 0 0 ;"})(),
                    "finish_reason": "stop",
                    "usage": type("U", (), {"completion_tokens": 4})(),
                },
            )()

    trace, result = _launch(_config(tmp_path), _task(max_steps=9), Refusing)
    assert result["outcome"] == "framework_stop_max_turns"
    assert result["validity"] == "valid" and result["infra_error"] is None
    assert result["success"] is False, "counted, not dropped: None deflates N"
    assert result["steps"] == 1
    assert trace.stop_condition == "max_turns"


def test_a_framework_stop_ends_the_loop_instead_of_replaying_the_last_turn(
    tmp_path, preparer
) -> None:
    """The other half of the same server path: once a turn has been served, a refusal
    returns that turn again with a 200 (`interception/server.py:401-407`). Without
    consulting `stop_condition` the harness re-dispatches it to `max_steps`.
    """

    class Stopping:
        def __init__(self, trace) -> None:
            self.trace = trace
            self.turns = 0

        async def get_response(self, *args, **kwargs):
            self.turns += 1
            if self.turns > 1:
                self.trace.stop("context_length")
            return type(
                "R",
                (),
                {
                    "message": type("M", (), {"content": "0 0 0 ; +LMB -LMB"})(),
                    "finish_reason": "stop",
                    "usage": type("U", (), {"completion_tokens": 6})(),
                },
            )()

    _, result = _launch(_config(tmp_path), _task(max_steps=9), Stopping)
    assert result["outcome"] == "framework_stop_context_length"
    assert result["steps"] == 2, "the served repeat is not dispatched a third time"
    assert result["validity"] == "valid" and result["success"] is False


def test_a_scripted_oracle_arm_that_passes_is_control_conformant(tmp_path, preparer) -> None:
    preparer.plan = ["0 0 0 ;", "0 0 0 ;"]
    preparer.probes = [
        {"postcondition_status": "ok", "postcondition_success": False},
        {"postcondition_status": "ok", "postcondition_success": True},
    ]
    config = _config(tmp_path, scripted=ScriptedConfig(enabled=True, negative=False))
    _, result, _ = _run(config, _task(max_steps=4))
    assert result["scripted"] is True and result["negative_control"] is False
    assert result["success"] is True and result["control_ok"] == 1.0


def test_a_scripted_negative_arm_that_fails_is_control_conformant(tmp_path, preparer) -> None:
    preparer.plan = ["0 0 0 ;"]
    config = _config(tmp_path, scripted=ScriptedConfig(enabled=True, negative=True))
    _, result, _ = _run(config, _task(max_steps=4))
    assert result["negative_control"] is True
    assert result["success"] is False and result["control_ok"] == 1.0


def test_a_negative_arm_that_passes_is_NOT_conformant(tmp_path, preparer) -> None:
    preparer.plan = ["0 0 0 ;"]
    preparer.probes = [
        {"postcondition_status": "ok", "postcondition_success": False},
        {"postcondition_status": "ok", "postcondition_success": True},
    ]
    config = _config(tmp_path, scripted=ScriptedConfig(enabled=True, negative=True))
    _, result, _ = _run(config, _task(max_steps=4))
    assert result["success"] is True and result["control_ok"] == 0.0


def test_a_scripted_arm_runs_out_of_script_rather_than_looping(tmp_path, preparer) -> None:
    preparer.plan = ["0 0 0 ;"]
    config = _config(tmp_path, scripted=ScriptedConfig(enabled=True, negative=False))
    _, result, _ = _run(config, _task(max_steps=6))
    assert result["outcome"] == "script_exhausted" and result["steps"] == 1


def test_a_scripted_arm_records_its_sampling_source_as_scripted(tmp_path, preparer) -> None:
    preparer.plan = ["0 0 0 ;", "0 0 0 ;"]
    config = _config(tmp_path, scripted=ScriptedConfig(enabled=True, negative=False))
    _, result, _ = _run(config, _task(max_steps=2))
    assert result["sampling"]["temperature_source"] == "scripted"
    assert result["sampling"]["model"] == "scripted"


def test_an_exhausted_script_does_not_erase_the_sampling_provenance(tmp_path, preparer) -> None:
    """`_decide` returns `(None, {})` once the script runs out, and `script_exhausted`
    is the *normal* end of every negative control. Assigning that empty dict over
    `sampling_record` would publish `sampling: {}` for the whole arm, leaving
    `SamplingProvenance` with no temperature and no source.
    """
    preparer.plan = ["0 0 0 ;"]  # one intent, two steps allowed
    config = _config(tmp_path, scripted=ScriptedConfig(enabled=True, negative=True))
    _, result, _ = _run(config, _task(max_steps=2))
    assert result["outcome"] == "script_exhausted"
    assert result["sampling"], "the arm's provenance must survive its own termination"
    assert result["sampling"]["temperature_source"] == "scripted"


def test_a_kind_with_no_scripted_arm_fails_before_a_vm_is_booted(tmp_path) -> None:
    """A config error, so it is refused at `launch` and not after a boot + guest setup."""

    class NoScript:
        kind = "no_script_kind"

        def prepare(self, session, task):
            raise AssertionError("prepare must not run: the arm is unrunnable")

        def probe(self, session, task):
            return {"postcondition_status": "ok", "postcondition_success": False}

    register_preparer(NoScript())
    try:
        config = _config(tmp_path, scripted=ScriptedConfig(enabled=True))
        with pytest.raises(LookupError, match="no scripted arm"):
            _run(config, make_task_data(kind="no_script_kind", max_steps=2))
    finally:
        PREPARERS.pop("no_script_kind", None)


def test_an_unknown_codec_is_refused_before_a_vm_is_booted(tmp_path, preparer) -> None:
    """`codec` is config, so resolving it must not wait for a checked-out desktop."""
    config = _config(tmp_path, codec="no_such_grammar")
    with pytest.raises(LookupError, match="no_such_grammar"):
        _run(config, _task(max_steps=1))
    assert preparer.prepared == 0, "no VM work before the config resolves"


def test_the_harness_provenance_metric_reports_the_calibration(tmp_path, preparer) -> None:
    preparer.plan = ["0 0 0 ;"]
    config = _config(tmp_path, scripted=ScriptedConfig(enabled=True, negative=True))
    trace, _, _ = _run(config, _task(max_steps=2))
    metrics = asyncio.run(DesktopHarness(config).harness_provenance(trace))
    assert metrics == {
        "scripted": 1.0,
        "negative_control": 1.0,
        "control_conformant": 1.0,
        "infra_valid": 1.0,
    }


def test_the_per_kind_settle_is_2s_for_chrome_and_0_75s_elsewhere(monkeypatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    settle = SettleConfig(min_delay_s=0.75, per_kind={"open_chrome": 2.0})
    session = FakeSession()
    _screenshot(session, settle, "open_chrome")
    _screenshot(session, settle, "terminal_command")
    assert slept == [2.0, 0.75], (
        "a global 2.0 s would triple every other cell's wall clock"
    )


def test_a_zero_delay_settle_does_not_sleep(monkeypatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    _screenshot(FakeSession(), SettleConfig(min_delay_s=0.0, per_kind={}), "any")
    assert slept == []


def test_a_stability_capable_session_is_polled_instead_of_slept(monkeypatch) -> None:
    calls = {}

    class Settling(FakeSession):
        def screenshot_settled(self, *, min_delay_s, stability_timeout_s, poll_s):
            calls.update(min_delay_s=min_delay_s, stability_timeout_s=stability_timeout_s)
            return jpeg()

    monkeypatch.setattr("time.sleep", lambda s: (_ for _ in ()).throw(AssertionError("slept")))
    settle = SettleConfig(min_delay_s=0.75, stability_timeout_s=5.0, per_kind={"open_chrome": 2.0})
    _screenshot(Settling(), settle, "open_chrome")
    assert calls == {"min_delay_s": 2.0, "stability_timeout_s": 5.0}


def test_stability_asked_for_but_unimplemented_is_refused_not_ignored() -> None:
    """A silently-downgraded settle is a silently different measurement."""
    settle = SettleConfig(min_delay_s=0.5, stability_timeout_s=5.0, per_kind={})
    with pytest.raises(LookupError, match="stability"):
        _screenshot(FakeSession(), settle, "any")


def test_the_turn_budget_ends_the_episode(tmp_path, preparer) -> None:
    config = _config(tmp_path, budget=BudgetConfig(model_turns=2))
    _, result, _ = _run(config, _task(max_steps=9), replies=["0 0 0 ;"] * 9)
    assert result["outcome"] == "budget_model_turns_exceeded"
    assert result["budget"]["failure"] == "model_turns_exceeded"


def test_the_operations_budget_ends_the_episode(tmp_path, preparer) -> None:
    config = _config(tmp_path, budget=BudgetConfig(operations=2))
    _, result, _ = _run(config, _task(max_steps=9), replies=["0 0 0 ; +LMB -LMB"] * 9)
    assert result["outcome"].startswith("budget_operations")


def test_the_output_token_budget_ends_the_episode(tmp_path, preparer) -> None:
    config = _config(tmp_path, budget=BudgetConfig(output_tokens=8))
    _, result, _ = _run(config, _task(max_steps=9), replies=["0 0 0 ;"] * 9)
    assert result["outcome"] == "budget_output_tokens_exceeded"
    assert result["budget"]["output_tokens"] == 12, "the third four-token turn spends it"


def test_the_published_token_count_is_the_one_the_server_reported(tmp_path, preparer) -> None:
    _, result, _ = _run(
        _config(tmp_path), _task(max_steps=2), replies=["0 0 0 ;", "NO_OP"]
    )
    assert result["budget"]["output_tokens"] == 5


def test_a_truncated_turn_still_spends_its_tokens(tmp_path, preparer) -> None:
    _, result, _ = _run(
        _config(tmp_path),
        _task(max_steps=4),
        replies=[("thinking\n10 10 0 ; +LM", "length")],
    )
    assert result["outcome"] == "truncated_action"
    assert result["budget"]["output_tokens"] == 6


def test_the_parse_error_streak_ends_the_episode(tmp_path, preparer) -> None:
    """`model_turns` cannot separate a checkpoint that acted for nine turns from one
    that emitted nine turns nothing could read: both spend the cell's whole VM and
    both publish `max_steps`."""
    config = _config(tmp_path, budget=BudgetConfig(consecutive_parse_errors=2))
    _, result, _ = _run(config, _task(max_steps=9), replies=["not an action"] * 9)
    assert result["outcome"] == "budget_consecutive_parse_errors_exceeded"
    assert result["steps"] == 3, "the third unreadable turn in a row exceeds a ceiling of 2"
    assert result["parse_errors"] == 3
    assert result["budget"]["consecutive_parse_errors"] == 3


def test_a_turn_that_parsed_resets_the_parse_error_streak(tmp_path, preparer) -> None:
    """Half the turns unreadable is a rate, not a collapse, and the ceiling counts
    only turns in a row — otherwise it would be `parse_errors` with a worse name."""
    config = _config(tmp_path, budget=BudgetConfig(consecutive_parse_errors=2))
    _, result, _ = _run(
        config, _task(max_steps=6), replies=["not an action", "0 0 0 ;"] * 3
    )
    assert result["outcome"] == "max_steps"
    assert result["parse_errors"] == 3
    assert result["budget"]["consecutive_parse_errors"] == 0


def test_an_unset_budget_never_fires() -> None:
    budget = _Budget(BudgetConfig())
    for _ in range(1000):
        budget.turn()
        budget.dispatched(100)
        budget.tokens(1000)
        budget.parsed(False)
    assert budget.failure is None, "a budget nobody set must never fire"


def test_the_wall_clock_budget_fires(monkeypatch) -> None:
    budget = _Budget(BudgetConfig(wall_time_s=1.0))
    budget.started = -100.0
    budget.turn()
    assert budget.failure == "wall_time_exceeded"


def test_the_first_budget_failure_is_the_one_reported() -> None:
    budget = _Budget(BudgetConfig(model_turns=1, operations=1))
    budget.turn()
    budget.turn()
    assert budget.failure == "model_turns_exceeded"
    budget.dispatched(50)
    assert budget.failure == "model_turns_exceeded", "the first failure sticks"


def test_the_budget_snapshot_is_json_serialisable() -> None:
    snapshot = _Budget(BudgetConfig()).snapshot()
    json.dumps(snapshot)
    assert set(snapshot) == {
        "model_turns",
        "operations",
        "output_tokens",
        "consecutive_parse_errors",
        "wall_time_s",
        "failure",
    }


def test_frames_prompts_and_result_json_are_written(tmp_path, preparer) -> None:
    task = _task(max_steps=2, name="cell_artifacts")
    _run(_config(tmp_path), task, replies=["0 0 0 ;"] * 2)
    root = tmp_path / "cell_artifacts"
    assert (root / "result.json").is_file()
    assert (root / "steps" / "step_000.jpg").is_file()
    assert (root / "steps" / "step_001.jpg").is_file()
    assert (root / "steps" / "prompt_001.json").is_file()
    payload = json.loads((root / "result.json").read_text())
    assert payload["schema_version"] == 1
    sidecar = json.loads((root / "steps" / "prompt_001.json").read_text())
    assert sidecar["messages"][0]["role"] == "system"
    assert "base64" not in json.dumps(sidecar), "image bytes are elided in the sidecar"


def test_the_result_json_write_is_atomic_and_leaves_no_temp_file(tmp_path, preparer) -> None:
    task = _task(max_steps=1, name="cell_atomic")
    _run(_config(tmp_path), task, replies=["0 0 0 ;"])
    root = tmp_path / "cell_atomic"
    leftovers = [p.name for p in root.iterdir() if p.name.startswith("result.json.")]
    assert leftovers == [], leftovers
    assert oct((root / "result.json").stat().st_mode)[-3:] == "600"


def test_a_gif_is_written_when_asked(tmp_path, preparer) -> None:
    config = _config(
        tmp_path,
        artifacts=ArtifactConfig(output_dir=str(tmp_path), write_gif=True, save_prompts=False),
    )
    _run(config, _task(max_steps=2, name="cell_gif"), replies=["0 0 0 ;"] * 2)
    assert (tmp_path / "cell_gif" / "rollout.gif").is_file()


def test_a_single_frame_rollout_writes_no_gif(tmp_path, preparer) -> None:
    from evals.harness import _write_gif

    target = tmp_path / "one.gif"
    _write_gif([jpeg()], target)
    assert not target.exists(), "a one-frame animation is not an animation"


def test_artifacts_can_be_switched_off_entirely(tmp_path, preparer) -> None:
    config = _config(
        tmp_path,
        artifacts=ArtifactConfig(
            output_dir=str(tmp_path),
            save_frames=False,
            save_prompts=False,
            write_gif=False,
            write_result_json=False,
        ),
    )
    _run(config, _task(max_steps=1, name="cell_quiet"), replies=["0 0 0 ;"])
    root = tmp_path / "cell_quiet"
    assert not (root / "result.json").exists() and not (root / "steps").exists()


def test_nothing_shells_out_to_register_external(tmp_path) -> None:
    """`labctl register-external` takes only --alias/--path/--kind/--cluster, so no
    invocation of it can populate `metadata.result` — and the rollout viewer is gated
    on `metadata.result.traj_path`. Registration goes through the recipe's
    `[outputs]` marker instead, so there is no knob and no subprocess.
    """
    from evals import harness as harness_module
    from pydantic import ValidationError

    assert not hasattr(harness_module, "_register_labctl")
    assert not hasattr(harness_module, "subprocess"), "nothing here shells out"
    for field in ("register_labctl", "labctl_alias"):
        with pytest.raises(ValidationError):
            ArtifactConfig(**{field: True})


def test_the_harness_refuses_to_run_without_an_artifact_root(tmp_path) -> None:
    """`output_dir` used to fall back to the system temp dir, so a dispatch that
    forgot it published `<root>/result.json` -- the file the recipe registers as its
    `eval_result` -- into a reaped directory, and every concurrent run wrote the
    same shared index. Arms are templates that carry no root, so the check belongs
    where a config becomes a run.
    """
    from evals.harness import DesktopHarness

    config = _config(tmp_path)
    assert config.artifacts.output_dir == str(tmp_path)
    assert DesktopHarness(config)._artifact_root == tmp_path

    rootless = config.model_copy(
        update={"artifacts": config.artifacts.model_copy(update={"output_dir": ""})}
    )
    with pytest.raises(ValueError, match="artifacts.output_dir is unset"):
        DesktopHarness(rootless)


def test_the_artifact_refuses_to_publish_frameless_rows(tmp_path) -> None:
    """Every trajectory row is fetched by frame filename, so a trajectory without
    frames is an artifact whose every frame is a 404."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="404"):
        ArtifactConfig(output_dir=str(tmp_path), save_frames=False, write_result_json=True)


def test_the_rollout_artifact_is_the_layout_labctl_and_convert_both_read(
    tmp_path, preparer
) -> None:
    """The whole contract in one place, read back through the real consumer.

    `<out>/result.json` (runs[] + traj_path) / `<out>/<subdir>/trajectory.jsonl` /
    `<out>/<subdir>/steps/step_%03d.jpg`, with `n_steps + 1` frames so that row
    ordinal, `step_num` and frame index are the same number.
    """
    convert = load_convert()
    session = FakeSession()
    _, result, ctx = _run(
        _config(tmp_path),
        _task(max_steps=3, name="cell_contract", instruction="do the thing"),
        replies=["10 10 0 ;", "10 10 0 ;", "10 10 0 ;"],
        session=session,
    )
    run_dir = tmp_path / "cell_contract"

    assert convert.discover_run_dirs(str(tmp_path)) == [run_dir], (
        "the reader discovers a rollout by result.json + trajectory.jsonl"
    )

    rows = [json.loads(line) for line in (run_dir / _TRAJECTORY).read_text().splitlines()]
    assert [row["step_num"] for row in rows] == [0, 1, 2, 3]
    assert rows[0]["action"] == "<reset>", "convert.py and evals/tasks.py both skip it"
    assert [row["done"] for row in rows] == [False, False, False, True]
    frames = sorted(p.name for p in (run_dir / "steps").glob("*.jpg"))
    assert frames == ["step_000.jpg", "step_001.jpg", "step_002.jpg", "step_003.jpg"], (
        "n_steps + 1 frames: frame 0 is the initial observation"
    )

    # The alignment itself: `convert.py:458` reads the frame a step SAW as
    # `step_{step_num - 1}`, and the harness observed screenshot k as frame k.
    observed = [
        (run_dir / "steps" / f"step_{index:03d}.jpg").read_bytes() for index in range(4)
    ]
    assert len(set(observed)) == 4, "the fake session hands out distinguishable frames"
    for row in rows[1:]:
        seen = run_dir / "steps" / f"step_{row['step_num'] - 1:03d}.jpg"
        assert seen.is_file()
        assert seen.read_bytes() == observed[row["step_num"] - 1]

    first_prompt = _wire_messages(ctx, 1)
    image_part = next(
        part
        for message in first_prompt
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    )
    data_url = image_part["image_url"]["url"]
    assert base64.b64decode(data_url.split(",", 1)[1]) == observed[0]

    # And through the reader, not around it: a rollout of ours is BC input, so every
    # row must convert, not be booked as a parse error for speaking its own grammar.
    target = load_codec("move_rel")
    record = convert.convert_rollout(
        run_dir, target, min_valid_actions=0, max_parse_error_frac=1.0
    )
    assert record is not None, "convert.py refuses a rollout with no screen_size"
    assert record["instruction"] == "do the thing"
    assert record["source_parse_errors"] == 0, (
        "the source grammar is declared by result.json['codec'], so none of these "
        "rows is a parse error"
    )
    assert result["codec"] == "deltatype_v2", "the declaration the reader keys on"

    # The actions themselves, not just the count: each assistant target recompiles
    # to exactly the operations the harness dispatched for that turn, from the same
    # cursor. `10 10 0` in deltatype_v2 and `move_rel [5, 9]` in move_rel are the
    # same move — the encoding changed and the guest-visible effect did not.
    geometry = DisplayGeometry(desktop_width=1920, desktop_height=1080)
    written = [
        message["content"][0]["text"]
        for message in record["messages"]
        if message["role"] == "assistant"
    ]
    assert len(written) == 3
    for row, assistant in zip(rows[1:], written, strict=True):
        cursor = tuple(row["info"]["cursor_before"])
        dispatched = [tuple(op["args"]) for op in row["info"]["operations"]]
        assert dispatched == [(110, 110)], "the fake guest moved +10,+10 each turn"
        recompiled = target.compile(assistant, geometry, cursor)
        assert [tuple(op.args) for op in recompiled] == dispatched, assistant
        assert '"action": "move_rel"' in assistant and '[5, 9]' in assistant


def test_our_own_terminate_survives_the_trip_into_another_grammar(
    tmp_path, preparer
) -> None:
    """The converter carries the harness's own `control` verdict, not a spelling.

    It now has only one spelling to write, which is the point: every target grammar
    renders the same control line, so a rollout in one grammar re-emits as a
    training target in any other without termination being re-derived from an
    Action dataclass.
    """
    convert = load_convert()
    _run(
        _config(tmp_path),
        _task(max_steps=2, name="cell_term", instruction="stop"),
        replies=["10 10 0 ;", "TERMINATE: success"],
    )
    run_dir = tmp_path / "cell_term"

    for name in ("deltatype_v2", "move_rel", "compact_raw", "ordered_events_v3"):
        record = convert.convert_rollout(
            run_dir, load_codec(name), min_valid_actions=0, max_parse_error_frac=1.0
        )
        written = [
            message["content"][0]["text"]
            for message in record["messages"]
            if message["role"] == "assistant"
        ]
        assert record["source_parse_errors"] == 0, name
        assert written[-1].splitlines()[-1] == "TERMINATE: success", (name, written)


def test_the_artifact_root_index_is_what_labctl_reads(tmp_path, preparer) -> None:
    """One `Harness` serves every rollout, so the runs[] index accumulates across
    them; `tasks` must stay a flat metric -> number dict or the UI falls back to a
    JSON tree."""
    import juergen_harness_pool

    juergen_harness_pool.Pool.session = FakeSession()
    config = _config(tmp_path)
    harness = DesktopHarness(config)
    for name, replies in (("cell_a", ["0 0 0 ;"]), ("cell_b", ["TERMINATE"])):
        trace = vf.Trace(
            task=vf.TraceTask(type="DesktopTask", data=_task(max_steps=1, name=name)),
            state=DesktopState(),
        )
        asyncio.run(harness.launch(make_ctx(replies=replies), trace, None, "", "", {}))

    index = json.loads((tmp_path / "result.json").read_text())
    assert [run["subdir"] for run in index["runs"]] == ["cell_a", "cell_b"]
    assert [run["index"] for run in index["runs"]] == [0, 1]
    assert index["primary"] == f"{config.id}/success_rate"
    assert index["primary"] in index["tasks"]
    assert all(isinstance(v, float) for v in index["tasks"].values()), (
        "a null or a nested object drops the whole table to a raw JSON tree"
    )
    assert index["tasks"][f"{config.id}/n_valid"] == 2.0
    assert index["traj_path"] == str((tmp_path / "cell_a" / _TRAJECTORY).resolve()), (
        "the viewer endpoint takes one path and derives steps/ from its parent"
    )


def test_an_infra_invalid_episode_is_counted_but_kept_out_of_the_rate(
    tmp_path, preparer
) -> None:
    preparer.probes = [{"postcondition_status": "ok", "postcondition_success": True}]
    _run(_config(tmp_path), _task(max_steps=1, name="cell_refused"), replies=["0 0 0 ;"])
    index = json.loads((tmp_path / "result.json").read_text())
    (run,) = index["runs"]
    assert run["validity"] == "infra_invalid" and run["success"] is None
    assert index["tasks"][f"{index['task']}/n_episodes"] == 1.0
    assert index["tasks"][f"{index['task']}/n_valid"] == 0.0
    assert index["primary"] is None, "a rate over zero episodes is not a measurement"
    assert f"{index['task']}/success_rate" not in index["tasks"], (
        "0.0 here reads as 'every episode failed' when nothing ran"
    )


def test_a_stray_frame_is_refused_by_name(tmp_path, preparer) -> None:
    """The viewer counts every `*.jpg` as a frame and fetches it by index, so a
    leftover from a longer earlier run offers frames that do not exist."""
    run_dir = tmp_path / "cell_stray"
    (run_dir / "steps").mkdir(parents=True)
    (run_dir / "steps" / "step_009.jpg").write_bytes(jpeg())
    with pytest.raises(RuntimeError, match="cell_stray/steps"):
        _run(_config(tmp_path), _task(max_steps=1, name="cell_stray"), replies=["0 0 0 ;"])


def test_a_missing_frame_is_refused_by_name(tmp_path, preparer) -> None:
    from evals.harness import _assert_frame_set

    steps = tmp_path / "steps"
    steps.mkdir()
    for index in (0, 2):
        (steps / f"step_{index:03d}.jpg").write_bytes(jpeg())
    with pytest.raises(RuntimeError, match=r"missing \['step_001.jpg'\]"):
        _assert_frame_set(steps, 3)
    assert _assert_frame_set(steps / "absent", 0) is None, "no frames, no rows, no error"


@pytest.mark.parametrize("stray", ["step_000.png", "step_001.bin"])
def test_a_non_jpeg_step_file_is_refused_by_name(tmp_path, stray) -> None:
    steps = tmp_path / "steps"
    steps.mkdir()
    (steps / "step_000.jpg").write_bytes(jpeg())
    (steps / stray).write_bytes(b"legacy")

    with pytest.raises(RuntimeError, match=rf"unexpected \['{stray}'\]"):
        _assert_frame_set(steps, 1)


def test_a_step_with_no_post_action_frame_gets_no_row(tmp_path, preparer) -> None:
    """The executor died mid-turn, so there is no screenshot to show for that step —
    but a row without a frame is a hole the viewer cannot fetch. `result.json` still
    carries the step in `steps_detail`."""

    class Broken(FakeSession):
        def execute_atomic(self, operations):
            raise ConnectionError("transport died")

    _, result, _ = _run(
        _config(tmp_path),
        _task(max_steps=2, name="cell_broken"),
        replies=["10 10 0 ; +LMB -LMB"],
        session=Broken(),
    )
    assert result["outcome"] == "executor_error" and len(result["steps_detail"]) == 1
    run_dir = tmp_path / "cell_broken"
    rows = [json.loads(line) for line in (run_dir / _TRAJECTORY).read_text().splitlines()]
    assert [row["step_num"] for row in rows] == [0]
    assert sorted(p.name for p in (run_dir / "steps").glob("*.jpg")) == ["step_000.jpg"]


def test_a_rerun_replaces_the_trajectory_rather_than_doubling_it(tmp_path, preparer) -> None:
    for _ in range(2):
        _run(_config(tmp_path), _task(max_steps=1, name="cell_rerun"), replies=["0 0 0 ;"])
    rows = (tmp_path / "cell_rerun" / _TRAJECTORY).read_text().splitlines()
    assert len(rows) == 2, rows


def test_an_unjustified_digest_mismatch_is_refused(tmp_path) -> None:
    """A digest that is not a prompt this codec renders means the checkpoint was
    trained under a different one, so its score is not the number it looks like.

    This used to be recorded and not enforced, which made "evaluated under the
    wrong prompt" a field in a JSON blob nobody reads rather than a failure.
    """
    from agent.agent import load_codec

    harness = DesktopHarness(_config(tmp_path, system_prompt_sha256="0" * 64))
    with pytest.raises(ValueError, match="not a prompt the 'deltatype_v2' codec renders"):
        harness._prompt_report(load_codec("deltatype_v2"))


def test_a_justified_mismatch_passes_and_the_reason_lands_in_the_record(tmp_path) -> None:
    """A prompt sealed before `describe()` existed cannot be recomputed from any
    codec, so only the arm's author can vouch for it -- in writing, as data."""
    from agent.agent import load_codec

    harness = DesktopHarness(
        _config(
            tmp_path,
            system_prompt_sha256="0" * 64,
            expect_prompt_mismatch="sealed before describe() existed",
        )
    )
    report = harness._prompt_report(load_codec("deltatype_v2"))
    assert report["matches_expected"] is False
    assert report["expect_prompt_mismatch"] == "sealed before describe() existed"
    assert report["comparable_to_sealed_baseline"] is False
    assert "33.9%" in report["baseline_note"] and "Re-measure" in report["baseline_note"]
    assert report["expected_prompt_sha256"] == "0" * 64
    assert len(report["prompt_sha256"]) == 64


def test_the_thinking_prompt_is_accepted_with_no_configuration(tmp_path) -> None:
    """`datasets/convert.py --keep_prose` trains on `THINKING_PREAMBLE +
    describe()` while eval renders the bare `describe()`. Both are prompts THIS
    codec produces, so the thinking digest is computed and accepted -- otherwise
    every thinking arm would need a written excuse for a difference we create."""
    import hashlib

    import grammars
    from agent.agent import load_codec

    codec = load_codec("deltatype_v2")
    thinking = hashlib.sha256(
        grammars.system_prompt(codec, thinking=True).encode()
    ).hexdigest()
    plain = hashlib.sha256(codec.describe().encode()).hexdigest()
    assert thinking != plain

    for digest in (thinking, plain):
        report = DesktopHarness(
            _config(tmp_path, system_prompt_sha256=digest)
        )._prompt_report(codec)
        assert report["matches_expected"] is True
        assert set(report["accepted_prompt_sha256"]) == {thinking, plain}


def test_no_expected_digest_reports_none_rather_than_a_false_match(tmp_path) -> None:
    from agent.agent import load_codec

    report = DesktopHarness(_config(tmp_path))._prompt_report(load_codec("deltatype_v2"))
    assert report["matches_expected"] is None


def test_every_episode_records_the_baseline_caveat(tmp_path, preparer) -> None:
    """The caveat rides every episode, including one whose digest legitimately
    differs -- the justification excuses the mismatch, not the incomparability."""
    trace, _, _ = _run(
        _config(
            tmp_path,
            system_prompt_sha256="a" * 64,
            expect_prompt_mismatch="sealed producer prompt",
        ),
        _task(max_steps=1),
        replies=["0 0 0 ;"],
    )
    prompt = trace.info["prompt"]
    assert prompt["comparable_to_sealed_baseline"] is False
    assert prompt["matches_expected"] is False
    assert "Qwen3-VL-8B=33.9%" in prompt["baseline_note"]
    assert prompt["codec"] == "deltatype_v2"


def test_a_justified_mismatch_does_not_change_the_episode_outcome(tmp_path, preparer) -> None:
    """The check gates whether the run happens, not what it scores. Once an arm has
    vouched for its digest, the mismatch must not perturb the episode itself."""
    good = _run(_config(tmp_path), _task(max_steps=1, name="a"), replies=["0 0 0 ;"])[1]
    mismatched = _run(
        _config(
            tmp_path,
            system_prompt_sha256="f" * 64,
            expect_prompt_mismatch="sealed producer prompt",
        ),
        _task(max_steps=1, name="b"),
        replies=["0 0 0 ;"],
    )[1]
    assert good["outcome"] == mismatched["outcome"] == "max_steps"
    assert good["validity"] == mismatched["validity"] == "valid"


def test_an_unjustified_mismatch_fails_the_run_rather_than_scoring_it(
    tmp_path, preparer
) -> None:
    """The replacement for the tree-wide "nothing may raise on a prompt digest"
    grep. That guard was written when `describe()` could not reproduce any sealed
    prompt, so any gate would have failed every run; now two renderings are
    computed and a mismatch means the checkpoint really is foreign. Refusing to
    produce a number beats producing an incomparable one."""
    with pytest.raises(ValueError, match="not a prompt the"):
        _run(
            _config(tmp_path, system_prompt_sha256="f" * 64),
            _task(max_steps=1, name="c"),
            replies=["0 0 0 ;"],
        )


def test_a_foreign_checkpoint_is_refused_before_a_vm_is_booted(
    tmp_path, preparer, monkeypatch
) -> None:
    """The digest needs the codec and nothing else, so it belongs with the rest of
    the config resolution. Checked after the boot it costs one VM and one guest
    setup per task across a 369-cell array before refusing every one of them."""
    captured = _captured_lease(monkeypatch)
    with pytest.raises(ValueError, match="not a prompt the"):
        _run(
            _config(tmp_path, system_prompt_sha256="e" * 64),
            _task(max_steps=1, name="d"),
            replies=["0 0 0 ;"],
        )
    assert captured == [], "no VM may be leased for a run that cannot be scored"
    assert preparer.prepared == 0


def test_a_system_prompt_override_is_honoured_and_hashed(tmp_path) -> None:
    import hashlib

    from agent.agent import load_codec

    harness = DesktopHarness(_config(tmp_path, system_prompt_override="SEALED PROMPT"))
    report = harness._prompt_report(load_codec("deltatype_v2"))
    assert report["prompt_sha256"] == hashlib.sha256(b"SEALED PROMPT").hexdigest()


def test_every_episode_publishes_the_image_encoding_it_sent(tmp_path, preparer) -> None:
    """`dump_prompt` elides the image bytes, so this is the only record of which
    pixels the checkpoint was scored on."""
    from image_domain import OSWORLD_CURSOR_JPEG_DOMAIN

    trace, result, _ = _run(_config(tmp_path), _task(max_steps=1), replies=["0 0 0 ;"])
    assert result["images"] == {
        "image_domain": OSWORLD_CURSOR_JPEG_DOMAIN,
        "max_images": 4,
    }
    assert trace.info["images"] == {"image_domain": OSWORLD_CURSOR_JPEG_DOMAIN}


def test_the_pool_target_injects_a_fake_and_receives_session_kwargs(tmp_path, preparer) -> None:
    """`pool_target` exists to inject a fake, not to select a backend."""
    config = _config(tmp_path)
    config.pool.session_kwargs = {"image": "x.qcow2", "max_sessions": 3}
    factory = DesktopHarness(config).pool_factory()
    built = factory()
    assert built.kwargs == {"image": "x.qcow2", "max_sessions": 3}


def _captured_lease(monkeypatch) -> list:
    """`launch` hands the lease to the scoring phase; the reaper does the release.

    So the observable contract at the end of `launch` is `finish()` having been
    called with the episode's verdict and a deadline `scoring_grace_s` out — not the
    session already being released (that happens up to `reap_interval_s` later).
    """
    captured: list = []
    original = dsk.LeasedDesktopPool.acquire

    def spy(self, trace_id):
        lease = original(self, trace_id)
        captured.append(lease)
        return lease

    monkeypatch.setattr(dsk.LeasedDesktopPool, "acquire", spy)
    return captured


def test_launch_finishes_the_lease_and_the_reaper_releases_it(tmp_path, preparer, monkeypatch) -> None:
    captured = _captured_lease(monkeypatch)
    session = FakeSession()
    _run(_config(tmp_path), _task(max_steps=1), replies=["0 0 0 ;"], session=session)
    (lease,) = captured
    assert lease.failed is False and lease.error is None, "a clean episode is not a failure"
    assert not lease.released, "launch must not release — scoring may still read the VM"
    assert lease.expired(), "with scoring_grace_s=0 the deadline is immediately past"
    lease.release()  # what the reaper does
    assert session.released == [(False, None)]


def test_the_grace_window_keeps_the_vm_readable_for_scoring(tmp_path, preparer, monkeypatch) -> None:
    captured = _captured_lease(monkeypatch)
    config = _config(tmp_path)
    config.pool.scoring_grace_s = 120.0
    session = FakeSession()
    _run(config, _task(max_steps=1), replies=["0 0 0 ;"], session=session)
    (lease,) = captured
    assert not lease.expired(), "a runtime-declaring reward can still probe the guest"
    assert dsk.lease_for_trace(lease.trace_id) is lease
    assert session.released == []
    lease.release()


def test_an_infra_invalid_episode_retires_the_vm(tmp_path, preparer, monkeypatch) -> None:
    """`failed=True` is what makes desktop retire a session instead of returning it
    to the pool as `ready` (`vm/pool.py:509-519`). `_run` publishes every episode
    exception as `infra_invalid` rather than re-raising, so `launch` must set
    `failed` from that too — otherwise a wedged guest (dead executor transport,
    unreadable state) is recycled into the next rollout as healthy.
    """
    captured = _captured_lease(monkeypatch)
    preparer.probes = [{"postcondition_status": "ok", "postcondition_success": True}]
    session = FakeSession()
    _, result, _ = _run(_config(tmp_path), _task(max_steps=1), replies=["0 0 0 ;"], session=session)
    assert result["validity"] == "infra_invalid"
    (lease,) = captured
    assert lease.failed is True, "an infra-invalid episode must retire its VM"
    assert "unsolved" in (lease.error or "")
    lease.release()
    assert session.released[0][0] is True


def test_an_executor_failure_also_retires_the_vm(tmp_path, preparer, monkeypatch) -> None:
    class Broken(FakeSession):
        def execute_atomic(self, operations):
            raise ConnectionError("transport died")

    captured = _captured_lease(monkeypatch)
    session = Broken()
    _, result, _ = _run(
        _config(tmp_path), _task(max_steps=2), replies=["0 0 0 ; +LMB -LMB"], session=session
    )
    assert result["outcome"] == "executor_error"
    (lease,) = captured
    assert lease.failed is True, "a dead executor transport must retire the VM"
    assert "ConnectionError" in (lease.error or "") and "execute" in (lease.error or "")


def test_a_clean_episode_returns_the_vm_for_reuse(tmp_path, preparer, monkeypatch) -> None:
    """The other half: a healthy VM must not be retired after every rollout."""
    captured = _captured_lease(monkeypatch)
    _, result, _ = _run(_config(tmp_path), _task(max_steps=1), replies=["0 0 0 ;"])
    assert result["validity"] == "valid"
    (lease,) = captured
    assert lease.failed is False and lease.error is None


def test_to_thread_defers_a_cancellation_until_the_thread_has_finished() -> None:
    """`asyncio.to_thread` cannot stop a function that has already started, so a bare
    `await` on it returns the instant the rollout is cancelled while the thread keeps
    driving the VM. `lease.finish` then hands that VM to the scoring phase with an
    `execute_atomic` still in flight against it.
    """
    running = threading.Event()
    finished: list[str] = []

    def blocking() -> str:
        running.set()
        time.sleep(0.2)
        finished.append("done")
        return "receipt"

    async def body() -> list[str]:
        task = asyncio.ensure_future(_to_thread(blocking))
        while not running.is_set():
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Read inside the loop: `asyncio.run` shuts the default executor down on
        # its way out, so after it returns the thread has finished either way.
        return list(finished)

    assert asyncio.run(body()) == ["done"], "the cancellation must not outrun the guest call"


def _cancel_once(event: threading.Event):
    """Drive `launch` and cancel it as soon as `event` is set by the guest thread."""

    async def body(harness, trace, ctx) -> None:
        task = asyncio.ensure_future(harness.launch(ctx, trace, None, "", "", {}))
        while not event.is_set():
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    return body


def test_a_cancellation_during_acquire_still_finishes_the_lease(
    tmp_path, preparer, monkeypatch
) -> None:
    """`lease` was assigned from the await's result, which a cancellation discards —
    so the VM and its node slot were checked out by a thread that ran to completion
    and then owned by nobody. At `max_node_slots=14` a handful of cancellations wedge
    the node.
    """
    import juergen_harness_pool

    captured = _captured_lease(monkeypatch)
    checking_out = threading.Event()
    session = FakeSession()

    def slow_checkout(self):
        checking_out.set()
        time.sleep(0.2)
        return session

    monkeypatch.setattr(juergen_harness_pool.Pool, "checkout", slow_checkout)
    config = _config(tmp_path)
    trace = vf.Trace(
        task=vf.TraceTask(type="DesktopTask", data=_task(max_steps=1)), state=DesktopState()
    )
    harness = DesktopHarness(config)
    asyncio.run(_cancel_once(checking_out)(harness, trace, make_ctx(replies=["0 0 0 ;"])))

    (lease,) = captured
    assert lease.failed is True, "a cancelled rollout must retire its VM"
    assert "Cancelled" in (lease.error or "")
    assert lease.expired(), "with scoring_grace_s=0 the reaper can take it now"
    lease.release()
    assert session.released == [(True, lease.error)]


def test_a_cancelled_episode_finishes_the_lease_only_after_the_guest_call_returns(
    tmp_path, preparer, monkeypatch
) -> None:
    """The ordering the shield buys: `lease.finish` starts the scoring window, and the
    reaper releases the VM into the pool from there, so it must not run while a
    detached thread is still dispatching into that guest.
    """
    order: list[str] = []
    dispatching = threading.Event()

    class Slow(FakeSession):
        def execute_atomic(self, operations):
            dispatching.set()
            time.sleep(0.2)
            order.append("dispatched")
            return super().execute_atomic(operations)

    original = dsk.DesktopLease.finish

    def spy(self, **kwargs):
        order.append("lease_finished")
        return original(self, **kwargs)

    monkeypatch.setattr(dsk.DesktopLease, "finish", spy)
    import juergen_harness_pool

    juergen_harness_pool.Pool.session = Slow()
    trace = vf.Trace(
        task=vf.TraceTask(type="DesktopTask", data=_task(max_steps=2)), state=DesktopState()
    )
    harness = DesktopHarness(_config(tmp_path))
    asyncio.run(
        _cancel_once(dispatching)(harness, trace, make_ctx(replies=["0 0 0 ; +LMB -LMB"] * 2))
    )
    assert order == ["dispatched", "lease_finished"], order


def test_the_desktop_session_id_rides_the_trace(tmp_path, preparer) -> None:
    trace, _, _ = _run(_config(tmp_path), _task(max_steps=1), replies=["0 0 0 ;"])
    assert trace.info["desktop_session"] == "fake-session"


def test_the_gpu_is_hidden_during_boot_when_asked(tmp_path, monkeypatch) -> None:
    from evals.harness import _hidden_gpu

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with _hidden_gpu(True):
        import os

        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3", "restored afterwards"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with _hidden_gpu(True):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    with _hidden_gpu(False):
        assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_a_button_left_down_is_lifted_at_teardown(tmp_path, preparer) -> None:
    """`+LMB` with no `-LMB` is a press the prompt advertises as surviving the turn
    (`grammars/move_rel/codec.py:199`) and the guest honours it. Closing only the
    agent ran `evaluate()`, and the whole scoring window, with the button down.
    """
    session = FakeSession()
    _, result, _ = _run(
        _config(tmp_path), _task(max_steps=1), replies=["10 10 0 ; +LMB"], session=session
    )
    assert result["released_holds"] == [{"kind": "mouse_up", "args": ["left"]}]
    (_, teardown) = session.operations_log
    assert [(op.kind, op.args) for op in teardown] == [("mouse_up", ("left",))]


def test_every_kind_of_hold_is_lifted_newest_first(tmp_path, preparer) -> None:
    session = FakeSession()
    _, result, _ = _run(
        _config(tmp_path),
        _task(max_steps=2),
        replies=["0 0 0 ; +ctrl", "0 0 0 ; +RMB"],
        session=session,
    )
    assert result["released_holds"] == [
        {"kind": "mouse_up", "args": ["right"]},
        {"kind": "key_up", "args": ["ctrl"]},
    ]


def test_a_rollout_that_releases_its_own_presses_leaves_nothing_to_lift(
    tmp_path, preparer
) -> None:
    """The other half: teardown must not synthesise a second release."""
    session = FakeSession()
    _, result, _ = _run(
        _config(tmp_path),
        _task(max_steps=2),
        replies=["10 10 0 ; +LMB", "0 0 0 ; -LMB"],
        session=session,
    )
    assert result["released_holds"] == []
    assert len(session.operations_log) == 2, "no teardown dispatch"


def test_the_holds_are_lifted_before_the_scorer_reads_the_guest(tmp_path, preparer) -> None:
    order: list[str] = []

    class Watching(FakeSession):
        def execute_atomic(self, operations):
            ops = list(operations)
            order.append("+".join(op.kind for op in ops))
            return super().execute_atomic(operations)

        def evaluate(self) -> float:
            order.append("evaluate")
            return 1.0

    session = Watching()
    config = _config(tmp_path, evaluate_on_finish=True)
    _, result, _ = _run(
        config, _task(max_steps=1), replies=["10 10 0 ; +LMB"], session=session
    )
    assert order == ["move_to+mouse_down", "mouse_up", "evaluate"], order
    assert result["task_reward"] == 1.0


def test_a_guest_that_cannot_be_unwedged_retires_its_vm(tmp_path, preparer, monkeypatch) -> None:
    class Stuck(FakeSession):
        def execute_atomic(self, operations):
            ops = list(operations)
            if [op.kind for op in ops] == ["mouse_up"]:
                raise ConnectionError("transport died during teardown")
            return super().execute_atomic(operations)

    captured = _captured_lease(monkeypatch)
    _, result, _ = _run(
        _config(tmp_path), _task(max_steps=1), replies=["10 10 0 ; +LMB"], session=Stuck()
    )
    assert result["validity"] == "infra_invalid" and result["success"] is None
    assert result["infra_error"]["type"] == "ConnectionError"
    (lease,) = captured
    assert lease.failed is True, "a guest with a stuck button must not be recycled"


def test_evaluate_on_finish_publishes_the_osworld_score(tmp_path, preparer) -> None:
    session = FakeSession()
    session.evaluate_value = 1.0
    config = _config(tmp_path, evaluate_on_finish=True)
    _, result, _ = _run(config, _task(max_steps=1), replies=["0 0 0 ;"], session=session)
    assert result["task_reward"] == 1.0


def test_the_declared_terminal_control_reaches_the_scorer(tmp_path, preparer) -> None:
    """OSWorld inverts the reward on `infeasible` tasks — declaring FAIL is the
    success condition there and forfeits everywhere else — and reads that off an
    action history we do not keep. So it is handed over explicitly."""
    session = FakeSession()
    session.evaluate_value = 1.0
    config = _config(tmp_path, evaluate_on_finish=True)
    _run(config, _task(max_steps=2), replies=["TERMINATE: failure"], session=session)
    assert session.declared_terminal == ["fail"]


def test_a_failing_evaluate_is_recorded_as_missing_never_as_zero(tmp_path, preparer) -> None:
    session = FakeSession()  # evaluate_value is None -> raises
    config = _config(tmp_path, evaluate_on_finish=True)
    _, result, _ = _run(config, _task(max_steps=1), replies=["0 0 0 ;"], session=session)
    assert result["task_reward"] is None, "0.0 would be trained as a task failure"
    assert result["validity"] == "infra_invalid" and result["success"] is None
    assert result["infra_error"]["stage"] == "evaluate"
    assert result["outcome"] == "max_steps", "the stop-reason census keeps the real stop"
    index = json.loads((tmp_path / "result.json").read_text())
    assert index["primary"] is None, "an unscorable episode is not a failed one"


def test_the_osworld_score_is_the_verdict_a_scored_arm_reports(tmp_path, preparer) -> None:
    """Jobs 141534/141535 reported `success_rate: 0.0` for both arms while episode
    rewards were 1.000: `success` was reading a postcondition probe that an OSWorld
    task never populates."""
    session = FakeSession()
    session.evaluate_value = 1.0
    config = _config(tmp_path, evaluate_on_finish=True)
    _, result, _ = _run(
        config, _task(max_steps=1, name="scored"), replies=["0 0 0 ;"], session=session
    )
    assert result["outcome"] == "max_steps", "no postcondition was ever reached"
    assert result["success"] is True
    index = json.loads((tmp_path / "result.json").read_text())
    assert index["tasks"][index["primary"]] == 1.0


@pytest.mark.parametrize("reward", [0.0, 0.5, 0.9989, 0.999999, 1.0])
def test_no_scored_episode_can_disagree_with_its_own_reward(
    tmp_path, preparer, reward
) -> None:
    """`success` is derived from the `task_reward` published in the same dict, so
    the pair cannot be constructed inconsistent. Partial credit is a partly-failed
    metric list in `DesktopEnv.evaluate()`, not a solved task."""
    session = FakeSession()
    session.evaluate_value = reward
    config = _config(tmp_path, evaluate_on_finish=True)
    _, result, _ = _run(config, _task(max_steps=1), replies=["0 0 0 ;"], session=session)
    assert result["task_reward"] == reward
    assert result["success"] is (reward == 1.0)


@pytest.mark.parametrize(
    "reward",
    [True, "1.0", float("nan"), float("inf"), -0.1, 1.1],
)
def test_an_invalid_osworld_score_is_infrastructure_invalid(
    tmp_path, preparer, reward
) -> None:
    session = FakeSession()
    session.evaluate_value = reward
    config = _config(tmp_path, evaluate_on_finish=True)
    _, result, _ = _run(config, _task(max_steps=1), replies=["0 0 0 ;"], session=session)
    assert result["task_reward"] is None
    assert result["validity"] == "infra_invalid" and result["success"] is None
    assert result["infra_error"]["stage"] == "evaluate"


def test_a_session_without_evaluate_refuses_the_flag_it_cannot_honour(
    tmp_path, preparer
) -> None:
    """`evaluate_on_finish` must not be silently ignored: the missing score would
    resurface as `OSWorldEvaluateOracle` complaining about a non-numeric reward, one
    layer away from the config that caused it."""

    class NoEval(FakeSession):
        evaluate = None

    config = _config(tmp_path, evaluate_on_finish=True)
    _, result, _ = _run(config, _task(max_steps=1), replies=["0 0 0 ;"], session=NoEval())
    assert result["validity"] == "infra_invalid"
    assert "evaluate_on_finish" in result["infra_error"]["message"]
    assert result["task_reward"] is None, "0.0 would be trained as a task failure"


def test_the_osworld_taskset_skips_a_task_that_already_has_a_result(tmp_path) -> None:
    """An interrupted 369-task array run is exactly when you need this."""
    from evals.tasks import OSWorldTaskset, OSWorldTasksetConfig

    root = tmp_path / "osworld"
    examples = root / "evaluation_examples" / "examples" / "chrome"
    examples.mkdir(parents=True)
    for task_id in ("t1", "t2"):
        (examples / f"{task_id}.json").write_text(
            json.dumps({"id": task_id, "instruction": f"do {task_id}", "config": []})
        )
    (root / "split.json").write_text(json.dumps({"chrome": ["t1", "t2"]}))
    resume = tmp_path / "resume"
    (resume / "chrome" / "t1").mkdir(parents=True)
    (resume / "chrome" / "t1" / "result.json").write_text("{}")
    config = OSWorldTasksetConfig(
        osworld_root=str(root), split_path=str(root / "split.json"), resume_dir=str(resume)
    )
    names = [t.data.name for t in OSWorldTaskset(config).load()]
    assert names == ["t2"], names
    without = OSWorldTasksetConfig(osworld_root=str(root), split_path=str(root / "split.json"))
    assert sorted(t.data.name for t in OSWorldTaskset(without).load()) == ["t1", "t2"]


def test_max_tasks_truncates_the_osworld_taskset(tmp_path) -> None:
    from evals.tasks import OSWorldTaskset, OSWorldTasksetConfig

    root = tmp_path / "osworld"
    examples = root / "evaluation_examples" / "examples" / "chrome"
    examples.mkdir(parents=True)
    for i in range(5):
        (examples / f"t{i}.json").write_text(
            json.dumps({"id": f"t{i}", "instruction": "x", "config": []})
        )
    (root / "split.json").write_text(json.dumps({"chrome": [f"t{i}" for i in range(5)]}))
    config = OSWorldTasksetConfig(
        osworld_root=str(root), split_path=str(root / "split.json"), max_tasks=2
    )
    assert len(list(OSWorldTaskset(config).load())) == 2


def test_the_freeroll_taskset_drops_blanks_and_comments(tmp_path) -> None:
    from evals.tasks import FreerollTaskset, FreerollTasksetConfig

    path = tmp_path / "instructions.txt"
    path.write_text("open a terminal\n\n# a comment\n   \nwrite a file\n")
    rows = list(FreerollTaskset(FreerollTasksetConfig(instructions_file=str(path))).load())
    assert [r.data.instruction for r in rows] == ["open a terminal", "write a file"]
    assert rows[0].data.name.startswith("task_00_open-a-terminal")


def test_an_empty_freeroll_list_still_yields_one_no_goal_rollout() -> None:
    from evals.tasks import FreerollTaskset, FreerollTasksetConfig

    rows = list(FreerollTaskset(FreerollTasksetConfig()).load())
    assert len(rows) == 1 and rows[0].data.instruction == ""
    assert rows[0].data.prompt is None


def test_the_freeroll_desktop_setup_selects_the_preparer() -> None:
    from evals.tasks import FreerollTaskset, FreerollTasksetConfig

    for setup in ("none", "terminal"):
        rows = list(
            FreerollTaskset(
                FreerollTasksetConfig(instructions=["do it"], desktop_setup=setup)
            ).load()
        )
        assert rows[0].data.kind == setup



def test_the_harness_declares_message_prompt_support() -> None:
    assert DesktopHarness.SUPPORTS_MESSAGE_PROMPT is True


def test_a_config_max_steps_overrides_the_task(tmp_path, preparer) -> None:
    config = _config(tmp_path, max_steps=1)
    _, result, _ = _run(config, _task(max_steps=9), replies=["0 0 0 ;"] * 9)
    assert result["steps"] == 1


def test_the_task_max_steps_is_used_when_the_config_leaves_it_zero(tmp_path, preparer) -> None:
    config = _config(tmp_path, max_steps=0)
    _, result, _ = _run(config, _task(max_steps=2), replies=["0 0 0 ;"] * 2)
    assert result["steps"] == 2


def test_a_config_without_a_codec_is_refused() -> None:
    """No default, so an arm that never names a grammar cannot run at all.

    A wrong codec parses and compiles the model's output under a grammar it was
    never trained on: it scores, and the score reads as a parse-failure collapse
    rather than as a config mistake. `verifiers` builds this config by
    `model_validate` on the harness payload (`loaders.narrow_plugin_field`), so the
    refusal lands at config parse, before a VM is booted.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="codec"):
        DesktopHarnessConfig(id="test_harness")


@pytest.mark.parametrize(
    "field,bad",
    [
        ("max_tokens", 0),
        ("max_tokens", True),
        ("temperature", float("nan")),
        ("temperature", -0.1),
        ("top_p", float("nan")),
        ("top_p", 0.0),
        ("top_p", 1.1),
        ("max_steps", -1),
    ],
)
def test_the_config_validates_its_bounds(field, bad) -> None:
    from pydantic import ValidationError

    # Named in `match`, and `codec` supplied: `codec` is required, so omitting it
    # raises on its own and the assertion would be satisfied by any input at all.
    with pytest.raises(ValidationError, match=field):
        DesktopHarnessConfig(**{"codec": "deltatype_v2", field: bad})


def test_a_scripted_arm_cannot_name_sampling_it_never_uses() -> None:
    """A scripted arm renders its own action, so a temperature on it would be
    published as the run's sampling and never reach anything. `None` is what says
    "this arm does not sample", and the six control arms depend on it meaning that.
    """
    from pydantic import ValidationError

    for knob in ("temperature", "top_p"):
        with pytest.raises(ValidationError, match="never calls a model"):
            DesktopHarnessConfig(
                codec="deltatype_v2", scripted=ScriptedConfig(enabled=True), **{knob: 0.7}
            )


def test_the_image_budget_config_accepts_only_an_image_count() -> None:
    from pydantic import ValidationError

    for kwargs in ({"max_images": 0}, {"media": "png"}, {"quality": 85}, {"max_pixels": 0}):
        with pytest.raises(ValidationError):
            ImageBudgetConfig(**kwargs)


def test_persist_instruction_is_refused_by_a_policy_that_would_ignore_it() -> None:
    """Only `InterleavedFrames` implements it; the other three took the field and
    dropped it."""
    from pydantic import ValidationError

    assert HistoryConfig(name="prose_summarised_window").persist_instruction is True
    with pytest.raises(ValidationError, match="interleaved_frames only"):
        HistoryConfig(name="prose_summarised_window", persist_instruction=False)
