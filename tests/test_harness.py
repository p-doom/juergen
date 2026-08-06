"""The episode driver, and the "also verify" list.

Every preserved single-runner behaviour is exercised through a real
`DesktopHarness.launch` against a fake pool injected by `pool_target` — which is
what `pool_target` exists for. No VM, no GPU, no network.

Covers: `stop_on_click`, `desktop_setup=terminal`, `reach_frame`, cached-trajectory
replay, per-kind settle (2.0 s Chrome / 0.75 s), the unsolved-start precondition,
`controls_ok`, resume-skip, GIF, prompt sidecars, atomic writes, and that all
prompt/digest raises are gone.
"""

from __future__ import annotations

import asyncio
import json
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
    _Budget,
    _is_left_click,
    _screenshot,
)
from evals.tasks import RESULT_KEY, DesktopState, DesktopTaskData, register_preparer, PREPARERS
from juergen_doubles import FakeSession, make_ctx, make_task_data, make_trace, png


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
    _, result, _ = _run(_config(tmp_path), _task(max_steps=4), replies=["TERMINATE"])
    assert result["outcome"] == "model_terminate_without_postcondition"
    assert result["control_terminate"] == "terminate" and result["terminate_step"] == 1
    assert result["success"] is False


def test_a_self_declared_fail_is_recorded_as_fail_not_terminate(tmp_path, preparer) -> None:
    _, result, _ = _run(_config(tmp_path), _task(max_steps=4), replies=["FAIL"])
    assert result["control_terminate"] == "fail"
    assert result["outcome"] == "model_fail_without_postcondition"


def test_a_parse_error_is_counted_and_the_episode_continues(tmp_path, preparer) -> None:
    _, result, _ = _run(
        _config(tmp_path), _task(max_steps=3), replies=["not an action", "0 0 0 ;", "0 0 0 ;"]
    )
    assert result["parse_errors"] == 1
    assert result["steps"] == 3, "a parse error is a scored outcome, not a stop"
    assert result["steps_detail"][0]["parse_ok"] is False


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
        return Decision(1, "t", "", None, tuple(ops), None, None, sampling)

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
    """The calibration failing loudly is the whole point."""
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
            return png()

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


def test_an_unset_budget_never_fires() -> None:
    budget = _Budget(BudgetConfig())
    for _ in range(1000):
        budget.turn()
        budget.dispatched(100)
        budget.tokens(1000)
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
        "wall_time_s",
        "failure",
    }


def test_frames_prompts_and_result_json_are_written(tmp_path, preparer) -> None:
    task = _task(max_steps=2, name="cell_artifacts")
    _run(_config(tmp_path), task, replies=["0 0 0 ;"] * 2)
    root = tmp_path / "cell_artifacts"
    assert (root / "result.json").is_file()
    assert (root / "steps" / "step_000.png").is_file()
    assert (root / "steps" / "step_001.png").is_file()
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
    _write_gif([png()], target)
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


def test_labctl_registration_is_best_effort(tmp_path, preparer, monkeypatch) -> None:
    """A registry hiccup must not tank a run."""
    from evals import harness as harness_module

    monkeypatch.setattr(harness_module, "_register_labctl", lambda alias, path: False)
    config = _config(
        tmp_path,
        artifacts=ArtifactConfig(
            output_dir=str(tmp_path), register_labctl=True, save_prompts=False, write_gif=False
        ),
    )
    trace, result, _ = _run(config, _task(max_steps=1, name="cell_labctl"), replies=["0 0 0 ;"])
    assert result["validity"] == "valid"
    assert trace.info["artifacts"]["labctl_registered"] is False


def test_register_labctl_survives_a_missing_binary(tmp_path) -> None:
    from evals.harness import _register_labctl

    assert _register_labctl("alias", tmp_path) in (True, False)


def test_the_prompt_report_never_raises_and_records_the_baseline_caveat(tmp_path) -> None:
    """`_prompt_report` returns data; it never raises on a digest mismatch."""
    from agent.agent import load_codec
    from evals import harness as harness_module

    assert not hasattr(harness_module, "_assert_prompt_pin")
    harness = DesktopHarness(_config(tmp_path, system_prompt_sha256="0" * 64))
    report = harness._prompt_report(load_codec("deltatype_v2"))
    assert report["comparable_to_sealed_baseline"] is False
    assert "33.9%" in report["baseline_note"] and "Re-measure" in report["baseline_note"]
    assert report["matches_expected"] is False, "a mismatch is recorded, not raised"
    assert report["expected_prompt_sha256"] == "0" * 64
    assert len(report["prompt_sha256"]) == 64


def test_no_expected_digest_reports_none_rather_than_a_false_match(tmp_path) -> None:
    from agent.agent import load_codec

    report = DesktopHarness(_config(tmp_path))._prompt_report(load_codec("deltatype_v2"))
    assert report["matches_expected"] is None


def test_every_episode_records_the_baseline_caveat(tmp_path, preparer) -> None:
    trace, _, _ = _run(
        _config(tmp_path, system_prompt_sha256="a" * 64), _task(max_steps=1), replies=["0 0 0 ;"]
    )
    prompt = trace.info["prompt"]
    assert prompt["comparable_to_sealed_baseline"] is False
    assert prompt["matches_expected"] is False
    assert "Qwen3-VL-8B=33.9%" in prompt["baseline_note"]
    assert prompt["codec"] == "deltatype_v2"


def test_a_digest_mismatch_does_not_change_the_episode_outcome(tmp_path, preparer) -> None:
    good = _run(_config(tmp_path), _task(max_steps=1, name="a"), replies=["0 0 0 ;"])[1]
    mismatched = _run(
        _config(tmp_path, system_prompt_sha256="f" * 64), _task(max_steps=1, name="b"),
        replies=["0 0 0 ;"],
    )[1]
    assert good["outcome"] == mismatched["outcome"] == "max_steps"
    assert good["validity"] == mismatched["validity"] == "valid"


def test_no_module_in_the_tree_raises_on_a_prompt_digest() -> None:
    repo = Path(__file__).resolve().parents[1]
    for module in ("agent", "evals", "rl"):
        for path in (repo / module).rglob("*.py"):
            text = path.read_text()
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("raise") and (
                    "prompt_sha" in stripped or "sealed_contract" in stripped
                ):
                    raise AssertionError(f"{path}: {stripped}")


def test_a_system_prompt_override_is_honoured_and_hashed(tmp_path) -> None:
    import hashlib

    from agent.agent import load_codec

    harness = DesktopHarness(_config(tmp_path, system_prompt_override="SEALED PROMPT"))
    report = harness._prompt_report(load_codec("deltatype_v2"))
    assert report["prompt_sha256"] == hashlib.sha256(b"SEALED PROMPT").hexdigest()


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
    """`failed=True` is what makes pixeldesk retire a session instead of returning it
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
    """The other half: a healthy VM must NOT be retired after every rollout."""
    captured = _captured_lease(monkeypatch)
    _, result, _ = _run(_config(tmp_path), _task(max_steps=1), replies=["0 0 0 ;"])
    assert result["validity"] == "valid"
    (lease,) = captured
    assert lease.failed is False and lease.error is None


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
    _run(config, _task(max_steps=2), replies=["FAIL"], session=session)
    assert session.declared_terminal == ["fail"]


def test_a_failing_evaluate_is_recorded_as_missing_never_as_zero(tmp_path, preparer) -> None:
    session = FakeSession()  # evaluate_value is None -> raises
    config = _config(tmp_path, evaluate_on_finish=True)
    _, result, _ = _run(config, _task(max_steps=1), replies=["0 0 0 ;"], session=session)
    assert result["task_reward"] is None, "0.0 would be trained as a task failure"


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


def test_the_grounding_taskset_is_the_target_by_regime_cross_product(tmp_path) -> None:
    from evals.tasks import GroundingTaskset, GroundingTasksetConfig

    steps = tmp_path / "run" / "task_a" / "steps"
    steps.mkdir(parents=True)
    (steps / "step_001.png").write_bytes(png())
    bboxes = tmp_path / "bboxes.jsonl"
    bboxes.write_text(
        json.dumps(
            {
                "idx": 0,
                "app": "chrome",
                "instruction": "click it",
                "bbox_xyxy": [10, 20, 30, 40],
                "image_path": str(steps / "step_001.png"),
            }
        )
        + "\n"
    )
    rows = list(
        GroundingTaskset(
            GroundingTasksetConfig(bboxes_jsonl=str(bboxes), osworld_root=str(tmp_path))
        ).load()
    )
    assert len(rows) == 3, "one task per (target, regime)"
    assert [r.data.regime for r in rows] == ["near", "medium", "far"]
    assert all(r.data.bbox == (10, 20, 30, 40) for r in rows)
    assert rows[0].data.name == "chrome/task_a/near"
    assert len({r.data.idx for r in rows}) == 3, "indices must be unique"


def test_a_malformed_image_path_is_refused(tmp_path) -> None:
    from evals.tasks import GroundingTaskset, GroundingTasksetConfig

    bboxes = tmp_path / "bboxes.jsonl"
    bboxes.write_text(
        json.dumps(
            {
                "idx": 0,
                "app": "chrome",
                "instruction": "x",
                "bbox_xyxy": [1, 2, 3, 4],
                "image_path": "/wrong/shape/step_001.png",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="unexpected image_path shape"):
        list(GroundingTaskset(GroundingTasksetConfig(bboxes_jsonl=str(bboxes))).load())


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


@pytest.mark.parametrize(
    "field,bad",
    [
        ("max_tokens", 0),
        ("max_steps", -1),
    ],
)
def test_the_config_validates_its_bounds(field, bad) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DesktopHarnessConfig(**{field: bad})


def test_the_image_budget_config_validates_quality_and_pixels() -> None:
    from pydantic import ValidationError

    for kwargs in ({"quality": 0}, {"quality": 101}, {"max_images": 0}, {"max_pixels": -1}):
        with pytest.raises(ValidationError):
            ImageBudgetConfig(**kwargs)


def test_the_image_budget_config_refuses_a_media_it_cannot_encode() -> None:
    """A bare `str` coerced to jpeg by anything but the word png would make
    `media="webp"` silently produce JPEG."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ImageBudgetConfig(media="webp")


def test_persist_instruction_is_refused_by_a_policy_that_would_ignore_it() -> None:
    """Only `InterleavedFrames` implements it; the other three took the field and
    dropped it."""
    from pydantic import ValidationError

    assert HistoryConfig(name="prose_summarised_window").persist_instruction is True
    with pytest.raises(ValidationError, match="interleaved_frames only"):
        HistoryConfig(name="prose_summarised_window", persist_instruction=False)
