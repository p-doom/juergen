"""Items 5, 6, 7 — indicator generalisation, the digit lattice, and `_never_moved`.

Item 5: A/B/D now read `native_absolute` / `move_rel` `calls` as well as the bare
grammars' `elements`, and legacy `(kind, value)` element pairs are still accepted so a
cached trace does not silently read 0 (which would look like a fixed defect).

Item 6: `on_lattice` / `delta_histogram` implement a documented finding
(`{0, ±1, ±10, ±100}`, mode `(±10, ±10)` = 14.1 px) whose test was never in-tree.

Item 7: `_never_moved` replaces a `distance=-1.0` sentinel.
"""

from __future__ import annotations

import asyncio
import json
import math

import pytest

from evals.indicators import (
    DIGIT_LATTICE,
    SUBMIT_KEYS,
    FailureModeIndicators,
    MouseIndicators,
    SamplingProvenance,
    delta_histogram,
    deltas,
    on_lattice,
    step_records,
    submit_keys,
    typed_texts,
)
from evals.tasks import RESULT_KEY, DesktopTask
from juergen_doubles import make_task_data, make_trace


def _tool_call(arguments: dict) -> str:
    return (
        "<tool_call>\n"
        + json.dumps({"name": "computer_use", "arguments": arguments})
        + "\n</tool_call>"
    )


def _parsed(codec_name: str, text: str):
    from agent.agent import _action_record, load_codec

    return _action_record(load_codec(codec_name).parse(text))


def _step(text: str, parsed, **extra):
    return {"raw_model_output": text, "parsed_action": parsed, **extra}


class Probe(FailureModeIndicators, MouseIndicators, SamplingProvenance, DesktopTask):
    pass


def _score(steps, *, no_submit=False, **result):
    data = make_task_data(no_submit=no_submit)
    payload = {"steps_detail": steps, **result}
    trace = make_trace(data, episode=payload)
    asyncio.run(Probe(data).score(trace))
    return trace


# =========================================================================== #
# extraction, both shapes
# =========================================================================== #


def test_step_records_ignores_a_missing_or_malformed_result() -> None:
    assert step_records(make_trace()) == []
    assert step_records(make_trace(episode={"steps_detail": "nope"})) == []
    assert step_records(make_trace(episode={"steps_detail": [1, {"a": 1}]})) == [{"a": 1}]


def test_typed_texts_reads_the_bare_token_element_shape() -> None:
    parsed = _parsed("deltatype_v2", '0 0 0 ; type("hello")')
    assert typed_texts(parsed) == ["hello"]


def test_typed_texts_reads_the_tool_call_shape() -> None:
    """Item 5: previously compact-only, so the identical defect went unmeasured natively."""
    for name in ("native_absolute", "move_rel"):
        parsed = _parsed(name, _tool_call({"action": "type", "text": "hello"}))
        assert typed_texts(parsed) == ["hello"], name


def test_typed_texts_reads_the_legacy_kind_value_pair_shape() -> None:
    """A cached pre-consolidation trace must not silently read 0."""
    assert typed_texts({"elements": [("type", "hello"), ("move", (1, 2))]}) == ["hello"]
    assert typed_texts({"elements": [["type", "bye"]]}) == ["bye"]


def test_typed_texts_tolerates_a_non_dict_action() -> None:
    for bad in (None, "text", 3, [], {"elements": "nope"}):
        assert typed_texts(bad) == []


def test_submit_keys_counts_presses_only() -> None:
    """Counting both halves of `+Return -Return` would double every B and D reading."""
    parsed = _parsed("deltatype_v2", "0 0 0 ; +Return -Return")
    assert submit_keys(parsed) == ["Return"]


def test_submit_keys_ignores_a_release_without_a_press() -> None:
    assert submit_keys({"elements": [{"kind": "event", "name": "Return", "pressed": False}]}) == []


def test_submit_keys_reads_the_tool_call_shape_for_key_and_key_down() -> None:
    """The tool-call grammars canonicalise `ENTER` -> `Return` at parse time, so the
    indicator sees the canonical spelling. Both are in `SUBMIT_KEYS`, so a cached
    trace carrying the pre-canonical form still reads as a submission."""
    for name in ("native_absolute", "move_rel"):
        parsed = _parsed(name, _tool_call({"action": "key", "keys": ["ENTER"]}))
        assert submit_keys(parsed) == ["Return"], name
    assert submit_keys({"action": "key", "keys": "Return"}) == ["Return"]
    assert submit_keys({"action": "key_down", "keys": ["Return"]}) == ["Return"]
    assert submit_keys({"action": "key", "keys": ["ENTER"]}) == ["ENTER"]


def test_submit_keys_reads_the_legacy_pair_shape() -> None:
    assert submit_keys({"elements": [("event", ("press", "Return"))]}) == ["Return"]
    assert submit_keys({"elements": [("event", ("release", "Return"))]}) == []


def test_a_non_submit_key_is_not_counted() -> None:
    parsed = _parsed("deltatype_v2", "0 0 0 ; +Tab -Tab")
    assert submit_keys(parsed) == []


def test_the_submit_key_vocabulary_is_case_sensitive() -> None:
    """Recorded, not fixed: `SUBMIT_KEYS` enumerates six spellings and matches exactly,
    so a grammar that ever emitted `return` or `enter` in lower case would read as a
    non-submission. None of the seven in-tree grammars does today (checked below), so
    the risk is a future grammar, not a live miss."""
    assert "return" not in SUBMIT_KEYS and "enter" not in SUBMIT_KEYS
    assert submit_keys({"elements": [{"kind": "event", "name": "return", "pressed": True}]}) == []
    import grammars

    for name in grammars.available():
        codec = grammars.load(name)
        for text in ("0 0 0 ; +Return -Return", _tool_call({"action": "key", "keys": ["ENTER"]})):
            try:
                parsed = _parsed(name, text)
            except Exception:
                continue
            for key in submit_keys(parsed):
                assert key in SUBMIT_KEYS, (name, key)


# =========================================================================== #
# ITEM 5 — A, B, C, D across both grammar families
# =========================================================================== #


@pytest.mark.parametrize(
    "codec_name,text",
    [
        ("deltatype_v2", '0 0 0 ; type("line one\\\\nline two")'),
        ("native_absolute", _tool_call({"action": "type", "text": "line one\\nline two"})),
        ("move_rel", _tool_call({"action": "type", "text": "line one\\nline two"})),
    ],
)
def test_indicator_A_sees_a_literal_escape_under_either_family(codec_name: str, text: str) -> None:
    parsed = _parsed(codec_name, text)
    assert any("\\n" in t for t in typed_texts(parsed)), typed_texts(parsed)
    trace = _score([_step(text, parsed)])
    assert trace.metrics["A_literal_escape_actions"] == 1.0
    assert trace.metrics["A_cell_has_literal_escape"] == 1.0


def test_indicator_A_is_silent_on_a_real_key_transition() -> None:
    text = '0 0 0 ; type("line one") +Return -Return'
    trace = _score([_step(text, _parsed("deltatype_v2", text))])
    assert trace.metrics["A_literal_escape_actions"] == 0.0


@pytest.mark.parametrize("escape", ["\\n", "\\r", "\\t"])
def test_indicator_A_covers_all_three_escapes(escape: str) -> None:
    trace = _score([_step("x", {"elements": [{"kind": "type", "text": f"a{escape}b"}]})])
    assert trace.metrics["A_cell_has_literal_escape"] == 1.0


def test_indicator_B_flags_type_and_submit_in_one_action() -> None:
    """The target composition: `type("...") +Return -Return`."""
    text = '0 0 0 ; type("ls") +Return -Return'
    trace = _score([_step(text, _parsed("deltatype_v2", text))])
    assert trace.metrics["B_same_action_submit_actions"] == 1.0
    assert trace.metrics["B_cell_has_same_action_submit"] == 1.0
    assert trace.info["indicators"]["failure_modes"]["B_examples"] == [text]


def test_indicator_B_is_silent_when_type_and_submit_are_separate_steps() -> None:
    typing = '0 0 0 ; type("ls")'
    submit = "0 0 0 ; +Return -Return"
    trace = _score(
        [
            _step(typing, _parsed("deltatype_v2", typing)),
            _step(submit, _parsed("deltatype_v2", submit)),
        ]
    )
    assert trace.metrics["B_same_action_submit_actions"] == 0.0


def test_indicator_B_reads_a_multi_call_native_turn() -> None:
    """One turn, two calls: the whole point of reading `calls` rather than one action."""
    text = _tool_call({"action": "type", "text": "ls"}) + "\n" + _tool_call(
        {"action": "key", "keys": ["ENTER"]}
    )
    parsed = _parsed("native_absolute", text)
    assert typed_texts(parsed) == ["ls"] and submit_keys(parsed) == ["Return"]
    trace = _score([_step(text, parsed)])
    assert trace.metrics["B_same_action_submit_actions"] == 1.0


def test_indicator_C_is_premature_only_when_the_cell_did_not_succeed() -> None:
    early = _score([], control_terminate="terminate", success=False)
    assert early.metrics["C_terminated"] == 1.0
    assert early.metrics["C_terminated_before_success"] == 1.0
    late = _score([], control_terminate="terminate", success=True)
    assert late.metrics["C_terminated"] == 1.0
    assert late.metrics["C_terminated_before_success"] == 0.0
    never = _score([], control_terminate=None, success=False)
    assert never.metrics["C_terminated"] == 0.0


def test_indicator_C_treats_a_self_declared_fail_as_a_termination() -> None:
    """This is what item 4's normalisation buys: `fail` is recorded and read as one."""
    trace = _score([], control_terminate="fail", success=False)
    assert trace.metrics["C_terminated"] == 1.0
    assert trace.info["indicators"]["failure_modes"]["C_termination_raw"] == "fail"


def test_indicator_D_only_fires_in_a_no_submit_cell() -> None:
    text = "0 0 0 ; +Return -Return"
    step = _step(text, _parsed("deltatype_v2", text))
    assert _score([step], no_submit=False).metrics["D_submitted_in_no_submit_cell"] == 0.0
    flagged = _score([step], no_submit=True)
    assert flagged.metrics["D_submitted_in_no_submit_cell"] == 1.0
    assert flagged.info["indicators"]["failure_modes"]["D_submitted_in_no_submit_cell"] == [text]


def test_indicator_D_reads_a_native_submission_too() -> None:
    text = _tool_call({"action": "key", "keys": ["ENTER"]})
    trace = _score([_step(text, _parsed("native_absolute", text))], no_submit=True)
    assert trace.metrics["D_submitted_in_no_submit_cell"] == 1.0


def test_the_offending_line_recorded_is_the_last_line_of_the_raw_output() -> None:
    raw = 'I will type it.\n0 0 0 ; type("a\\\\nb")'
    trace = _score([_step(raw, _parsed("deltatype_v2", raw))])
    assert trace.info["indicators"]["failure_modes"]["A_offending"] == [raw.splitlines()[-1]]
    assert "I will type it." not in trace.info["indicators"]["failure_modes"]["A_offending"][0]


def test_the_error_counters_ride_the_same_metric() -> None:
    trace = _score([], parse_errors=2, action_errors=1, executor_errors=3)
    assert trace.metrics["parse_errors"] == 2.0
    assert trace.metrics["action_errors"] == 1.0
    assert trace.metrics["executor_errors"] == 3.0


# =========================================================================== #
# ITEM 6 — the digit lattice
# =========================================================================== #


def test_the_lattice_is_the_documented_output_support() -> None:
    assert DIGIT_LATTICE == frozenset({0, 1, 10, 100})


@pytest.mark.parametrize(
    "delta,expected",
    [
        ((0, 0), True),
        ((10, 10), True),
        ((-10, -10), True),
        ((100, -1), True),
        ((0, 100), True),
        ((1, 1), True),
        ((11, 10), False),
        ((10, 11), False),
        ((2, 0), False),
        ((-99, 100), False),
        ((1000, 0), False),
    ],
)
def test_on_lattice_is_per_axis_and_sign_free(delta, expected) -> None:
    assert on_lattice(delta) is expected


def test_the_documented_mode_is_on_lattice_at_14_1_px() -> None:
    assert on_lattice((10, 10))
    assert abs(math.hypot(10, 10) - 14.142) < 0.01


def test_a_healthy_relative_policy_reads_far_off_lattice() -> None:
    arbitrary = [(37, -412), (5, 9), (233, 71), (-88, 14)]
    assert not any(on_lattice(d) for d in arbitrary)


def test_delta_histogram_bins_are_log2_and_every_bin_key_exists() -> None:
    bins = delta_histogram([])
    assert set(bins) == {f"delta_bin_{2 ** k}" for k in range(0, 12, 2)} | {"delta_bin_0"}
    assert all(value == 0.0 for value in bins.values())
    assert "delta_mean_px" not in bins, "no summary statistics from an empty sample"


def test_delta_histogram_separates_a_collapsed_from_a_healthy_distribution() -> None:
    """Bins rather than a mean because the failure mode is bimodal."""
    collapsed = delta_histogram([(10, 10)] * 20)
    assert collapsed["delta_bin_4"] == 20.0, collapsed
    assert collapsed["delta_median_px"] == pytest.approx(14.142, abs=0.01)
    healthy = delta_histogram([(400, 300), (600, 100), (900, 20)])
    assert healthy["delta_bin_256"] == 3.0, healthy
    assert healthy["delta_mean_px"] > 400


def test_delta_histogram_counts_a_zero_delta_in_its_own_bin() -> None:
    bins = delta_histogram([(0, 0), (0, 0), (1, 0)])
    assert bins["delta_bin_0"] == 2.0 and bins["delta_bin_1"] == 1.0
    assert bins["delta_max_px"] == 1.0


def test_delta_histogram_saturates_the_top_bin() -> None:
    bins = delta_histogram([(100000, 100000)])
    assert bins["delta_bin_1024"] == 1.0, "no KeyError above 2**10"


@pytest.mark.parametrize("magnitude_source", [(1, 0), (2, 0), (4, 0), (16, 0), (64, 0), (256, 0), (1024, 0)])
def test_every_integer_magnitude_lands_in_a_declared_bin(magnitude_source) -> None:
    bins = delta_histogram([magnitude_source])
    assert sum(v for k, v in bins.items() if k.startswith("delta_bin_")) == 1.0


def test_deltas_reads_all_three_relative_shapes_and_ignores_absolute() -> None:
    compact = _parsed("deltatype_v2", "10 -20 0 ; +LMB -LMB")
    assert deltas(compact) == [(10, -20)]
    rel = _parsed("move_rel", _tool_call({"action": "move_rel", "coordinate": [5, 7]}))
    assert deltas(rel) == [(5, 7)]
    absolute = _parsed("native_absolute", _tool_call({"action": "left_click", "coordinate": [5, 7]}))
    assert deltas(absolute) == [], (
        "an absolute grammar carries a target, not a delta; differencing targets "
        "would fabricate a distribution"
    )


def test_deltas_reads_the_element_move_shape_and_the_legacy_pair() -> None:
    assert deltas({"elements": [{"kind": "move", "delta": [3, 4]}]}) == [(3, 4)]
    assert deltas({"elements": [("move", (3, 4))]}) == [(3, 4)]


def test_deltas_survives_a_malformed_coordinate() -> None:
    assert deltas({"dx": "a", "dy": 1}) == []
    assert deltas({"elements": [{"kind": "move", "delta": ["x", "y"]}]}) == []
    assert deltas({"action": "move_rel", "coordinate": [1]}) == []
    assert deltas(None) == []


def test_the_mouse_metric_reports_lattice_collapse_end_to_end() -> None:
    steps = [_step("t", _parsed("deltatype_v2", f"{d[0]} {d[1]} 0 ;")) for d in [(10, 10)] * 4]
    trace = _score(steps)
    assert trace.metrics["n_deltas"] == 4.0
    assert trace.metrics["on_lattice_rate"] == 1.0
    assert trace.metrics["zero_delta_rate"] == 0.0
    assert trace.metrics["delta_bin_4"] == 4.0


def test_the_mouse_metric_reports_a_zero_delta_collapse_separately() -> None:
    steps = [_step("t", _parsed("deltatype_v2", "0 0 0 ;"))] * 4
    trace = _score(steps)
    assert trace.metrics["zero_delta_rate"] == 1.0
    assert trace.metrics["on_lattice_rate"] == 1.0, "(0,0) is on the lattice by definition"


def test_the_mouse_metric_is_defined_on_an_empty_rollout() -> None:
    trace = _score([])
    assert trace.metrics["n_deltas"] == 0.0
    assert trace.metrics["on_lattice_rate"] == 0.0
    assert trace.metrics["in_bbox_rate"] == 0.0
    assert trace.metrics["no_op_rate"] == 0.0


def test_the_mouse_metric_reads_no_op_parse_error_and_in_bbox_rates() -> None:
    steps = [
        _step("a", None, control="no_op", probe={"in_bbox": False}),
        _step("b", None, parse_error={"type": "X"}, probe={"in_bbox": True}),
        _step("c", None, control=None, probe={}),
        _step("d", None, control="no_op", probe={"in_bbox": True}),
    ]
    trace = _score(steps, control_terminate="terminate")
    assert trace.metrics["no_op_rate"] == 0.5
    assert trace.metrics["parse_error_rate"] == 0.25
    assert trace.metrics["in_bbox_rate"] == 0.5
    assert trace.metrics["terminate_rate"] == 1.0


# =========================================================================== #
# ITEM 7 — `_never_moved` replaces the distance=-1.0 sentinel
# =========================================================================== #


def test_never_moved_lives_in_the_grounding_taskset_not_in_evals() -> None:
    """Item 7's successor is `rl.grounding.taskset._never_moved`.

    Recorded here because that is not where a reader of `evals/indicators.py` would
    look for it: the `distance = -1.0` sentinel it replaces lived on the *result*, so
    the natural home would be alongside the other indicators. It is exercised in
    `tests/test_rl_tasksets.py`.
    """
    import evals.harness as harness_module
    import evals.indicators as ind
    from rl.grounding.taskset import _never_moved

    assert callable(_never_moved)
    assert not hasattr(ind, "_never_moved") and not hasattr(harness_module, "_never_moved")


def test_the_two_sentinels_the_result_still_carries_are_distinct_fields() -> None:
    """`reach_frame == -1` means "never entered the bbox"; `best_distance == -1.0`
    means "distance undefined". Two fields, so neither has to double as the other —
    which is what the single shared `-1.0` used to be asked to do."""
    from evals.tasks import DesktopState

    state = DesktopState()
    assert state.reach_frame == -1 and state.best_distance == -1.0
    trace = _score([], reach_frame=-1, best_distance=-1.0)
    assert trace.metrics["in_bbox_rate"] == 0.0
    reached = _score([_step("a", None, probe={"in_bbox": True})], reach_frame=1, best_distance=0.0)
    assert reached.metrics["in_bbox_rate"] == 1.0


# =========================================================================== #
# sampling provenance metric
# =========================================================================== #


def test_the_sampling_metric_reports_the_temperature_and_its_source() -> None:
    trace = _score([], sampling={"temperature": 0.25, "temperature_source": "ctx.sampling", "max_tokens": 64})
    assert trace.metrics["temperature"] == 0.25
    assert trace.metrics["temperature_from_ctx_sampling"] == 1.0
    assert trace.metrics["max_tokens"] == 64.0


def test_an_absent_temperature_reads_as_minus_one_not_zero() -> None:
    """0.0 is a real temperature (the parity runs used it), so it cannot double as
    'missing'."""
    trace = _score([], sampling={"temperature": None, "temperature_source": "harness_default"})
    assert trace.metrics["temperature"] == -1.0
    zero = _score([], sampling={"temperature": 0.0, "temperature_source": "harness_default"})
    assert zero.metrics["temperature"] == 0.0
    assert zero.metrics["temperature_from_ctx_sampling"] == 0.0


def test_a_scripted_arm_reports_its_own_source() -> None:
    trace = _score([], sampling={"temperature": None, "temperature_source": "scripted"})
    assert trace.metrics["temperature_from_ctx_sampling"] == 0.0
