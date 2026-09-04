"""Indicator generalisation and the digit lattice.

The indicators read all three parsed-action containers — `elements` (bare-token),
`primitives` (`ordered_events_v3`), `calls` (tool-call) — and a shape that is none
of them raises instead of reading empty. `test_every_grammar_...` below drives the
loop over `grammars.available()` that stops a fourth shape from appearing unnoticed;
legacy `(kind, value)` element pairs are still accepted so a cached trace does not
silently read 0 (which would look like a fixed defect).

`on_lattice` / `delta_histogram` implement a documented finding
(`{0, ±1, ±10, ±100}`, mode `(±10, ±10)` = 14.1 px).
"""

from __future__ import annotations

import asyncio
import json
import math

import pytest

import grammars
from desktop.geometry import DisplayGeometry
from desktop.ir import Operation
from evals.indicators import (
    DIGIT_LATTICE,
    SUBMIT_KEYS,
    FailureModeIndicators,
    MouseIndicators,
    SamplingProvenance,
    UnreadableAction,
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


def test_an_episode_that_published_no_step_records_reads_as_absent_not_as_zero() -> None:
    """The false zero this test used to assert as correct behaviour.

    Every rate in `MouseIndicators` is a `sum`/`len` over these records, so an
    absent `steps_detail` read as `[]` reported `no_op_rate` and `zero_delta_rate`
    as 0.0 — unmeasured and measured-none were the same number. 24,832 of the
    24,912 archived eval cells were in that state.
    """
    for episode in (None, {}, {"steps_detail": "nope"}, {"steps_detail": {}}):
        trace = make_trace() if episode is None else make_trace(episode=episode)
        assert step_records(trace) is None


def test_step_records_keeps_an_empty_list_and_drops_non_records() -> None:
    assert step_records(make_trace(episode={"steps_detail": []})) == []
    assert step_records(make_trace(episode={"steps_detail": [1, {"a": 1}]})) == [{"a": 1}]


def test_the_indicators_publish_no_rate_at_all_when_the_episode_published_no_steps() -> None:
    """A missing metric is conspicuous; a 0.0 averages into a conclusion."""
    data = make_task_data()
    trace = make_trace(data)
    asyncio.run(Probe(data).score(trace))
    for absent in ("no_op_rate", "zero_delta_rate", "on_lattice_rate", "n_deltas",
                   "dropped_move_rate", "move_requests"):
        assert absent not in trace.metrics, absent
    assert trace.metrics, "the result-derived metrics still report"


_GEOMETRY = DisplayGeometry(desktop_width=1920, desktop_height=1080)
_CURSOR = (100, 200)


def _own_spelling(name: str, operations):
    """What grammar `name` writes for `operations`, as the harness records it.

    Through each codec's own lift rather than a hand-written line per grammar, so
    a grammar cannot drift out of this test's coverage by respelling itself.
    """
    codec = grammars.load(name)
    action = codec.action_from_operations(
        operations, geometry=_GEOMETRY, cursor=_CURSOR
    )
    return action.to_dict()


def test_every_grammar_reports_its_own_typing_and_submission() -> None:
    """The blindness this guards: `typed_texts` sniffed `elements` then `calls`,
    so `ordered_events_v3` — whose container is `primitives` — read `[]` for both,
    and every typing and submission indicator reported zero for the format a live
    training arm was using.
    """
    cannot_type = set()
    for name in grammars.available():
        submission = _own_spelling(
            name, [Operation("key_down", ("Return",)), Operation("key_up", ("Return",))]
        )
        assert submit_keys(submission) == ["Return"], name
        try:
            typing = _own_spelling(name, [Operation("coalesced_type", ("hi",))])
        except ValueError:
            cannot_type.add(name)
            continue
        assert typed_texts(typing) == ["hi"], name
    assert cannot_type == {"diffabs"}, (
        "diffabs is the one grammar with no type() production — it spells literal "
        "text as key transitions. Every other grammar must report what it typed"
    )


def test_every_relative_grammar_reports_its_own_move() -> None:
    """`MouseIndicators` was blind the same way: the lattice-collapse rate read
    zero on `ordered_events_v3` because its move lives in `primitives`."""
    absolute = set()
    for name in grammars.available():
        parsed = _own_spelling(name, [Operation("move_to", (110, 220))])
        read = deltas(parsed)
        if not read:
            absolute.add(name)
            continue
        assert read == [(10, 20)] or name == "move_rel", (name, read)
    assert absolute == {"native_absolute", "compact_absolute"}, (
        "only the two absolute grammars carry a target rather than a delta"
    )


def test_an_unrecognised_action_shape_raises_instead_of_reading_empty() -> None:
    """A silent `[]` is what kept the `primitives` miss invisible for a whole arm."""
    for reader in (typed_texts, submit_keys, deltas):
        with pytest.raises(UnreadableAction, match="publishes exactly one"):
            reader({"no_op": False, "terminate": None})
        with pytest.raises(UnreadableAction, match="publishes exactly one"):
            reader({"elements": [], "primitives": []})
        with pytest.raises(UnreadableAction, match="ordered list"):
            reader({"elements": "nope"})
        with pytest.raises(UnreadableAction, match="to_dict"):
            reader("0 0 0 ; type(\"hi\")")


def test_a_step_that_parsed_nothing_reads_empty_rather_than_raising() -> None:
    """`parsed_action` is None on a parse error and on a bare termination, and both
    are ordinary steps a metric has to score."""
    assert typed_texts(None) == [] and submit_keys(None) == [] and deltas(None) == []


def test_typed_texts_reads_the_bare_token_element_shape() -> None:
    parsed = _parsed("deltatype_v2", '0 0 0 ; type("hello")')
    assert typed_texts(parsed) == ["hello"]


def test_typed_texts_reads_the_ordered_events_v3_primitive_shape() -> None:
    parsed = _parsed("ordered_events_v3", 'type("hello")')
    assert typed_texts(parsed) == ["hello"]
    assert submit_keys(_parsed("ordered_events_v3", "down(Return); up(Return)")) == [
        "Return"
    ]
    assert deltas(_parsed("ordered_events_v3", "move(10,-20); down(LMB); up(LMB)")) == [
        (10, -20)
    ]


def test_typed_texts_reads_the_tool_call_shape() -> None:
    """The same defect must be measurable in a tool-call arm, not compact-only."""
    for name in ("native_absolute", "move_rel"):
        parsed = _parsed(name, _tool_call({"action": "type", "text": "hello"}))
        assert typed_texts(parsed) == ["hello"], name


def test_typed_texts_reads_the_kind_value_pair_shape() -> None:
    """A cached pre-consolidation trace must not silently read 0."""
    assert typed_texts({"elements": [("type", "hello"), ("move", (1, 2))]}) == ["hello"]
    assert typed_texts({"elements": [["type", "bye"]]}) == ["bye"]


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


def test_submit_keys_reads_the_kind_value_pair_shape() -> None:
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
    """One turn, two calls, so `calls` is read rather than a single action."""
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
    """`fail` is recorded, and read as a termination."""
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
    assert deltas({"dx": "a", "dy": 1, "elements": []}) == []
    assert deltas({"elements": [{"kind": "move", "delta": ["x", "y"]}]}) == []
    assert deltas({"primitives": [{"kind": "move", "dx": "x", "dy": "y"}]}) == []
    assert deltas({"action": "move_rel", "coordinate": [1]}) == []


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


def test_the_two_sentinels_the_result_still_carries_are_distinct_fields() -> None:
    """`reach_frame == -1` means "never entered the bbox"; `best_distance == -1.0`
    means "distance undefined". Two fields, so neither has to double as the
    other."""
    from evals.tasks import DesktopState

    state = DesktopState()
    assert state.reach_frame == -1 and state.best_distance == -1.0
    trace = _score([], reach_frame=-1, best_distance=-1.0)
    assert trace.metrics["in_bbox_rate"] == 0.0
    reached = _score([_step("a", None, probe={"in_bbox": True})], reach_frame=1, best_distance=0.0)
    assert reached.metrics["in_bbox_rate"] == 1.0


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


def _dstep(intended, before, operations, **extra):
    return _step(
        "x",
        None,
        intended_cursor=intended,
        cursor_before=list(before),
        operations=list(operations),
        **extra,
    )


_MOVE = {"kind": "move_to", "args": [5, 5]}
_CLICK = ({"kind": "mouse_down", "args": ["left"]}, {"kind": "mouse_up", "args": ["left"]})


def test_the_displacement_metric_sees_a_dropped_move_the_no_op_label_cannot() -> None:
    """The case that makes this metric necessary rather than redundant.

    Both turns request the same displacement and both dispatch a click, so BOTH
    have a non-`no_op` control and `no_op_rate` reads them identically. Only the
    requested-vs-realised comparison separates them. Retrospectively this is 6.16%
    of 186,051 archived movement requests, 2.1x what the label could see, and the
    two worst arms at ~19% had no `no_op` turn at all.
    """
    dropped = _dstep({"x": -95, "y": 990, "clamped": True}, (0, 1079), _CLICK)
    realised = _dstep({"x": 865, "y": 530, "clamped": False}, (960, 540), (_MOVE, *_CLICK))
    trace = _score([dropped, realised], control_terminate=None)
    assert trace.metrics["move_requests"] == 2.0
    assert trace.metrics["moves_dropped"] == 1.0
    assert trace.metrics["dropped_move_rate"] == 0.5
    assert trace.metrics["clamped_request_rate"] == 0.5
    assert trace.metrics["no_op_rate"] == 0.0, (
        "neither turn is a no_op: the click dispatched in both, which is exactly "
        "why a rate over `control` cannot see the dropped move"
    )


def test_the_displacement_metric_does_not_count_a_turn_that_asked_to_stay_put() -> None:
    """The mirror. A zero delta and an absent request are both non-requests, so a
    metric that counted them would report a dropped move for a turn that never
    asked to move."""
    stayed = _dstep({"x": 960, "y": 540, "clamped": False}, (960, 540), ())
    named_nothing = _dstep(None, (960, 540), _CLICK)
    trace = _score([stayed, named_nothing])
    assert trace.metrics["move_requests"] == 0.0
    assert trace.metrics["moves_dropped"] == 0.0
    assert trace.metrics["dropped_move_rate"] == 0.0


def test_a_dropped_move_is_counted_even_when_the_turn_also_terminated() -> None:
    """A terminating turn's move is as erasable as any other, and `control` is
    `terminate` there, so the label is blind to this one too."""
    trace = _score(
        [_dstep({"x": -95, "y": 990, "clamped": True}, (0, 1079), (), control="terminate")]
    )
    assert trace.metrics["move_requests"] == 1.0 and trace.metrics["moves_dropped"] == 1.0
