"""Packing-geometry unit tests: runtime-mirroring boundary arithmetic, span
partitioning, goal-mode eligibility and sampling, and the agreement-gate action
comparison. Pure arithmetic on synthetic inputs — no artifact, no frame store,
no labeler.
"""

from __future__ import annotations

import itertools
import sys
from collections import Counter
from pathlib import Path

import pytest

from realigned_pipeline.lib.sequential_packing import (
    DEFAULT_MODE_WEIGHTS,
    MODES,
    MOVE_ZERO_DELTA,
    PackingConfig,
    actions_agree,
    boundary_events,
    eligible_modes,
    packing_config_hash,
    sample_mode,
    segments_from_boundaries,
)

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
from action_parser import _CUA_V4_REQUIRED_ARGS  # noqa: E402

DAY = "u0_20260101"
# fraction_low == fraction_high, and capacity * fraction is exact in binary, so
# the threshold is pinned at 5 and boundaries are checkable by hand.
FIXED = PackingConfig(capacity=10, fraction_low=0.5, fraction_high=0.5, seed=3)
# 3 * 0.3 < 1 -> ceil 1 -> the floor of 2 applies to every segment.
FLOOR = PackingConfig(capacity=3, fraction_low=0.3, fraction_high=0.3, seed=3)
JITTER = PackingConfig(capacity=12, seed=7)
# Config x day-length sweep for the structural invariants.
SWEEP = [
    (PackingConfig(capacity=capacity, fraction_low=low, fraction_high=high, seed=seed),
     n_events)
    for capacity, low, high in ((3, 0.3, 0.3), (4, 0.5, 0.9), (8, 0.5, 0.85),
                                (16, 0.4, 1.0), (32, 0.6, 0.75))
    for seed in (0, 1, 17)
    for n_events in (0, 1, 2, 3, 5, 13, 40, 97)
]


def _node(level: str, start: int, end: int, goal_id: str, text: str) -> dict:
    return {"level": level, "start_event_index": start, "end_event_index": end,
            "goal_id": goal_id, "text": text}


# long over the whole day, two mids partitioning it, one short inside mid 1.
TREE = [
    _node("long", 0, 9, "L1", "Prepare the quarterly report"),
    _node("mid", 0, 4, "M1", "Open the revenue spreadsheet"),
    _node("mid", 5, 9, "M2", "Chart the revenue column"),
    _node("short", 2, 3, "S1", "Type the file name"),
]


def _call(action: str, **arguments) -> dict:
    return {"name": "computer_use", "arguments": {"action": action, **arguments}}


# ---------------------------------------------------------------------------
# PackingConfig + hash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"capacity": 2}, {"capacity": 0}, {"capacity": -1},
    {"capacity": 8, "fraction_low": 0.0},
    {"capacity": 8, "fraction_low": -0.1},
    {"capacity": 8, "fraction_low": 0.9, "fraction_high": 0.5},
    {"capacity": 8, "fraction_high": 1.5},
    {"capacity": 8, "n_packings": 0},
])
def test_packing_config_rejects_impossible_geometry(kwargs) -> None:
    with pytest.raises(ValueError):
        PackingConfig(**kwargs)


def test_packing_config_hash_is_stable_and_field_sensitive() -> None:
    base = PackingConfig(capacity=10)
    digest = packing_config_hash(base)
    assert len(digest) == 64 and int(digest, 16) >= 0
    assert digest == packing_config_hash(PackingConfig(capacity=10))
    variants = [
        {"capacity": 11}, {"fraction_low": 0.55}, {"fraction_high": 0.9},
        {"seed": 1}, {"n_packings": 2},
    ]
    digests = {packing_config_hash(PackingConfig(**{"capacity": 10, **kwargs}))
               for kwargs in variants}
    assert digest not in digests
    assert len(digests) == len(variants)


# ---------------------------------------------------------------------------
# boundary_events
# ---------------------------------------------------------------------------

def test_fixed_fraction_boundaries_match_the_hand_count() -> None:
    # threshold 5 -> a segment holds 5 screenshots, so anchors step by 4.
    assert boundary_events(30, day_tag=DAY, cfg=FIXED) == [4, 8, 12, 16, 20, 24, 28]


def test_threshold_floor_guarantees_progress() -> None:
    boundaries = boundary_events(6, day_tag=DAY, cfg=FLOOR)
    assert boundaries == [1, 2, 3, 4]
    spans = segments_from_boundaries(6, boundaries)
    assert spans == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5)]


def test_boundaries_are_deterministic_per_seed_day_and_packing_index() -> None:
    chain = boundary_events(60, day_tag=DAY, cfg=JITTER)
    assert chain == boundary_events(60, day_tag=DAY, cfg=JITTER)
    assert len(chain) > 1
    by_day = {tuple(boundary_events(60, day_tag=f"u{i}_20260101", cfg=JITTER))
              for i in range(8)}
    by_packing = {tuple(boundary_events(60, day_tag=DAY, cfg=JITTER, packing_index=i))
                  for i in range(8)}
    assert len(by_day) > 1 and len(by_packing) > 1


def test_different_seeds_move_boundaries() -> None:
    chains = {tuple(boundary_events(60, day_tag=DAY,
                                    cfg=PackingConfig(capacity=12, seed=seed)))
              for seed in range(8)}
    assert len(chains) > 1


def test_n_packings_does_not_shift_an_existing_packing() -> None:
    # 03c unions anchors over packings; raising n_packings must not move them.
    for packing_index in range(3):
        chains = {tuple(boundary_events(60, day_tag=DAY, packing_index=packing_index,
                                        cfg=PackingConfig(capacity=12, seed=7,
                                                          n_packings=n_packings)))
                  for n_packings in (3, 5, 9)}
        assert len(chains) == 1


@pytest.mark.parametrize("n_events", [0, 1, 2])
def test_tiny_days_have_no_boundaries(n_events) -> None:
    for cfg in (FIXED, FLOOR, JITTER):
        assert boundary_events(n_events, day_tag=DAY, cfg=cfg) == []
    spans = segments_from_boundaries(n_events, [])
    assert spans == ([] if n_events == 0 else [(0, n_events - 1)])


def test_boundary_never_lands_on_the_final_event() -> None:
    for cfg, n_events in SWEEP:
        boundaries = boundary_events(n_events, day_tag=DAY, cfg=cfg)
        assert boundaries == sorted(set(boundaries)), (cfg, n_events)
        assert all(1 <= boundary <= n_events - 2 for boundary in boundaries), (cfg, n_events)


def test_boundary_events_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="n_events"):
        boundary_events(-1, day_tag=DAY, cfg=FIXED)
    with pytest.raises(ValueError, match="packing_index"):
        boundary_events(10, day_tag=DAY, cfg=FIXED, packing_index=-1)


# ---------------------------------------------------------------------------
# segments_from_boundaries
# ---------------------------------------------------------------------------

def test_spans_partition_the_day_exactly_once() -> None:
    for cfg, n_events in SWEEP:
        boundaries = boundary_events(n_events, day_tag=DAY, cfg=cfg)
        spans = segments_from_boundaries(n_events, boundaries)
        covered = [index for start, end in spans for index in range(start, end + 1)]
        assert covered == list(range(n_events)), (cfg, n_events)
        if n_events == 0:
            assert spans == []
            continue
        assert len(spans) == len(boundaries) + 1, (cfg, n_events)
        assert [start for start, _end in spans[1:]] == boundaries, (cfg, n_events)


def test_segment_screenshots_never_exceed_capacity() -> None:
    # A non-final record also shows the boundary frame in its control turn.
    for cfg, n_events in SWEEP:
        boundaries = boundary_events(n_events, day_tag=DAY, cfg=cfg)
        spans = segments_from_boundaries(n_events, boundaries)
        for index, (start, end) in enumerate(spans):
            n_images = end - start + 1 + (index < len(boundaries))
            assert n_images <= cfg.capacity, (cfg, n_events, index)


@pytest.mark.parametrize(("n_events", "boundaries"), [
    (10, [0]), (10, [3, 3]), (10, [5, 4]), (10, [10]), (10, [-1]), (0, [1]),
])
def test_segments_from_boundaries_rejects_bad_boundaries(n_events, boundaries) -> None:
    with pytest.raises(ValueError):
        segments_from_boundaries(n_events, boundaries)


# ---------------------------------------------------------------------------
# eligible_modes / sample_mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("span", "expected"), [
    ((0, 4), ["explicit_mid", "explicit_long", "proactive"]),
    ((1, 3), ["explicit_mid", "explicit_long", "proactive"]),
    ((5, 9), ["explicit_mid", "explicit_long", "proactive"]),
    ((2, 3), ["explicit_mid", "explicit_long", "proactive"]),  # a short is never a mode
    ((3, 6), ["explicit_long", "proactive"]),                  # two mids, no single one
    ((0, 9), ["explicit_long", "proactive"]),
    ((4, 5), ["explicit_long", "proactive"]),
])
def test_eligible_modes_needs_one_covering_node(span, expected) -> None:
    assert eligible_modes(span, TREE) == expected


def test_proactive_is_the_only_mode_without_goal_coverage() -> None:
    assert eligible_modes((0, 3), []) == ["proactive"]
    partial = [_node("long", 0, 7, "L1", "Prepare the report"),
               _node("mid", 0, 7, "M1", "Open the spreadsheet")]
    assert eligible_modes((6, 9), partial) == ["proactive"]
    assert eligible_modes((7, 7), partial) == ["explicit_mid", "explicit_long", "proactive"]


def test_eligible_modes_rejects_an_empty_span() -> None:
    with pytest.raises(ValueError, match="empty action span"):
        eligible_modes((4, 3), TREE)


def test_sample_mode_is_deterministic_and_order_independent() -> None:
    key = {"seed": 11, "day_tag": DAY, "packing_index": 0, "segment_index": 2}
    mode = sample_mode(list(MODES), DEFAULT_MODE_WEIGHTS, **key)
    assert mode == sample_mode(list(MODES), DEFAULT_MODE_WEIGHTS, **key)
    assert mode == sample_mode(list(reversed(MODES)), DEFAULT_MODE_WEIGHTS, **key)
    drawn = {sample_mode(list(MODES), DEFAULT_MODE_WEIGHTS, seed=11, day_tag=DAY,
                         packing_index=0, segment_index=index)
             for index in range(40)}
    assert drawn == set(MODES)


@pytest.mark.parametrize("field", ["seed", "day_tag", "packing_index", "segment_index"])
def test_sample_mode_is_keyed_by_every_identity_field(field) -> None:
    key = {"seed": 4, "day_tag": DAY, "packing_index": 0, "segment_index": 0}
    values = {"seed": range(24), "day_tag": [f"u{i}_20260101" for i in range(24)],
              "packing_index": range(24), "segment_index": range(24)}[field]
    drawn = {sample_mode(list(MODES), DEFAULT_MODE_WEIGHTS, **{**key, field: value})
             for value in values}
    assert len(drawn) > 1


def test_sample_mode_renormalizes_over_eligible_modes() -> None:
    eligible = ["explicit_mid", "proactive"]
    draws = 4000
    picks = Counter(
        sample_mode(eligible, DEFAULT_MODE_WEIGHTS, seed=5, day_tag=DAY,
                    packing_index=0, segment_index=index)
        for index in range(draws)
    )
    assert set(picks) == set(eligible)
    # 0.45 / (0.45 + 0.30) — the dropped explicit_long mass is redistributed.
    assert picks["explicit_mid"] / draws == pytest.approx(0.6, abs=0.03)


def test_sample_mode_respects_a_single_option_and_zero_mass() -> None:
    weights = {"explicit_mid": 1.0, "proactive": 0.0}
    for index in range(50):
        key = {"seed": 2, "day_tag": DAY, "packing_index": 1, "segment_index": index}
        assert sample_mode(["proactive"], DEFAULT_MODE_WEIGHTS, **key) == "proactive"
        assert sample_mode(["explicit_mid", "proactive"], weights, **key) == "explicit_mid"


@pytest.mark.parametrize(("eligible", "weights", "match"), [
    (["explicit_short"], DEFAULT_MODE_WEIGHTS, "unknown packing mode"),
    ([], DEFAULT_MODE_WEIGHTS, "at least one eligible mode"),
    (["proactive"], {"proactive": 0.0}, "sum to zero"),
    (["proactive"], {}, "sum to zero"),
    (["proactive"], {"proactive": -1.0}, "negative mode weight"),
])
def test_sample_mode_validates_modes_and_weights(eligible, weights, match) -> None:
    with pytest.raises(ValueError, match=match):
        sample_mode(eligible, weights, seed=0, day_tag=DAY, packing_index=0,
                    segment_index=0)


# ---------------------------------------------------------------------------
# actions_agree
# ---------------------------------------------------------------------------

REPRESENTATIVE = {
    "key": {"keys": ["ctrl", "s"]},
    "type": {"text": "hello"},
    "mouse_move_rel": {"delta": [120, -60]},
    "left_click": {}, "right_click": {}, "middle_click": {},
    "double_click": {}, "triple_click": {},
    "button_down": {"button": "left"}, "button_up": {"button": "left"},
    "key_down": {"key": "ctrl"}, "key_up": {"key": "ctrl"},
    "scroll": {"pixels": -120}, "hscroll": {"pixels": 90},
    "wait": {"time": 1.5}, "terminate": {"status": "success"},
}


def test_every_contract_action_has_an_agreement_rule() -> None:
    assert set(REPRESENTATIVE) == set(_CUA_V4_REQUIRED_ARGS)
    for action, arguments in REPRESENTATIVE.items():
        call = [_call(action, **arguments)]
        assert actions_agree(call, call), action
    for first, second in itertools.combinations(sorted(REPRESENTATIVE), 2):
        assert not actions_agree([_call(first, **REPRESENTATIVE[first])],
                                 [_call(second, **REPRESENTATIVE[second])])


@pytest.mark.parametrize(("predicted", "actual", "agree"), [
    # --- list shape -------------------------------------------------------
    pytest.param([], [], True, id="both_empty"),
    pytest.param([_call("left_click")], [], False, id="length_mismatch"),
    pytest.param([_call("left_click")], [_call("left_click"), _call("wait", time=1)],
                 False, id="length_mismatch_suffix"),
    pytest.param([_call("key_down", key="ctrl"), _call("type", text="x"),
                  _call("key_up", key="ctrl")],
                 [_call("key_down", key="ctrl"), _call("type", text="x"),
                  _call("key_up", key="ctrl")], True, id="sequence_identical"),
    pytest.param([_call("type", text="x"), _call("key_down", key="ctrl")],
                 [_call("key_down", key="ctrl"), _call("type", text="x")],
                 False, id="sequence_reordered"),
    pytest.param([{"name": "computer_use"}], [_call("left_click")], False,
                 id="missing_arguments"),
    pytest.param([{"name": "computer_use"}], [{"name": "computer_use"}], False,
                 id="both_malformed"),
    pytest.param([_call("teleport")], [_call("teleport")], False, id="unknown_action"),
    # --- key / key_down / key_up ------------------------------------------
    pytest.param([_call("key", keys=[" Ctrl ", "S"])], [_call("key", keys=["ctrl", "s"])],
                 True, id="key_casefold_and_strip"),
    pytest.param([_call("key", keys="Ctrl")], [_call("key", keys=["ctrl"])], True,
                 id="key_scalar_tolerated"),
    pytest.param([_call("key", keys=5)], [_call("key", keys=["5"])], False,
                 id="key_non_sequence_ignored"),
    pytest.param([_call("key", keys=["ctrl", "s"])], [_call("key", keys=["ctrl", "a"])],
                 False, id="key_different_letter"),
    pytest.param([_call("key", keys=["s", "ctrl"])], [_call("key", keys=["ctrl", "s"])],
                 False, id="key_order_matters"),
    pytest.param([_call("key", keys=["ctrl"])],
                 [_call("key", keys=["ctrl", "shift", "s"])], False, id="key_length"),
    pytest.param([_call("key_down", key="Ctrl")], [_call("key_down", key="ctrl")], True,
                 id="key_down_casefold"),
    pytest.param([_call("key_down", key="alt")], [_call("key_down", key="ctrl")], False,
                 id="key_down_other_key"),
    pytest.param([_call("key_down", key="ctrl")], [_call("key_up", key="ctrl")], False,
                 id="key_down_vs_key_up"),
    # --- type -------------------------------------------------------------
    pytest.param([_call("type", text=" hello   world\n")],
                 [_call("type", text="hello world")], True, id="type_whitespace_normalized"),
    pytest.param([_call("type", text="Hello world")], [_call("type", text="hello world")],
                 False, id="type_case_matters"),
    pytest.param([_call("type", text="hello")], [_call("type", text="helo")], False,
                 id="type_typo"),
    pytest.param([_call("type")], [_call("type", text="hello")], False,
                 id="type_missing_text"),
    # --- mouse_move_rel ---------------------------------------------------
    pytest.param([_call("mouse_move_rel", delta=[100, 0])],
                 [_call("mouse_move_rel", delta=[120, 0])], True, id="move_similar"),
    pytest.param([_call("mouse_move_rel", delta=[100, 0])],
                 [_call("mouse_move_rel", delta=[250, 0])], True, id="move_ratio_low_edge"),
    pytest.param([_call("mouse_move_rel", delta=[100, 0])],
                 [_call("mouse_move_rel", delta=[251, 0])], False, id="move_ratio_low_over"),
    pytest.param([_call("mouse_move_rel", delta=[250, 0])],
                 [_call("mouse_move_rel", delta=[100, 0])], True, id="move_ratio_high_edge"),
    pytest.param([_call("mouse_move_rel", delta=[251, 0])],
                 [_call("mouse_move_rel", delta=[100, 0])], False, id="move_ratio_high_over"),
    pytest.param([_call("mouse_move_rel", delta=[MOVE_ZERO_DELTA, 0])],
                 [_call("mouse_move_rel", delta=[-MOVE_ZERO_DELTA, 0])], False,
                 id="move_zero_delta_boundary_is_signed"),
    pytest.param([_call("mouse_move_rel", delta=[MOVE_ZERO_DELTA - 1, 0])],
                 [_call("mouse_move_rel", delta=[-(MOVE_ZERO_DELTA - 1), 0])], True,
                 id="move_both_near_zero"),
    pytest.param([_call("mouse_move_rel", delta=[39, 0])],
                 [_call("mouse_move_rel", delta=[-500, 0])], False,
                 id="move_near_zero_axis_but_travel_differs"),
    pytest.param([_call("mouse_move_rel", delta=[200, 30])],
                 [_call("mouse_move_rel", delta=[200, -20])], True,
                 id="move_minor_axis_near_zero"),
    pytest.param([_call("mouse_move_rel", delta=[100, 100])],
                 [_call("mouse_move_rel", delta=[100, -100])], False,
                 id="move_axis_sign_mismatch"),
    pytest.param([_call("mouse_move_rel", delta=[0, 0])],
                 [_call("mouse_move_rel", delta=[0, 0])], True, id="move_zero_both"),
    pytest.param([_call("mouse_move_rel", delta=[0, 0])],
                 [_call("mouse_move_rel", delta=[100, 0])], False, id="move_zero_vs_travel"),
    pytest.param([_call("mouse_move_rel")], [_call("mouse_move_rel", delta=[100, 0])],
                 False, id="move_missing_delta"),
    # --- scroll / hscroll -------------------------------------------------
    pytest.param([_call("scroll", pixels=120)], [_call("scroll", pixels=4)], True,
                 id="scroll_same_sign"),
    pytest.param([_call("scroll", pixels=120)], [_call("scroll", pixels=-4)], False,
                 id="scroll_opposite_sign"),
    pytest.param([_call("scroll", pixels=0)], [_call("scroll", pixels=4)], False,
                 id="scroll_zero_vs_signed"),
    pytest.param([_call("scroll", pixels=0)], [_call("scroll", pixels=0)], True,
                 id="scroll_zero_both"),
    pytest.param([_call("hscroll", pixels=-50)], [_call("hscroll", pixels=-5)], True,
                 id="hscroll_same_sign"),
    pytest.param([_call("hscroll", pixels=50)], [_call("scroll", pixels=50)], False,
                 id="hscroll_vs_scroll"),
    # --- clicks and buttons ----------------------------------------------
    pytest.param([_call("left_click")], [_call("left_click")], True, id="click_same"),
    pytest.param([_call("double_click")], [_call("triple_click")], False,
                 id="click_count_differs"),
    pytest.param([_call("button_down", button="Left")],
                 [_call("button_down", button="left")], True, id="button_casefold"),
    pytest.param([_call("button_down", button="left")],
                 [_call("button_down", button="right")], False, id="button_differs"),
    pytest.param([_call("button_down", button="left")],
                 [_call("button_up", button="left")], False, id="button_down_vs_up"),
    # --- wait / terminate -------------------------------------------------
    pytest.param([_call("wait", time=0.5)], [_call("wait", time=12)], True,
                 id="wait_duration_ignored"),
    pytest.param([_call("terminate", status="success")],
                 [_call("terminate", status="SUCCESS")], True, id="terminate_casefold"),
    pytest.param([_call("terminate", status="success")],
                 [_call("terminate", status="failure")], False, id="terminate_status"),
])
def test_actions_agree_per_action_semantics(predicted, actual, agree) -> None:
    assert actions_agree(predicted, actual) is agree
    assert actions_agree(actual, predicted) is agree  # the relation is symmetric
