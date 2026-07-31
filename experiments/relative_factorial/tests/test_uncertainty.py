from __future__ import annotations

import math

from experiments.relative_factorial.capacity import (
    ARMS,
    capacity_contributions,
    factorial_effect,
)
from experiments.relative_factorial.effects import CELLS, calculate
from experiments.relative_factorial.uncertainty import (
    exact_sign_flip_p,
    factorial_contributions,
    failure_overlap,
    vector_diagnostics,
)


def _row(*, value: bool, kind: str = "long") -> dict:
    return {
        "in_box": value,
        "kind": kind,
        "parse_ok": True,
        "schema_ok": True,
        "endpoint_err_px": 0.0 if value else 100.0,
        "coord": [9, 0],
        "ideal_coord": [10, 0],
    }


def test_scene_contributions_match_factorial_calculation():
    values = {
        "abs_tool_act": 1,
        "abs_bare_act": 0,
        "abs_tool_pre": 1,
        "abs_bare_pre": 0,
        "rel_tool_act": 1,
        "rel_bare_act": 1,
        "rel_tool_pre": 0,
        "rel_bare_pre": 0,
    }
    rows = {cell: {"scene": _row(value=bool(value))} for cell, value in values.items()}
    contributions = factorial_contributions(rows, ["scene"])
    expected = calculate({cell: float(value) for cell, value in values.items()})
    assert contributions["grand_mean"] == [expected["grand_mean"]]
    for name, result in expected["effects"].items():
        assert contributions[name] == [result["effect"]]


def test_exact_sign_flip_is_two_sided_and_exact():
    assert exact_sign_flip_p([-0.25] * 4) == 0.125
    assert exact_sign_flip_p([0.0] * 4) == 1.0


def test_vector_error_decomposition_identity():
    diagnostic = vector_diagnostics({"coord": [9, 1], "ideal_coord": [10, 0]})
    assert diagnostic is not None
    ratio = diagnostic["magnitude_ratio"]
    cosine = diagnostic["cosine"]
    total = ratio * ratio + 1.0 - 2.0 * ratio * cosine
    components = (
        diagnostic["normalized_squared_radial_error"]
        + diagnostic["normalized_squared_angular_error"]
    )
    assert math.isclose(total, components, abs_tol=1e-12)


def test_failure_overlap_enumerates_shared_and_unique_scenes():
    scene_ids = ("a", "b", "c")
    rows = {}
    for cell in CELLS:
        rows[cell] = {scene: _row(value=True) for scene in scene_ids}
    rows["rel_bare_act"]["a"] = _row(value=False)
    rows["rel_bare_pre"]["a"] = _row(value=False)
    rows["rel_tool_act"]["b"] = _row(value=False)
    result = failure_overlap(rows)
    assert result["any_relative_failure_count"] == 2
    assert result["all_pass_count"] == 1
    assert result["pairwise"]["rel_bare_act|rel_bare_pre"]["intersection_scene_ids"] == ["a"]


def test_capacity_contrasts_use_conventional_factorial_effect_scaling():
    rows = {
        rank: {arm: {"scene": _row(value=False)} for arm in ARMS}
        for rank in (32, 64, 256)
    }
    rows[64]["reltool_act"]["scene"] = _row(value=True)
    rows[64]["reltool_pre"]["scene"] = _row(value=True)
    rows[256]["reltool_pre"]["scene"] = _row(value=True)

    r64 = capacity_contributions(rows, rank=64, scene_ids=["scene"])
    assert r64["capacity"] == [0.5]
    assert r64["capacity×grammar"] == [1.0]
    assert r64["capacity×preamble"] == [0.0]
    assert r64["capacity×grammar×preamble"] == [0.0]

    r256 = capacity_contributions(rows, rank=256, scene_ids=["scene"])
    assert r256["capacity"] == [0.25]
    assert r256["capacity×grammar"] == [0.5]
    assert r256["capacity×preamble"] == [0.5]
    assert r256["capacity×grammar×preamble"] == [0.5]

    assert factorial_effect({arm: 1.0 for arm in ARMS}, ()) == 1.0
