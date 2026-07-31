from __future__ import annotations

import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "matched_analysis.py"
SPEC = importlib.util.spec_from_file_location("phaseb_matched_analysis", PATH)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def _row(task: str, regime: str, *, a_ok: bool, r_ok: bool) -> dict:
    coord = regime != "non_coordinate"
    return {
        "sample_id": f"{task}-{regime}-{a_ok}-{r_ok}",
        "task_id": task,
        "is_coord_record": coord,
        "distance_regime": regime,
        "absolute": {"parse_ok": a_ok, "action_match": a_ok,
                     "coord_emitted": a_ok if coord else None,
                     "err_px": 0.0 if coord and a_ok else None},
        "relative": {"parse_ok": r_ok, "action_match": r_ok,
                     "coord_emitted": r_ok if coord else None,
                     "err_px": 0.0 if coord and r_ok else None},
    }


def test_distance_regime_boundaries() -> None:
    assert analysis.distance_regime(False, None) == "non_coordinate"
    assert analysis.distance_regime(True, 2) == "stationary_0_2px"
    assert analysis.distance_regime(True, 2.1) == "short_gt2_lt150px"
    assert analysis.distance_regime(True, 150) == "medium_150_lt500px"
    assert analysis.distance_regime(True, 500) == "far_ge500px"


def test_paired_cluster_bootstrap_preserves_direction() -> None:
    rows = [
        _row("a", "stationary_0_2px", a_ok=False, r_ok=True),
        _row("a", "short_gt2_lt150px", a_ok=False, r_ok=True),
        _row("b", "medium_150_lt500px", a_ok=False, r_ok=True),
        _row("b", "non_coordinate", a_ok=False, r_ok=True),
    ]
    result = analysis.bootstrap_ci(
        rows, analysis.PAIRED_METRICS["parse_rate"], n_boot=1000, seed=7)
    assert result["relative_minus_absolute"] == 1
    assert result["paired_task_cluster_bootstrap_95ci"] == [1, 1]


def test_missing_coordinates_are_failures_and_penalized() -> None:
    rows = [
        _row("a", "stationary_0_2px", a_ok=True, r_ok=False),
        _row("b", "short_gt2_lt150px", a_ok=True, r_ok=True),
    ]
    relative = analysis.arm_summary(rows, "relative")
    assert relative["coord_emit_rate"] == 0.5
    assert relative["within_100px_rate"] == 0.5
    assert relative["mean_capped_err_px_missing_as_screen_diagonal"] == (
        analysis.SCREEN_DIAGONAL_PX / 2)
