from __future__ import annotations

from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.fixtures import load_manifest
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.oracle import (
    evaluate_state,
    initial_state,
    reset_signature,
    scripted_state,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.replay import (
    run_build_replay,
)


def test_oracle_rejects_reset_and_near_miss_and_accepts_gold() -> None:
    for fixture in load_manifest("development").fixtures:
        reset = initial_state(fixture)
        assert reset_signature(fixture, reset) == reset_signature(fixture, initial_state(fixture))
        assert evaluate_state(fixture, reset).MOUSE_SOLVED is False
        assert evaluate_state(fixture, scripted_state(fixture, near_miss=True)).MOUSE_SOLVED is False
        assert evaluate_state(fixture, scripted_state(fixture, near_miss=False)).MOUSE_SOLVED is True


def test_build_replay_is_bounded_and_never_opens_sealed_eval() -> None:
    report = run_build_replay("development")
    assert report["status"] == "passed"
    assert report["sealed_eval_executed"] is False
    assert {row["app"] for row in report["rows"]} == {
        "writer",
        "calc",
        "files",
        "chrome",
    }
