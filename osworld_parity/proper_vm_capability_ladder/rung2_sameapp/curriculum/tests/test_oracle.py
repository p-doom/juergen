from __future__ import annotations

import os

from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.manifests import (
    load_materialized_curriculum,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.oracle import (
    evaluate_in_fresh_process,
    evaluate_state,
    initial_state,
    reset_signature,
    scripted_state,
)


def _tasks():
    return [
        task
        for manifest in load_materialized_curriculum().values()
        for task in manifest.tasks
    ]


def test_every_fixture_rejects_reset_and_near_miss_and_accepts_fresh_gold() -> None:
    for task in _tasks():
        reset_one = initial_state(task)
        reset_two = initial_state(task)
        assert reset_signature(task, reset_one) == reset_signature(task, reset_two)
        assert evaluate_state(task, reset_one).MOUSE_SOLVED is False
        assert evaluate_state(
            task, scripted_state(task, near_miss=True)
        ).MOUSE_SOLVED is False
        gold_state = scripted_state(task, near_miss=False)
        assert gold_state["held_inputs"] == []
        gold = evaluate_in_fresh_process(task, gold_state)
        assert gold.oracle_pid != os.getpid()
        assert gold.oracle_status == "ok"
        assert gold.MOUSE_SOLVED is True


def test_final_oracle_rejects_held_input_even_when_app_state_is_gold() -> None:
    task = _tasks()[0]
    state = scripted_state(task, near_miss=False)
    state["held_inputs"] = ["left"]
    result = evaluate_state(task, state)
    assert result.oracle_status == "ok"
    assert result.MOUSE_SOLVED is False
    assert "not fully released" in result.reason
