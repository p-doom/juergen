from __future__ import annotations

import json
import os

import pytest

from osworld_parity.mixed_action_short_vm.hidden_oracle import (
    evaluate_in_fresh_process,
)
from osworld_parity.mixed_action_short_vm.manifest import load_authorized_tasks
from osworld_parity.mixed_action_short_vm.replay import replay
from osworld_parity.mixed_action_short_vm.runtime import Episode, EpisodeError
from osworld_parity.mixed_action_short_vm.teacher import (
    convert_native_actions,
    native_gold_actions,
)


def test_policy_api_exposes_horizon_reward_done_without_hidden_oracle_values() -> None:
    task = load_authorized_tasks("development")[0]
    episode = Episode(task, "native_absolute_control")
    receipt = episode.reset()
    serialized = json.dumps(receipt.observation.as_dict(), sort_keys=True)
    assert task.target_text not in serialized
    assert "geometry" not in serialized
    assert "expected" not in serialized
    assert receipt.observation.horizon == task.horizon

    actions = native_gold_actions(task)
    first = episode.step(actions[0])
    assert first.reward == 0
    assert first.done is False
    final = episode.step(actions[1])
    assert final.reward == 1
    assert final.done is True
    assert final.truncated is False
    with pytest.raises(EpisodeError, match="after done"):
        episode.step({"action": "wait", "time": 0})


def test_second_reset_removes_contamination_and_hidden_oracle_is_fresh() -> None:
    task = next(
        item
        for item in load_authorized_tasks("development")
        if item.sequence_id == "focus_type_drag"
    )
    episode = Episode(task, "compact_raw_phaseb")
    reset_a = episode.reset()
    first_raw = convert_native_actions(
        (native_gold_actions(task)[0],), task.geometry.initial_cursor
    )[0]
    episode.step(first_raw)
    reset_b = episode.reset()
    assert reset_a.reset_fingerprint == reset_b.reset_fingerprint
    result = evaluate_in_fresh_process(task, episode._trainer_hidden_snapshot())
    assert result.oracle_pid != os.getpid()
    assert result.oracle_status == "ok"
    assert result.MOUSE_SOLVED is False


def test_all_development_gold_and_near_miss_replays_in_both_formats() -> None:
    for task in load_authorized_tasks("development"):
        for arm in ("native_absolute_control", "compact_raw_phaseb"):
            gold = replay(task, arm=arm, near_miss=False)
            miss = replay(task, arm=arm, near_miss=True)
            assert gold.oracle.MOUSE_SOLVED is True
            assert gold.reward == 1
            assert gold.done is True
            assert gold.final_pointer_mask == 0
            assert miss.oracle.MOUSE_SOLVED is False
            assert miss.reward == 0
            assert miss.done is True
            assert miss.truncated is True
            assert miss.action_count <= miss.horizon
