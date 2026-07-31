from __future__ import annotations

from dataclasses import replace

import pytest

from stage5_rft.collector import EpisodeCollector, EpisodeStore
from stage5_rft.contamination import (
    ContaminationBlocklist,
    assert_clean,
    audit_tasks,
)
from stage5_rft.gates import construction_metrics
from stage5_rft.replay import replay_episodes, validate_collection, validate_deterministic_reset
from stage5_rft.util import ContractError

from conftest import MockActor, MockEnv, make_policy, make_task


def _collection(tmp_path, blocklist):
    policy = make_policy()
    collector = EpisodeCollector(
        store=EpisodeStore(tmp_path),
        environment=MockEnv({"t1": 2}),
        actor=MockActor(policy),
        actor_id="actor",
        contamination_blocklist=blocklist,
    )
    collector.collect_many([make_task("ep1", task_id="t1")])
    return EpisodeStore(tmp_path).load_all()


def test_offline_and_live_replay_pass(tmp_path, blocklist):
    episodes = _collection(tmp_path, blocklist)
    assert validate_collection(tmp_path).passed
    live = replay_episodes(episodes, MockEnv({"t1": 2}))
    assert live.passed and live.pass_rate == 1.0


def test_live_replay_detects_state_and_screenshot_drift(tmp_path, blocklist):
    episodes = _collection(tmp_path, blocklist)
    report = replay_episodes(episodes, MockEnv({"t1": 2}, drift=True))
    assert not report.passed
    assert {d.field for d in report.divergences} >= {"state", "screenshot"}


def test_offline_replay_detects_corrupt_artifact(tmp_path, blocklist):
    episodes = _collection(tmp_path, blocklist)
    ref = episodes[0].steps[0].screenshot_before
    (tmp_path / ref.uri).write_bytes(b"corrupt")
    report = validate_collection(tmp_path)
    assert not report.passed
    assert any(d.field == "screenshot_before.sha256" for d in report.divergences)


def test_deterministic_reset_repeated_cpu_mock(tmp_path, blocklist):
    episode = _collection(tmp_path, blocklist)[0]
    report = validate_deterministic_reset(episode, MockEnv({"t1": 2}), repeats=3)
    assert report.passed and report.pass_rate == 1.0


def test_construction_metrics_cover_replay_resume_and_provenance(tmp_path, blocklist):
    episodes = _collection(tmp_path, blocklist)
    live = replay_episodes(episodes, MockEnv({"t1": 2})).as_dict()
    reset = validate_deterministic_reset(episodes[0], MockEnv({"t1": 2})).as_dict()
    metrics = construction_metrics(
        rollout_root=tmp_path,
        blocklist=blocklist,
        live_replay_report=live,
        deterministic_reset_report=reset,
    )
    assert metrics["trace"]["completeness_rate"] == 1.0
    assert metrics["replay"]["pass_rate"] == 1.0
    assert metrics["resume"]["atomic_rate"] == 1.0
    assert metrics["provenance"]["mismatch_count"] == 0


def test_contamination_blocks_id_digest_and_split(blocklist):
    task = make_task("ep1", task_id="blocked")
    by_id = ContaminationBlocklist(frozenset({"blocked"}), frozenset(), "heldout")
    with pytest.raises(ContractError, match="CONTAMINATION"):
        assert_clean(audit_tasks([task], by_id))
    by_digest = ContaminationBlocklist(
        frozenset(), frozenset({task.reset.task_content_sha256}), "heldout"
    )
    with pytest.raises(ContractError, match="content_digests"):
        assert_clean(audit_tasks([task], by_digest))
    with pytest.raises(ContractError, match="not collection-authorized"):
        replace(task.reset, source_split="official_eval").validate()
    empty_production = ContaminationBlocklist(frozenset(), frozenset(), "missing")
    with pytest.raises(ContractError, match="blocklist_usable=False"):
        assert_clean(audit_tasks([task], empty_production))
