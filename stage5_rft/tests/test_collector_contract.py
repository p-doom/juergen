from __future__ import annotations

from dataclasses import replace

import pytest

from stage5_rft.collector import EpisodeCollector, EpisodeStore
from stage5_rft.schema import FailureKind
from stage5_rft.util import ContractError, read_json

from conftest import MockActor, MockEnv, digest, make_policy, make_task


def test_collects_complete_per_step_trace_and_manifest(tmp_path, blocklist):
    policy = make_policy()
    actor = MockActor(policy)
    collector = EpisodeCollector(
        store=EpisodeStore(tmp_path),
        environment=MockEnv({"t1": 2}),
        actor=actor,
        actor_id="actor-1",
        contamination_blocklist=blocklist,
    )
    manifest = collector.collect_many([make_task("ep1", task_id="t1")])
    episode = EpisodeStore(tmp_path).load_complete("ep1")
    assert episode is not None
    assert episode.success and episode.total_reward == 1.0
    assert len(episode.steps) == 2
    assert episode.steps[0].state_after.sha256 == episode.steps[1].state_before.sha256
    assert episode.steps[-1].done
    assert manifest["on_policy"] is True
    assert manifest["actor_policy_fingerprint"] == policy.fingerprint
    assert manifest["episodes"]["ep1"] == episode.trace_sha256


def test_incomplete_episode_restarts_from_reset_after_preemption(tmp_path, blocklist):
    policy = make_policy()
    store = EpisodeStore(tmp_path)
    first = EpisodeCollector(
        store=store,
        environment=MockEnv({"t1": 2}),
        actor=MockActor(policy, interrupt_on_call=2),
        actor_id="actor-1",
        contamination_blocklist=blocklist,
    )
    with pytest.raises(KeyboardInterrupt):
        first.collect(make_task("ep1", task_id="t1"))
    partial = read_json(store.partial_path("ep1"))
    assert partial["collection_attempt"] == 1
    assert partial["n_durable_steps"] == 1
    assert store.load_complete("ep1") is None

    second_actor = MockActor(policy)
    second = EpisodeCollector(
        store=store,
        environment=MockEnv({"t1": 2}),
        actor=second_actor,
        actor_id="actor-2",
        contamination_blocklist=blocklist,
    )
    episode = second.collect(make_task("ep1", task_id="t1"))
    assert episode.collection_attempt == 2
    assert episode.steps[0].state_before.payload["position"] == 0
    assert len(episode.steps) == 2


def test_complete_episode_is_idempotently_skipped(tmp_path, blocklist):
    policy = make_policy()
    actor = MockActor(policy)
    collector = EpisodeCollector(
        store=EpisodeStore(tmp_path),
        environment=MockEnv({"t1": 1}),
        actor=actor,
        actor_id="actor",
        contamination_blocklist=blocklist,
    )
    task = make_task("ep1", task_id="t1", max_steps=1)
    first = collector.collect(task)
    second = collector.collect(task)
    assert first.trace_sha256 == second.trace_sha256
    assert actor.calls == 1


def test_policy_version_mismatch_fails_before_commit(tmp_path, blocklist):
    policy = make_policy()
    actor = MockActor(policy, served_fingerprint=digest("f"))
    collector = EpisodeCollector(
        store=EpisodeStore(tmp_path),
        environment=MockEnv(),
        actor=actor,
        actor_id="actor",
        contamination_blocklist=blocklist,
    )
    with pytest.raises(ContractError, match="different checkpoint"):
        collector.collect(make_task("ep1"))
    assert EpisodeStore(tmp_path).load_complete("ep1") is None


def test_invalid_action_is_traced_as_nondispatched_terminal_failure(tmp_path, blocklist):
    policy = make_policy()
    collector = EpisodeCollector(
        store=EpisodeStore(tmp_path),
        environment=MockEnv(),
        actor=MockActor(policy, invalid=True),
        actor_id="actor",
        contamination_blocklist=blocklist,
    )
    episode = collector.collect(make_task("ep1"))
    step = episode.steps[0]
    assert not step.action.valid and not step.action.dispatched
    assert step.failure_kind == FailureKind.PARSE_ERROR
    assert step.screenshot_before.sha256 == step.screenshot_after.sha256


def test_reset_hash_mismatch_fails_closed(tmp_path, blocklist):
    task = make_task("ep1")
    task = replace(
        task,
        reset=replace(task.reset, expected_initial_state_sha256=digest("9")),
    )
    collector = EpisodeCollector(
        store=EpisodeStore(tmp_path),
        environment=MockEnv(),
        actor=MockActor(make_policy()),
        actor_id="actor",
        contamination_blocklist=blocklist,
    )
    with pytest.raises(ContractError, match="deterministic reset state mismatch"):
        collector.collect(task)
