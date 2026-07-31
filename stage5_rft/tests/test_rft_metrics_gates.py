from __future__ import annotations

import json
from dataclasses import replace

import pytest

from stage5_rft.collector import EpisodeCollector, EpisodeStore
from stage5_rft.gates import evaluate_gates
from stage5_rft.learner import build_learner_plan, validate_resume_state, write_learner_plan
from stage5_rft.metrics import matched_native_absolute_parity, summarize_condition, summarize_separately
from stage5_rft.pipeline import StageJournal
from stage5_rft.rft import RFTConfig, build_rft_dataset
from stage5_rft.util import ContractError, atomic_write_json, read_json, read_jsonl

from conftest import MockActor, MockEnv, make_policy, make_task


def _collect(root, blocklist, policy, tasks, goals):
    collector = EpisodeCollector(
        store=EpisodeStore(root),
        environment=MockEnv(goals),
        actor=MockActor(policy),
        actor_id=f"actor-{policy.role}",
        contamination_blocklist=blocklist,
    )
    collector.collect_many(tasks)
    return EpisodeStore(root).load_all()


def test_rejection_rft_keeps_only_success_and_uses_unit_weight(tmp_path, blocklist):
    rollout = tmp_path / "rollout"
    tasks = [
        make_task("success", task_id="success", max_steps=2),
        make_task("failure", task_id="failure", max_steps=2),
    ]
    _collect(rollout, blocklist, make_policy(), tasks, {"success": 2, "failure": 5})
    manifest = build_rft_dataset(
        rollout_root=rollout,
        output_dir=tmp_path / "dataset",
        blocklist=blocklist,
        config=RFTConfig(val_fraction=0.5),
    )
    rows = read_jsonl(tmp_path / "dataset" / "train.jsonl") + read_jsonl(
        tmp_path / "dataset" / "val.jsonl"
    )
    assert manifest["method"] == "rejection"
    assert manifest["accepted_episode_count"] == 1
    assert manifest["rejection_reasons"] == {"task_not_successful": 1}
    assert rows[0]["weight"] == 1.0 and rows[0]["episode_id"] == "success"


def test_reward_weighting_requires_explicit_experimental_opt_in():
    with pytest.raises(ContractError, match="requires"):
        RFTConfig(mode="reward_weighted").validate()
    RFTConfig(mode="reward_weighted", enable_reward_weighting_experiment=True).validate()


def test_task_level_split_never_scatter_sibling_episodes(tmp_path, blocklist):
    rollout = tmp_path / "rollout"
    tasks = [
        make_task("sibling-a", task_id="shared", max_steps=1),
        make_task("sibling-b", task_id="shared", max_steps=1),
    ]
    _collect(rollout, blocklist, make_policy(), tasks, {"shared": 1})
    build_rft_dataset(
        rollout_root=rollout,
        output_dir=tmp_path / "dataset",
        blocklist=blocklist,
        config=RFTConfig(val_fraction=0.5),
    )
    train = {r["task_id"] for r in read_jsonl(tmp_path / "dataset" / "train.jsonl")}
    val = {r["task_id"] for r in read_jsonl(tmp_path / "dataset" / "val.jsonl")}
    assert not train & val
    assert train | val == {"shared"}


def test_single_and_multi_step_metrics_are_separate(tmp_path, blocklist):
    tasks = [
        make_task("single", task_id="single", max_steps=1),
        make_task("multi", task_id="multi", max_steps=2),
    ]
    episodes = _collect(
        tmp_path / "rollout", blocklist, make_policy(), tasks, {"single": 1, "multi": 2}
    )
    report = summarize_separately(episodes)
    assert set(report) == {"single_step", "multi_step"}
    with pytest.raises(ContractError, match="never be pooled"):
        summarize_condition(episodes)


def test_exact_matched_native_absolute_baseline(tmp_path, blocklist):
    tasks = [
        make_task("single", task_id="single", max_steps=1),
        make_task("multi", task_id="multi", max_steps=2),
    ]
    candidate = _collect(
        tmp_path / "candidate", blocklist, make_policy(), tasks, {"single": 1, "multi": 2}
    )
    baseline_policy = make_policy(
        role="native_absolute_baseline", action_schema="native_absolute.v1"
    )
    baseline = _collect(
        tmp_path / "baseline", blocklist, baseline_policy, tasks, {"single": 1, "multi": 2}
    )
    report = matched_native_absolute_parity(candidate, baseline)
    assert report["single_step"]["pair_coverage"] == 1.0
    assert report["multi_step"]["success_delta_pp"] == 0.0

    mismatched = [replace(baseline[0], policy=replace(baseline[0].policy, sampling={"temperature": 1.0, "top_p": 1.0, "max_tokens": 64}))] + baseline[1:]
    with pytest.raises(ContractError, match="sampling tuple"):
        matched_native_absolute_parity(candidate, mismatched)


def test_missing_gate_metric_fails_closed():
    config = {
        "phases": {
            "construction": [
                {"name": "replay", "metric": "replay.pass_rate", "op": ">=", "threshold": 1.0}
            ]
        }
    }
    missing = evaluate_gates({}, config, phase="construction")
    assert not missing["passed"] and missing["results"][0]["missing"]
    passed = evaluate_gates({"replay": {"pass_rate": 1.0}}, config, phase="construction")
    assert passed["passed"] and not passed["launch_authorized"]


def test_learner_plan_pins_parent_and_resume_identity(tmp_path, blocklist):
    rollout = tmp_path / "rollout"
    _collect(
        rollout,
        blocklist,
        make_policy(),
        [make_task("success", task_id="success", max_steps=1)],
        {"success": 1},
    )
    dataset = tmp_path / "dataset"
    build_rft_dataset(
        rollout_root=rollout,
        output_dir=dataset,
        blocklist=blocklist,
        config=RFTConfig(val_fraction=0.5),
    )
    plan = build_learner_plan(
        dataset_dir=dataset,
        output_checkpoint_dir=tmp_path / "child",
        learner_run_id="learner-1",
        trainer_adapter="mock:trainer",
    )
    assert plan.method == "rejection" and not plan.launch_authorized
    write_learner_plan(plan, tmp_path / "plan.json")
    payload = read_json(tmp_path / "plan.json")
    assert payload["output_eligible_only_next_iteration"] is True

    state = {
        "dataset_manifest_sha256": plan.dataset_manifest_sha256,
        "parent_actor_policy_fingerprint": plan.parent_actor_policy_fingerprint,
        "step": 12,
    }
    atomic_write_json(tmp_path / "state.json", state)
    assert validate_resume_state(plan, tmp_path / "state.json")["step"] == 12
    state["parent_actor_policy_fingerprint"] = "0" * 64
    atomic_write_json(tmp_path / "state.json", state)
    with pytest.raises(ContractError, match="parent checkpoint changed"):
        validate_resume_state(plan, tmp_path / "state.json")


def test_stage_journal_reuses_only_exact_inputs(tmp_path):
    journal = StageJournal(tmp_path)
    assert not journal.reusable("replay", {"collection": "a"})
    journal.complete("replay", inputs={"collection": "a"}, outputs={"passed": True})
    assert journal.reusable("replay", {"collection": "a"})
    with pytest.raises(ContractError, match="different inputs"):
        journal.reusable("replay", {"collection": "b"})
