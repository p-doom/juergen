from __future__ import annotations

import copy

import pytest

from ..aggregate import aggregate_results
from ..contracts import (
    ARMS,
    ExecutionReceipt,
    InfrastructureFailure,
    Observation,
    RequestedAction,
    SessionStart,
    VerifierState,
)
from ..manifest import load_evaluation_manifest
from ..planning import build_plan
from ..readiness import consume_executor_ready
from ..runner import PairedEvaluationRunner
from .helpers import evaluation_manifest, ready_marker, sealed_file, task_manifest


class FakeSession:
    def __init__(self, task, arm, mode, prefix, fail: bool = False) -> None:
        self.task = task
        self.arm = arm
        self.mode = mode
        self.prefix = prefix
        self.fail = fail
        self.turn = 0
        self.cursor = task.cursor_for_prefix(prefix)
        self.start = SessionStart(
            task.task_id,
            task.snapshot_id,
            task.parameter_seed,
            self.cursor,
            f"reset-{task.task_id}-{prefix}",
        )

    def observe(self):
        if self.fail:
            raise InfrastructureFailure("observation_capture", "injected fake outage")
        return Observation({"frame": self.turn, "task": self.task.task_id}, "application/json")

    def request_action(self, *, observation, history, generation_seed, budget):
        value = (
            {"action": "key", "keys": ["ControlLeft", "KeyS"]}
            if self.arm.name == ARMS[0]
            else "0 0 0; +ControlLeft +KeyS -KeyS -ControlLeft"
        )
        return RequestedAction(value, f"{self.arm.name}-{self.turn}", {"output_tokens": 4})

    def execute(self, requested):
        before = self.cursor
        # Canonical semantic operations make the cross-interface action comparable.
        semantic = ({"kind": "hotkey", "keys": ["ControlLeft", "KeyS"]},)
        return ExecutionReceipt(
            executed_action=requested.value,
            cursor_before=before,
            cursor_after=before,
            action_classes=("hotkey",),
            semantic_operations=semantic,
            lowered_operations=({"backend": self.arm.action_interface},),
            operations=({"executed": "key_down/up"},),
            backend_primitives=("key",),
            executor_evidence={"cursor_readback_verified": True},
        )

    def verify(self, *, expected_target_ref):
        self.turn += 1
        index = self.prefix + 1 if self.mode == "gold_history_one_step" else self.task.semantic_step_count
        target = expected_target_ref if expected_target_ref is not None else self.task.semantic_steps[-1].target_ref
        return VerifierState(
            "ok",
            index >= self.task.semantic_step_count,
            index,
            {"semantic_step_index": index, "target": target},
            target,
        )

    def close(self):
        return None


class FakeRuntime:
    def __init__(self, fail_arm: str | None = None) -> None:
        self.fail_arm = fail_arm

    def open_session(self, *, task, arm, mode, gold_prefix_length, horizon, generation_seed):
        return FakeSession(task, arm, mode, gold_prefix_length, arm.name == self.fail_arm)


def _setup(tmp_path, *, attempts: int = 8):
    marker_path, marker_sha = ready_marker(tmp_path / "EXECUTOR_READY.json")
    tasks, task_seal = task_manifest()
    task_path, _ = sealed_file(tmp_path / "tasks.json", tasks)
    evaluation = evaluation_manifest(task_seal, marker_sha, attempts=attempts)
    eval_path, _ = sealed_file(tmp_path / "evaluation.json", evaluation)
    manifest = load_evaluation_manifest(eval_path, task_path)
    readiness = consume_executor_ready(marker_path, expected_sha256=marker_sha)
    return manifest, readiness


def test_runner_logs_paired_trace_and_semantic_next_state(tmp_path) -> None:
    manifest, readiness = _setup(tmp_path, attempts=1)
    trial = next(
        trial for trial in build_plan(manifest) if trial.mode == "gold_history_one_step"
    )
    row = PairedEvaluationRunner(manifest, readiness, FakeRuntime()).run_trial(trial)
    assert row["comparison_scope"] == "complete_system"
    assert row["pairing"]["arm_order"] == list(trial.arm_order)
    assert {arm["arm"] for arm in row["arms"]} == set(ARMS)
    assert all(arm["score_name"] == "semantic_next_state" for arm in row["arms"])
    assert all(arm["success"] is True for arm in row["arms"])
    assert all(arm["turns"][0]["observation_sha256"] for arm in row["arms"])
    assert all("requested_action" in arm["turns"][0] for arm in row["arms"])
    assert all("executed_action" in arm["turns"][0] for arm in row["arms"])
    assert all(arm["turns"][0]["cursor_before"] for arm in row["arms"])
    assert row["first_divergence"] is None
    assert row["first_semantic_divergence"] is None
    assert row["exclusion"]["excluded"] is False


def test_known_infrastructure_failure_excludes_whole_pair_arm_blind(tmp_path) -> None:
    manifest, readiness = _setup(tmp_path, attempts=1)
    trial = build_plan(manifest)[0]
    row = PairedEvaluationRunner(
        manifest,
        readiness,
        FakeRuntime(fail_arm=ARMS[1]),
    ).run_trial(trial)
    assert row["exclusion"] == {
        "excluded": True,
        "policy": "arm_blind_whole_pair_infrastructure_only",
        "infra_failure_classes": ["observation_capture"],
        "decision_inputs_contain_arm_identity": False,
    }
    assert next(arm for arm in row["arms"] if arm["arm"] == ARMS[1])["success"] is None


def test_aggregation_bootstrap_mcnemar_and_pass_at_k(tmp_path) -> None:
    manifest, readiness = _setup(tmp_path, attempts=8)
    plan = build_plan(manifest)
    rows = PairedEvaluationRunner(manifest, readiness, FakeRuntime()).run(plan)
    report = aggregate_results(manifest, plan, rows)
    assert report["overall"]["paired_difference_compact_minus_native"] == 0.0
    assert report["overall"]["paired_task_cluster_bootstrap"]["confidence_interval_95"] == [0.0, 0.0]
    assert report["overall"]["mcnemar_descriptive"]["label"].startswith("descriptive_only")
    assert report["pass_at_k_feasibility"]["pass@1"]["feasible"] is True
    assert report["pass_at_k_feasibility"]["pass@4"]["feasible"] is True
    assert report["pass_at_k_feasibility"]["pass@8"]["feasible"] is True
    assert report["pass_at_k_feasibility"]["pass@8"]["estimate_by_arm"][ARMS[1]] == 1.0

    with pytest.raises(ValueError, match="missing paired results"):
        aggregate_results(manifest, plan, rows[:-1])


def test_mcnemar_is_descriptive_for_a_discordant_pair(tmp_path) -> None:
    manifest, readiness = _setup(tmp_path, attempts=1)
    plan = build_plan(manifest)
    rows = PairedEvaluationRunner(manifest, readiness, FakeRuntime()).run(plan)
    compact = next(arm for arm in rows[0]["arms"] if arm["arm"] == ARMS[1])
    compact["success"] = False
    report = aggregate_results(manifest, plan, rows)
    assert report["overall"]["mcnemar_descriptive"]["native_success_compact_failure"] == 1
    assert report["overall"]["mcnemar_descriptive"]["exact_two_sided_p_value"] == 1.0
    assert report["pass_at_k_feasibility"]["pass@4"]["feasible"] is False
