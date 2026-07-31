from __future__ import annotations

import copy
import os

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
from ..runner import PairedEvaluationRunner, PairingViolation
from .helpers import (
    evaluation_manifest,
    labctl_context,
    ready_marker,
    sealed_file,
    task_manifest,
)


class FakeSession:
    def __init__(
        self,
        task,
        arm,
        mode,
        prefix,
        fail: bool = False,
        semantic_fail: bool = False,
        reset_mismatch: bool = False,
        execution_status: str = "ok",
        primitive_action_count: int = 1,
    ) -> None:
        self.task = task
        self.arm = arm
        self.mode = mode
        self.prefix = prefix
        self.fail = fail
        self.semantic_fail = semantic_fail
        self.execution_status = execution_status
        self.primitive_action_count = primitive_action_count
        self.turn = 0
        self.cursor = (960, 540) if prefix == 0 else (820, 520)
        self.start = SessionStart(
            task.task_id,
            task.snapshot_id,
            task.parameter_seed,
            task.cursor_ref_for_prefix(prefix),
            self.cursor,
            (
                f"reset-{task.task_id}-{prefix}-{arm.name}"
                if reset_mismatch
                else f"reset-{task.task_id}-{prefix}"
            ),
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
            parse_status=self.execution_status,
            dispatch_status=self.execution_status,
            primitive_action_count=self.primitive_action_count,
        )

    def probe_state(self):
        self.turn += 1
        index = self.prefix + 1 if self.mode == "gold_history_one_step" else self.task.semantic_step_count
        target = (
            self.task.expected_target(self.prefix)
            if self.mode == "gold_history_one_step"
            else self.task.semantic_steps[-1].target_ref
        )
        if self.semantic_fail:
            index = self.prefix
            target = "wrong.semantic.target"
        return {
            "fixture_id": self.task.task_id,
            "fixture_sha256": self.task.fixture_sha256,
            "semantic_step_index": index,
            "target": target,
            "task_solved": index >= self.task.semantic_step_count,
        }

    def close(self):
        return None


class FakeRuntime:
    def __init__(
        self,
        fail_arm: str | None = None,
        semantic_fail_arm: str | None = None,
        semantic_fail_seed: int | None = None,
        reset_mismatch: bool = False,
        execution_failure_arm: str | None = None,
        primitive_action_count: int = 1,
    ) -> None:
        self.fail_arm = fail_arm
        self.semantic_fail_arm = semantic_fail_arm
        self.semantic_fail_seed = semantic_fail_seed
        self.reset_mismatch = reset_mismatch
        self.execution_failure_arm = execution_failure_arm
        self.primitive_action_count = primitive_action_count
        self.contract = {
            "schema": "proper_vm_paired_runtime_v1",
            "runtime_id": "fake-paired-runtime-v1",
            "executor_commit": "6" * 40,
            "interfaces": {
                "native_absolute_control": "native_absolute_sequence_v1",
                "compact_raw_phaseb": "compact_raw_phaseb_v1",
            },
        }

    def open_session(self, *, task, arm, mode, gold_prefix_length, horizon, generation_seed):
        semantic_fail = (
            arm.name == self.semantic_fail_arm
            and generation_seed == self.semantic_fail_seed
        )
        return FakeSession(
            task,
            arm,
            mode,
            gold_prefix_length,
            arm.name == self.fail_arm,
            semantic_fail,
            self.reset_mismatch,
            "error" if arm.name == self.execution_failure_arm else "ok",
            self.primitive_action_count,
        )


def _setup(tmp_path, *, attempts: int = 8):
    marker_path, marker_sha = ready_marker(tmp_path / "EXECUTOR_READY.json")
    tasks, task_seal = task_manifest()
    task_path, _ = sealed_file(tmp_path / "tasks.json", tasks)
    evaluation = evaluation_manifest(task_seal, marker_sha, attempts=attempts)
    eval_path, _ = sealed_file(tmp_path / "evaluation.json", evaluation)
    manifest = load_evaluation_manifest(eval_path, task_path)
    readiness = consume_executor_ready(
        marker_path,
        expected_sha256=marker_sha,
        expected_artifact_id="artifact-executor-ready-test",
        labctl_context_path=labctl_context(tmp_path / "context.json", tmp_path),
    )
    return manifest, readiness


def test_runner_logs_paired_trace_and_semantic_next_state(tmp_path) -> None:
    manifest, readiness = _setup(tmp_path)
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
    assert all(
        arm["turns"][0]["verifier_state"]["oracle_pid"] != os.getpid()
        for arm in row["arms"]
    )
    assert row["first_divergence"] is None
    assert row["first_semantic_divergence"] is None
    assert row["exclusion"]["excluded"] is False


def test_known_infrastructure_failure_excludes_whole_pair_arm_blind(tmp_path) -> None:
    manifest, readiness = _setup(tmp_path)
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


def test_reset_runtime_and_execution_failures_cannot_masquerade_as_success(tmp_path) -> None:
    manifest, readiness = _setup(tmp_path)
    trial = build_plan(manifest)[0]
    with pytest.raises(PairingViolation, match="reset_signature differs"):
        PairedEvaluationRunner(
            manifest,
            readiness,
            FakeRuntime(reset_mismatch=True),
        ).run_trial(trial)

    runtime = FakeRuntime(execution_failure_arm=ARMS[1])
    row = PairedEvaluationRunner(manifest, readiness, runtime).run_trial(trial)
    compact = next(arm for arm in row["arms"] if arm["arm"] == ARMS[1])
    assert compact["success"] is False
    assert compact["infra_failure_class"] is None

    bad_runtime = FakeRuntime()
    bad_runtime.contract["interfaces"][ARMS[1]] = "invented_schema_v1"
    with pytest.raises(PairingViolation, match="runtime identity/interface"):
        PairedEvaluationRunner(manifest, readiness, bad_runtime)


def test_budget_overrun_forces_scored_failure(tmp_path) -> None:
    manifest, readiness = _setup(tmp_path)
    trial = build_plan(manifest)[0]
    row = PairedEvaluationRunner(
        manifest,
        readiness,
        FakeRuntime(primitive_action_count=99),
    ).run_trial(trial)
    assert row["exclusion"]["excluded"] is False
    assert all(arm["success"] is False for arm in row["arms"])
    assert all(arm["budget_failure"] == "primitive_actions_exceeded" for arm in row["arms"])


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
    manifest, readiness = _setup(tmp_path)
    plan = build_plan(manifest)
    runtime = FakeRuntime(
        semantic_fail_arm=ARMS[1],
        semantic_fail_seed=plan[0].generation_seed,
    )
    rows = PairedEvaluationRunner(manifest, readiness, runtime).run(plan)
    report = aggregate_results(manifest, plan, rows)
    assert report["overall"]["mcnemar_descriptive"]["native_success_compact_failure"] == 1
    assert report["overall"]["mcnemar_descriptive"]["exact_two_sided_p_value"] == 1.0
    assert report["pass_at_k_feasibility"]["pass@4"]["feasible"] is True


def test_aggregate_rejects_stored_success_or_verifier_tampering(tmp_path) -> None:
    manifest, readiness = _setup(tmp_path)
    plan = build_plan(manifest)
    trial = plan[0]
    row = PairedEvaluationRunner(manifest, readiness, FakeRuntime()).run_trial(trial)
    tampered = copy.deepcopy(row)
    tampered["arms"][0]["success"] = not tampered["arms"][0]["success"]
    with pytest.raises(ValueError, match="record payload hash mismatch"):
        aggregate_results(manifest, [trial], [tampered])
    tampered = copy.deepcopy(row)
    tampered["arms"][0]["turns"][0]["verifier_state"]["semantic_state"]["target"] = "tampered"
    with pytest.raises(ValueError, match="record payload hash mismatch"):
        aggregate_results(manifest, [trial], [tampered])
