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
    resolved_segment_budget_payload,
    SessionStart,
    sha256_json,
    StateProbe,
    VerifierState,
)
from ..manifest import load_evaluation_manifest
from ..planning import build_plan
from ..readiness import consume_executor_ready
from ..runner import PairedEvaluationRunner, PairingViolation, write_jsonl_atomic
from ..setup_validation import consume_task_setup_validation
from .helpers import (
    evaluation_manifest,
    labctl_context,
    ready_marker,
    sealed_file,
    task_setup_validation,
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
        cursor_precentered: bool = False,
        hidden_intervention: bool = False,
        active_window_verified: bool = True,
        native_click_proof: bool = True,
        click: bool = False,
        alternative_execution: bool = False,
        state_probe_intervention: bool = False,
    ) -> None:
        self.task = task
        self.arm = arm
        self.mode = mode
        self.prefix = prefix
        self.fail = fail
        self.semantic_fail = semantic_fail
        self.execution_status = execution_status
        self.primitive_action_count = primitive_action_count
        self.hidden_intervention = hidden_intervention
        self.active_window_verified = active_window_verified
        self.native_click_proof = native_click_proof
        self.click = click
        self.alternative_execution = alternative_execution
        self.state_probe_intervention = state_probe_intervention
        self.turn = 0
        self.cursor = (960, 540) if prefix == 0 else (820, 520)
        self.binding_sha256 = sha256_json(
            {
                "task_id": task.task_id,
                "fixture_sha256": task.fixture_sha256,
                "prefix": prefix,
                "cursor": self.cursor,
                "reset_probe_count": 2,
            }
        )
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
            "live_probe_before_policy",
            cursor_precentered,
            2,
            self.binding_sha256,
            (),
        )

    def observe(self):
        if self.fail:
            raise InfrastructureFailure("observation_capture", "injected fake outage")
        return Observation({"frame": self.turn, "task": self.task.task_id}, "application/json")

    def request_action(self, *, observation, history, generation_seed, budget):
        if self.click:
            value = (
                {
                    "schema": "native_absolute_sequence_v1",
                    "semantic_step": 1,
                    "operations": [
                        {"action": "click", "coordinate": [500, 300], "button": "left"}
                    ],
                }
                if self.arm.name == ARMS[0]
                else "0 0 0; +LMB -LMB"
            )
        else:
            value = (
                {"action": "key", "keys": ["ControlLeft", "KeyS"]}
                if self.arm.name == ARMS[0]
                else "0 0 0; +ControlLeft +KeyS -KeyS -ControlLeft"
            )
        return RequestedAction(value, f"{self.arm.name}-{self.turn}", {"output_tokens": 4})

    def execute(self, requested):
        before = self.cursor
        after = (500, 300) if self.click and self.arm.name == ARMS[0] else before
        executed_action = (
            {"semantically_equivalent_alternative": True}
            if self.alternative_execution
            else requested.value
        )
        resolved_actions = tuple(
            executed_action for _ in range(self.primitive_action_count)
        )
        semantic_step_index = min(
            self.prefix + (0 if self.semantic_fail else self.turn) + 1,
            self.task.semantic_step_count,
        )
        budget_payload = resolved_segment_budget_payload(
            task_id=self.task.task_id,
            fixture_sha256=self.task.fixture_sha256,
            action_schema=self.arm.action_interface,
            semantic_step_index=semantic_step_index,
            actions=resolved_actions,
            resolved_primitive_actions=self.primitive_action_count,
            resolved_primitive_events=1,
            binding_sha256=self.binding_sha256,
        )
        # Canonical semantic operations make the cross-interface action comparable.
        semantic = (
            ({"kind": "click", "button": "left"},)
            if self.click
            else ({"kind": "hotkey", "keys": ["ControlLeft", "KeyS"]},)
        )
        executor_evidence = {
            "cursor_readback_verified": True,
            "interventions_between_policy_turns": (
                [{"kind": "hotkey", "keys": ["ControlLeft", "KeyC"]}]
                if self.hidden_intervention
                else []
            ),
            "active_window": {
                "verified": self.active_window_verified,
                "method": "x11_getactivewindow",
                "window_id": "0x1234",
                "expected_application": self.task.app,
                "observed_application": self.task.app,
            },
        }
        if self.click and self.arm.name == ARMS[0]:
            executor_evidence.update(
                {
                    "native_click_dispatches": (
                        [
                            {
                                "requested_operation_index": 0,
                                "lowered_operation_index": 0,
                                "requested_coordinate": [500, 300],
                                "dispatched_coordinate": [500, 300],
                                "post_click_cursor": [500, 300],
                            }
                        ]
                        if self.native_click_proof
                        else []
                    ),
                    "post_action_cursor_verified": self.native_click_proof,
                    "post_action_cursor": list(after),
                }
            )
        return ExecutionReceipt(
            executed_action=executed_action,
            cursor_before=before,
            cursor_after=after,
            action_classes=(("click",) if self.click else ("hotkey",)),
            semantic_operations=semantic,
            lowered_operations=(
                (
                    {
                        "backend": self.arm.action_interface,
                        "source_operation_index": 0,
                        "action": "click",
                        "coordinate": [500, 300],
                    },
                )
                if self.click and self.arm.name == ARMS[0]
                else ({"backend": self.arm.action_interface},)
            ),
            operations=({"executed": "key_down/up"},),
            backend_primitives=("key",),
            executor_evidence=executor_evidence,
            parse_status=self.execution_status,
            dispatch_status=self.execution_status,
            primitive_action_count=self.primitive_action_count,
            resolved_actions=resolved_actions,
            semantic_step_index=semantic_step_index,
            resolved_primitive_actions=self.primitive_action_count,
            resolved_primitive_events=1,
            resolved_budget_sha256=sha256_json(budget_payload),
            binding_sha256=self.binding_sha256,
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
        return StateProbe(
            {
                "fixture_id": self.task.task_id,
                "fixture_sha256": self.task.fixture_sha256,
                "semantic_step_index": index,
                "target": target,
                "task_solved": index >= self.task.semantic_step_count,
            },
            {
                "read_only": True,
                "input_events": (
                    [{"kind": "hotkey", "keys": ["ControlLeft", "KeyS"]}]
                    if self.state_probe_intervention
                    else []
                ),
                "application": self.task.app,
                "method": "test_readonly_state_probe",
            },
        )

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
        cursor_precentered: bool = False,
        hidden_intervention: bool = False,
        active_window_verified: bool = True,
        native_click_proof: bool = True,
        click: bool = False,
        alternative_execution: bool = False,
        state_probe_intervention: bool = False,
    ) -> None:
        self.fail_arm = fail_arm
        self.semantic_fail_arm = semantic_fail_arm
        self.semantic_fail_seed = semantic_fail_seed
        self.reset_mismatch = reset_mismatch
        self.execution_failure_arm = execution_failure_arm
        self.primitive_action_count = primitive_action_count
        self.cursor_precentered = cursor_precentered
        self.hidden_intervention = hidden_intervention
        self.active_window_verified = active_window_verified
        self.native_click_proof = native_click_proof
        self.click = click
        self.alternative_execution = alternative_execution
        self.state_probe_intervention = state_probe_intervention
        self.contract = {
            "schema": "proper_vm_paired_runtime_v1",
            "runtime_id": "fake-paired-runtime-v1",
            "executor_commit": "6" * 40,
            "interfaces": {
                "native_absolute_control": "native_absolute_sequence_v1",
                "compact_raw_phaseb": "compact_raw_phaseb_v1",
            },
            "cursor_initialization": "live_unmodified_snapshot",
            "native_coordinate_dispatch": "requested_to_lowered_to_post_cursor",
            "between_turn_interventions": "forbidden",
            "active_window_check": "true_active_window_only",
            "live_binding": "provisional_contract_test_only_reject_all",
            "resolved_budget_receipts": "provisional_contract_test_only_reject_all",
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
            self.cursor_precentered,
            self.hidden_intervention,
            self.active_window_verified,
            self.native_click_proof,
            self.click,
            self.alternative_execution,
            self.state_probe_intervention,
        )


def _setup(tmp_path, *, attempts: int = 8):
    marker_path, marker_sha = ready_marker(tmp_path / "EXECUTOR_READY.json")
    tasks, task_seal = task_manifest()
    setup_path, setup_sha = task_setup_validation(
        tmp_path / "task_setup_validation.json", tasks, task_seal
    )
    task_path, _ = sealed_file(tmp_path / "tasks.json", tasks)
    evaluation = evaluation_manifest(
        task_seal,
        marker_sha,
        attempts=attempts,
        setup_validation_sha=setup_sha,
    )
    eval_path, _ = sealed_file(tmp_path / "evaluation.json", evaluation)
    manifest = load_evaluation_manifest(eval_path, task_path)
    readiness = consume_executor_ready(
        marker_path,
        expected_sha256=marker_sha,
        expected_artifact_id="artifact-executor-ready-test",
        labctl_context_path=labctl_context(tmp_path / "context.json", tmp_path),
    )
    setup = consume_task_setup_validation(
        setup_path,
        manifest=manifest,
        labctl_context_path=tmp_path / "context.json",
    )
    return manifest, readiness, setup


def _runner(manifest, readiness, setup, runtime):
    return PairedEvaluationRunner._for_contract_tests(
        manifest,
        readiness,
        setup,
        runtime,
    )


def test_runner_logs_paired_trace_and_semantic_next_state(tmp_path) -> None:
    manifest, readiness, setup = _setup(tmp_path)
    trial = next(
        trial for trial in build_plan(manifest) if trial.mode == "gold_history_one_step"
    )
    row = _runner(manifest, readiness, setup, FakeRuntime()).run_trial(trial)
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


def test_production_runner_is_reject_all_until_binding_approval(tmp_path) -> None:
    manifest, readiness, setup = _setup(tmp_path)
    with pytest.raises(PairingViolation, match="independently approved"):
        PairedEvaluationRunner(manifest, readiness, setup, FakeRuntime())


def test_known_infrastructure_failure_excludes_whole_pair_arm_blind(tmp_path) -> None:
    manifest, readiness, setup = _setup(tmp_path)
    trial = build_plan(manifest)[0]
    row = _runner(
        manifest,
        readiness,
        setup,
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
    manifest, readiness, setup = _setup(tmp_path)
    trial = build_plan(manifest)[0]
    with pytest.raises(PairingViolation, match="reset_signature differs"):
        _runner(
            manifest,
            readiness,
            setup,
            FakeRuntime(reset_mismatch=True),
        ).run_trial(trial)

    runtime = FakeRuntime(execution_failure_arm=ARMS[1])
    row = _runner(manifest, readiness, setup, runtime).run_trial(trial)
    compact = next(arm for arm in row["arms"] if arm["arm"] == ARMS[1])
    assert compact["success"] is False
    assert compact["infra_failure_class"] is None

    bad_runtime = FakeRuntime()
    bad_runtime.contract["interfaces"][ARMS[1]] = "invented_schema_v1"
    with pytest.raises(PairingViolation, match="runtime identity/interface"):
        _runner(manifest, readiness, setup, bad_runtime)


def test_budget_overrun_forces_scored_failure(tmp_path) -> None:
    manifest, readiness, setup = _setup(tmp_path)
    trial = build_plan(manifest)[0]
    row = _runner(
        manifest,
        readiness,
        setup,
        FakeRuntime(primitive_action_count=99),
    ).run_trial(trial)
    assert row["exclusion"]["excluded"] is False
    assert all(arm["success"] is False for arm in row["arms"])
    assert all(arm["budget_failure"] == "primitive_actions_exceeded" for arm in row["arms"])


@pytest.mark.parametrize(
    ("runtime", "message"),
    [
        (FakeRuntime(cursor_precentered=True), "target pre-centering"),
        (FakeRuntime(hidden_intervention=True), "hidden between-turn intervention"),
        (FakeRuntime(active_window_verified=False), "true active-window evidence"),
        (FakeRuntime(state_probe_intervention=True), "input-free/read-only"),
        (FakeRuntime(click=True, native_click_proof=False), "native requested click coordinate"),
    ],
)
def test_runner_rejects_cursor_and_executor_audit_violations(
    tmp_path, runtime, message
) -> None:
    manifest, readiness, setup = _setup(tmp_path)
    with pytest.raises(PairingViolation, match=message):
        _runner(manifest, readiness, setup, runtime).run_trial(
            build_plan(manifest)[0]
        )


def test_native_coordinate_dispatch_uses_post_cursor_and_semantic_state_wins(tmp_path) -> None:
    manifest, readiness, setup = _setup(tmp_path)
    trial = build_plan(manifest)[0]
    row = _runner(
        manifest,
        readiness,
        setup,
        FakeRuntime(click=True, alternative_execution=True),
    ).run_trial(trial)
    native = next(arm for arm in row["arms"] if arm["arm"] == ARMS[0])
    turn = native["turns"][0]
    assert turn["executed_action"] != turn["requested_action"]
    assert turn["cursor_before"] != turn["cursor_after"]
    assert turn["executor_evidence"]["post_action_cursor"] == turn["cursor_after"]
    assert native["success"] is True


def test_aggregation_bootstrap_mcnemar_and_pass_at_k(tmp_path) -> None:
    manifest, readiness, setup = _setup(tmp_path, attempts=8)
    plan = build_plan(manifest)
    rows = _runner(manifest, readiness, setup, FakeRuntime()).run(plan)
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
    manifest, readiness, setup = _setup(tmp_path)
    plan = build_plan(manifest)
    runtime = FakeRuntime(
        semantic_fail_arm=ARMS[1],
        semantic_fail_seed=plan[0].generation_seed,
    )
    rows = _runner(manifest, readiness, setup, runtime).run(plan)
    report = aggregate_results(manifest, plan, rows)
    assert report["overall"]["mcnemar_descriptive"]["native_success_compact_failure"] == 1
    assert report["overall"]["mcnemar_descriptive"]["exact_two_sided_p_value"] == 1.0
    assert report["pass_at_k_feasibility"]["pass@4"]["feasible"] is True


def test_aggregate_rejects_stored_success_or_verifier_tampering(tmp_path) -> None:
    manifest, readiness, setup = _setup(tmp_path)
    plan = build_plan(manifest)
    trial = plan[0]
    row = _runner(manifest, readiness, setup, FakeRuntime()).run_trial(trial)
    tampered = copy.deepcopy(row)
    tampered["arms"][0]["success"] = not tampered["arms"][0]["success"]
    with pytest.raises(ValueError, match="record payload hash mismatch"):
        aggregate_results(manifest, [trial], [tampered])

    tampered = copy.deepcopy(row)
    turn = tampered["arms"][0]["turns"][0]
    turn["resolved_primitive_events"] += 1
    unsigned_turn = dict(turn)
    unsigned_turn.pop("turn_payload_sha256")
    turn["turn_payload_sha256"] = sha256_json(unsigned_turn)
    unsigned_row = dict(tampered)
    unsigned_row.pop("record_payload_sha256")
    tampered["record_payload_sha256"] = sha256_json(unsigned_row)
    with pytest.raises(ValueError, match="live-resolved budget receipt"):
        aggregate_results(manifest, [trial], [tampered])
    tampered = copy.deepcopy(row)
    tampered["arms"][0]["turns"][0]["verifier_state"]["semantic_state"]["target"] = "tampered"
    with pytest.raises(ValueError, match="record payload hash mismatch"):
        aggregate_results(manifest, [trial], [tampered])


def test_result_writer_never_leaves_partial_output_or_marker(tmp_path) -> None:
    output = tmp_path / "scored-results.jsonl"

    def records():
        yield {"pair_id": "complete"}
        raise RuntimeError("injected mid-shard failure")

    with pytest.raises(RuntimeError, match="mid-shard"):
        write_jsonl_atomic(output, records())
    assert not output.exists()
    assert not output.with_suffix(".jsonl.tmp").exists()
