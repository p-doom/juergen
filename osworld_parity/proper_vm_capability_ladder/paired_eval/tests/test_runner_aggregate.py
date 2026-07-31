from __future__ import annotations

import copy
import os
import time

import pytest

from ..aggregate import aggregate_results
from ..contracts import (
    APPROVED_CURRICULUM_COMMIT,
    APPROVED_CURRICULUM_RUNTIME_BINDING_SCHEMA,
    ARMS,
    ExecutionReceipt,
    InfrastructureFailure,
    Observation,
    RequestedAction,
    SessionStart,
    sha256_json,
    StateProbe,
    VerifierState,
)
from ..manifest import load_evaluation_manifest
from ..planning import build_plan
from ..readiness import consume_executor_ready
from ..receipts import validate_binding_receipt, validate_binding_successor
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


def _seal(value, field):
    value[field] = sha256_json(value)
    return value


def _binding(task, arm_name):
    now = time.monotonic_ns()
    fresh_until = now + 30_000_000_000
    provider_session = f"provider-{arm_name}"
    session = f"session-{arm_name}"
    path_sha = sha256_json({"provider": "test"})
    cycles = []
    prior = sha256_json({"generation": arm_name, "index": 0})
    for index in (1, 2):
        after = sha256_json({"generation": arm_name, "index": index})
        started = now - (5 - index * 2) * 1_000_000
        row = {
            "session_id": session,
            "reset_id": f"reset-{arm_name}-{index}",
            "generation_id": after,
            "sequence": index,
            "provider_reset_sequence": index,
            "provider_session_id": provider_session,
            "prior_provider_generation_id": prior,
            "provider_reset_receipt_sha256": sha256_json({"reset": arm_name, "index": index}),
            "provider_state_before_sha256": prior,
            "provider_state_after_sha256": after,
            "provider_path_sha256": path_sha,
            "prior_provider_transition_index": (index - 1) * 2 + 1,
            "new_provider_transition_index": index * 2 + 1,
            "provider_transition_labels": ["loadvm[osworld_ready]", "loadvm_guest_ready"],
            "provider_transition_records_sha256": sha256_json({"transitions": index}),
            "guest_sentinel_path_sha256": sha256_json({"sentinel": "path"}),
            "guest_sentinel_nonce_sha256": sha256_json({"nonce": index}),
            "reset_started_monotonic_ns": started,
            "provider_reset_completed_monotonic_ns": started + 200_000,
            "probe_completed_monotonic_ns": started + 400_000,
            "captured_wall_time_ns": time.time_ns(),
            "vm_snapshot_id": task.snapshot_id,
            "setup_commit": "7" * 40,
            "reset_provider": "/test/provider",
            "transport_endpoint_sha256": sha256_json({"endpoint": arm_name}),
            "probe_sha256": sha256_json({"probe": arm_name, "index": index}),
            "evidence_sha256": sha256_json({"evidence": arm_name, "index": index}),
        }
        cycles.append(row)
        prior = after
    geometry = {"editor": [820, 520]}
    binding_sha = sha256_json({"binding": arm_name, "task": task.task_id})
    return _seal(
        {
            "schema_version": 1,
            "task_id": task.task_id,
            "fixture_sha256": task.fixture_sha256,
            "binding_revision": 1,
            "binding_sha256": binding_sha,
            "parent_binding_sha256": None,
            "refresh_evidence_sha256": None,
            "evidence_fresh_until_monotonic_ns": fresh_until,
            "reset_cycles": cycles,
            "resolved_initial_cursor": [960, 540],
            "initial_geometry": geometry,
            "initial_geometry_sha256": sha256_json(geometry),
            "refresh_transitions": [],
        },
        "binding_receipt_sha256",
    )


def _dispatch_result(action_schema, payload, operation_index, before):
    if action_schema == "native_absolute_sequence_v1":
        operation = payload
        kind = operation["action"]
        coordinate = operation.get("coordinate")
        after = list(coordinate) if coordinate is not None and kind in {
            "click", "mouse_down", "mouse_move", "mouse_up"
        } else list(before)
        operations = []
        if coordinate is not None and kind in {"click", "mouse_down", "mouse_move", "mouse_up"}:
            operations.append({"kind": "move_to", "args": list(coordinate)})
        atomic = None
        if kind == "click":
            atomic_ops = [
                {"kind": "mouse_down", "args": ["left"]},
                {"kind": "mouse_up", "args": ["left"]},
            ]
            operations += atomic_ops
            atomic = {"ok": True, "operations": atomic_ops}
            action_class = "click"
        elif kind == "key_chord":
            operations.append({"kind": "key_chord", "args": list(operation["keys"])})
            action_class = "key_chord"
        else:
            raise AssertionError(kind)
        adapter = "native_absolute_control"
    else:
        after = list(before)
        if payload.startswith("-140 -20"):
            after = [before[0] - 140, before[1] - 20]
            operations = [
                {"kind": "move_to", "args": after},
                {"kind": "mouse_down", "args": ["left"]},
                {"kind": "mouse_up", "args": ["left"]},
            ]
            action_class = "button_hold+button_release+mouse_move"
        elif "+LMB" in payload:
            operations = [
                {"kind": "mouse_down", "args": ["left"]},
                {"kind": "mouse_up", "args": ["left"]},
            ]
            action_class = "button_hold+button_release"
        else:
            operations = [
                {"kind": "key_down", "args": ["ControlLeft"]},
                {"kind": "key_down", "args": ["KeyS"]},
                {"kind": "key_up", "args": ["KeyS"]},
                {"kind": "key_up", "args": ["ControlLeft"]},
            ]
            action_class = "key_chord"
        atomic = {"ok": True, "operations": operations}
        adapter = "compact_raw_phaseb"
    value = {
        "adapter": adapter,
        "parse_status": "ok",
        "executor_dispatch_status": "ok",
        "action_class": action_class,
        "operations": operations,
        "atomic_state": atomic,
        "compiled_payload_sha256": sha256_json(payload),
        "compiled_operation_index": operation_index,
        "cursor_before": list(before),
        "cursor_after": after,
        "atomic_state_sha256": sha256_json(atomic) if atomic is not None else None,
    }
    return _seal(value, "dispatch_result_sha256")


def _segment_receipts(task, arm, binding, semantic_step, actions, cursor_before):
    dispatches = []
    cursor = list(cursor_before)
    events = 0
    for action in actions:
        if arm.action_interface == "native_absolute_sequence_v1":
            results = []
            for index, operation in enumerate(action["operations"]):
                result = _dispatch_result(arm.action_interface, operation, index, cursor)
                cursor = result["cursor_after"]
                events += len(result["operations"])
                results.append(result)
        else:
            result = _dispatch_result(arm.action_interface, action, 0, cursor)
            cursor = result["cursor_after"]
            events += len(result["operations"])
            results = [result]
        dispatches.append(results)
    segment = {
        "task_id": task.task_id,
        "fixture_sha256": task.fixture_sha256,
        "action_schema": arm.action_interface,
        "semantic_step_index": semantic_step,
        "actions": actions,
        "resolved_primitive_actions": len(actions),
        "resolved_primitive_events": events,
        "resolved_budget_sha256": "",
        "binding_revision": binding["binding_revision"],
        "binding_sha256": binding["binding_sha256"],
        "expected_cursor_before": list(cursor_before),
        "expected_cursor_after": cursor,
    }
    budget_payload = dict(segment)
    budget_payload.pop("resolved_budget_sha256")
    budget_payload = {"schema_version": 1, **budget_payload}
    segment["resolved_budget_sha256"] = sha256_json(budget_payload)
    dispatch_sha = sha256_json(
        {
            "schema_version": 1,
            "task_id": task.task_id,
            "semantic_step_index": semantic_step,
            "compiled_actions": actions,
            "dispatches": dispatches,
        }
    )
    started = time.monotonic_ns()
    receipt = _seal(
        {
            "schema_version": 1,
            "task_id": task.task_id,
            "fixture_sha256": task.fixture_sha256,
            "action_schema": arm.action_interface,
            "semantic_step_index": semantic_step,
            "resolved_primitive_actions": len(actions),
            "resolved_primitive_events": events,
            "resolved_budget_sha256": segment["resolved_budget_sha256"],
            "binding_revision": binding["binding_revision"],
            "binding_sha256": binding["binding_sha256"],
            "dispatch_receipt_sha256": dispatch_sha,
            "execution_started_monotonic_ns": started,
            "execution_completed_monotonic_ns": started + 1,
        },
        "executed_receipt_sha256",
    )
    return segment, tuple(tuple(row) for row in dispatches), receipt


def _refreshed_binding(task, binding, executed_receipt_sha256):
    last_probe = binding["reset_cycles"][-1]["probe_completed_monotonic_ns"]
    refresh = {
        "session_id": binding["reset_cycles"][-1]["session_id"],
        "refresh_id": "refresh-test",
        "sequence": 1,
        "task_id": task.task_id,
        "fixture_sha256": task.fixture_sha256,
        "reset_generation_id": binding["reset_cycles"][-1]["generation_id"],
        "completed_step": 2,
        "prior_binding_sha256": binding["binding_sha256"],
        "executed_segment_sha256": executed_receipt_sha256,
        "action_started_monotonic_ns": last_probe + 100,
        "action_completed_monotonic_ns": last_probe + 200,
        "probe_started_monotonic_ns": last_probe + 300,
        "probe_completed_monotonic_ns": last_probe + 400,
        "captured_wall_time_ns": time.time_ns(),
        "before_scroll_y": 0,
        "after_scroll_y": 200,
        "observed_scroll_delta": 200,
        "required_minimum_delta": 100,
        "expected_scroll_direction": "down",
        "probe_sha256": sha256_json({"refresh_probe": True}),
        "issuer_mac": "a" * 64,
    }
    _seal(refresh, "evidence_sha256")
    binding_sha = sha256_json({"binding": "refreshed", "parent": binding["binding_sha256"]})
    transition = _seal(
        {
            "pre_binding_revision": 1,
            "post_binding_revision": 2,
            "pre_binding_sha256": binding["binding_sha256"],
            "post_binding_sha256": binding_sha,
            "refresh_evidence": refresh,
        },
        "transition_receipt_sha256",
    )
    return _seal(
        {
            **{key: copy.deepcopy(value) for key, value in binding.items()
               if key != "binding_receipt_sha256"},
            "binding_revision": 2,
            "binding_sha256": binding_sha,
            "parent_binding_sha256": binding["binding_sha256"],
            "refresh_evidence_sha256": refresh["evidence_sha256"],
            "refresh_transitions": [transition],
        },
        "binding_receipt_sha256",
    )


class FakeSession:
    def __init__(
        self, task, arm, mode, prefix, fail=False, semantic_fail=False,
        reset_mismatch=False, execution_status="ok", primitive_action_count=1,
        cursor_precentered=False, hidden_intervention=False,
        active_window_verified=True, native_click_proof=True, click=False,
        alternative_execution=False, state_probe_intervention=False,
    ) -> None:
        self.task, self.arm, self.mode, self.prefix = task, arm, mode, prefix
        self.fail, self.semantic_fail = fail, semantic_fail
        self.execution_status = execution_status
        self.primitive_action_count = primitive_action_count
        self.hidden_intervention = hidden_intervention
        self.active_window_verified = active_window_verified
        self.native_click_proof = native_click_proof
        self.click, self.alternative_execution = click, alternative_execution
        self.state_probe_intervention = state_probe_intervention
        self.executed_turns = 0
        self.binding = _binding(task, arm.name)
        self.cursor = (960, 540)
        prefix_replay = []
        if prefix:
            prefix_action = (
                {"schema": arm.action_interface, "semantic_step": 1, "operations": [
                    {"action": "click", "coordinate": [820, 520], "button": "left"}
                ]}
                if arm.name == ARMS[0]
                else "-140 -20 0; +LMB -LMB"
            )
            segment, dispatches, receipt = _segment_receipts(
                task, arm, self.binding, 1, [prefix_action], self.cursor
            )
            prefix_replay.append(
                {
                    "semantic_step": 1,
                    "binding_receipt": self.binding,
                    "binding_sha256": self.binding["binding_sha256"],
                    "compiled_segment": segment,
                    "executed_receipt": receipt,
                    "actions": [
                        {"action_index": 0, "screenshot": None, "action": prefix_action,
                         "dispatch": list(dispatches[0])}
                    ],
                }
            )
            self.cursor = (820, 520)
        self.start = SessionStart(
            task.task_id, task.snapshot_id, task.parameter_seed,
            task.cursor_ref_for_prefix(prefix), self.cursor,
            f"reset-{task.task_id}-{prefix}-{arm.name}" if reset_mismatch
            else f"reset-{task.task_id}-{prefix}",
            "live_probe_before_policy", cursor_precentered, self.binding,
            tuple(prefix_replay),
        )

    def observe(self):
        if self.fail:
            raise InfrastructureFailure("observation_capture", "injected fake outage")
        return Observation(
            {"frame": self.executed_turns, "task": self.task.task_id},
            "application/json",
        )

    def request_action(self, *, observation, history, generation_seed, budget):
        if self.click:
            value = (
                {"schema": self.arm.action_interface, "semantic_step": 1,
                 "operations": [{"action": "click", "coordinate": [500, 300], "button": "left"}]}
                if self.arm.name == ARMS[0] else "0 0 0; +LMB -LMB"
            )
        else:
            value = (
                {"schema": self.arm.action_interface, "semantic_step": 1,
                 "operations": [{"action": "key_chord", "keys": ["ControlLeft", "KeyS"]}]}
                if self.arm.name == ARMS[0]
                else "0 0 0; +ControlLeft +KeyS -KeyS -ControlLeft"
            )
        return RequestedAction(
            value, f"{self.arm.name}-{self.executed_turns}", {"output_tokens": 4}
        )

    def execute(self, requested):
        before = self.cursor
        self.executed_turns += 1
        evidence = {
            "cursor_readback_verified": True,
            "interventions_between_policy_turns": (
                [{"kind": "hotkey", "keys": ["ControlLeft", "KeyC"]}]
                if self.hidden_intervention else []
            ),
            "active_window": {
                "verified": self.active_window_verified,
                "method": "x11_getactivewindow",
                "window_id": "0x1234",
                "expected_application": self.task.app,
                "observed_application": self.task.app,
            },
        }
        semantic = (
            ({"kind": "click", "button": "left"},) if self.click
            else ({"kind": "hotkey", "keys": ["ControlLeft", "KeyS"]},)
        )
        if self.execution_status != "ok":
            return ExecutionReceipt(
                executed_action=None, cursor_before=before, cursor_after=before,
                parse_status="error", dispatch_status="error",
                executor_evidence=evidence, binding_receipt=self.binding,
            )
        actions = [requested.value for _ in range(self.primitive_action_count)]
        semantic_step = min(
            self.prefix + (0 if self.semantic_fail else self.executed_turns - 1) + 1,
            self.task.semantic_step_count,
        )
        segment, dispatches, receipt = _segment_receipts(
            self.task, self.arm, self.binding, semantic_step, actions, before
        )
        after = tuple(segment["expected_cursor_after"])
        self.cursor = after
        if self.click and self.arm.name == ARMS[0]:
            evidence.update(
                {
                    "native_click_dispatches": ([{
                        "requested_operation_index": 0,
                        "lowered_operation_index": 0,
                        "requested_coordinate": [500, 300],
                        "dispatched_coordinate": [500, 300],
                        "post_click_cursor": [500, 300],
                    }] if self.native_click_proof else []),
                    "post_action_cursor_verified": self.native_click_proof,
                    "post_action_cursor": list(after),
                }
            )
        lowered = (
            ({"backend": self.arm.action_interface, "source_operation_index": 0,
              "action": "click", "coordinate": [500, 300]},)
            if self.click and self.arm.name == ARMS[0]
            else ({"backend": self.arm.action_interface},)
        )
        return ExecutionReceipt(
            executed_action=(
                {"semantically_equivalent_alternative": True}
                if self.alternative_execution else requested.value
            ),
            cursor_before=before, cursor_after=after,
            action_classes=(("click",) if self.click else ("hotkey",)),
            semantic_operations=semantic, lowered_operations=lowered,
            operations=tuple(
                operation for action_dispatch in dispatches for result in action_dispatch
                for operation in result["operations"]
            ),
            backend_primitives=("key",), executor_evidence=evidence,
            primitive_action_count=len(actions), resolved_actions=tuple(actions),
            semantic_step_index=semantic_step,
            resolved_primitive_actions=len(actions),
            resolved_primitive_events=segment["resolved_primitive_events"],
            resolved_budget_sha256=segment["resolved_budget_sha256"],
            binding_sha256=self.binding["binding_sha256"],
            binding_revision=self.binding["binding_revision"],
            binding_receipt=self.binding, compiled_segment=segment,
            dispatches=dispatches, executed_segment_receipt=receipt,
        )

    def probe_state(self):
        if self.executed_turns == 0:
            index = self.prefix
            target = (
                self.task.semantic_steps[self.prefix - 1].target_ref
                if self.prefix else "initial"
            )
        else:
            index = (
                self.prefix + 1
                if self.mode == "gold_history_one_step"
                else self.task.semantic_step_count
            )
            target = (
                self.task.expected_target(self.prefix)
                if self.mode == "gold_history_one_step"
                else self.task.semantic_steps[-1].target_ref
            )
            if self.semantic_fail:
                index, target = self.prefix, "wrong.semantic.target"
        return StateProbe(
            {"fixture_id": self.task.task_id, "fixture_sha256": self.task.fixture_sha256,
             "semantic_step_index": index, "target": target,
             "task_solved": index >= self.task.semantic_step_count},
            {"read_only": True,
             "input_events": ([{"kind": "hotkey", "keys": ["ControlLeft", "KeyS"]}]
                              if self.state_probe_intervention else []),
             "application": self.task.app, "method": "test_readonly_state_probe"},
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
            "curriculum_commit": APPROVED_CURRICULUM_COMMIT,
            "live_binding": APPROVED_CURRICULUM_RUNTIME_BINDING_SCHEMA,
            "resolved_budget_receipts": "executed_segment_receipt_v1",
            "ordered_executed_aggregate": "compiled_program_receipt_v1",
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


class FakeFreshVerifier:
    def verify(
        self, *, task, state, expected_step_index, expected_target_ref, timeout_seconds
    ):
        index = int(state["semantic_step_index"])
        matches = expected_step_index is None or (
            index == expected_step_index and state["target"] == expected_target_ref
        )
        return VerifierState(
            status="ok",
            task_solved=bool(state["task_solved"] and matches),
            semantic_step_index=index,
            semantic_state=dict(state),
            matched_target_ref=state["target"] if matches else None,
            reason="isolated fake verifier contract",
            oracle_pid=os.getpid() + 100_000,
            verifier_module=task.verifier_module,
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
        verifier=FakeFreshVerifier(),
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


def test_production_runner_accepts_only_the_pinned_approved_binding(tmp_path) -> None:
    manifest, readiness, setup = _setup(tmp_path)
    PairedEvaluationRunner(manifest, readiness, setup, FakeRuntime())
    runtime = FakeRuntime()
    runtime.contract["live_binding"] = "invented_binding_v2"
    with pytest.raises(PairingViolation, match="runtime identity/interface"):
        PairedEvaluationRunner(manifest, readiness, setup, runtime)


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
    with pytest.raises(ValueError, match="turn/executed-segment mismatch"):
        aggregate_results(manifest, [trial], [tampered])
    tampered = copy.deepcopy(row)
    tampered["arms"][0]["turns"][0]["verifier_state"]["semantic_state"]["target"] = "tampered"
    with pytest.raises(ValueError, match="record payload hash mismatch"):
        aggregate_results(manifest, [trial], [tampered])


def test_provider_generation_and_aab_refresh_receipts_are_fail_closed(tmp_path) -> None:
    manifest, _, _ = _setup(tmp_path)
    task = manifest.tasks[0]
    initial = _binding(task, ARMS[0])
    validate_binding_receipt(
        initial,
        task_id=task.task_id,
        fixture_sha256=task.fixture_sha256,
        snapshot_id=task.snapshot_id,
        setup_commit="7" * 40,
        require_fresh=True,
    )
    tampered = copy.deepcopy(initial)
    tampered["reset_cycles"][1]["prior_provider_generation_id"] = "f" * 64
    tampered["binding_receipt_sha256"] = sha256_json(
        {key: value for key, value in tampered.items() if key != "binding_receipt_sha256"}
    )
    with pytest.raises(ValueError, match="provider generation"):
        validate_binding_receipt(
            tampered,
            task_id=task.task_id,
            fixture_sha256=task.fixture_sha256,
            snapshot_id=task.snapshot_id,
            setup_commit="7" * 40,
            require_fresh=True,
        )

    step_2 = "b" * 64
    refreshed = _refreshed_binding(task, initial, step_2)
    validate_binding_receipt(
        refreshed,
        task_id=task.task_id,
        fixture_sha256=task.fixture_sha256,
        snapshot_id=task.snapshot_id,
        setup_commit="7" * 40,
        require_fresh=True,
    )
    validate_binding_successor(
        initial,
        refreshed,
        completed_step_2_receipt_sha256=step_2,
    )
    with pytest.raises(ValueError, match="A/A/B successor"):
        validate_binding_successor(
            initial,
            refreshed,
            completed_step_2_receipt_sha256="c" * 64,
        )


def test_result_writer_never_leaves_partial_output_or_marker(tmp_path) -> None:
    output = tmp_path / "scored-results.jsonl"

    def records():
        yield {"pair_id": "complete"}
        raise RuntimeError("injected mid-shard failure")

    with pytest.raises(RuntimeError, match="mid-shard"):
        write_jsonl_atomic(output, records())
    assert not output.exists()
    assert not output.with_suffix(".jsonl.tmp").exists()
