from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    ACTION_INTERFACES,
    APPROVED_CURRICULUM_RUNTIME_BINDING_SCHEMA,
    ARMS,
    ExecutionReceipt,
    InfrastructureFailure,
    PairedRuntime,
    RUNTIME_CONTRACT_SCHEMA,
    RequestedAction,
    resolved_segment_budget_payload,
    StateProbe,
    VerifierState,
    canonical_json,
    sha256_json,
)
from .manifest import Arm, EvaluationManifest, Task
from .planning import TrialSpec
from .readiness import ConsumedReadiness
from .setup_validation import ConsumedTaskSetupValidation
from .verifier import FreshProcessTaskVerifier


class PairingViolation(RuntimeError):
    """A harness contract violation that must halt instead of changing a score."""


class _BudgetTracker:
    def __init__(self, limits: dict[str, int | float]) -> None:
        self.limits = limits
        self.started = time.monotonic()
        self.model_turns = 0
        self.primitive_actions = 0
        self.emitted_primitive_events = 0
        self.output_tokens = 0
        self.logical_semantic_steps = 0
        self.failure: str | None = None

    def snapshot(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started
        used = {
            "model_turns": self.model_turns,
            "logical_semantic_steps": self.logical_semantic_steps,
            "primitive_actions": self.primitive_actions,
            "emitted_primitive_events": self.emitted_primitive_events,
            "output_tokens": self.output_tokens,
            "wall_time_seconds": elapsed,
        }
        remaining = {
            "model_turns": max(0, int(self.limits["model_turns"]) - self.model_turns),
            "logical_semantic_steps": max(
                0,
                int(self.limits["logical_semantic_steps"])
                - self.logical_semantic_steps,
            ),
            "primitive_actions": max(
                0, int(self.limits["primitive_actions"]) - self.primitive_actions
            ),
            "emitted_primitive_events": max(
                0,
                int(self.limits["emitted_primitive_events"])
                - self.emitted_primitive_events,
            ),
            "output_tokens": max(
                0, int(self.limits["total_output_tokens"]) - self.output_tokens
            ),
            "wall_time_seconds": max(
                0.0, float(self.limits["wall_time_seconds"]) - elapsed
            ),
        }
        return {"limits": dict(self.limits), "used": used, "remaining": remaining}

    def start_model_turn(self) -> None:
        self.model_turns += 1
        self._limit("model_turns", self.model_turns)
        self._time()

    def add_output_tokens(self, usage: dict[str, int]) -> None:
        tokens = usage.get("output_tokens")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            raise PairingViolation("model usage must report non-negative output_tokens")
        if tokens > int(self.limits["output_tokens_per_turn"]):
            self.failure = "output_tokens_per_turn_exceeded"
        self.output_tokens += tokens
        self._limit("total_output_tokens", self.output_tokens)
        self._time()

    def add_execution(self, receipt: ExecutionReceipt) -> None:
        if (
            not isinstance(receipt.primitive_action_count, int)
            or isinstance(receipt.primitive_action_count, bool)
            or receipt.primitive_action_count < 0
        ):
            raise PairingViolation("executor primitive_action_count is invalid")
        self.primitive_actions += receipt.primitive_action_count
        self.emitted_primitive_events += len(receipt.operations)
        self._limit("primitive_actions", self.primitive_actions)
        self._limit("emitted_primitive_events", self.emitted_primitive_events)
        self._time()

    def add_semantic_progress(self, current: int, prefix: int) -> None:
        self.logical_semantic_steps = max(0, current - prefix)
        self._limit("logical_semantic_steps", self.logical_semantic_steps)
        self._time()

    def _limit(self, key: str, used: int) -> None:
        if used > int(self.limits[key]) and self.failure is None:
            self.failure = f"{key}_exceeded"

    def _time(self) -> None:
        if (
            time.monotonic() - self.started > float(self.limits["wall_time_seconds"])
            and self.failure is None
        ):
            self.failure = "wall_time_seconds_exceeded"


class PairedEvaluationRunner:
    def __init__(
        self,
        manifest: EvaluationManifest,
        readiness: ConsumedReadiness,
        setup_validation: ConsumedTaskSetupValidation,
        runtime: PairedRuntime,
        verifier: FreshProcessTaskVerifier | None = None,
    ) -> None:
        if APPROVED_CURRICULUM_RUNTIME_BINDING_SCHEMA is None:
            raise PairingViolation(
                "production paired execution is disabled until the final curriculum "
                "runtime binding is independently approved"
            )
        self._initialize(manifest, readiness, setup_validation, runtime, verifier)

    @classmethod
    def _for_contract_tests(
        cls,
        manifest: EvaluationManifest,
        readiness: ConsumedReadiness,
        setup_validation: ConsumedTaskSetupValidation,
        runtime: PairedRuntime,
        verifier: FreshProcessTaskVerifier | None = None,
    ) -> "PairedEvaluationRunner":
        """Exercise independent contracts without enabling production execution."""

        value = object.__new__(cls)
        value._initialize(manifest, readiness, setup_validation, runtime, verifier)
        return value

    def _initialize(
        self,
        manifest: EvaluationManifest,
        readiness: ConsumedReadiness,
        setup_validation: ConsumedTaskSetupValidation,
        runtime: PairedRuntime,
        verifier: FreshProcessTaskVerifier | None,
    ) -> None:
        if not readiness.consumed:
            raise PairingViolation("executor readiness was not consumed")
        if readiness.marker_sha256 != manifest.expected_executor_ready_sha256:
            raise PairingViolation("consumed readiness does not match sealed manifest")
        if readiness.artifact_id != manifest.expected_executor_ready_artifact_id:
            raise PairingViolation("consumed readiness artifact does not match sealed manifest")
        if readiness.certification_schema != manifest.expected_executor_certification_schema:
            raise PairingViolation("consumed readiness schema does not match sealed manifest")
        if not setup_validation.consumed:
            raise PairingViolation("task setup validation was not consumed")
        if (
            setup_validation.artifact_sha256
            != manifest.expected_task_setup_validation_sha256
            or setup_validation.artifact_id
            != manifest.expected_task_setup_validation_artifact_id
            or setup_validation.schema_id
            != manifest.expected_task_setup_validation_schema
            or setup_validation.task_manifest_payload_sha256
            != manifest.task_manifest_payload_sha256
            or setup_validation.vm_snapshot_id != readiness.vm_snapshot_id
        ):
            raise PairingViolation(
                "consumed task setup validation does not match sealed dependencies"
            )
        expected_runtime_contract = {
            "schema": RUNTIME_CONTRACT_SCHEMA,
            "runtime_id": manifest.runtime.runtime_id,
            "executor_commit": readiness.executor_commit,
            "interfaces": {
                arm: ACTION_INTERFACES[arm]
                for arm in ARMS
            },
            "cursor_initialization": "live_unmodified_snapshot",
            "native_coordinate_dispatch": "requested_to_lowered_to_post_cursor",
            "between_turn_interventions": "forbidden",
            "active_window_check": "true_active_window_only",
            "live_binding": "provisional_contract_test_only_reject_all",
            "resolved_budget_receipts": "provisional_contract_test_only_reject_all",
        }
        if getattr(runtime, "contract", None) != expected_runtime_contract:
            raise PairingViolation("runtime identity/interface/executor contract mismatch")
        self.runtime_contract = expected_runtime_contract
        snapshots = {task.snapshot_id for task in manifest.tasks}
        if snapshots != {readiness.vm_snapshot_id}:
            raise PairingViolation(
                "task snapshots are not bound to the consumed executor marker: "
                f"{sorted(snapshots)} != {readiness.vm_snapshot_id!r}"
            )
        self.manifest = manifest
        self.readiness = readiness
        self.setup_validation = setup_validation
        self.runtime = runtime
        self.verifier = verifier or FreshProcessTaskVerifier()

    def run(self, plan: Iterable[TrialSpec]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for trial in plan:
            if trial.pair_id in seen:
                raise PairingViolation(f"duplicate pair in execution plan: {trial.pair_id}")
            seen.add(trial.pair_id)
            records.append(self.run_trial(trial))
        return records

    def run_trial(self, trial: TrialSpec) -> dict[str, Any]:
        task = self.manifest.task(trial.task_id)
        self._validate_trial(task, trial)
        by_name: dict[str, dict[str, Any]] = {}
        for arm_name in trial.arm_order:
            by_name[arm_name] = self._run_arm(
                task,
                self.manifest.arm(arm_name),
                trial,
            )
        arms = [by_name[name] for name in ARMS]
        start_cursor_refs = [arm["start_cursor_ref"] for arm in arms]
        start_cursors = [arm["start_cursor"] for arm in arms]
        resolved_initial_cursor = None
        if all(value is not None for value in start_cursor_refs + start_cursors):
            if (
                len(set(start_cursor_refs)) != 1
                or start_cursor_refs[0] != trial.initial_cursor_ref
            ):
                raise PairingViolation(f"{trial.pair_id}: live cursor refs differ across arms")
            if len({tuple(cursor) for cursor in start_cursors}) != 1:
                raise PairingViolation(f"{trial.pair_id}: resolved live cursors differ across arms")
            resolved_initial_cursor = start_cursors[0]
        reset_signatures = [arm["reset_signature"] for arm in arms]
        if all(signature is not None for signature in reset_signatures) and len(
            set(reset_signatures)
        ) != 1:
            raise PairingViolation(
                f"{trial.pair_id}: reset_signature differs across paired arms"
            )
        initial_bindings = [arm["start_binding_sha256"] for arm in arms]
        if all(value is not None for value in initial_bindings) and len(
            set(initial_bindings)
        ) != 1:
            raise PairingViolation(
                f"{trial.pair_id}: initial live binding differs across paired arms"
            )
        reset_probe_counts = [arm["reset_probe_count"] for arm in arms]
        if all(value is not None for value in reset_probe_counts) and any(
            int(value) < 2 for value in reset_probe_counts
        ):
            raise PairingViolation(f"{trial.pair_id}: repeated reset probes are missing")
        infra_classes = sorted(
            {
                result["infra_failure_class"]
                for result in arms
                if result["infra_failure_class"] is not None
            }
        )
        excluded = bool(infra_classes)
        record = {
            "schema_version": 1,
            "record_type": "paired_complete_system_trial",
            "suite": self.manifest.suite,
            "evaluator_commit": self.manifest.evaluator_commit,
            "split": "development",
            "development_only": True,
            "comparison_scope": "complete_system",
            "comparison_label": self.manifest.comparison_label,
            "pair_id": trial.pair_id,
            "cell_id": trial.cell_id,
            "attempt_id": trial.attempt_id,
            "task_id": trial.task_id,
            "family_id": task.family_id,
            "app": task.app,
            "fixture_sha256": trial.fixture_sha256,
            "mode": trial.mode,
            "gold_prefix_length": trial.gold_prefix_length,
            "horizon": trial.horizon,
            "pairing": {
                "snapshot_id": trial.snapshot_id,
                "parameter_seed": trial.parameter_seed,
                "initial_cursor_ref": trial.initial_cursor_ref,
                "initial_cursor": resolved_initial_cursor,
                "generation_seed": trial.generation_seed,
                "sampling_seed_policy": "paired_fixed_per_attempt_v1",
                "budget": trial.budget,
                "arm_order": list(trial.arm_order),
                "shard_index": trial.shard_index,
                "shard_count": trial.shard_count,
            },
            "readiness": {
                "marker_sha256": self.readiness.marker_sha256,
                "artifact_id": self.readiness.artifact_id,
                "certification_schema": self.readiness.certification_schema,
                "capability_report_sha256": self.readiness.capability_report_sha256,
                "executor_commit": self.readiness.executor_commit,
            },
            "task_setup_validation": {
                "artifact_sha256": self.setup_validation.artifact_sha256,
                "artifact_id": self.setup_validation.artifact_id,
                "schema_id": self.setup_validation.schema_id,
                "task_manifest_payload_sha256": (
                    self.setup_validation.task_manifest_payload_sha256
                ),
                "vm_snapshot_id": self.setup_validation.vm_snapshot_id,
                "setup_commit": self.setup_validation.setup_commit,
            },
            "systems": {
                arm.name: {
                    "action_interface": arm.action_interface,
                    "checkpoint": arm.checkpoint,
                    "checkpoint_sha256": arm.checkpoint_sha256,
                    "prompt_id": arm.prompt_id,
                    "prompt_sha256": arm.prompt_sha256,
                    "generation": arm.generation,
                }
                for arm in self.manifest.arms
            },
            "runtime": self.runtime_contract,
            "arms": arms,
            "first_divergence": _first_divergence(arms),
            "first_semantic_divergence": _first_semantic_divergence(arms),
            "exclusion": {
                "excluded": excluded,
                "policy": "arm_blind_whole_pair_infrastructure_only",
                "infra_failure_classes": infra_classes,
                "decision_inputs_contain_arm_identity": False,
            },
        }
        record["record_payload_sha256"] = sha256_json(record)
        return record

    def _validate_trial(self, task: Task, trial: TrialSpec) -> None:
        if trial.fixture_sha256 != task.fixture_sha256:
            raise PairingViolation("trial/task fixture hash mismatch")
        if trial.snapshot_id != task.snapshot_id:
            raise PairingViolation("trial/task snapshot mismatch")
        if trial.parameter_seed != task.parameter_seed:
            raise PairingViolation("trial/task parameter seed mismatch")
        if trial.initial_cursor_ref != task.cursor_ref_for_prefix(trial.gold_prefix_length):
            raise PairingViolation("trial gold-prefix cursor ref mismatch")
        if trial.horizon != trial.budget.get("model_turns"):
            raise PairingViolation("trial horizon/model-turn budget mismatch")
        if set(trial.arm_order) != set(ARMS) or len(trial.arm_order) != 2:
            raise PairingViolation("trial arm order is not a permutation of both arms")
        if (
            trial.mode == "gold_history_one_step"
            and trial.budget.get("logical_semantic_steps") != 1
        ):
            raise PairingViolation("one-step mode must budget one logical semantic step")
        if trial.mode == "natural_closed_loop" and not 2 <= task.semantic_step_count <= 4:
            raise PairingViolation("natural closed loop requires a 2-4-step task")

    def _run_arm(self, task: Task, arm: Arm, trial: TrialSpec) -> dict[str, Any]:
        session = None
        turns: list[dict[str, Any]] = []
        infra_class: str | None = None
        infra_phase: str | None = None
        infra_message: str | None = None
        reset_signature: str | None = None
        start_cursor: tuple[int, int] | None = None
        start_cursor_ref: str | None = None
        start_cursor_source: str | None = None
        start_cursor_precentered: bool | None = None
        reset_probe_count: int | None = None
        start_binding_sha256: str | None = None
        binding_refreshed_after_steps: tuple[int, ...] | None = None
        budget = _BudgetTracker(trial.budget)
        try:
            session = self.runtime.open_session(
                task=task,
                arm=arm,
                mode=trial.mode,
                gold_prefix_length=trial.gold_prefix_length,
                horizon=trial.horizon,
                generation_seed=trial.generation_seed,
            )
            start = session.start
            if start.task_id != task.task_id:
                raise PairingViolation(f"{arm.name}: runtime started the wrong task")
            if start.snapshot_id != trial.snapshot_id:
                raise PairingViolation(f"{arm.name}: runtime started the wrong snapshot")
            if start.parameter_seed != trial.parameter_seed:
                raise PairingViolation(f"{arm.name}: runtime started the wrong task seed")
            if start.cursor_ref != trial.initial_cursor_ref:
                raise PairingViolation(
                    f"{arm.name}: cursor ref mismatch {start.cursor_ref} != {trial.initial_cursor_ref}"
                )
            if start.cursor_source != "live_probe_before_policy" or start.cursor_precentered:
                raise PairingViolation(f"{arm.name}: target pre-centering is forbidden")
            if start.reset_probe_count < 2:
                raise PairingViolation(f"{arm.name}: fewer than two exact reset probes")
            if not _is_sha256(start.binding_sha256):
                raise PairingViolation(f"{arm.name}: live runtime binding hash is invalid")
            expected_refreshes = (
                (2,)
                if task.app == "chrome" and trial.gold_prefix_length >= 2
                else ()
            )
            if start.binding_refreshed_after_steps != expected_refreshes:
                raise PairingViolation(
                    f"{arm.name}: live binding refresh evidence does not match prefix"
                )
            start_cursor = start.cursor
            start_cursor_ref = start.cursor_ref
            start_cursor_source = start.cursor_source
            start_cursor_precentered = start.cursor_precentered
            reset_probe_count = start.reset_probe_count
            start_binding_sha256 = start.binding_sha256
            binding_refreshed_after_steps = start.binding_refreshed_after_steps
            reset_signature = start.reset_signature
            current_cursor = start.cursor
            current_binding_sha256 = start.binding_sha256
            refreshed_after_steps = set(start.binding_refreshed_after_steps)
            history: list[dict[str, Any]] = []
            for turn_index in range(trial.horizon):
                budget_before = budget.snapshot()
                budget.start_model_turn()
                observation = session.observe()
                requested = session.request_action(
                    observation=observation,
                    history=tuple(history),
                    generation_seed=trial.generation_seed,
                    budget=budget.snapshot(),
                )
                budget.add_output_tokens(requested.usage)
                if budget.failure is None:
                    receipt = session.execute(requested)
                    budget.add_execution(receipt)
                else:
                    receipt = ExecutionReceipt(
                        executed_action=None,
                        cursor_before=current_cursor,
                        cursor_after=current_cursor,
                        parse_status="ok",
                        dispatch_status="budget_rejected",
                        primitive_action_count=0,
                    )
                expected_semantic_step = min(
                    trial.gold_prefix_length + budget.logical_semantic_steps + 1,
                    task.semantic_step_count,
                )
                current_binding_sha256 = self._validate_receipt(
                    task,
                    arm,
                    current_cursor,
                    current_binding_sha256,
                    refreshed_after_steps,
                    expected_semantic_step,
                    requested,
                    receipt,
                )
                current_cursor = receipt.cursor_after
                expected_target = (
                    task.expected_target(trial.gold_prefix_length)
                    if trial.mode == "gold_history_one_step"
                    else None
                )
                state_probe = session.probe_state()
                self._validate_state_probe(task, arm, state_probe)
                verified = self.verifier.verify(
                    task=task,
                    state=state_probe.state,
                    expected_step_index=(
                        trial.gold_prefix_length + 1
                        if trial.mode == "gold_history_one_step"
                        else None
                    ),
                    expected_target_ref=expected_target,
                    timeout_seconds=float(trial.budget.get("wall_time_seconds", 30.0)),
                )
                budget.add_semantic_progress(
                    verified.semantic_step_index,
                    trial.gold_prefix_length,
                )
                turn = _turn_record(
                    turn_index,
                    observation.sha256,
                    observation.media_type,
                    requested,
                    receipt,
                    verified,
                    state_probe.evidence,
                    budget_before,
                    budget.snapshot(),
                )
                turns.append(turn)
                history.append(
                    {
                        "observation_sha256": observation.sha256,
                        "requested_action": requested.value,
                        "executed_action": receipt.executed_action,
                        "verifier_state_sha256": verified.semantic_state_sha256,
                    }
                )
                if verified.status != "ok":
                    infra_class = "verifier"
                    infra_phase = "verify"
                    infra_message = "verifier returned non-ok status"
                    break
                if budget.failure is not None:
                    break
                if receipt.parse_status != "ok" or receipt.dispatch_status != "ok":
                    break
                if (
                    trial.mode == "gold_history_one_step"
                    and verified.semantic_step_index == trial.gold_prefix_length + 1
                    and verified.matched_target_ref == expected_target
                ):
                    break
                if verified.task_solved:
                    break
        except InfrastructureFailure as exc:
            infra_class = exc.failure_class
            infra_phase = "runtime"
            infra_message = str(exc)
        finally:
            if session is not None:
                try:
                    session.close()
                except InfrastructureFailure as exc:
                    if infra_class is None:
                        infra_class = exc.failure_class
                        infra_phase = "close"
                        infra_message = str(exc)

        final_verifier = turns[-1]["verifier_state"] if turns else None
        score = _score_arm(task, trial, turns, infra_class, budget.failure)
        resolved_budget_aggregate = _resolved_budget_aggregate(
            task,
            arm,
            turns,
        )
        return {
            "arm": arm.name,
            "action_interface": arm.action_interface,
            "reset_signature": reset_signature,
            "start_cursor_ref": start_cursor_ref,
            "start_cursor": None if start_cursor is None else list(start_cursor),
            "start_cursor_source": start_cursor_source,
            "start_cursor_precentered": start_cursor_precentered,
            "reset_probe_count": reset_probe_count,
            "start_binding_sha256": start_binding_sha256,
            "binding_refreshed_after_steps": (
                None
                if binding_refreshed_after_steps is None
                else list(binding_refreshed_after_steps)
            ),
            "turns": turns,
            "actions_requested": len(turns),
            "actions_executed": sum(
                turn["parse_status"] == "ok" and turn["dispatch_status"] == "ok"
                for turn in turns
            ),
            "final_verifier_state": final_verifier,
            "MOUSE_SOLVED": (
                None if final_verifier is None else bool(final_verifier["task_solved"])
            ),
            "success": score["success"],
            "score_name": score["score_name"],
            "first_divergence_from_gold": score["first_divergence_from_gold"],
            "budget_accounting": budget.snapshot(),
            "budget_failure": budget.failure,
            "resolved_budget_aggregate": resolved_budget_aggregate,
            "infra_failure_class": infra_class,
            "infra_failure_phase": infra_phase,
            "infra_failure_message": infra_message,
        }

    @staticmethod
    def _validate_receipt(
        task: Task,
        arm: Arm,
        current_cursor: tuple[int, int],
        current_binding_sha256: str,
        refreshed_after_steps: set[int],
        expected_semantic_step: int,
        requested: RequestedAction,
        receipt: ExecutionReceipt,
    ) -> str:
        if receipt.cursor_before != current_cursor:
            raise PairingViolation(
                f"{arm.name}: executor cursor_before mismatch "
                f"{receipt.cursor_before} != {current_cursor}"
            )
        for label, cursor in (
            ("cursor_before", receipt.cursor_before),
            ("cursor_after", receipt.cursor_after),
        ):
            if len(cursor) != 2 or not all(isinstance(value, int) for value in cursor):
                raise PairingViolation(f"{arm.name}: invalid {label}")
        if receipt.dispatch_status == "budget_rejected":
            if (
                receipt.executed_action is not None
                or receipt.action_classes
                or receipt.semantic_operations
                or receipt.lowered_operations
                or receipt.operations
                or receipt.backend_primitives
                or receipt.primitive_action_count != 0
                or receipt.cursor_after != receipt.cursor_before
            ):
                raise PairingViolation(f"{arm.name}: invalid budget-rejected receipt")
            return current_binding_sha256
        if (
            receipt.semantic_step_index != expected_semantic_step
            or receipt.resolved_primitive_actions != receipt.primitive_action_count
            or receipt.resolved_primitive_events != len(receipt.operations)
            or len(receipt.resolved_actions) != receipt.resolved_primitive_actions
            or not _is_sha256(receipt.binding_sha256)
            or not _is_sha256(receipt.resolved_budget_sha256)
        ):
            raise PairingViolation(f"{arm.name}: invalid live-resolved budget receipt")
        if task.app == "chrome" and expected_semantic_step == 3 and 2 not in refreshed_after_steps:
            transition = receipt.executor_evidence.get("binding_refresh")
            expected_transition = {
                "after_completed_step": 2,
                "previous_binding_sha256": current_binding_sha256,
                "refreshed_binding_sha256": receipt.binding_sha256,
                "live_probe": True,
            }
            if transition != expected_transition or receipt.binding_sha256 == current_binding_sha256:
                raise PairingViolation(
                    f"{arm.name}: Chrome step 3 lacks a refreshed post-scroll binding"
                )
            refreshed_after_steps.add(2)
        elif receipt.binding_sha256 != current_binding_sha256:
            raise PairingViolation(f"{arm.name}: unregistered live binding transition")
        resolved_payload = resolved_segment_budget_payload(
            task_id=task.task_id,
            fixture_sha256=task.fixture_sha256,
            action_schema=arm.action_interface,
            semantic_step_index=receipt.semantic_step_index,
            actions=receipt.resolved_actions,
            resolved_primitive_actions=receipt.resolved_primitive_actions,
            resolved_primitive_events=receipt.resolved_primitive_events,
            binding_sha256=receipt.binding_sha256,
        )
        if sha256_json(resolved_payload) != receipt.resolved_budget_sha256:
            raise PairingViolation(f"{arm.name}: resolved budget receipt hash mismatch")
        evidence = receipt.executor_evidence
        if evidence.get("cursor_readback_verified") is not True:
            raise PairingViolation(f"{arm.name}: post-dispatch cursor readback is missing")
        if evidence.get("interventions_between_policy_turns") != []:
            raise PairingViolation(f"{arm.name}: hidden between-turn intervention detected")
        active = evidence.get("active_window")
        if (
            not isinstance(active, dict)
            or active.get("verified") is not True
            or active.get("method")
            not in {"x11_getactivewindow", "wayland_foreground_surface"}
            or not isinstance(active.get("window_id"), str)
            or not active["window_id"]
            or active.get("expected_application") != task.app
            or active.get("observed_application") != task.app
        ):
            raise PairingViolation(f"{arm.name}: true active-window evidence missing")
        if arm.name == ARMS[0] and "click" in receipt.action_classes:
            value = requested.value
            operations = value.get("operations") if isinstance(value, dict) else None
            if not isinstance(operations, list):
                raise PairingViolation("native click request is not a native action sequence")
            requested_clicks = [
                (index, operation.get("coordinate"))
                for index, operation in enumerate(operations)
                if isinstance(operation, dict) and operation.get("action") == "click"
            ]
            if any(
                not isinstance(coordinate, list)
                or len(coordinate) != 2
                or not all(isinstance(value, int) and not isinstance(value, bool) for value in coordinate)
                for _, coordinate in requested_clicks
            ):
                raise PairingViolation("native click coordinate is not [int, int]")
            dispatches = evidence.get("native_click_dispatches")
            if not requested_clicks or not isinstance(dispatches, list):
                raise PairingViolation("native click dispatch evidence is missing")
            lowered_clicks = [
                (index, operation)
                for index, operation in enumerate(receipt.lowered_operations)
                if isinstance(operation, dict) and operation.get("action") == "click"
            ]
            expected_dispatches: list[dict[str, Any]] = []
            if len(lowered_clicks) == len(requested_clicks):
                for (requested_index, coordinate), (
                    lowered_index,
                    lowered,
                ) in zip(requested_clicks, lowered_clicks, strict=True):
                    if (
                        lowered.get("source_operation_index") != requested_index
                        or lowered.get("coordinate") != coordinate
                    ):
                        break
                    expected_dispatches.append(
                        {
                            "requested_operation_index": requested_index,
                            "lowered_operation_index": lowered_index,
                            "requested_coordinate": coordinate,
                            "dispatched_coordinate": coordinate,
                            "post_click_cursor": coordinate,
                        }
                    )
            if (
                len(expected_dispatches) != len(requested_clicks)
                or dispatches != expected_dispatches
            ):
                raise PairingViolation(
                    "native requested click coordinate did not drive lowered dispatch"
                )
            if (
                evidence.get("post_action_cursor_verified") is not True
                or evidence.get("post_action_cursor") != list(receipt.cursor_after)
            ):
                raise PairingViolation("native post-action cursor was not read back")
        return receipt.binding_sha256

    @staticmethod
    def _validate_state_probe(task: Task, arm: Arm, probe: StateProbe) -> None:
        if not isinstance(probe, StateProbe) or not isinstance(probe.state, dict):
            raise PairingViolation(f"{arm.name}: state extractor contract drift")
        evidence = probe.evidence
        if (
            not isinstance(evidence, dict)
            or evidence.get("read_only") is not True
            or evidence.get("input_events") != []
            or evidence.get("application") != task.app
            or evidence.get("method")
            not in {
                "uno_readonly_state_probe",
                "filesystem_readonly_state_probe",
                "browser_dom_readonly_state_probe",
                "editor_readonly_state_probe",
                "test_readonly_state_probe",
            }
        ):
            raise PairingViolation(
                f"{arm.name}: verifier state extraction was not proven input-free/read-only"
            )


def _turn_record(
    turn_index: int,
    observation_sha256: str,
    media_type: str,
    requested: RequestedAction,
    receipt: ExecutionReceipt,
    verifier: VerifierState,
    state_probe_evidence: dict[str, Any],
    budget_before: dict[str, Any],
    budget_after: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "turn_index": turn_index,
        "observation_sha256": observation_sha256,
        "observation_media_type": media_type,
        "requested_action": requested.value,
        "requested_action_sha256": sha256_json(requested.value),
        "model_call_id": requested.model_call_id,
        "model_usage": requested.usage,
        "budget_before": budget_before,
        "budget_after": budget_after,
        "executed_action": receipt.executed_action,
        "executed_action_sha256": sha256_json(receipt.executed_action),
        "semantic_operations": list(receipt.semantic_operations),
        "semantic_operations_sha256": sha256_json(receipt.semantic_operations),
        "lowered_operations": list(receipt.lowered_operations),
        "operations": list(receipt.operations),
        "backend_primitives": list(receipt.backend_primitives),
        "primitive_action_count": receipt.primitive_action_count,
        "resolved_actions": list(receipt.resolved_actions),
        "semantic_step_index": receipt.semantic_step_index,
        "resolved_primitive_actions": receipt.resolved_primitive_actions,
        "resolved_primitive_events": receipt.resolved_primitive_events,
        "resolved_budget_sha256": receipt.resolved_budget_sha256,
        "binding_sha256": receipt.binding_sha256,
        "executor_evidence": receipt.executor_evidence,
        "parse_status": receipt.parse_status,
        "dispatch_status": receipt.dispatch_status,
        "cursor_before": list(receipt.cursor_before),
        "cursor_after": list(receipt.cursor_after),
        "action_classes": list(receipt.action_classes),
        "state_probe_evidence": state_probe_evidence,
        "verifier_state": {
            "status": verifier.status,
            "task_solved": verifier.task_solved,
            "semantic_step_index": verifier.semantic_step_index,
            "matched_target_ref": verifier.matched_target_ref,
            "semantic_state_sha256": verifier.semantic_state_sha256,
            "semantic_state": verifier.semantic_state,
            "verifier_module": verifier.verifier_module,
            "oracle_pid": verifier.oracle_pid,
            "reason": verifier.reason,
        },
        "infra_failure_class": "verifier" if verifier.status != "ok" else None,
    }
    record["turn_payload_sha256"] = sha256_json(record)
    return record


def _resolved_budget_aggregate(
    task: Task,
    arm: Arm,
    turns: list[dict[str, Any]],
) -> dict[str, Any]:
    resolved = [
        turn for turn in turns if turn.get("dispatch_status") != "budget_rejected"
    ]
    payload = {
        "schema_version": 1,
        "schema_id": "proper_vm_runtime_budget_aggregate_v1",
        "task_id": task.task_id,
        "fixture_sha256": task.fixture_sha256,
        "action_schema": arm.action_interface,
        "segment_budget_sha256": [
            turn["resolved_budget_sha256"] for turn in resolved
        ],
        "binding_sha256": [turn["binding_sha256"] for turn in resolved],
        "resolved_primitive_actions": sum(
            int(turn["resolved_primitive_actions"]) for turn in resolved
        ),
        "resolved_primitive_events": sum(
            int(turn["resolved_primitive_events"]) for turn in resolved
        ),
    }
    payload["aggregate_sha256"] = sha256_json(payload)
    return payload


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value.lower() == value


def _score_arm(
    task: Task,
    trial: TrialSpec,
    turns: list[dict[str, Any]],
    infra_class: str | None,
    budget_failure: str | None,
) -> dict[str, Any]:
    if infra_class is not None:
        return {
            "success": None,
            "score_name": "excluded_infrastructure",
            "first_divergence_from_gold": None,
        }
    if not turns:
        return {
            "success": False,
            "score_name": "budget_failure",
            "first_divergence_from_gold": {
                "turn_index": None,
                "reason": budget_failure or "no_verified_turn",
            },
        }
    final = turns[-1]["verifier_state"]
    execution_ok = budget_failure is None and all(
        turn["parse_status"] == "ok" and turn["dispatch_status"] == "ok"
        for turn in turns
    )
    if trial.mode == "gold_history_one_step":
        target = task.expected_target(trial.gold_prefix_length)
        success = (
            execution_ok
            and final["status"] == "ok"
            and final["matched_target_ref"] == target
            and final["semantic_step_index"] == trial.gold_prefix_length + 1
        )
        divergence = None if success else {
            "turn_index": 0,
            "reason": "semantic_next_state_mismatch",
            "expected_target_ref": target,
            "observed_target_ref": final["matched_target_ref"],
        }
        return {
            "success": bool(success),
            "score_name": "semantic_next_state",
            "first_divergence_from_gold": divergence,
        }
    success = execution_ok and final["status"] == "ok" and final["task_solved"] is True
    return {
        "success": bool(success),
        "score_name": "semantic_final_state",
        "first_divergence_from_gold": None if success else {
            "turn_index": None,
            "reason": "final_semantic_goal_not_reached",
            "detectability": "right_censored_at_horizon",
        },
    }


def _first_divergence(arms: list[dict[str, Any]]) -> dict[str, Any] | None:
    left, right = arms
    left_turns, right_turns = left["turns"], right["turns"]
    for index in range(max(len(left_turns), len(right_turns))):
        if index >= len(left_turns) or index >= len(right_turns):
            return {
                "turn_index": index,
                "field": "trajectory_length",
                "native_value": len(left_turns),
                "compact_value": len(right_turns),
            }
        a, b = left_turns[index], right_turns[index]
        fields = (
            "observation_sha256",
            "semantic_operations_sha256",
            "parse_status",
            "dispatch_status",
            "cursor_after",
        )
        for field in fields:
            if a[field] != b[field]:
                return {
                    "turn_index": index,
                    "field": field,
                    "native_value": a[field],
                    "compact_value": b[field],
                    "raw_action_hashes": {
                        "native_requested": a["requested_action_sha256"],
                        "compact_requested": b["requested_action_sha256"],
                        "native_executed": a["executed_action_sha256"],
                        "compact_executed": b["executed_action_sha256"],
                    },
                }
    return None


def _first_semantic_divergence(arms: list[dict[str, Any]]) -> dict[str, Any] | None:
    left_turns, right_turns = arms[0]["turns"], arms[1]["turns"]
    for index in range(min(len(left_turns), len(right_turns))):
        a = left_turns[index]["verifier_state"]
        b = right_turns[index]["verifier_state"]
        comparable = (
            a["semantic_state_sha256"],
            a["semantic_step_index"],
            a["task_solved"],
        )
        other = (
            b["semantic_state_sha256"],
            b["semantic_step_index"],
            b["task_solved"],
        )
        if comparable != other:
            return {
                "turn_index": index,
                "native_verifier_state_sha256": a["semantic_state_sha256"],
                "compact_verifier_state_sha256": b["semantic_state_sha256"],
                "native_semantic_step_index": a["semantic_step_index"],
                "compact_semantic_step_index": b["semantic_step_index"],
            }
    if len(left_turns) != len(right_turns):
        return {
            "turn_index": min(len(left_turns), len(right_turns)),
            "reason": "semantic_trajectory_length_differs",
        }
    return None


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write complete records atomically; never leave a partial scored shard."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(
            "refusing to overwrite an existing final or partial scored shard"
        )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)
