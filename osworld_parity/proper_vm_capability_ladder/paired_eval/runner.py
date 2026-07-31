from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    ACTION_INTERFACES,
    ARMS,
    ExecutionReceipt,
    InfrastructureFailure,
    PairedRuntime,
    RUNTIME_CONTRACT_SCHEMA,
    RequestedAction,
    VerifierState,
    canonical_json,
    sha256_json,
)
from .manifest import Arm, EvaluationManifest, Task
from .planning import TrialSpec
from .readiness import ConsumedReadiness
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
        runtime: PairedRuntime,
        verifier: FreshProcessTaskVerifier | None = None,
    ) -> None:
        if not readiness.consumed:
            raise PairingViolation("executor readiness was not consumed")
        if readiness.marker_sha256 != manifest.expected_executor_ready_sha256:
            raise PairingViolation("consumed readiness does not match sealed manifest")
        if readiness.artifact_id != manifest.expected_executor_ready_artifact_id:
            raise PairingViolation("consumed readiness artifact does not match sealed manifest")
        if readiness.certification_schema != manifest.expected_executor_certification_schema:
            raise PairingViolation("consumed readiness schema does not match sealed manifest")
        expected_runtime_contract = {
            "schema": RUNTIME_CONTRACT_SCHEMA,
            "runtime_id": manifest.runtime.runtime_id,
            "executor_commit": readiness.executor_commit,
            "interfaces": {
                arm: ACTION_INTERFACES[arm]
                for arm in ARMS
            },
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
            start_cursor = start.cursor
            start_cursor_ref = start.cursor_ref
            reset_signature = start.reset_signature
            current_cursor = start.cursor
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
                self._validate_receipt(arm, current_cursor, receipt)
                current_cursor = receipt.cursor_after
                expected_target = (
                    task.expected_target(trial.gold_prefix_length)
                    if trial.mode == "gold_history_one_step"
                    else None
                )
                state = session.probe_state()
                verified = self.verifier.verify(
                    task=task,
                    state=state,
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
        return {
            "arm": arm.name,
            "action_interface": arm.action_interface,
            "reset_signature": reset_signature,
            "start_cursor_ref": start_cursor_ref,
            "start_cursor": None if start_cursor is None else list(start_cursor),
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
            "infra_failure_class": infra_class,
            "infra_failure_phase": infra_phase,
            "infra_failure_message": infra_message,
        }

    @staticmethod
    def _validate_receipt(
        arm: Arm,
        current_cursor: tuple[int, int],
        receipt: ExecutionReceipt,
    ) -> None:
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


def _turn_record(
    turn_index: int,
    observation_sha256: str,
    media_type: str,
    requested: RequestedAction,
    receipt: ExecutionReceipt,
    verifier: VerifierState,
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
        "executor_evidence": receipt.executor_evidence,
        "parse_status": receipt.parse_status,
        "dispatch_status": receipt.dispatch_status,
        "cursor_before": list(receipt.cursor_before),
        "cursor_after": list(receipt.cursor_after),
        "action_classes": list(receipt.action_classes),
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
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
