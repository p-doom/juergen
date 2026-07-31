from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    APPROVED_CURRICULUM_COMMIT,
    APPROVED_CURRICULUM_RUNTIME_BINDING_SCHEMA,
    ARMS,
    GENERATION_SEED_DERIVATION,
    INFRA_FAILURE_CLASSES,
    INFRA_FAILURE_CLASS_PHASES,
    INFRA_FAILURE_EVENT_RECEIPT_TYPE,
    INFRA_FAILURE_OPERATIONS,
    SAMPLING_SEED_POLICY,
    sha256_json,
    validate_infrastructure_failure_source_receipt,
)
from .manifest import EvaluationManifest
from .planning import TrialSpec
from .receipts import (
    executed_aggregate,
    is_sha256,
    ordered_trace_aggregate,
    validate_binding_receipt,
    validate_binding_successor,
    validate_executed_segment,
    validate_prefix_replay,
)


class IncompleteEvaluationError(ValueError):
    """A validated result cannot contribute to any statistical estimate."""

    def __init__(self, pair_id: str, failure_classes: Iterable[str]) -> None:
        classes = sorted(set(failure_classes))
        self.status = {
            "status": "invalid_incomplete_evaluation",
            "reason": "infrastructure_exclusion_lacks_external_attestation",
            "pair_id": pair_id,
            "infra_failure_classes": classes,
            "estimates_emitted": False,
            "pass_at_k_emitted": False,
            "automatic_replacement_or_retry": False,
            "required_action": "operator_review_and_new_preregistered_run",
        }
        super().__init__(
            f"{pair_id}: incomplete evaluation: infrastructure exclusion "
            "cannot be authenticated from a self-contained result row; no "
            "estimates were produced; operator review and a new preregistered "
            "run are required"
        )


def load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("row is not an object")
                    rows.append(value)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"cannot load result {path}: {exc}") from exc
    return rows


def aggregate_results(
    manifest: EvaluationManifest,
    plan: Iterable[TrialSpec],
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    trials = {trial.pair_id: trial for trial in plan}
    _validate_planned_generation_seeds(trials.values())
    rows: dict[str, dict[str, Any]] = {}
    for row in records:
        pair_id = row.get("pair_id")
        if pair_id in rows:
            raise ValueError(f"duplicate pair result: {pair_id}")
        if pair_id not in trials:
            raise ValueError(f"result is not in the deterministic plan: {pair_id}")
        _validate_row(manifest, trials[pair_id], row)
        rows[pair_id] = row
    missing = sorted(set(trials) - set(rows))
    if missing:
        raise ValueError(f"missing paired results: {missing[:5]}")

    ordered = [rows[pair_id] for pair_id in sorted(rows)]
    included = [row for row in ordered if not row["exclusion"]["excluded"]]
    excluded = [row for row in ordered if row["exclusion"]["excluded"]]
    arm_values = {
        arm: [_arm(row, arm)["success"] for row in included]
        for arm in ARMS
    }
    for arm, values in arm_values.items():
        if not all(isinstance(value, bool) for value in values):
            raise ValueError(f"included rows have non-boolean scores for {arm}")
    native = [int(value) for value in arm_values[ARMS[0]]]
    compact = [int(value) for value in arm_values[ARMS[1]]]
    differences = [right - left for left, right in zip(native, compact, strict=True)]

    strata: dict[str, Any] = {}
    stratum_keys = sorted({(row["mode"], row["horizon"]) for row in included})
    for mode, horizon in stratum_keys:
        subset = [
            row for row in included if row["mode"] == mode and row["horizon"] == horizon
        ]
        strata[f"{mode}/h{horizon}"] = _simple_metrics(subset)

    bootstrap = _paired_task_cluster_bootstrap(
        included,
        seed=manifest.bootstrap_seed,
        resamples=manifest.bootstrap_resamples,
    )
    mcnemar = _mcnemar_descriptive(native, compact)
    failure_counts: dict[str, int] = defaultdict(int)
    for row in excluded:
        for failure_class in row["exclusion"]["infra_failure_classes"]:
            failure_counts[failure_class] += 1
    return {
        "schema_version": 1,
        "report_type": "paired_complete_system_development",
        "suite": manifest.suite,
        "evaluator_commit": manifest.evaluator_commit,
        "split": "development",
        "development_only": True,
        "comparison_scope": "complete_system",
        "comparison_label": manifest.comparison_label,
        "systems": {
            arm.name: {
                "checkpoint": arm.checkpoint,
                "checkpoint_sha256": arm.checkpoint_sha256,
                "prompt_id": arm.prompt_id,
                "prompt_sha256": arm.prompt_sha256,
                "action_interface": arm.action_interface,
            }
            for arm in manifest.arms
        },
        "generation_seed_contract": {
            "sampling_seed_policy": SAMPLING_SEED_POLICY,
            "generation_seed_derivation": GENERATION_SEED_DERIVATION,
            "unique_per_planned_attempt_and_arm": True,
            "nested_arm_generation_seed_forbidden": True,
        },
        "pair_accounting": {
            "planned": len(trials),
            "complete": len(ordered),
            "included": len(included),
            "excluded_whole_pair": len(excluded),
            "infra_failure_class_counts": dict(sorted(failure_counts.items())),
            "exclusion_policy": "arm_blind_whole_pair_infrastructure_only",
        },
        "overall": {
            "n_pairs": len(included),
            "arm_success_rate": {
                ARMS[0]: _mean(native),
                ARMS[1]: _mean(compact),
            },
            "paired_difference_compact_minus_native": _mean(differences),
            "paired_task_cluster_bootstrap": bootstrap,
            "mcnemar_descriptive": mcnemar,
        },
        "by_mode_horizon": strata,
        "pass_at_k_feasibility": _pass_at_k(manifest, included, trials.values()),
    }


def _validate_row(
    manifest: EvaluationManifest,
    trial: TrialSpec,
    row: dict[str, Any],
) -> None:
    unsigned_row = dict(row)
    row_seal = unsigned_row.pop("record_payload_sha256", None)
    if row_seal != sha256_json(unsigned_row):
        raise ValueError(f"{trial.pair_id}: record payload hash mismatch")
    if row.get("schema_version") != 1 or row.get("record_type") != "paired_complete_system_trial":
        raise ValueError(f"{trial.pair_id}: result schema drift")
    if row.get("split") != "development" or row.get("development_only") is not True:
        raise ValueError(f"{trial.pair_id}: non-development result forbidden")
    if row.get("fixture_sha256") != trial.fixture_sha256:
        raise ValueError(f"{trial.pair_id}: fixture hash mismatch")
    if row.get("cell_id") != trial.cell_id or row.get("attempt_id") != trial.attempt_id:
        raise ValueError(f"{trial.pair_id}: attempt identity mismatch")
    if row.get("task_id") != trial.task_id:
        raise ValueError(f"{trial.pair_id}: task mismatch")
    if row.get("evaluator_commit") != manifest.evaluator_commit:
        raise ValueError(f"{trial.pair_id}: evaluator commit mismatch")
    readiness = row.get("readiness")
    if not isinstance(readiness, dict):
        raise ValueError(f"{trial.pair_id}: readiness provenance missing")
    if readiness.get("marker_sha256") != manifest.expected_executor_ready_sha256:
        raise ValueError(f"{trial.pair_id}: readiness marker hash mismatch")
    if readiness.get("artifact_id") != manifest.expected_executor_ready_artifact_id:
        raise ValueError(f"{trial.pair_id}: readiness artifact mismatch")
    if readiness.get("certification_schema") != manifest.expected_executor_certification_schema:
        raise ValueError(f"{trial.pair_id}: readiness schema mismatch")
    setup = row.get("task_setup_validation")
    expected_setup = {
        "artifact_sha256": manifest.expected_task_setup_validation_sha256,
        "artifact_id": manifest.expected_task_setup_validation_artifact_id,
        "schema_id": manifest.expected_task_setup_validation_schema,
        "task_manifest_payload_sha256": manifest.task_manifest_payload_sha256,
        "vm_snapshot_id": trial.snapshot_id,
        "setup_commit": setup.get("setup_commit") if isinstance(setup, dict) else None,
    }
    setup_commit = expected_setup["setup_commit"]
    if (
        setup != expected_setup
        or not isinstance(setup_commit, str)
        or len(setup_commit) != 40
        or setup_commit.lower() != setup_commit
        or any(character not in "0123456789abcdef" for character in setup_commit)
    ):
        raise ValueError(f"{trial.pair_id}: task setup-validation provenance mismatch")
    pairing = row.get("pairing")
    expected_pairing = {
        "snapshot_id": trial.snapshot_id,
        "parameter_seed": trial.parameter_seed,
        "initial_cursor_ref": trial.initial_cursor_ref,
        "initial_cursor": row.get("pairing", {}).get("initial_cursor"),
        "sampling_draw_seed": trial.sampling_draw_seed,
        "generation_seeds_by_arm": trial.generation_seeds_by_arm,
        "generation_seed_derivation": GENERATION_SEED_DERIVATION,
        "sampling_seed_policy": SAMPLING_SEED_POLICY,
        "budget": trial.budget,
        "arm_order": list(trial.arm_order),
        "shard_index": trial.shard_index,
        "shard_count": trial.shard_count,
    }
    if pairing != expected_pairing:
        raise ValueError(f"{trial.pair_id}: pair invariants changed")
    arms = row.get("arms")
    if not isinstance(arms, list) or len(arms) != 2:
        raise ValueError(f"{trial.pair_id}: result must contain two arm rows")
    if {value.get("arm") for value in arms if isinstance(value, dict)} != set(ARMS):
        raise ValueError(f"{trial.pair_id}: missing or duplicate arm")
    reset_signatures = [value.get("reset_signature") for value in arms]
    if all(value is not None for value in reset_signatures) and len(set(reset_signatures)) != 1:
        raise ValueError(f"{trial.pair_id}: paired reset signatures differ")
    start_refs = [value.get("start_cursor_ref") for value in arms]
    start_cursors = [value.get("start_cursor") for value in arms]
    start_sources = [value.get("start_cursor_source") for value in arms]
    start_precentered = [value.get("start_cursor_precentered") for value in arms]
    for value, source, precentered in zip(
        arms, start_sources, start_precentered, strict=True
    ):
        failure_event = value.get("infra_failure_evidence")
        prestart_infra = (
            isinstance(failure_event, dict)
            and failure_event.get("phase") == "open_session"
            and value.get("reset_signature") is None
        )
        if prestart_infra:
            if source is not None or precentered is not None:
                raise ValueError(f"{trial.pair_id}: partial pre-start cursor evidence")
            continue
        if source != "live_probe_before_policy":
            raise ValueError(f"{trial.pair_id}: cursor was not live-probed before policy")
        if precentered is not False:
            raise ValueError(f"{trial.pair_id}: target pre-centering is forbidden")
        if not is_sha256(value.get("start_binding_sha256")):
            raise ValueError(f"{trial.pair_id}: live binding evidence missing")
    start_geometries = [
        value.get("start_binding_receipt", {}).get("initial_geometry")
        for value in arms
        if value.get("start_binding_receipt") is not None
    ]
    if len(start_geometries) == 2 and start_geometries[0] != start_geometries[1]:
        raise ValueError(f"{trial.pair_id}: paired live geometry differs")
    reset_evidence = [
        {
            cycle["evidence_sha256"]
            for cycle in value["start_binding_receipt"]["reset_cycles"]
        }
        for value in arms
        if value.get("start_binding_receipt") is not None
    ]
    if len(reset_evidence) == 2 and reset_evidence[0] & reset_evidence[1]:
        raise ValueError(f"{trial.pair_id}: paired arms reused reset evidence")
    if all(value is not None for value in start_refs + start_cursors):
        if start_refs != [trial.initial_cursor_ref, trial.initial_cursor_ref]:
            raise ValueError(f"{trial.pair_id}: paired cursor refs differ")
        if start_cursors[0] != start_cursors[1]:
            raise ValueError(f"{trial.pair_id}: paired live cursor values differ")
        if row["pairing"].get("initial_cursor") != start_cursors[0]:
            raise ValueError(f"{trial.pair_id}: paired live cursor evidence mismatch")
    exclusion = row.get("exclusion")
    if not isinstance(exclusion, dict):
        raise ValueError(f"{trial.pair_id}: exclusion decision missing")
    if exclusion.get("policy") != "arm_blind_whole_pair_infrastructure_only":
        raise ValueError(f"{trial.pair_id}: exclusion policy drift")
    if exclusion.get("decision_inputs_contain_arm_identity") is not False:
        raise ValueError(f"{trial.pair_id}: arm-aware exclusion forbidden")
    classes = exclusion.get("infra_failure_classes")
    if not isinstance(classes, list) or classes != sorted(set(classes)):
        raise ValueError(f"{trial.pair_id}: invalid infrastructure exclusion classes")
    if any(value not in INFRA_FAILURE_CLASSES for value in classes):
        raise ValueError(f"{trial.pair_id}: unknown infrastructure exclusion class")
    if exclusion.get("excluded") is not bool(classes):
        raise ValueError(f"{trial.pair_id}: exclusion/class mismatch")
    validated_classes: set[str] = set()
    for arm in manifest.arms:
        system = row.get("systems", {}).get(arm.name)
        expected = {
            "action_interface": arm.action_interface,
            "checkpoint": arm.checkpoint,
            "checkpoint_sha256": arm.checkpoint_sha256,
            "prompt_id": arm.prompt_id,
            "prompt_sha256": arm.prompt_sha256,
            "generation": arm.generation,
        }
        if system != expected:
            raise ValueError(f"{trial.pair_id}: complete-system provenance drift for {arm.name}")
        if _arm(row, arm.name).get("generation_seed") != trial.generation_seed_for(
            arm.name
        ):
            raise ValueError(f"{trial.pair_id}: arm stochastic seed evidence mismatch")
        validated_class = _validate_and_recompute_arm(
            manifest.task(trial.task_id),
            trial,
            _arm(row, arm.name),
            setup_commit=setup_commit,
        )
        if validated_class is not None:
            validated_classes.add(validated_class)
    if classes != sorted(validated_classes):
        raise ValueError(
            f"{trial.pair_id}: exclusion was not derived from validated evidence"
        )
    expected_runtime = {
        "schema": "proper_vm_paired_runtime_v1",
        "runtime_id": manifest.runtime.runtime_id,
        "executor_commit": row["readiness"].get("executor_commit"),
        "interfaces": {arm.name: arm.action_interface for arm in manifest.arms},
        "cursor_initialization": "live_unmodified_snapshot",
        "native_coordinate_dispatch": "requested_to_lowered_to_post_cursor",
        "between_turn_interventions": "forbidden",
        "active_window_check": "true_active_window_only",
        "curriculum_commit": APPROVED_CURRICULUM_COMMIT,
        "live_binding": APPROVED_CURRICULUM_RUNTIME_BINDING_SCHEMA,
        "resolved_budget_receipts": "executed_segment_receipt_v1",
        "ordered_execution_trace_aggregate": "paired_policy_turn_receipt_trace_v1",
        "complete_program_aggregate": "c603_compiled_program_receipt_v1",
    }
    if row.get("runtime") != expected_runtime:
        raise ValueError(f"{trial.pair_id}: runtime contract drift")
    if validated_classes:
        # The nested receipts prove internal consistency and bind the exact
        # runner phase, but their SHA-256 seals are carried inside the same
        # mutable row.  Without an independently pinned signature or
        # attestation boundary, accepting them would let a re-sealed row alter
        # the paired denominator and pass@k.  Refuse the entire aggregate.
        raise IncompleteEvaluationError(trial.pair_id, validated_classes)


def _validate_infrastructure_failure_evidence(
    task: Any,
    trial: TrialSpec,
    arm: dict[str, Any],
    turns: list[dict[str, Any]],
) -> str | None:
    outer_failure_class = arm.get("infra_failure_class")
    outer_phase = arm.get("infra_failure_phase")
    outer_message = arm.get("infra_failure_message")
    event = arm.get("infra_failure_evidence")
    if event is None:
        if (
            outer_failure_class is not None
            or outer_phase is not None
            or outer_message is not None
        ):
            raise ValueError(
                f"{trial.pair_id}: partial infrastructure failure declaration"
            )
        return None
    expected_fields = {
        "schema_version",
        "receipt_type",
        "pair_id",
        "task_id",
        "fixture_sha256",
        "arm",
        "generation_seed",
        "failure_class",
        "phase",
        "message_sha256",
        "start_binding_sha256",
        "prior_turn_payload_sha256",
        "source_receipt",
        "event_receipt_sha256",
    }
    if not isinstance(event, dict) or set(event) != expected_fields:
        raise ValueError(
            f"{trial.pair_id}: infrastructure failure event schema mismatch"
        )
    unsigned_event = dict(event)
    event_seal = unsigned_event.pop("event_receipt_sha256")
    if event_seal != sha256_json(unsigned_event):
        raise ValueError(f"{trial.pair_id}: infrastructure failure event hash mismatch")
    failure_class = event.get("failure_class")
    phase = event.get("phase")
    if failure_class not in INFRA_FAILURE_CLASSES:
        raise ValueError(f"{trial.pair_id}: unknown infrastructure failure class")
    if phase not in INFRA_FAILURE_CLASS_PHASES[failure_class]:
        raise ValueError(f"{trial.pair_id}: infrastructure class/phase mismatch")
    if not isinstance(outer_message, str) or not outer_message:
        raise ValueError(f"{trial.pair_id}: infrastructure failure message missing")
    if outer_failure_class != failure_class or outer_phase != phase:
        raise ValueError(
            f"{trial.pair_id}: outer infrastructure summary disagrees with "
            "structured phase evidence"
        )
    if (
        event.get("schema_version") != 1
        or event.get("receipt_type") != INFRA_FAILURE_EVENT_RECEIPT_TYPE
        or event.get("pair_id") != trial.pair_id
        or event.get("task_id") != task.task_id
        or event.get("fixture_sha256") != task.fixture_sha256
        or event.get("arm") != arm.get("arm")
        or event.get("generation_seed")
        != trial.generation_seed_for(arm.get("arm"))
        or event.get("message_sha256") != sha256_json(outer_message)
        or event.get("start_binding_sha256") != arm.get("start_binding_sha256")
        or event.get("prior_turn_payload_sha256")
        != [turn["turn_payload_sha256"] for turn in turns]
    ):
        raise ValueError(
            f"{trial.pair_id}: infrastructure failure event binding mismatch"
        )
    try:
        source_receipt = validate_infrastructure_failure_source_receipt(
            event.get("source_receipt"),
            expected_class=failure_class,
            expected_operation=INFRA_FAILURE_OPERATIONS[phase],
        )
    except ValueError as exc:
        raise ValueError(
            f"{trial.pair_id}: invalid infrastructure source evidence: {exc}"
        ) from exc
    raw_evidence = source_receipt["raw_evidence"]
    raw_event = raw_evidence.get("event")
    if not isinstance(raw_event, str) or not raw_event:
        raise ValueError(f"{trial.pair_id}: infrastructure source event missing")
    if raw_event == "verifier_non_ok_result":
        if failure_class != "verifier" or phase != "verifier" or not turns:
            raise ValueError(f"{trial.pair_id}: invalid non-ok verifier evidence")
        final_turn = turns[-1]
        verifier = final_turn["verifier_state"]
        if (
            verifier.get("status") == "ok"
            or final_turn.get("infra_failure_class") != "verifier"
            or raw_evidence
            != {
                "event": "verifier_non_ok_result",
                "task_id": task.task_id,
                "fixture_sha256": task.fixture_sha256,
                "turn_index": len(turns) - 1,
                "turn_payload_sha256": final_turn["turn_payload_sha256"],
                "verifier_status": verifier.get("status"),
                "oracle_pid": verifier.get("oracle_pid"),
                "verifier_module": verifier.get("verifier_module"),
            }
        ):
            raise ValueError(f"{trial.pair_id}: non-ok verifier evidence mismatch")
    elif failure_class == "verifier":
        if (
            raw_event != "fresh_process_verifier_failure"
            or raw_evidence.get("task_id") != task.task_id
            or raw_evidence.get("fixture_sha256") != task.fixture_sha256
            or not isinstance(raw_evidence.get("reason_code"), str)
            or not raw_evidence["reason_code"]
        ):
            raise ValueError(f"{trial.pair_id}: verifier failure evidence mismatch")
    return failure_class


def _validate_and_recompute_arm(
    task: Any,
    trial: TrialSpec,
    arm: dict[str, Any],
    *,
    setup_commit: str,
) -> str | None:
    start_receipt = arm.get("start_binding_receipt")
    if start_receipt is not None:
        binding = validate_binding_receipt(
            start_receipt,
            task_id=task.task_id,
            fixture_sha256=task.fixture_sha256,
            snapshot_id=task.snapshot_id,
            setup_commit=setup_commit,
            require_fresh=False,
        )
        if arm.get("start_binding_sha256") != binding["binding_sha256"]:
            raise ValueError(f"{trial.pair_id}: start binding hash mismatch")
        terminal, cursor, _ = validate_prefix_replay(
            replay=arm.get("prefix_replay"),
            prefix_length=trial.gold_prefix_length,
            start_binding=binding,
            task_id=task.task_id,
            fixture_sha256=task.fixture_sha256,
            snapshot_id=task.snapshot_id,
            setup_commit=setup_commit,
            app=task.app,
            action_schema=arm.get("action_interface"),
            require_fresh=False,
        )
        if terminal != binding or arm.get("start_cursor") != cursor:
            raise ValueError(f"{trial.pair_id}: prefix terminal evidence mismatch")
        verifier = arm.get("start_prefix_verifier")
        state_probe = arm.get("start_state_probe_evidence")
        expected_target = (
            task.semantic_steps[trial.gold_prefix_length - 1].target_ref
            if trial.gold_prefix_length
            else None
        )
        if (
            not isinstance(verifier, dict)
            or verifier.get("status") != "ok"
            or verifier.get("semantic_step_index") != trial.gold_prefix_length
            or (
                trial.gold_prefix_length
                and verifier.get("matched_target_ref") != expected_target
            )
            or verifier.get("verifier_module") != task.verifier_module
            or not isinstance(verifier.get("oracle_pid"), int)
            or verifier["oracle_pid"] <= 0
            or not isinstance(verifier.get("semantic_state"), dict)
            or sha256_json(verifier["semantic_state"])
            != verifier.get("semantic_state_sha256")
        ):
            raise ValueError(f"{trial.pair_id}: prefix fresh-verifier evidence mismatch")
        _validate_state_probe_evidence(
            task,
            trial,
            {"state_probe_evidence": state_probe},
        )
    turns = arm.get("turns")
    if not isinstance(turns, list):
        raise ValueError(f"{trial.pair_id}: arm trace is not a list")
    if turns:
        _validate_turn_trace_evidence(
            task,
            trial,
            arm,
            turns,
            setup_commit=setup_commit,
        )
    infra = _validate_infrastructure_failure_evidence(task, trial, arm, turns)
    if not turns and infra is not None:
        expected_success: bool | None = None
        expected_score = "excluded_infrastructure"
        expected_divergence = None
        if arm.get("final_verifier_state") is not None or arm.get("MOUSE_SOLVED") is not None:
            raise ValueError(f"{trial.pair_id}: empty trace has final verifier evidence")
    else:
        if not turns:
            if arm.get("budget_failure") is None:
                raise ValueError(f"{trial.pair_id}: unexcluded arm has no trace")
            expected_success = False
            expected_score = "budget_failure"
            expected_divergence = {
                "turn_index": None,
                "reason": arm["budget_failure"],
            }
            if arm.get("final_verifier_state") is not None or arm.get("MOUSE_SOLVED") is not None:
                raise ValueError(f"{trial.pair_id}: empty trace has final verifier evidence")
        else:
            final = turns[-1]["verifier_state"]
            budget_failure = _recompute_budget_failure(trial, arm, turns)
            execution_ok = budget_failure is None and all(
                turn.get("parse_status") == "ok" and turn.get("dispatch_status") == "ok"
                for turn in turns
            )
            if trial.mode == "gold_history_one_step":
                target = task.expected_target(trial.gold_prefix_length)
                expected_success = bool(
                    execution_ok
                    and final.get("status") == "ok"
                    and final.get("matched_target_ref") == target
                    and final.get("semantic_step_index")
                    == trial.gold_prefix_length + 1
                )
                expected_score = "semantic_next_state"
                expected_divergence = None if expected_success else {
                    "turn_index": 0,
                    "reason": "semantic_next_state_mismatch",
                    "expected_target_ref": target,
                    "observed_target_ref": final.get("matched_target_ref"),
                }
            else:
                expected_success = bool(
                    execution_ok
                    and final.get("status") == "ok"
                    and final.get("task_solved") is True
                )
                expected_score = "semantic_final_state"
                expected_divergence = None if expected_success else {
                    "turn_index": None,
                    "reason": "final_semantic_goal_not_reached",
                    "detectability": "right_censored_at_horizon",
                }
            if arm.get("final_verifier_state") != final:
                raise ValueError(f"{trial.pair_id}: final verifier evidence mismatch")
            if arm.get("MOUSE_SOLVED") is not bool(final.get("task_solved")):
                raise ValueError(f"{trial.pair_id}: MOUSE_SOLVED mismatch")
            if (
                expected_success
                and infra is not None
                and arm["infra_failure_evidence"]["phase"] != "close"
            ):
                raise ValueError(
                    f"{trial.pair_id}: successful trace has non-close infrastructure exclusion"
                )
    if arm.get("score_name") != expected_score:
        raise ValueError(f"{trial.pair_id}: score name mismatch")
    if arm.get("first_divergence_from_gold") != expected_divergence:
        raise ValueError(f"{trial.pair_id}: stored divergence disagrees with trace")
    if arm.get("success") is not expected_success:
        raise ValueError(f"{trial.pair_id}: stored success disagrees with trace")
    return infra


def _validate_turn_trace_evidence(
    task: Any,
    trial: TrialSpec,
    arm: dict[str, Any],
    turns: list[dict[str, Any]],
    *,
    setup_commit: str,
) -> None:
    current_binding = validate_binding_receipt(
        arm.get("start_binding_receipt"),
        task_id=task.task_id,
        fixture_sha256=task.fixture_sha256,
        snapshot_id=task.snapshot_id,
        setup_commit=setup_commit,
        require_fresh=False,
    )
    _, current_cursor, completed_step_2_receipt = validate_prefix_replay(
        replay=arm.get("prefix_replay"),
        prefix_length=trial.gold_prefix_length,
        start_binding=current_binding,
        task_id=task.task_id,
        fixture_sha256=task.fixture_sha256,
        snapshot_id=task.snapshot_id,
        setup_commit=setup_commit,
        app=task.app,
        action_schema=arm.get("action_interface"),
        require_fresh=False,
    )
    logical_progress = 0
    for turn in turns:
        unsigned_turn = dict(turn)
        turn_seal = unsigned_turn.pop("turn_payload_sha256", None)
        if turn_seal != sha256_json(unsigned_turn):
            raise ValueError(f"{trial.pair_id}: turn payload hash mismatch")
        if turn.get("model_generation_seed") != trial.generation_seed_for(arm["arm"]):
            raise ValueError(f"{trial.pair_id}: model stochastic seed evidence mismatch")
        verifier = turn.get("verifier_state")
        if not isinstance(verifier, dict):
            raise ValueError(f"{trial.pair_id}: verifier evidence missing")
        state = verifier.get("semantic_state")
        if not isinstance(state, dict) or sha256_json(state) != verifier.get(
            "semantic_state_sha256"
        ):
            raise ValueError(f"{trial.pair_id}: verifier state hash mismatch")
        if verifier.get("verifier_module") != task.verifier_module:
            raise ValueError(f"{trial.pair_id}: verifier module mismatch")
        if not isinstance(verifier.get("oracle_pid"), int) or verifier["oracle_pid"] <= 0:
            raise ValueError(f"{trial.pair_id}: fresh verifier PID missing")
        _validate_executor_evidence(task, trial, arm, turn)
        _validate_state_probe_evidence(task, trial, turn)
        expected_step = min(
            trial.gold_prefix_length + logical_progress + 1,
            task.semantic_step_count,
        )
        if turn.get("dispatch_status") == "budget_rejected":
            if any(
                turn.get(key) not in (None, 0, "", [])
                for key in (
                    "binding_receipt",
                    "compiled_segment",
                    "dispatches",
                    "executed_segment_receipt",
                    "binding_sha256",
                    "binding_revision",
                    "resolved_budget_sha256",
                    "resolved_primitive_actions",
                    "resolved_primitive_events",
                    "resolved_actions",
                )
            ):
                raise ValueError(f"{trial.pair_id}: invalid budget-rejected receipt")
        elif turn.get("parse_status") != "ok" or turn.get("dispatch_status") != "ok":
            if any(
                turn.get(key) not in (None, 0, "", [])
                for key in (
                    "compiled_segment",
                    "dispatches",
                    "executed_segment_receipt",
                    "resolved_budget_sha256",
                    "resolved_primitive_actions",
                    "resolved_primitive_events",
                    "resolved_actions",
                )
            ):
                raise ValueError(f"{trial.pair_id}: failed execution forged a receipt")
            failed_binding_raw = turn.get("binding_receipt")
            if failed_binding_raw is not None:
                failed_binding = validate_binding_receipt(
                    failed_binding_raw,
                    task_id=task.task_id,
                    fixture_sha256=task.fixture_sha256,
                    snapshot_id=task.snapshot_id,
                    setup_commit=setup_commit,
                    require_fresh=False,
                )
                validate_binding_successor(
                    current_binding,
                    failed_binding,
                    completed_step_2_receipt_sha256=completed_step_2_receipt,
                )
                current_binding = failed_binding
        else:
            binding = validate_binding_receipt(
                turn.get("binding_receipt"),
                task_id=task.task_id,
                fixture_sha256=task.fixture_sha256,
                snapshot_id=task.snapshot_id,
                setup_commit=setup_commit,
                require_fresh=False,
            )
            validate_binding_successor(
                current_binding,
                binding,
                completed_step_2_receipt_sha256=completed_step_2_receipt,
            )
            if (task.app == "chrome" and (
                (expected_step <= 2 and binding["binding_revision"] != 1)
                or (expected_step >= 3 and binding["binding_revision"] != 2)
            )) or (task.app != "chrome" and binding["binding_revision"] != 1):
                raise ValueError(f"{trial.pair_id}: binding revision/semantic-step mismatch")
            executed = validate_executed_segment(
                compiled_segment=turn.get("compiled_segment"),
                dispatches=turn.get("dispatches"),
                executed_receipt=turn.get("executed_segment_receipt"),
                binding_receipt=binding,
                task_id=task.task_id,
                fixture_sha256=task.fixture_sha256,
                action_schema=arm.get("action_interface"),
                expected_semantic_step=expected_step,
                expected_cursor_before=current_cursor,
            )
            segment = turn["compiled_segment"]
            if (
                turn.get("semantic_step_index") != segment["semantic_step_index"]
                or turn.get("resolved_primitive_actions")
                != segment["resolved_primitive_actions"]
                or turn.get("resolved_primitive_events")
                != segment["resolved_primitive_events"]
                or len(turn.get("operations", []))
                != segment["resolved_primitive_events"]
                or turn.get("primitive_action_count")
                != segment["resolved_primitive_actions"]
                or turn.get("resolved_budget_sha256")
                != segment["resolved_budget_sha256"]
                or turn.get("binding_sha256") != segment["binding_sha256"]
                or turn.get("binding_revision") != segment["binding_revision"]
                or turn.get("resolved_actions") != segment["actions"]
                or turn.get("cursor_before") != segment["expected_cursor_before"]
                or turn.get("cursor_after") != segment["expected_cursor_after"]
                or executed != turn.get("executed_segment_receipt")
            ):
                raise ValueError(f"{trial.pair_id}: turn/executed-segment mismatch")
            current_binding = binding
            current_cursor = list(segment["expected_cursor_after"])
        verifier_index = verifier.get("semantic_step_index")
        if isinstance(verifier_index, int):
            logical_progress = max(
                logical_progress,
                verifier_index - trial.gold_prefix_length,
            )
            if (
                turn.get("semantic_step_index") == 2
                and verifier_index >= 2
                and isinstance(turn.get("executed_segment_receipt"), dict)
            ):
                completed_step_2_receipt = turn["executed_segment_receipt"][
                    "executed_receipt_sha256"
                ]
    _validate_execution_aggregates(task, trial, arm, turns)


def _validate_executor_evidence(
    task: Any,
    trial: TrialSpec,
    arm: dict[str, Any],
    turn: dict[str, Any],
) -> None:
    if turn.get("dispatch_status") == "budget_rejected":
        return
    evidence = turn.get("executor_evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"{trial.pair_id}: executor evidence missing")
    if evidence.get("cursor_readback_verified") is not True:
        raise ValueError(f"{trial.pair_id}: post-dispatch cursor readback missing")
    if evidence.get("interventions_between_policy_turns") != []:
        raise ValueError(f"{trial.pair_id}: hidden between-turn intervention detected")
    active = evidence.get("active_window")
    if (
        not isinstance(active, dict)
        or active.get("verified") is not True
        or active.get("method") not in {"x11_getactivewindow", "wayland_foreground_surface"}
        or not isinstance(active.get("window_id"), str)
        or not active["window_id"]
        or active.get("expected_application") != task.app
        or active.get("observed_application") != task.app
    ):
        raise ValueError(f"{trial.pair_id}: true active-window evidence missing")
    if arm.get("arm") != ARMS[0] or "click" not in turn.get("action_classes", []):
        return
    requested = turn.get("requested_action")
    operations = requested.get("operations") if isinstance(requested, dict) else None
    if not isinstance(operations, list):
        raise ValueError(f"{trial.pair_id}: native click request schema mismatch")
    click_coordinates = [
        operation.get("coordinate")
        for operation in operations
        if isinstance(operation, dict) and operation.get("action") == "click"
    ]
    if any(
        not isinstance(coordinate, list)
        or len(coordinate) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in coordinate)
        for coordinate in click_coordinates
    ):
        raise ValueError(f"{trial.pair_id}: invalid native click coordinate")
    requested_clicks = [
        (index, operation.get("coordinate"))
        for index, operation in enumerate(operations)
        if isinstance(operation, dict) and operation.get("action") == "click"
    ]
    lowered_clicks = [
        (index, operation)
        for index, operation in enumerate(turn.get("lowered_operations", []))
        if isinstance(operation, dict) and operation.get("action") == "click"
    ]
    compiled = turn.get("compiled_segment")
    compiled_clicks = [
        operation.get("coordinate")
        for action in compiled.get("actions", [])
        if isinstance(compiled, dict) and isinstance(action, dict)
        for operation in action.get("operations", [])
        if isinstance(operation, dict) and operation.get("action") == "click"
    ] if isinstance(compiled, dict) else []
    expected: list[dict[str, Any]] = []
    if len(lowered_clicks) == len(requested_clicks):
        for (requested_index, coordinate), (lowered_index, lowered) in zip(
            requested_clicks, lowered_clicks, strict=True
        ):
            if (
                lowered.get("source_operation_index") != requested_index
                or lowered.get("coordinate") != coordinate
            ):
                break
            expected.append(
                {
                    "requested_operation_index": requested_index,
                    "lowered_operation_index": lowered_index,
                    "requested_coordinate": coordinate,
                    "dispatched_coordinate": coordinate,
                    "post_click_cursor": coordinate,
                }
            )
    if (
        len(expected) != len(requested_clicks)
        or not expected
        or evidence.get("native_click_dispatches") != expected
        or compiled_clicks != [coordinate for _, coordinate in requested_clicks]
    ):
        raise ValueError(f"{trial.pair_id}: native click dispatch evidence mismatch")
    if (
        evidence.get("post_action_cursor_verified") is not True
        or evidence.get("post_action_cursor") != turn.get("cursor_after")
    ):
        raise ValueError(f"{trial.pair_id}: native post-action cursor evidence mismatch")


def _validate_state_probe_evidence(
    task: Any,
    trial: TrialSpec,
    turn: dict[str, Any],
) -> None:
    evidence = turn.get("state_probe_evidence")
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
        }
    ):
        raise ValueError(f"{trial.pair_id}: state probe was not input-free/read-only")


def _validate_execution_aggregates(
    task: Any,
    trial: TrialSpec,
    arm: dict[str, Any],
    turns: list[dict[str, Any]],
) -> None:
    resolved = [
        turn["executed_segment_receipt"]
        for turn in turns
        if turn.get("parse_status") == "ok"
        and turn.get("dispatch_status") == "ok"
        and isinstance(turn.get("executed_segment_receipt"), dict)
    ]
    expected_trace = ordered_trace_aggregate(
        task_id=task.task_id,
        fixture_sha256=task.fixture_sha256,
        action_schema=arm["action_interface"],
        receipts=resolved,
    )
    if arm.get("ordered_execution_trace_aggregate") != expected_trace:
        raise ValueError(f"{trial.pair_id}: ordered execution trace aggregate mismatch")
    complete_coverage = [item["semantic_step_index"] for item in resolved] == list(
        range(1, task.semantic_step_count + 1)
    )
    if complete_coverage:
        expected_complete = executed_aggregate(
            task_id=task.task_id,
            fixture_sha256=task.fixture_sha256,
            action_schema=arm["action_interface"],
            app=task.app,
            semantic_step_count=task.semantic_step_count,
            primitive_action_cap=task.budget_contract["primitive_action_caps"][
                arm["action_interface"]
            ],
            primitive_event_cap=task.budget_contract["primitive_event_caps"][
                arm["action_interface"]
            ],
            receipts=resolved,
        )
    else:
        expected_complete = None
    if arm.get("complete_program_aggregate") != expected_complete:
        raise ValueError(f"{trial.pair_id}: complete program aggregate mismatch")
    expected_status = (
        "validated_c603_complete_program"
        if expected_complete is not None
        else "not_complete_semantic_coverage"
    )
    if arm.get("complete_program_aggregate_status") != expected_status:
        raise ValueError(f"{trial.pair_id}: complete program aggregate status mismatch")


def _recompute_budget_failure(
    trial: TrialSpec,
    arm: dict[str, Any],
    turns: list[dict[str, Any]],
) -> str | None:
    limits = trial.budget
    model_turns = len(turns)
    actions = sum(int(turn.get("primitive_action_count", -1)) for turn in turns)
    events = sum(len(turn.get("operations", [])) for turn in turns)
    token_values = [turn.get("model_usage", {}).get("output_tokens") for turn in turns]
    if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in token_values):
        raise ValueError(f"{trial.pair_id}: invalid output-token evidence")
    tokens = sum(token_values)
    final_step = int(turns[-1]["verifier_state"]["semantic_step_index"])
    logical_steps = max(0, final_step - trial.gold_prefix_length)
    failures: list[str] = []
    if model_turns > int(limits["model_turns"]):
        failures.append("model_turns_exceeded")
    if logical_steps > int(limits["logical_semantic_steps"]):
        failures.append("logical_semantic_steps_exceeded")
    if actions > int(limits["primitive_actions"]):
        failures.append("primitive_actions_exceeded")
    if events > int(limits["emitted_primitive_events"]):
        failures.append("emitted_primitive_events_exceeded")
    if any(value > int(limits["output_tokens_per_turn"]) for value in token_values):
        failures.append("output_tokens_per_turn_exceeded")
    if tokens > int(limits["total_output_tokens"]):
        failures.append("total_output_tokens_exceeded")
    accounting = arm.get("budget_accounting")
    if not isinstance(accounting, dict) or accounting.get("limits") != limits:
        raise ValueError(f"{trial.pair_id}: budget accounting limits mismatch")
    used = accounting.get("used", {})
    expected_used = {
        "model_turns": model_turns,
        "logical_semantic_steps": logical_steps,
        "primitive_actions": actions,
        "emitted_primitive_events": events,
        "output_tokens": tokens,
    }
    if any(used.get(key) != value for key, value in expected_used.items()):
        raise ValueError(f"{trial.pair_id}: budget accounting usage mismatch")
    elapsed = used.get("wall_time_seconds")
    if not isinstance(elapsed, (int, float)) or elapsed < 0:
        raise ValueError(f"{trial.pair_id}: invalid wall-time evidence")
    if elapsed > float(limits["wall_time_seconds"]):
        failures.append("wall_time_seconds_exceeded")
    stored = arm.get("budget_failure")
    if failures:
        if stored not in failures:
            raise ValueError(f"{trial.pair_id}: budget failure evidence mismatch")
    elif stored is not None:
        raise ValueError(f"{trial.pair_id}: spurious budget failure")
    return stored


def _arm(row: dict[str, Any], arm: str) -> dict[str, Any]:
    matches = [value for value in row["arms"] if value["arm"] == arm]
    if len(matches) != 1:
        raise ValueError(f"arm is not unique in result: {arm}")
    return matches[0]


def _simple_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    native = [int(_arm(row, ARMS[0])["success"]) for row in rows]
    compact = [int(_arm(row, ARMS[1])["success"]) for row in rows]
    return {
        "n_pairs": len(rows),
        "arm_success_rate": {ARMS[0]: _mean(native), ARMS[1]: _mean(compact)},
        "paired_difference_compact_minus_native": _mean(
            [right - left for left, right in zip(native, compact, strict=True)]
        ),
    }


def _paired_task_cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    if not rows:
        return {
            "resamples": resamples,
            "seed": seed,
            "cluster": "task_id",
            "confidence_interval_95": [None, None],
        }
    clusters: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        clusters[row["task_id"]].append(
            int(_arm(row, ARMS[1])["success"]) - int(_arm(row, ARMS[0])["success"])
        )
    names = sorted(clusters)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        sampled = [rng.choice(names) for _ in names]
        observations = [value for name in sampled for value in clusters[name]]
        values.append(sum(observations) / len(observations))
    values.sort()
    return {
        "resamples": resamples,
        "seed": seed,
        "cluster": "task_id",
        "confidence_interval_95": [_percentile(values, 0.025), _percentile(values, 0.975)],
    }


def _mcnemar_descriptive(native: list[int], compact: list[int]) -> dict[str, Any]:
    native_only = sum(left == 1 and right == 0 for left, right in zip(native, compact, strict=True))
    compact_only = sum(left == 0 and right == 1 for left, right in zip(native, compact, strict=True))
    discordant = native_only + compact_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(native_only, compact_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "label": "descriptive_only_no_independence_claim",
        "native_success_compact_failure": native_only,
        "native_failure_compact_success": compact_only,
        "discordant_pairs": discordant,
        "exact_two_sided_p_value": p_value,
    }


def _pass_at_k(
    manifest: EvaluationManifest,
    rows: list[dict[str, Any]],
    planned_trials: Iterable[TrialSpec],
) -> dict[str, Any]:
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[row["cell_id"]].append(row)
    planned_cells = {trial.cell_id for trial in planned_trials}
    output: dict[str, Any] = {}
    for k in (1, 4, 8):
        eligible = [
            cells[cell_id]
            for cell_id in sorted(planned_cells)
            if len(cells[cell_id]) >= k
        ]
        enough = bool(planned_cells) and len(eligible) == len(planned_cells)
        stochastic = {
            arm.name: _stochastic_generation(arm.generation) if k > 1 else True
            for arm in manifest.arms
        }
        feasible = enough and all(stochastic.values())
        estimates: dict[str, float | None] = {name: None for name in ARMS}
        if feasible:
            for arm in ARMS:
                estimates[arm] = _mean(
                    [
                        _unbiased_pass_at_k(
                            len(values),
                            sum(bool(_arm(row, arm)["success"]) for row in values),
                            k,
                        )
                        for values in eligible
                    ]
                )
        output[f"pass@{k}"] = {
            "feasible": feasible,
            "enough_complete_attempts_per_cell": enough,
            "cells_total": len(planned_cells),
            "cells_with_at_least_k_attempts": len(eligible),
            "stochastic_generation_configured": stochastic,
            "paired_task_reset_per_attempt": True,
            "unique_fixed_generation_seed_per_attempt": True,
            "estimate_by_arm": estimates,
        }
    return output


def _validate_planned_generation_seeds(trials: Iterable[TrialSpec]) -> None:
    draws_by_cell: dict[str, set[int]] = defaultdict(set)
    all_draw_seeds: set[int] = set()
    all_arm_seeds: set[int] = set()
    for trial in trials:
        draws = draws_by_cell[trial.cell_id]
        if trial.sampling_draw_seed in draws:
            raise ValueError("duplicate stochastic sampling draw within pass@k cell")
        draws.add(trial.sampling_draw_seed)
        if trial.sampling_draw_seed in all_draw_seeds:
            raise ValueError("stochastic sampling draw seed reused across cells")
        all_draw_seeds.add(trial.sampling_draw_seed)
        arm_seeds = [trial.generation_seed_for(arm) for arm in ARMS]
        if len(set(arm_seeds)) != len(ARMS):
            raise ValueError("paired arms share a stochastic generation seed")
        for seed in arm_seeds:
            if seed in all_arm_seeds:
                raise ValueError("stochastic generation seed reused across draws/arms")
            all_arm_seeds.add(seed)


def _stochastic_generation(generation: dict[str, Any]) -> bool:
    if generation.get("do_sample") is True:
        return float(generation.get("temperature", 1.0)) > 0
    return float(generation.get("temperature", 0.0)) > 0


def _unbiased_pass_at_k(n: int, successes: int, k: int) -> float:
    if n < k:
        raise ValueError("pass@k needs at least k attempts")
    failures = n - successes
    if failures < k:
        return 1.0
    return 1.0 - math.comb(failures, k) / math.comb(n, k)


def _mean(values: list[int] | list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _percentile(values: list[float], probability: float) -> float:
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction
