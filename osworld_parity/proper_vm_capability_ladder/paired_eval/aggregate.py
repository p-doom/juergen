from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .contracts import ARMS, resolved_segment_budget_payload, sha256_json
from .manifest import EvaluationManifest
from .planning import TrialSpec


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
        "generation_seed": trial.generation_seed,
        "sampling_seed_policy": "paired_fixed_per_attempt_v1",
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
        prestart_infra = (
            value.get("infra_failure_class") is not None
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
        if not _is_sha256(value.get("start_binding_sha256")):
            raise ValueError(f"{trial.pair_id}: live binding evidence missing")
    start_bindings = [value.get("start_binding_sha256") for value in arms]
    present_bindings = [value for value in start_bindings if value is not None]
    if present_bindings and (not all(
        _is_sha256(value) for value in present_bindings
    ) or len(set(present_bindings)) != 1):
        raise ValueError(f"{trial.pair_id}: paired initial live bindings differ")
    if any(
        value.get("reset_probe_count") is not None
        and (
            not isinstance(value.get("reset_probe_count"), int)
            or value["reset_probe_count"] < 2
        )
        for value in arms
    ):
        raise ValueError(f"{trial.pair_id}: repeated reset probe evidence missing")
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
    if exclusion.get("excluded") is not bool(classes):
        raise ValueError(f"{trial.pair_id}: exclusion/class mismatch")
    actual_classes = sorted(
        {
            value.get("infra_failure_class")
            for value in arms
            if value.get("infra_failure_class") is not None
        }
    )
    if classes != actual_classes:
        raise ValueError(f"{trial.pair_id}: exclusion was not derived from both arms")
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
        _validate_and_recompute_arm(manifest.task(trial.task_id), trial, _arm(row, arm.name))
    expected_runtime = {
        "schema": "proper_vm_paired_runtime_v1",
        "runtime_id": manifest.runtime.runtime_id,
        "executor_commit": row["readiness"].get("executor_commit"),
        "interfaces": {arm.name: arm.action_interface for arm in manifest.arms},
        "cursor_initialization": "live_unmodified_snapshot",
        "native_coordinate_dispatch": "requested_to_lowered_to_post_cursor",
        "between_turn_interventions": "forbidden",
        "active_window_check": "true_active_window_only",
        "live_binding": "provisional_contract_test_only_reject_all",
        "resolved_budget_receipts": "provisional_contract_test_only_reject_all",
    }
    if row.get("runtime") != expected_runtime:
        raise ValueError(f"{trial.pair_id}: runtime contract drift")


def _validate_and_recompute_arm(task: Any, trial: TrialSpec, arm: dict[str, Any]) -> bool | None:
    turns = arm.get("turns")
    if not isinstance(turns, list):
        raise ValueError(f"{trial.pair_id}: arm trace is not a list")
    if turns:
        _validate_turn_trace_evidence(task, trial, arm, turns)
    infra = arm.get("infra_failure_class")
    if infra is not None:
        expected_success: bool | None = None
    else:
        if not turns:
            if arm.get("budget_failure") is None:
                raise ValueError(f"{trial.pair_id}: unexcluded arm has no trace")
            expected_success = False
            if arm.get("score_name") != "budget_failure":
                raise ValueError(f"{trial.pair_id}: empty budget trace score mismatch")
            if arm.get("final_verifier_state") is not None or arm.get("MOUSE_SOLVED") is not None:
                raise ValueError(f"{trial.pair_id}: empty trace has final verifier evidence")
            if arm.get("success") is not expected_success:
                raise ValueError(f"{trial.pair_id}: stored success disagrees with trace")
            return expected_success
        final = turns[-1]["verifier_state"]
        budget_failure = _recompute_budget_failure(trial, arm, turns)
        execution_ok = budget_failure is None and all(
            turn.get("parse_status") == "ok" and turn.get("dispatch_status") == "ok"
            for turn in turns
        )
        if trial.mode == "gold_history_one_step":
            expected_success = bool(
                execution_ok
                and final.get("status") == "ok"
                and final.get("matched_target_ref")
                == task.expected_target(trial.gold_prefix_length)
                and final.get("semantic_step_index") == trial.gold_prefix_length + 1
            )
            expected_score = "semantic_next_state"
        else:
            expected_success = bool(
                execution_ok
                and final.get("status") == "ok"
                and final.get("task_solved") is True
            )
            expected_score = "semantic_final_state"
        if arm.get("score_name") != expected_score:
            raise ValueError(f"{trial.pair_id}: score name mismatch")
        if arm.get("final_verifier_state") != final:
            raise ValueError(f"{trial.pair_id}: final verifier evidence mismatch")
        if arm.get("MOUSE_SOLVED") is not bool(final.get("task_solved")):
            raise ValueError(f"{trial.pair_id}: MOUSE_SOLVED mismatch")
    if arm.get("success") is not expected_success:
        raise ValueError(f"{trial.pair_id}: stored success disagrees with trace")
    return expected_success


def _validate_turn_trace_evidence(
    task: Any,
    trial: TrialSpec,
    arm: dict[str, Any],
    turns: list[dict[str, Any]],
) -> None:
    current_binding = arm.get("start_binding_sha256")
    refreshed = arm.get("binding_refreshed_after_steps")
    expected_initial_refreshes = (
        [2] if task.app == "chrome" and trial.gold_prefix_length >= 2 else []
    )
    if refreshed != expected_initial_refreshes:
        raise ValueError(f"{trial.pair_id}: live binding refresh/prefix mismatch")
    refreshed_steps = set(refreshed)
    logical_progress = 0
    for turn in turns:
        unsigned_turn = dict(turn)
        turn_seal = unsigned_turn.pop("turn_payload_sha256", None)
        if turn_seal != sha256_json(unsigned_turn):
            raise ValueError(f"{trial.pair_id}: turn payload hash mismatch")
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
        current_binding = _validate_resolved_budget_receipt(
            task,
            trial,
            arm,
            turn,
            current_binding,
            refreshed_steps,
            expected_step,
        )
        verifier_index = verifier.get("semantic_step_index")
        if isinstance(verifier_index, int):
            logical_progress = max(
                logical_progress,
                verifier_index - trial.gold_prefix_length,
            )
    _validate_resolved_budget_aggregate(task, trial, arm, turns)


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
            "test_readonly_state_probe",
        }
    ):
        raise ValueError(f"{trial.pair_id}: state probe was not input-free/read-only")


def _validate_resolved_budget_receipt(
    task: Any,
    trial: TrialSpec,
    arm: dict[str, Any],
    turn: dict[str, Any],
    current_binding: str,
    refreshed_steps: set[int],
    expected_step: int,
) -> str:
    if turn.get("dispatch_status") == "budget_rejected":
        if any(
            turn.get(key) not in (0, "", [])
            for key in (
                "semantic_step_index",
                "resolved_primitive_actions",
                "resolved_primitive_events",
                "resolved_budget_sha256",
                "binding_sha256",
                "resolved_actions",
            )
        ):
            raise ValueError(f"{trial.pair_id}: invalid budget-rejected receipt")
        return current_binding
    actions = turn.get("resolved_actions")
    resolved_actions = turn.get("resolved_primitive_actions")
    resolved_events = turn.get("resolved_primitive_events")
    binding = turn.get("binding_sha256")
    if (
        turn.get("semantic_step_index") != expected_step
        or not isinstance(actions, list)
        or not isinstance(resolved_actions, int)
        or len(actions) != resolved_actions
        or resolved_actions != turn.get("primitive_action_count")
        or not isinstance(resolved_events, int)
        or resolved_events != len(turn.get("operations", []))
        or not _is_sha256(binding)
    ):
        raise ValueError(f"{trial.pair_id}: invalid live-resolved budget receipt")
    if task.app == "chrome" and expected_step == 3 and 2 not in refreshed_steps:
        expected_transition = {
            "after_completed_step": 2,
            "previous_binding_sha256": current_binding,
            "refreshed_binding_sha256": binding,
            "live_probe": True,
        }
        if (
            turn.get("executor_evidence", {}).get("binding_refresh")
            != expected_transition
            or binding == current_binding
        ):
            raise ValueError(f"{trial.pair_id}: Chrome live binding refresh missing")
        refreshed_steps.add(2)
    elif binding != current_binding:
        raise ValueError(f"{trial.pair_id}: unregistered live binding transition")
    payload = resolved_segment_budget_payload(
        task_id=task.task_id,
        fixture_sha256=task.fixture_sha256,
        action_schema=arm["action_interface"],
        semantic_step_index=expected_step,
        actions=tuple(actions),
        resolved_primitive_actions=resolved_actions,
        resolved_primitive_events=resolved_events,
        binding_sha256=binding,
    )
    if turn.get("resolved_budget_sha256") != sha256_json(payload):
        raise ValueError(f"{trial.pair_id}: resolved budget receipt hash mismatch")
    return binding


def _validate_resolved_budget_aggregate(
    task: Any,
    trial: TrialSpec,
    arm: dict[str, Any],
    turns: list[dict[str, Any]],
) -> None:
    resolved = [
        turn for turn in turns if turn.get("dispatch_status") != "budget_rejected"
    ]
    payload = {
        "schema_version": 1,
        "schema_id": "proper_vm_runtime_budget_aggregate_v1",
        "task_id": task.task_id,
        "fixture_sha256": task.fixture_sha256,
        "action_schema": arm["action_interface"],
        "segment_budget_sha256": [
            turn["resolved_budget_sha256"] for turn in resolved
        ],
        "binding_sha256": [turn["binding_sha256"] for turn in resolved],
        "resolved_primitive_actions": sum(
            turn["resolved_primitive_actions"] for turn in resolved
        ),
        "resolved_primitive_events": sum(
            turn["resolved_primitive_events"] for turn in resolved
        ),
    }
    expected = dict(payload)
    expected["aggregate_sha256"] = sha256_json(payload)
    if arm.get("resolved_budget_aggregate") != expected:
        raise ValueError(f"{trial.pair_id}: resolved budget aggregate mismatch")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value.lower() == value
        and all(character in "0123456789abcdef" for character in value)
    )


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
