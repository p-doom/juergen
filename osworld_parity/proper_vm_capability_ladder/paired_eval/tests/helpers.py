from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..contracts import ARMS, MODES, canonical_json


def task_row(task_id: str = "writer-dev-1", seed: int = 101) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": 1,
        "suite": "real_vm_curriculum_v1",
        "task_id": task_id,
        "family_id": "writer_replace_save",
        "app": "writer",
        "split": "development",
        "parameter_seed": seed,
        "semantic_steps": [
            {
                "step_id": "replace",
                "intent": "replace text",
                "capabilities": ["click", "coalesced_type"],
                "target_ref": "writer.body.replaced",
                "value_ref": "expected.text",
            },
            {
                "step_id": "save",
                "intent": "save document",
                "capabilities": ["hotkey", "ctrl_s"],
                "target_ref": "writer.document.saved",
            },
        ],
        "semantic_step_count": 2,
        "budgets": {
            "semantic_steps": 2,
            "primitive_actions": {
                "native_absolute_sequence_v1": 3,
                "compact_raw_phaseb_v1": 4,
            },
            "primitive_events": {
                "native_absolute_sequence_v1": 8,
                "compact_raw_phaseb_v1": 10,
            },
        },
        "instruction": "Replace the text and save the Writer document.",
        "snapshot": {
            "id": "osworld_ready",
            "reset_strategy": "restore_snapshot_then_seeded_setup",
        },
        "assets": [],
        "params": {"initial_text": "old"},
        "expected": {"text": "new", "saved": True},
        "near_miss": {"text": "new", "saved": False},
        "verifier": {
            "kind": "writer_state",
            "module": "osworld_parity.proper_vm_capability_ladder.paired_eval.tests.fake_verifier",
            "fresh_process": True,
            "entrypoint": "main",
            "result_schema": "semantic_oracle_result_v2",
            "state_extractor_entrypoint": "extract_state",
        },
        "geometry_contract": {
            "source": "live_probe",
            "probe_version": "test_geometry_probe_v1",
            "state_probe_version": "test_state_probe_v1",
            "required_targets": ["editor"],
        },
        "initial_cursor": {
            "source": "live_probe",
            "probe_version": "rung1_cursor_position_v1",
        },
        "gold_cursor_history": [
            {
                "prefix_length": 1,
                "step_id": "replace",
                "target_ref": "writer.body.replaced",
                "cursor_before_ref": "runtime.initial_cursor",
                "cursor_after_ref": "geometry.editor",
            },
            {
                "prefix_length": 2,
                "step_id": "save",
                "target_ref": "writer.document.saved",
                "cursor_before_ref": "geometry.editor",
                "cursor_after_ref": "geometry.editor",
            },
        ],
        "coverage": {"click": True, "coalesced_type": True, "ctrl_s": True},
        "exclusions": ["horizontal_scroll", "timing_sensitive_double_click"],
        "transport_requirements": ["unicode_safe_type", "cursor_readback"],
        "reset_contract": {"second_reset_removes_first_episode_state": True},
    }
    unsigned = dict(row)
    row["fixture_sha256"] = hashlib.sha256(canonical_json(unsigned)).hexdigest()
    return row


def task_manifest() -> tuple[dict[str, Any], str]:
    value = {
        "schema_version": 1,
        "suite": "real_vm_curriculum_v1",
        "split": "development",
        "sealed": False,
        "tasks": [task_row()],
    }
    seal = hashlib.sha256(canonical_json(value)).hexdigest()
    return value, seal


def evaluation_manifest(task_seal: str, readiness_sha: str, attempts: int = 8) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite": "paired_real_vm_development_v1",
        "split": "development",
        "development_only": True,
        "heldout_access": False,
        "task_manifest_payload_sha256": task_seal,
        "expected_executor_ready_sha256": readiness_sha,
        "expected_executor_ready_artifact_id": "artifact-executor-ready-test",
        "expected_executor_certification_schema": "proper_vm_executor_cert_v1",
        "evaluator_commit": "8" * 40,
        "arms": [
            {
                "name": ARMS[0],
                "action_interface": "native_absolute_sequence_v1",
                "checkpoint": "upstream/native",
                "checkpoint_sha256": "1" * 64,
                "prompt_id": "native-prompt-v1",
                "prompt_sha256": "2" * 64,
                "generation": {"do_sample": True, "temperature": 0.7},
            },
            {
                "name": ARMS[1],
                "action_interface": "compact_raw_phaseb_v1",
                "checkpoint": "phase-b/raw-relative",
                "checkpoint_sha256": "3" * 64,
                "prompt_id": "compact-prompt-v1",
                "prompt_sha256": "4" * 64,
                "generation": {"do_sample": True, "temperature": 0.7},
            },
        ],
        "runtime": {
            "runtime_id": "fake-paired-runtime-v1",
            "module": "example.runtime",
            "factory": "create_runtime",
            "source_sha256": "9" * 64,
            "contract_schema": "proper_vm_paired_runtime_v1",
        },
        "budget": {
            "max_model_turns_per_trial": 8,
            "max_model_turns_per_semantic_step": 8,
            "max_logical_semantic_steps": 4,
            "max_primitive_actions_per_trial": 16,
            "max_emitted_primitive_events_per_trial": 64,
            "max_output_tokens_per_turn": 256,
            "max_total_output_tokens": 2048,
            "wall_time_seconds": 120,
        },
        "modes": list(MODES),
        "gold_prefix_horizons": [2, 4, 8],
        "attempts_per_cell": attempts,
        "order_seed": 20260731,
        "shard_seed": 10101,
        "sampling_seed": 424242,
        "sampling_seed_policy": "paired_fixed_per_attempt_v1",
        "bootstrap_seed": 20260731,
        "bootstrap_resamples": 200,
        "exclusions": [],
    }


def ready_marker(path: Path) -> tuple[Path, str]:
    marker = {
        "schema_version": 1,
        "certification_schema": "proper_vm_executor_cert_v1",
        "status": "ready",
        "development_only": True,
        "scored_execution_completed": False,
        "validated_interfaces": [
            "native_absolute_control",
            "compact_raw_phaseb",
            "shared_atomic_gui_executor",
            "http_vm_transport",
        ],
        "checks": {
            "clean_build_at_least_109_tests": True,
            "narrow_click_preflight_10_trials": True,
            "forced_failure_artifact_probe_with_png": True,
            "full_click_100_trials_per_arm": True,
            "rung1a_16_cells": True,
            "rung1b_12_counterbalanced_cells": True,
            "sameapp_8_cells": True,
            "vm_isolation_and_provenance": True,
        },
        "executor_commit": "6" * 40,
        "vm_snapshot_id": "osworld_ready",
    }
    marker["capability_report_sha256"] = hashlib.sha256(
        canonical_json(marker)
    ).hexdigest()
    raw = (json.dumps(marker, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def labctl_context(path: Path, artifact_root: Path) -> Path:
    value = {
        "inputs": [
            {
                "role": "executor_readiness",
                "artifact_id": "artifact-executor-ready-test",
                "resolved_path": str(artifact_root.resolve()),
            }
        ]
    }
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def sealed_file(path: Path, value: dict[str, Any]) -> tuple[Path, str]:
    seal = hashlib.sha256(canonical_json(value)).hexdigest()
    payload = dict(value)
    payload["manifest_payload_sha256"] = seal
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path, seal
