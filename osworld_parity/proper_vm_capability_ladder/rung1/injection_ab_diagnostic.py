from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .fixtures import Fixture, load_manifest
from .selfcheck import (
    _assert_dispatch_journal,
    _atomic_json,
    _execute,
    _validate_loaded_geometry,
)
from .server import (
    BROWSER_AUDIT_EVENTS,
    BROWSER_AUDIT_SCHEMA_VERSION,
    FixtureHttpServer,
)
from .transport import HttpVmTransport, Operation
from .trajectory import build_trajectory
from .transport_diagnostic import (
    FIXTURE_ID,
    FIXTURE_SHA256,
    MANIFEST_PAYLOAD_SHA256,
)
from .vm import (
    DEFAULT_PROVIDER,
    DEFAULT_QCOW,
    DEFAULT_QEMU,
    READY_SNAPSHOT,
    KvmFixtureSession,
    sha256_file,
)


SPEC_PATH = Path(__file__).with_name("injection_ab_spec.json")
EXPECTED_SPEC_SHA256 = "68c612969e88ae7f2122c3321be98286fff7b02818c80e07fb4d49f33ad444e4"
SUITE = "rung1_click_release_injection_ab_v3"
ORDER_BLOCK = ("A", "B", "B", "A", "B", "A", "A", "B")
TRIAL_ORDER = ORDER_BLOCK * 6
BACKEND_BY_ARM = {
    "A": "pyautogui_release_motion",
    "B": "direct_xtest_no_release_motion",
}
RELEASE_MOTION_BY_ARM = {"A": True, "B": False}
AUDIT_DOM_EVENTS = (
    "pointerdown",
    "pointerup",
    "pointermove",
    "mousedown",
    "mouseup",
    "mousemove",
    "click",
    "input",
    "change",
    "focus",
    "blur",
)
AUDIT_REQUIRED_FIELDS = (
    "audit_sequence",
    "browser_wall_time_ms",
    "client_monotonic_ms",
    "event_time_stamp_ms",
    "is_trusted",
    "default_prevented",
    "target",
    "target_checked",
    "checkbox_state",
    "active_element",
    "document_has_focus",
    "visibility_state",
)
AUDIT_DETAIL_FIELDS = (
    "button",
    "buttons",
    "pointer_type",
    "client_x",
    "client_y",
    "screen_x",
    "screen_y",
)
AUDIT_ENVELOPE_FIELDS = (
    "schema_version",
    "generation",
    "audit_sequence",
    "event",
    "host_audit_request_id",
    "host_monotonic_ns",
    "host_wall_time_ns",
)
TIMESTAMP_STAGES = (
    "click_started",
    "press_call_before",
    "press_call_after",
    "press_sync_completed",
    "dwell_started",
    "dwell_completed",
    "release_call_before",
    "release_call_after",
    "release_sync_completed",
    "click_completed",
)
OBSERVATION_WINDOW_S = 3.0
POST_WINDOW_HEARTBEAT_WAIT_S = 3.0
X_BUTTON_PRESS = 4
X_BUTTON_RELEASE = 5
X_MOTION_NOTIFY = 6


class InjectionAbIntegrityError(RuntimeError):
    def __init__(self, message: str, *, evidence: Any = None) -> None:
        super().__init__(message)
        self.evidence = evidence


class BackendBoundTransport:
    """Select an experimental click backend without changing action parsing."""

    def __init__(self, transport: HttpVmTransport, backend: str) -> None:
        self._transport = transport
        self.backend = backend
        self.audit = transport.audit

    def __getattr__(self, name: str) -> Any:
        return getattr(self._transport, name)

    def execute_atomic(
        self, operations: tuple[Operation, ...]
    ) -> Any:
        return self._transport.execute_atomic(
            operations, click_backend=self.backend
        )


def load_injection_ab_spec(path: Path = SPEC_PATH) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if path == SPEC_PATH and digest != EXPECTED_SPEC_SHA256:
        raise InjectionAbIntegrityError(
            f"injection A/B preregistration hash drifted: {digest}"
        )
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InjectionAbIntegrityError("injection A/B spec is invalid JSON") from exc
    if not isinstance(spec, dict):
        raise InjectionAbIntegrityError("injection A/B spec must be an object")
    exact = {
        "schema_version": 1,
        "suite": SUITE,
        "status": "preregistered_cpu_development_diagnostic",
        "manifest_payload_sha256": MANIFEST_PAYLOAD_SHA256,
        "fixture_id": FIXTURE_ID,
        "fixture_sha256": FIXTURE_SHA256,
        "development_only": True,
        "canonical_action_format": "compact_raw_phaseb",
        "canonical_semantic_operations": [
            "move_relative",
            "mouse_down",
            "mouse_up",
        ],
        "canonical_lowered_operations": ["move_relative", "click"],
        "dwell_ms": 50,
        "observation_window_s": OBSERVATION_WINDOW_S,
        "post_window_heartbeat_wait_s": POST_WINDOW_HEARTBEAT_WAIT_S,
        "order_block": list(ORDER_BLOCK),
        "order_block_repetitions": 6,
        "trial_count": len(TRIAL_ORDER),
        "trials_per_arm": 24,
        "reset_snapshot": READY_SNAPSHOT,
        "reset_before_every_trial": True,
        "fresh_chromium_every_trial": True,
        "dispatches_per_trial": 1,
        "guest_processes_per_dispatch": 1,
        "retry_count": 0,
        "replacement_count": 0,
        "stop_policy": "integrity_abort_only",
        "guest_timestamp_clock": "time.monotonic_ns",
        "required_guest_timestamp_stages": list(TIMESTAMP_STAGES),
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
        "qualification_authorized": False,
    }
    differing = [key for key, value in exact.items() if spec.get(key) != value]
    if differing:
        raise InjectionAbIntegrityError(
            f"injection A/B preregistration drifted: {sorted(differing)}"
        )
    if spec.get("supersedes") != {
        "suite": "rung1_click_release_injection_ab_v2",
        "pipeline_id": "pipeline_019fb9ebfdcf73e1af0eaf02b6a12c73",
        "run_id": "run_019fb9ebfdcf73e1af0eaef40d8fbdff",
        "job_id": "136178",
        "terminal_state": "trial_6_integrity_abort",
        "reason": (
            "v2 used best-effort heartbeat delivery followed by a fixed 750 ms "
            "sleep and one-shot snapshot, and did not durably checkpoint the raw "
            "active-trial browser snapshot before validation"
        ),
    }:
        raise InjectionAbIntegrityError("successor identity provenance drifted")
    if spec.get("control_plane_guardrail") != {
        "authoritative_host": "hai-login2.haicore.berlin",
        "required_invocation": "ssh hai-login2.haicore.berlin labctl ...",
        "reason": (
            "the PostgreSQL Unix socket is host-local despite its shared NFS path"
        ),
        "direct_compute_node_failure": (
            "ECONNREFUSED followed by a misleading 30-second pool timeout"
        ),
        "submission_authorized": False,
        "manual_reconcile_authorized": False,
    }:
        raise InjectionAbIntegrityError("control-plane guardrail drifted")
    if spec.get("backend_arms") != {
        "A": {
            "name": BACKEND_BY_ARM["A"],
            "order": [
                "pyautogui_top_level_same_coordinate_motion",
                "pyautogui_press",
                "x_sync",
                "dwell_50ms",
                "pyautogui_release_side_same_coordinate_motion",
                "x_sync",
                "pyautogui_release",
                "x_sync",
            ],
            "release_side_motion_notify": True,
        },
        "B": {
            "name": BACKEND_BY_ARM["B"],
            "order": [
                "pyautogui_top_level_same_coordinate_motion",
                "xtest_press",
                "x_sync",
                "dwell_50ms",
                "arm_neutral_release_pre_button_x_sync",
                "xtest_release",
                "x_sync",
            ],
            "release_side_motion_notify": False,
        },
    }:
        raise InjectionAbIntegrityError("backend arm contract drifted")
    if spec.get("common_pyautogui_click_premove") != {
        "source": (
            "PyAutoGUI 0.9.54 top-level click _mouseMoveDrag before platform _click"
        ),
        "phase": "click_premove",
        "xtest_sequence": ["motion_notify"],
        "event_type": X_MOTION_NOTIFY,
        "detail": 0,
        "same_coordinates_as_press": True,
        "arm_neutral": True,
    }:
        raise InjectionAbIntegrityError("common click premove contract drifted")
    if spec.get("post_window_heartbeat_wait_basis") != {
        "wait_bound_s": POST_WINDOW_HEARTBEAT_WAIT_S,
        "nominal_heartbeat_interval_ms": 500,
        "observed_successful_trial_count": 5,
        "max_observed_post_deadline_marker_arrival_lag_ns": 460_922_709,
        "max_observed_heartbeat_host_interarrival_ns": 999_610_841,
        "rationale": (
            "3.0 seconds is more than three times the maximum observed host "
            "interarrival and more than six times the maximum observed "
            "post-deadline marker lag in v2 job 136178"
        ),
    }:
        raise InjectionAbIntegrityError("post-window heartbeat wait basis drifted")
    if spec.get("failure_evidence_contract") != {
        "schema_version": "rung1_atomic_output_failure_v2",
        "lifecycle_global_attempt_hooks": True,
        "x_test_attempt_fields": [
            "sequence",
            "phase",
            "event",
            "event_type",
            "detail",
            "x",
            "y",
            "attempted",
            "success",
            "error",
            "started_guest_monotonic_ns",
            "completed_guest_monotonic_ns",
            "duration_ns",
        ],
        "successful_x_test_phase_event_order_by_arm": {
            "A": [
                ["canonical_move", "motion_notify"],
                ["click_premove", "motion_notify"],
                ["press", "motion_notify"],
                ["press", "button_press"],
                ["release", "motion_notify"],
                ["release", "button_release"],
            ],
            "B": [
                ["canonical_move", "motion_notify"],
                ["click_premove", "motion_notify"],
                ["press", "motion_notify"],
                ["press", "button_press"],
                ["release", "button_release"],
            ],
        },
        "global_sync_attempt_fields": [
            "sequence",
            "phase",
            "attempted",
            "success",
            "error",
            "started_guest_monotonic_ns",
            "completed_guest_monotonic_ns",
            "duration_ns",
        ],
        "successful_sync_attempt_phase_order_both_arms": [
            "initial_readback",
            "canonical_move",
            "click_premove",
            "press",
            "press",
            "press_sync",
            "release",
            "release",
            "release_sync",
            "verification_readback",
            "final_readback",
        ],
        "causal_sync_attempt_fields": [
            "supported",
            "flush_attempted",
            "flush",
            "sync_attempted",
            "sync",
            "success",
            "error",
        ],
        "final_pointer_readback_fields": [
            "attempted",
            "success",
            "error",
            "cursor",
            "pointer_button_mask",
        ],
        "post_output_error_fields": [
            "click_backend_expected",
            "expected",
            "observed",
            "execute_result",
            "raw_stdout",
            "raw_result_markers",
            "raw_payload",
            "raw_backend_primitives",
            "raw_x_event_sync_evidence",
            "raw_x_sync_attempt_evidence",
            "raw_x_injection_timestamps",
            "raw_x_injection_evidence",
            "guest_error",
            "guest_failure_kind",
            "pointer_masks",
            "final_pointer_readback",
            "attempt_hook_restore_errors",
        ],
        "checkpoint_immediately_after_dispatch": True,
    }:
        raise InjectionAbIntegrityError("failure evidence contract drifted")
    audit = spec.get("browser_audit")
    if not isinstance(audit, dict) or (
        audit.get("transport")
        != (
            "independent_navigator_sendBeacon_for_dom_events_and_"
            "fetch_ack_chain_for_heartbeats"
        )
        or audit.get("dom_events") != list(AUDIT_DOM_EVENTS)
        or audit.get("required_fields") != list(AUDIT_REQUIRED_FIELDS)
        or audit.get("wire_schema_version") != BROWSER_AUDIT_SCHEMA_VERSION
        or audit.get("heartbeat_event") != "audit_heartbeat"
        or audit.get("heartbeat_interval_ms") != 500
        or audit.get("require_heartbeat_causally_generated_after_observation_deadline")
        is not True
        or audit.get("heartbeat_acknowledgement_fields")
        != [
            "acknowledged_heartbeat_audit_sequence",
            "acknowledged_host_audit_request_id",
            "acknowledged_host_monotonic_ns",
        ]
        or audit.get("heartbeat_expected_sequence_fields")
        != [
            "expected_previous_audit_sequence",
            "expected_audit_count_through_marker",
        ]
        or audit.get("classification_uses_only_sequence_sealed_prefix") is not True
        or audit.get("arm_neutral") is not True
        or audit.get("serialized_post_queue_independent") is not True
        or audit.get("cdp_independent") is not True
        or audit.get("bounded_condition_wait") is not True
        or audit.get("raw_snapshot_checkpoint_before_validation") is not True
    ):
        raise InjectionAbIntegrityError("browser audit contract drifted")
    if spec.get("browser_failure_evidence_contract") != {
        "checkpoint_stage": "browser_audit_captured",
        "checkpoint_before_validation": True,
        "active_trial_fields": [
            "raw_browser_snapshot",
            "post_window_heartbeat_wait",
        ],
        "wait_evidence_fields": [
            "schema_version",
            "generation",
            "observation_deadline_host_monotonic_ns",
            "wait_started_host_monotonic_ns",
            "wait_completed_host_monotonic_ns",
            "wait_duration_ns",
            "wait_timeout_s",
            "wait_deadline_host_monotonic_ns",
            "timed_out",
            "candidate_audit_sequences",
            "heartbeat_summaries",
        ],
        "timeout_remains_integrity_abort": True,
    }:
        raise InjectionAbIntegrityError("browser failure evidence contract drifted")
    passive = spec.get("passive_x_observer")
    if passive != {
        "enabled": False,
        "installed": False,
        "observer_process_count": 0,
        "additional_x_connection_count": 0,
        "assessment": "omitted_not_demonstrably_non_perturbing",
        "limitation": (
            "not installed: a same-process XRecord/XI2 observer requires a second X "
            "connection and concurrent event consumption, which is not demonstrably "
            "non-perturbing for this timing experiment"
        ),
    }:
        raise InjectionAbIntegrityError("passive X observer limitation drifted")
    parent = spec.get("parent_evidence")
    if not isinstance(parent, dict) or parent.get("job_id") != "136131":
        raise InjectionAbIntegrityError("registered parent evidence drifted")
    if spec.get("pre_science_integrity_abort_history") != {
        "pipeline_id": "pipeline_019fb9aa926f7ba198bc237a383e9a49",
        "run_id": "run_019fb9aa926f7ba198bc2362ead2f37a",
        "job_id": "136152",
        "completed_trial_count": 0,
        "retry_count": 0,
        "replacement_count": 0,
        "integrity_error": "atomic guest action controlled X identity drifted",
        "root_cause": (
            "validator omitted the arm-neutral PyAutoGUI top-level same-coordinate "
            "pre-click MotionNotify"
        ),
        "artifact_index_sha256": (
            "92988882cdd0cecd3df94ce404af16f37f8439351515ac51ed8620c4d023ff03"
        ),
        "content_address": (
            "sha256:1a7a7eaf80f51803b5f557311d61c9961007719407f8662b189dea226621e3b4"
        ),
        "failure_sha256": (
            "a1b40b559be6a575e01ecdbc911f1a86b6d2cb26d2406c64399df092a9a55b4e"
        ),
        "progress_sha256": (
            "15bfeefb8fbfff19b8d017ab955b280bc194aee6f6a10955e13cc025ca8ec550"
        ),
        "vm_closed": True,
        "overlay_removed": True,
        "accidental_control_plane_action": {
            "command": "labctl reconcile",
            "result": "duplicate artifact primary key artifact_7143accb7e53783d",
            "effect": "no run revival, relabel, replacement, or additional Slurm job",
        },
    }:
        raise InjectionAbIntegrityError("pre-science failure provenance drifted")
    if spec.get("alias_mismatch_abort_history") != {
        "pipeline_id": "pipeline_019fb9e86a027952a8d3fd92795c7d1b",
        "build_run_id": "run_019fb9e86a027952a8d3fd73687c5054",
        "diagnostic_run_id": "run_019fb9e86a027952a8d3fd845f839cee",
        "build_job_id": "136174",
        "registered_git_commit": "4c523b798a0da4bbaf7c667374818eecfe3ca45e",
        "required_git_commit": "aa7f7365012c23d8b79c02ffe509414cbf81f1c2",
        "build_terminal_state": "cancelled",
        "diagnostic_terminal_state": "failed_without_job",
        "completed_trial_count": 0,
        "retry_count": 0,
        "replacement_count": 0,
        "manual_reconcile_count": 0,
    }:
        raise InjectionAbIntegrityError("alias mismatch abort provenance drifted")
    if spec.get("v2_integrity_abort_history") != {
        "pipeline_id": "pipeline_019fb9ebfdcf73e1af0eaf02b6a12c73",
        "build_run_id": "run_019fb9ebfdcf73e1af0eaee1baff7172",
        "diagnostic_run_id": "run_019fb9ebfdcf73e1af0eaef40d8fbdff",
        "build_job_id": "136177",
        "diagnostic_job_id": "136178",
        "artifact_id": "artifact_c7fa29e3aa080c44",
        "artifact_index_sha256": (
            "858b4b9f71c5b9f830cf76745f53ca5278f146189ab1c75509ead74a95d6c161"
        ),
        "content_address": (
            "sha256:816d9388ff12ec865334a7bcee0ffec2f72b73f2bffd8c39cc4a491ec3d02b73"
        ),
        "failure_sha256": (
            "a02afcf8ff14e642cf68a18f84123ef64e247e4cc1412d94723d5a4688bf73c8"
        ),
        "progress_sha256": (
            "c35dea0115509aead7abac3356c776db365bb8c33ab93f0725b059a819e0b292"
        ),
        "git_commit": "aa7f7365012c23d8b79c02ffe509414cbf81f1c2",
        "git_tree": "a61a5979e6d861631a5cfb04d8b1942f854b48ce",
        "spec_sha256": (
            "7a8bea3b23442e30b4b25de7262972b7a4019832f842ce26c4bf3e37b7c2d615"
        ),
        "lock_sha256": (
            "391e50f4a1a0d2b0aa3776ca10ed2828e8d41cbc42ffc0ba84570729df4c3163"
        ),
        "completed_trial_count": 5,
        "arm_completed_counts": {"A": 2, "B": 3},
        "active_trial": "injection-ab-06-a",
        "retry_count": 0,
        "replacement_count": 0,
        "integrity_error": (
            "independent browser audit has no post-window heartbeat"
        ),
        "integrity_error_evidence": None,
        "root_cause": (
            "best-effort heartbeat delivery plus a fixed 750 ms sleep and "
            "one-shot snapshot; raw active-trial browser state was not "
            "checkpointed before validation"
        ),
        "vm_closed": True,
        "overlay_removed": True,
        "cleanup_errors": [],
        "manual_reconcile_count": 0,
    }:
        raise InjectionAbIntegrityError("v2 integrity abort provenance drifted")
    return spec, digest


def fixed_trial_schedule() -> list[dict[str, Any]]:
    counts = {"A": 0, "B": 0}
    schedule: list[dict[str, Any]] = []
    for trial_index, arm in enumerate(TRIAL_ORDER, start=1):
        counts[arm] += 1
        schedule.append(
            {
                "trial_index": trial_index,
                "trial_id": f"injection-ab-{trial_index:02d}-{arm.lower()}",
                "block_index": (trial_index - 1) // len(ORDER_BLOCK) + 1,
                "block_position": (trial_index - 1) % len(ORDER_BLOCK) + 1,
                "arm": arm,
                "arm_trial_index": counts[arm],
                "backend": BACKEND_BY_ARM[arm],
            }
        )
    return schedule


def _select_fixture(spec: dict[str, Any]) -> Fixture:
    manifest = load_manifest()
    if manifest.manifest_payload_sha256 != spec["manifest_payload_sha256"]:
        raise InjectionAbIntegrityError("fixture manifest seal drifted")
    fixture = manifest.by_id(str(spec["fixture_id"]))
    if (
        fixture.id != FIXTURE_ID
        or fixture.fixture_sha256 != FIXTURE_SHA256
        or fixture.split != "development"
        or fixture.template != "click"
    ):
        raise InjectionAbIntegrityError("fixed development fixture drifted")
    return fixture


def _wait_for_audit_ready(
    server: FixtureHttpServer,
    fixture: Fixture,
    generation: int,
    *,
    timeout_s: float = 2.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        snapshot = server.store.snapshot(fixture.id)
        ready = [
            item
            for item in snapshot["browser_audit_events"]
            if item.get("generation") == generation
            and item.get("event") == "audit_ready"
        ]
        if len(ready) == 1:
            return snapshot
        if len(ready) > 1:
            raise InjectionAbIntegrityError("duplicate browser audit_ready events")
        time.sleep(0.005)
    raise InjectionAbIntegrityError("browser audit_ready did not arrive before dispatch")


def validate_audit_trace(
    snapshot: dict[str, Any],
    generation: int,
    *,
    allow_request_id_gaps_after_sequence_seal: bool = False,
) -> list[dict[str, Any]]:
    if snapshot.get("browser_audit_dropped") != 0:
        raise InjectionAbIntegrityError("browser audit ring overflowed")
    raw = snapshot.get("browser_audit_events")
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise InjectionAbIntegrityError("browser audit trace malformed")
    if not all(
        isinstance(item.get("audit_sequence"), int)
        and not isinstance(item.get("audit_sequence"), bool)
        and item["audit_sequence"] >= 1
        for item in raw
    ):
        raise InjectionAbIntegrityError("browser audit sequence type/range malformed")
    events = sorted(raw, key=lambda item: item["audit_sequence"])
    sequences = [item.get("audit_sequence") for item in events]
    if sequences != list(range(1, len(events) + 1)):
        raise InjectionAbIntegrityError(
            "browser audit sequence omitted, duplicated, or reordered",
            evidence={"sequences": sequences},
        )
    ready = [item for item in events if item.get("event") == "audit_ready"]
    if len(ready) != 1 or ready[0].get("audit_sequence") != 1:
        raise InjectionAbIntegrityError("browser audit_ready identity mismatch")
    if (
        not _is_finite_number(ready[0].get("page_time_origin_ms"), minimum=0)
        or not isinstance(ready[0].get("url"), str)
        or FIXTURE_ID not in ready[0]["url"]
    ):
        raise InjectionAbIntegrityError("fresh browser audit identity malformed")
    prior_browser_wall_time_ms = -1.0
    prior_client_monotonic_ms = -1.0
    for event in events:
        if (
            not isinstance(event.get("generation"), int)
            or isinstance(event.get("generation"), bool)
            or event.get("generation") != generation
        ):
            raise InjectionAbIntegrityError("browser audit generation crossed reset")
        if (
            type(event.get("schema_version")) is not int
            or event["schema_version"] != BROWSER_AUDIT_SCHEMA_VERSION
        ):
            raise InjectionAbIntegrityError("browser audit schema version drifted")
        if event.get("event") not in BROWSER_AUDIT_EVENTS:
            raise InjectionAbIntegrityError("browser audit event set drifted")
        expected_fields = set(AUDIT_ENVELOPE_FIELDS + AUDIT_REQUIRED_FIELDS + AUDIT_DETAIL_FIELDS)
        if event["event"] == "audit_ready":
            expected_fields.update({"page_time_origin_ms", "url"})
        elif event["event"] == "audit_heartbeat":
            expected_fields.update(
                {
                    "acknowledged_heartbeat_audit_sequence",
                    "acknowledged_host_audit_request_id",
                    "acknowledged_host_monotonic_ns",
                    "expected_previous_audit_sequence",
                    "expected_audit_count_through_marker",
                }
            )
        if set(event) != expected_fields:
            raise InjectionAbIntegrityError("browser audit exact field set drifted")
        missing = [field for field in AUDIT_REQUIRED_FIELDS if field not in event]
        if missing:
            raise InjectionAbIntegrityError(
                f"browser audit required fields missing: {missing}"
            )
        for field in (
            "host_audit_request_id",
            "host_monotonic_ns",
            "host_wall_time_ns",
        ):
            value = event.get(field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise InjectionAbIntegrityError(
                    f"browser audit host field malformed: {field}"
                )
        browser_wall_time_ms = event.get("browser_wall_time_ms")
        client_monotonic_ms = event.get("client_monotonic_ms")
        if (
            not _is_finite_number(browser_wall_time_ms, minimum=0)
            or not _is_finite_number(client_monotonic_ms, minimum=0)
            or float(browser_wall_time_ms) < prior_browser_wall_time_ms
            or float(client_monotonic_ms) < prior_client_monotonic_ms
        ):
            raise InjectionAbIntegrityError("browser audit clocks malformed or reordered")
        prior_browser_wall_time_ms = float(browser_wall_time_ms)
        prior_client_monotonic_ms = float(client_monotonic_ms)
        checkbox_state = event.get("checkbox_state")
        if (
            not isinstance(checkbox_state, dict)
            or set(checkbox_state) != {"target", "decoy"}
            or not all(type(checkbox_state[key]) is bool for key in checkbox_state)
        ):
            raise InjectionAbIntegrityError("browser audit checkbox state malformed")
        _validate_audit_element(event.get("active_element"), allow_none=True)
        if not isinstance(event.get("document_has_focus"), bool):
            raise InjectionAbIntegrityError("browser audit focus state malformed")
        if event.get("visibility_state") not in {"visible", "hidden"}:
            raise InjectionAbIntegrityError("browser audit visibility state malformed")
        if event.get("event") in AUDIT_DOM_EVENTS:
            if (
                not _is_finite_number(event.get("event_time_stamp_ms"), minimum=0)
                or float(event["event_time_stamp_ms"])
                > float(client_monotonic_ms) + 1000.0
                or type(event.get("is_trusted")) is not bool
                or type(event.get("default_prevented")) is not bool
            ):
                raise InjectionAbIntegrityError("browser DOM audit fields malformed")
            _validate_audit_element(event.get("target"), allow_none=False)
            target_checked = event.get("target_checked")
            if target_checked is not None and type(target_checked) is not bool:
                raise InjectionAbIntegrityError("browser audit target state malformed")
            if target_checked != event["target"]["checked"]:
                raise InjectionAbIntegrityError("browser audit target state disagreed")
            target_id = event["target"]["id"]
            if target_id in checkbox_state and target_checked != checkbox_state[target_id]:
                raise InjectionAbIntegrityError(
                    "browser audit target and page checkbox state disagreed"
                )
            active_element = event.get("active_element")
            if (
                isinstance(active_element, dict)
                and active_element["id"] in checkbox_state
                and active_element["checked"]
                != checkbox_state[active_element["id"]]
            ):
                raise InjectionAbIntegrityError(
                    "browser audit active element and page state disagreed"
                )
            _validate_dom_event_details(event)
        else:
            if any(
                event.get(field) is not None
                for field in (
                    "event_time_stamp_ms",
                    "is_trusted",
                    "default_prevented",
                    "target",
                    "target_checked",
                    "button",
                    "buttons",
                    "pointer_type",
                    "client_x",
                    "client_y",
                    "screen_x",
                    "screen_y",
                )
            ):
                raise InjectionAbIntegrityError("browser audit marker fields malformed")
            if event["event"] == "audit_heartbeat" and (
                type(event.get("expected_previous_audit_sequence")) is not int
                or type(event.get("expected_audit_count_through_marker")) is not int
                or event["expected_previous_audit_sequence"]
                != event["audit_sequence"] - 1
                or event["expected_audit_count_through_marker"]
                != event["audit_sequence"]
            ):
                raise InjectionAbIntegrityError(
                    "browser audit heartbeat sequence/count malformed"
                )
            if event["event"] == "audit_heartbeat":
                acknowledgement = (
                    event.get("acknowledged_heartbeat_audit_sequence"),
                    event.get("acknowledged_host_audit_request_id"),
                    event.get("acknowledged_host_monotonic_ns"),
                )
                if acknowledgement != (None, None, None) and any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in acknowledgement
                ):
                    raise InjectionAbIntegrityError(
                        "browser audit heartbeat acknowledgement malformed"
                    )
    request_ids = [int(event["host_audit_request_id"]) for event in events]
    if len(set(request_ids)) != len(request_ids) or (
        not allow_request_id_gaps_after_sequence_seal
        and sorted(request_ids) != list(range(1, len(events) + 1))
    ):
        raise InjectionAbIntegrityError(
            "browser audit host request identities omitted or duplicated"
        )
    by_sequence = {int(event["audit_sequence"]): event for event in events}
    for event in events:
        if event.get("event") != "audit_heartbeat":
            continue
        acknowledged_sequence = event.get(
            "acknowledged_heartbeat_audit_sequence"
        )
        if acknowledged_sequence is None:
            continue
        acknowledged = by_sequence.get(acknowledged_sequence)
        if (
            acknowledged_sequence >= event["audit_sequence"]
            or not isinstance(acknowledged, dict)
            or acknowledged.get("event") != "audit_heartbeat"
            or acknowledged.get("host_audit_request_id")
            != event.get("acknowledged_host_audit_request_id")
            or acknowledged.get("host_monotonic_ns")
            != event.get("acknowledged_host_monotonic_ns")
        ):
            raise InjectionAbIntegrityError(
                "browser audit heartbeat acknowledgement identity drifted"
            )
    return events


def _is_finite_number(value: Any, *, minimum: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= minimum
    )


def _validate_audit_element(value: Any, *, allow_none: bool) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, dict) or set(value) != {"id", "tag", "checked"}:
        raise InjectionAbIntegrityError("browser audit element identity malformed")
    if not isinstance(value["id"], str) or not isinstance(value["tag"], str):
        raise InjectionAbIntegrityError("browser audit element identity malformed")
    if value["checked"] is not None and type(value["checked"]) is not bool:
        raise InjectionAbIntegrityError("browser audit element checked state malformed")


def _validate_dom_event_details(event: dict[str, Any]) -> None:
    for field in ("button", "buttons", "client_x", "client_y", "screen_x", "screen_y"):
        value = event.get(field)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            raise InjectionAbIntegrityError(f"browser DOM audit field malformed: {field}")
    button = event.get("button")
    buttons = event.get("buttons")
    if button is not None and not -1 <= button <= 4:
        raise InjectionAbIntegrityError("browser DOM audit button range malformed")
    if buttons is not None and not 0 <= buttons <= 31:
        raise InjectionAbIntegrityError("browser DOM audit buttons range malformed")
    pointer_type = event.get("pointer_type")
    if pointer_type is not None and pointer_type not in {"", "mouse", "pen", "touch"}:
        raise InjectionAbIntegrityError("browser DOM audit pointer type malformed")
    name = event["event"]
    pointer_or_mouse = name.startswith(("pointer", "mouse")) or name == "click"
    coordinates = tuple(event.get(field) for field in ("client_x", "client_y", "screen_x", "screen_y"))
    if pointer_or_mouse:
        if button is None or buttons is None or any(value is None for value in coordinates):
            raise InjectionAbIntegrityError("browser pointer/mouse audit details missing")
        if name.startswith("pointer") and pointer_type not in {"mouse", "pen", "touch"}:
            raise InjectionAbIntegrityError("browser pointer audit type missing")
    elif any(
        event.get(field) is not None
        for field in AUDIT_DETAIL_FIELDS
    ):
        raise InjectionAbIntegrityError("browser non-pointer audit details malformed")


def validate_post_window_audit_heartbeat(
    audit_events: list[dict[str, Any]], observation_deadline_ns: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    heartbeat_timing_evidence = {
        "observation_deadline_host_monotonic_ns": observation_deadline_ns,
        "heartbeats": [
            {
                "audit_sequence": event.get("audit_sequence"),
                "client_monotonic_ms": event.get("client_monotonic_ms"),
                "host_monotonic_ns": event.get("host_monotonic_ns"),
                "host_arrival_lag_ns": (
                    event["host_monotonic_ns"] - observation_deadline_ns
                    if isinstance(event.get("host_monotonic_ns"), int)
                    and not isinstance(event.get("host_monotonic_ns"), bool)
                    else None
                ),
                "acknowledged_heartbeat_audit_sequence": event.get(
                    "acknowledged_heartbeat_audit_sequence"
                ),
                "acknowledged_host_audit_request_id": event.get(
                    "acknowledged_host_audit_request_id"
                ),
                "acknowledged_host_monotonic_ns": event.get(
                    "acknowledged_host_monotonic_ns"
                ),
                "acknowledged_receipt_lag_ns": (
                    event["acknowledged_host_monotonic_ns"]
                    - observation_deadline_ns
                    if isinstance(
                        event.get("acknowledged_host_monotonic_ns"), int
                    )
                    and not isinstance(
                        event.get("acknowledged_host_monotonic_ns"), bool
                    )
                    else None
                ),
            }
            for event in audit_events
            if event.get("event") == "audit_heartbeat"
        ],
    }
    heartbeats = [
        event
        for event in audit_events
        if event.get("event") == "audit_heartbeat"
        and isinstance(event.get("acknowledged_host_monotonic_ns"), int)
        and not isinstance(event.get("acknowledged_host_monotonic_ns"), bool)
        and event["acknowledged_host_monotonic_ns"] > observation_deadline_ns
    ]
    if not heartbeats:
        raise InjectionAbIntegrityError(
            "independent browser audit has no causally generated post-window heartbeat",
            evidence=heartbeat_timing_evidence,
        )
    marker = min(heartbeats, key=lambda event: int(event["audit_sequence"]))
    marker_sequence = marker["audit_sequence"]
    if (
        marker.get("expected_previous_audit_sequence") != marker_sequence - 1
        or marker.get("expected_audit_count_through_marker") != marker_sequence
        or marker_sequence > len(audit_events)
    ):
        raise InjectionAbIntegrityError(
            "independent browser audit marker sequence/count mismatch",
            evidence=heartbeat_timing_evidence,
        )
    sealed = audit_events[:marker_sequence]
    if len(sealed) != marker_sequence or sealed[-1] is not marker:
        raise InjectionAbIntegrityError(
            "independent browser audit tail was not sealed",
            evidence=heartbeat_timing_evidence,
        )
    return sealed, marker


def sequence_sealed_audit_snapshot(
    snapshot: dict[str, Any], marker_sequence: int
) -> dict[str, Any]:
    if not isinstance(marker_sequence, int) or isinstance(marker_sequence, bool):
        raise InjectionAbIntegrityError("causal audit marker sequence malformed")
    raw = snapshot.get("browser_audit_events")
    if not isinstance(raw, list):
        raise InjectionAbIntegrityError("raw browser audit trace malformed")
    sealed_events = [
        event
        for event in raw
        if isinstance(event, dict)
        and isinstance(event.get("audit_sequence"), int)
        and not isinstance(event.get("audit_sequence"), bool)
        and event["audit_sequence"] <= marker_sequence
    ]
    sealed_snapshot = dict(snapshot)
    sealed_snapshot["browser_audit_events"] = sealed_events
    return sealed_snapshot


def validate_atomic_contract(
    atomic: dict[str, Any], *, arm: str, expected_endpoint: tuple[int, int] | None = None
) -> dict[str, Any]:
    semantic = atomic.get("semantic_operations")
    lowered = atomic.get("lowered_operations")
    semantic_kinds = [item.get("kind") for item in semantic or []]
    lowered_kinds = [item.get("kind") for item in lowered or []]
    if semantic_kinds != ["move_relative", "mouse_down", "mouse_up"]:
        raise InjectionAbIntegrityError("canonical semantic operations drifted")
    if lowered_kinds != ["move_relative", "click"]:
        raise InjectionAbIntegrityError("canonical lowered operations drifted")
    if atomic.get("guest_process_count") != 1:
        raise InjectionAbIntegrityError("atomic action used other than one guest process")
    if atomic.get("pointer_button_mask") != 0 or atomic.get("ok") is not True:
        raise InjectionAbIntegrityError("atomic action failed or ended with held button")
    if atomic.get("click_backend") != BACKEND_BY_ARM[arm]:
        raise InjectionAbIntegrityError("atomic action backend identity drifted")
    timestamps = atomic.get("x_injection_timestamps")
    if (
        not isinstance(timestamps, list)
        or len(timestamps) != 1
        or not isinstance(timestamps[0], dict)
    ):
        raise InjectionAbIntegrityError("guest injection timestamp record drifted")
    timestamp = timestamps[0]
    if (
        timestamp.get("click_backend") != BACKEND_BY_ARM[arm]
        or timestamp.get("clock") != "time.monotonic_ns"
        or timestamp.get("dwell_requested_ns") != 50_000_000
        or timestamp.get("press_call_success") is not True
        or timestamp.get("press_call_error") is not None
        or timestamp.get("dwell_success") is not True
        or timestamp.get("dwell_error") is not None
        or timestamp.get("release_call_success") is not True
        or timestamp.get("release_call_error") is not None
        or timestamp.get("release_side_motion_notify")
        is not RELEASE_MOTION_BY_ARM[arm]
    ):
        raise InjectionAbIntegrityError("guest injection timestamp identity drifted")
    timestamp_fields = [f"{stage}_guest_monotonic_ns" for stage in TIMESTAMP_STAGES]
    values = [timestamp.get(field) for field in timestamp_fields]
    if not all(
        isinstance(value, int) and not isinstance(value, bool) for value in values
    ) or values != sorted(values):
        raise InjectionAbIntegrityError("guest injection timestamps are not monotonic")
    primitives = atomic.get("backend_primitives")
    click = next(
        (
            item
            for item in primitives or []
            if isinstance(item, dict) and item.get("kind") == "click"
        ),
        None,
    )
    if (
        not isinstance(click, dict)
        or click.get("button") != "left"
        or click.get("click_backend") != BACKEND_BY_ARM[arm]
        or click.get("call") != "pyautogui.click(clicks=1, interval=0.05)"
        or click.get("x11_per_event_sync_hooked") is not True
        or click.get("ordering")
        != [
            "click_premove_motion",
            "mouse_down",
            "flush",
            "sync",
            "dwell",
            "mouse_up",
            "flush",
            "sync",
        ]
        or click.get("dwell_ms") != 50
        or click.get("click_premove_same_coordinate_motion_notify") is not True
        or click.get("release_side_motion_notify")
        is not RELEASE_MOTION_BY_ARM[arm]
        or click.get("injection_attempt_count") != 1
        or click.get("retry_count") != 0
        or click.get("click_premove_xtest_sequence") != ["motion_notify"]
        or click.get("press_xtest_sequence")
        != ["motion_notify", "button_press"]
        or click.get("release_xtest_sequence")
        != (
            ["motion_notify", "button_release"]
            if arm == "A"
            else ["button_release"]
        )
    ):
        raise InjectionAbIntegrityError("one-detail backend primitive contract drifted")
    evidence = atomic.get("x_injection_evidence")
    if not isinstance(evidence, list) or not all(
        isinstance(item, dict) for item in evidence
    ):
        raise InjectionAbIntegrityError("X injection evidence missing")
    sequences = [item.get("sequence") for item in evidence]
    if (
        not all(isinstance(value, int) and not isinstance(value, bool) for value in sequences)
        or sequences != list(range(1, len(evidence) + 1))
    ):
        raise InjectionAbIntegrityError("X injection global sequence drifted")
    for item in evidence:
        if (
            item.get("phase")
            not in {"canonical_move", "click_premove", "press", "release"}
            or not isinstance(item.get("event_type"), int)
            or isinstance(item.get("event_type"), bool)
            or not isinstance(item.get("detail"), int)
            or isinstance(item.get("detail"), bool)
            or item.get("attempted") is not True
            or item.get("success") is not True
            or item.get("error") is not None
            or not isinstance(item.get("started_guest_monotonic_ns"), int)
            or not isinstance(item.get("completed_guest_monotonic_ns"), int)
            or item["completed_guest_monotonic_ns"] < item["started_guest_monotonic_ns"]
            or item.get("duration_ns")
            != item["completed_guest_monotonic_ns"] - item["started_guest_monotonic_ns"]
        ):
            raise InjectionAbIntegrityError("X injection evidence schema drifted")
    sync_evidence = atomic.get("x_event_sync_evidence")
    if (
        not isinstance(sync_evidence, list)
        or not all(isinstance(item, dict) for item in sync_evidence)
        or [item.get("event") for item in sync_evidence]
        != ["mouse_down", "mouse_up"]
    ):
        raise InjectionAbIntegrityError("X sync evidence sequence drifted")
    for item in sync_evidence:
        if (
            item.get("supported") is not True
            or item.get("flush_attempted") is not True
            or item.get("flush") is not True
            or item.get("sync_attempted") is not True
            or item.get("sync") is not True
            or item.get("success") is not True
            or item.get("error") is not None
            or not isinstance(item.get("started_guest_monotonic_ns"), int)
            or isinstance(item.get("started_guest_monotonic_ns"), bool)
            or not isinstance(item.get("completed_guest_monotonic_ns"), int)
            or isinstance(item.get("completed_guest_monotonic_ns"), bool)
            or item["completed_guest_monotonic_ns"]
            < item["started_guest_monotonic_ns"]
            or item.get("duration_ns")
            != item["completed_guest_monotonic_ns"]
            - item["started_guest_monotonic_ns"]
        ):
            raise InjectionAbIntegrityError("X sync attempt evidence drifted")
    global_sync_attempts = atomic.get("x_sync_attempt_evidence")
    expected_sync_phases = [
        "initial_readback",
        "canonical_move",
        "click_premove",
        "press",
        "press",
        "press_sync",
        "release",
        "release",
        "release_sync",
        "verification_readback",
        "final_readback",
    ]
    if (
        not isinstance(global_sync_attempts, list)
        or not all(isinstance(item, dict) for item in global_sync_attempts)
        or [item.get("sequence") for item in global_sync_attempts]
        != list(range(1, len(expected_sync_phases) + 1))
        or [item.get("phase") for item in global_sync_attempts]
        != expected_sync_phases
    ):
        raise InjectionAbIntegrityError("global X sync attempt sequence drifted")
    for item in global_sync_attempts:
        if (
            item.get("attempted") is not True
            or item.get("success") is not True
            or item.get("error") is not None
            or not isinstance(item.get("started_guest_monotonic_ns"), int)
            or isinstance(item.get("started_guest_monotonic_ns"), bool)
            or not isinstance(item.get("completed_guest_monotonic_ns"), int)
            or isinstance(item.get("completed_guest_monotonic_ns"), bool)
            or item["completed_guest_monotonic_ns"]
            < item["started_guest_monotonic_ns"]
            or item.get("duration_ns")
            != item["completed_guest_monotonic_ns"]
            - item["started_guest_monotonic_ns"]
        ):
            raise InjectionAbIntegrityError("global X sync attempt evidence drifted")
    controlled = [
        item for item in evidence if item.get("phase") in {"press", "release"}
    ]
    press_events = [item.get("event") for item in controlled if item.get("phase") == "press"]
    release_events = [
        item.get("event") for item in controlled if item.get("phase") == "release"
    ]
    if press_events != ["motion_notify", "button_press"] or release_events != (
        ["motion_notify", "button_release"]
        if arm == "A"
        else ["button_release"]
    ):
        raise InjectionAbIntegrityError("low-level XTest event sequence drifted")
    motion_events = [
        item
        for item in controlled
        if item.get("phase") == "release" and item.get("event") == "motion_notify"
    ]
    if bool(motion_events) is not RELEASE_MOTION_BY_ARM[arm]:
        raise InjectionAbIntegrityError("release-side MotionNotify delta drifted")
    start_sequence = timestamp.get("x_injection_start_sequence")
    expected_events = [
        ("press", "motion_notify", X_MOTION_NOTIFY, 0),
        ("press", "button_press", X_BUTTON_PRESS, 1),
        *(
            [("release", "motion_notify", X_MOTION_NOTIFY, 0)]
            if arm == "A"
            else []
        ),
        ("release", "button_release", X_BUTTON_RELEASE, 1),
    ]
    expected_click_window = [
        ("click_premove", "motion_notify", X_MOTION_NOTIFY, 0),
        *expected_events,
    ]
    end_sequence = timestamp.get("x_injection_end_sequence")
    if (
        not isinstance(start_sequence, int)
        or isinstance(start_sequence, bool)
        or start_sequence != 1
        or not isinstance(end_sequence, int)
        or isinstance(end_sequence, bool)
        or end_sequence != start_sequence + len(expected_click_window)
        or len(evidence) != end_sequence
        or timestamp.get("click_premove_xtest_sequence") != ["motion_notify"]
        or [item["sequence"] for item in controlled]
        != list(range(start_sequence + 2, end_sequence + 1))
    ):
        raise InjectionAbIntegrityError("controlled XTest event order drifted")
    click_premove = evidence[start_sequence]
    if (
        click_premove.get("sequence") != start_sequence + 1
        or (
            click_premove.get("phase"),
            click_premove.get("event"),
            click_premove.get("event_type"),
            click_premove.get("detail"),
        )
        != expected_click_window[0]
    ):
        raise InjectionAbIntegrityError("click premove XTest identity drifted")
    for item, (phase, name, event_type, detail) in zip(controlled, expected_events):
        if (
            item.get("phase") != phase
            or item.get("event") != name
            or item.get("event_type") != event_type
            or item.get("detail") != detail
        ):
            raise InjectionAbIntegrityError("controlled XTest event identity drifted")
    click_clock_values = [
        value
        for item in [click_premove, *controlled]
        for value in (
            item["started_guest_monotonic_ns"],
            item["completed_guest_monotonic_ns"],
        )
    ]
    if click_clock_values != sorted(click_clock_values):
        raise InjectionAbIntegrityError("controlled XTest event clock order drifted")
    press_motion = controlled[0]
    press_coordinates = (press_motion.get("x"), press_motion.get("y"))
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in press_coordinates):
        raise InjectionAbIntegrityError("press MotionNotify coordinates malformed")
    if expected_endpoint is not None and press_coordinates != expected_endpoint:
        raise InjectionAbIntegrityError("press MotionNotify missed the fixed endpoint")
    if (click_premove.get("x"), click_premove.get("y")) != press_coordinates:
        raise InjectionAbIntegrityError(
            "click premove MotionNotify was not at the same cursor coordinates"
        )
    canonical_move = evidence[0]
    if (
        canonical_move.get("sequence") != 1
        or canonical_move.get("phase") != "canonical_move"
        or canonical_move.get("event") != "motion_notify"
        or canonical_move.get("event_type") != X_MOTION_NOTIFY
        or canonical_move.get("detail") != 0
        or (canonical_move.get("x"), canonical_move.get("y"))
        != press_coordinates
    ):
        raise InjectionAbIntegrityError("canonical move XTest evidence drifted")
    cursor_after = atomic.get("cursor_after")
    if (
        not isinstance(cursor_after, list)
        or len(cursor_after) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in cursor_after)
        or list(press_coordinates) != cursor_after
    ):
        raise InjectionAbIntegrityError("XTest click coordinates disagree with cursor readback")
    if atomic.get("final_pointer_readback") != {
        "attempted": True,
        "success": True,
        "error": None,
        "cursor": cursor_after,
        "pointer_button_mask": 0,
    }:
        raise InjectionAbIntegrityError("final pointer readback evidence drifted")
    for item in controlled:
        if item["event"] in {"button_press", "button_release"} and (
            item.get("x") is not None or item.get("y") is not None
        ):
            raise InjectionAbIntegrityError("XTest button event coordinates drifted")
    if arm == "A" and (
        motion_events[0].get("x"), motion_events[0].get("y")
    ) != press_coordinates:
        raise InjectionAbIntegrityError(
            "release MotionNotify was not at the same cursor coordinates"
        )
    if not (
        timestamp["click_started_guest_monotonic_ns"]
        <= click_premove["started_guest_monotonic_ns"]
        <= click_premove["completed_guest_monotonic_ns"]
        <= timestamp["press_call_before_guest_monotonic_ns"]
        <= controlled[0]["started_guest_monotonic_ns"]
        <= controlled[1]["completed_guest_monotonic_ns"]
        <= timestamp["press_call_after_guest_monotonic_ns"]
        <= timestamp["press_sync_completed_guest_monotonic_ns"]
        <= timestamp["dwell_started_guest_monotonic_ns"]
        <= timestamp["dwell_completed_guest_monotonic_ns"]
        <= timestamp["release_call_before_guest_monotonic_ns"]
        <= controlled[2]["started_guest_monotonic_ns"]
        <= controlled[-1]["completed_guest_monotonic_ns"]
        <= timestamp["release_call_after_guest_monotonic_ns"]
        <= timestamp["release_sync_completed_guest_monotonic_ns"]
        <= timestamp["click_completed_guest_monotonic_ns"]
    ):
        raise InjectionAbIntegrityError("XTest event/timestamp ordering drifted")
    actual_dwell_ns = timestamp.get("dwell_duration_ns")
    if (
        not isinstance(actual_dwell_ns, int)
        or isinstance(actual_dwell_ns, bool)
        or actual_dwell_ns < 50_000_000
    ):
        raise InjectionAbIntegrityError("explicit 50ms dwell was not observed")
    passive = atomic.get("passive_x_observer")
    if passive != {
        "installed": False,
        "observer_process_count": 0,
        "additional_x_connection_count": 0,
        "assessment": "omitted_not_demonstrably_non_perturbing",
        "limitation": (
            "not installed: a same-process XRecord/XI2 observer requires a second X "
            "connection and concurrent event consumption, which is not demonstrably "
            "non-perturbing for this timing experiment"
        ),
    }:
        raise InjectionAbIntegrityError("passive X observer limitation drifted")
    return {
        "semantic_kinds": semantic_kinds,
        "lowered_kinds": lowered_kinds,
        "click_primitive": click,
        "x_injection_timestamps": timestamps,
        "x_injection_evidence": evidence,
        "x_event_sync_evidence": sync_evidence,
        "x_sync_attempt_evidence": global_sync_attempts,
        "final_pointer_readback": atomic.get("final_pointer_readback"),
        "passive_x_observer": atomic.get("passive_x_observer"),
    }


def classify_trial_outcome(
    audit_events: list[dict[str, Any]], snapshot: dict[str, Any]
) -> dict[str, Any]:
    dom_events = [
        event for event in audit_events if event["event"] in AUDIT_DOM_EVENTS
    ]

    def trusted_target(name: str) -> list[dict[str, Any]]:
        return [
            event
            for event in dom_events
            if event.get("event") == name
            and event.get("is_trusted") is True
            and isinstance(event.get("target"), dict)
            and event["target"].get("id") == "target"
        ]

    clicks = trusted_target("click")
    inputs = trusted_target("input")
    changes = trusted_target("change")
    pointer_down = trusted_target("pointerdown")
    pointer_up = trusted_target("pointerup")
    primary_success = bool(
        pointer_down and pointer_up and clicks and inputs and changes
    ) and all(
        events[-1].get("checkbox_state", {}).get("target") is True
        for events in (clicks, inputs, changes)
    )
    if primary_success:
        outcome = "trusted_click_input_change"
    elif pointer_down and pointer_up and not clicks:
        outcome = "trusted_pointer_release_without_click"
    elif pointer_down and not pointer_up:
        outcome = "trusted_pointerdown_without_pointerup"
    elif not pointer_down:
        outcome = "no_trusted_pointerdown"
    else:
        outcome = "partial_dom_activation"
    host_events = [
        event
        for event in snapshot.get("events", [])
        if event.get("kind") in {"pointer", "click"}
    ]
    return {
        "primary_success": primary_success,
        "outcome": outcome,
        "audit_event_sequence": [event["event"] for event in dom_events],
        "audit_target_event_sequence": [
            event["event"]
            for event in dom_events
            if isinstance(event.get("target"), dict)
            and event["target"].get("id") == "target"
        ],
        "trusted_pointerdown_count": len(pointer_down),
        "trusted_pointerup_count": len(pointer_up),
        "trusted_click_count": len(clicks),
        "trusted_input_count": len(inputs),
        "trusted_change_count": len(changes),
        "host_reporter_event_sequence": [
            event.get("event") if event.get("kind") == "pointer" else "click"
            for event in host_events
        ],
        "host_reporter_checked": snapshot.get("current", {}).get("checked"),
        "focus_states": [event.get("document_has_focus") for event in dom_events],
        "visibility_states": [event.get("visibility_state") for event in dom_events],
        "default_prevented_events": [
            event["event"]
            for event in dom_events
            if event.get("default_prevented") is True
        ],
    }


def interpret_results(trials: list[dict[str, Any]]) -> dict[str, Any]:
    failures = {
        arm: sum(
            trial["arm"] == arm and not trial["outcome"]["primary_success"]
            for trial in trials
        )
        for arm in ("A", "B")
    }
    if failures["A"] == 0 and failures["B"] == 0:
        interpretation = "failure_not_reproduced"
    elif failures["A"] > 0 and failures["B"] == 0:
        interpretation = "supports_release_motion_hypothesis"
    elif failures["A"] == 0 and failures["B"] > 0:
        interpretation = "contradicts_release_motion_hypothesis"
    elif failures["A"] > failures["B"]:
        interpretation = "directional_support_with_shared_failures_inconclusive"
    elif failures["A"] == failures["B"]:
        interpretation = "nondifferential_failures_inconclusive"
    else:
        interpretation = "directional_contradiction_with_shared_failures"
    return {
        "interpretation": interpretation,
        "failure_counts": failures,
        "success_counts": {arm: 24 - failures[arm] for arm in ("A", "B")},
        "descriptive_only": True,
        "qualification_authorized": False,
    }


def _checkpoint(
    output: Path,
    *,
    trials: list[dict[str, Any]],
    active_trial: dict[str, Any] | None,
    stage: str,
    integrity_error: dict[str, Any] | None = None,
) -> None:
    _atomic_json(
        output / "injection_ab_progress.json",
        {
            "schema_version": 1,
            "suite": SUITE,
            "status": (
                "integrity_abort"
                if integrity_error
                else "completed"
                if stage == "completed"
                else "running"
            ),
            "stage": stage,
            "expected_trial_count": len(TRIAL_ORDER),
            "completed_trial_count": len(trials),
            "arm_completed_counts": {
                arm: sum(trial["arm"] == arm for trial in trials)
                for arm in ("A", "B")
            },
            "active_trial": active_trial,
            "integrity_error": integrity_error,
            "retry_count": 0,
            "replacement_count": 0,
            "gpu_count": 0,
            "model_access": False,
            "sealed_evaluation_access": False,
            "trials": trials,
        },
    )


def _checkpoint_dispatched_trial(
    *,
    output: Path,
    trials: list[dict[str, Any]],
    active: dict[str, Any],
    fixture: Fixture,
    dispatch: list[dict[str, Any]],
    journal: dict[str, Any],
    arm: str,
    expected_endpoint: tuple[int, int],
    dispatch_started_ns: int,
    dispatch_completed_ns: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    active = {
        **active,
        "dispatch": dispatch,
        "journal": journal,
        "dispatch_started_host_monotonic_ns": dispatch_started_ns,
        "dispatch_completed_host_monotonic_ns": dispatch_completed_ns,
    }
    _checkpoint(
        output,
        trials=trials,
        active_trial=active,
        stage="dispatched",
    )
    _assert_dispatch_journal(
        fixture, "compact_raw_phaseb", "injection A/B", journal
    )
    atomic_states = journal.get("atomic_action_states")
    if (
        len(dispatch) != 1
        or journal.get("completed_action_count") != 1
        or not isinstance(atomic_states, list)
        or len(atomic_states) != 1
        or not isinstance(atomic_states[0], dict)
    ):
        raise InjectionAbIntegrityError("dispatch count or atomic state drifted")
    atomic_contract = validate_atomic_contract(
        atomic_states[0], arm=arm, expected_endpoint=expected_endpoint
    )
    active = {**active, "atomic_contract": atomic_contract}
    _checkpoint(
        output,
        trials=trials,
        active_trial=active,
        stage="atomic_validated",
    )
    return atomic_contract, active


def _checkpoint_browser_audit_capture(
    *,
    output: Path,
    trials: list[dict[str, Any]],
    active: dict[str, Any],
    snapshot: dict[str, Any],
    wait_evidence: dict[str, Any],
) -> dict[str, Any]:
    active = {
        **active,
        "raw_browser_snapshot": snapshot,
        "post_window_heartbeat_wait": wait_evidence,
    }
    _checkpoint(
        output,
        trials=trials,
        active_trial=active,
        stage="browser_audit_captured",
    )
    return active


def validate_injection_ab() -> dict[str, Any]:
    spec, spec_sha256 = load_injection_ab_spec()
    fixture = _select_fixture(spec)
    schedule = fixed_trial_schedule()
    if [item["arm"] for item in schedule] != list(TRIAL_ORDER):
        raise InjectionAbIntegrityError("fixed A/B schedule drifted")
    return {
        "schema_version": 1,
        "status": "passed",
        "mode": "validate",
        "suite": spec["suite"],
        "spec_sha256": spec_sha256,
        "fixture_id": fixture.id,
        "fixture_sha256": fixture.fixture_sha256,
        "trial_count": len(schedule),
        "arm_trial_counts": {
            arm: sum(item["arm"] == arm for item in schedule)
            for arm in ("A", "B")
        },
        "ordered_arms": [item["arm"] for item in schedule],
        "retry_count": 0,
        "replacement_count": 0,
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
        "qualification_authorized": False,
    }


def run_vm_injection_ab(
    *,
    output: Path,
    qcow: Path,
    qemu: Path,
    provider_path: Path,
    expected_provider_sha256: str | None,
) -> dict[str, Any]:
    spec, spec_sha256 = load_injection_ab_spec()
    fixture = _select_fixture(spec)
    provider_sha256 = sha256_file(provider_path)
    if expected_provider_sha256 and provider_sha256 != expected_provider_sha256:
        raise InjectionAbIntegrityError("KVM provider hash mismatch")
    schedule = fixed_trial_schedule()
    trials: list[dict[str, Any]] = []
    _checkpoint(
        output, trials=trials, active_trial=None, stage="starting_vm"
    )
    with FixtureHttpServer(
        load_manifest(), enable_browser_audit=True
    ) as server, KvmFixtureSession(
        qcow=qcow,
        qemu=qemu,
        provider_path=provider_path,
        vm_log_dir=output / "vm_logs",
    ) as session:
        for scheduled in schedule:
            active = {**scheduled, "fixture_id": fixture.id}
            _checkpoint(
                output,
                trials=trials,
                active_trial=active,
                stage="resetting_trial",
            )
            transport = session.reset_to_ready()
            initial = session.launch_fixture(server, fixture)
            _validate_loaded_geometry(fixture, initial, transport.screen_size())
            if initial.get("current") != {
                "checked": False,
                "decoy_checked": False,
            }:
                raise InjectionAbIntegrityError("trial reset state was not clean")
            generation = int(initial["generation"])
            ready_snapshot = _wait_for_audit_ready(
                server, fixture, generation
            )
            audit_ready = next(
                event
                for event in ready_snapshot["browser_audit_events"]
                if event.get("event") == "audit_ready"
            )
            baseline = transport.cursor_position()
            trajectory = build_trajectory(
                fixture,
                initial,
                arm="compact_raw_phaseb",
                cursor=baseline,
                near_miss=False,
            )
            if len(trajectory.actions) != 1 or trajectory.expected_endpoint is None:
                raise InjectionAbIntegrityError("fixed click trajectory drifted")
            bound = BackendBoundTransport(transport, scheduled["backend"])
            dispatch_started_ns = time.monotonic_ns()
            dispatch, journal = _execute(
                "compact_raw_phaseb", bound, trajectory
            )
            dispatch_completed_ns = time.monotonic_ns()
            atomic_contract, active = _checkpoint_dispatched_trial(
                output=output,
                trials=trials,
                active=active,
                fixture=fixture,
                dispatch=dispatch,
                journal=journal,
                arm=scheduled["arm"],
                expected_endpoint=trajectory.expected_endpoint,
                dispatch_started_ns=dispatch_started_ns,
                dispatch_completed_ns=dispatch_completed_ns,
            )
            observation_deadline_ns = dispatch_completed_ns + int(
                OBSERVATION_WINDOW_S * 1_000_000_000
            )
            remaining_ns = observation_deadline_ns - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns / 1_000_000_000)
            snapshot, heartbeat_wait = (
                server.store.wait_for_causal_post_window_heartbeat(
                    fixture.id,
                    generation=generation,
                    observation_deadline_host_monotonic_ns=(
                        observation_deadline_ns
                    ),
                    timeout_s=POST_WINDOW_HEARTBEAT_WAIT_S,
                )
            )
            active = _checkpoint_browser_audit_capture(
                output=output,
                trials=trials,
                active=active,
                snapshot=snapshot,
                wait_evidence=heartbeat_wait,
            )
            if heartbeat_wait["timed_out"]:
                raise InjectionAbIntegrityError(
                    "bounded causal post-window heartbeat wait timed out",
                    evidence=heartbeat_wait,
                )
            marker_sequence = min(
                heartbeat_wait["candidate_audit_sequences"]
            )
            sealed_snapshot = sequence_sealed_audit_snapshot(
                snapshot, marker_sequence
            )
            audit_events = validate_audit_trace(
                sealed_snapshot,
                generation,
                allow_request_id_gaps_after_sequence_seal=True,
            )
            sealed_audit_events, audit_heartbeat = validate_post_window_audit_heartbeat(
                audit_events, observation_deadline_ns
            )
            outcome = classify_trial_outcome(sealed_audit_events, snapshot)
            trial = {
                **scheduled,
                "status": "observed",
                "fixture_id": fixture.id,
                "fixture_sha256": fixture.fixture_sha256,
                "reset_snapshot": READY_SNAPSHOT,
                "fresh_chromium": {
                    "generation": generation,
                    "page_time_origin_ms": audit_ready["page_time_origin_ms"],
                    "url": audit_ready["url"],
                },
                "canonical_action_format": "compact_raw_phaseb",
                "baseline": list(baseline),
                "endpoint": list(trajectory.expected_endpoint),
                "dispatch_count": 1,
                "retry_count": 0,
                "replacement_count": 0,
                "dispatch": dispatch,
                "atomic_contract": atomic_contract,
                "final_pointer_button_mask": journal[
                    "final_pointer_button_mask"
                ],
                "audit_events": audit_events,
                "sealed_audit_events": sealed_audit_events,
                "post_window_audit_heartbeat": audit_heartbeat,
                "post_window_heartbeat_wait": heartbeat_wait,
                "outcome": outcome,
                "timings": {
                    "dispatch_started_host_monotonic_ns": dispatch_started_ns,
                    "dispatch_completed_host_monotonic_ns": dispatch_completed_ns,
                    "observation_deadline_host_monotonic_ns": observation_deadline_ns,
                    "audit_capture_completed_host_monotonic_ns": time.monotonic_ns(),
                    "observation_window_s": OBSERVATION_WINDOW_S,
                    "post_window_heartbeat_wait_s": (
                        POST_WINDOW_HEARTBEAT_WAIT_S
                    ),
                },
            }
            trials.append(trial)
            _checkpoint(
                output,
                trials=trials,
                active_trial=active,
                stage="trial_observed",
            )
    if len(trials) != len(schedule):
        raise InjectionAbIntegrityError("fixed trial horizon incomplete")
    if {arm: sum(t["arm"] == arm for t in trials) for arm in ("A", "B")} != {
        "A": 24,
        "B": 24,
    }:
        raise InjectionAbIntegrityError("arm allocation drifted")
    browser_identities = [
        (
            trial["fresh_chromium"]["generation"],
            trial["fresh_chromium"]["page_time_origin_ms"],
        )
        for trial in trials
    ]
    if len(set(browser_identities)) != len(browser_identities):
        raise InjectionAbIntegrityError("fresh Chromium identity repeated")
    interpretation = interpret_results(trials)
    result = {
        "schema_version": 1,
        "status": "completed",
        "suite": spec["suite"],
        "mode": "vm",
        "spec_sha256": spec_sha256,
        "parent_evidence": spec["parent_evidence"],
        "manifest_payload_sha256": MANIFEST_PAYLOAD_SHA256,
        "fixture_id": fixture.id,
        "fixture_sha256": fixture.fixture_sha256,
        "snapshot_name": READY_SNAPSHOT,
        "trial_count": len(trials),
        "arm_trial_counts": {"A": 24, "B": 24},
        "ordered_arms": [trial["arm"] for trial in trials],
        "reset_count": len(trials),
        "fresh_chromium_count": len(trials),
        "dispatch_count": len(trials),
        "retry_count": 0,
        "replacement_count": 0,
        "stop_policy": "integrity_abort_only",
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
        "qualification_authorized": False,
        "passive_x_observer": spec["passive_x_observer"],
        "provider": {
            "path": str(provider_path.resolve()),
            "sha256": provider_sha256,
        },
        "qcow": {"path": str(qcow.resolve()), "size": qcow.stat().st_size},
        "qemu": str(qemu.resolve()),
        "interpretation": interpretation,
        "trials": trials,
    }
    _checkpoint(
        output, trials=trials, active_trial=None, stage="completed"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "vm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qcow", type=Path, default=DEFAULT_QCOW)
    parser.add_argument("--qemu", type=Path, default=DEFAULT_QEMU)
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument(
        "--expected-provider-sha256",
        "--expected_provider_sha256",
        dest="expected_provider_sha256",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "injection_ab_result.json"
    failure_path = args.output / "injection_ab_failure.json"
    result_path.unlink(missing_ok=True)
    failure_path.unlink(missing_ok=True)
    if args.mode == "vm":
        _checkpoint(
            args.output,
            trials=[],
            active_trial=None,
            stage="entry",
        )
    try:
        result = (
            validate_injection_ab()
            if args.mode == "validate"
            else run_vm_injection_ab(
                output=args.output,
                qcow=args.qcow,
                qemu=args.qemu,
                provider_path=args.provider,
                expected_provider_sha256=args.expected_provider_sha256,
            )
        )
        _atomic_json(result_path, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        integrity_error = {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "evidence": getattr(exc, "evidence", None),
        }
        progress_path = args.output / "injection_ab_progress.json"
        progress = (
            json.loads(progress_path.read_text()) if progress_path.exists() else {}
        )
        if progress:
            progress.update(
                {
                    "status": "integrity_abort",
                    "stage": "integrity_abort",
                    "integrity_error": integrity_error,
                }
            )
            _atomic_json(progress_path, progress)
        _atomic_json(
            failure_path,
            {
                "schema_version": 1,
                "status": "integrity_abort",
                "suite": SUITE,
                "failure_kind": "infrastructure",
                "integrity_error": integrity_error,
                "progress": progress,
                "retry_count": 0,
                "replacement_count": 0,
                "gpu_count": 0,
                "model_access": False,
                "sealed_evaluation_access": False,
                "traceback": traceback.format_exc(),
            },
        )
        print(
            json.dumps(
                {"status": "integrity_abort", **integrity_error},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
