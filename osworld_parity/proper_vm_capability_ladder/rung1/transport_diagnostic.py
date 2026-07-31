from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from .fixtures import Fixture, FixtureManifest, load_manifest
from .selfcheck import (
    _assert_dispatch_journal,
    _atomic_json,
    _execute,
    _validate_loaded_geometry,
)
from .server import FixtureHttpServer
from .transport import ATOMIC_RESULT_PREFIX
from .trajectory import Arm, build_trajectory
from .vm import (
    DEFAULT_PROVIDER,
    DEFAULT_QCOW,
    DEFAULT_QEMU,
    READY_SNAPSHOT,
    KvmFixtureSession,
    sha256_file,
)


SPEC_PATH = Path(__file__).with_name("transport_diagnostic_spec.json")
CERTIFICATION_SPEC_PATH = Path(__file__).with_name(
    "transport_certification_spec.json"
)
FIXTURE_ID = "r1a-click-dev-1101"
FIXTURE_SHA256 = "0124b5dab062e69ed83c37f9b91396b152b1f27a6cd3b9de72a0f9fa18ff5c0e"
MANIFEST_PAYLOAD_SHA256 = (
    "5d4ea3ab33c084f1a5de1b716429c242a97452416f5b74efc3654b7d4b338097"
)
PAIR_COUNT = 5
CERTIFICATION_SHARD_COUNT = 4
CERTIFICATION_PAIRS_PER_SHARD = 25
ARM_ORDER: tuple[Arm, ...] = (
    "native_absolute_control",
    "compact_raw_phaseb",
)
SEMANTIC_KINDS = ("move_to", "mouse_down", "mouse_up")
LOWERED_KINDS = ("click",)
BROWSER_SEQUENCE = ("pointerdown", "pointerup", "click")
X_EVENT_SEQUENCE = ("mouse_down", "mouse_up")
CLICK_CALL = "pyautogui.click(clicks=1, interval=0.05)"
ATTEMPT_EVIDENCE_SCHEMA = "rung1_transport_attempt_evidence_v1"
POST_TIMEOUT_OBSERVATION_GRACE_S = 0.25
TIMEOUT_CLASSIFIER_RULES = {
    "guest_input_path": (
        "passive X observer lacks button release and the post-grace pointer mask is held"
    ),
    "chromium_input_delivery": (
        "passive X observer has button release with pointer mask zero but the browser "
        "local ring lacks pointerup"
    ),
    "browser_reporter": (
        "browser local ring has pointerup and click but its serialized reporter remains "
        "queued or unresolved"
    ),
    "host_harness": (
        "one request-correlated required event was rejected before timeout, or the full "
        "required predicate was committed and acknowledged with its quiet window strictly "
        "complete before timeout but the waiter still timed out"
    ),
    "inconclusive": "none of the identifying evidence rules is satisfied",
}


class TransportDiagnosticError(RuntimeError):
    def __init__(self, message: str, *, evidence: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.evidence = evidence


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def certification_trial_identity(
    spec_sha256: str, shard_index: int, pair_index: int, arm: Arm
) -> tuple[int, str, str]:
    if len(spec_sha256) != 64:
        raise TransportDiagnosticError("certification identity requires a spec SHA256")
    if shard_index not in range(CERTIFICATION_SHARD_COUNT):
        raise TransportDiagnosticError("certification shard index must be in [0, 3]")
    if pair_index not in range(1, CERTIFICATION_PAIRS_PER_SHARD + 1):
        raise TransportDiagnosticError("certification pair index must be in [1, 25]")
    if arm not in ARM_ORDER:
        raise TransportDiagnosticError(f"unsupported certification arm: {arm}")
    global_pair_index = shard_index * CERTIFICATION_PAIRS_PER_SHARD + pair_index
    pair_id = (
        f"cert-{spec_sha256[:12]}-s{shard_index}-pair-{global_pair_index:03d}"
    )
    arm_slug = "native" if arm == ARM_ORDER[0] else "compact"
    return global_pair_index, pair_id, f"{pair_id}-{arm_slug}"


def load_transport_diagnostic_spec(
    path: Path = SPEC_PATH,
) -> tuple[dict[str, Any], str]:
    raw_bytes = path.read_bytes()
    try:
        spec = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise TransportDiagnosticError("transport diagnostic spec is invalid JSON") from exc
    if not isinstance(spec, dict):
        raise TransportDiagnosticError("transport diagnostic spec must be an object")
    expected = {
        "schema_version": 1,
        "suite": "rung1_shared_click_transport_diagnostic",
        "status": "authorized_narrow_diagnostic",
        "manifest_payload_sha256": MANIFEST_PAYLOAD_SHA256,
        "fixture_id": FIXTURE_ID,
        "fixture_sha256": FIXTURE_SHA256,
        "pair_count": PAIR_COUNT,
        "arm_order": list(ARM_ORDER),
        "reset_before_every_trial": True,
        "stop_on_first_mismatch": True,
        "required_browser_sequence": list(BROWSER_SEQUENCE),
        "required_semantic_operations": list(SEMANTIC_KINDS),
        "required_lowered_operations": list(LOWERED_KINDS),
        "required_backend_primitive": CLICK_CALL,
        "required_x_event_sync": list(X_EVENT_SEQUENCE),
        "required_final_pointer_mask": 0,
        "dispatches_per_trial": 1,
        "retry_count": 0,
        "oracle_conditioned_dispatch": False,
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
    }
    if spec != expected:
        differing = sorted(
            key
            for key in set(spec) | set(expected)
            if spec.get(key) != expected.get(key)
        )
        raise TransportDiagnosticError(
            f"transport diagnostic spec drifted in fields: {differing}"
        )
    return spec, hashlib.sha256(raw_bytes).hexdigest()


def load_transport_certification_spec(
    path: Path = CERTIFICATION_SPEC_PATH,
) -> tuple[dict[str, Any], str]:
    raw_bytes = path.read_bytes()
    try:
        spec = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise TransportDiagnosticError(
            "transport certification spec is invalid JSON"
        ) from exc
    expected = {
        "schema_version": 1,
        "suite": "rung1_shared_click_transport_certification",
        "status": "authorized_full_certification",
        "manifest_payload_sha256": MANIFEST_PAYLOAD_SHA256,
        "fixture_id": FIXTURE_ID,
        "fixture_sha256": FIXTURE_SHA256,
        "shard_count": CERTIFICATION_SHARD_COUNT,
        "pairs_per_shard": CERTIFICATION_PAIRS_PER_SHARD,
        "total_pair_count": 100,
        "trials_per_shard": 50,
        "total_trial_count": 200,
        "arm_order": list(ARM_ORDER),
        "reset_before_every_trial": True,
        "stop_on_first_mismatch": True,
        "required_browser_sequence": list(BROWSER_SEQUENCE),
        "required_semantic_operations": list(SEMANTIC_KINDS),
        "required_lowered_operations": list(LOWERED_KINDS),
        "required_backend_primitive": CLICK_CALL,
        "required_x_event_sync": list(X_EVENT_SEQUENCE),
        "required_final_pointer_mask": 0,
        "dispatches_per_trial": 1,
        "retry_count": 0,
        "oracle_conditioned_dispatch": False,
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
    }
    if spec != expected:
        differing = sorted(
            key
            for key in set(spec) | set(expected)
            if spec.get(key) != expected.get(key)
        )
        raise TransportDiagnosticError(
            f"transport certification spec drifted in fields: {differing}"
        )
    return spec, hashlib.sha256(raw_bytes).hexdigest()


def _select_fixture(manifest: FixtureManifest, spec: dict[str, Any]) -> Fixture:
    if manifest.manifest_payload_sha256 != spec["manifest_payload_sha256"]:
        raise TransportDiagnosticError("frozen fixture manifest seal drifted")
    fixture = manifest.by_id(str(spec["fixture_id"]))
    if (
        fixture.id != FIXTURE_ID
        or fixture.fixture_sha256 != spec["fixture_sha256"]
        or fixture.split != "development"
        or fixture.template != "click"
    ):
        raise TransportDiagnosticError("fixed development click fixture contract drifted")
    return fixture


def _semantic_contract(
    dispatch: list[dict[str, Any]], endpoint: tuple[int, int]
) -> list[dict[str, Any]]:
    if len(dispatch) != 1:
        raise TransportDiagnosticError(
            f"trial dispatched {len(dispatch)} actions instead of exactly one"
        )
    record = dispatch[0]
    if record.get("parse_status") != "ok" or record.get("executor_dispatch_status") != "ok":
        raise TransportDiagnosticError(f"adapter dispatch failed: {record}")
    operations = record.get("operations")
    expected = [
        {"kind": "move_to", "args": [endpoint[0], endpoint[1]]},
        {"kind": "mouse_down", "args": ["left"]},
        {"kind": "mouse_up", "args": ["left"]},
    ]
    if operations != expected:
        raise TransportDiagnosticError(
            f"semantic operation contract mismatch: {operations} != {expected}"
        )
    return operations


def _atomic_contract(journal: dict[str, Any]) -> dict[str, Any]:
    states = journal.get("atomic_action_states")
    if not isinstance(states, list) or len(states) != 1:
        raise TransportDiagnosticError("trial did not produce exactly one atomic action state")
    state = states[0]
    if not isinstance(state, dict):
        raise TransportDiagnosticError("atomic action state is not an object")
    required = {
        "ok": True,
        "pointer_button_mask": 0,
        "observed_pointer_button_mask": 0,
        "expected_pointer_button_mask": 0,
        "guest_process_count": 1,
        "guest_returncode": 0,
        "cleanup_attempted": False,
        "error": None,
        "failure_kind": None,
    }
    mismatches = {
        key: {"observed": state.get(key), "expected": value}
        for key, value in required.items()
        if state.get(key) != value
    }
    if mismatches:
        raise TransportDiagnosticError(f"atomic state contract mismatch: {mismatches}")
    if not str(state.get("raw_result_marker", "")).startswith(ATOMIC_RESULT_PREFIX):
        raise TransportDiagnosticError("atomic state lacks raw result marker evidence")

    for field in (
        "cursor_before",
        "cursor_after",
        "semantic_operations",
        "lowered_operations",
    ):
        if not isinstance(state.get(field), list):
            raise TransportDiagnosticError(f"atomic state lacks {field} evidence")
    if state.get("cursor") != state.get("cursor_after"):
        raise TransportDiagnosticError("atomic cursor alias/readback mismatch")
    lowered = state["lowered_operations"]
    if [item.get("kind") for item in lowered][-1:] != ["click"]:
        raise TransportDiagnosticError(f"semantic click was not lowered once: {lowered}")

    primitives = state.get("backend_primitives")
    if not isinstance(primitives, list):
        raise TransportDiagnosticError("atomic state lacks backend primitives")
    click_primitives = [
        item
        for item in primitives
        if isinstance(item, dict) and item.get("kind") == "click"
    ]
    if len(click_primitives) != 1:
        raise TransportDiagnosticError("click did not lower to exactly one backend primitive")
    primitive = click_primitives[0]
    if not isinstance(primitive, dict) or {
        "kind": primitive.get("kind"),
        "button": primitive.get("button"),
        "call": primitive.get("call"),
        "x11_per_event_sync_hooked": primitive.get("x11_per_event_sync_hooked"),
    } != {
        "kind": "click",
        "button": "left",
        "call": CLICK_CALL,
        "x11_per_event_sync_hooked": True,
    }:
        raise TransportDiagnosticError(f"lowered click primitive mismatch: {primitive}")

    sync = state.get("x_event_sync_evidence")
    if not isinstance(sync, list) or [item.get("event") for item in sync] != list(
        X_EVENT_SEQUENCE
    ):
        raise TransportDiagnosticError(f"X event sync sequence mismatch: {sync}")
    for item in sync:
        if (
            not isinstance(item, dict)
            or item.get("flush") is not True
            or item.get("sync") is not True
            or "x11" not in str(item.get("backend", "")).lower()
        ):
            raise TransportDiagnosticError(f"unsupported X event sync evidence: {item}")
    return {
        "lowered_operations": [str(primitive["kind"])],
        "backend_primitives": click_primitives,
        "x_event_sync_evidence": sync,
        "real_cursor_before": list(state["cursor_before"]),
        "real_cursor_after": list(state["cursor_after"]),
        "final_pointer_button_mask": int(state["pointer_button_mask"]),
    }


def _browser_contract(
    acknowledgement: dict[str, Any], endpoint: tuple[int, int]
) -> dict[str, Any]:
    events = acknowledgement.get("events")
    if not isinstance(events, list):
        raise TransportDiagnosticError("browser acknowledgement lacks events")
    causal: list[tuple[str, dict[str, Any]]] = []
    for event in events:
        if not isinstance(event, dict):
            raise TransportDiagnosticError("browser event is not an object")
        label = ""
        if event.get("kind") == "pointer" and event.get("event") in {
            "pointerdown",
            "pointerup",
        }:
            label = str(event["event"])
        elif event.get("kind") == "click":
            label = "click"
        if label:
            causal.append((label, event))
    labels = [label for label, _ in causal]
    if labels != list(BROWSER_SEQUENCE):
        raise TransportDiagnosticError(f"browser causal sequence mismatch: {labels}")
    sequences = [int(event.get("client_sequence", -1)) for _, event in causal]
    if any(value <= 0 for value in sequences) or sequences != sorted(set(sequences)):
        raise TransportDiagnosticError(
            f"browser client sequences are not strictly causal: {sequences}"
        )
    down = causal[0][1]
    up = causal[1][1]
    for label, event, buttons in (("pointerdown", down, 1), ("pointerup", up, 0)):
        if (
            int(event.get("button", -1)) != 0
            or int(event.get("buttons", -1)) != buttons
            or event.get("hit_id") != "target"
            or [int(event.get("screen_x", -1)), int(event.get("screen_y", -1))]
            != [endpoint[0], endpoint[1]]
        ):
            raise TransportDiagnosticError(f"{label} hit contract mismatch: {event}")
    if acknowledgement.get("pointer_buttons") != 0:
        raise TransportDiagnosticError("browser acknowledgement ended with a held button")
    return {
        "sequence": labels,
        "client_sequences": sequences,
        "pointer_hits": [
            {
                "event": label,
                "screen": [int(event["screen_x"]), int(event["screen_y"])],
                "buttons": int(event["buttons"]),
                "hit_id": str(event["hit_id"]),
            }
            for label, event in causal[:2]
        ],
        "state_event": "change/click",
        "final_pointer_buttons": 0,
    }


def _run_trial(
    *,
    session: KvmFixtureSession,
    server: FixtureHttpServer,
    fixture: Fixture,
    pair_index: int,
    trial_index: int,
    arm: Arm,
    pair_id: str,
    trial_id: str,
    global_pair_index: int,
    checkpoint_attempt: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    timings: dict[str, Any] = {
        "clock": "time.monotonic_ns",
        "trial_started_host_monotonic_ns": time.monotonic_ns(),
        "trial_started_host_wall_time_ns": time.time_ns(),
    }
    reset_started = time.monotonic_ns()
    transport = session.reset_to_ready()
    reset_completed = time.monotonic_ns()
    timings.update(
        {
            "reset_started_host_monotonic_ns": reset_started,
            "reset_completed_host_monotonic_ns": reset_completed,
            "reset_duration_ns": reset_completed - reset_started,
        }
    )
    fixture_launch_started = time.monotonic_ns()
    initial = session.launch_fixture(server, fixture)
    fixture_launch_completed = time.monotonic_ns()
    timings.update(
        {
            "fixture_launch_started_host_monotonic_ns": fixture_launch_started,
            "fixture_launch_completed_host_monotonic_ns": fixture_launch_completed,
            "fixture_launch_duration_ns": (
                fixture_launch_completed - fixture_launch_started
            ),
        }
    )
    _validate_loaded_geometry(fixture, initial, transport.screen_size())
    if initial.get("current") != {"checked": False, "decoy_checked": False}:
        raise TransportDiagnosticError(f"trial {trial_index} reset state was not clean")
    baseline = transport.cursor_position()
    trajectory = build_trajectory(
        fixture, initial, arm=arm, cursor=baseline, near_miss=False
    )
    if len(trajectory.actions) != 1 or trajectory.expected_endpoint is None:
        raise TransportDiagnosticError("fixed click trajectory contract drifted")
    after_sequence = int(server.store.snapshot(fixture.id)["last_client_sequence"])
    dispatch_started = time.monotonic_ns()
    timings["dispatch_started_host_wall_time_ns"] = time.time_ns()
    dispatch, journal = _execute(arm, transport, trajectory)
    dispatch_completed = time.monotonic_ns()
    timings.update(
        {
            "dispatch_started_host_monotonic_ns": dispatch_started,
            "dispatch_completed_host_monotonic_ns": dispatch_completed,
            "dispatch_completed_host_wall_time_ns": time.time_ns(),
            "dispatch_duration_ns": dispatch_completed - dispatch_started,
        }
    )
    failed_atomic_states = [
        state
        for state in journal.get("atomic_action_states", [])
        if isinstance(state, dict) and state.get("ok") is not True
    ]
    if failed_atomic_states:
        raise TransportDiagnosticError(
            "atomic guest action failed during the transport diagnostic",
            evidence={"dispatch": dispatch, "journal": journal},
        )
    _assert_dispatch_journal(fixture, arm, "transport diagnostic", journal)
    endpoint = trajectory.expected_endpoint
    semantic = _semantic_contract(dispatch, endpoint)
    atomic = _atomic_contract(journal)
    atomic_states = journal.get("atomic_action_states", [])
    atomic_result = atomic_states[0] if len(atomic_states) == 1 else None
    attempt = {
        "schema_version": 1,
        "evidence_schema": ATTEMPT_EVIDENCE_SCHEMA,
        "status": "attempted",
        "stage": "before_browser_acknowledgement",
        "trial": {
            "pair_index": pair_index,
            "global_pair_index": global_pair_index,
            "pair_id": pair_id,
            "trial_index": trial_index,
            "trial_id": trial_id,
            "arm": arm,
            "fixture_id": fixture.id,
            "fixture_sha256": fixture.fixture_sha256,
        },
        "progress": {
            "dispatch_count": 1,
            "retry_count": 0,
            "atomic_guest_process_count": (
                atomic_result.get("guest_process_count")
                if isinstance(atomic_result, dict)
                else None
            ),
            "browser_acknowledged": False,
        },
        "browser_acknowledgement_request": {
            "after_client_sequence": after_sequence,
            "required_kinds": ["click"],
            "require_pointer_down": True,
            "require_pointer_up": True,
            "expected_pointer_buttons": 0,
        },
        "baseline": list(baseline),
        "endpoint": list(endpoint),
        "dispatch": dispatch,
        "journal": journal,
        "atomic_result": atomic_result,
        "semantic_contract": semantic,
        "atomic_contract": atomic,
        "timings": timings,
    }
    checkpoint_started = time.monotonic_ns()
    timings["pre_ack_checkpoint_started_host_monotonic_ns"] = checkpoint_started
    if checkpoint_attempt is not None:
        checkpoint_attempt(attempt)
    checkpoint_completed = time.monotonic_ns()
    timings.update(
        {
            "pre_ack_checkpoint_completed_host_monotonic_ns": checkpoint_completed,
            "pre_ack_checkpoint_duration_ns": checkpoint_completed - checkpoint_started,
        }
    )
    wait_started = time.monotonic_ns()
    timings["browser_ack_wait_started_host_monotonic_ns"] = wait_started
    timings["browser_ack_wait_started_host_wall_time_ns"] = time.time_ns()
    try:
        acknowledgement = server.store.wait_for_browser_quiescence(
            fixture.id,
            after_sequence=after_sequence,
            required_kinds=("click",),
            require_pointer_down=True,
            require_pointer_up=True,
            expected_pointer_buttons=0,
        )
    except TimeoutError as exc:
        wait_completed = time.monotonic_ns()
        timings.update(
            {
                "browser_ack_wait_completed_host_monotonic_ns": wait_completed,
                "browser_ack_wait_completed_host_wall_time_ns": time.time_ns(),
                "browser_ack_wait_duration_ns": wait_completed - wait_started,
            }
        )
        immediate_host_snapshot = _capture_timeout_component(
            "immediate_host_oracle_snapshot",
            lambda: server.store.snapshot(fixture.id),
            timings,
        )
        immediate_live_browser = _capture_timeout_component(
            "immediate_live_browser",
            lambda: session.capture_browser_diagnostics(fixture),
            timings,
        )
        immediate_live_guest_pointer = _capture_timeout_component(
            "immediate_live_guest_pointer",
            session.capture_guest_pointer_state,
            timings,
        )
        grace_started = time.monotonic_ns()
        timings["observation_grace_started_host_monotonic_ns"] = grace_started
        timings["observation_grace_started_host_wall_time_ns"] = time.time_ns()
        time.sleep(POST_TIMEOUT_OBSERVATION_GRACE_S)
        grace_completed = time.monotonic_ns()
        timings.update(
            {
                "observation_grace_completed_host_monotonic_ns": grace_completed,
                "observation_grace_completed_host_wall_time_ns": time.time_ns(),
                "observation_grace_duration_ns": grace_completed - grace_started,
                "observation_grace_requested_s": POST_TIMEOUT_OBSERVATION_GRACE_S,
            }
        )
        host_snapshot = _capture_timeout_component(
            "post_grace_host_oracle_snapshot",
            lambda: server.store.snapshot(fixture.id),
            timings,
        )
        live_browser = _capture_timeout_component(
            "post_grace_live_browser",
            lambda: session.capture_browser_diagnostics(fixture),
            timings,
        )
        live_guest_pointer = _capture_timeout_component(
            "post_grace_live_guest_pointer",
            session.capture_guest_pointer_state,
            timings,
        )
        chrome_log = _capture_timeout_component(
            "chrome_log",
            session.capture_chrome_log,
            timings,
        )
        immediate_browser_page = _browser_page_from_capture(immediate_live_browser)
        browser_page = _browser_page_from_capture(live_browser)
        page_diagnostics = (
            browser_page.get("diagnostics", {})
            if isinstance(browser_page, dict)
            and isinstance(browser_page.get("diagnostics"), dict)
            else {}
        )
        attempt.update(
            {
                "status": "failed",
                "stage": "browser_acknowledgement_timeout",
                "progress": {
                    **attempt["progress"],
                    "browser_acknowledged": False,
                    "failure_kind": "infrastructure",
                },
                "timeout": {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                "observation_grace": {
                    "requested_s": POST_TIMEOUT_OBSERVATION_GRACE_S,
                    "no_input_dispatched": True,
                    "immediate": {
                        "host_oracle_snapshot": immediate_host_snapshot,
                        "live_browser": immediate_live_browser,
                        "live_guest_pointer": immediate_live_guest_pointer,
                    },
                    "post_grace": {
                        "host_oracle_snapshot": host_snapshot,
                        "live_browser": live_browser,
                        "live_guest_pointer": live_guest_pointer,
                    },
                },
                "host_oracle_snapshot": host_snapshot,
                "live_browser": live_browser,
                "browser_page_event_log": page_diagnostics.get("page_events"),
                "report_queue": page_diagnostics.get("report_queue"),
                "dom_state": (
                    browser_page.get("dom")
                    if isinstance(browser_page, dict)
                    else None
                ),
                "live_guest_pointer": live_guest_pointer,
                "chrome_log": chrome_log,
                "cross_clock_calibration": {
                    "host": {
                        key: value
                        for key, value in timings.items()
                        if "host_" in key
                    },
                    "browser_immediate": _browser_clock_sample(
                        immediate_browser_page
                    ),
                    "browser_post_grace": _browser_clock_sample(browser_page),
                    "guest_immediate": _guest_clock_sample(
                        immediate_live_guest_pointer
                    ),
                    "guest_post_grace": _guest_clock_sample(live_guest_pointer),
                    "limitations": [
                        "X server event time is unavailable without a passive "
                        "XRecord/XI2 observer"
                    ],
                },
                "instrumentation_limitations": [
                    "no passive guest XRecord/XI2 observer was installed by this "
                    "semantics-preserving core patch",
                    "the atomic journal proves per-event X sync but does not contain "
                    "timestamped XQueryPointer samples after each button event",
                ],
            }
        )
        attempt["outcome_classifier"] = _classify_timeout_outcome(attempt)
        timings["timeout_evidence_completed_host_monotonic_ns"] = time.monotonic_ns()
        timings["trial_completed_host_monotonic_ns"] = timings[
            "timeout_evidence_completed_host_monotonic_ns"
        ]
        timings["trial_completed_host_wall_time_ns"] = time.time_ns()
        timings["trial_duration_ns"] = (
            timings["trial_completed_host_monotonic_ns"]
            - timings["trial_started_host_monotonic_ns"]
        )
        if checkpoint_attempt is not None:
            checkpoint_attempt(attempt)
        evidence = {
            "schema_version": 1,
            "evidence_schema": ATTEMPT_EVIDENCE_SCHEMA,
            "status": "failed",
            "failure_kind": "infrastructure",
            "failure_stage": "browser_acknowledgement_timeout",
            "attempted_trial": attempt,
        }
        raise TransportDiagnosticError(
            f"trial {trial_id} browser acknowledgement timed out",
            evidence=evidence,
        ) from exc
    wait_completed = time.monotonic_ns()
    timings.update(
        {
            "browser_ack_wait_completed_host_monotonic_ns": wait_completed,
            "browser_ack_wait_completed_host_wall_time_ns": time.time_ns(),
            "browser_ack_wait_duration_ns": wait_completed - wait_started,
        }
    )
    browser = _browser_contract(acknowledgement, endpoint)
    final_state = server.store.snapshot(fixture.id)
    if final_state.get("current") != {"checked": True, "decoy_checked": False}:
        raise TransportDiagnosticError(
            f"click state acknowledgement mismatch: {final_state.get('current')}"
        )
    timings["trial_completed_host_monotonic_ns"] = time.monotonic_ns()
    timings["trial_completed_host_wall_time_ns"] = time.time_ns()
    timings["trial_duration_ns"] = (
        timings["trial_completed_host_monotonic_ns"]
        - timings["trial_started_host_monotonic_ns"]
    )
    attempt.update(
        {
            "status": "passed",
            "stage": "browser_acknowledged",
            "progress": {
                **attempt["progress"],
                "browser_acknowledged": True,
                "failure_kind": None,
            },
            "browser_acknowledgement": acknowledgement,
            "browser_contract": browser,
            "final_state": final_state["current"],
        }
    )
    if checkpoint_attempt is not None:
        checkpoint_attempt(attempt)
    return {
        "pair_index": pair_index,
        "global_pair_index": global_pair_index,
        "pair_id": pair_id,
        "trial_index": trial_index,
        "trial_id": trial_id,
        "arm": arm,
        "status": "passed",
        "reset_snapshot": READY_SNAPSHOT,
        "reset_before_trial": True,
        "fixture_id": fixture.id,
        "fixture_sha256": fixture.fixture_sha256,
        "dispatch_count": 1,
        "retry_count": 0,
        "oracle_invocation_count": 0,
        "oracle_conditioned_dispatch": False,
        "baseline": list(baseline),
        "endpoint": list(endpoint),
        "semantic_operations": semantic,
        **atomic,
        "browser_contract": browser,
        "final_state": final_state["current"],
        "attempt_evidence": attempt,
    }


def _captured_value(capture: Any) -> dict[str, Any]:
    if (
        isinstance(capture, dict)
        and capture.get("status") == "captured"
        and isinstance(capture.get("value"), dict)
    ):
        return capture["value"]
    return {}


def _browser_page_from_capture(capture: Any) -> dict[str, Any]:
    value = _captured_value(capture)
    page = value.get("page")
    return page if isinstance(page, dict) else {}


def _browser_clock_sample(page: dict[str, Any]) -> dict[str, Any]:
    return {
        key: page.get(key)
        for key in (
            "captured_browser_wall_time_ms",
            "performance_time_origin_ms",
            "performance_now_ms",
            "captured_client_monotonic_ms",
        )
    }


def _guest_clock_sample(capture: Any) -> dict[str, Any]:
    value = _captured_value(capture)
    return {
        key: value.get(key)
        for key in (
            "guest_wall_before_ns",
            "guest_wall_after_ns",
            "guest_monotonic_before_ns",
            "guest_monotonic_after_ns",
        )
    }


def _classify_timeout_outcome(attempt: dict[str, Any]) -> dict[str, Any]:
    browser_page = _browser_page_from_capture(attempt.get("live_browser"))
    diagnostics = browser_page.get("diagnostics", {})
    page_events = (
        diagnostics.get("page_events", []) if isinstance(diagnostics, dict) else []
    )
    page_events = page_events if isinstance(page_events, list) else []
    pointer_up_sequences = {
        int(event["client_sequence"])
        for event in page_events
        if isinstance(event, dict)
        and event.get("kind") == "pointer"
        and event.get("event") == "pointerup"
        and isinstance(event.get("client_sequence"), int)
    }
    click_sequences = {
        int(event["client_sequence"])
        for event in page_events
        if isinstance(event, dict)
        and event.get("kind") == "click"
        and isinstance(event.get("client_sequence"), int)
    }
    relevant_sequences = pointer_up_sequences | click_sequences
    page_has_pointer_up = bool(pointer_up_sequences)
    page_has_click = bool(click_sequences)
    report_queue = (
        diagnostics.get("report_queue", {}) if isinstance(diagnostics, dict) else {}
    )
    report_queue = report_queue if isinstance(report_queue, dict) else {}
    records = report_queue.get("records", [])
    records = records if isinstance(records, list) else []
    record_states = {
        int(record["client_sequence"]): record.get("state")
        for record in records
        if isinstance(record, dict) and isinstance(record.get("client_sequence"), int)
    }
    unresolved_relevant_sequences = sorted(
        sequence
        for sequence in relevant_sequences
        if record_states.get(sequence) != "resolved"
    )

    host_snapshot = _captured_value(attempt.get("host_oracle_snapshot"))
    raw_journal = host_snapshot.get("diagnostic_journal", [])
    journal = (
        [
            item
            for item in raw_journal
            if isinstance(item, dict)
            and isinstance(item.get("details"), dict)
            and isinstance(item.get("host_monotonic_ns"), int)
            and not isinstance(item.get("host_monotonic_ns"), bool)
        ]
        if isinstance(raw_journal, list)
        else []
    )

    def journal_order(item: dict[str, Any]) -> tuple[int, int]:
        journal_sequence = item.get("journal_sequence")
        return (
            int(item["host_monotonic_ns"]),
            int(journal_sequence)
            if isinstance(journal_sequence, int)
            and not isinstance(journal_sequence, bool)
            else 0,
        )

    journal.sort(key=journal_order)
    timeout_decisions = [
        item
        for item in journal
        if item.get("stage") == "waiter_decision"
        and item["details"].get("decision") == "timeout"
    ]
    timeout_decision = timeout_decisions[-1] if timeout_decisions else None
    decision_ns = (
        int(timeout_decision["host_monotonic_ns"])
        if timeout_decision is not None
        else None
    )
    deadline_value = (
        timeout_decision["details"].get("deadline_host_monotonic_ns")
        if timeout_decision is not None
        else None
    )
    deadline_ns = (
        int(deadline_value)
        if isinstance(deadline_value, int) and not isinstance(deadline_value, bool)
        else None
    )
    cutoff_ns = (
        min(decision_ns, deadline_ns)
        if decision_ns is not None and deadline_ns is not None
        else None
    )
    waiter_starts = [
        item
        for item in journal
        if item.get("stage") == "waiter_started"
        and (decision_ns is None or int(item["host_monotonic_ns"]) <= decision_ns)
    ]
    waiter_start = waiter_starts[-1] if waiter_starts else None
    waiter_start_deadline = (
        waiter_start["details"].get("deadline_host_monotonic_ns")
        if waiter_start is not None
        else None
    )
    waiter_scope_exact = (
        waiter_start is not None
        and timeout_decision is not None
        and deadline_ns is not None
        and isinstance(waiter_start_deadline, int)
        and not isinstance(waiter_start_deadline, bool)
        and waiter_start_deadline == deadline_ns
        and journal_order(waiter_start) < journal_order(timeout_decision)
        and all(
            waiter_start["details"].get(field)
            == timeout_decision["details"].get(field)
            for field in (
                "after_sequence",
                "required_kinds",
                "require_pointer_up",
                "require_pointer_down",
                "expected_pointer_buttons",
                "quiet_s",
            )
        )
    )
    requirements: dict[str, Any] = {}
    if waiter_start is not None:
        requirements.update(waiter_start["details"])
    if timeout_decision is not None:
        requirements.update(timeout_decision["details"])
    after_sequence = requirements.get("after_sequence")
    after_sequence = (
        int(after_sequence)
        if isinstance(after_sequence, int) and not isinstance(after_sequence, bool)
        else -1
    )
    required_kinds = requirements.get("required_kinds", [])
    required_kinds = (
        {str(kind) for kind in required_kinds}
        if isinstance(required_kinds, list)
        else set()
    )
    require_pointer_down = requirements.get("require_pointer_down") is True
    require_pointer_up = requirements.get("require_pointer_up") is True
    expected_buttons = requirements.get("expected_pointer_buttons")
    expected_buttons = (
        int(expected_buttons)
        if isinstance(expected_buttons, int) and not isinstance(expected_buttons, bool)
        else None
    )
    quiet_s = requirements.get("quiet_s")
    quiet_s = (
        float(quiet_s)
        if isinstance(quiet_s, (int, float)) and not isinstance(quiet_s, bool)
        else None
    )

    request_stages = {
        "http_ingress",
        "http_body_received",
        "store_apply_started",
        "store_apply_committed",
        "store_apply_rejected",
        "http_response_started",
        "http_response_completed",
    }
    by_request: dict[Any, list[dict[str, Any]]] = {}
    for item in journal:
        request_id = item["details"].get("host_request_id")
        if (
            item.get("stage") in request_stages
            and isinstance(request_id, int)
            and not isinstance(request_id, bool)
        ):
            by_request.setdefault(request_id, []).append(item)

    correlations: list[dict[str, Any]] = []
    correlated_commits: list[dict[str, Any]] = []
    correlated_rejections: list[dict[str, Any]] = []
    ingress_sequences: set[int] = set()
    for request_id, entries in by_request.items():
        entries.sort(key=journal_order)
        phases: dict[str, list[dict[str, Any]]] = {}
        for item in entries:
            phases.setdefault(str(item.get("stage")), []).append(item)
        sequence_values = {
            int(value)
            for item in entries
            if isinstance((value := item["details"].get("client_sequence")), int)
            and not isinstance(value, bool)
        }
        sequence = next(iter(sequence_values)) if len(sequence_values) == 1 else None
        body = phases.get("http_body_received", [None])[0]
        apply_started = phases.get("store_apply_started", [None])[0]
        ingress = phases.get("http_ingress", [None])[0]
        commit = phases.get("store_apply_committed", [None])[0]
        rejection = phases.get("store_apply_rejected", [None])[0]
        response_started = phases.get("http_response_started", [None])[0]
        response_completed = phases.get("http_response_completed", [None])[0]
        terminal = commit if commit is not None else rejection
        ordered = (
            ingress is not None
            and body is not None
            and apply_started is not None
            and terminal is not None
            and response_started is not None
            and response_completed is not None
            and journal_order(ingress)
            < journal_order(body)
            < journal_order(apply_started)
            < journal_order(terminal)
            < journal_order(response_started)
            < journal_order(response_completed)
        )
        payload = body["details"].get("payload") if body is not None else None
        payload = payload if isinstance(payload, dict) else {}
        event_kind = payload.get("kind")
        event_name = payload.get("event")
        event_buttons = payload.get("buttons")
        expected_status = 200 if commit is not None else 400
        correlated_sequence_stages = (
            body,
            apply_started,
            terminal,
            response_started,
            response_completed,
        )
        sequence_matches = sequence is not None and all(
            item is not None and item["details"].get("client_sequence") == sequence
            for item in correlated_sequence_stages
        )
        request_payload_matches = (
            sequence is not None
            and payload.get("client_sequence") == sequence
            and body is not None
            and body["details"].get("kind") == event_kind
            and apply_started is not None
            and apply_started["details"].get("kind") == event_kind
        )
        terminal_matches = terminal is not None and (
            rejection is not None
            or (
                all(
                    field in terminal["details"]
                    for field in ("kind", "event", "buttons")
                )
                and terminal["details"].get("kind") == event_kind
                and terminal["details"].get("event") == event_name
                and terminal["details"].get("buttons") == event_buttons
            )
        )
        response_matches = (
            response_started is not None
            and response_completed is not None
            and response_started["details"].get("status") == expected_status
            and response_completed["details"].get("status") == expected_status
        )
        exact = (
            sequence is not None
            and ordered
            and sequence_matches
            and request_payload_matches
            and len(phases.get("http_ingress", [])) == 1
            and len(phases.get("http_body_received", [])) == 1
            and len(phases.get("store_apply_started", [])) == 1
            and len(phases.get("store_apply_committed", []))
            + len(phases.get("store_apply_rejected", []))
            == 1
            and len(phases.get("http_response_started", [])) == 1
            and len(phases.get("http_response_completed", [])) == 1
            and terminal_matches
            and response_matches
        )
        if (
            sequence is not None
            and ingress is not None
            and body is not None
            and int(ingress["host_monotonic_ns"])
            <= int(body["host_monotonic_ns"])
        ):
            ingress_sequences.add(sequence)
        correlation = {
            "host_request_id": request_id,
            "client_sequence": sequence,
            "phases": [str(item.get("stage")) for item in entries],
            "exact": exact,
            "kind": event_kind,
            "event": event_name,
            "buttons": event_buttons,
            "terminal_stage": terminal.get("stage") if terminal is not None else None,
            "terminal_host_monotonic_ns": (
                int(terminal["host_monotonic_ns"])
                if terminal is not None
                else None
            ),
        }
        correlations.append(correlation)
        if exact and sequence > after_sequence and commit is not None:
            correlated_commits.append({**correlation, "item": commit})
        if exact and sequence > after_sequence and rejection is not None:
            correlated_rejections.append({**correlation, "item": rejection})

    def label(item: dict[str, Any]) -> str:
        if item.get("kind") == "pointer":
            return str(item.get("event"))
        return str(item.get("kind"))

    required_labels = set(required_kinds)
    if require_pointer_down:
        required_labels.add("pointerdown")
    if require_pointer_up:
        required_labels.add("pointerup")
    commits_before_cutoff = [
        item
        for item in correlated_commits
        if cutoff_ns is not None
        and int(item["terminal_host_monotonic_ns"]) < cutoff_ns
    ]
    commits_before_cutoff.sort(key=lambda item: int(item["terminal_host_monotonic_ns"]))
    committed_labels = {label(item) for item in commits_before_cutoff}
    final_commit = commits_before_cutoff[-1] if commits_before_cutoff else None
    final_buttons = (
        final_commit["item"]["details"].get("last_pointer_buttons")
        if final_commit is not None
        else None
    )
    full_predicate_committed = (
        waiter_scope_exact
        and cutoff_ns is not None
        and required_labels <= committed_labels
        and expected_buttons is not None
        and final_buttons == expected_buttons
    )
    exact_rejections_before_cutoff = [
        item
        for item in correlated_rejections
        if waiter_scope_exact
        and cutoff_ns is not None
        and int(item["terminal_host_monotonic_ns"]) < cutoff_ns
        and label(item) in required_labels
    ]
    late_terminal_sequences = sorted(
        int(item["client_sequence"])
        for item in (*correlated_commits, *correlated_rejections)
        if cutoff_ns is not None
        and int(item["terminal_host_monotonic_ns"]) >= cutoff_ns
    )

    acknowledged_observations = [
        item
        for item in journal
        if item.get("stage") == "waiter_observation"
        and item["details"].get("acknowledged") is True
        and cutoff_ns is not None
        and int(item["host_monotonic_ns"]) < cutoff_ns
    ]
    acknowledged_observation = (
        acknowledged_observations[-1] if acknowledged_observations else None
    )
    observation_ns = (
        int(acknowledged_observation["host_monotonic_ns"])
        if acknowledged_observation is not None
        else None
    )
    quiet_window_started_value = (
        acknowledged_observation["details"].get(
            "quiet_window_started_host_monotonic_ns"
        )
        if acknowledged_observation is not None
        else None
    )
    observation_deadline_value = (
        acknowledged_observation["details"].get("deadline_host_monotonic_ns")
        if acknowledged_observation is not None
        else None
    )
    observation_quiet_s_value = (
        acknowledged_observation["details"].get("quiet_s")
        if acknowledged_observation is not None
        else None
    )
    waiter_start_ns = (
        int(waiter_start["host_monotonic_ns"])
        if waiter_start is not None
        else None
    )
    quiet_window_started_ns = (
        int(quiet_window_started_value)
        if isinstance(quiet_window_started_value, int)
        and not isinstance(quiet_window_started_value, bool)
        and observation_ns is not None
        and waiter_start_ns is not None
        and waiter_start_ns <= quiet_window_started_value
        and quiet_window_started_value <= observation_ns
        and isinstance(observation_deadline_value, int)
        and not isinstance(observation_deadline_value, bool)
        and observation_deadline_value == deadline_ns
        and isinstance(observation_quiet_s_value, (int, float))
        and not isinstance(observation_quiet_s_value, bool)
        and quiet_s is not None
        and float(observation_quiet_s_value) == quiet_s
        else None
    )
    observation_sequence = (
        acknowledged_observation["details"].get("last_client_sequence")
        if acknowledged_observation is not None
        else None
    )
    observation_predicate = (
        acknowledged_observation is not None
        and (
            not require_pointer_down
            or acknowledged_observation["details"].get("pointer_down_observed") is True
        )
        and (
            not require_pointer_up
            or acknowledged_observation["details"].get("pointer_up_observed") is True
        )
        and required_kinds
        <= set(acknowledged_observation["details"].get("observed_kinds", []))
        and acknowledged_observation["details"].get("pointer_buttons")
        == expected_buttons
    )
    quiet_ready_ns = (
        quiet_window_started_ns + int(quiet_s * 1_000_000_000)
        if quiet_window_started_ns is not None
        and quiet_s is not None
        and quiet_s >= 0
        else None
    )
    quiet_completed_strictly_before_timeout = (
        quiet_ready_ns is not None
        and cutoff_ns is not None
        and quiet_ready_ns < cutoff_ns
    )
    required_commit_ns = max(
        (int(item["terminal_host_monotonic_ns"]) for item in commits_before_cutoff),
        default=None,
    )
    observation_after_commits = (
        quiet_window_started_ns is not None
        and required_commit_ns is not None
        and required_commit_ns <= quiet_window_started_ns
    )
    later_sequence_change = False
    if quiet_window_started_ns is not None and cutoff_ns is not None:
        for item in journal:
            item_ns = int(item["host_monotonic_ns"])
            if not quiet_window_started_ns < item_ns < cutoff_ns:
                continue
            if item.get("stage") == "store_apply_committed" and item["details"].get(
                "last_client_sequence"
            ) != observation_sequence:
                later_sequence_change = True
            if item.get("stage") == "waiter_observation" and (
                item["details"].get("last_client_sequence") != observation_sequence
                or item["details"].get("acknowledged") is not True
            ):
                later_sequence_change = True
    waiter_miss_proven = (
        full_predicate_committed
        and observation_predicate
        and observation_after_commits
        and quiet_completed_strictly_before_timeout
        and not later_sequence_change
    )
    proven_waiter_missed_sequences = (
        sorted(int(item["client_sequence"]) for item in commits_before_cutoff)
        if waiter_miss_proven
        else []
    )
    relevant_absent_from_host = sorted(relevant_sequences - ingress_sequences)

    x_observer = attempt.get("passive_x_observer")
    x_observer_available = isinstance(x_observer, dict) and x_observer.get(
        "status"
    ) == "captured"
    x_events = x_observer.get("events", []) if x_observer_available else []
    x_has_release = any(
        isinstance(event, dict) and event.get("event") == "button_release"
        for event in x_events
    )
    pointer = _captured_value(attempt.get("live_guest_pointer"))
    pointer_mask = pointer.get("pointer_button_mask")

    classification = "inconclusive"
    if x_observer_available and not x_has_release and pointer_mask not in {None, 0}:
        classification = "guest_input_path"
    elif x_observer_available and x_has_release and pointer_mask == 0 and not page_has_pointer_up:
        classification = "chromium_input_delivery"
    elif exact_rejections_before_cutoff or waiter_miss_proven:
        classification = "host_harness"
    elif page_has_pointer_up and page_has_click and (
        unresolved_relevant_sequences or relevant_absent_from_host
    ):
        classification = "browser_reporter"
    limitations = []
    if not x_observer_available:
        limitations.append(
            "passive XRecord/XI2 event trace unavailable; guest-vs-Chromium rules cannot fire"
        )
    return {
        "schema_version": 1,
        "classification": classification,
        "rules": TIMEOUT_CLASSIFIER_RULES,
        "observed": {
            "page_has_pointer_up": page_has_pointer_up,
            "page_has_click": page_has_click,
            "pointer_up_client_sequences": sorted(pointer_up_sequences),
            "click_client_sequences": sorted(click_sequences),
            "reporter_record_states": {
                str(sequence): record_states.get(sequence)
                for sequence in sorted(relevant_sequences)
            },
            "reporter_unresolved_relevant_sequences": (
                unresolved_relevant_sequences
            ),
            "relevant_sequences_absent_from_host": relevant_absent_from_host,
            "host_request_correlations": correlations,
            "waiter_timeout_decision_host_monotonic_ns": decision_ns,
            "waiter_deadline_host_monotonic_ns": deadline_ns,
            "waiter_timeout_cutoff_host_monotonic_ns": cutoff_ns,
            "waiter_scope_exact": waiter_scope_exact,
            "required_host_labels": sorted(required_labels),
            "committed_host_labels_before_timeout": sorted(committed_labels),
            "full_required_predicate_committed": full_predicate_committed,
            "exact_rejected_sequences_before_timeout": sorted(
                int(item["client_sequence"])
                for item in exact_rejections_before_cutoff
            ),
            "host_ingress_missing_store_sequences": sorted(
                int(item["client_sequence"])
                for item in exact_rejections_before_cutoff
            ),
            "host_waiter_missed_sequences": proven_waiter_missed_sequences,
            "late_terminal_sequences": late_terminal_sequences,
            "acknowledged_observation_host_monotonic_ns": observation_ns,
            "acknowledged_observation_last_client_sequence": observation_sequence,
            "quiet_window_started_host_monotonic_ns": quiet_window_started_ns,
            "quiet_ready_host_monotonic_ns": quiet_ready_ns,
            "quiet_completed_strictly_before_timeout": (
                quiet_completed_strictly_before_timeout
            ),
            "later_sequence_change_before_timeout": later_sequence_change,
            "waiter_miss_proven": waiter_miss_proven,
            "post_grace_pointer_button_mask": pointer_mask,
            "passive_x_observer_available": x_observer_available,
            "passive_x_release_observed": x_has_release,
        },
        "limitations": limitations,
    }


def _capture_timeout_component(
    name: str,
    capture: Callable[[], Any],
    timings: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic_ns()
    started_wall = time.time_ns()
    timings[f"{name}_capture_started_host_monotonic_ns"] = started
    timings[f"{name}_capture_started_host_wall_time_ns"] = started_wall
    try:
        value = capture()
        return {
            "schema_version": 1,
            "status": "captured",
            "value": value,
        }
    except Exception as exc:
        return {
            "schema_version": 1,
            "status": "capture_error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        completed = time.monotonic_ns()
        completed_wall = time.time_ns()
        timings[f"{name}_capture_completed_host_monotonic_ns"] = completed
        timings[f"{name}_capture_completed_host_wall_time_ns"] = completed_wall
        timings[f"{name}_capture_duration_ns"] = completed - started


def _matching_contract(trial: dict[str, Any]) -> dict[str, Any]:
    return {
        "endpoint": trial["endpoint"],
        "semantic_operations": trial["semantic_operations"],
        "lowered_operations": trial["lowered_operations"],
        "backend_primitives": trial["backend_primitives"],
        "x_event_sync_evidence": trial["x_event_sync_evidence"],
        "real_cursor_before": trial["real_cursor_before"],
        "real_cursor_after": trial["real_cursor_after"],
        "browser_sequence": trial["browser_contract"]["sequence"],
        "pointer_hits": trial["browser_contract"]["pointer_hits"],
        "final_pointer_button_mask": trial["final_pointer_button_mask"],
        "final_state": trial["final_state"],
    }


def _checkpoint(
    output: Path,
    *,
    trials: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    active_trial: dict[str, Any] | None,
    stage: str,
    attempted_trial: dict[str, Any] | None = None,
    expected_pair_count: int = PAIR_COUNT,
    suite: str = "rung1_shared_click_transport_diagnostic",
    shard_index: int | None = None,
) -> None:
    attempted_progress = (
        attempted_trial.get("progress", {})
        if isinstance(attempted_trial, dict)
        else {}
    )
    attempted_failure_kind = (
        attempted_progress.get("failure_kind")
        if isinstance(attempted_progress, dict)
        else None
    )
    failed = attempted_trial is not None and attempted_trial.get("status") == "failed"
    _atomic_json(
        output / "transport_diagnostic_progress.json",
        {
            "status": "failed" if failed else "running",
            "suite": suite,
            "shard_index": shard_index,
            "expected_pair_count": expected_pair_count,
            "expected_trial_count": expected_pair_count * len(ARM_ORDER),
            "completed_pair_count": len(pairs),
            "completed_trial_count": len(trials),
            "stop_on_first_mismatch": True,
            "retry_count": 0,
            "infrastructure_error_count": int(
                failed and attempted_failure_kind == "infrastructure"
            ),
            "verifier_failure_count": int(
                failed and attempted_failure_kind == "verification"
            ),
            "injected_failure_count": int(
                failed and attempted_failure_kind == "injected"
            ),
            "gpu_count": 0,
            "model_access": False,
            "sealed_evaluation_access": False,
            "stage": stage,
            "active_trial": active_trial,
            "attempted_trial": attempted_trial,
            "pairs": pairs,
            "trials": trials,
        },
    )


def validate_transport_diagnostic() -> dict[str, Any]:
    spec, spec_sha256 = load_transport_diagnostic_spec()
    manifest = load_manifest()
    fixture = _select_fixture(manifest, spec)
    return {
        "schema_version": 1,
        "status": "passed",
        "suite": spec["suite"],
        "mode": "validate",
        "spec_sha256": spec_sha256,
        "manifest_payload_sha256": manifest.manifest_payload_sha256,
        "fixture_id": fixture.id,
        "fixture_sha256": fixture.fixture_sha256,
        "pair_count": PAIR_COUNT,
        "trial_count": PAIR_COUNT * len(ARM_ORDER),
        "arm_trial_counts": {arm: PAIR_COUNT for arm in ARM_ORDER},
        "retry_count": 0,
        "infrastructure_error_count": 0,
        "verifier_failure_count": 0,
        "injected_failure_count": 0,
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
    }


def validate_transport_certification(shard_index: int) -> dict[str, Any]:
    if shard_index not in range(CERTIFICATION_SHARD_COUNT):
        raise TransportDiagnosticError("certification shard index must be in [0, 3]")
    spec, spec_sha256 = load_transport_certification_spec()
    manifest = load_manifest()
    fixture = _select_fixture(manifest, spec)
    first_global_pair = shard_index * CERTIFICATION_PAIRS_PER_SHARD + 1
    return {
        "schema_version": 1,
        "status": "passed",
        "suite": spec["suite"],
        "mode": "validate",
        "spec_sha256": spec_sha256,
        "shard_index": shard_index,
        "shard_count": CERTIFICATION_SHARD_COUNT,
        "global_pair_range": [
            first_global_pair,
            first_global_pair + CERTIFICATION_PAIRS_PER_SHARD - 1,
        ],
        "manifest_payload_sha256": manifest.manifest_payload_sha256,
        "fixture_id": fixture.id,
        "fixture_sha256": fixture.fixture_sha256,
        "pair_count": CERTIFICATION_PAIRS_PER_SHARD,
        "trial_count": CERTIFICATION_PAIRS_PER_SHARD * len(ARM_ORDER),
        "arm_trial_counts": {
            arm: CERTIFICATION_PAIRS_PER_SHARD for arm in ARM_ORDER
        },
        "retry_count": 0,
        "infrastructure_error_count": 0,
        "verifier_failure_count": 0,
        "injected_failure_count": 0,
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
    }


def run_vm_transport_diagnostic(
    *,
    output: Path,
    qcow: Path,
    qemu: Path,
    provider_path: Path,
    expected_provider_sha256: str | None,
    certification_shard_index: int | None = None,
) -> dict[str, Any]:
    if certification_shard_index is None:
        spec, spec_sha256 = load_transport_diagnostic_spec()
        pair_count = PAIR_COUNT
        shard_index = None
        global_pair_offset = 0
        identity_prefix = "preflight"
    else:
        if certification_shard_index not in range(CERTIFICATION_SHARD_COUNT):
            raise TransportDiagnosticError(
                "certification shard index must be in [0, 3]"
            )
        spec, spec_sha256 = load_transport_certification_spec()
        pair_count = CERTIFICATION_PAIRS_PER_SHARD
        shard_index = certification_shard_index
        global_pair_offset = shard_index * pair_count
    manifest = load_manifest()
    fixture = _select_fixture(manifest, spec)
    provider_sha256 = sha256_file(provider_path)
    if expected_provider_sha256 and provider_sha256 != expected_provider_sha256:
        raise TransportDiagnosticError(
            f"KVM provider hash mismatch: {provider_sha256} != {expected_provider_sha256}"
        )
    trials: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    vm_log_dir = output / "vm_logs"
    _checkpoint(
        output,
        trials=trials,
        pairs=pairs,
        active_trial=None,
        stage="starting_vm",
        expected_pair_count=pair_count,
        suite=str(spec["suite"]),
        shard_index=shard_index,
    )
    with FixtureHttpServer(manifest) as server, KvmFixtureSession(
        qcow=qcow,
        qemu=qemu,
        provider_path=provider_path,
        vm_log_dir=vm_log_dir,
    ) as session:
        for pair_index in range(1, pair_count + 1):
            if shard_index is None:
                global_pair_index = global_pair_offset + pair_index
                pair_id = f"{identity_prefix}-pair-{global_pair_index:03d}"
            else:
                global_pair_index, pair_id, _ = certification_trial_identity(
                    spec_sha256, shard_index, pair_index, ARM_ORDER[0]
                )
            pair_trials: list[dict[str, Any]] = []
            for arm in ARM_ORDER:
                trial_index = len(trials) + 1
                if shard_index is None:
                    arm_slug = "native" if arm == ARM_ORDER[0] else "compact"
                    trial_id = f"{pair_id}-{arm_slug}"
                else:
                    _, _, trial_id = certification_trial_identity(
                        spec_sha256, shard_index, pair_index, arm
                    )
                active = {
                    "pair_index": pair_index,
                    "global_pair_index": global_pair_index,
                    "pair_id": pair_id,
                    "trial_index": trial_index,
                    "trial_id": trial_id,
                    "arm": arm,
                    "fixture_id": fixture.id,
                }
                _checkpoint(
                    output,
                    trials=trials,
                    pairs=pairs,
                    active_trial=active,
                    attempted_trial=None,
                    stage="resetting_trial",
                    expected_pair_count=pair_count,
                    suite=str(spec["suite"]),
                    shard_index=shard_index,
                )
                def checkpoint_attempt(attempt: dict[str, Any]) -> None:
                    _checkpoint(
                        output,
                        trials=trials,
                        pairs=pairs,
                        active_trial=active,
                        attempted_trial=attempt,
                        stage=str(attempt["stage"]),
                        expected_pair_count=pair_count,
                        suite=str(spec["suite"]),
                        shard_index=shard_index,
                    )

                trial = _run_trial(
                    session=session,
                    server=server,
                    fixture=fixture,
                    pair_index=pair_index,
                    trial_index=trial_index,
                    arm=arm,
                    pair_id=pair_id,
                    trial_id=trial_id,
                    global_pair_index=global_pair_index,
                    checkpoint_attempt=checkpoint_attempt,
                )
                pair_trials.append(trial)
                trials.append(trial)
                _checkpoint(
                    output,
                    trials=trials,
                    pairs=pairs,
                    active_trial=active,
                    attempted_trial=trial["attempt_evidence"],
                    stage="trial_passed",
                    expected_pair_count=pair_count,
                    suite=str(spec["suite"]),
                    shard_index=shard_index,
                )
            native_contract = _matching_contract(pair_trials[0])
            compact_contract = _matching_contract(pair_trials[1])
            if native_contract != compact_contract:
                raise TransportDiagnosticError(
                    f"pair {pair_index} native/compact contract mismatch: "
                    f"native={native_contract}, compact={compact_contract}"
                )
            pair = {
                "pair_index": pair_index,
                "global_pair_index": global_pair_index,
                "pair_id": pair_id,
                "status": "passed",
                "arm_order": list(ARM_ORDER),
                "trial_indices": [item["trial_index"] for item in pair_trials],
                "trial_ids": [item["trial_id"] for item in pair_trials],
                "matched_contract_sha256": _payload_sha256(native_contract),
                "matched_contract": native_contract,
            }
            pairs.append(pair)
            _checkpoint(
                output,
                trials=trials,
                pairs=pairs,
                active_trial=None,
                stage="pair_passed",
                expected_pair_count=pair_count,
                suite=str(spec["suite"]),
                shard_index=shard_index,
            )
    expected_trials = pair_count * len(ARM_ORDER)
    if len(trials) != expected_trials or len(pairs) != pair_count:
        raise TransportDiagnosticError(
            "diagnostic ended before its fixed pair/trial horizon"
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "suite": spec["suite"],
        "mode": "vm",
        "spec_sha256": spec_sha256,
        "shard_index": shard_index,
        "shard_count": (
            CERTIFICATION_SHARD_COUNT if shard_index is not None else None
        ),
        "global_pair_range": (
            [global_pair_offset + 1, global_pair_offset + pair_count]
            if shard_index is not None
            else None
        ),
        "snapshot_name": READY_SNAPSHOT,
        "manifest_payload_sha256": manifest.manifest_payload_sha256,
        "fixture_id": fixture.id,
        "fixture_sha256": fixture.fixture_sha256,
        "pair_count": len(pairs),
        "trial_count": len(trials),
        "passed_trial_count": len(trials),
        "arm_trial_counts": {
            arm: sum(trial["arm"] == arm for trial in trials)
            for arm in ARM_ORDER
        },
        "ordered_pair_ids": [pair["pair_id"] for pair in pairs],
        "ordered_trial_ids": [trial["trial_id"] for trial in trials],
        "reset_count": len(trials),
        "dispatch_count": len(trials),
        "retry_count": 0,
        "infrastructure_error_count": 0,
        "verifier_failure_count": 0,
        "injected_failure_count": 0,
        "oracle_invocation_count": 0,
        "oracle_conditioned_dispatch": False,
        "stop_on_first_mismatch": True,
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
        "provider": {
            "path": str(provider_path.resolve()),
            "sha256": provider_sha256,
        },
        "qcow": {"path": str(qcow.resolve()), "size": qcow.stat().st_size},
        "qemu": str(qemu.resolve()),
        "pairs": pairs,
        "trials": trials,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "vm"), required=True)
    parser.add_argument(
        "--suite", choices=("preflight", "certification"), default="preflight"
    )
    parser.add_argument("--shard-index", type=int)
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
    certification = args.suite == "certification"
    marker = (
        args.output / f"transport_certification_shard_{args.shard_index}.json"
        if certification
        else args.output / "transport_diagnostic.json"
    )
    marker.unlink(missing_ok=True)
    try:
        if certification and args.shard_index not in range(CERTIFICATION_SHARD_COUNT):
            raise TransportDiagnosticError(
                "certification requires --shard-index in [0, 3]"
            )
        if not certification and args.shard_index is not None:
            raise TransportDiagnosticError(
                "--shard-index is only valid for the certification suite"
            )
        payload = (
            (
                validate_transport_certification(args.shard_index)
                if certification
                else validate_transport_diagnostic()
            )
            if args.mode == "validate"
            else run_vm_transport_diagnostic(
                output=args.output,
                qcow=args.qcow,
                qemu=args.qemu,
                provider_path=args.provider,
                expected_provider_sha256=args.expected_provider_sha256,
                certification_shard_index=(
                    args.shard_index if certification else None
                ),
            )
        )
        _atomic_json(marker, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        evidence = getattr(exc, "evidence", None)
        evidence_failure_kind = (
            evidence.get("failure_kind") if isinstance(evidence, dict) else None
        )
        failure_kind = (
            evidence_failure_kind
            if evidence_failure_kind in {"verification", "infrastructure", "injected"}
            else (
                "verification"
                if isinstance(exc, TransportDiagnosticError)
                else "infrastructure"
            )
        )
        failure = {
            "schema_version": 1,
            "status": "failed",
            "mode": args.mode,
            "suite": args.suite,
            "shard_index": args.shard_index,
            "retry_count": 0,
            "infrastructure_error_count": int(failure_kind == "infrastructure"),
            "verifier_failure_count": int(failure_kind == "verification"),
            "injected_failure_count": int(failure_kind == "injected"),
            "gpu_count": 0,
            "model_access": False,
            "sealed_evaluation_access": False,
            "error_type": type(exc).__name__,
            "failure_kind": failure_kind,
            "message": str(exc),
            "evidence": evidence,
            "traceback": traceback.format_exc(),
        }
        failure_marker = (
            args.output
            / f"transport_certification_failure_shard_{args.shard_index}.json"
            if certification
            else args.output / "failure.json"
        )
        _atomic_json(failure_marker, failure)
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
