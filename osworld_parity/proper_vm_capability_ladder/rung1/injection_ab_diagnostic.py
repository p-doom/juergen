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
from .server import BROWSER_AUDIT_EVENTS, FixtureHttpServer
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
EXPECTED_SPEC_SHA256 = "36c804467519014d96310ae85608639a19ffe8b322e2529b7c90648784d0f176"
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
    "press_call_before",
    "press_call_after",
    "press_sync_completed",
    "dwell_started",
    "dwell_completed",
    "release_call_before",
    "release_call_after",
    "release_sync_completed",
)
OBSERVATION_WINDOW_S = 3.0
AUDIT_DRAIN_S = 0.75
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
        "suite": "rung1_click_release_injection_ab_v1",
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
        "audit_drain_s": AUDIT_DRAIN_S,
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
    if spec.get("backend_arms") != {
        "A": {
            "name": BACKEND_BY_ARM["A"],
            "order": [
                "pyautogui_press",
                "x_sync",
                "dwell_50ms",
                "pyautogui_release_side_same_coordinate_motion",
                "pyautogui_release",
                "x_sync",
            ],
            "release_side_motion_notify": True,
        },
        "B": {
            "name": BACKEND_BY_ARM["B"],
            "order": [
                "xtest_press",
                "x_sync",
                "dwell_50ms",
                "xtest_release",
                "x_sync",
            ],
            "release_side_motion_notify": False,
        },
    }:
        raise InjectionAbIntegrityError("backend arm contract drifted")
    audit = spec.get("browser_audit")
    if not isinstance(audit, dict) or (
        audit.get("dom_events") != list(AUDIT_DOM_EVENTS)
        or audit.get("required_fields") != list(AUDIT_REQUIRED_FIELDS)
        or audit.get("heartbeat_event") != "audit_heartbeat"
        or audit.get("heartbeat_interval_ms") != 500
        or audit.get("require_heartbeat_received_after_observation_deadline")
        is not True
        or audit.get("heartbeat_expected_sequence_fields")
        != [
            "expected_previous_audit_sequence",
            "expected_audit_count_through_marker",
        ]
        or audit.get("classification_uses_only_sequence_sealed_prefix") is not True
        or audit.get("arm_neutral") is not True
        or audit.get("serialized_post_queue_independent") is not True
        or audit.get("cdp_independent") is not True
    ):
        raise InjectionAbIntegrityError("browser audit contract drifted")
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
    snapshot: dict[str, Any], generation: int
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
        if type(event.get("schema_version")) is not int or event["schema_version"] != 1:
            raise InjectionAbIntegrityError("browser audit schema version drifted")
        if event.get("event") not in BROWSER_AUDIT_EVENTS:
            raise InjectionAbIntegrityError("browser audit event set drifted")
        expected_fields = set(AUDIT_ENVELOPE_FIELDS + AUDIT_REQUIRED_FIELDS + AUDIT_DETAIL_FIELDS)
        if event["event"] == "audit_ready":
            expected_fields.update({"page_time_origin_ms", "url"})
        elif event["event"] == "audit_heartbeat":
            expected_fields.update(
                {
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
    request_ids = [int(event["host_audit_request_id"]) for event in events]
    if sorted(request_ids) != list(range(1, len(events) + 1)):
        raise InjectionAbIntegrityError(
            "browser audit host request identities omitted or duplicated"
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
    heartbeats = [
        event
        for event in audit_events
        if event.get("event") == "audit_heartbeat"
        and isinstance(event.get("host_monotonic_ns"), int)
        and event["host_monotonic_ns"] > observation_deadline_ns
    ]
    if not heartbeats:
        raise InjectionAbIntegrityError(
            "independent browser audit has no post-window heartbeat"
        )
    marker = min(heartbeats, key=lambda event: int(event["audit_sequence"]))
    marker_sequence = marker["audit_sequence"]
    if (
        marker.get("expected_previous_audit_sequence") != marker_sequence - 1
        or marker.get("expected_audit_count_through_marker") != marker_sequence
        or marker_sequence > len(audit_events)
    ):
        raise InjectionAbIntegrityError(
            "independent browser audit marker sequence/count mismatch"
        )
    sealed = audit_events[:marker_sequence]
    if len(sealed) != marker_sequence or sealed[-1] is not marker:
        raise InjectionAbIntegrityError("independent browser audit tail was not sealed")
    return sealed, marker


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
        or click.get("dwell_ms") != 50
        or click.get("release_side_motion_notify")
        is not RELEASE_MOTION_BY_ARM[arm]
        or click.get("injection_attempt_count") != 1
        or click.get("retry_count") != 0
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
            item.get("phase") not in {"outside_click", "press", "release"}
            or not isinstance(item.get("event_type"), int)
            or isinstance(item.get("event_type"), bool)
            or not isinstance(item.get("detail"), int)
            or isinstance(item.get("detail"), bool)
            or not isinstance(item.get("started_guest_monotonic_ns"), int)
            or not isinstance(item.get("completed_guest_monotonic_ns"), int)
            or item["completed_guest_monotonic_ns"] < item["started_guest_monotonic_ns"]
            or item.get("duration_ns")
            != item["completed_guest_monotonic_ns"] - item["started_guest_monotonic_ns"]
        ):
            raise InjectionAbIntegrityError("X injection evidence schema drifted")
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
    if (
        not isinstance(start_sequence, int)
        or isinstance(start_sequence, bool)
        or start_sequence != 1
        or len(evidence) != start_sequence + len(expected_events)
        or [item["sequence"] for item in controlled]
        != list(range(start_sequence + 1, start_sequence + 1 + len(expected_events)))
    ):
        raise InjectionAbIntegrityError("controlled XTest event order drifted")
    for item, (phase, name, event_type, detail) in zip(controlled, expected_events):
        if (
            item.get("phase") != phase
            or item.get("event") != name
            or item.get("event_type") != event_type
            or item.get("detail") != detail
        ):
            raise InjectionAbIntegrityError("controlled XTest event identity drifted")
    controlled_clock_values = [
        value
        for item in controlled
        for value in (
            item["started_guest_monotonic_ns"],
            item["completed_guest_monotonic_ns"],
        )
    ]
    if controlled_clock_values != sorted(controlled_clock_values):
        raise InjectionAbIntegrityError("controlled XTest event clock order drifted")
    press_motion = controlled[0]
    press_coordinates = (press_motion.get("x"), press_motion.get("y"))
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in press_coordinates):
        raise InjectionAbIntegrityError("press MotionNotify coordinates malformed")
    if expected_endpoint is not None and press_coordinates != expected_endpoint:
        raise InjectionAbIntegrityError("press MotionNotify missed the fixed endpoint")
    outside_motion = evidence[0]
    if (
        outside_motion.get("phase") != "outside_click"
        or outside_motion.get("event") != "motion_notify"
        or outside_motion.get("event_type") != X_MOTION_NOTIFY
        or outside_motion.get("detail") != 0
        or (outside_motion.get("x"), outside_motion.get("y")) != press_coordinates
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
        timestamp["press_call_before_guest_monotonic_ns"]
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
            "suite": "rung1_click_release_injection_ab_v1",
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
            _assert_dispatch_journal(
                fixture, "compact_raw_phaseb", "injection A/B", journal
            )
            if len(dispatch) != 1 or journal["completed_action_count"] != 1:
                raise InjectionAbIntegrityError("dispatch count drifted")
            atomic = journal["atomic_action_states"][0]
            atomic_contract = validate_atomic_contract(
                atomic,
                arm=scheduled["arm"],
                expected_endpoint=trajectory.expected_endpoint,
            )
            observation_deadline_ns = dispatch_completed_ns + int(
                OBSERVATION_WINDOW_S * 1_000_000_000
            )
            remaining_ns = observation_deadline_ns - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns / 1_000_000_000)
            time.sleep(AUDIT_DRAIN_S)
            snapshot = server.store.snapshot(fixture.id)
            audit_events = validate_audit_trace(snapshot, generation)
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
                "outcome": outcome,
                "timings": {
                    "dispatch_started_host_monotonic_ns": dispatch_started_ns,
                    "dispatch_completed_host_monotonic_ns": dispatch_completed_ns,
                    "observation_deadline_host_monotonic_ns": observation_deadline_ns,
                    "audit_capture_completed_host_monotonic_ns": time.monotonic_ns(),
                    "observation_window_s": OBSERVATION_WINDOW_S,
                    "audit_drain_s": AUDIT_DRAIN_S,
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
                "suite": "rung1_click_release_injection_ab_v1",
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
