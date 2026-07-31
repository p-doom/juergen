from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from osworld_parity.proper_vm_capability_ladder.rung1.injection_ab_diagnostic import (
    AUDIT_REQUIRED_FIELDS,
    BACKEND_BY_ARM,
    ORDER_BLOCK,
    TIMESTAMP_STAGES,
    BackendBoundTransport,
    InjectionAbIntegrityError,
    classify_trial_outcome,
    fixed_trial_schedule,
    interpret_results,
    load_injection_ab_spec,
    validate_atomic_contract,
    validate_audit_trace,
    validate_injection_ab,
)
from osworld_parity.proper_vm_capability_ladder.rung1.transport import Operation


def test_preregistration_and_schedule_are_exact_and_development_only() -> None:
    spec, digest = load_injection_ab_spec()
    assert len(digest) == 64
    assert spec["parent_evidence"]["job_id"] == "136131"
    assert spec["parent_evidence"]["observed_signature"]["classification"] == "inconclusive"
    assert spec["development_only"] is True
    assert spec["qualification_authorized"] is False
    assert spec["gpu_count"] == 0
    assert spec["model_access"] is False
    assert spec["sealed_evaluation_access"] is False
    assert spec["passive_x_observer"]["enabled"] is False
    schedule = fixed_trial_schedule()
    assert len(schedule) == 48
    assert [item["arm"] for item in schedule] == list(ORDER_BLOCK) * 6
    assert [item["arm"] for item in schedule].count("A") == 24
    assert [item["arm"] for item in schedule].count("B") == 24
    assert len({item["trial_id"] for item in schedule}) == 48
    validated = validate_injection_ab()
    assert validated["status"] == "passed"
    assert validated["arm_trial_counts"] == {"A": 24, "B": 24}


def test_preregistration_rejects_schedule_drift(tmp_path: Path) -> None:
    spec, _ = load_injection_ab_spec()
    spec["order_block"] = ["A", "B"] * 4
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(InjectionAbIntegrityError, match="preregistration drifted"):
        load_injection_ab_spec(path)


def _audit_event(sequence: int, event: str, *, checked: bool = False) -> dict:
    value = {
        "schema_version": 1,
        "generation": 7,
        "audit_sequence": sequence,
        "event": event,
        "browser_wall_time_ms": 1000 + sequence,
        "client_monotonic_ms": float(sequence),
        "event_time_stamp_ms": float(sequence) if event != "audit_ready" else None,
        "is_trusted": event != "audit_ready",
        "default_prevented": False if event != "audit_ready" else None,
        "target": (
            {"id": "target", "tag": "input", "checked": checked}
            if event != "audit_ready"
            else None
        ),
        "target_checked": checked if event != "audit_ready" else None,
        "checkbox_state": {"target": checked, "decoy": False},
        "active_element": {"id": "target", "tag": "input", "checked": checked},
        "document_has_focus": True,
        "visibility_state": "visible",
        "button": 0 if event.startswith(("pointer", "mouse")) else None,
        "buttons": 0,
        "pointer_type": "mouse" if event.startswith("pointer") else None,
        "client_x": 295,
        "client_y": 231,
        "screen_x": 365,
        "screen_y": 345,
        "host_audit_request_id": sequence,
        "host_monotonic_ns": sequence,
        "host_wall_time_ns": sequence,
    }
    if event == "audit_ready":
        value["page_time_origin_ms"] = 123.5
        value["url"] = "http://10.0.2.2/fixture/r1a-click-dev-1101"
    assert all(field in value for field in AUDIT_REQUIRED_FIELDS)
    return value


def _audit_snapshot(events: list[dict]) -> dict:
    return {
        "browser_audit_events": events,
        "browser_audit_dropped": 0,
        "events": [],
        "current": {"checked": False, "decoy_checked": False},
    }


def test_audit_trace_is_independent_ordered_and_primary() -> None:
    events = [
        _audit_event(1, "audit_ready"),
        _audit_event(2, "pointerdown"),
        _audit_event(3, "mousedown"),
        _audit_event(4, "focus"),
        _audit_event(5, "pointerup"),
        _audit_event(6, "mouseup"),
        _audit_event(7, "click", checked=True),
        _audit_event(8, "input", checked=True),
        _audit_event(9, "change", checked=True),
    ]
    # HTTP request completion can reorder; browser audit_sequence is canonical.
    trace = validate_audit_trace(_audit_snapshot(list(reversed(events))), 7)
    assert [event["audit_sequence"] for event in trace] == list(range(1, 10))
    outcome = classify_trial_outcome(trace, _audit_snapshot(events))
    assert outcome["primary_success"] is True
    assert outcome["outcome"] == "trusted_click_input_change"
    assert outcome["host_reporter_event_sequence"] == []


@pytest.mark.parametrize(
    "mutation",
    ["omission", "duplicate", "missing_field", "stale", "overflow", "ready_duplicate"],
)
def test_audit_integrity_mutations_abort(mutation: str) -> None:
    events = [_audit_event(1, "audit_ready"), _audit_event(2, "pointerdown")]
    snapshot = _audit_snapshot(events)
    if mutation == "omission":
        events[1]["audit_sequence"] = 3
    elif mutation == "duplicate":
        events[1]["audit_sequence"] = 1
    elif mutation == "missing_field":
        events[1].pop("is_trusted")
    elif mutation == "stale":
        events[1]["generation"] = 6
    elif mutation == "overflow":
        snapshot["browser_audit_dropped"] = 1
    else:
        duplicate = copy.deepcopy(events[0])
        duplicate["audit_sequence"] = 2
        events[:] = [events[0], duplicate]
    with pytest.raises(InjectionAbIntegrityError):
        validate_audit_trace(snapshot, 7)


def _atomic(arm: str) -> dict:
    timestamp = {
        "click_backend": BACKEND_BY_ARM[arm],
        "backend_identity": BACKEND_BY_ARM[arm],
        "release_side_motion_notify": arm == "A",
        "clock": "time.monotonic_ns",
        "dwell_requested_ns": 50_000_000,
        "dwell_duration_ns": 50_000_000,
    }
    timestamp.update(
        {
            f"{stage}_guest_monotonic_ns": 1_000_000_000 + index
            for index, stage in enumerate(TIMESTAMP_STAGES)
        }
    )
    evidence = [
        {"phase": "press", "event": "motion_notify"},
        {"phase": "press", "event": "button_press"},
        {"phase": "release", "event": "button_release"},
    ]
    if arm == "A":
        evidence.insert(
            2,
            {
                "phase": "release",
                "event": "motion_notify",
                "x": 365,
                "y": 345,
            },
        )
    return {
        "ok": True,
        "guest_process_count": 1,
        "pointer_button_mask": 0,
        "semantic_operations": [
            {"kind": "move_relative", "args": [-1363, -627]},
            {"kind": "mouse_down", "args": ["left"]},
            {"kind": "mouse_up", "args": ["left"]},
        ],
        "lowered_operations": [
            {"kind": "move_relative", "args": [-1363, -627]},
            {"kind": "click", "args": ["left"]},
        ],
        "backend_primitives": [
            {"kind": "move_to", "call": "pyautogui.moveTo"},
            {
                "kind": "click",
                "click_backend": BACKEND_BY_ARM[arm],
                "dwell_ms": 50,
                "release_side_motion_notify": arm == "A",
                "injection_attempt_count": 1,
                "retry_count": 0,
                "press_xtest_sequence": ["motion_notify", "button_press"],
                "release_xtest_sequence": (
                    ["motion_notify", "button_release"]
                    if arm == "A"
                    else ["button_release"]
                ),
            },
        ],
        "x_injection_timestamps": [timestamp],
        "x_injection_evidence": evidence,
        "passive_x_observer": {
            "installed": False,
            "observer_process_count": 0,
            "additional_x_connection_count": 0,
            "assessment": "omitted_not_demonstrably_non_perturbing",
            "limitation": (
                "not installed: a same-process XRecord/XI2 observer requires a second X "
                "connection and concurrent event consumption, which is not demonstrably "
                "non-perturbing for this timing experiment"
            ),
        },
    }


def test_backend_contract_varies_only_preregistered_release_motion() -> None:
    a = validate_atomic_contract(_atomic("A"), arm="A")
    b = validate_atomic_contract(_atomic("B"), arm="B")
    assert a["semantic_kinds"] == b["semantic_kinds"]
    assert a["lowered_kinds"] == b["lowered_kinds"]
    assert a["click_primitive"]["dwell_ms"] == b["click_primitive"]["dwell_ms"] == 50
    assert a["click_primitive"]["release_side_motion_notify"] is True
    assert b["click_primitive"]["release_side_motion_notify"] is False


@pytest.mark.parametrize(
    ("arm", "field", "value"),
    [
        ("A", "guest_process_count", 2),
        ("A", "pointer_button_mask", 256),
        ("B", "semantic_operations", []),
        ("B", "lowered_operations", []),
        ("B", "x_injection_timestamps", []),
    ],
)
def test_backend_contract_rejects_integrity_drift(arm: str, field: str, value) -> None:
    atomic = _atomic(arm)
    atomic[field] = value
    with pytest.raises(InjectionAbIntegrityError):
        validate_atomic_contract(atomic, arm=arm)


def _trials(a_failures: int, b_failures: int) -> list[dict]:
    return [
        {
            "arm": arm,
            "outcome": {
                "primary_success": index >= (a_failures if arm == "A" else b_failures)
            },
        }
        for arm in ("A", "B")
        for index in range(24)
    ]


@pytest.mark.parametrize(
    ("a_failures", "b_failures", "expected"),
    [
        (0, 0, "failure_not_reproduced"),
        (1, 0, "supports_release_motion_hypothesis"),
        (0, 1, "contradicts_release_motion_hypothesis"),
        (2, 1, "directional_support_with_shared_failures_inconclusive"),
        (1, 1, "nondifferential_failures_inconclusive"),
        (1, 2, "directional_contradiction_with_shared_failures"),
    ],
)
def test_interpretation_table_is_total(
    a_failures: int, b_failures: int, expected: str
) -> None:
    result = interpret_results(_trials(a_failures, b_failures))
    assert result["interpretation"] == expected
    assert result["failure_counts"] == {"A": a_failures, "B": b_failures}
    assert result["descriptive_only"] is True
    assert result["qualification_authorized"] is False


def test_backend_bound_transport_does_not_change_canonical_operations() -> None:
    class Recording:
        def __init__(self) -> None:
            self.audit = object()
            self.seen = None

        def execute_atomic(self, operations, *, click_backend):
            self.seen = (operations, click_backend)
            return "result"

    base = Recording()
    bound = BackendBoundTransport(base, BACKEND_BY_ARM["B"])
    operations = (
        Operation("move_relative", (1, 2)),
        Operation("mouse_down", ("left",)),
        Operation("mouse_up", ("left",)),
    )
    assert bound.execute_atomic(operations) == "result"
    assert base.seen == (operations, BACKEND_BY_ARM["B"])
