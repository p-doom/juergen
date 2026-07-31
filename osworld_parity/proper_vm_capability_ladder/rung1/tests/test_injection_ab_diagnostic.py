from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import osworld_parity.proper_vm_capability_ladder.rung1.injection_ab_diagnostic as injection_ab

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
    validate_post_window_audit_heartbeat,
)
from osworld_parity.proper_vm_capability_ladder.rung1.transport import (
    PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
    Operation,
    TransportError,
    compile_atomic_guest_program,
)


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
    assert spec["common_pyautogui_click_premove"]["arm_neutral"] is True
    assert spec["control_plane_guardrail"]["authoritative_host"] == (
        "hai-login2.haicore.berlin"
    )
    assert spec["control_plane_guardrail"]["submission_authorized"] is False
    assert spec["control_plane_guardrail"]["manual_reconcile_authorized"] is False
    assert spec["failure_evidence_contract"]["schema_version"] == (
        "rung1_atomic_output_failure_v2"
    )
    assert spec["failure_evidence_contract"][
        "checkpoint_immediately_after_dispatch"
    ] is True
    history = spec["pre_science_integrity_abort_history"]
    assert history["job_id"] == "136152"
    assert history["completed_trial_count"] == 0
    assert history["retry_count"] == history["replacement_count"] == 0
    assert history["vm_closed"] is history["overlay_removed"] is True
    assert history["accidental_control_plane_action"] == {
        "command": "labctl reconcile",
        "result": "duplicate artifact primary key artifact_7143accb7e53783d",
        "effect": "no run revival, relabel, replacement, or additional Slurm job",
    }
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
    marker = event in {"audit_ready", "audit_heartbeat"}
    pointer_or_mouse = event.startswith(("pointer", "mouse")) or event == "click"
    value = {
        "schema_version": 1,
        "generation": 7,
        "audit_sequence": sequence,
        "event": event,
        "browser_wall_time_ms": 1000 + sequence,
        "client_monotonic_ms": float(sequence),
        "event_time_stamp_ms": None if marker else float(sequence),
        "is_trusted": None if marker else True,
        "default_prevented": None if marker else False,
        "target": (
            {"id": "target", "tag": "input", "checked": checked}
            if not marker
            else None
        ),
        "target_checked": None if marker else checked,
        "checkbox_state": {"target": checked, "decoy": False},
        "active_element": {"id": "target", "tag": "input", "checked": checked},
        "document_has_focus": True,
        "visibility_state": "visible",
        "button": 0 if pointer_or_mouse else None,
        "buttons": 0 if pointer_or_mouse else None,
        "pointer_type": "mouse" if event.startswith("pointer") else None,
        "client_x": 295 if pointer_or_mouse else None,
        "client_y": 231 if pointer_or_mouse else None,
        "screen_x": 365 if pointer_or_mouse else None,
        "screen_y": 345 if pointer_or_mouse else None,
        "host_audit_request_id": sequence,
        "host_monotonic_ns": sequence,
        "host_wall_time_ns": sequence,
    }
    if event == "audit_ready":
        value["page_time_origin_ms"] = 123.5
        value["url"] = "http://10.0.2.2/fixture/r1a-click-dev-1101"
    elif event == "audit_heartbeat":
        value["expected_previous_audit_sequence"] = sequence - 1
        value["expected_audit_count_through_marker"] = sequence
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


def test_post_window_heartbeat_seals_independent_audit_completeness() -> None:
    ready = _audit_event(1, "audit_ready")
    heartbeat = _audit_event(2, "audit_heartbeat")
    heartbeat["host_monotonic_ns"] = 3_000_000_001
    trace = validate_audit_trace(_audit_snapshot([ready, heartbeat]), 7)
    sealed, marker = validate_post_window_audit_heartbeat(trace, 3_000_000_000)
    assert sealed == [ready, heartbeat]
    assert marker == heartbeat
    with pytest.raises(InjectionAbIntegrityError, match="post-window heartbeat"):
        validate_post_window_audit_heartbeat(trace, 3_000_000_001)


def test_post_window_marker_detects_lost_tail_and_excludes_post_marker_events() -> None:
    ready = _audit_event(1, "audit_ready")
    heartbeat = _audit_event(2, "audit_heartbeat")
    heartbeat["host_monotonic_ns"] = 101
    post_marker_click = _audit_event(3, "click", checked=True)
    trace = validate_audit_trace(
        _audit_snapshot([ready, heartbeat, post_marker_click]), 7
    )
    sealed, marker = validate_post_window_audit_heartbeat(trace, 100)
    assert marker == heartbeat
    assert sealed == [ready, heartbeat]
    heartbeat["expected_audit_count_through_marker"] = 3
    with pytest.raises(InjectionAbIntegrityError, match="sequence/count"):
        validate_post_window_audit_heartbeat(trace, 100)

    lost_sendbeacon_marker = _audit_event(3, "audit_heartbeat")
    lost_sendbeacon_marker["host_monotonic_ns"] = 101
    with pytest.raises(InjectionAbIntegrityError, match="omitted"):
        validate_audit_trace(
            _audit_snapshot([ready, lost_sendbeacon_marker]), 7
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("browser_wall_time_ms", float("nan")),
        ("client_monotonic_ms", -1),
        ("event_time_stamp_ms", float("inf")),
        ("is_trusted", 1),
        ("default_prevented", "false"),
        ("target", {"id": "target", "tag": "input"}),
        ("target_checked", None),
        ("checkbox_state", {"target": False, "decoy": 0}),
        ("active_element", {"id": 4, "tag": "input", "checked": False}),
        ("document_has_focus", 1),
        ("visibility_state", "prerender"),
        ("button", 17),
        ("buttons", -1),
        ("pointer_type", "trackball"),
    ],
)
def test_audit_schema_corruption_is_integrity_abort(field: str, value) -> None:
    events = [_audit_event(1, "audit_ready"), _audit_event(2, "pointerdown")]
    events[1][field] = value
    with pytest.raises(InjectionAbIntegrityError):
        validate_audit_trace(_audit_snapshot(events), 7)


@pytest.mark.parametrize(
    "mutation",
    [
        "omission",
        "duplicate",
        "missing_field",
        "malformed_field",
        "stale",
        "overflow",
        "ready_duplicate",
        "request_id_duplicate",
    ],
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
    elif mutation == "malformed_field":
        events[1]["document_has_focus"] = "yes"
    elif mutation == "stale":
        events[1]["generation"] = 6
    elif mutation == "overflow":
        snapshot["browser_audit_dropped"] = 1
    elif mutation == "ready_duplicate":
        duplicate = copy.deepcopy(events[0])
        duplicate["audit_sequence"] = 2
        events[:] = [events[0], duplicate]
    else:
        events[1]["host_audit_request_id"] = 1
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
        "press_call_success": True,
        "press_call_error": None,
        "dwell_success": True,
        "dwell_error": None,
        "release_call_success": True,
        "release_call_error": None,
        "x_injection_start_sequence": 0,
        "x_injection_end_sequence": 5 if arm == "A" else 4,
        "click_started_guest_monotonic_ns": 90,
        "press_call_before_guest_monotonic_ns": 100,
        "press_call_after_guest_monotonic_ns": 130,
        "press_sync_completed_guest_monotonic_ns": 140,
        "dwell_started_guest_monotonic_ns": 150,
        "dwell_completed_guest_monotonic_ns": 50_000_150,
        "release_call_before_guest_monotonic_ns": 50_000_160,
        "release_call_after_guest_monotonic_ns": 50_000_190,
        "release_sync_completed_guest_monotonic_ns": 50_000_200,
        "click_completed_guest_monotonic_ns": 50_000_210,
        "click_premove_xtest_sequence": ["motion_notify"],
        "press_xtest_sequence": ["motion_notify", "button_press"],
        "release_xtest_sequence": (
            ["motion_notify", "button_release"]
            if arm == "A"
            else ["button_release"]
        ),
    }

    def x_event(
        sequence: int,
        phase: str,
        event: str,
        event_type: int,
        detail: int,
        started: int,
        *,
        x: int | None = None,
        y: int | None = None,
    ) -> dict:
        return {
            "sequence": sequence,
            "phase": phase,
            "event": event,
            "event_type": event_type,
            "detail": detail,
            "x": x,
            "y": y,
            "attempted": True,
            "success": True,
            "error": None,
            "started_guest_monotonic_ns": started,
            "completed_guest_monotonic_ns": started + 1,
            "duration_ns": 1,
        }

    evidence = [
        x_event(1, "click_premove", "motion_notify", 6, 0, 95, x=365, y=345),
        x_event(2, "press", "motion_notify", 6, 0, 110, x=365, y=345),
        x_event(3, "press", "button_press", 4, 1, 120),
        x_event(
            4 if arm == "B" else 5,
            "release",
            "button_release",
            5,
            1,
            50_000_170 if arm == "B" else 50_000_180,
        ),
    ]
    if arm == "A":
        evidence.insert(
            3,
            x_event(
                4,
                "release",
                "motion_notify",
                6,
                0,
                50_000_170,
                x=365,
                y=345,
            ),
        )
    return {
        "ok": True,
        "click_backend": BACKEND_BY_ARM[arm],
        "guest_process_count": 1,
        "pointer_button_mask": 0,
        "cursor_after": [365, 345],
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
                "button": "left",
                "call": "pyautogui.click(clicks=1, interval=0.05)",
                "click_backend": BACKEND_BY_ARM[arm],
                "x11_per_event_sync_hooked": True,
                "dwell_ms": 50,
                "ordering": [
                    "click_premove_motion",
                    "mouse_down",
                    "flush",
                    "sync",
                    "dwell",
                    "mouse_up",
                    "flush",
                    "sync",
                ],
                "click_premove_same_coordinate_motion_notify": True,
                "release_side_motion_notify": arm == "A",
                "injection_attempt_count": 1,
                "retry_count": 0,
                "click_premove_xtest_sequence": ["motion_notify"],
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
        "x_event_sync_evidence": [
            {
                "event": "mouse_down",
                "backend": "fake_x11",
                "supported": True,
                "flush_attempted": True,
                "flush": True,
                "sync_attempted": True,
                "sync": True,
                "success": True,
                "error": None,
                "started_guest_monotonic_ns": 130,
                "completed_guest_monotonic_ns": 140,
                "duration_ns": 10,
            },
            {
                "event": "mouse_up",
                "backend": "fake_x11",
                "supported": True,
                "flush_attempted": True,
                "flush": True,
                "sync_attempted": True,
                "sync": True,
                "success": True,
                "error": None,
                "started_guest_monotonic_ns": 50_000_190,
                "completed_guest_monotonic_ns": 50_000_200,
                "duration_ns": 10,
            },
        ],
        "final_pointer_readback": {
            "attempted": True,
            "success": True,
            "error": None,
            "cursor": [365, 345],
            "pointer_button_mask": 0,
        },
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
    a = validate_atomic_contract(_atomic("A"), arm="A", expected_endpoint=(365, 345))
    b = validate_atomic_contract(_atomic("B"), arm="B", expected_endpoint=(365, 345))
    assert a["semantic_kinds"] == b["semantic_kinds"]
    assert a["lowered_kinds"] == b["lowered_kinds"]
    assert a["click_primitive"]["dwell_ms"] == b["click_primitive"]["dwell_ms"] == 50
    assert a["click_primitive"]["release_side_motion_notify"] is True
    assert b["click_primitive"]["release_side_motion_notify"] is False


def test_existing_default_compiles_identically_to_explicit_a_backend() -> None:
    operations = (
        Operation("move_relative", (1, 2)),
        Operation("mouse_down", ("left",)),
        Operation("mouse_up", ("left",)),
    )
    default, default_mask = compile_atomic_guest_program(
        operations, initial_buttons=set(), initial_keys=set()
    )
    explicit, explicit_mask = compile_atomic_guest_program(
        operations,
        initial_buttons=set(),
        initial_keys=set(),
        click_backend=PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
    )
    assert default == explicit
    assert default_mask == explicit_mask == 0


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


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("coordinates", (9999, -9999)),
        ("detail", 17),
        ("event_type", 5),
        ("sequence", 400),
        ("phase", "press"),
    ],
)
def test_a_release_motion_identity_corruption_aborts(mutation: str, value) -> None:
    atomic = _atomic("A")
    release_motion = atomic["x_injection_evidence"][3]
    if mutation == "coordinates":
        release_motion["x"], release_motion["y"] = value
    else:
        release_motion[mutation] = value
    with pytest.raises(InjectionAbIntegrityError):
        validate_atomic_contract(
            atomic, arm="A", expected_endpoint=(365, 345)
        )


@pytest.mark.parametrize(
    "mutation",
    ["omission", "phase", "coordinates", "event_type", "order"],
)
def test_click_premove_identity_corruption_aborts(mutation: str) -> None:
    atomic = _atomic("A")
    records = atomic["x_injection_evidence"]
    premove = records[0]
    if mutation == "omission":
        records.pop(0)
        for sequence, record in enumerate(records, 1):
            record["sequence"] = sequence
        atomic["x_injection_timestamps"][0]["x_injection_end_sequence"] = 4
    elif mutation == "phase":
        premove["phase"] = "press"
    elif mutation == "coordinates":
        premove["x"], premove["y"] = 9999, -9999
    elif mutation == "event_type":
        premove["event_type"] = 5
    else:
        premove["sequence"], records[1]["sequence"] = 2, 1
    with pytest.raises(InjectionAbIntegrityError):
        validate_atomic_contract(
            atomic, arm="A", expected_endpoint=(365, 345)
        )


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


def test_vm_integrity_abort_always_persists_failure_and_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(**_kwargs):
        raise InjectionAbIntegrityError("sealed test abort", evidence={"seal": "bad"})

    monkeypatch.setattr(injection_ab, "run_vm_injection_ab", fail)
    rc = injection_ab.main(
        [
            "--mode=vm",
            f"--output={tmp_path}",
            f"--qcow={tmp_path / 'vm.qcow2'}",
            f"--qemu={tmp_path / 'qemu'}",
            f"--provider={tmp_path / 'provider.py'}",
        ]
    )
    assert rc == 2
    failure = json.loads((tmp_path / "injection_ab_failure.json").read_text())
    progress = json.loads((tmp_path / "injection_ab_progress.json").read_text())
    assert failure["status"] == "integrity_abort"
    assert failure["integrity_error"]["evidence"] == {"seal": "bad"}
    assert progress["status"] == "integrity_abort"
    assert progress["stage"] == "integrity_abort"
    assert not (tmp_path / "injection_ab_result.json").exists()


def test_vm_transport_identity_abort_persists_full_raw_x_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = {
        "schema_version": "rung1_atomic_output_failure_v2",
        "click_backend_expected": BACKEND_BY_ARM["A"],
        "expected": [{"phase": "click_premove", "event": "motion_notify"}],
        "observed": [],
        "execute_result": {"status": "error", "returncode": 1, "error": None},
        "raw_stdout": "RUNG1A_ATOMIC_RESULT={...}",
        "raw_result_markers": ["RUNG1A_ATOMIC_RESULT={...}"],
        "raw_payload": {"error": "injected identity failure"},
        "raw_backend_primitives": [{"kind": "click"}],
        "raw_x_event_sync_evidence": [{"event": "mouse_down"}],
        "raw_x_injection_timestamps": [{"x_injection_start_sequence": 0}],
        "raw_x_injection_evidence": [
            {"sequence": 1, "phase": "click_premove", "event": "motion_notify"}
        ],
        "guest_error": "injected identity failure",
        "guest_failure_kind": "infrastructure",
        "pointer_masks": {"final": 0, "observed": -1, "expected": 0},
        "final_pointer_readback": {"attempted": True, "success": True},
    }

    def fail(**_kwargs):
        raise TransportError("atomic guest action click X identity drifted", evidence=evidence)

    monkeypatch.setattr(injection_ab, "run_vm_injection_ab", fail)
    rc = injection_ab.main(
        [
            "--mode=vm",
            f"--output={tmp_path}",
            f"--qcow={tmp_path / 'vm.qcow2'}",
            f"--qemu={tmp_path / 'qemu'}",
            f"--provider={tmp_path / 'provider.py'}",
        ]
    )
    assert rc == 2
    failure = json.loads((tmp_path / "injection_ab_failure.json").read_text())
    progress = json.loads((tmp_path / "injection_ab_progress.json").read_text())
    assert failure["integrity_error"]["evidence"] == evidence
    assert progress["integrity_error"]["evidence"] == evidence


@pytest.mark.parametrize("failure_stage", ["dispatch_assert", "atomic", "audit"])
def test_post_dispatch_abort_preserves_current_dispatch_and_journal(
    failure_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = [{"action": "sealed-dispatch"}]
    journal = {
        "completed_action_count": 1,
        "atomic_action_states": [{"raw": "sealed-atomic-state"}],
    }

    def assert_dispatch(*_args, **_kwargs) -> None:
        if failure_stage == "dispatch_assert":
            raise InjectionAbIntegrityError("injected dispatch assertion failure")

    def validate_atomic(*_args, **_kwargs) -> dict:
        if failure_stage == "atomic":
            raise InjectionAbIntegrityError("injected atomic validation failure")
        return {"sealed": True}

    monkeypatch.setattr(injection_ab, "_assert_dispatch_journal", assert_dispatch)
    monkeypatch.setattr(injection_ab, "validate_atomic_contract", validate_atomic)

    def fail(**kwargs):
        _, active = injection_ab._checkpoint_dispatched_trial(
            output=kwargs["output"],
            trials=[],
            active={"trial_id": "trial-001"},
            fixture=object(),
            dispatch=dispatch,
            journal=journal,
            arm="A",
            expected_endpoint=(365, 345),
            dispatch_started_ns=10,
            dispatch_completed_ns=20,
        )
        assert active["journal"] == journal
        raise InjectionAbIntegrityError("injected audit failure")

    monkeypatch.setattr(injection_ab, "run_vm_injection_ab", fail)
    rc = injection_ab.main(
        [
            "--mode=vm",
            f"--output={tmp_path}",
            f"--qcow={tmp_path / 'vm.qcow2'}",
            f"--qemu={tmp_path / 'qemu'}",
            f"--provider={tmp_path / 'provider.py'}",
        ]
    )
    assert rc == 2
    progress = json.loads((tmp_path / "injection_ab_progress.json").read_text())
    assert progress["status"] == "integrity_abort"
    assert progress["active_trial"]["dispatch"] == dispatch
    assert progress["active_trial"]["journal"] == journal
    expected_checkpoint_stage = (
        "atomic_validated" if failure_stage == "audit" else "dispatched"
    )
    assert progress["active_trial"].get("atomic_contract") == (
        {"sealed": True} if expected_checkpoint_stage == "atomic_validated" else None
    )
