from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import pytest

import osworld_parity.proper_vm_capability_ladder.rung1.injection_ab_diagnostic as injection_ab
import osworld_parity.proper_vm_capability_ladder.rung1.server as server_module

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
    recover_sequence_sealed_audit_snapshot,
    select_science_window_audit_events,
    validate_atomic_contract,
    validate_audit_trace,
    validate_injection_ab,
    validate_post_window_audit_heartbeat,
)
from osworld_parity.proper_vm_capability_ladder.rung1.fixtures import load_manifest
from osworld_parity.proper_vm_capability_ladder.rung1.server import FixtureStateStore
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
    assert spec["suite"] == "rung1_click_release_injection_ab_v4"
    assert spec["browser_audit"]["wire_schema_version"] == 3
    assert spec["browser_audit"][
        "require_heartbeat_causally_generated_after_observation_deadline"
    ] is True
    assert spec["browser_audit"][
        "acknowledged_host_record_must_precede_marker"
    ] is True
    assert spec["post_window_heartbeat_wait_basis"]["absolute_cap"] is True
    assert spec["post_window_heartbeat_wait_basis"][
        "deadline_basis"
    ] == "observation_deadline_host_monotonic_ns"
    assert spec["science_window_contract"] == {
        "duration_s": 3.0,
        "deadline_basis": "dispatch_completed_host_monotonic_ns",
        "audit_event_inclusion_rule": (
            "direct events require host_monotonic_ns <= "
            "observation_deadline_host_monotonic_ns; recovered DOM events "
            "require a later directly received browser-sequence event whose "
            "host_monotonic_ns <= observation_deadline_host_monotonic_ns"
        ),
        "recovered_event_without_host_before_deadline_proof": "integrity_abort",
        "host_reporter_event_inclusion_rule": (
            "host_monotonic_ns <= observation_deadline_host_monotonic_ns"
        ),
        "post_window_sequence_prefix_retained_as_evidence": True,
        "post_window_events_are_not_outcome_bearing": True,
    }
    assert spec["browser_failure_evidence_contract"][
        "checkpoint_before_validation"
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
    assert spec["alias_mismatch_abort_history"]["build_job_id"] == "136174"
    v2_history = spec["v2_integrity_abort_history"]
    assert v2_history["diagnostic_job_id"] == "136178"
    assert v2_history["completed_trial_count"] == 5
    assert v2_history["arm_completed_counts"] == {"A": 2, "B": 3}
    assert v2_history["integrity_error_evidence"] is None
    assert v2_history["vm_closed"] is v2_history["overlay_removed"] is True
    v3_history = spec["v3_integrity_abort_history"]
    assert v3_history["diagnostic_job_id"] == "136206"
    assert v3_history["completed_trial_count"] == 8
    assert v3_history["arm_completed_counts"] == {"A": 4, "B": 4}
    assert v3_history["primary_success_counts"] == {"A": 3, "B": 1}
    assert v3_history["interpretation"] == (
        "protocol_inconclusive_truncated_prefix_not_inferentially_poolable"
    )
    assert v3_history["vm_closed"] is v3_history["overlay_removed"] is True
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
        "schema_version": 3,
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
        value["acknowledged_heartbeat_audit_sequence"] = None
        value["acknowledged_host_audit_request_id"] = None
        value["acknowledged_host_monotonic_ns"] = None
        value["sealed_journal_through_sequence"] = sequence - 1
        value["sealed_journal_json"] = "[]"
        value["sealed_journal_sha256"] = hashlib.sha256(b"[]").hexdigest()
    assert all(field in value for field in AUDIT_REQUIRED_FIELDS)
    return value


def _audit_snapshot(events: list[dict]) -> dict:
    return {
        "browser_audit_events": events,
        "browser_audit_dropped": 0,
        "events": [],
        "current": {"checked": False, "decoy_checked": False},
    }


def _acknowledge_heartbeat(
    marker: dict, acknowledged: dict
) -> None:
    marker["acknowledged_heartbeat_audit_sequence"] = acknowledged[
        "audit_sequence"
    ]
    marker["acknowledged_host_audit_request_id"] = acknowledged[
        "host_audit_request_id"
    ]
    marker["acknowledged_host_monotonic_ns"] = acknowledged[
        "host_monotonic_ns"
    ]


def _journal_record(event: dict) -> dict:
    omitted = {
        "host_audit_request_id",
        "host_monotonic_ns",
        "host_wall_time_ns",
        "sealed_journal_json",
        "transport_recovered_from_marker_sequence",
        "transport_recovery_journal_sha256",
        "recovered_host_monotonic_upper_bound_ns",
    }
    return {
        key: copy.deepcopy(value)
        for key, value in event.items()
        if key not in omitted
    }


def _seal_heartbeat(marker: dict, prior_events: list[dict]) -> None:
    marker["sealed_journal_through_sequence"] = marker["audit_sequence"] - 1
    journal_json = json.dumps(
        [_journal_record(event) for event in prior_events],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    marker["sealed_journal_json"] = journal_json
    marker["sealed_journal_sha256"] = hashlib.sha256(
        journal_json.encode("utf-8")
    ).hexdigest()


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
    outcome = classify_trial_outcome(trace, _audit_snapshot(events), 10_000)
    assert outcome["primary_success"] is True
    assert outcome["outcome"] == "trusted_click_input_change"
    assert outcome["host_reporter_event_sequence"] == []


@pytest.mark.parametrize(
    ("arrival_delta_ns", "expected_success"),
    [(0, True), (1, False)],
)
def test_classifier_enforces_exact_science_deadline_boundary(
    arrival_delta_ns: int, expected_success: bool
) -> None:
    deadline = 1_000
    events = [_audit_event(1, "audit_ready")]
    for sequence, name in enumerate(
        ("pointerdown", "pointerup", "click", "input", "change"), start=2
    ):
        event = _audit_event(sequence, name, checked=True)
        event["host_monotonic_ns"] = deadline + arrival_delta_ns
        events.append(event)

    outcome = classify_trial_outcome(
        events,
        _audit_snapshot(events),
        deadline,
    )

    assert outcome["primary_success"] is expected_success
    assert outcome["audit_event_sequence"] == (
        ["pointerdown", "pointerup", "click", "input", "change"]
        if expected_success
        else []
    )


def test_classifier_filters_secondary_reporter_at_science_deadline() -> None:
    deadline = 1_000
    snapshot = _audit_snapshot([_audit_event(1, "audit_ready")])
    snapshot["events"] = [
        {
            "kind": "pointer",
            "event": "pointerdown",
            "host_monotonic_ns": deadline,
        },
        {
            "kind": "pointer",
            "event": "pointerup",
            "host_monotonic_ns": deadline + 1,
        },
    ]

    outcome = classify_trial_outcome(
        snapshot["browser_audit_events"], snapshot, deadline
    )

    assert outcome["host_reporter_event_sequence"] == ["pointerdown"]


def test_post_window_heartbeat_seals_independent_audit_completeness() -> None:
    ready = _audit_event(1, "audit_ready")
    acknowledged = _audit_event(2, "audit_heartbeat")
    _seal_heartbeat(acknowledged, [ready])
    acknowledged["host_monotonic_ns"] = 3_000_000_001
    heartbeat = _audit_event(3, "audit_heartbeat")
    heartbeat["host_monotonic_ns"] = 3_000_000_002
    _acknowledge_heartbeat(heartbeat, acknowledged)
    _seal_heartbeat(heartbeat, [ready, acknowledged])
    trace = validate_audit_trace(
        _audit_snapshot([ready, acknowledged, heartbeat]), 7
    )
    sealed, marker = validate_post_window_audit_heartbeat(
        trace, 3_000_000_000, 3_000_000_100
    )
    assert sealed == [ready, acknowledged, heartbeat]
    assert marker == heartbeat
    with pytest.raises(
        InjectionAbIntegrityError,
        match="causally generated post-window heartbeat",
    ) as caught:
        validate_post_window_audit_heartbeat(
            trace, 3_000_000_001, 3_000_000_100
        )
    assert caught.value.evidence["heartbeats"][-1][
        "acknowledged_receipt_lag_ns"
    ] == 0


def test_post_window_marker_detects_lost_tail_and_excludes_post_marker_events() -> None:
    ready = _audit_event(1, "audit_ready")
    acknowledged = _audit_event(2, "audit_heartbeat")
    _seal_heartbeat(acknowledged, [ready])
    acknowledged["host_monotonic_ns"] = 101
    heartbeat = _audit_event(3, "audit_heartbeat")
    heartbeat["host_monotonic_ns"] = 102
    _acknowledge_heartbeat(heartbeat, acknowledged)
    _seal_heartbeat(heartbeat, [ready, acknowledged])
    post_marker_click = _audit_event(4, "click", checked=True)
    trace = validate_audit_trace(
        _audit_snapshot([ready, acknowledged, heartbeat, post_marker_click]), 7
    )
    sealed, marker = validate_post_window_audit_heartbeat(trace, 100, 200)
    assert marker == heartbeat
    assert sealed == [ready, acknowledged, heartbeat]
    heartbeat["expected_audit_count_through_marker"] = 4
    with pytest.raises(InjectionAbIntegrityError, match="sequence/count"):
        validate_post_window_audit_heartbeat(trace, 100, 200)

    lost_sendbeacon_marker = _audit_event(4, "audit_heartbeat")
    _acknowledge_heartbeat(lost_sendbeacon_marker, acknowledged)
    with pytest.raises(InjectionAbIntegrityError, match="omitted"):
        validate_audit_trace(
            _audit_snapshot([ready, acknowledged, lost_sendbeacon_marker]), 7
        )


def test_post_window_validator_rejects_marker_after_absolute_wait_cap() -> None:
    ready = _audit_event(1, "audit_ready")
    acknowledged = _audit_event(2, "audit_heartbeat")
    _seal_heartbeat(acknowledged, [ready])
    acknowledged["host_monotonic_ns"] = 101
    marker = _audit_event(3, "audit_heartbeat")
    marker["host_monotonic_ns"] = 201
    _acknowledge_heartbeat(marker, acknowledged)
    _seal_heartbeat(marker, [ready, acknowledged])
    trace = validate_audit_trace(
        _audit_snapshot([ready, acknowledged, marker]), 7
    )

    with pytest.raises(
        InjectionAbIntegrityError,
        match="causally generated post-window heartbeat",
    ) as caught:
        validate_post_window_audit_heartbeat(trace, 100, 200)

    assert caught.value.evidence["wait_deadline_host_monotonic_ns"] == 200
    assert caught.value.evidence["heartbeats"][-1]["host_monotonic_ns"] == 201


def _journal_recovery_case(
    dropped_sequences: set[int], *, witness_host_ns: int = 90
) -> tuple[dict, int]:
    ready = _audit_event(1, "audit_ready")
    pointerdown = _audit_event(2, "pointerdown")
    pointerup = _audit_event(3, "pointerup")
    pointerup["host_monotonic_ns"] = witness_host_ns
    witness = _audit_event(4, "mousemove")
    witness["host_monotonic_ns"] = witness_host_ns
    acknowledged = _audit_event(5, "audit_heartbeat")
    acknowledged["host_monotonic_ns"] = 150
    _seal_heartbeat(
        acknowledged, [ready, pointerdown, pointerup, witness]
    )
    marker = _audit_event(6, "audit_heartbeat")
    marker["host_monotonic_ns"] = 160
    _acknowledge_heartbeat(marker, acknowledged)
    all_events = [ready, pointerdown, pointerup, witness, acknowledged]
    _seal_heartbeat(marker, all_events)
    raw = [
        event
        for event in [ready, pointerdown, pointerup, witness, acknowledged, marker]
        if event["audit_sequence"] not in dropped_sequences
    ]
    return _audit_snapshot(raw), marker["audit_sequence"]


@pytest.mark.parametrize("dropped_sequences", [{2}, {2, 3}])
def test_sealed_browser_journal_recovers_one_or_multiple_dropped_beacons(
    dropped_sequences: set[int],
) -> None:
    snapshot, marker_sequence = _journal_recovery_case(dropped_sequences)
    sealed_snapshot, recovery = recover_sequence_sealed_audit_snapshot(
        snapshot, marker_sequence
    )
    trace = validate_audit_trace(
        sealed_snapshot,
        7,
        allow_request_id_gaps_after_sequence_seal=True,
    )

    assert [event["audit_sequence"] for event in trace] == list(range(1, 7))
    assert recovery["recovered_audit_sequences"] == sorted(dropped_sequences)
    assert recovery["raw_host_audit_sequences"] == sorted(
        set(range(1, 7)) - dropped_sequences
    )
    for sequence in dropped_sequences:
        recovered = trace[sequence - 1]
        assert recovered["transport_recovered_from_marker_sequence"] == 6
        assert recovered["host_monotonic_ns"] is None
        assert recovered["recovered_host_monotonic_upper_bound_ns"] == 90


@pytest.mark.parametrize(
    "mutation",
    [
        "hash",
        "direct_event",
        "json_type",
        "seal_count_type",
        "sequence",
        "reserved_field",
        "duplicate_key",
    ],
)
def test_tampered_browser_journal_is_integrity_abort(mutation: str) -> None:
    snapshot, marker_sequence = _journal_recovery_case({2})
    marker = next(
        event
        for event in snapshot["browser_audit_events"]
        if event["audit_sequence"] == marker_sequence
    )
    journal = json.loads(marker["sealed_journal_json"])
    if mutation == "hash":
        journal[1]["event"] = "change"
        marker["sealed_journal_json"] = json.dumps(
            journal, separators=(",", ":")
        )
    elif mutation == "direct_event":
        journal[3]["client_x"] += 1
        marker["sealed_journal_json"] = json.dumps(
            journal, separators=(",", ":")
        )
        marker["sealed_journal_sha256"] = hashlib.sha256(
            marker["sealed_journal_json"].encode()
        ).hexdigest()
    elif mutation == "json_type":
        journal[3]["is_trusted"] = 1
        marker["sealed_journal_json"] = json.dumps(
            journal, separators=(",", ":")
        )
        marker["sealed_journal_sha256"] = hashlib.sha256(
            marker["sealed_journal_json"].encode()
        ).hexdigest()
    elif mutation == "seal_count_type":
        marker["sealed_journal_through_sequence"] = float(
            marker["sealed_journal_through_sequence"]
        )
    elif mutation == "sequence":
        journal[1]["audit_sequence"] = 9
        marker["sealed_journal_json"] = json.dumps(
            journal, separators=(",", ":")
        )
        marker["sealed_journal_sha256"] = hashlib.sha256(
            marker["sealed_journal_json"].encode()
        ).hexdigest()
    elif mutation == "reserved_field":
        journal[1]["host_monotonic_ns"] = 50
        marker["sealed_journal_json"] = json.dumps(
            journal, separators=(",", ":")
        )
        marker["sealed_journal_sha256"] = hashlib.sha256(
            marker["sealed_journal_json"].encode()
        ).hexdigest()
    else:
        marker["sealed_journal_json"] = marker[
            "sealed_journal_json"
        ].replace(
            '"audit_sequence":1',
            '"audit_sequence":1,"audit_sequence":1',
            1,
        )
        marker["sealed_journal_sha256"] = hashlib.sha256(
            marker["sealed_journal_json"].encode()
        ).hexdigest()

    with pytest.raises(InjectionAbIntegrityError, match="journal"):
        recover_sequence_sealed_audit_snapshot(snapshot, marker_sequence)


def test_journal_recovery_accepts_reordered_transport_and_rejects_duplicate() -> None:
    snapshot, marker_sequence = _journal_recovery_case({2})
    snapshot["browser_audit_events"].reverse()
    sealed_snapshot, _recovery = recover_sequence_sealed_audit_snapshot(
        snapshot, marker_sequence
    )
    assert [
        event["audit_sequence"]
        for event in sealed_snapshot["browser_audit_events"]
    ] == list(range(1, 7))

    duplicate = copy.deepcopy(snapshot)
    duplicate["browser_audit_events"].append(
        copy.deepcopy(duplicate["browser_audit_events"][-1])
    )
    with pytest.raises(InjectionAbIntegrityError, match="duplicated"):
        recover_sequence_sealed_audit_snapshot(duplicate, marker_sequence)


@pytest.mark.parametrize("malformed_sequence", [None, True, 0, "7"])
def test_journal_recovery_rejects_malformed_raw_sequence(
    malformed_sequence,
) -> None:
    snapshot, marker_sequence = _journal_recovery_case({2})
    snapshot["browser_audit_events"].append(
        {"audit_sequence": malformed_sequence}
    )

    with pytest.raises(
        InjectionAbIntegrityError,
        match="raw browser audit sequence type/range malformed",
    ):
        recover_sequence_sealed_audit_snapshot(snapshot, marker_sequence)


def test_journal_recovery_cannot_substitute_for_direct_ready_receipt() -> None:
    snapshot, marker_sequence = _journal_recovery_case({1})
    sealed_snapshot, _recovery = recover_sequence_sealed_audit_snapshot(
        snapshot, marker_sequence
    )

    with pytest.raises(
        InjectionAbIntegrityError,
        match="audit_ready was not directly received",
    ):
        validate_audit_trace(
            sealed_snapshot,
            7,
            allow_request_id_gaps_after_sequence_seal=True,
        )


@pytest.mark.parametrize(
    ("witness_host_ns", "accepted"), [(100, True), (101, False)]
)
def test_recovered_event_requires_exact_host_deadline_causal_bound(
    witness_host_ns: int, accepted: bool
) -> None:
    snapshot, marker_sequence = _journal_recovery_case(
        {2}, witness_host_ns=witness_host_ns
    )
    sealed_snapshot, _recovery = recover_sequence_sealed_audit_snapshot(
        snapshot, marker_sequence
    )
    trace = validate_audit_trace(
        sealed_snapshot,
        7,
        allow_request_id_gaps_after_sequence_seal=True,
    )

    if accepted:
        science = select_science_window_audit_events(trace, 100)
        assert 2 in [event["audit_sequence"] for event in science]
    else:
        with pytest.raises(
            InjectionAbIntegrityError,
            match="host-before-deadline proof",
        ):
            select_science_window_audit_events(trace, 100)


def _recovered_change_case() -> dict:
    ready = _audit_event(1, "audit_ready")
    events = [ready]
    for sequence, name in enumerate(
        ("pointerdown", "pointerup", "click", "input", "change"), start=2
    ):
        event = _audit_event(sequence, name, checked=True)
        event["host_monotonic_ns"] = 50 + sequence
        events.append(event)
    witness = _audit_event(7, "audit_heartbeat")
    witness["host_monotonic_ns"] = 100
    _seal_heartbeat(witness, events)
    acknowledged = _audit_event(8, "audit_heartbeat")
    acknowledged["host_monotonic_ns"] = 150
    _seal_heartbeat(acknowledged, [*events, witness])
    marker = _audit_event(9, "audit_heartbeat")
    marker["host_monotonic_ns"] = 160
    _acknowledge_heartbeat(marker, acknowledged)
    _seal_heartbeat(marker, [*events, witness, acknowledged])
    raw = [ready, *events[1:-1], witness, acknowledged, marker]
    return _audit_snapshot(raw)


def test_recovered_change_counts_only_with_causal_predeadline_witness() -> None:
    snapshot = _recovered_change_case()

    sealed_snapshot, recovery = recover_sequence_sealed_audit_snapshot(
        snapshot, 9
    )
    trace = validate_audit_trace(
        sealed_snapshot,
        7,
        allow_request_id_gaps_after_sequence_seal=True,
    )
    outcome = classify_trial_outcome(trace, snapshot, 100)

    assert recovery["recovered_audit_sequences"] == [6]
    assert trace[5]["event"] == "change"
    assert trace[5]["recovered_host_monotonic_upper_bound_ns"] == 100
    assert outcome["primary_success"] is True


def test_conflicting_overlapping_heartbeat_journals_abort_recovery() -> None:
    snapshot = _recovered_change_case()
    marker = next(
        event
        for event in snapshot["browser_audit_events"]
        if event["audit_sequence"] == 9
    )
    journal = json.loads(marker["sealed_journal_json"])
    assert journal[5]["event"] == "change"
    journal[5]["event"] = "input"
    marker["sealed_journal_json"] = json.dumps(
        journal, ensure_ascii=False, separators=(",", ":")
    )
    marker["sealed_journal_sha256"] = hashlib.sha256(
        marker["sealed_journal_json"].encode("utf-8")
    ).hexdigest()

    with pytest.raises(InjectionAbIntegrityError, match="not append-only"):
        recover_sequence_sealed_audit_snapshot(snapshot, 9)


def _apply_store_audit(
    store: FixtureStateStore, fixture, event: dict
) -> dict[str, int]:
    payload = copy.deepcopy(event)
    for field in (
        "host_audit_request_id",
        "host_monotonic_ns",
        "host_wall_time_ns",
    ):
        payload.pop(field, None)
    payload["generation"] = store.snapshot(fixture.id)["generation"]
    return store.apply_browser_audit(fixture, payload)


def _set_heartbeat_ack_from_response(marker: dict, response: dict) -> None:
    marker["acknowledged_heartbeat_audit_sequence"] = response[
        "audit_sequence"
    ]
    marker["acknowledged_host_audit_request_id"] = response[
        "host_audit_request_id"
    ]
    marker["acknowledged_host_monotonic_ns"] = response[
        "host_monotonic_ns"
    ]


def test_causal_wait_accepts_journal_marker_despite_direct_gap() -> None:
    fixture = next(
        item
        for item in load_manifest().select(split="development")
        if item.id == injection_ab.FIXTURE_ID
    )
    store = FixtureStateStore(load_manifest())
    generation = store.snapshot(fixture.id)["generation"]
    ready = _audit_event(1, "audit_ready")
    _apply_store_audit(store, fixture, ready)
    acknowledged_event = _audit_event(2, "audit_heartbeat")
    _seal_heartbeat(acknowledged_event, [ready])
    acknowledged = _apply_store_audit(
        store, fixture, acknowledged_event
    )
    marker = _audit_event(4, "audit_heartbeat")
    _set_heartbeat_ack_from_response(marker, acknowledged)
    _seal_heartbeat(
        marker,
        [ready, acknowledged_event, _audit_event(3, "pointermove")],
    )
    _apply_store_audit(store, fixture, marker)
    _apply_store_audit(store, fixture, _audit_event(6, "pointermove"))
    snapshot, wait = store.wait_for_causal_post_window_heartbeat(
        fixture.id,
        generation=generation,
        observation_deadline_host_monotonic_ns=(
            acknowledged["host_monotonic_ns"] - 1
        ),
        timeout_s=0.5,
    )

    assert [event["audit_sequence"] for event in snapshot["browser_audit_events"]] == [
        1,
        2,
        4,
        6,
    ]
    assert wait["timed_out"] is False
    assert wait["candidate_audit_sequences"] == [4]


@pytest.mark.parametrize("failure", ["stale_ack", "missing_journal_seal"])
def test_causal_wait_times_out_with_exact_raw_evidence(
    failure: str,
) -> None:
    fixture = next(
        item
        for item in load_manifest().select(split="development")
        if item.id == injection_ab.FIXTURE_ID
    )
    store = FixtureStateStore(load_manifest())
    generation = store.snapshot(fixture.id)["generation"]
    ready = _audit_event(1, "audit_ready")
    _apply_store_audit(store, fixture, ready)
    acknowledged_event = _audit_event(2, "audit_heartbeat")
    _seal_heartbeat(acknowledged_event, [ready])
    acknowledged = _apply_store_audit(
        store, fixture, acknowledged_event
    )
    marker_sequence = 3
    marker = _audit_event(marker_sequence, "audit_heartbeat")
    _set_heartbeat_ack_from_response(marker, acknowledged)
    _seal_heartbeat(marker, [ready, acknowledged_event])
    if failure == "stale_ack":
        marker["acknowledged_host_audit_request_id"] += 1
    else:
        marker.pop("sealed_journal_json")
    _apply_store_audit(store, fixture, marker)

    snapshot, wait = store.wait_for_causal_post_window_heartbeat(
        fixture.id,
        generation=generation,
        observation_deadline_host_monotonic_ns=(
            acknowledged["host_monotonic_ns"] - 1
        ),
        timeout_s=0.005,
    )

    assert wait["timed_out"] is True
    assert wait["candidate_audit_sequences"] == []
    assert wait["heartbeat_summaries"][-1]["audit_sequence"] == marker_sequence
    assert snapshot["diagnostic_journal"][-1]["stage"] == (
        "causal_audit_wait_completed"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "partial",
        "future",
        "wrong_request",
        "wrong_host",
        "nonheartbeat",
        "request_order",
        "host_order",
    ],
)
def test_heartbeat_acknowledgement_identity_corruption_aborts(
    mutation: str,
) -> None:
    ready = _audit_event(1, "audit_ready")
    acknowledged = _audit_event(2, "audit_heartbeat")
    _seal_heartbeat(acknowledged, [ready])
    marker = _audit_event(3, "audit_heartbeat")
    _acknowledge_heartbeat(marker, acknowledged)
    _seal_heartbeat(marker, [ready, acknowledged])
    if mutation == "partial":
        marker["acknowledged_host_monotonic_ns"] = None
    elif mutation == "future":
        marker["acknowledged_heartbeat_audit_sequence"] = 3
        marker["acknowledged_host_audit_request_id"] = 3
        marker["acknowledged_host_monotonic_ns"] = 3
    elif mutation == "wrong_request":
        marker["acknowledged_host_audit_request_id"] += 1
    elif mutation == "wrong_host":
        marker["acknowledged_host_monotonic_ns"] += 1
    elif mutation == "request_order":
        marker["host_audit_request_id"] = marker[
            "acknowledged_host_audit_request_id"
        ]
    elif mutation == "host_order":
        marker["host_monotonic_ns"] = marker[
            "acknowledged_host_monotonic_ns"
        ]
    else:
        marker["acknowledged_heartbeat_audit_sequence"] = 1
        marker["acknowledged_host_audit_request_id"] = 1
        marker["acknowledged_host_monotonic_ns"] = 1
    expected_error = (
        "host request identities|acknowledgement"
        if mutation == "request_order"
        else "acknowledgement"
    )
    with pytest.raises(InjectionAbIntegrityError, match=expected_error):
        validate_audit_trace(
            _audit_snapshot([ready, acknowledged, marker]), 7
        )


def test_late_prestored_marker_is_rejected_by_absolute_wait_cap() -> None:
    observation_deadline = 100
    wait_deadline = 200
    acknowledged = {
        "audit_sequence": 2,
        "event": "audit_heartbeat",
        "host_audit_request_id": 2,
        "host_monotonic_ns": 150,
    }

    def state(marker_host_ns: int) -> dict:
        return {
            "browser_audit_events": [
                {
                    "audit_sequence": 1,
                    "event": "audit_ready",
                    "host_audit_request_id": 1,
                    "host_monotonic_ns": 90,
                },
                acknowledged,
                {
                    "audit_sequence": 3,
                    "event": "audit_heartbeat",
                    "host_audit_request_id": 3,
                    "host_monotonic_ns": marker_host_ns,
                    "acknowledged_heartbeat_audit_sequence": 2,
                    "acknowledged_host_audit_request_id": 2,
                    "acknowledged_host_monotonic_ns": 150,
                    "sealed_journal_through_sequence": 2,
                    "sealed_journal_json": "[]",
                    "sealed_journal_sha256": "0" * 64,
                },
            ]
        }

    assert FixtureStateStore._causal_post_window_marker_sequences(
        state(wait_deadline), observation_deadline, wait_deadline
    ) == [3]
    assert FixtureStateStore._causal_post_window_marker_sequences(
        state(wait_deadline + 1), observation_deadline, wait_deadline
    ) == []
    duplicate_marker = state(wait_deadline)
    duplicate_marker["browser_audit_events"].append(
        copy.deepcopy(duplicate_marker["browser_audit_events"][-1])
    )
    assert FixtureStateStore._causal_post_window_marker_sequences(
        duplicate_marker, observation_deadline, wait_deadline
    ) == []
    float_count_marker = state(wait_deadline)
    float_count_marker["browser_audit_events"][-1][
        "sealed_journal_through_sequence"
    ] = 2.0
    assert FixtureStateStore._causal_post_window_marker_sequences(
        float_count_marker, observation_deadline, wait_deadline
    ) == []


def test_overslept_waiter_uses_absolute_observation_deadline_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = next(
        item
        for item in load_manifest().select(split="development")
        if item.id == injection_ab.FIXTURE_ID
    )
    store = FixtureStateStore(load_manifest())
    generation = store.snapshot(fixture.id)["generation"]
    observation_deadline = 100
    monkeypatch.setattr(server_module.time, "monotonic_ns", lambda: 4_000_000_100)

    _snapshot, wait = store.wait_for_causal_post_window_heartbeat(
        fixture.id,
        generation=generation,
        observation_deadline_host_monotonic_ns=observation_deadline,
        timeout_s=3.0,
    )

    assert wait["wait_started_host_monotonic_ns"] == 4_000_000_100
    assert wait["wait_deadline_host_monotonic_ns"] == 3_000_000_100
    assert wait["timed_out"] is True
    assert wait["wait_duration_ns"] == 0


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
        "x_injection_start_sequence": 1,
        "x_injection_end_sequence": 6 if arm == "A" else 5,
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
        x_event(1, "canonical_move", "motion_notify", 6, 0, 50, x=365, y=345),
        x_event(2, "click_premove", "motion_notify", 6, 0, 95, x=365, y=345),
        x_event(3, "press", "motion_notify", 6, 0, 110, x=365, y=345),
        x_event(4, "press", "button_press", 4, 1, 120),
        x_event(
            5 if arm == "B" else 6,
            "release",
            "button_release",
            5,
            1,
            50_000_170 if arm == "B" else 50_000_180,
        ),
    ]
    if arm == "A":
        evidence.insert(
            4,
            x_event(
                5,
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
        "x_sync_attempt_evidence": [
            {
                "sequence": sequence,
                "phase": phase,
                "attempted": True,
                "success": True,
                "error": None,
                "started_guest_monotonic_ns": sequence,
                "completed_guest_monotonic_ns": sequence + 1,
                "duration_ns": 1,
            }
            for sequence, phase in enumerate(
                [
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
                1,
            )
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
    identity_fields = ("phase", "event", "event_type", "detail", "x", "y")
    a_without_release_motion = [
        tuple(item[field] for field in identity_fields)
        for item in a["x_injection_evidence"]
        if not (item["phase"] == "release" and item["event"] == "motion_notify")
    ]
    b_identities = [
        tuple(item[field] for field in identity_fields)
        for item in b["x_injection_evidence"]
    ]
    assert a_without_release_motion == b_identities
    assert [
        item["phase"] for item in a["x_sync_attempt_evidence"]
    ] == [item["phase"] for item in b["x_sync_attempt_evidence"]]


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
    release_motion = atomic["x_injection_evidence"][4]
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
    premove = records[1]
    if mutation == "omission":
        records.pop(1)
        for sequence, record in enumerate(records, 1):
            record["sequence"] = sequence
        atomic["x_injection_timestamps"][0]["x_injection_end_sequence"] = 5
    elif mutation == "phase":
        premove["phase"] = "press"
    elif mutation == "coordinates":
        premove["x"], premove["y"] = 9999, -9999
    elif mutation == "event_type":
        premove["event_type"] = 5
    else:
        premove["sequence"], records[2]["sequence"] = 3, 2
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


def test_browser_timeout_persists_raw_snapshot_journal_and_timing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = _audit_event(1, "audit_ready")
    heartbeat = _audit_event(2, "audit_heartbeat")
    raw_snapshot = {
        **_audit_snapshot([ready, heartbeat]),
        "diagnostic_journal": [
            {"journal_sequence": 1, "stage": "causal_audit_wait_completed"}
        ],
    }
    wait_evidence = {
        "schema_version": 1,
        "generation": 7,
        "observation_deadline_host_monotonic_ns": 100,
        "wait_started_host_monotonic_ns": 100,
        "wait_completed_host_monotonic_ns": 200,
        "wait_duration_ns": 100,
        "wait_timeout_s": 3.0,
        "wait_deadline_host_monotonic_ns": 3_000_000_100,
        "timed_out": True,
        "candidate_audit_sequences": [],
        "heartbeat_summaries": [
            {
                "audit_sequence": 2,
                "client_monotonic_ms": 2.0,
                "host_monotonic_ns": 2,
                "host_arrival_lag_ns": -98,
                "acknowledged_heartbeat_audit_sequence": None,
                "acknowledged_host_audit_request_id": None,
                "acknowledged_host_monotonic_ns": None,
                "acknowledged_receipt_lag_ns": None,
            }
        ],
    }

    def fail(**kwargs):
        injection_ab._checkpoint_browser_audit_capture(
            output=kwargs["output"],
            trials=[],
            active={"trial_id": "injection-ab-01-a"},
            snapshot=raw_snapshot,
            wait_evidence=wait_evidence,
        )
        raise InjectionAbIntegrityError(
            "bounded causal post-window heartbeat wait timed out",
            evidence=wait_evidence,
        )

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
    active = progress["active_trial"]
    assert active["raw_browser_snapshot"] == raw_snapshot
    assert active["post_window_heartbeat_wait"] == wait_evidence
    assert failure["progress"]["active_trial"] == active
    evidence = failure["integrity_error"]["evidence"]
    assert evidence["observation_deadline_host_monotonic_ns"] == 100
    assert evidence["heartbeat_summaries"][0]["audit_sequence"] == 2
    assert evidence["heartbeat_summaries"][0]["host_arrival_lag_ns"] == -98


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
        "raw_x_sync_attempt_evidence": [{"sequence": 1}],
        "raw_x_injection_timestamps": [{"x_injection_start_sequence": 0}],
        "raw_x_injection_evidence": [
            {"sequence": 1, "phase": "click_premove", "event": "motion_notify"}
        ],
        "guest_error": "injected identity failure",
        "guest_failure_kind": "infrastructure",
        "pointer_masks": {"final": 0, "observed": -1, "expected": 0},
        "final_pointer_readback": {"attempted": True, "success": True},
        "attempt_hook_restore_errors": [],
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
