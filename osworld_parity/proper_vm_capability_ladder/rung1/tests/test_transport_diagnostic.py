from __future__ import annotations

import json
from copy import deepcopy
from itertools import count
from types import SimpleNamespace

import pytest

from osworld_parity.proper_vm_capability_ladder.rung1 import transport_diagnostic
from osworld_parity.proper_vm_capability_ladder.rung1.transport_diagnostic import (
    ATTEMPT_EVIDENCE_SCHEMA,
    BROWSER_SEQUENCE,
    CLICK_CALL,
    FIXTURE_ID,
    MANIFEST_PAYLOAD_SHA256,
    PAIR_COUNT,
    CERTIFICATION_PAIRS_PER_SHARD,
    CERTIFICATION_SHARD_COUNT,
    TransportDiagnosticError,
    _atomic_contract,
    _browser_contract,
    _classify_timeout_outcome,
    _checkpoint,
    _run_trial,
    certification_trial_identity,
    _select_fixture,
    _semantic_contract,
    load_transport_diagnostic_spec,
    load_transport_certification_spec,
    main as transport_diagnostic_main,
    validate_transport_certification,
    validate_transport_diagnostic,
)
from osworld_parity.proper_vm_capability_ladder.rung1.fixtures import load_manifest


def test_transport_diagnostic_spec_is_fixed_and_development_only() -> None:
    spec, spec_sha256 = load_transport_diagnostic_spec()
    fixture = _select_fixture(load_manifest(), spec)
    assert fixture.id == FIXTURE_ID
    assert fixture.split == "development"
    assert fixture.template == "click"
    assert spec["manifest_payload_sha256"] == MANIFEST_PAYLOAD_SHA256
    assert spec["pair_count"] == PAIR_COUNT == 5
    assert spec["gpu_count"] == 0
    assert spec["model_access"] is False
    assert spec["sealed_evaluation_access"] is False
    assert len(spec_sha256) == 64
    report = validate_transport_diagnostic()
    assert report["trial_count"] == 10
    assert report["manifest_payload_sha256"] == MANIFEST_PAYLOAD_SHA256


def test_transport_certification_spec_has_four_disjoint_fixed_shards() -> None:
    spec, spec_sha256 = load_transport_certification_spec()
    assert spec["shard_count"] == CERTIFICATION_SHARD_COUNT == 4
    assert spec["pairs_per_shard"] == CERTIFICATION_PAIRS_PER_SHARD == 25
    reports = [
        validate_transport_certification(index)
        for index in range(CERTIFICATION_SHARD_COUNT)
    ]
    assert [report["global_pair_range"] for report in reports] == [
        [1, 25],
        [26, 50],
        [51, 75],
        [76, 100],
    ]
    assert all(report["trial_count"] == 50 for report in reports)
    assert all(report["retry_count"] == 0 for report in reports)
    assert all(report["infrastructure_error_count"] == 0 for report in reports)
    assert all(report["gpu_count"] == 0 for report in reports)
    assert all(report["model_access"] is False for report in reports)
    assert all(report["sealed_evaluation_access"] is False for report in reports)
    assert len(spec_sha256) == 64
    identities = [
        certification_trial_identity(spec_sha256, shard, pair, arm)[2]
        for shard in range(CERTIFICATION_SHARD_COUNT)
        for pair in range(1, CERTIFICATION_PAIRS_PER_SHARD + 1)
        for arm in ("native_absolute_control", "compact_raw_phaseb")
    ]
    assert len(identities) == len(set(identities)) == 200
    assert all(spec_sha256[:12] in trial_id for trial_id in identities)


def test_invalid_certification_shard_fails_nonzero_with_artifact(tmp_path) -> None:
    assert (
        transport_diagnostic_main(
            [
                "--mode",
                "validate",
                "--suite",
                "certification",
                "--shard-index",
                "4",
                "--output",
                str(tmp_path),
            ]
        )
        == 2
    )
    failure = json.loads(
        (tmp_path / "transport_certification_failure_shard_4.json").read_text()
    )
    assert failure["status"] == "failed"
    assert failure["failure_kind"] == "verification"
    assert failure["verifier_failure_count"] == 1
    assert failure["retry_count"] == 0


def test_transport_diagnostic_requires_exact_semantic_and_atomic_contracts() -> None:
    endpoint = (300, 400)
    dispatch = [
        {
            "parse_status": "ok",
            "executor_dispatch_status": "ok",
            "operations": [
                {"kind": "move_to", "args": [300, 400]},
                {"kind": "mouse_down", "args": ["left"]},
                {"kind": "mouse_up", "args": ["left"]},
            ],
        }
    ]
    assert [item["kind"] for item in _semantic_contract(dispatch, endpoint)] == [
        "move_to",
        "mouse_down",
        "mouse_up",
    ]
    journal = {
        "atomic_action_states": [
            {
                "ok": True,
                "pointer_button_mask": 0,
                "observed_pointer_button_mask": 0,
                "expected_pointer_button_mask": 0,
                "guest_process_count": 1,
                "guest_returncode": 0,
                "raw_result_marker": "RUNG1A_ATOMIC_RESULT={}",
                "cleanup_attempted": False,
                "error": None,
                "failure_kind": None,
                "cursor": [300, 400],
                "cursor_before": [10, 20],
                "cursor_after": [300, 400],
                "semantic_operations": [
                    {"kind": "move_relative", "args": [290, 380]},
                    {"kind": "mouse_down", "args": ["left"]},
                    {"kind": "mouse_up", "args": ["left"]},
                ],
                "lowered_operations": [
                    {"kind": "move_relative", "args": [290, 380]},
                    {"kind": "click", "args": ["left"]},
                ],
                "backend_primitives": [
                    {
                        "kind": "click",
                        "button": "left",
                        "call": CLICK_CALL,
                        "x11_per_event_sync_hooked": True,
                    }
                ],
                "x_event_sync_evidence": [
                    {
                        "event": "mouse_down",
                        "backend": "pyautogui._pyautogui_x11",
                        "flush": True,
                        "sync": True,
                    },
                    {
                        "event": "mouse_up",
                        "backend": "pyautogui._pyautogui_x11",
                        "flush": True,
                        "sync": True,
                    },
                ],
            }
        ]
    }
    assert _atomic_contract(journal)["lowered_operations"] == ["click"]
    held = deepcopy(journal)
    held["atomic_action_states"][0]["pointer_button_mask"] = 256
    with pytest.raises(TransportDiagnosticError, match="atomic state contract mismatch"):
        _atomic_contract(held)


def test_transport_diagnostic_requires_exact_causal_browser_sequence() -> None:
    endpoint = (300, 400)
    events = [
        {
            "kind": "pointer",
            "event": "pointerdown",
            "client_sequence": 5,
            "button": 0,
            "buttons": 1,
            "screen_x": 300,
            "screen_y": 400,
            "hit_id": "target",
        },
        {
            "kind": "pointer",
            "event": "pointerup",
            "client_sequence": 6,
            "button": 0,
            "buttons": 0,
            "screen_x": 300,
            "screen_y": 400,
            "hit_id": "target",
        },
        {"kind": "click", "client_sequence": 7},
    ]
    acknowledgement = {"events": events, "pointer_buttons": 0}
    contract = _browser_contract(acknowledgement, endpoint)
    assert contract["sequence"] == list(BROWSER_SEQUENCE)
    assert contract["state_event"] == "change/click"
    missing_up = deepcopy(acknowledgement)
    del missing_up["events"][1]
    with pytest.raises(TransportDiagnosticError, match="causal sequence mismatch"):
        _browser_contract(missing_up, endpoint)


def _trial_test_doubles(monkeypatch, *, timeout: bool):
    fixture = load_manifest().by_id(FIXTURE_ID)
    endpoint = (300, 400)
    dispatch = [
        {
            "parse_status": "ok",
            "executor_dispatch_status": "ok",
            "operations": [
                {"kind": "move_to", "args": [300, 400]},
                {"kind": "mouse_down", "args": ["left"]},
                {"kind": "mouse_up", "args": ["left"]},
            ],
        }
    ]
    atomic_result = {
        "ok": True,
        "guest_process_count": 1,
        "guest_returncode": 0,
        "raw_result_marker": 'RUNG1A_ATOMIC_RESULT={"ok":true}',
        "cursor": [300, 400],
        "cursor_before": [10, 20],
        "cursor_after": [300, 400],
        "pointer_button_mask": 0,
        "observed_pointer_button_mask": 0,
        "expected_pointer_button_mask": 0,
        "cleanup_attempted": False,
        "error": None,
        "failure_kind": None,
        "backend_primitives": [
            {"kind": "click", "call": CLICK_CALL, "button": "left"}
        ],
    }
    journal = {
        "atomic_action_states": [atomic_result],
        "atomic_invocation_count": 1,
    }
    atomic_contract = {
        "lowered_operations": ["click"],
        "backend_primitives": atomic_result["backend_primitives"],
        "x_event_sync_evidence": [
            {"event": "mouse_down", "flush": True, "sync": True},
            {"event": "mouse_up", "flush": True, "sync": True},
        ],
        "real_cursor_before": [10, 20],
        "real_cursor_after": [300, 400],
        "final_pointer_button_mask": 0,
    }
    acknowledgement = {
        "after_sequence": 4,
        "last_sequence": 7,
        "pointer_buttons": 0,
        "events": [{"kind": "click", "client_sequence": 7}],
    }

    class FakeTransport:
        def screen_size(self):
            return (1920, 1080)

        def cursor_position(self):
            return (10, 20)

    class FakeStore:
        checkpoint_seen = False

        def __init__(self):
            self.snapshot_count = 0

        def snapshot(self, fixture_id):
            assert fixture_id == fixture.id
            self.snapshot_count += 1
            if self.snapshot_count == 1:
                return {"last_client_sequence": 4}
            if timeout:
                return {
                    "last_client_sequence": 5,
                    "last_pointer_buttons": 1,
                    "current": {"checked": False, "decoy_checked": False},
                    "events": [
                        {
                            "kind": "pointer",
                            "event": "pointerdown",
                            "client_sequence": 5,
                            "host_monotonic_ns": 1500,
                        }
                    ],
                }
            return {"current": {"checked": True, "decoy_checked": False}}

        def wait_for_browser_quiescence(self, fixture_id, **kwargs):
            assert self.checkpoint_seen is True
            if timeout:
                raise TimeoutError("deterministic browser acknowledgement timeout")
            return acknowledgement

    store = FakeStore()

    class FakeSession:
        def reset_to_ready(self):
            return FakeTransport()

        def launch_fixture(self, server, selected_fixture):
            assert selected_fixture is fixture
            return {"current": {"checked": False, "decoy_checked": False}}

        def capture_browser_diagnostics(self, selected_fixture):
            assert selected_fixture is fixture
            return {
                "status": "captured",
                "transport": "cdp_runtime_evaluate",
                "page": {
                    "captured_client_monotonic_ms": 42.5,
                    "captured_browser_wall_time_ms": 1_700_000_000_000,
                    "performance_time_origin_ms": 1_699_999_999_000,
                    "performance_now_ms": 1000.0,
                    "diagnostics": {
                        "page_events": [
                            {
                                "kind": "pointer",
                                "event": "pointerdown",
                                "client_sequence": 5,
                            },
                            {
                                "kind": "pointer",
                                "event": "pointerup",
                                "client_sequence": 6,
                            },
                            {"kind": "click", "client_sequence": 7},
                        ],
                        "report_queue": {
                            "enqueued": 7,
                            "send_started": 5,
                            "acknowledged": 4,
                            "failed": 0,
                            "pending": 3,
                            "records": [],
                        },
                    },
                    "dom": {
                        "target": {"checked": True},
                        "decoy": {"checked": False},
                        "outer_html": "<html>fixture</html>",
                    },
                },
            }

        def capture_guest_pointer_state(self):
            return {
                "status": "captured",
                "guest_returncode": 0,
                "raw_result_marker": "RUNG1A_POINTER_STATE={}",
                "cursor": [300, 400],
                "raw_x_mask": 0,
                "pointer_button_mask": 0,
                "guest_wall_before_ns": 2_000,
                "guest_wall_after_ns": 2_100,
                "guest_monotonic_before_ns": 3_000,
                "guest_monotonic_after_ns": 3_100,
            }

        def capture_chrome_log(self):
            return {
                "status": "captured",
                "total_bytes": 12,
                "captured_bytes": 12,
                "sha256": "a" * 64,
                "truncated": False,
                "content_tail": "chrome trace",
                "guest_returncode": 0,
            }

    clock = count(1000, 100)
    monkeypatch.setattr(transport_diagnostic.time, "monotonic_ns", lambda: next(clock))
    monkeypatch.setattr(transport_diagnostic.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(transport_diagnostic, "_validate_loaded_geometry", lambda *a: None)
    monkeypatch.setattr(
        transport_diagnostic,
        "build_trajectory",
        lambda *a, **k: SimpleNamespace(actions=(object(),), expected_endpoint=endpoint),
    )
    dispatch_count = {"value": 0}

    def execute(*args):
        dispatch_count["value"] += 1
        return dispatch, journal

    monkeypatch.setattr(transport_diagnostic, "_execute", execute)
    monkeypatch.setattr(transport_diagnostic, "_assert_dispatch_journal", lambda *a: None)
    monkeypatch.setattr(
        transport_diagnostic, "_semantic_contract", lambda *a: dispatch[0]["operations"]
    )
    monkeypatch.setattr(transport_diagnostic, "_atomic_contract", lambda *a: atomic_contract)
    monkeypatch.setattr(
        transport_diagnostic,
        "_browser_contract",
        lambda *a: {"sequence": list(BROWSER_SEQUENCE), "final_pointer_buttons": 0},
    )
    return fixture, FakeSession(), SimpleNamespace(store=store), store, dispatch_count


def test_trial_checkpoints_full_atomic_evidence_before_success_wait(monkeypatch) -> None:
    fixture, session, server, store, dispatch_count = _trial_test_doubles(
        monkeypatch, timeout=False
    )
    checkpoints = []

    def checkpoint(attempt):
        checkpoints.append(deepcopy(attempt))
        store.checkpoint_seen = True

    trial = _run_trial(
        session=session,
        server=server,
        fixture=fixture,
        pair_index=1,
        trial_index=1,
        arm="native_absolute_control",
        pair_id="pair-001",
        trial_id="pair-001-native",
        global_pair_index=1,
        checkpoint_attempt=checkpoint,
    )
    before_wait, passed = checkpoints
    assert before_wait["evidence_schema"] == ATTEMPT_EVIDENCE_SCHEMA
    assert before_wait["stage"] == "before_browser_acknowledgement"
    assert before_wait["dispatch"] == trial["attempt_evidence"]["dispatch"]
    assert before_wait["journal"]["atomic_action_states"][0]["guest_returncode"] == 0
    assert before_wait["atomic_result"]["raw_result_marker"].startswith(
        "RUNG1A_ATOMIC_RESULT="
    )
    assert before_wait["progress"] == {
        "dispatch_count": 1,
        "retry_count": 0,
        "atomic_guest_process_count": 1,
        "browser_acknowledged": False,
    }
    assert passed["status"] == trial["status"] == "passed"
    assert passed["progress"]["browser_acknowledged"] is True
    assert dispatch_count["value"] == 1
    assert trial["backend_primitives"][0]["call"] == CLICK_CALL


def test_trial_timeout_retains_live_page_x_log_and_timing_evidence(
    monkeypatch, tmp_path
) -> None:
    fixture, session, server, store, dispatch_count = _trial_test_doubles(
        monkeypatch, timeout=True
    )
    checkpoints = []

    def checkpoint(attempt):
        checkpoints.append(deepcopy(attempt))
        store.checkpoint_seen = True
        _checkpoint(
            tmp_path,
            trials=[],
            pairs=[],
            active_trial=attempt["trial"],
            attempted_trial=attempt,
            stage=attempt["stage"],
        )

    with pytest.raises(TransportDiagnosticError) as raised:
        _run_trial(
            session=session,
            server=server,
            fixture=fixture,
            pair_index=1,
            trial_index=1,
            arm="compact_raw_phaseb",
            pair_id="pair-001",
            trial_id="pair-001-compact",
            global_pair_index=1,
            checkpoint_attempt=checkpoint,
        )
    evidence = raised.value.evidence
    assert evidence["status"] == "failed"
    assert evidence["failure_kind"] == "infrastructure"
    assert evidence["failure_stage"] == "browser_acknowledgement_timeout"
    attempt = evidence["attempted_trial"]
    assert attempt["status"] == "failed"
    assert attempt["journal"]["atomic_action_states"][0]["guest_returncode"] == 0
    assert attempt["atomic_result"]["cursor"] == [300, 400]
    assert attempt["atomic_result"]["pointer_button_mask"] == 0
    assert [item.get("event") for item in attempt["browser_page_event_log"][:2]] == [
        "pointerdown",
        "pointerup",
    ]
    assert attempt["report_queue"]["enqueued"] == 7
    assert attempt["report_queue"]["send_started"] == 5
    assert attempt["report_queue"]["acknowledged"] == 4
    assert attempt["dom_state"]["target"]["checked"] is True
    assert attempt["live_guest_pointer"]["value"]["pointer_button_mask"] == 0
    assert attempt["chrome_log"]["value"]["sha256"] == "a" * 64
    assert attempt["observation_grace"]["requested_s"] == 0.25
    assert attempt["observation_grace"]["no_input_dispatched"] is True
    assert attempt["outcome_classifier"]["classification"] == "browser_reporter"
    assert attempt["cross_clock_calibration"]["browser_post_grace"][
        "performance_time_origin_ms"
    ] == 1_699_999_999_000
    assert attempt["cross_clock_calibration"]["guest_post_grace"][
        "guest_monotonic_after_ns"
    ] == 3_100
    assert attempt["timings"]["dispatch_duration_ns"] == 100
    assert attempt["timings"]["browser_ack_wait_duration_ns"] == 100
    assert dispatch_count["value"] == 1
    progress = json.loads(
        (tmp_path / "transport_diagnostic_progress.json").read_text()
    )
    assert progress["status"] == "failed"
    assert progress["infrastructure_error_count"] == 1
    assert progress["attempted_trial"]["evidence_schema"] == ATTEMPT_EVIDENCE_SCHEMA
    failure_output = tmp_path / "failure-artifact"

    def raise_timeout(**kwargs):
        raise raised.value

    monkeypatch.setattr(
        transport_diagnostic, "run_vm_transport_diagnostic", raise_timeout
    )
    assert (
        transport_diagnostic_main(
            ["--mode", "vm", "--output", str(failure_output)]
        )
        == 2
    )
    failure = json.loads((failure_output / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["failure_kind"] == "infrastructure"
    assert failure["infrastructure_error_count"] == 1
    assert failure["evidence"]["attempted_trial"]["report_queue"]["enqueued"] == 7
    assert not (failure_output / "transport_diagnostic.json").exists()


def _classifier_attempt(
    *,
    page_events=None,
    records=None,
    journal=None,
    pointer_mask=0,
    x_events=None,
):
    captured = lambda value: {"status": "captured", "value": value}
    attempt = {
        "live_browser": captured(
            {
                "page": {
                    "diagnostics": {
                        "page_events": page_events or [],
                        "report_queue": {"records": records or []},
                    }
                }
            }
        ),
        "host_oracle_snapshot": captured({"diagnostic_journal": journal or []}),
        "live_guest_pointer": captured(
            {"pointer_button_mask": pointer_mask}
        ),
    }
    if x_events is not None:
        attempt["passive_x_observer"] = {
            "status": "captured",
            "events": x_events,
        }
    return attempt


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [
        (
            _classifier_attempt(
                pointer_mask=256,
                x_events=[{"event": "button_press"}],
            ),
            "guest_input_path",
        ),
        (
            _classifier_attempt(
                pointer_mask=0,
                x_events=[
                    {"event": "button_press"},
                    {"event": "button_release"},
                ],
                page_events=[
                    {"kind": "pointer", "event": "pointerdown", "client_sequence": 5}
                ],
            ),
            "chromium_input_delivery",
        ),
        (
            _classifier_attempt(
                page_events=[
                    {"kind": "pointer", "event": "pointerup", "client_sequence": 6},
                    {"kind": "click", "client_sequence": 7},
                ],
                records=[
                    {"client_sequence": 6, "state": "resolved"},
                    {"client_sequence": 7, "state": "fetch_started"},
                ],
                journal=[
                    {
                        "stage": "http_body_received",
                        "details": {"client_sequence": 6},
                    },
                    {
                        "stage": "store_apply_committed",
                        "details": {"client_sequence": 6},
                    },
                ],
            ),
            "browser_reporter",
        ),
        (
            _classifier_attempt(
                page_events=[
                    {"kind": "pointer", "event": "pointerup", "client_sequence": 6},
                    {"kind": "click", "client_sequence": 7},
                ],
                records=[
                    {"client_sequence": 6, "state": "resolved"},
                    {"client_sequence": 7, "state": "fetch_started"},
                ],
                journal=[
                    {
                        "stage": "http_body_received",
                        "details": {"client_sequence": 7},
                    }
                ],
            ),
            "host_harness",
        ),
        (_classifier_attempt(pointer_mask=0), "inconclusive"),
    ],
)
def test_timeout_outcome_classifier_rules_are_sequence_specific(
    attempt, expected
) -> None:
    result = _classify_timeout_outcome(attempt)
    assert result["classification"] == expected
    if expected in {"browser_reporter", "host_harness"}:
        assert result["observed"]["click_client_sequences"] == [7]
