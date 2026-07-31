from __future__ import annotations

import json
from copy import deepcopy

import pytest

from osworld_parity.proper_vm_capability_ladder.rung1.transport_diagnostic import (
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
