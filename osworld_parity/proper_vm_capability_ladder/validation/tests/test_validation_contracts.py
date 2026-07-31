from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from osworld_parity.proper_vm_capability_ladder.validation.aggregate import (
    CertificationError,
    capability_report_sha256,
    validate_click_shards,
)
from osworld_parity.proper_vm_capability_ladder.validation.artifact_index import (
    ArtifactIndexError,
    _input_bindings,
    canonical_bytes,
    validate_index,
)
from osworld_parity.proper_vm_capability_ladder.validation.failure_artifact_probe import (
    INJECTED_MESSAGE,
    validate_injected_failure,
)
from osworld_parity.proper_vm_capability_ladder.validation import qualification_result


def _click_trial(shard: int, pair: int, arm: str) -> dict:
    slug = "native" if arm == "native_absolute_control" else "compact"
    trial = 2 * ((pair - 1) % 25) + (1 if slug == "native" else 2)
    return {
        "pair_index": (pair - 1) % 25 + 1,
        "global_pair_index": pair,
        "trial_index": trial,
        "trial_id": f"cert-3059bd4c8057-s{shard}-pair-{pair:03d}-{slug}",
        "arm": arm,
        "status": "passed",
        "retry_count": 0,
        "dispatch_count": 1,
        "reset_before_trial": True,
        "oracle_invocation_count": 0,
        "oracle_conditioned_dispatch": False,
        "final_pointer_button_mask": 0,
        "final_state": {"checked": True, "decoy_checked": False},
        "lowered_operations": ["click"],
        "backend_primitives": [
            {
                "kind": "click",
                "button": "left",
                "call": "pyautogui.click(clicks=1, interval=0.05)",
                "x11_per_event_sync_hooked": True,
            }
        ],
        "x_event_sync_evidence": [
            {"event": "mouse_down", "flush": True, "sync": True},
            {"event": "mouse_up", "flush": True, "sync": True},
        ],
    }


def _click_shards() -> dict[int, dict]:
    arms = ("native_absolute_control", "compact_raw_phaseb")
    values = {}
    for shard in range(4):
        trials = [
            _click_trial(shard, pair, arm)
            for pair in range(shard * 25 + 1, shard * 25 + 26)
            for arm in arms
        ]
        values[shard] = {
            "schema_version": 1,
            "status": "passed",
            "mode": "vm",
            "retry_count": 0,
            "infrastructure_error_count": 0,
            "verifier_failure_count": 0,
            "gpu_count": 0,
            "model_access": False,
            "sealed_evaluation_access": False,
            "shard_index": shard,
            "shard_count": 4,
            "pair_count": 25,
            "trial_count": 50,
            "arm_trial_counts": {arm: 25 for arm in arms},
            "spec_sha256": "3059bd4c8057f0922652a7320fc0e6362ab3a850aeecc0ac6027855dd4f943b6",
            "trials": trials,
        }
    return values


def test_full_click_gate_requires_every_preregistered_trial() -> None:
    values = _click_shards()
    validate_click_shards(values)
    values[2]["trials"].pop()
    with pytest.raises(CertificationError, match="incomplete"):
        validate_click_shards(values)


def test_failure_probe_requires_durable_png(tmp_path: Path) -> None:
    png = tmp_path / "failure.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nbody")
    semantic = [
        {"kind": "mouse_down", "args": ["left"]},
        {"kind": "raise_for_test", "args": [INJECTED_MESSAGE]},
    ]
    raw_marker = 'RUNG1A_ATOMIC_RESULT={"ok":false}'
    atomic = {
        "ok": False,
        "guest_returncode": 17,
        "raw_result_marker": raw_marker,
        "failure_kind": "injected",
        "error": f"RuntimeError: {INJECTED_MESSAGE}",
        "semantic_operations": semantic,
        "lowered_operations": semantic,
        "operations": semantic[:1],
        "cursor_before": [1, 2],
        "cursor_after": [1, 2],
        "cursor": [1, 2],
        "cleanup_attempted": True,
        "pointer_button_mask": 0,
        "expected_pointer_button_mask": 256,
        "observed_pointer_button_mask": -1,
        "guest_process_count": 1,
    }
    screenshot = {
        "path": str(png),
        "bytes": png.stat().st_size,
        "sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
    }
    checks = validate_injected_failure(
        raw_guest={"returncode": 17, "output": raw_marker + "\n"},
        atomic_state=atomic,
        forbidden_success_markers=[tmp_path / "success.json"],
        screenshot=screenshot,
    )
    assert all(checks.values())
    with pytest.raises(Exception, match="exactly the bound marker"):
        validate_injected_failure(
            raw_guest={
                "returncode": 17,
                "output": raw_marker + "\n" + raw_marker + "\n",
            },
            atomic_state=atomic,
            forbidden_success_markers=[],
            screenshot=screenshot,
        )
    png.write_bytes(b"not-png")
    with pytest.raises(Exception, match="screenshot"):
        validate_injected_failure(
            raw_guest={"returncode": 17, "output": raw_marker + "\n"},
            atomic_state=atomic,
            forbidden_success_markers=[],
            screenshot=screenshot,
        )


def test_capability_report_hash_has_a_fixed_canonical_roundtrip() -> None:
    vector = {"schema_version": 1, "status": "ready", "unicode": "Grüße Δ"}
    expected = "1dd0ae1c5628e005a3638289ecdec17e1480a7786772ef66f04fb09703f84f8a"
    assert capability_report_sha256(vector) == expected
    vector["capability_report_sha256"] = expected
    assert capability_report_sha256(vector) == expected
    assert canonical_bytes({"b": 1, "a": "Δ"}) == b'{"a":"\xce\x94","b":1}'


def test_content_address_tampering_fails_closed(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_text("{}\n", encoding="utf-8")
    payload = {
        "schema_version": "proper_vm_executor_artifact_index_v1",
        "terminal_results": {
            "result": {
                "path": str(result),
                "size": result.stat().st_size,
                "sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
            }
        },
    }
    payload["content_address"] = "sha256:" + hashlib.sha256(
        canonical_bytes(payload)
    ).hexdigest()
    validate_index(payload)
    payload["kind"] = "tampered"
    with pytest.raises(ArtifactIndexError, match="content address"):
        validate_index(payload)


def test_labctl_input_array_is_normalized_without_losing_artifact_ids() -> None:
    context = {
        "inputs": [
            {"role": "build", "artifact_id": "artifact_1", "resolved_path": "/a/build"},
            {"role": "vm", "artifact_id": None, "resolved_path": "/a/vm"},
        ]
    }
    assert _input_bindings(context) == {
        "build": {"artifact_id": "artifact_1", "resolved_path": "/a/build"},
        "vm": {"artifact_id": None, "resolved_path": "/a/vm"},
    }
    context["inputs"].append(
        {"role": "build", "artifact_id": "artifact_2", "resolved_path": "/b"}
    )
    with pytest.raises(ArtifactIndexError, match="duplicate input role"):
        _input_bindings(context)


@pytest.mark.parametrize(
    ("kind", "producer_fields"),
    [
        (
            "rung1a",
            {
                "cells": [{}] * 16,
                "provider_shape": "nested",
            },
        ),
        (
            "sameapp",
            {
                "mode": "vm",
                "split": "development",
                "sealed_eval_executed": False,
                "rows": [{}] * 8,
                "provider_shape": "nested",
            },
        ),
    ],
)
def test_validate_result_accepts_producer_shaped_payload_and_rejects_omission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    producer_fields: dict,
) -> None:
    provider = tmp_path / "provider.py"
    provider.write_text("# pinned test provider\n", encoding="utf-8")
    provider_sha256 = hashlib.sha256(provider.read_bytes()).hexdigest()
    monkeypatch.setitem(
        qualification_result.PINNED_SUBSTRATE_SHA256,
        "provider",
        provider_sha256,
    )
    value = {
        "status": "passed",
        "retry_count": 0,
        "infrastructure_error_count": 0,
        "gpu_count": 0,
        "model_access": False,
        "sealed_evaluation_access": False,
        **producer_fields,
    }
    value.pop("provider_shape")
    value["provider"] = {
        "path": str(provider.resolve()),
        "sha256": provider_sha256,
    }
    result = tmp_path / f"{kind}.json"
    result.write_text(json.dumps(value), encoding="utf-8")
    assert qualification_result.validate_result(
        kind=kind, path=result, provider=provider
    ) == value

    del value["retry_count"]
    result.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(
        qualification_result.QualificationResultError,
        match="must emit retry_count=0 explicitly",
    ):
        qualification_result.validate_result(
            kind=kind, path=result, provider=provider
        )
