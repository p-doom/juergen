from __future__ import annotations

from pathlib import Path

import pytest

from osworld_parity.proper_vm_capability_ladder.rung5_official_pilot.authorize import main
from osworld_parity.proper_vm_capability_ladder.rung5_official_pilot.gates import (
    GateBundle,
    GateError,
    SignedGatePaths,
    canonical_json,
    verify_gate_bundle,
    with_authorized_source,
)

from .conftest import NOW, SignedBundleFixture


def test_valid_signed_bundle_releases_sanitized_contract(
    signed_bundle: SignedBundleFixture,
) -> None:
    authorization = verify_gate_bundle(signed_bundle.bundle, now=NOW)
    assert authorization.pilot_id == "pilot-mock-rung5-v1"
    assert authorization.task_count == 8
    assert authorization.max_episodes == 32
    assert "path" not in authorization.as_dict()


def test_missing_gates_stop_before_source_factory(tmp_path: Path) -> None:
    called = False

    def poison_source() -> object:
        nonlocal called
        called = True
        raise AssertionError("heldout source became reachable")

    missing = GateBundle(
        prerequisites=SignedGatePaths(tmp_path / "missing-a", tmp_path / "missing-a.sig"),
        pilot_release=SignedGatePaths(tmp_path / "missing-b", tmp_path / "missing-b.sig"),
        allowed_signers=tmp_path / "missing-signers",
        signer_identity="proper-vm-roadmap-release",
    )
    with pytest.raises(GateError):
        with_authorized_source(missing, poison_source, lambda _auth, _source: None, now=NOW)
    assert called is False


def test_tampered_gate_stops_before_source_factory(
    signed_bundle: SignedBundleFixture,
) -> None:
    called = False
    payload_path = signed_bundle.bundle.pilot_release.payload
    payload = dict(signed_bundle.release_payload)
    payload["decision"] = "authorize-tampered"
    payload_path.write_bytes(canonical_json(payload))

    def poison_source() -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(GateError, match="signature"):
        with_authorized_source(
            signed_bundle.bundle, poison_source, lambda _auth, _source: None, now=NOW
        )
    assert called is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("task_ids", ["mock-secret"]),
        ("task_path", "/not/allowed"),
        ("task_hash", "a" * 64),
    ],
)
def test_even_signed_gates_cannot_disclose_official_details(
    signed_bundle: SignedBundleFixture, key: str, value: object
) -> None:
    payload = dict(signed_bundle.release_payload)
    payload[key] = value
    signed_bundle.sign(signed_bundle.bundle.pilot_release.payload, payload)
    with pytest.raises(GateError, match="forbidden|filesystem|SHA-256"):
        verify_gate_bundle(signed_bundle.bundle, now=NOW)


def test_unsigned_authorize_cli_refuses_and_writes_nothing(tmp_path: Path) -> None:
    output = tmp_path / "authorization.json"
    status = main(
        [
            "--prerequisites-gate",
            str(tmp_path / "missing-a"),
            "--prerequisites-signature",
            str(tmp_path / "missing-a.sig"),
            "--pilot-release-gate",
            str(tmp_path / "missing-b"),
            "--pilot-release-signature",
            str(tmp_path / "missing-b.sig"),
            "--allowed-signers",
            str(tmp_path / "missing-signers"),
            "--signer-identity",
            "proper-vm-roadmap-release",
            "--output",
            str(output),
        ]
    )
    assert status == 2
    assert not output.exists()
