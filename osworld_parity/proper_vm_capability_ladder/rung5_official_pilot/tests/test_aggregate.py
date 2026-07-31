from __future__ import annotations

import json
from pathlib import Path

import pytest

from osworld_parity.proper_vm_capability_ladder.rung5_official_pilot.aggregate import (
    AggregateError,
    aggregate_authorized,
    aggregate_rows,
    main,
)
from osworld_parity.proper_vm_capability_ladder.rung5_official_pilot.contract import (
    COMPACT_RAW_ARM,
)
from osworld_parity.proper_vm_capability_ladder.rung5_official_pilot.gates import (
    GateBundle,
    GateError,
    SignedGatePaths,
    verify_gate_bundle,
)
from osworld_parity.proper_vm_capability_ladder.rung5_official_pilot.records import (
    RecordError,
)

from .conftest import NOW, SignedBundleFixture, mock_rows


def test_paired_bootstrap_and_noninferiority_pass(
    signed_bundle: SignedBundleFixture,
) -> None:
    authorization = verify_gate_bundle(signed_bundle.bundle, now=NOW)
    first = aggregate_rows(mock_rows(), authorization)
    second = aggregate_rows(mock_rows(), authorization)
    assert first == second
    assert first["paired_cells"] == 16
    assert first["episodes"] == 32
    assert first["arms"][COMPACT_RAW_ARM]["task_success"]["rate"] == 0.75
    assert first["paired_compact_raw_minus_native_absolute"]["estimate"] == 0.0
    assert first["noninferiority"]["pass"] is True


def test_parse_and_executor_failure_is_reported_separately_and_fails_gate(
    signed_bundle: SignedBundleFixture,
) -> None:
    authorization = verify_gate_bundle(signed_bundle.bundle, now=NOW)
    result = aggregate_rows(mock_rows(compact_parse_failure=True), authorization)
    compact = result["arms"][COMPACT_RAW_ARM]
    assert compact["task_success"]["rate"] == 0.75
    assert compact["parse_executor_failure_rate"] == 1 / 16
    assert result["noninferiority"]["criteria"][
        "compact_raw_parse_executor_failure_at_most_ceiling"
    ] is False
    assert result["noninferiority"]["pass"] is False


def test_incomplete_pair_grid_refuses(signed_bundle: SignedBundleFixture) -> None:
    authorization = verify_gate_bundle(signed_bundle.bundle, now=NOW)
    with pytest.raises(AggregateError, match="expected 32"):
        aggregate_rows(mock_rows()[:-1], authorization)


def test_infrastructure_failure_refuses_instead_of_counting_failure(
    signed_bundle: SignedBundleFixture,
) -> None:
    authorization = verify_gate_bundle(signed_bundle.bundle, now=NOW)
    with pytest.raises(AggregateError, match="infrastructure-invalid"):
        aggregate_rows(mock_rows(infrastructure_failure=True), authorization)


def test_trace_schema_rejects_instruction_or_raw_output(
    signed_bundle: SignedBundleFixture,
) -> None:
    authorization = verify_gate_bundle(signed_bundle.bundle, now=NOW)
    rows = mock_rows()
    rows[0]["instruction"] = "mock content that must never be accepted"
    with pytest.raises(RecordError, match="sanitized schema"):
        aggregate_rows(rows, authorization)


def test_aggregate_does_not_open_rows_until_both_gates_pass(tmp_path: Path) -> None:
    called = False

    def poison_rows() -> list[object]:
        nonlocal called
        called = True
        raise AssertionError("rows became reachable before authorization")

    missing = GateBundle(
        prerequisites=SignedGatePaths(tmp_path / "missing-a", tmp_path / "missing-a.sig"),
        pilot_release=SignedGatePaths(tmp_path / "missing-b", tmp_path / "missing-b.sig"),
        allowed_signers=tmp_path / "missing-signers",
        signer_identity="proper-vm-roadmap-release",
    )
    with pytest.raises(GateError):
        aggregate_authorized(missing, poison_rows, now=NOW)
    assert called is False


def test_aggregate_cli_reverifies_signatures(
    signed_bundle: SignedBundleFixture, tmp_path: Path
) -> None:
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in mock_rows()),
        encoding="utf-8",
    )
    output = tmp_path / "aggregate.json"
    bundle = signed_bundle.bundle
    status = main(
        [
            "--prerequisites-gate",
            str(bundle.prerequisites.payload),
            "--prerequisites-signature",
            str(bundle.prerequisites.signature),
            "--pilot-release-gate",
            str(bundle.pilot_release.payload),
            "--pilot-release-signature",
            str(bundle.pilot_release.signature),
            "--allowed-signers",
            str(bundle.allowed_signers),
            "--signer-identity",
            bundle.signer_identity,
            "--rows",
            str(rows_path),
            "--output",
            str(output),
        ]
    )
    assert status == 0
    assert json.loads(output.read_text(encoding="utf-8"))["noninferiority"]["pass"]
