from __future__ import annotations

from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.vm import (
    AppReadinessError,
)


def test_readiness_failure_is_phase_specific_and_preserves_evidence() -> None:
    error = AppReadinessError(
        fixture_id="fixture",
        failed_phase="browser_document_ready",
        evidence={"last_error": "no event", "diagnostics": {"chrome": "running"}},
    )
    assert error.failed_phase == "browser_document_ready"
    assert error.evidence["diagnostics"]["chrome"] == "running"
    assert "readiness failed at browser_document_ready" in str(error)
