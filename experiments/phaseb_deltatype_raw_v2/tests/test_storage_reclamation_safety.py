from __future__ import annotations

import pytest

import storage_emergency_reclaim_intermediate_payloads as reclaim


@pytest.mark.parametrize("state", ["RUNNING", "SUSPENDED"])
def test_active_job_must_retain_a_durable_resumable_payload(state: str):
    with pytest.raises(
        reclaim.ReclamationSafetyError,
        match="active job would have no durable resumable checkpoint",
    ):
        reclaim.validate_reclamation_safety(
            active_job_state=state,
            durable_resumable_payloads_after=0,
            downstream_references=[],
        )


def test_downstream_export_reference_blocks_last_payload_removal():
    with pytest.raises(
        reclaim.ReclamationSafetyError,
        match="downstream export references checkpoint payloads",
    ):
        reclaim.validate_reclamation_safety(
            active_job_state="COMPLETED",
            durable_resumable_payloads_after=0,
            downstream_references=["run_held_export/context.json:checkpoint"],
        )


def test_cleanup_can_only_proceed_with_an_independent_durable_payload():
    reclaim.validate_reclamation_safety(
        active_job_state="RUNNING",
        durable_resumable_payloads_after=1,
        downstream_references=["run_held_export/context.json:checkpoint"],
    )
