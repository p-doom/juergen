from __future__ import annotations

import time
import uuid
from dataclasses import replace

from osworld_parity.proper_vm_capability_ladder.rung1.vm import ProviderResetReceipt
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.runtime import (
    RuntimeEvidenceLedger,
    RuntimeProbe,
)


class FakeProviderResetAttestor:
    """Independent test authority mirroring the KVM provider transition API."""

    def __init__(self) -> None:
        self.session_id = uuid.uuid4().hex
        self.generation_id = uuid.uuid4().hex
        self.sequence = 0
        self.issued: dict[str, ProviderResetReceipt] = {}
        self.consumed: set[str] = set()

    def issue(self) -> ProviderResetReceipt:
        started = time.monotonic_ns()
        completed = time.monotonic_ns()
        self.sequence += 1
        prior = self.generation_id
        self.generation_id = uuid.uuid4().hex
        receipt = ProviderResetReceipt(
            provider_session_id=self.session_id,
            reset_id=uuid.uuid4().hex,
            reset_sequence=self.sequence,
            prior_generation_id=prior,
            new_generation_id=self.generation_id,
            snapshot_id="osworld_ready",
            reset_started_monotonic_ns=started,
            reset_completed_monotonic_ns=completed,
            provider_state_before_sha256="1" * 64,
            provider_state_after_sha256="2" * 64,
            provider_path_sha256="3" * 64,
            attestor_mac="4" * 64,
            receipt_sha256=uuid.uuid4().hex + uuid.uuid4().hex,
        )
        self.issued[receipt.receipt_sha256] = receipt
        return receipt

    def consume_provider_reset_receipt(self, receipt: ProviderResetReceipt) -> None:
        if self.issued.get(receipt.receipt_sha256) != receipt:
            raise RuntimeError("unissued provider reset receipt")
        if receipt.receipt_sha256 in self.consumed:
            raise RuntimeError("provider reset receipt replay")
        self.consumed.add(receipt.receipt_sha256)


def make_ledger() -> tuple[RuntimeEvidenceLedger, FakeProviderResetAttestor]:
    attestor = FakeProviderResetAttestor()
    return (
        RuntimeEvidenceLedger(
            setup_commit="a" * 40,
            reset_provider="test",
            reset_attestor=attestor,
        ),
        attestor,
    )


def attribute_new_observation(
    task,
    ledger: RuntimeEvidenceLedger,
    attestor: FakeProviderResetAttestor,
    probe: RuntimeProbe,
    *,
    endpoint: str,
) -> RuntimeProbe:
    receipt = attestor.issue()
    observation = replace(
        probe,
        observation_id=uuid.uuid4().hex,
        observed_monotonic_ns=time.monotonic_ns(),
        reset_cycle_evidence=None,
        refresh_evidence=None,
    )
    return ledger.issue_reset_probe(
        task,
        observation,
        provider_reset_receipt=receipt,
        transport_endpoint=endpoint,
    )
