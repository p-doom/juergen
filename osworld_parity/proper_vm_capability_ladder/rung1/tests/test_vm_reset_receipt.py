from __future__ import annotations

from dataclasses import replace

import pytest

from osworld_parity.proper_vm_capability_ladder.rung1.vm import (
    KvmFixtureSession,
    READY_SNAPSHOT,
    VmHarnessError,
)


class _FakeProvider:
    def __init__(self) -> None:
        self.generation = 0
        self.loads: list[tuple[str, str]] = []

    def state(self, qcow: str) -> dict[str, object]:
        return {
            "qcow": qcow,
            "generation": self.generation,
            "ports": {"server": 8765},
        }

    def load_state(self, qcow: str, snapshot: str) -> None:
        self.loads.append((qcow, snapshot))
        self.generation += 1


def test_provider_receipt_attests_actual_reset_and_generation_chain(tmp_path) -> None:
    session = KvmFixtureSession(
        qcow=tmp_path / "vm.qcow2",
        qemu=tmp_path / "qemu",
        provider_path=tmp_path / "provider.py",
        vm_log_dir=tmp_path / "logs",
    )
    provider = _FakeProvider()
    session.provider = provider

    _, first = session.reset_to_ready_with_receipt()
    assert provider.loads == [(str(session.qcow), READY_SNAPSHOT)]
    assert first.snapshot_id == READY_SNAPSHOT
    assert first.prior_generation_id != first.new_generation_id
    assert first.provider_state_before_sha256 != first.provider_state_after_sha256
    assert first.reset_started_monotonic_ns < first.reset_completed_monotonic_ns
    with pytest.raises(VmHarnessError, match="must be consumed"):
        session.reset_to_ready_with_receipt()
    session.consume_provider_reset_receipt(first)

    with pytest.raises(VmHarnessError, match="replay"):
        session.consume_provider_reset_receipt(first)

    _, second = session.reset_to_ready_with_receipt()
    assert second.reset_sequence == first.reset_sequence + 1
    assert second.prior_generation_id == first.new_generation_id
    session.consume_provider_reset_receipt(second)

    with pytest.raises(VmHarnessError, match="attestation mismatch"):
        session.consume_provider_reset_receipt(
            replace(second, new_generation_id="0" * 32)
        )
