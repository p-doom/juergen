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
        self.loads: list[tuple[str, str]] = []
        self.timings: list[tuple[str, float]] = []
        self.live_state = {
            "ports": {"server": 8765},
            "snapshots": {READY_SNAPSHOT},
        }
        self.on_load = lambda: None

    def state(self, qcow: str) -> dict[str, object]:
        self.live_state["qcow"] = qcow
        return self.live_state

    def load_state(self, qcow: str, snapshot: str) -> None:
        self.loads.append((qcow, snapshot))
        self.timings.extend(
            ((f"loadvm[{snapshot}]", 0.01), ("loadvm_guest_ready", 0.02))
        )
        self.on_load()


class _NoTransitionProvider(_FakeProvider):
    def load_state(self, qcow: str, snapshot: str) -> None:
        self.loads.append((qcow, snapshot))


def _wire_guest_sentinel(session: KvmFixtureSession, provider: _FakeProvider) -> None:
    guest_files: set[str] = set()
    session._plant_reset_sentinel = (  # type: ignore[method-assign]
        lambda path, _nonce: guest_files.add(path)
    )

    def verify(path: str) -> None:
        if path in guest_files:
            raise VmHarnessError(
                "provider reset did not rewind the pre-reset guest sentinel"
            )

    session._verify_reset_sentinel_removed = verify  # type: ignore[method-assign]
    provider.on_load = guest_files.clear


def test_provider_receipt_attests_actual_reset_and_generation_chain(tmp_path) -> None:
    session = KvmFixtureSession(
        qcow=tmp_path / "vm.qcow2",
        qemu=tmp_path / "qemu",
        provider_path=tmp_path / "provider.py",
        vm_log_dir=tmp_path / "logs",
    )
    provider = _FakeProvider()
    session.provider = provider
    _wire_guest_sentinel(session, provider)

    _, first = session.reset_to_ready_with_receipt()
    assert provider.loads == [(str(session.qcow), READY_SNAPSHOT)]
    assert first.snapshot_id == READY_SNAPSHOT
    assert first.prior_generation_id != first.new_generation_id
    assert first.provider_state_before_sha256 != first.provider_state_after_sha256
    assert first.prior_provider_transition_index == 0
    assert first.new_provider_transition_index == 2
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


def test_provider_reset_rejects_equal_state_without_native_transition(tmp_path) -> None:
    session = KvmFixtureSession(
        qcow=tmp_path / "vm.qcow2",
        qemu=tmp_path / "qemu",
        provider_path=tmp_path / "provider.py",
        vm_log_dir=tmp_path / "logs",
    )
    session.provider = _NoTransitionProvider()
    _wire_guest_sentinel(session, session.provider)
    with pytest.raises(VmHarnessError, match="no native loadvm transition"):
        session.reset_to_ready_with_receipt()


def test_provider_telemetry_without_guest_rewind_is_rejected(tmp_path) -> None:
    session = KvmFixtureSession(
        qcow=tmp_path / "vm.qcow2",
        qemu=tmp_path / "qemu",
        provider_path=tmp_path / "provider.py",
        vm_log_dir=tmp_path / "logs",
    )
    provider = _FakeProvider()
    session.provider = provider
    _wire_guest_sentinel(session, provider)
    provider.on_load = lambda: None
    with pytest.raises(VmHarnessError, match="did not rewind"):
        session.reset_to_ready_with_receipt()
