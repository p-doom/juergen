from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import os
import secrets
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

from .fixtures import Fixture
from .server import FixtureHttpServer
from .transport import HttpVmTransport


DEFAULT_QCOW = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/osworld_vm/Ubuntu.qcow2"
)
DEFAULT_QEMU = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/qemu/bin/"
    "qemu-system-x86_64-wrapped"
)
DEFAULT_PROVIDER = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/osworld_fastreset/"
    "qemu_fast_reset.py"
)
READY_SNAPSHOT = "osworld_ready"


class VmHarnessError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderResetReceipt:
    provider_session_id: str
    reset_id: str
    reset_sequence: int
    prior_generation_id: str
    new_generation_id: str
    snapshot_id: str
    reset_started_monotonic_ns: int
    reset_completed_monotonic_ns: int
    provider_state_before_sha256: str
    provider_state_after_sha256: str
    provider_path_sha256: str
    attestor_mac: str
    receipt_sha256: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_provider(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("rung1a_qemu_fast_reset", path)
    if spec is None or spec.loader is None:
        raise VmHarnessError(f"cannot load KVM provider: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KvmFixtureSession:
    """One KVM VM, explicitly restored to ``osworld_ready`` per episode."""

    def __init__(
        self,
        *,
        qcow: Path = DEFAULT_QCOW,
        qemu: Path = DEFAULT_QEMU,
        provider_path: Path = DEFAULT_PROVIDER,
        vm_log_dir: Path = Path("/tmp/rung1a_vm_logs"),
        smp: int = 4,
        memory: str = "8G",
    ) -> None:
        self.qcow = qcow.resolve()
        self.qemu = qemu.resolve()
        self.provider_path = provider_path.resolve()
        self.vm_log_dir = vm_log_dir.resolve()
        self.smp = smp
        self.memory = memory
        self.provider = None
        self.transport: HttpVmTransport | None = None
        self._module: ModuleType | None = None
        self._provider_session_id = uuid.uuid4().hex
        self._provider_generation_id = uuid.uuid4().hex
        self._provider_reset_sequence = 0
        self._provider_attestor_secret = secrets.token_bytes(32)
        self._consumed_provider_reset_receipts: set[str] = set()
        self._last_consumed_provider_sequence = 0
        self._last_consumed_provider_generation_id = self._provider_generation_id
        self._outstanding_provider_reset_receipt_sha256: str | None = None

    def start(self) -> None:
        if not os.access("/dev/kvm", os.R_OK | os.W_OK):
            raise VmHarnessError("/dev/kvm is not readable and writable")
        for path in (self.qcow, self.qemu, self.provider_path):
            if not path.exists():
                raise VmHarnessError(f"required VM input missing: {path}")
        if not os.access(self.qemu, os.X_OK):
            raise VmHarnessError(f"qemu is not executable: {self.qemu}")
        self.vm_log_dir.mkdir(parents=True, exist_ok=True)
        os.environ["OSWORLD_QCOW2"] = str(self.qcow)
        os.environ["OSWORLD_QEMU_BIN"] = str(self.qemu)
        os.environ["OSWORLD_FAST_SNAPSHOT"] = "1"
        os.environ["OSWORLD_VM_LOG_DIR"] = str(self.vm_log_dir)
        os.environ["OSWORLD_QMP_DIR"] = "/tmp"
        os.environ["OSWORLD_VM_SMP"] = str(self.smp)
        os.environ["OSWORLD_VM_MEM"] = self.memory
        module = _load_provider(self.provider_path)
        # The provider's ordinary boot snapshot is semantically the pinned clean
        # QCOW state. Name it exactly as preregistered before the VM is started.
        module.BOOT_SNAPSHOT = READY_SNAPSHOT
        provider = module.FastResetKvmProvider()
        provider.start_emulator(str(self.qcow))
        if not provider.has_snapshot(str(self.qcow), READY_SNAPSHOT):
            provider.stop_emulator(str(self.qcow))
            raise VmHarnessError("provider did not create osworld_ready snapshot")
        port = int(provider.state(str(self.qcow))["ports"]["server"])
        self.provider = provider
        self.transport = HttpVmTransport(f"http://127.0.0.1:{port}")
        self._module = module

    def reset_to_ready(self) -> HttpVmTransport:
        transport, receipt = self.reset_to_ready_with_receipt()
        self.consume_provider_reset_receipt(receipt)
        return transport

    def reset_to_ready_with_receipt(
        self,
    ) -> tuple[HttpVmTransport, ProviderResetReceipt]:
        if self.provider is None:
            raise VmHarnessError("VM session is not started")
        if self._outstanding_provider_reset_receipt_sha256 is not None:
            raise VmHarnessError(
                "previous provider reset receipt must be consumed before another reset"
            )
        before = self.provider.state(str(self.qcow))
        started = time.monotonic_ns()
        self.provider.load_state(str(self.qcow), READY_SNAPSHOT)
        completed = time.monotonic_ns()
        after = self.provider.state(str(self.qcow))
        port = int(after["ports"]["server"])
        # Sessions and host-side input audits must never cross an episode reset.
        self.transport = HttpVmTransport(f"http://127.0.0.1:{port}")
        self._provider_reset_sequence += 1
        prior_generation_id = self._provider_generation_id
        new_generation_id = uuid.uuid4().hex
        payload = {
            "provider_session_id": self._provider_session_id,
            "reset_id": uuid.uuid4().hex,
            "reset_sequence": self._provider_reset_sequence,
            "prior_generation_id": prior_generation_id,
            "new_generation_id": new_generation_id,
            "snapshot_id": READY_SNAPSHOT,
            "reset_started_monotonic_ns": started,
            "reset_completed_monotonic_ns": completed,
            "provider_state_before_sha256": hashlib.sha256(
                _canonical_json(before)
            ).hexdigest(),
            "provider_state_after_sha256": hashlib.sha256(
                _canonical_json(after)
            ).hexdigest(),
            "provider_path_sha256": hashlib.sha256(
                str(self.provider_path).encode("utf-8")
            ).hexdigest(),
        }
        attestor_mac = hmac.new(
            self._provider_attestor_secret,
            _canonical_json(payload),
            hashlib.sha256,
        ).hexdigest()
        receipt_sha256 = hashlib.sha256(
            _canonical_json({**payload, "attestor_mac": attestor_mac})
        ).hexdigest()
        receipt = ProviderResetReceipt(
            **payload,
            attestor_mac=attestor_mac,
            receipt_sha256=receipt_sha256,
        )
        self._provider_generation_id = new_generation_id
        self._outstanding_provider_reset_receipt_sha256 = receipt.receipt_sha256
        return self.transport, receipt

    def consume_provider_reset_receipt(self, receipt: ProviderResetReceipt) -> None:
        if not isinstance(receipt, ProviderResetReceipt):
            raise VmHarnessError("provider reset receipt type mismatch")
        payload = asdict(receipt)
        receipt_sha256 = payload.pop("receipt_sha256")
        attestor_mac = payload.pop("attestor_mac")
        expected_mac = hmac.new(
            self._provider_attestor_secret,
            _canonical_json(payload),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(attestor_mac, expected_mac):
            raise VmHarnessError("provider reset receipt attestation mismatch")
        observed = hashlib.sha256(
            _canonical_json({**payload, "attestor_mac": attestor_mac})
        ).hexdigest()
        if observed != receipt_sha256:
            raise VmHarnessError("provider reset receipt was mutated")
        if receipt.provider_session_id != self._provider_session_id:
            raise VmHarnessError("provider reset receipt belongs to another session")
        if receipt.snapshot_id != READY_SNAPSHOT:
            raise VmHarnessError("provider reset receipt snapshot drift")
        if (
            receipt.reset_started_monotonic_ns < 1
            or receipt.reset_completed_monotonic_ns
            <= receipt.reset_started_monotonic_ns
            or not receipt.reset_id
            or not receipt.prior_generation_id
            or not receipt.new_generation_id
            or receipt.prior_generation_id == receipt.new_generation_id
        ):
            raise VmHarnessError("provider reset receipt field contract drift")
        lowercase_hex = set("0123456789abcdef")
        for value in (
            receipt.provider_state_before_sha256,
            receipt.provider_state_after_sha256,
            receipt.provider_path_sha256,
            receipt.attestor_mac,
            receipt.receipt_sha256,
        ):
            if len(value) != 64 or any(char not in lowercase_hex for char in value):
                raise VmHarnessError("provider reset receipt hash contract drift")
        if receipt.receipt_sha256 in self._consumed_provider_reset_receipts:
            raise VmHarnessError("provider reset receipt replay detected")
        if receipt.receipt_sha256 != self._outstanding_provider_reset_receipt_sha256:
            raise VmHarnessError("provider reset receipt is not the active transition")
        if receipt.reset_sequence != self._last_consumed_provider_sequence + 1 or (
            receipt.prior_generation_id
            != self._last_consumed_provider_generation_id
        ):
            raise VmHarnessError("provider reset transition is out of order")
        self._consumed_provider_reset_receipts.add(receipt.receipt_sha256)
        self._last_consumed_provider_sequence = receipt.reset_sequence
        self._last_consumed_provider_generation_id = receipt.new_generation_id
        self._outstanding_provider_reset_receipt_sha256 = None

    def launch_fixture(
        self,
        fixture_server: FixtureHttpServer,
        fixture: Fixture,
        *,
        timeout_s: float = 60.0,
    ) -> dict:
        if self.transport is None:
            raise VmHarnessError("VM session is not started")
        fixture_server.store.reset(fixture)
        url = fixture_server.guest_url(fixture)
        script = f"""
set -euo pipefail
browser="$(command -v google-chrome || command -v chromium || command -v chromium-browser)"
test -n "$browser"
nohup "$browser" --no-first-run --no-default-browser-check \
  --disable-session-crashed-bubble --disable-features=TranslateUI \
  --start-maximized {url!r} >/tmp/rung1a_chrome.log 2>&1 </dev/null &
""".strip()
        self.transport.execute_argv(["bash", "-lc", script])
        return fixture_server.store.wait_ready(fixture.id, timeout_s=timeout_s)

    def probe_pointer_buttons(
        self, fixture_server: FixtureHttpServer, fixture: Fixture
    ) -> int:
        if self.transport is None:
            raise VmHarnessError("VM session is not started")
        x, y = self.transport.cursor_position()
        self.transport.move_to(x + 2, y + 1)
        time.sleep(0.25)
        return int(fixture_server.store.snapshot(fixture.id)["last_pointer_buttons"])

    def close(self) -> None:
        if self.provider is not None:
            self.provider.stop_emulator(str(self.qcow))
            self.provider = None
        self.transport = None

    def __enter__(self) -> "KvmFixtureSession":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
