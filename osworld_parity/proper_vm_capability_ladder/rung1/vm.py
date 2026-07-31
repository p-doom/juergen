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
    prior_provider_transition_index: int
    new_provider_transition_index: int
    provider_transition_labels: tuple[str, ...]
    provider_transition_records_sha256: str
    guest_sentinel_path_sha256: str
    guest_sentinel_nonce_sha256: str
    attestor_mac: str
    receipt_sha256: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _provider_state_value(value: object) -> object:
    """Copy provider state into an immutable, stable JSON value.

    The production provider returns its live internal dictionary directly, so
    retaining that object would let an in-place mutation rewrite the alleged
    pre-reset observation.  This conversion captures only stable external
    identity/state fields and owns every returned container.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}
    if isinstance(value, dict):
        return {
            str(key): _provider_state_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_provider_state_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        rows = [_provider_state_value(item) for item in value]
        return sorted(rows, key=lambda item: _canonical_json(item))
    if callable(getattr(value, "poll", None)) and hasattr(value, "pid"):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "pid": int(value.pid),
            "returncode": value.poll(),
        }
    if isinstance(getattr(value, "path", None), str):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "path": value.path,
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _provider_timings(provider: object) -> tuple[tuple[str, float], ...]:
    raw = getattr(provider, "timings", None)
    if not isinstance(raw, list):
        raise VmHarnessError("provider exposes no native transition telemetry")
    rows: list[tuple[str, float]] = []
    for item in raw:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], (int, float))
            or float(item[1]) < 0
        ):
            raise VmHarnessError("provider transition telemetry schema drift")
        rows.append((item[0], float(item[1])))
    return tuple(rows)


def _provider_observation(provider: object, qcow: Path) -> tuple[bytes, tuple[tuple[str, float], ...]]:
    state = _provider_state_value(provider.state(str(qcow)))
    timings = _provider_timings(provider)
    return _canonical_json({"state": state, "timings": timings}), timings


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
        expected_provider_sha256: str | None = None,
    ) -> None:
        self.qcow = qcow.resolve()
        self.qemu = qemu.resolve()
        self.provider_path = provider_path.resolve()
        self.vm_log_dir = vm_log_dir.resolve()
        self.smp = smp
        self.memory = memory
        self.expected_provider_sha256 = expected_provider_sha256
        self.provider = None
        self.transport: HttpVmTransport | None = None
        self._module: ModuleType | None = None
        self._provider_session_id = uuid.uuid4().hex
        self._provider_reset_sequence = 0
        self._provider_attestor_secret = secrets.token_bytes(32)
        self._consumed_provider_reset_receipts: set[str] = set()
        self._last_consumed_provider_sequence = 0
        self._last_consumed_provider_generation_id: str | None = None
        self._last_consumed_provider_transition_index: int | None = None
        self._outstanding_provider_reset_receipt_sha256: str | None = None

    def start(self) -> None:
        if not os.access("/dev/kvm", os.R_OK | os.W_OK):
            raise VmHarnessError("/dev/kvm is not readable and writable")
        for path in (self.qcow, self.qemu, self.provider_path):
            if not path.exists():
                raise VmHarnessError(f"required VM input missing: {path}")
        if not os.access(self.qemu, os.X_OK):
            raise VmHarnessError(f"qemu is not executable: {self.qemu}")
        if self.expected_provider_sha256 is not None:
            expected = self.expected_provider_sha256
            if len(expected) != 64 or any(
                char not in "0123456789abcdef" for char in expected
            ):
                raise VmHarnessError("expected provider SHA-256 is invalid")
            observed = sha256_file(self.provider_path)
            if observed != expected:
                raise VmHarnessError(
                    f"provider SHA-256 mismatch: {observed} != {expected}"
                )
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
        reset_sequence = self._provider_reset_sequence + 1
        sentinel_path = (
            f"/tmp/osworld_reset_attestation_{self._provider_session_id}_"
            f"{reset_sequence}.nonce"
        )
        sentinel_nonce = secrets.token_hex(32)
        started = time.monotonic_ns()
        self._plant_reset_sentinel(sentinel_path, sentinel_nonce)
        before_observation, before_timings = _provider_observation(
            self.provider, self.qcow
        )
        self.provider.load_state(str(self.qcow), READY_SNAPSHOT)
        after_observation, after_timings = _provider_observation(
            self.provider, self.qcow
        )
        if after_timings[: len(before_timings)] != before_timings:
            raise VmHarnessError("provider transition telemetry rewrote prior history")
        appended = after_timings[len(before_timings) :]
        expected_labels = (f"loadvm[{READY_SNAPSHOT}]", "loadvm_guest_ready")
        if len(appended) < len(expected_labels) or tuple(
            row[0] for row in appended[: len(expected_labels)]
        ) != expected_labels:
            raise VmHarnessError(
                "provider reset produced no native loadvm transition evidence"
            )
        if before_observation == after_observation:
            raise VmHarnessError("provider reset observation did not change")
        prior_generation_id = hashlib.sha256(before_observation).hexdigest()
        new_generation_id = hashlib.sha256(after_observation).hexdigest()
        if prior_generation_id == new_generation_id:
            raise VmHarnessError("provider reset generation did not advance")
        after = self.provider.state(str(self.qcow))
        port = int(after["ports"]["server"])
        # Sessions and host-side input audits must never cross an episode reset.
        self.transport = HttpVmTransport(f"http://127.0.0.1:{port}")
        self._verify_reset_sentinel_removed(sentinel_path)
        completed = time.monotonic_ns()
        self._provider_reset_sequence = reset_sequence
        transition_payload = [list(row) for row in appended]
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
                before_observation
            ).hexdigest(),
            "provider_state_after_sha256": hashlib.sha256(
                after_observation
            ).hexdigest(),
            "provider_path_sha256": (
                sha256_file(self.provider_path)
                if self.provider_path.is_file()
                else hashlib.sha256(
                    str(self.provider_path).encode("utf-8")
                ).hexdigest()
            ),
            "prior_provider_transition_index": len(before_timings),
            "new_provider_transition_index": len(after_timings),
            "provider_transition_labels": tuple(row[0] for row in appended),
            "provider_transition_records_sha256": hashlib.sha256(
                _canonical_json(transition_payload)
            ).hexdigest(),
            "guest_sentinel_path_sha256": hashlib.sha256(
                sentinel_path.encode("utf-8")
            ).hexdigest(),
            "guest_sentinel_nonce_sha256": hashlib.sha256(
                sentinel_nonce.encode("utf-8")
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
        self._outstanding_provider_reset_receipt_sha256 = receipt.receipt_sha256
        return self.transport, receipt

    def _plant_reset_sentinel(self, path: str, nonce: str) -> None:
        if self.transport is None:
            raise VmHarnessError("VM transport is not started")
        code = (
            "from pathlib import Path;"
            f"p=Path({path!r});v={nonce!r};"
            "p.write_text(v,encoding='utf-8');"
            "assert p.read_text(encoding='utf-8')==v"
        )
        try:
            self.transport.execute_argv(["python3", "-c", code])
        except Exception as exc:
            raise VmHarnessError(f"could not plant pre-reset guest sentinel: {exc}") from exc

    def _verify_reset_sentinel_removed(self, path: str) -> None:
        if self.transport is None:
            raise VmHarnessError("VM transport is not started")
        code = f"from pathlib import Path;assert not Path({path!r}).exists()"
        try:
            self.transport.execute_argv(["python3", "-c", code])
        except Exception as exc:
            raise VmHarnessError(
                "provider reset did not rewind the pre-reset guest sentinel"
            ) from exc

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
            receipt.prior_generation_id,
            receipt.new_generation_id,
            receipt.provider_state_before_sha256,
            receipt.provider_state_after_sha256,
            receipt.provider_path_sha256,
            receipt.provider_transition_records_sha256,
            receipt.guest_sentinel_path_sha256,
            receipt.guest_sentinel_nonce_sha256,
            receipt.attestor_mac,
            receipt.receipt_sha256,
        ):
            if len(value) != 64 or any(char not in lowercase_hex for char in value):
                raise VmHarnessError("provider reset receipt hash contract drift")
        if receipt.receipt_sha256 in self._consumed_provider_reset_receipts:
            raise VmHarnessError("provider reset receipt replay detected")
        if receipt.receipt_sha256 != self._outstanding_provider_reset_receipt_sha256:
            raise VmHarnessError("provider reset receipt is not the active transition")
        if receipt.reset_sequence != self._last_consumed_provider_sequence + 1:
            raise VmHarnessError("provider reset transition is out of order")
        if self._last_consumed_provider_generation_id is not None and (
            receipt.prior_generation_id
            != self._last_consumed_provider_generation_id
            or receipt.prior_provider_transition_index
            != self._last_consumed_provider_transition_index
        ):
            raise VmHarnessError("provider reset generation chain is out of order")
        if (
            receipt.provider_state_before_sha256 != receipt.prior_generation_id
            or receipt.provider_state_after_sha256 != receipt.new_generation_id
            or receipt.new_provider_transition_index
            <= receipt.prior_provider_transition_index
            or receipt.new_provider_transition_index
            - receipt.prior_provider_transition_index
            != len(receipt.provider_transition_labels)
            or tuple(receipt.provider_transition_labels[:2])
            != (f"loadvm[{READY_SNAPSHOT}]", "loadvm_guest_ready")
        ):
            raise VmHarnessError("provider-native reset transition contract drift")
        if self.provider is None:
            raise VmHarnessError("VM session is not started")
        current_observation, current_timings = _provider_observation(
            self.provider, self.qcow
        )
        if (
            hashlib.sha256(current_observation).hexdigest()
            != receipt.new_generation_id
            or len(current_timings) != receipt.new_provider_transition_index
        ):
            raise VmHarnessError("provider reset receipt is no longer current")
        transition_start = receipt.prior_provider_transition_index
        transition_end = receipt.new_provider_transition_index
        current_transition = current_timings[transition_start:transition_end]
        if (
            tuple(row[0] for row in current_transition)
            != receipt.provider_transition_labels
            or hashlib.sha256(
                _canonical_json([list(row) for row in current_transition])
            ).hexdigest()
            != receipt.provider_transition_records_sha256
        ):
            raise VmHarnessError("provider reset transition receipt mismatch")
        self._consumed_provider_reset_receipts.add(receipt.receipt_sha256)
        self._last_consumed_provider_sequence = receipt.reset_sequence
        self._last_consumed_provider_generation_id = receipt.new_generation_id
        self._last_consumed_provider_transition_index = (
            receipt.new_provider_transition_index
        )
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
