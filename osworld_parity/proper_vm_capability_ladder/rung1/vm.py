from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import os
import re
import secrets
import shutil
import socket
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import fcntl

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


_VM_ID_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_component(value: str, *, fallback: str) -> str:
    cleaned = _VM_ID_COMPONENT.sub("-", value).strip("-.")
    return (cleaned or fallback)[:40]


def task_unique_vm_id() -> str:
    """Return an auditable id that is unique across jobs, tasks, and processes."""

    job = _safe_component(
        os.environ.get("SLURM_JOB_ID", os.environ.get("LABCTL_JOB_ID", "local")),
        fallback="local",
    )
    task = _safe_component(os.environ.get("SLURM_PROCID", "0"), fallback="0")
    run = _safe_component(os.environ.get("LABCTL_RUN_ID", "no-run"), fallback="no-run")
    return f"{job}-{task}-{run[:16]}-{os.getpid()}-{uuid.uuid4().hex[:10]}"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(raw, path)
    finally:
        Path(raw).unlink(missing_ok=True)


@contextmanager
def _node_port_allocation_lock() -> Iterator[None]:
    """Serialize bind(0)->QEMU handoff across cooperating processes on a node."""

    # The provider retries a lost bind race, but serializing allocation through
    # QEMU readiness closes that race between all certification jobs on a node.
    path = Path(f"/tmp/proper-vm-port-allocation-{os.getuid()}.lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


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
        scratch_root: Path | None = None,
        vm_id: str | None = None,
    ) -> None:
        self.qcow = qcow.resolve()
        self.qemu = qemu.resolve()
        self.provider_path = provider_path.resolve()
        self.vm_log_dir = vm_log_dir.resolve()
        self.smp = smp
        self.memory = memory
        self.expected_provider_sha256 = expected_provider_sha256
        self.scratch_root = scratch_root
        self.vm_id = vm_id or task_unique_vm_id()
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
        self._task_lock_handle: Any | None = None
        self._saved_environment: dict[str, str | None] = {}
        self.vm_scratch_dir: Path | None = None
        self.scratch_source: str | None = None
        self._scratch_root: Path | None = None
        self._scratch_root_owned = False
        self._task_lock_path: Path | None = None
        self.metadata_path = self.vm_log_dir.parent / "vm_metadata.json"

    def _set_environment(self, key: str, value: str) -> None:
        if key not in self._saved_environment:
            self._saved_environment[key] = os.environ.get(key)
        os.environ[key] = value

    def _restore_environment(self) -> None:
        for key, value in self._saved_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._saved_environment.clear()

    def _prepare_isolation(self) -> None:
        if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
            raise VmHarnessError("GPU visibility is forbidden for CPU KVM certification")
        ntasks = os.environ.get("SLURM_NTASKS")
        if ntasks is not None and ntasks != "1":
            raise VmHarnessError(f"exactly one SLURM task is required, got {ntasks}")
        root = self.scratch_root
        if root is None:
            raw = os.environ.get("SLURM_TMPDIR")
            if raw:
                root = Path(raw)
                self.scratch_source = "SLURM_TMPDIR"
            else:
                job = _safe_component(
                    os.environ.get("SLURM_JOB_ID", os.environ.get("LABCTL_JOB_ID", "local")),
                    fallback="local",
                )
                task = _safe_component(os.environ.get("SLURM_PROCID", "0"), fallback="0")
                root = Path("/tmp") / f"proper-vm-job-{os.getuid()}-{job}-{task}"
                self.scratch_source = "job_unique_tmp_fallback"
                self._scratch_root_owned = True
        else:
            self.scratch_source = "explicit"
        root = root.resolve()
        slurm_tmp = os.environ.get("SLURM_TMPDIR")
        if slurm_tmp and not root.is_relative_to(Path(slurm_tmp).resolve()):
            raise VmHarnessError("VM scratch root must be below SLURM_TMPDIR")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.stat().st_uid != os.geteuid():
            raise VmHarnessError("VM scratch root is not owned by this job user")
        if self._scratch_root_owned:
            os.chmod(root, 0o700)
        self._scratch_root = root
        self.vm_scratch_dir = root / f"proper-vm-{self.vm_id}"
        self.vm_scratch_dir.mkdir(mode=0o700, parents=False, exist_ok=False)

        # A task may own at most one live VM.  The lock is intentionally scoped
        # to the SLURM scratch root so separate array tasks do not share state.
        lock_path = root / f"proper-vm-task-{os.environ.get('SLURM_PROCID', '0')}.lock"
        self._task_lock_path = lock_path
        handle = lock_path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise VmHarnessError("another VM is already live in this SLURM task") from exc
        self._task_lock_handle = handle

        # QEMU -snapshot creates and immediately unlinks a temporary qcow
        # overlay.  TMPDIR is the only reliable placement boundary for that
        # file; keeping it in job-unique scratch makes even SIGKILL cleanup local
        # to the allocation. QMP remains under /tmp to stay far below sockaddr_un
        # path limits.
        self._set_environment("TMPDIR", str(self.vm_scratch_dir))
        self._set_environment("OSWORLD_QMP_DIR", "/tmp")

    def _overlay_fd_targets(self, proc: Any) -> list[str]:
        pid = getattr(proc, "pid", None)
        if not isinstance(pid, int):
            raise VmHarnessError("provider did not expose the QEMU process id")
        fd_root = Path(f"/proc/{pid}/fd")
        targets: list[str] = []
        for entry in fd_root.iterdir():
            try:
                target = os.readlink(entry)
            except OSError:
                continue
            if self.vm_scratch_dir is not None and target.startswith(
                str(self.vm_scratch_dir) + os.sep
            ):
                targets.append(target)
        if not targets:
            raise VmHarnessError(
                "QEMU temporary qcow overlay is not under the per-VM SLURM scratch directory"
            )
        return sorted(targets)

    def _metadata(self, *, state: dict[str, Any], overlay_fds: list[str]) -> dict[str, Any]:
        ports = state.get("ports")
        if not isinstance(ports, dict) or set(ports) != {
            "server",
            "chromium",
            "vnc",
            "vlc",
        }:
            raise VmHarnessError(f"provider returned invalid port map: {ports!r}")
        parsed_ports = {str(key): int(value) for key, value in ports.items()}
        if len(set(parsed_ports.values())) != len(parsed_ports):
            raise VmHarnessError("provider returned colliding host ports")
        qmp_path = str(state.get("qmp_path", ""))
        if not qmp_path.startswith("/tmp/") or len(qmp_path.encode("utf-8")) >= 80:
            raise VmHarnessError(f"QMP path is not short and /tmp-scoped: {qmp_path!r}")
        log_path = Path(state.get("log", "")).resolve()
        try:
            log_path.relative_to(self.vm_log_dir)
        except ValueError as exc:
            raise VmHarnessError(f"QEMU log escaped its unique output directory: {log_path}") from exc
        return {
            "schema_version": "proper_vm_isolation_v1",
            "vm_id": self.vm_id,
            "hostname": socket.gethostname(),
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "proc_id": os.environ.get("SLURM_PROCID", "0"),
                "node_id": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
                "tmpdir": os.environ.get("SLURM_TMPDIR"),
            },
            "labctl": {
                "run_id": os.environ.get("LABCTL_RUN_ID"),
                "run_dir": os.environ.get("LABCTL_RUN_DIR"),
            },
            "base_qcow": str(self.qcow),
            "overlay": {
                "kind": "qemu_snapshot_temporary_qcow",
                "directory": str(self.vm_scratch_dir),
                "fd_targets": overlay_fds,
                "under_slurm_tmpdir": bool(os.environ.get("SLURM_TMPDIR")),
                "scratch_source": self.scratch_source,
                "job_unique_scratch": self.scratch_source
                in {"SLURM_TMPDIR", "job_unique_tmp_fallback"},
            },
            "qemu": str(self.qemu),
            "provider": str(self.provider_path),
            "qmp_path": qmp_path,
            "ports": parsed_ports,
            "qemu_log": str(log_path),
            "one_vm_per_task": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "closed": False,
        }

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
        self.vm_log_dir.mkdir(parents=True, exist_ok=False)
        try:
            self._prepare_isolation()
            for key, value in {
                "OSWORLD_VM_ID": self.vm_id,
                "OSWORLD_QCOW2": str(self.qcow),
                "OSWORLD_QEMU_BIN": str(self.qemu),
                "OSWORLD_FAST_SNAPSHOT": "1",
                "OSWORLD_VM_LOG_DIR": str(self.vm_log_dir),
                "OSWORLD_VM_SMP": str(self.smp),
                "OSWORLD_VM_MEM": self.memory,
            }.items():
                self._set_environment(key, value)
            module = _load_provider(self.provider_path)
            # The provider's ordinary boot snapshot is semantically the pinned clean
            # QCOW state. Name it exactly as preregistered before the VM is started.
            module.BOOT_SNAPSHOT = READY_SNAPSHOT
            provider = module.FastResetKvmProvider()
            self.provider = provider
            with _node_port_allocation_lock():
                provider.start_emulator(str(self.qcow))
            state = provider.state(str(self.qcow))
            if not provider.has_snapshot(str(self.qcow), READY_SNAPSHOT):
                provider.stop_emulator(str(self.qcow))
                raise VmHarnessError("provider did not create osworld_ready snapshot")
            overlay_fds = self._overlay_fd_targets(state.get("proc"))
            metadata = self._metadata(state=state, overlay_fds=overlay_fds)
            _atomic_json(self.metadata_path, metadata)
            port = int(metadata["ports"]["server"])
            self.transport = HttpVmTransport(f"http://127.0.0.1:{port}")
            self._module = module
        except Exception:
            self.close()
            raise

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
        cleanup_errors: list[str] = []
        provider_stopped = self.provider is None
        if self.provider is not None:
            provider = self.provider
            proc = None
            try:
                proc = provider.state(str(self.qcow)).get("proc")
            except (KeyError, AttributeError):
                pass
            try:
                provider.stop_emulator(str(self.qcow))
                provider_stopped = proc is None or proc.poll() is not None
                if not provider_stopped:
                    cleanup_errors.append("QEMU process remained live after provider stop")
            except Exception as exc:
                cleanup_errors.append(f"provider stop failed: {exc}")
            self.provider = None
        self.transport = None
        scratch_path = self.vm_scratch_dir
        if self.vm_scratch_dir is not None:
            try:
                shutil.rmtree(self.vm_scratch_dir)
            except OSError as exc:
                cleanup_errors.append(f"VM scratch cleanup failed: {exc}")
            self.vm_scratch_dir = None
        overlay_removed = scratch_path is None or not scratch_path.exists()
        if not overlay_removed:
            cleanup_errors.append("VM scratch directory still exists after cleanup")
        if self._task_lock_handle is not None:
            fcntl.flock(self._task_lock_handle.fileno(), fcntl.LOCK_UN)
            self._task_lock_handle.close()
            self._task_lock_handle = None
        if self._task_lock_path is not None:
            try:
                self._task_lock_path.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_errors.append(f"VM task lock cleanup failed: {exc}")
            self._task_lock_path = None
        if self._scratch_root_owned and self._scratch_root is not None:
            try:
                self._scratch_root.rmdir()
            except OSError as exc:
                cleanup_errors.append(f"job scratch root cleanup failed: {exc}")
        self._scratch_root = None
        self._scratch_root_owned = False
        self._restore_environment()
        if self.metadata_path.is_file():
            try:
                metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                metadata["closed"] = provider_stopped
                metadata["overlay"]["removed"] = overlay_removed
                metadata["cleanup_errors"] = cleanup_errors
                _atomic_json(self.metadata_path, metadata)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                cleanup_errors.append(f"VM metadata finalization failed: {exc}")
        if cleanup_errors and sys.exc_info()[0] is None:
            raise VmHarnessError("; ".join(cleanup_errors))

    def __enter__(self) -> "KvmFixtureSession":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
