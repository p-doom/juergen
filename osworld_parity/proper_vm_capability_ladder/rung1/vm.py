from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import json
import os
import re
import secrets
import shutil
import socket
import struct
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import fcntl

from .fixtures import Fixture
from .server import FixtureHttpServer
from .transport import ALL_POINTER_BUTTON_MASK, HttpVmTransport


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
POINTER_STATE_PREFIX = "RUNG1A_POINTER_STATE="


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


_WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_MAX_CDP_MESSAGE_BYTES = 4 * 1024 * 1024
_MAX_CDP_TARGET_LIST_BYTES = 1024 * 1024
_MAX_CHROME_LOG_BYTES = 1024 * 1024
CHROME_LOG_PREFIX = "RUNG1A_CHROME_LOG="


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise VmHarnessError("CDP websocket closed before its response completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_websocket_frame(sock: socket.socket, payload: bytes, *, opcode: int) -> None:
    first = 0x80 | opcode
    length = len(payload)
    if length < 126:
        header = bytes((first, 0x80 | length))
    elif length <= 0xFFFF:
        header = bytes((first, 0x80 | 126)) + struct.pack("!H", length)
    else:
        header = bytes((first, 0x80 | 127)) + struct.pack("!Q", length)
    mask = os.urandom(4)
    masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    sock.sendall(header + mask + masked)


def _recv_websocket_message(sock: socket.socket) -> bytes:
    message = bytearray()
    message_opcode: int | None = None
    while True:
        first, second = _recv_exact(sock, 2)
        final = bool(first & 0x80)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", _recv_exact(sock, 2))[0]
        elif length == 127:
            length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
        if length > _MAX_CDP_MESSAGE_BYTES:
            raise VmHarnessError(f"CDP websocket frame exceeded {length} bytes")
        mask = _recv_exact(sock, 4) if masked else b""
        payload = _recv_exact(sock, length)
        if masked:
            payload = bytes(
                value ^ mask[index % 4] for index, value in enumerate(payload)
            )
        if opcode == 0x8:
            raise VmHarnessError("CDP websocket closed before the requested response")
        if opcode == 0x9:
            _send_websocket_frame(sock, payload, opcode=0xA)
            continue
        if opcode == 0xA:
            continue
        if opcode in {0x1, 0x2}:
            message_opcode = opcode
        elif opcode != 0x0 or message_opcode is None:
            raise VmHarnessError(f"unsupported CDP websocket opcode {opcode}")
        message.extend(payload)
        if len(message) > _MAX_CDP_MESSAGE_BYTES:
            raise VmHarnessError("CDP websocket message exceeded the evidence bound")
        if final:
            return bytes(message)


def _cdp_evaluate(websocket_url: str, expression: str, *, timeout_s: float) -> Any:
    parsed = urllib.parse.urlsplit(websocket_url)
    if parsed.scheme != "ws" or parsed.hostname is None:
        raise VmHarnessError(f"unsupported CDP websocket URL: {websocket_url!r}")
    port = parsed.port or 80
    path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    expected_accept = base64.b64encode(
        hashlib.sha1((key + _WEBSOCKET_GUID).encode("ascii")).digest()
    ).decode("ascii")
    with socket.create_connection((parsed.hostname, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        handshake = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode("ascii")
        sock.sendall(handshake)
        response = bytearray()
        while b"\r\n\r\n" not in response:
            if len(response) > 65536:
                raise VmHarnessError("CDP websocket handshake exceeded its bound")
            response.extend(_recv_exact(sock, 1))
        header_text = bytes(response).decode("iso-8859-1")
        lines = header_text.split("\r\n")
        if not lines or " 101 " not in lines[0]:
            raise VmHarnessError(f"CDP websocket upgrade failed: {lines[0]!r}")
        headers = {
            name.strip().lower(): value.strip()
            for line in lines[1:]
            if ":" in line
            for name, value in [line.split(":", 1)]
        }
        if headers.get("sec-websocket-accept") != expected_accept:
            raise VmHarnessError("CDP websocket returned an invalid accept key")
        request_id = 1
        request = json.dumps(
            {
                "id": request_id,
                "method": "Runtime.evaluate",
                "params": {
                    "expression": expression,
                    "returnByValue": True,
                    "awaitPromise": False,
                },
            },
            separators=(",", ":"),
        ).encode("utf-8")
        _send_websocket_frame(sock, request, opcode=0x1)
        while True:
            raw = _recv_websocket_message(sock)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise VmHarnessError("CDP websocket returned invalid JSON") from exc
            if not isinstance(payload, dict) or payload.get("id") != request_id:
                continue
            if "error" in payload:
                raise VmHarnessError(f"CDP Runtime.evaluate failed: {payload['error']}")
            result = payload.get("result", {}).get("result", {})
            if not isinstance(result, dict):
                raise VmHarnessError("CDP Runtime.evaluate returned no result object")
            if "exceptionDetails" in payload.get("result", {}):
                raise VmHarnessError(
                    f"CDP Runtime.evaluate raised: {payload['result']['exceptionDetails']}"
                )
            if "value" not in result:
                raise VmHarnessError(f"CDP Runtime.evaluate returned no value: {result}")
            return result["value"]


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
        self._chromium_port: int | None = None

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
            self._chromium_port = int(metadata["ports"]["chromium"])
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
        ports = after["ports"]
        port = int(ports["server"])
        self._chromium_port = int(ports["chromium"])
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

    def capture_browser_diagnostics(
        self, fixture: Fixture, *, timeout_s: float = 2.0
    ) -> dict[str, Any]:
        """Read the live fixture page through CDP without modifying page state."""
        if self._chromium_port is None:
            raise VmHarnessError("VM session has no forwarded Chromium CDP port")
        list_url = f"http://127.0.0.1:{self._chromium_port}/json/list"
        with urllib.request.urlopen(list_url, timeout=timeout_s) as response:
            target_bytes = response.read(_MAX_CDP_TARGET_LIST_BYTES + 1)
        if len(target_bytes) > _MAX_CDP_TARGET_LIST_BYTES:
            raise VmHarnessError("Chromium CDP target list exceeded its evidence bound")
        targets = json.loads(target_bytes)
        if not isinstance(targets, list):
            raise VmHarnessError("Chromium CDP target list was not an array")
        matching = [
            target
            for target in targets
            if isinstance(target, dict)
            and target.get("type") == "page"
            and fixture.id in str(target.get("url", ""))
            and isinstance(target.get("webSocketDebuggerUrl"), str)
        ]
        if len(matching) != 1:
            summary = [
                {
                    "id": target.get("id"),
                    "type": target.get("type"),
                    "url": target.get("url"),
                }
                for target in targets
                if isinstance(target, dict)
            ]
            raise VmHarnessError(
                f"expected one live CDP fixture target, found {len(matching)}: {summary}"
            )
        target = matching[0]
        advertised_websocket_url = str(target["webSocketDebuggerUrl"])
        advertised = urllib.parse.urlsplit(advertised_websocket_url)
        local_websocket_url = urllib.parse.urlunsplit(
            (
                "ws",
                f"127.0.0.1:{self._chromium_port}",
                advertised.path,
                advertised.query,
                "",
            )
        )
        expression = r"""
(() => {
  const elementState = (element) => element ? {
    id: element.id || '',
    tag: element.tagName ? element.tagName.toLowerCase() : '',
    checked: typeof element.checked === 'boolean' ? element.checked : null,
    value: 'value' in element ? String(element.value) : null,
    disabled: Boolean(element.disabled),
    outer_html: element.outerHTML
  } : null;
  return {
    schema_version: 1,
    captured_browser_wall_time_ms: Date.now(),
    performance_time_origin_ms: performance.timeOrigin,
    performance_now_ms: performance.now(),
    captured_client_monotonic_ms: Math.round(performance.now() * 1000) / 1000,
    url: location.href,
    title: document.title,
    ready_state: document.readyState,
    visibility_state: document.visibilityState,
    has_focus: document.hasFocus(),
    diagnostics: window.__RUNG1A_DIAGNOSTICS__
      ? JSON.parse(JSON.stringify(window.__RUNG1A_DIAGNOSTICS__)) : null,
    dom: {
      active_element: elementState(document.activeElement),
      target: elementState(document.getElementById('target')),
      decoy: elementState(document.getElementById('decoy')),
      scroll_x: Math.round(window.scrollX),
      scroll_y: Math.round(window.scrollY),
      body_text: document.body ? document.body.innerText : null,
      outer_html: document.documentElement ? document.documentElement.outerHTML : null
    }
  };
})()
""".strip()
        page = _cdp_evaluate(local_websocket_url, expression, timeout_s=timeout_s)
        if not isinstance(page, dict):
            raise VmHarnessError("CDP page diagnostic result was not an object")
        return {
            "schema_version": 1,
            "status": "captured",
            "transport": "cdp_runtime_evaluate",
            "host_forwarded_port": self._chromium_port,
            "target": {
                "id": target.get("id"),
                "type": target.get("type"),
                "url": target.get("url"),
                "title": target.get("title"),
                "advertised_websocket_url": advertised_websocket_url,
                "local_websocket_url": local_websocket_url,
            },
            "page": page,
        }

    def capture_guest_pointer_state(self) -> dict[str, Any]:
        """Query the live X root pointer without issuing an input operation."""
        if self.transport is None:
            raise VmHarnessError("VM session is not started")
        program = (
            "import json,time\n"
            "from Xlib import display\n"
            "d=display.Display()\n"
            "wall_before=time.time_ns()\n"
            "monotonic_before=time.monotonic_ns()\n"
            "q=d.screen().root.query_pointer()\n"
            "monotonic_after=time.monotonic_ns()\n"
            "wall_after=time.time_ns()\n"
            "payload={'schema_version':1,'cursor':[int(q.root_x),int(q.root_y)],"
            f"'raw_x_mask':int(q.mask),'pointer_button_mask':int(q.mask)&"
            f"{ALL_POINTER_BUTTON_MASK},'guest_wall_before_ns':wall_before,"
            "'guest_wall_after_ns':wall_after,"
            "'guest_monotonic_before_ns':monotonic_before,"
            "'guest_monotonic_after_ns':monotonic_after}\n"
            f"print({POINTER_STATE_PREFIX!r}+json.dumps(payload,sort_keys=True))\n"
            "d.close()\n"
        )
        raw = self.transport.execute_argv(["python", "-c", program], check=False)
        output = raw.get("output")
        markers = (
            [line for line in output.splitlines() if line.startswith(POINTER_STATE_PREFIX)]
            if isinstance(output, str)
            else []
        )
        parsed: dict[str, Any] | None = None
        if len(markers) == 1:
            candidate = json.loads(markers[0][len(POINTER_STATE_PREFIX) :])
            if isinstance(candidate, dict):
                parsed = candidate
        return {
            "schema_version": 1,
            "status": "captured" if parsed is not None else "capture_failed",
            "guest_returncode": raw.get("returncode"),
            "guest_status": raw.get("status"),
            "raw_result_marker": markers[0] if len(markers) == 1 else None,
            "cursor": parsed.get("cursor") if parsed is not None else None,
            "raw_x_mask": parsed.get("raw_x_mask") if parsed is not None else None,
            "pointer_button_mask": (
                parsed.get("pointer_button_mask") if parsed is not None else None
            ),
            "guest_wall_before_ns": (
                parsed.get("guest_wall_before_ns") if parsed is not None else None
            ),
            "guest_wall_after_ns": (
                parsed.get("guest_wall_after_ns") if parsed is not None else None
            ),
            "guest_monotonic_before_ns": (
                parsed.get("guest_monotonic_before_ns")
                if parsed is not None
                else None
            ),
            "guest_monotonic_after_ns": (
                parsed.get("guest_monotonic_after_ns")
                if parsed is not None
                else None
            ),
            "raw_guest_result": raw,
        }

    def capture_chrome_log(self) -> dict[str, Any]:
        if self.transport is None:
            raise VmHarnessError("VM session is not started")
        program = (
            "import hashlib,json\n"
            "from pathlib import Path\n"
            "p=Path('/tmp/rung1a_chrome.log')\n"
            "digest=hashlib.sha256()\n"
            "tail=bytearray()\n"
            "total=0\n"
            "with p.open('rb') as source:\n"
            "  while True:\n"
            "    chunk=source.read(65536)\n"
            "    if not chunk: break\n"
            "    total+=len(chunk)\n"
            "    digest.update(chunk)\n"
            "    tail.extend(chunk)\n"
            f"    if len(tail)>{_MAX_CHROME_LOG_BYTES}: "
            f"del tail[:-{_MAX_CHROME_LOG_BYTES}]\n"
            "payload={'schema_version':1,'total_bytes':total,"
            "'captured_bytes':len(tail),'sha256':digest.hexdigest(),"
            "'truncated':len(tail)!=total,'tail':tail.decode('utf-8','replace')}\n"
            f"print({CHROME_LOG_PREFIX!r}+json.dumps(payload,sort_keys=True))\n"
        )
        raw = self.transport.execute_argv(["python", "-c", program], check=False)
        output = raw.get("output")
        markers = (
            [line for line in output.splitlines() if line.startswith(CHROME_LOG_PREFIX)]
            if isinstance(output, str)
            else []
        )
        parsed: dict[str, Any] | None = None
        if len(markers) == 1:
            candidate = json.loads(markers[0][len(CHROME_LOG_PREFIX) :])
            if isinstance(candidate, dict):
                parsed = candidate
        return {
            "schema_version": 1,
            "status": "captured" if parsed is not None else "capture_failed",
            "path": "/tmp/rung1a_chrome.log",
            "guest_returncode": raw.get("returncode"),
            "guest_status": raw.get("status"),
            "total_bytes": parsed.get("total_bytes") if parsed is not None else None,
            "captured_bytes": (
                parsed.get("captured_bytes") if parsed is not None else None
            ),
            "sha256": parsed.get("sha256") if parsed is not None else None,
            "truncated": parsed.get("truncated") if parsed is not None else None,
            "content_tail": parsed.get("tail") if parsed is not None else None,
            "guest_error": raw.get("error"),
            "raw_guest_result": {
                key: value for key, value in raw.items() if key != "output"
            },
        }

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
        self._chromium_port = None
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
