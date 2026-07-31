from __future__ import annotations

import hashlib
import importlib.util
import os
import time
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
        if self.provider is None:
            raise VmHarnessError("VM session is not started")
        self.provider.load_state(str(self.qcow), READY_SNAPSHOT)
        port = int(self.provider.state(str(self.qcow))["ports"]["server"])
        # Sessions and host-side input audits must never cross an episode reset.
        self.transport = HttpVmTransport(f"http://127.0.0.1:{port}")
        return self.transport

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
