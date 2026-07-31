from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from rl.runtime.ports import PortLease, allocate_worker_ports

PROVIDER_SOURCE = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/"
    "osworld_eval_pinned/code/qemu_kvm_provider.py"
)
SPEC = importlib.util.spec_from_file_location("pinned_qemu_kvm_provider", PROVIDER_SOURCE)
assert SPEC is not None and SPEC.loader is not None
provider = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = provider
SPEC.loader.exec_module(provider)


def _port_env(lease: PortLease) -> dict[str, str]:
    return {
        "OSWORLD_APPTAINER_SERVER_PORT": str(lease.ports.server),
        "OSWORLD_APPTAINER_CHROMIUM_PORT": str(lease.ports.chromium),
        "OSWORLD_APPTAINER_VNC_PORT": str(lease.ports.vnc),
        "OSWORLD_APPTAINER_VLC_PORT": str(lease.ports.vlc),
    }


def test_provider_consumes_two_concurrent_disjoint_leases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSWORLD_PORT_BASE", "42000")
    first = allocate_worker_ports(lock_dir=tmp_path / "locks")
    second = allocate_worker_ports(lock_dir=tmp_path / "locks")
    try:
        first_ports = provider._leased_ports_from_env(_port_env(first))
        second_ports = provider._leased_ports_from_env(_port_env(second))
        assert first_ports == {
            "server": 42000,
            "chromium": 42001,
            "vnc": 42002,
            "vlc": 42003,
        }
        assert second_ports == {
            "server": 42010,
            "chromium": 42011,
            "vnc": 42012,
            "vlc": 42013,
        }
        assert set(first_ports.values()).isdisjoint(second_ports.values())
    finally:
        first.release()
        second.release()


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({}, "must be set"),
        (
            {
                "OSWORLD_APPTAINER_SERVER_PORT": "not-a-port",
                "OSWORLD_APPTAINER_CHROMIUM_PORT": "30001",
                "OSWORLD_APPTAINER_VNC_PORT": "30002",
                "OSWORLD_APPTAINER_VLC_PORT": "30003",
            },
            "decimal TCP port",
        ),
        (
            {
                "OSWORLD_APPTAINER_SERVER_PORT": "80",
                "OSWORLD_APPTAINER_CHROMIUM_PORT": "30001",
                "OSWORLD_APPTAINER_VNC_PORT": "30002",
                "OSWORLD_APPTAINER_VLC_PORT": "30003",
            },
            "allowed range",
        ),
        (
            {
                "OSWORLD_APPTAINER_SERVER_PORT": "30000",
                "OSWORLD_APPTAINER_CHROMIUM_PORT": "30000",
                "OSWORLD_APPTAINER_VNC_PORT": "30002",
                "OSWORLD_APPTAINER_VLC_PORT": "30003",
            },
            "must be unique",
        ),
    ],
)
def test_provider_rejects_missing_invalid_and_duplicate_ports(
    env: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        provider._leased_ports_from_env(env)


class _FakeProcess:
    returncode: int | None = None
    terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0

    def kill(self) -> None:
        self.returncode = -9


def test_qemu_argv_uses_exact_leased_ports_without_reallocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in {
        "OSWORLD_APPTAINER_SERVER_PORT": "30000",
        "OSWORLD_APPTAINER_CHROMIUM_PORT": "30001",
        "OSWORLD_APPTAINER_VNC_PORT": "30002",
        "OSWORLD_APPTAINER_VLC_PORT": "30003",
        "OSWORLD_VM_LOG_DIR": str(tmp_path / "logs"),
    }.items():
        monkeypatch.setenv(name, value)
    captured: dict[str, Any] = {}
    fake_process = _FakeProcess()

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return fake_process

    monkeypatch.setattr(provider.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(provider.KvmProvider, "_wait_ready", lambda *args: None)
    vm = provider.KvmProvider()
    vm.qemu_bin = "/pinned/qemu"

    vm.start_emulator(str(tmp_path / "Ubuntu.qcow2"))

    command = captured["command"]
    netdev = command[command.index("-netdev") + 1]
    assert netdev == (
        "user,id=net0,hostfwd=tcp::30000-:5000,"
        "hostfwd=tcp::30001-:9222,hostfwd=tcp::30002-:5900,"
        "hostfwd=tcp::30003-:8080"
    )
    assert "-enable-kvm" in command
    assert "snapshot=on" in command[command.index("-drive") + 1]
    assert captured["kwargs"]["stderr"] is subprocess.STDOUT
    assert vm.get_ip_address(str(tmp_path / "Ubuntu.qcow2")) == (
        "localhost:30000:30001:30002:30003"
    )
    vm.stop_emulator(str(tmp_path / "Ubuntu.qcow2"))
    assert fake_process.terminated
