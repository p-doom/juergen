"""Re-clone-proof local-qemu (KVM) DesktopEnv provider for OSWorld on hai-* nodes.

Background
----------
The team's original ``apptainer`` DesktopEnv provider was wiped when
``$OSWORLD_ROOT`` was re-cloned to upstream on 2026-07-23, and it was fragile
anyway: running qemu *inside* an apptainer userns strips the KVM ioctls, so the
VM fell back to slow software emulation (see osworld_grounding_runner.py notes).

This module restores native-OSWorld VM eval WITHOUT touching the OSWorld
checkout. It boots the OSWorld qcow2 with the bundled ``qemu-system-x86_64-wrapped``
binary (glibc/ld bundled from tianon/qemu; talks to /dev/kvm directly, no userns
seccomp), the exact KVM-accelerated path the grounding/typing evals already use.

It plugs into upstream ``DesktopEnv`` by monkeypatching the provider factory at
runtime (``install()``), so the OSWorld checkout stays pristine and disposable
and ALL custom code lives in this (version-controlled, re-clone-proof) file.

DesktopEnv's ``_start_emulator`` accepts a provider whose ``get_ip_address``
returns ``"<host>:<server>:<chromium>:<vnc>:<vlc>"`` — so we hand back the four
qemu user-mode hostfwd ports and DesktopEnv wires its controllers to them. The
host process must supply the four ports from its node-shared advisory lease;
this provider validates and consumes them without a bind-and-close race.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import requests

_LOG = logging.getLogger("qemu_kvm_provider")

DEFAULT_QEMU_BIN = os.environ.get(
    "OSWORLD_QEMU_BIN",
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/qemu/bin/qemu-system-x86_64-wrapped",
)
DEFAULT_QCOW2 = os.environ.get(
    "OSWORLD_QCOW2",
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/osworld_vm/Ubuntu.qcow2",
)
# Guest-side ports the OSWorld in-VM Flask server / debug interfaces listen on.
_GUEST_SERVER = 5000
_GUEST_CHROMIUM = 9222
_GUEST_VNC = 5900
_GUEST_VLC = 8080
_LEASED_PORT_ENV = {
    "server": "OSWORLD_APPTAINER_SERVER_PORT",
    "chromium": "OSWORLD_APPTAINER_CHROMIUM_PORT",
    "vnc": "OSWORLD_APPTAINER_VNC_PORT",
    "vlc": "OSWORLD_APPTAINER_VLC_PORT",
}


def _leased_ports_from_env(env: Mapping[str, str]) -> dict[str, int]:
    ports: dict[str, int] = {}
    for name, variable in _LEASED_PORT_ENV.items():
        raw = env.get(variable)
        if raw is None or not raw.isascii() or not raw.isdigit():
            raise RuntimeError(f"{variable} must be set to a decimal TCP port")
        port = int(raw)
        if not 1024 <= port <= 65535:
            raise RuntimeError(f"{variable} is outside the allowed range: {port}")
        ports[name] = port
    if len(set(ports.values())) != len(ports):
        raise RuntimeError(f"leased OSWorld ports must be unique: {ports}")
    return ports


class KvmProvider:
    """DesktopEnv Provider backed by a locally-booted, KVM-accelerated qemu VM.

    Not importing OSWorld's ``Provider`` ABC keeps this file importable without
    ``$OSWORLD_ROOT`` on ``sys.path`` (duck-typing is all DesktopEnv needs).
    """

    def __init__(self, region: str | None = None) -> None:
        self.region = region
        self.qemu_bin = DEFAULT_QEMU_BIN
        self.mem = os.environ.get("OSWORLD_VM_MEM", "8G")
        self.smp = os.environ.get("OSWORLD_VM_SMP", "8")
        self.boot_timeout_s = int(os.environ.get("OSWORLD_VM_BOOT_TIMEOUT_S", "300"))
        # path_to_vm -> {"proc": Popen, "ports": {...}, "log": Path}
        self._vms: dict[str, dict] = {}

    # -- lifecycle -------------------------------------------------------
    def start_emulator(self, path_to_vm: str, headless: bool = True, os_type: str = "Ubuntu") -> None:
        st = self._vms.get(path_to_vm)
        if st and st["proc"].poll() is None:
            return  # already running
        ports = _leased_ports_from_env(os.environ)
        hostfwd = ",".join(
            [
                f"hostfwd=tcp::{ports['server']}-:{_GUEST_SERVER}",
                f"hostfwd=tcp::{ports['chromium']}-:{_GUEST_CHROMIUM}",
                f"hostfwd=tcp::{ports['vnc']}-:{_GUEST_VNC}",
                f"hostfwd=tcp::{ports['vlc']}-:{_GUEST_VLC}",
            ]
        )
        log_path = (
            Path(os.environ.get("OSWORLD_VM_LOG_DIR", "/tmp"))
            / f"qemu_{ports['server']}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.qemu_bin,
            "-enable-kvm",
            "-cpu", "host",
            "-smp", str(self.smp),
            "-m", str(self.mem),
            "-machine", "type=q35,accel=kvm",
            "-drive", f"file={path_to_vm},if=virtio,format=qcow2,snapshot=on",
            "-netdev", f"user,id=net0,{hostfwd}",
            "-device", "virtio-net-pci,netdev=net0",
            "-display", "none", "-nographic",
        ]
        _LOG.info("booting VM: %s (server_port=%d)", path_to_vm, ports["server"])
        with log_path.open("w", encoding="utf-8") as log_handle:
            proc = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        self._vms[path_to_vm] = {"proc": proc, "ports": ports, "log": log_path}
        try:
            self._wait_ready(ports["server"], proc)
        except BaseException:
            self.stop_emulator(path_to_vm)
            raise

    def _wait_ready(self, server_port: int, proc: subprocess.Popen) -> None:
        url = f"http://localhost:{server_port}/screenshot"
        start = time.time()
        while time.time() - start < self.boot_timeout_s:
            if proc.poll() is not None:
                raise RuntimeError(f"qemu died early (rc={proc.returncode})")
            try:
                if requests.get(url, timeout=3).status_code == 200:
                    _LOG.info("VM controller ready after %.0fs (:%d)", time.time() - start, server_port)
                    return
            except requests.RequestException:
                pass
            time.sleep(3)
        raise TimeoutError(f"VM controller not ready after {self.boot_timeout_s}s (:{server_port})")

    def get_ip_address(self, path_to_vm: str) -> str:
        p = self._vms[path_to_vm]["ports"]
        # DesktopEnv rsplit(':',4) -> host, server, chromium, vnc, vlc
        return f"localhost:{p['server']}:{p['chromium']}:{p['vnc']}:{p['vlc']}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        return None  # snapshot=on overlay is throwaway; nothing to persist

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str) -> str:
        # Clean state == fresh boot (snapshot=on discards writes). Kill + reboot.
        self.stop_emulator(path_to_vm)
        self.start_emulator(path_to_vm)
        return path_to_vm

    def stop_emulator(self, path_to_vm: str, region: str | None = None) -> None:
        st = self._vms.get(path_to_vm)
        if not st:
            return
        proc = st["proc"]
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self._vms.pop(path_to_vm, None)


class KvmVMManager:
    """Minimal VMManager. We always pass an explicit path_to_vm, so the registry
    methods are inert; get_vm_path just echoes the configured qcow2."""

    checked_and_cleaned = True

    def initialize_registry(self, **kwargs):
        return None

    def add_vm(self, vm_path, *a, **k):
        return None

    def delete_vm(self, vm_path, *a, **k):
        return None

    def occupy_vm(self, vm_path, pid, *a, **k):
        return None

    def list_free_vms(self, *a, **k):
        return []

    def check_and_clean(self, *a, **k):
        return None

    def get_vm_path(self, *a, **k):
        return DEFAULT_QCOW2


def install(alias_names: tuple[str, ...] = ("docker", "qemukvm", "manual")) -> None:
    """Monkeypatch DesktopEnv's provider factory so ``alias_names`` resolve to
    the local KVM provider. Import ``desktop_env`` (needs $OSWORLD_ROOT on path)
    BEFORE calling this. Use provider_name='docker' at the DesktopEnv level:
    it is in DesktopEnv's clean-start set and sets client_password='password'
    (the OSWorld Ubuntu image's password)."""
    import desktop_env.desktop_env as dm

    _orig = dm.create_vm_manager_and_provider

    def _patched(provider_name, region=None, use_proxy=False):
        if str(provider_name).lower().strip() in alias_names:
            _LOG.info("KVM provider serving provider_name=%r", provider_name)
            return KvmVMManager(), KvmProvider(region)
        return _orig(provider_name, region, use_proxy)

    dm.create_vm_manager_and_provider = _patched
    _LOG.info("qemu_kvm_provider installed for aliases: %s", alias_names)
