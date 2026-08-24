from __future__ import annotations

import contextlib
import os
import socket

os.environ.setdefault(
    "OSWORLD_ROOT", "/fast/project/HFMI_SynergyUnit/yll/osworld-pinned"
)

_VM_PORT_ENV_VARS = (
    "OSWORLD_APPTAINER_SERVER_PORT",
    "OSWORLD_APPTAINER_CHROMIUM_PORT",
    "OSWORLD_APPTAINER_VNC_PORT",
    "OSWORLD_APPTAINER_VLC_PORT",
)


def _lease_vm_ports() -> None:
    missing = [name for name in _VM_PORT_ENV_VARS if not os.environ.get(name)]
    if not missing:
        return
    socks: list[socket.socket] = []
    try:
        for _ in missing:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("0.0.0.0", 0))
            socks.append(s)
        for name, s in zip(missing, socks, strict=True):
            os.environ[name] = str(s.getsockname()[1])
    finally:
        for s in socks:
            s.close()


def main() -> None:
    _lease_vm_ports()

    import osworld_fullbench_runner as runner
    import qemu_kvm_provider

    qemu_kvm_provider.install()

    warm_url = os.environ.get("SGLANG_URL", "").rstrip("/")
    if warm_url:
        runner.sglang_server = lambda **_kwargs: contextlib.nullcontext(warm_url)

    from absl import app

    app.run(runner.main)


if __name__ == "__main__":
    main()
