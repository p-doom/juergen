from __future__ import annotations

import contextlib
import os

os.environ.setdefault(
    "OSWORLD_ROOT", "/fast/project/HFMI_SynergyUnit/yll/osworld-pinned"
)

from osworld_fullbench_kvm import _lease_vm_ports


def main() -> None:
    _lease_vm_ports()

    import osworld_fullbench_runner as runner
    import qemu_kvm_provider
    from mm_agents import qwen3vl_agent

    from oev3_agent import Oev3Agent

    qemu_kvm_provider.install()
    qwen3vl_agent.Qwen3VLAgent = Oev3Agent

    warm_url = os.environ.get("SGLANG_URL", "").rstrip("/")
    if warm_url:
        runner.sglang_server = lambda **_kwargs: contextlib.nullcontext(warm_url)

    from absl import app

    app.run(runner.main)


if __name__ == "__main__":
    main()
