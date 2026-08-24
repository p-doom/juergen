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


def _patch_agent_sampling() -> None:
    import openai
    from mm_agents import qwen3vl_agent

    def _call_llm_openai(self, messages, model):
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        api_key = os.environ.get("OPENAI_API_KEY", "sk-123")
        client = openai.OpenAI(base_url=base_url, api_key=api_key)
        for attempt in range(1, qwen3vl_agent.MAX_RETRY_TIMES + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                return response.choices[0].message.content
            except Exception:
                if attempt < qwen3vl_agent.MAX_RETRY_TIMES:
                    import time as _time

                    _time.sleep(5)
                    continue
                break
        return ""

    qwen3vl_agent.Qwen3VLAgent._call_llm_openai = _call_llm_openai


def main() -> None:
    _lease_vm_ports()

    import osworld_fullbench_runner as runner
    import qemu_kvm_provider

    qemu_kvm_provider.install()
    _patch_agent_sampling()

    warm_url = os.environ.get("SGLANG_URL", "").rstrip("/")
    if warm_url:
        runner.sglang_server = lambda **_kwargs: contextlib.nullcontext(warm_url)

    from absl import app

    app.run(runner.main)


if __name__ == "__main__":
    main()
