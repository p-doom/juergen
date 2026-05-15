"""SGLang server lifecycle as a context manager.

Spawns ``sglang.launch_server`` as a subprocess in this same uv venv,
polls /health_generate until ready (cold flashinfer JIT compilation can
take several minutes), yields the OpenAI-compatible base URL, and tears
down on context exit.

Why we run our own server rather than letting inspect_ai auto-manage it:
inspect_ai's auto-managed startup timeout is too short for cold JIT
compiles on first launch. Manual lifecycle keeps us in control of the
ready-check loop.
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _reserve_free_port(host: str = "127.0.0.1") -> int:
    """Ask the kernel for a free ephemeral TCP port and release it.

    Used when the caller passes ``port=0`` (meaning "any free port"). SGLang
    itself accepts ``--port 0`` and binds to a kernel-assigned port, but its
    internal ``_execute_server_warmup`` and our own ``_wait_for_ready`` both
    construct probe URLs from the *original* args.port value — and a URL of
    ``http://localhost:0/...`` is normalised by urllib to drop the port,
    sending the request to port 80 instead. Result: warmup never reaches the
    actual server, the readiness probe times out, and the parent SIGKILLs
    SGLang. By picking a real free port here and passing it to both
    ``--port`` and the probe URL, we sidestep the SGLang bug.

    Small race window: another process could grab the port between
    ``close()`` and SGLang's bind. In our cluster topology each ifeval job
    is on its own node with no contender, so this is fine in practice.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def sglang_server(
    *,
    model_path: str,
    port: int,
    api_key: str,
    log_path: Path,
    mem_fraction_static: float = 0.80,
    chunked_prefill_size: int = 2048,
    ready_timeout_s: int = 1500,
    served_model_name: str | None = None,
):
    """Yield ``http://localhost:<port>/v1`` once SGLang is /health_generate-ready.

    ``port=0`` is treated as "auto-pick a free ephemeral port" — see
    ``_reserve_free_port`` for the reason we can't just hand 0 to SGLang.

    ``served_model_name`` (optional) is forwarded as
    ``--served-model-name`` so chat-completion requests can address the
    model under a short alias rather than the full ``--model-path``.
    """
    if port == 0:
        port = _reserve_free_port()
        print(f"[sglang] auto-picked free port: {port}", flush=True)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SGLANG_DISABLE_CUDNN_CHECK": "1"}
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--api-key",
        api_key,
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--chunked-prefill-size",
        str(chunked_prefill_size),
    ]
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])
    print(f"[sglang] launching: {' '.join(cmd)}", flush=True)
    log_f = log_path.open("w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    try:
        _wait_for_ready(port, api_key, timeout_s=ready_timeout_s, proc=proc, log_path=log_path)
        url = f"http://localhost:{port}/v1"
        print(f"[sglang] ready at {url}", flush=True)
        yield url
    finally:
        print("[sglang] terminating", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log_f.close()


def _wait_for_ready(
    port: int, api_key: str, *, timeout_s: int, proc: subprocess.Popen, log_path: Path
) -> None:
    url = f"http://localhost:{port}/health_generate"
    headers = {"Authorization": f"Bearer {api_key}"}
    start = time.time()
    last_log = 0
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            tail = "\n".join(log_path.read_text().splitlines()[-50:])
            raise RuntimeError(
                f"sglang exited before ready (rc={proc.returncode}). Last 50 log lines:\n{tail}"
            )
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status < 500:
                    return
        except Exception:
            pass
        elapsed = int(time.time() - start)
        if elapsed - last_log >= 60:
            print(f"[sglang] waiting for ready... {elapsed}s", flush=True)
            last_log = elapsed
        time.sleep(5)
    raise RuntimeError(f"sglang not ready after {timeout_s}s")
