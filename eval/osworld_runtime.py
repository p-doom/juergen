"""Shared OSWorld eval runtime helpers — sglang call, readiness polling,
PIL→data-URL encoding, VM/qemu binary defaults.

Extracted from freeroll.py so the grounding runner (and any future OSWorld
eval that boots a VM + serves a model via sglang) can reuse the same
primitives without reaching into the freeroll script.
"""

from __future__ import annotations

import base64
import io
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

_LOGGER = logging.getLogger(__name__)

# Defaults inherited from the original freeroll.py setup. Paths are
# cluster-specific (hai-* OSWorld VM image + qemu wrapper) and intentionally
# left as overridable by --qcow2 / --qemu_bin flags in each runner.
_DEFAULT_QCOW2 = "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/osworld_vm/Ubuntu.qcow2"
_DEFAULT_QEMU_BIN = "/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/qemu/bin/qemu-system-x86_64-wrapped"
# uv project root for the inner sglang.launch_server call. The outer
# `uv run` (in the labctl recipe command) uses the same path; keeping it
# as a module constant survives the sys.path-shim removal in juergen 39d6d5f.
_EVAL_DIR = Path("/fast/home/franz.srambical/juergen/eval")


def _pil_to_data_url(img: Image.Image, *, quality: int = 85) -> str:
    """Encode a PIL image as a base64 JPEG data URL for the chat API."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _call_model(
    *,
    sglang_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    instruction: str | None,
    recent_frames: list[Image.Image],
    max_tokens: int,
    temperature: float,
    request_timeout_s: float = 120.0,
) -> str:
    """One chat-completion call: system + user(turn1-instruction? + frames).

    The user-message shape mirrors the BC training convention:
      - turn 1 contains the natural-language goal (when ``instruction`` is set).
      - subsequent turns send only the most recent N frames.
    History of the model's own assistant tokens is intentionally NOT
    interleaved into the conversation here — this matches what
    freeroll.py has been doing in practice. The trade-off is documented
    in the grounding-eval design discussion.
    """
    user_content: list[dict[str, Any]] = []
    if instruction:
        user_content.append({"type": "text", "text": instruction})
    for f in recent_frames:
        user_content.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(f)}})
    r = requests.post(
        sglang_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=request_timeout_s,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def _wait_for(
    url: str,
    *,
    headers: dict | None = None,
    proc: subprocess.Popen,
    poll_s: float,
    max_polls: int,
    label: str,
) -> None:
    """Poll a health URL until it returns 200 or the spawning process dies.

    Used to wait for the in-VM Flask agent (/screenshot) and sglang
    (/health_generate) without busy-looping on a process that already
    exited (which we'd otherwise spend the full timeout window on).
    """
    for i in range(1, max_polls + 1):
        if proc.poll() is not None:
            raise RuntimeError(f"{label} died early (rc={proc.returncode})")
        try:
            r = requests.get(url, headers=headers or {}, timeout=3)
            if r.status_code == 200:
                _LOGGER.info("%s ready after %.0fs", label, i * poll_s)
                return
        except requests.RequestException:
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"{label} not ready after {max_polls * poll_s:.0f}s")
