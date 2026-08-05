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
import re
import subprocess
import time
from dataclasses import asdict, dataclass
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
_EVAL_DIR = Path(__file__).resolve().parent


def parse_resolution(value: str | None) -> tuple[int, int] | None:
    """Parse a ``WIDTHxHEIGHT`` string (e.g. ``1280x720``) into a (w, h) tuple.

    ``None``/empty means "native" (no resizing) and returns ``None``. Usable as
    an argparse ``type=`` callable — argparse turns the ValueError into a clean
    usage error.
    """
    if not value:
        return None
    m = re.fullmatch(r"(\d+)\s*[xX]\s*(\d+)", value.strip())
    if not m:
        raise ValueError(f"resolution must look like 1280x720, got {value!r}")
    return int(m.group(1)), int(m.group(2))


def _pil_to_data_url(img: Image.Image, *, quality: int = 85) -> str:
    """Encode a PIL image as a base64 JPEG data URL for the chat API."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def append_turn(
    recent_frames: list[Image.Image],
    recent_actions: list[str],
    frame: Image.Image,
    action_text: str,
    *,
    n_history_frames: int,
) -> None:
    """Append one completed turn to the rolling history, then block-evict.

    A "turn" is ``(frame, action_text)`` where ``action_text`` is the action the
    model produced *from the previous* current frame; ``frame`` is the resulting
    new screen. The invariant maintained is
    ``len(recent_actions) == len(recent_frames) - 1`` — every frame except the
    latest has an action that followed it.

    Eviction is StreamingLLM-style *block* eviction rather than slide-by-one:
    once the window exceeds ``n_history_frames`` we keep only the newest
    ``n_history_frames // 2`` frames (and their aligned actions). This preserves
    sglang's RadixAttention prefix cache — while the window grows the prompt is
    append-only (full cache reuse) and only the ~``N/2`` frames retained on a
    slide are re-prefilled, roughly once every ``N/2`` steps. Slide-by-one would
    instead invalidate the whole window on every step past ``N``.
    """
    recent_actions.append(action_text)
    recent_frames.append(frame)
    if len(recent_frames) > n_history_frames:
        keep = max(1, n_history_frames // 2)
        recent_frames[:] = recent_frames[-keep:]
        # Actions align to every frame but the latest.
        recent_actions[:] = recent_actions[-(len(recent_frames) - 1):] if len(recent_frames) > 1 else []


def _interleave_messages(
    system_prompt: str,
    instruction: str | None,
    image_parts: list[Any],
    recent_actions: list[str] | None,
    current_message: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble the interleaved chat message list from per-frame ``image_parts``.

    Shared by ``_call_model`` (parts are base64 ``image_url`` dicts) and
    ``build_loggable_messages`` (parts are ``<image …>`` placeholders), so the
    persisted prompt trace is structurally identical to the payload actually
    sent — only the image representation differs. ``image_parts[i]`` is the
    representation of ``recent_frames[i]``; ``recent_actions[i]`` is the action
    that followed it (``len(recent_actions) == len(image_parts) - 1``).

    ``current_message`` (optional) is extra user text attached to the LAST
    (current) frame's user turn — a message sent to the model alongside the
    current screenshot. ``None`` leaves the turn image-only (default).
    """
    recent_actions = recent_actions or []
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    n = len(image_parts)
    for i, part in enumerate(image_parts):
        content: list[Any] = []
        if i == 0 and instruction:
            content.append({"type": "text", "text": instruction})
        content.append(part)
        if i == n - 1 and current_message:
            content.append({"type": "text", "text": current_message})
        messages.append({"role": "user", "content": content})
        if i < len(recent_actions):
            messages.append({"role": "assistant", "content": recent_actions[i]})
    return messages


def window_frame_labels(step: int, n_frames: int) -> list[str]:
    """PNG filenames for the ``n_frames`` in the window at the start of ``step``.

    The window is always a contiguous tail of the saved frames ending at the
    latest screenshot (``step_000.png`` is the initial frame; ``step_{k}.png`` is
    the post-action screenshot of step ``k``). At step ``S`` with ``L`` frames in
    the window the current frame is ``step_{S-1}.png``, so the ids are
    ``[S-L .. S-1]``. Used to label images in the persisted prompt trace.
    """
    base = step - n_frames
    return [f"step_{base + i:03d}.png" for i in range(n_frames)]


def build_loggable_messages(
    *,
    system_prompt: str,
    instruction: str | None,
    recent_actions: list[str] | None,
    frame_labels: list[str],
    current_message: str | None = None,
) -> list[dict[str, Any]]:
    """The message list as sent, but with each image replaced by ``<image name>``.

    Suitable for writing a small, human-readable ``prompt_NNN.json`` sidecar that
    lets you audit exactly what the model saw each step (turn sequence, verbatim
    text, instruction placement, eviction window) without duplicating base64
    image bytes — the referenced ``step_NNN.png`` files already hold the pixels.
    """
    parts = [{"type": "image", "image": f"<image {lbl}>"} for lbl in frame_labels]
    return _interleave_messages(system_prompt, instruction, parts, recent_actions, current_message)


@dataclass(frozen=True)
class SamplingOverrides:
    """Optional sampling knobs sent alongside ``max_tokens``/``temperature``.

    Every field is tri-state: ``None`` means "do not put this key in the request
    body", which is *not* the same as sending the OpenAI default. sglang resolves
    an omitted key as ``user value > the model's generation_config.json > OpenAI
    default`` (``ChatCompletionRequest.to_sampling_params``), and the server-side
    ``--sampling-defaults`` defaults to ``model``, so omitting ``top_p``/``top_k``/
    ``repetition_penalty``/``min_p`` inherits the checkpoint's own recommended
    values (Qwen3-VL ships top_p=0.8, top_k=20, repetition_penalty=1.0). Sending
    an explicit value pins it instead, which is what makes runs comparable across
    lineages whose exports carry different generation_configs.

    ``presence_penalty``/``frequency_penalty`` are the exception: sglang reads
    those straight off the request and never consults generation_config for them,
    so they are 0.0 unless set here. Qwen3-VL's recommended
    ``presence_penalty=1.5`` is only reachable this way.
    """

    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    def to_request_fields(self) -> dict[str, Any]:
        """The subset to merge into the chat-completions body (drops ``None``)."""
        return {k: v for k, v in asdict(self).items() if v is not None}


def _call_model(
    *,
    sglang_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    instruction: str | None,
    recent_frames: list[Image.Image],
    recent_actions: list[str] | None = None,
    max_tokens: int,
    temperature: float,
    sampling: SamplingOverrides | None = None,
    current_message: str | None = None,
    request_timeout_s: float = 120.0,
) -> str:
    """One chat-completion call, interleaving frames and the model's prior actions.

    Builds a training-shaped conversation: each frame is its own ``user`` turn and
    the action the model took after a frame is fed back as the following
    ``assistant`` turn, mirroring the BC training convention
    (``user(image) -> assistant(action)``, one screen per turn). The last frame is
    the current screen and has no action yet::

        [ system,
          user[instruction? + frame_0], assistant[action_0],
          user[frame_1],                assistant[action_1],
          ...
          user[frame_{N-1}] ]          # current screen, awaiting the next action

    ``recent_actions[i]`` is the action taken after ``recent_frames[i]``, so
    ``len(recent_actions) == len(recent_frames) - 1`` (see ``append_turn``). When
    it is empty (e.g. ``n_history_frames == 1``) the list collapses to
    ``[system, user[instruction? + frame]]`` — the legacy single-frame shape.

    ``instruction`` rides the first (earliest in-window) user turn only. Callers
    pass it every step to keep the goal in context (``persist_instruction``), or
    only on step 1 to match the training distribution.
    """
    image_parts = [
        {"type": "image_url", "image_url": {"url": _pil_to_data_url(f)}}
        for f in recent_frames
    ]
    messages = _interleave_messages(
        system_prompt, instruction, image_parts, recent_actions, current_message)
    r = requests.post(
        sglang_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **(sampling.to_request_fields() if sampling else {}),
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
