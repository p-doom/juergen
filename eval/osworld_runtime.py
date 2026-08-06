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
import math
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from sampling import SamplingParams

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


class ScreenshotCheckpointController:
    """Trigger compaction near a configured screenshot-context capacity.

    The controller counts screenshots in the current segment (starting at 1 for
    the frame already in context) and is ``due`` once the count reaches
    ``ceil(capacity * fraction)``. ``reset_to_current`` restarts the count at 1
    because the boundary screenshot carries over into the next segment — it is
    the last frame of the compacted record and the first frame of the resumed
    one (see ``sequential_packing.boundary_events`` /
    ``segments_from_boundaries``, which mirror these semantics exactly).

    Jitter note: the runtime keeps a FIXED ``fraction`` (0.7 by default), while
    training deliberately covers a jittered range — Stage 04 draws a per-segment
    fraction uniformly from ``[PackingConfig.fraction_low,
    PackingConfig.fraction_high]`` = ``[0.5, 0.85]``, seeded per
    (seed, day_tag, packing_index). The training distribution therefore brackets
    the single runtime fraction on both sides, so the model sees compaction
    requested anywhere in that band and the deployed 0.7 is in-distribution. No
    runtime change is needed to match training; the only hard requirement is the
    single-eviction guarantee below (``validate_single_eviction``).
    """

    def __init__(self, capacity: int, fraction: float = 0.7):
        if capacity <= 0:
            raise ValueError("checkpoint capacity must be positive")
        if not 0 < fraction <= 1:
            raise ValueError("checkpoint fraction must be in (0, 1]")
        self.capacity = int(capacity)
        self.fraction = float(fraction)
        self.threshold = max(1, math.ceil(self.capacity * self.fraction))
        self.screenshots = 1

    @property
    def due(self) -> bool:
        return self.screenshots >= self.threshold

    def note_screenshot(self) -> None:
        self.screenshots += 1

    def reset_to_current(self) -> None:
        self.screenshots = 1


def validate_single_eviction(
    *,
    n_history_frames: int,
    controller: ScreenshotCheckpointController,
) -> None:
    """Fail fast unless the controller is the FIRST thing that drops a frame.

    Two independent mechanisms can shrink the runtime image window: this
    controller (a checkpoint request, then a full compaction down to the
    boundary frame) and ``append_turn``'s StreamingLLM block eviction (keep the
    newest ``n_history_frames // 2``). A training record contains every frame
    since the last compaction, so on the sequential path block eviction must
    never fire first — if it did, the model would be conditioned on a window
    with a hole in it that no training segment ever contains.

    The controller counts screenshots 1:1 with the frames ``append_turn``
    accumulates, so ``threshold <= capacity <= n_history_frames`` is exactly the
    condition that makes the controller strictly earlier (``threshold`` is
    reached at ``threshold`` frames; eviction needs ``n_history_frames + 1``).

    Residual case, deliberately left as the bounded fallback: if a checkpoint
    reply fails to parse the caller resets the controller without compacting, so
    the window keeps growing and block eviction can eventually fire. That run has
    no valid checkpoint and is already out of distribution; bounded eviction is
    preferable to an unbounded prompt.
    """
    if n_history_frames < controller.capacity:
        raise ValueError(
            "single-eviction violation: rolling image window "
            f"n_history_frames={n_history_frames} is smaller than checkpoint "
            f"capacity={controller.capacity}, so append_turn block eviction "
            f"would drop frames before the controller fires at "
            f"{controller.threshold} screenshots; a training segment contains "
            "every frame since the last compaction, so the runtime must too "
            "(raise n_history_frames to at least the capacity)"
        )


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

    On the sequential goal-memory path a ``ScreenshotCheckpointController`` owns
    compaction instead, and this eviction must never preempt it — the caller
    enforces that at setup via ``validate_single_eviction``.
    """
    recent_actions.append(action_text)
    recent_frames.append(frame)
    if len(recent_frames) > n_history_frames:
        keep = max(1, n_history_frames // 2)
        recent_frames[:] = recent_frames[-keep:]
        # Actions align to every frame but the latest.
        recent_actions[:] = recent_actions[-(len(recent_frames) - 1):] if len(recent_frames) > 1 else []


def compact_to_current(
    recent_frames: list[Image.Image],
    recent_actions: list[str],
) -> None:
    """Drop everything but the current screenshot after a stored checkpoint.

    The image-window twin of ``ScreenshotCheckpointController.reset_to_current``:
    the boundary screenshot stays (it is the last frame of the record just
    closed and the first frame of the one being opened), every older frame and
    every replayed assistant action go, and the causal state they carried lives
    on only in the checkpoint text now folded into the goal conditioning.

    Deliberately NOT bundled with ``reset_to_current``: the caller resets the
    controller whether or not the checkpoint reply parsed, but compacts only on
    success — an unparseable checkpoint must not silently discard the context it
    failed to summarize.
    """
    recent_frames[:] = recent_frames[-1:]
    recent_actions.clear()


def _interleave_messages(
    system_prompt: str,
    instruction: str | None,
    image_parts: list[Any],
    recent_actions: list[str] | None,
    current_text: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble the interleaved chat message list from per-frame ``image_parts``.

    Shared by ``_call_model`` (parts are base64 ``image_url`` dicts) and
    ``build_loggable_messages`` (parts are ``<image …>`` placeholders), so the
    persisted prompt trace is structurally identical to the payload actually
    sent — only the image representation differs. ``image_parts[i]`` is the
    representation of ``recent_frames[i]``; ``recent_actions[i]`` is the action
    that followed it (``len(recent_actions) == len(image_parts) - 1``).
    """
    recent_actions = recent_actions or []
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for i, part in enumerate(image_parts):
        content: list[Any] = []
        if i == 0 and instruction:
            content.append({"type": "text", "text": instruction})
        if i == len(image_parts) - 1 and current_text:
            content.append({"type": "text", "text": current_text})
        content.append(part)
        messages.append({"role": "user", "content": content})
        if i < len(recent_actions):
            messages.append({"role": "assistant", "content": recent_actions[i]})
    return messages


def _fresh_visual_messages(
    system_prompt: str,
    instruction: str | None,
    image_parts: list[Any],
    current_text: str | None = None,
) -> list[dict[str, Any]]:
    """One decision record: system + one user turn containing goal and images.

    This is the runtime twin of stage 04's ``--context-images`` mode. It never
    replays prior assistant actions, so every prediction is conditioned on a
    fresh visual state rather than on model-generated history.
    """
    content: list[Any] = []
    if instruction:
        content.append({"type": "text", "text": instruction})
    if current_text:
        content.append({"type": "text", "text": current_text})
    content.extend(image_parts)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


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
    fresh_visual_context: bool = False,
    current_text: str | None = None,
) -> list[dict[str, Any]]:
    """The message list as sent, but with each image replaced by ``<image name>``.

    Suitable for writing a small, human-readable ``prompt_NNN.json`` sidecar that
    lets you audit exactly what the model saw each step (turn sequence, verbatim
    text, instruction placement, eviction window) without duplicating base64
    image bytes — the referenced ``step_NNN.png`` files already hold the pixels.
    """
    parts = [{"type": "image", "image": f"<image {lbl}>"} for lbl in frame_labels]
    if fresh_visual_context:
        return _fresh_visual_messages(system_prompt, instruction, parts, current_text)
    return _interleave_messages(
        system_prompt, instruction, parts, recent_actions, current_text)


def step_messages(
    *,
    system_prompt: str,
    instruction: str | None,
    step: int,
    n_frames: int,
    recent_actions: list[str] | None,
    current_text: str | None = None,
    fresh_visual_context: bool = False,
) -> tuple[list[str], list[dict[str, Any]]]:
    """One turn's ``(frame_labels, loggable_messages)`` pair.

    Pure composition of ``window_frame_labels`` and ``build_loggable_messages``
    — the two are always used together (the labels both name the images in the
    message list and identify the current frame for logging), and every runner
    that assembles a turn needs the same pair. Factored out so the assembled
    turn sequence is testable without a VM or an sglang server: see
    ``eval/test_sequential_reply_contract.py``, which replays a whole
    goal → actions → checkpoint → resume flow through this function and asserts
    turn-by-turn identity with the Stage-04 record shape.
    """
    labels = window_frame_labels(step, n_frames)
    return labels, build_loggable_messages(
        system_prompt=system_prompt,
        instruction=instruction,
        recent_actions=recent_actions,
        frame_labels=labels,
        fresh_visual_context=fresh_visual_context,
        current_text=current_text,
    )


def _call_model(
    *,
    sglang_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    instruction: str | None,
    recent_frames: list[Image.Image],
    recent_actions: list[str] | None = None,
    fresh_visual_context: bool = False,
    current_text: str | None = None,
    sampling: SamplingParams,
    request_timeout_s: float = 120.0,
) -> tuple[str, str | None]:
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

    ``sampling`` is the single source of truth for decoding parameters (see
    ``eval/sampling.py``): the FULL Qwen-recommended tuple (temperature, top_p,
    top_k, repetition_penalty, presence_penalty, max_tokens) is sent to sglang,
    not just temperature — so the checkpoint's partial ``generation_config`` can
    no longer silently fill in the rest (top_p/top_k) or drop presence_penalty.

    Returns ``(content, finish_reason)``. ``finish_reason == "length"``
    means the reply was truncated at ``max_tokens`` — callers MUST NOT
    dispatch a truncated action (a half-emitted ``down(...)`` would leave a
    key held and trigger OS key-repeat).
    """
    image_parts = [
        {"type": "image_url", "image_url": {"url": _pil_to_data_url(f)}}
        for f in recent_frames
    ]
    messages = (
        _fresh_visual_messages(system_prompt, instruction, image_parts, current_text)
        if fresh_visual_context
        else _interleave_messages(
            system_prompt, instruction, image_parts, recent_actions, current_text)
    )
    r = requests.post(
        sglang_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": messages,
            **sampling.as_request_json(),
        },
        timeout=request_timeout_s,
    )
    r.raise_for_status()
    choice = r.json()["choices"][0]
    return choice["message"]["content"] or "", choice.get("finish_reason")


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
