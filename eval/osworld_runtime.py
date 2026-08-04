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

from sampling import SamplingParams
from shortgoal_grammar import FRAME_JPEG_QUALITY, IMAGE_PLACEHOLDER, K_IMAGES, KEEP_IMAGES

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
        content.append(part)
        messages.append({"role": "user", "content": content})
        if i < len(recent_actions):
            messages.append({"role": "assistant", "content": recent_actions[i]})
    return messages


def _fresh_visual_messages(
    system_prompt: str,
    instruction: str | None,
    image_parts: list[Any],
) -> list[dict[str, Any]]:
    """One decision record: system + one user turn containing goal and images.

    This is the runtime twin of stage 04's ``--context-images`` mode. It never
    replays prior assistant actions, so every prediction is conditioned on a
    fresh visual state rather than on model-generated history.
    """
    content: list[Any] = []
    if instruction:
        content.append({"type": "text", "text": instruction})
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
) -> list[dict[str, Any]]:
    """The message list as sent, but with each image replaced by ``<image name>``.

    Suitable for writing a small, human-readable ``prompt_NNN.json`` sidecar that
    lets you audit exactly what the model saw each step (turn sequence, verbatim
    text, instruction placement, eviction window) without duplicating base64
    image bytes — the referenced ``step_NNN.png`` files already hold the pixels.
    """
    parts = [{"type": "image", "image": f"<image {lbl}>"} for lbl in frame_labels]
    if fresh_visual_context:
        return _fresh_visual_messages(system_prompt, instruction, parts)
    return _interleave_messages(system_prompt, instruction, parts, recent_actions)


def keep_text_eviction_points(
    n_frames: int,
    k: int = K_IMAGES,
    keep: int = KEEP_IMAGES,
) -> list[int]:
    """Frame indices whose arrival makes keep-text block eviction fire.

    Pure and deterministic. Appending frame ``j`` makes it the ``j+1``-th live
    image, so the first eviction is at ``j == k`` (live *would* be ``k+1``); it
    drops the oldest live images in one block down to ``keep``, hence the next
    eviction lands ``k + 1 - keep`` frames later. Exposed because the training
    record builder cuts one record per eviction point — a record's context is
    the window as it stands right after an eviction — so runtime window and
    record boundaries come from this single function.
    """
    if not isinstance(n_frames, int) or n_frames < 1:
        raise ValueError(f"n_frames must be a positive int, got {n_frames!r}")
    _check_keep_text_window(k, keep)
    step = k + 1 - keep
    return [j for j in range(k, n_frames) if (j - k) % step == 0]


def _check_keep_text_window(k: int, keep: int) -> None:
    if not isinstance(k, int) or not isinstance(keep, int) or not 1 <= keep < k:
        raise ValueError(f"keep-text window needs 1 <= keep < k, got k={k!r} keep={keep!r}")


def _require_live_frame(frame: Image.Image) -> Image.Image:
    if frame is None:
        raise ValueError("a keep-text turn needs a live frame (None marks an evicted one)")
    return frame


class KeepTextWindow:
    """Keep-all-text context window: every action line stays, only pixels expire.

    ``append_turn``'s rolling window drops old text together with old frames;
    this sibling keeps the whole episode's action history and expires images
    only. ``frames[i] is None`` marks an evicted frame: its user turn keeps its
    position and renders the fixed ``IMAGE_PLACEHOLDER`` text instead of an
    image, so assistant turns never move and the prompt is append-only between
    evictions (RadixAttention prefix reuse, exactly the reason eviction is a
    block drop to ``keep`` rather than a slide-by-one).
    """

    def __init__(
        self,
        frame: Image.Image,
        *,
        k: int = K_IMAGES,
        keep: int = KEEP_IMAGES,
    ) -> None:
        _check_keep_text_window(k, keep)
        self.k = k
        self.keep = keep
        self.frames: list[Image.Image | None] = [_require_live_frame(frame)]
        self.actions: list[str] = []
        self.evicted_at: list[int] = []

    def live_count(self) -> int:
        """How many user turns still carry an image."""
        return sum(f is not None for f in self.frames)

    def liveness(self) -> list[bool]:
        """Per-frame liveness, positionally aligned with ``frames``."""
        return [f is not None for f in self.frames]

    def frame_labels(self) -> list[str]:
        """The PNG filename of every frame, evicted ones included."""
        return keep_text_frame_labels(len(self.frames))

    def append_turn(self, action_text: str, frame: Image.Image) -> None:
        """Record the action produced from the current frame, plus the frame it led to."""
        if not isinstance(action_text, str) or not action_text:
            raise ValueError(f"action text must be a non-empty str, got {action_text!r}")
        _require_live_frame(frame)
        self.actions.append(action_text)
        self.frames.append(frame)
        if self.live_count() > self.k:
            live = [i for i, f in enumerate(self.frames) if f is not None]
            for i in live[:-self.keep]:
                self.frames[i] = None
            self.evicted_at.append(len(self.frames) - 1)


def keep_text_frame_labels(n_frames: int) -> list[str]:
    """PNG filenames of a keep-text window — always the whole episode from frame 0.

    Unlike ``window_frame_labels`` (a contiguous tail), a keep-text window keeps
    every turn's position for the whole episode, so the ids are ``[0 .. n-1]``.
    """
    if not isinstance(n_frames, int) or n_frames < 1:
        raise ValueError(f"n_frames must be a positive int, got {n_frames!r}")
    return [f"step_{i:03d}.png" for i in range(n_frames)]


def keep_text_messages(
    system_prompt: str,
    goal: str | None,
    frame_parts: list[dict[str, Any] | None],
    actions: list[str] | None,
) -> list[dict[str, Any]]:
    """Assemble the keep-text message list; ``frame_parts[i] is None`` == evicted.

    Shape: system, then one user turn per frame carrying that frame's image part
    — or the ``IMAGE_PLACEHOLDER`` text part once evicted — with the GOAL text
    pinned to the FIRST user turn only, and ``actions[i]`` as the assistant turn
    following frame ``i``. The last frame is the current screen and has no
    action yet, so ``len(actions) == len(frame_parts) - 1``::

        [ system,
          user[goal? + frame_0 | placeholder], assistant[action_0],
          user[frame_1 | placeholder],         assistant[action_1],
          ...
          user[frame_{N-1}] ]                  # current screen, always live

    A training record instead ends on its own final assistant turn (the
    ``TERMINATE`` that follows the last frame), so ``len(actions) ==
    len(frame_parts)`` is accepted too — those are the only two legal shapes.

    Public because the training-record builder assembles the same list from
    file-path image refs: same structure, same text, same placement — which is
    what makes a record and the runtime prompt for that step byte-identical.
    """
    actions = actions or []
    if not frame_parts or len(actions) not in (len(frame_parts) - 1, len(frame_parts)):
        raise ValueError(
            "keep-text needs len(actions) == len(frame_parts) - 1 (runtime) or "
            f"len(frame_parts) (trailing TERMINATE), got {len(actions)} actions "
            f"and {len(frame_parts)} frames"
        )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for i, part in enumerate(frame_parts):
        content: list[Any] = []
        if i == 0 and goal:
            content.append({"type": "text", "text": goal})
        content.append(part if part is not None else {"type": "text", "text": IMAGE_PLACEHOLDER})
        messages.append({"role": "user", "content": content})
        if i < len(actions):
            messages.append({"role": "assistant", "content": actions[i]})
    return messages


def build_keep_text_messages(
    *,
    system_prompt: str,
    goal: str | None,
    frames: list[Image.Image | None],
    actions: list[str] | None,
    quality: int = FRAME_JPEG_QUALITY,
) -> list[dict[str, Any]]:
    """The keep-text message list as sent: live frames as base64 JPEG data URLs.

    ``quality`` defaults to the training frame quality, NOT ``_pil_to_data_url``'s
    legacy 85 (which stays as it is for ``append_turn``'s callers): a closed-loop
    prompt must carry the same JPEG bytes as the training record for that step,
    or rung 1(b) fails on pixels the checkpoint never saw.
    """
    parts = [
        None if f is None
        else {"type": "image_url", "image_url": {"url": _pil_to_data_url(f, quality=quality)}}
        for f in frames
    ]
    return keep_text_messages(system_prompt, goal, parts, actions)


def build_loggable_keep_text_messages(
    *,
    system_prompt: str,
    goal: str | None,
    actions: list[str] | None,
    frame_labels: list[str],
    liveness: list[bool],
) -> list[dict[str, Any]]:
    """The keep-text list as sent, with each live image replaced by ``<image name>``.

    Evicted turns keep their placeholder text verbatim, so the persisted
    ``prompt_NNN.json`` sidecar shows exactly which turns went blind and when,
    without duplicating image bytes (see ``build_loggable_messages``).
    """
    parts = [
        {"type": "image", "image": f"<image {lbl}>"} if live else None
        for lbl, live in zip(frame_labels, liveness, strict=True)
    ]
    return keep_text_messages(system_prompt, goal, parts, actions)


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
    sampling: SamplingParams,
    seed: int | None = None,
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

    ``seed``, when given, rides along in the request so a sampled rollout is
    reproducible run to run; greedy passes leave it ``None``.

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
        _fresh_visual_messages(system_prompt, instruction, image_parts)
        if fresh_visual_context
        else _interleave_messages(system_prompt, instruction, image_parts, recent_actions)
    )
    request_json = {
        "model": model,
        "messages": messages,
        **sampling.as_request_json(),
    }
    if seed is not None:
        request_json["seed"] = int(seed)
    r = requests.post(
        sglang_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=request_json,
        timeout=request_timeout_s,
    )
    r.raise_for_status()
    choice = r.json()["choices"][0]
    return choice["message"]["content"] or "", choice.get("finish_reason")


def call_model_messages(
    *,
    sglang_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    sampling: SamplingParams,
    seed: int | None = None,
    request_timeout_s: float = 120.0,
) -> tuple[str, str | None]:
    """One chat-completion call for an already-assembled message list.

    The keep-text path assembles its own messages (``build_keep_text_messages``),
    so it needs the request half of ``_call_model`` without the frame/action
    interleaving. Decoding parameters, seed handling and the
    ``(content, finish_reason)`` contract are identical — including the rule that
    a ``finish_reason == "length"`` reply MUST NOT be dispatched.
    """
    request_json = {
        "model": model,
        "messages": messages,
        **sampling.as_request_json(),
    }
    if seed is not None:
        request_json["seed"] = int(seed)
    r = requests.post(
        sglang_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=request_json,
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
