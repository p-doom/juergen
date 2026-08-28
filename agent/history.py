"""History policy: the injected object that decides what the model sees each step.

The four shapes in the tree, and where each came from:

  * `eval/freeroll.py` + `eval/osworld_grounding_runner.py` (via
    `osworld_runtime._interleave_messages` / `append_turn`) — one user turn per
    frame, the model's prior output fed back as the following assistant turn,
    slide-by-one eviction at `n_history_frames`, and the goal re-anchored on the
    earliest in-window user turn every step.
  * `sign_of_life_v2/compact_relative.build_phaseb_messages` — at most five
    images; actions evicted out of the image window survive as prose in a
    `Previous actions:` block on the first user turn; the instruction rides that
    same block.
  * `rl/osworld/harness.py` (target_box) — the full conversation accumulates but
    every image older than the newest is rewritten to a placeholder.
  * `rl/movebox/rollout.py` — no history at all; one fresh single-turn prompt per
    step, the pixel cursor being the only threaded state.

All four are `HistoryPolicy` implementations over one mutable `History` window.
The policy only renders; the window owns append and eviction.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import verifiers.v1 as vf

from history_policy import History, Turn, render_interleaved
from image_domain import OSWORLD_CURSOR_JPEG_DOMAIN

__all__ = [
    "History",
    "HistoryPolicy",
    "ImageBudget",
    "InterleavedFrames",
    "LatestImageOnly",
    "POLICIES",
    "ProseSummarisedWindow",
    "StatelessSingleTurn",
    "Turn",
    "history_policy",
    "prose_summary",
]

IMAGE_PLACEHOLDER = "Previous screenshot omitted."


@dataclass(frozen=True)
class ImageBudget:
    """How many received desktop images may ride a prompt."""

    max_images: int = 16

    def data_url(self, image: bytes) -> str:
        if not image.startswith(b"\xff\xd8\xff"):
            raise ValueError(
                "desktop observations must be "
                f"{OSWORLD_CURSOR_JPEG_DOMAIN}, not a non-JPEG payload"
            )
        return f"data:image/jpeg;base64,{base64.b64encode(image).decode('ascii')}"

    def image_part(self, image: bytes) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": self.data_url(image)}}


@runtime_checkable
class HistoryPolicy(Protocol):
    """Renders a window into wire messages. Stateless: all state is in `History`."""

    name: str

    def render(
        self,
        *,
        history: History,
        system: str,
        instruction: str | None,
        step: int,
        budget: ImageBudget,
    ) -> vf.Messages: ...


def _interleave(
    *,
    system: str,
    instruction: str | None,
    parts: Sequence[Any],
    outputs: Sequence[str],
) -> vf.Messages:
    """[system, user(instr? + img0), assistant(out0), user(img1), ...].

    `parts[i]` represents frame i and `outputs[i]` is the action that followed it,
    so `len(outputs) == len(parts) - 1`. Verbatim structure of
    `osworld_runtime._interleave_messages`.
    """
    history: History[Any] = History(n_history_frames=len(parts))
    history.start(parts[0])
    for index, output in enumerate(outputs):
        history.append(output, parts[index + 1])
    wire = render_interleaved(
        history=history,
        system=system,
        instruction=instruction,
        image_part=lambda part: part,
    )
    return [
        vf.SystemMessage(content=message["content"])
        if message["role"] == "system"
        else vf.UserMessage(content=message["content"])
        if message["role"] == "user"
        else vf.AssistantMessage(content=message["content"])
        for message in wire
    ]


@dataclass(frozen=True)
class InterleavedFrames:
    """freeroll / grounding / OSWorld-task shape.

    One user turn per in-window frame, the model's prior text as the following
    assistant turn. The goal rides the earliest in-window user turn.
    """

    name: str = "interleaved_frames"

    def render(
        self,
        *,
        history: History,
        system: str,
        instruction: str | None,
        step: int,
        budget: ImageBudget,
    ) -> vf.Messages:
        del step
        images = history.images[-budget.max_images :]
        outputs = history.outputs[-(len(images) - 1) :] if len(images) > 1 else []
        return _interleave(
            system=system,
            instruction=instruction,
            parts=[budget.image_part(image) for image in images],
            outputs=outputs,
        )


def prose_summary(raw_output: str) -> str:
    """Recover the natural-language action description Phase-B history carries.

    The checkpoint emits reasoning prose then a final bare action line; the prose
    is what an evicted turn is remembered by. Verbatim from
    `sign_of_life_v2/compact_relative._prose_summary`.
    """
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if len(lines) < 2:
        return lines[0] if lines else "No parseable action description."
    prose = " ".join(lines[:-1]).strip()
    return prose.removeprefix("Action:").strip() or "No action description."


@dataclass(frozen=True)
class ProseSummarisedWindow:
    """Phase-B deltatype-v2 shape (`build_phaseb_messages`).

    At most `budget.max_images` images (the records the checkpoint trained on hold
    five: four completed turns plus the current screen). Actions older than the
    image window are not dropped — they are summarised to prose and listed in a
    `Previous actions:` block on the first user turn, which also carries the
    instruction. The image comes before the text on that first turn, matching the
    sealed teacher-forced evaluator.
    """

    header: str = (
        "\nPlease generate the next move according to the UI screenshot, "
        "instruction and previous actions.\n"
    )
    name: str = "prose_summarised_window"

    def render(
        self,
        *,
        history: History,
        system: str,
        instruction: str | None,
        step: int,
        budget: ImageBudget,
    ) -> vf.Messages:
        del step
        images = history.images
        outputs = history.all_outputs
        # `images` is the in-window frames; `outputs` is every action ever taken.
        # The two live in different index spaces once `History` evicts,
        # and `len(history.evicted)` is exactly the number of frames that left the
        # window (one output per dropped turn). Comparing `outputs` against
        # `images` alone raised for every window past `n_history_frames`, and
        # indexing `outputs` with a position derived from `images` paired the wrong
        # action with each frame.
        dropped = len(history.evicted)
        if len(outputs) != dropped + len(images) - 1:
            raise ValueError(
                "prose-summarised history requires one action per completed frame "
                f"(evicted={dropped}, frames={len(images)}, outputs={len(outputs)})"
            )
        first = max(0, len(images) - max(1, budget.max_images))
        offset = dropped + first
        visible_images, visible_outputs = images[first:], outputs[offset:]
        earlier = outputs[:offset]
        previous = (
            "\n".join(
                f"Step {index}: {prose_summary(raw)}"
                for index, raw in enumerate(earlier, start=1)
            )
            if earlier
            else "None"
        )
        first_text = (
            f"{self.header}\nInstruction: {instruction or ''}"
            f"\n\nPrevious actions:\n{previous}"
        )
        messages: vf.Messages = [vf.SystemMessage(content=system)]
        for index, image in enumerate(visible_images):
            content: list[Any] = [budget.image_part(image)]
            if index == 0:
                content.append({"type": "text", "text": first_text})
            messages.append(vf.UserMessage(content=content))
            if index < len(visible_outputs):
                # `vf.AssistantMessage.content` is `str | None` (`types.py:79`), not
                # `MessageContent` — only user/system/tool turns take content parts. A
                # one-element text list raised `ValidationError` here for every window
                # with a completed turn, i.e. every step after the first.
                messages.append(vf.AssistantMessage(content=visible_outputs[index]))
        return messages


@dataclass(frozen=True)
class LatestImageOnly:
    """target_box shape: accumulate every turn, send only the newest image.

    The conversation grows (so the model's own reasoning and tool results stay in
    context) but each older image part is rewritten to a text placeholder, one
    replacement per message. Cheap in tokens, and the only shape that kept a
    VM-in-the-loop 10-step rollout inside the renderer's image cache.
    """

    initial_suffix: str = "Initial observation."
    later_suffix: str = "Newest observation."
    name: str = "latest_image_only"

    def render(
        self,
        *,
        history: History,
        system: str,
        instruction: str | None,
        step: int,
        budget: ImageBudget,
    ) -> vf.Messages:
        del step
        images = history.images
        outputs = history.outputs
        messages: vf.Messages = [vf.SystemMessage(content=system)]
        for index, image in enumerate(images):
            newest = index == len(images) - 1
            text = (
                f"Instruction: {instruction}\n{self.initial_suffix}"
                if index == 0 and instruction
                else (self.later_suffix if newest else IMAGE_PLACEHOLDER)
            )
            content: list[Any] = [{"type": "text", "text": text}]
            if newest:
                content.append(budget.image_part(image))
            messages.append(vf.UserMessage(content=content))
            if index < len(outputs):
                messages.append(vf.AssistantMessage(content=outputs[index]))
        return messages


@dataclass(frozen=True)
class StatelessSingleTurn:
    """movebox / single-step grounding shape: no history at all.

    One fresh `[system, user(text + image)]` prompt per step. The only threaded
    state is whatever the environment carries (the pixel cursor). An ablation
    partner for `InterleavedFrames`: a stateless prompt makes the per-step
    grounding decision identifiable.
    """

    name: str = "stateless_single_turn"

    def render(
        self,
        *,
        history: History,
        system: str,
        instruction: str | None,
        step: int,
        budget: ImageBudget,
    ) -> vf.Messages:
        del step
        content: list[Any] = []
        if instruction:
            content.append({"type": "text", "text": instruction})
        content.append(budget.image_part(history.current))
        return [vf.SystemMessage(content=system), vf.UserMessage(content=content)]


POLICIES: dict[str, Any] = {
    "interleaved_frames": InterleavedFrames,
    "prose_summarised_window": ProseSummarisedWindow,
    "latest_image_only": LatestImageOnly,
    "stateless_single_turn": StatelessSingleTurn,
}


def history_policy(name: str) -> HistoryPolicy:
    """Build a policy by name, so the A/B is a config field
    (`--harness.history.name=prose_summarised_window`) rather than a fork."""
    try:
        factory = POLICIES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown history policy {name!r}; known: {sorted(POLICIES)}"
        ) from exc
    return factory()
