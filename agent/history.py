"""History policy: the injected object that decides what the model sees each step.

Before this module the history shape was implicit in whichever episode driver you
had forked, which is precisely *why* people forked:

  * `eval/freeroll.py` + `eval/osworld_grounding_runner.py` (via
    `osworld_runtime._interleave_messages` / `append_turn`) — one user turn per
    frame, the model's prior output fed back as the following assistant turn,
    StreamingLLM *block* eviction at `n_history_frames`, and the goal re-anchored
    on the earliest in-window user turn every step (`persist_instruction`).
  * `sign_of_life_v2/compact_relative.build_phaseb_messages` — at most five
    images; actions evicted out of the image window survive as prose in a
    `Previous actions:` block on the first user turn; the instruction rides that
    same block.
  * `rl/osworld/harness.py` (target_box) — the full conversation accumulates but
    every image older than the newest is rewritten to a placeholder.
  * `rl/movebox/rollout.py` — no history at all; one fresh single-turn prompt per
    step, the pixel cursor being the only threaded state.

All four are here as `HistoryPolicy` implementations over one mutable `History`
window. The policy only *renders*; the window owns append and eviction.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

import verifiers.v1 as vf

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


# --------------------------------------------------------------------------- #
# image budget
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ImageBudget:
    """How many images may ride a prompt, and how each is encoded.

    `max_images` is the hard cap the *policy* honours; the window's
    `n_history_frames` is the eviction trigger. They are separate because the
    Phase-B contract evicts at 5 images while keeping older *actions*, and
    target_box accumulates turns while sending exactly one image.

    JPEG at quality 85 is what every runner used (`_pil_to_data_url`); PNG is
    kept for the RL renderers, which composite synthetic markers and want them
    lossless.
    """

    max_images: int = 16
    media: Literal["jpeg", "png"] = "jpeg"
    quality: int = 85
    max_pixels: int = 0
    """If > 0, downscale (preserving aspect) until w*h <= max_pixels. 0 = never."""

    def data_url(self, image: bytes) -> str:
        payload, mime = self._encode(image)
        return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"

    def image_part(self, image: bytes) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": self.data_url(image)}}

    def _encode(self, image: bytes) -> tuple[bytes, str]:
        if self.media == "png" and not self.max_pixels:
            return image, "image/png"
        from PIL import Image  # local: keep PIL off the import path of pure-text users

        with Image.open(io.BytesIO(image)) as handle:
            frame = handle.convert("RGB")
            if self.max_pixels and frame.width * frame.height > self.max_pixels:
                scale = (self.max_pixels / (frame.width * frame.height)) ** 0.5
                frame = frame.resize(
                    (max(1, int(frame.width * scale)), max(1, int(frame.height * scale))),
                    Image.LANCZOS,
                )
            buffer = io.BytesIO()
            if self.media == "png":
                frame.save(buffer, format="PNG")
                return buffer.getvalue(), "image/png"
            frame.save(buffer, format="JPEG", quality=self.quality, optimize=False)
            return buffer.getvalue(), "image/jpeg"


# --------------------------------------------------------------------------- #
# the window
# --------------------------------------------------------------------------- #


@dataclass
class Turn:
    """One completed turn: the frame the model saw, and the text it produced from it."""

    image: bytes
    output: str | None = None


@dataclass
class History:
    """The rolling (frame, action) window, with StreamingLLM block eviction.

    Invariant: `turns[-1].output is None` — the newest frame is the current screen
    and has no action yet. Every earlier turn has the action that followed it.
    That is exactly `osworld_runtime.append_turn`'s
    `len(recent_actions) == len(recent_frames) - 1`.

    Eviction is *block* eviction, not slide-by-one: once the window exceeds
    `n_history_frames` we keep the newest `n_history_frames // 2`. This preserves
    the server-side prefix cache — while the window grows the prompt is
    append-only, and only ~N/2 frames are re-prefilled, roughly once every N/2
    steps. Slide-by-one invalidates the whole window every step past N.
    """

    n_history_frames: int = 16
    turns: list[Turn] = field(default_factory=list)
    evicted: list[str] = field(default_factory=list)
    """Outputs that have fallen out of the window, oldest first, in order."""

    def start(self, frame: bytes) -> None:
        self.turns = [Turn(frame)]
        self.evicted = []

    def append(self, output: str, frame: bytes) -> None:
        """Record the action taken from the current frame, then the resulting frame."""
        if not self.turns:
            raise RuntimeError("History.append before History.start")
        self.turns[-1].output = output
        self.turns.append(Turn(frame))
        if len(self.turns) > self.n_history_frames:
            keep = max(1, self.n_history_frames // 2)
            self.evicted.extend(
                turn.output or "" for turn in self.turns[:-keep] if turn.output is not None
            )
            self.turns = self.turns[-keep:]

    @property
    def images(self) -> list[bytes]:
        return [turn.image for turn in self.turns]

    @property
    def outputs(self) -> list[str]:
        return [turn.output for turn in self.turns if turn.output is not None]

    @property
    def all_outputs(self) -> list[str]:
        return [*self.evicted, *self.outputs]

    @property
    def current(self) -> bytes:
        return self.turns[-1].image


# --------------------------------------------------------------------------- #
# policies
# --------------------------------------------------------------------------- #


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
    messages: vf.Messages = [vf.SystemMessage(content=system)]
    for index, part in enumerate(parts):
        content: list[Any] = []
        if index == 0 and instruction:
            content.append({"type": "text", "text": instruction})
        content.append(part)
        messages.append(vf.UserMessage(content=content))
        if index < len(outputs):
            messages.append(vf.AssistantMessage(content=outputs[index]))
    return messages


@dataclass(frozen=True)
class InterleavedFrames:
    """freeroll / grounding / OSWorld-task shape.

    One user turn per in-window frame, the model's prior text as the following
    assistant turn. `persist_instruction=True` re-anchors the goal on the
    earliest in-window user turn *every* step so it survives eviction of the
    first frame; `False` reverts to goal-on-step-1, which is the training
    distribution.
    """

    persist_instruction: bool = True
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
        images = history.images[-budget.max_images :]
        outputs = history.outputs[-(len(images) - 1) :] if len(images) > 1 else []
        goal = instruction if (step == 1 or self.persist_instruction) else None
        return _interleave(
            system=system,
            instruction=goal,
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
    instruction. The image comes *before* the text on that first turn, matching
    the sealed teacher-forced evaluator.
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
        # `images` is the in-window frames; `outputs` is every action ever taken. The
        # two live in different index spaces the moment `History` block-evicts, and
        # `len(history.evicted)` is exactly the number of frames that left the window
        # (one output per dropped turn). Comparing `outputs` against `images` alone
        # raised for every window past `n_history_frames` — fatal for any rollout
        # longer than the window under this policy — and indexing `outputs` with a
        # position derived from `images` paired the wrong action with each frame.
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
    replacement per message. Cheap in tokens, and it was the only shape that kept
    a VM-in-the-loop 10-step rollout inside the renderer's image cache.
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
    state is whatever the environment carries (the pixel cursor). This is a
    deliberate ablation partner for `InterleavedFrames`, not an oversight — a
    stateless prompt makes the per-step grounding decision identifiable.
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


def history_policy(name: str, **kwargs: Any) -> HistoryPolicy:
    """Build a policy by name. This is the whole point of the module: the A/B is a
    config field (`--harness.history.name=prose_summarised_window`), not a fork."""
    try:
        factory = POLICIES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown history policy {name!r}; known: {sorted(POLICIES)}"
        ) from exc
    return factory(**kwargs)
