from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Sequence, TypeVar

Frame = TypeVar("Frame")


@dataclass
class Turn(Generic[Frame]):
    image: Frame
    output: str | None = None


@dataclass
class History(Generic[Frame]):
    n_history_frames: int = 16
    turns: list[Turn[Frame]] = field(default_factory=list)
    evicted: list[str] = field(default_factory=list)
    note: str | None = None

    def __post_init__(self) -> None:
        if self.n_history_frames < 1:
            raise ValueError("n_history_frames must be at least 1")

    def start(self, frame: Frame) -> None:
        self.turns = [Turn(frame)]
        self.evicted = []
        self.note = None

    def append(self, output: str, frame: Frame, note: str | None = None) -> None:
        if not self.turns:
            raise RuntimeError("History.append before History.start")
        self.turns[-1].output = output
        self.turns.append(Turn(frame))
        self.note = note
        if len(self.turns) > self.n_history_frames:
            dropped = self.turns.pop(0)
            if dropped.output is not None:
                self.evicted.append(dropped.output)

    @property
    def images(self) -> list[Frame]:
        return [turn.image for turn in self.turns]

    @property
    def outputs(self) -> list[str]:
        return [turn.output for turn in self.turns if turn.output is not None]

    @property
    def all_outputs(self) -> list[str]:
        return [*self.evicted, *self.outputs]

    @property
    def current(self) -> Frame:
        return self.turns[-1].image


def render_interleaved(
    *,
    history: History[Frame],
    system: str,
    instruction: str | None,
    image_part: Callable[[Frame], dict[str, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for index, turn in enumerate(history.turns):
        content: list[dict[str, Any]] = []
        if index == 0 and instruction:
            content.append({"type": "text", "text": instruction})
        content.append(image_part(turn.image))
        messages.append({"role": "user", "content": content})
        if turn.output is not None:
            messages.append({"role": "assistant", "content": turn.output})
    return messages


def replay_training_messages(
    *,
    turns: Sequence[tuple[Frame, str]],
    n_history_frames: int,
    system: str,
    instruction: str | None,
    image_part: Callable[[Frame], dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    if not turns:
        return []
    history: History[Frame] = History(n_history_frames=n_history_frames)
    history.start(turns[0][0])
    records: list[list[dict[str, Any]]] = []
    for index, (_, target) in enumerate(turns):
        messages = render_interleaved(
            history=history,
            system=system,
            instruction=instruction,
            image_part=image_part,
        )
        messages.append({"role": "assistant", "content": target})
        records.append(messages)
        if index + 1 < len(turns):
            history.append(target, turns[index + 1][0])
    return records
