"""Crowd-Cast keylogs to canonical ``deltatype_v2`` labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from grammars._support import Element
from grammars.deltatype_v2 import CODEC, DeltatypeV2Action
from pipeline.lib.common import ActionBin
from pipeline.lib.events import (
    DeadZone,
    LabeledEvent,
    RawEvent,
    Window,
    apply_label_policy,
)


@dataclass(frozen=True)
class WindowKeyboard:
    names: tuple[str, ...]


@dataclass(frozen=True)
class FormatResult:
    labels: list[str]
    keyboard: list[WindowKeyboard]


TEXT_KEYS = frozenset(
    {
        "Space",
        "BackQuote",
        "Minus",
        "Equal",
        "BracketLeft",
        "BracketRight",
        "BackSlash",
        "SemiColon",
        "Quote",
        "Comma",
        "Period",
        "Slash",
        *(f"Key{char}" for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        *(f"Num{digit}" for digit in "0123456789"),
    }
)


def _owned(labeled: Sequence[LabeledEvent]) -> list[LabeledEvent]:
    return sorted(
        (event for event in labeled if event.window is not None),
        key=lambda event: (event.label_t, event.event.seq),
    )


def _action(action_bin: ActionBin) -> DeltatypeV2Action:
    dx = round(action_bin.move_dx)
    dy = round(action_bin.move_dy)
    scroll = round(action_bin.scroll)
    if dx == 0 and dy == 0 and scroll == 0 and not action_bin.events:
        return DeltatypeV2Action(no_op=True)
    return DeltatypeV2Action(
        dx,
        dy,
        scroll,
        elements=tuple(
            Element("event", name=name, pressed=sign == "+")
            for sign, name in action_bin.events
        ),
    )


def format_segment(
    events: Sequence[RawEvent],
    windows: Sequence[Window],
    dead_zones: Sequence[DeadZone],
    *,
    master_fps: float,
) -> FormatResult:
    for event in events:
        if event.kind not in {"move", "scroll", "press", "release"}:
            raise ValueError(f"unexpected event kind: {event.kind!r}")
    labeled = apply_label_policy(events, windows, dead_zones, master_fps=master_fps)
    bins = [ActionBin() for _ in windows]
    for labeled_event in _owned(labeled):
        event = labeled_event.event
        action_bin = bins[labeled_event.window]
        if event.kind == "move":
            action_bin.move_dx += event.dx
            action_bin.move_dy += event.dy
        elif event.kind == "scroll":
            action_bin.scroll += event.scroll
        elif event.kind == "press":
            action_bin.events.append(("+", event.name))
        elif event.kind == "release":
            action_bin.events.append(("-", event.name))
        else:
            raise ValueError(f"unexpected event kind: {event.kind!r}")
    actions = [_action(action_bin) for action_bin in bins]
    return FormatResult(
        labels=[CODEC.format(action) for action in actions],
        keyboard=[
            WindowKeyboard(
                names=tuple(
                    element.name
                    for element in action.elements
                    if element.kind == "event"
                )
            )
            for action in actions
        ],
    )
