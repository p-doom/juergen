"""The ordered-events-v3 action contract used by this traced CUA stream."""

from __future__ import annotations

from typing import Any

from ._support import CONTROL_SPEC, CONTROL_TOKEN, Control, NoAction, split_control

GRAMMAR_NAME = "ordered_events_v3"
THINKING_PREAMBLE = (
    "For each step, first reason in a single <think>...</think> block — your current "
    "sub-goal and what you observe on the screen — then a one-line `Action:` describing "
    "the move, then the action itself.\n\n"
)


def available() -> tuple[str]:
    return (GRAMMAR_NAME,)


def load(name: str) -> Any:
    if name != GRAMMAR_NAME:
        raise KeyError(f"unknown grammar {name!r}; required: {GRAMMAR_NAME!r}")
    from .ordered_events_v3.codec import CODEC

    return CODEC


def codecs() -> dict[str, Any]:
    return {GRAMMAR_NAME: load(GRAMMAR_NAME)}


def describe(name: str) -> str:
    return load(name).describe()


def system_prompt(codec: Any, *, thinking: bool) -> str:
    described = codec.describe()
    return (THINKING_PREAMBLE + described) if thinking else described


__all__ = [
    "CONTROL_SPEC",
    "CONTROL_TOKEN",
    "GRAMMAR_NAME",
    "THINKING_PREAMBLE",
    "Control",
    "NoAction",
    "available",
    "codecs",
    "describe",
    "load",
    "split_control",
    "system_prompt",
]
