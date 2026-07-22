"""Action formatters: (events, windows, dead_zones) -> one label per window.

This is the single place the on-disk keylog becomes assistant-turn text, so
action-format ablations are a stage-04 flag, not a pipeline rerun. A formatter
receives the FULL raw event stream (state layer) plus the dead-zone label
policy's dispositions (label layer) via ``events.apply_label_policy`` — a
stateful format (e.g. cumulative cursor position) folds over every event while
emitting labels only for owned ones.

``CanonicalFormatter`` reproduces the historical format byte-for-byte on
dead-zone-free stretches (``common.format_action`` over per-window bins with
segment-global held-set dedup) — that identity is the regression gate
(tests/test_action_format.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from realigned_pipeline.lib.common import ActionBin, format_action
from realigned_pipeline.lib.events import (
    DeadZone,
    LabeledEvent,
    PolicyCounters,
    RawEvent,
    Window,
    apply_label_policy,
)


@dataclass
class FormatResult:
    labels: list[str]  # one per window, same order
    counters: PolicyCounters


class ActionFormatter(Protocol):
    name: str

    def format_segment(
        self,
        events: Sequence[RawEvent],
        windows: Sequence[Window],
        dead_zones: Sequence[DeadZone],
        *,
        master_fps: float,
    ) -> FormatResult: ...


def _ordered_owned(labeled: Sequence[LabeledEvent]) -> list[LabeledEvent]:
    """Owned events in label order: by clamped label time, ties by original
    stream position (so a press clamped onto a tick boundary precedes that
    tick's native events, and a clamped release follows its window's)."""
    return sorted(
        (le for le in labeled if le.window is not None),
        key=lambda le: (le.label_t, le.event.seq),
    )


class CanonicalFormatter:
    """Per-window ``round(dx) round(dy) round(scroll) [; +KEY -KEY]`` / ``NO_OP``."""

    name = "canonical"

    def format_segment(
        self,
        events: Sequence[RawEvent],
        windows: Sequence[Window],
        dead_zones: Sequence[DeadZone],
        *,
        master_fps: float,
    ) -> FormatResult:
        labeled, counters = apply_label_policy(
            events, windows, dead_zones, master_fps=master_fps
        )
        bins = [ActionBin() for _ in windows]
        for le in _ordered_owned(labeled):
            e = le.event
            b = bins[le.window]
            if e.kind == "move":
                b.move_dx += e.dx
                b.move_dy += e.dy
            elif e.kind == "scroll":
                b.scroll += e.scroll
            elif e.kind == "press":
                b.events.append(("+", e.name))
            else:
                b.events.append(("-", e.name))
        return FormatResult(
            labels=[format_action(b) for b in bins],
            counters=counters,
        )


FORMATTERS: dict[str, ActionFormatter] = {
    "canonical": CanonicalFormatter(),
}


def get_formatter(name: str) -> ActionFormatter:
    try:
        return FORMATTERS[name]
    except KeyError:
        raise KeyError(
            f"unknown action format {name!r} (available: {sorted(FORMATTERS)})"
        ) from None
