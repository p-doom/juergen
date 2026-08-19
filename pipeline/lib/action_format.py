"""Action formatters: (events, windows, dead_zones) -> one label per window.

This is the single place the on-disk keylog becomes assistant-turn text. A
formatter receives the full raw event stream (state layer) plus the dead-zone
label policy's dispositions (label layer) via ``events.apply_label_policy`` — a
stateful format (e.g. cumulative cursor position) folds over every event while
emitting labels only for owned ones.

A formatter owns the extraction (binning, motor grid, label policy) and nothing
else: the label text itself is rendered by the codec of the grammar named in its
``grammar`` attribute, and stage 04 takes that grammar's ``describe()`` as the
system prompt. So the emitter cannot write a line the eval parser calls
malformed, and cannot be trained under a prompt describing another syntax.

``CanonicalFormatter`` reproduces the historical format byte-for-byte on
dead-zone-free stretches (``common.format_action`` over per-window bins with
segment-global held-set dedup) — that identity is the regression gate
(tests/test_action_format.py).

``OrderedFormatter`` (``ordered_events_v2``, ported from the yll/action-format
branch) preserves the relative order of movement, scrolling, and key/button
transitions inside each window as one mini-program:
``move(4,-1); down(LMB); move(2,0); up(LMB)``. Continuous motion is
accumulated on an internal ``continuous_action_hz`` motor grid (default 10 Hz
== 100 ms ticks; not a frame rate); every press/release is an ordering barrier
at its exact position. The aggregate format cannot represent
``move -> click -> move``; this one can. Held-state anomaly accounting
(redundant press / dangling release / held at end) lives in the shared label
policy's ``PolicyCounters``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from grammars._support import Element
from grammars.deltatype_v2 import CODEC as DELTATYPE_V2
from grammars.deltatype_v2 import DeltatypeV2Action
from grammars.ordered_events_v3 import CODEC as ORDERED_EVENTS_V3
from grammars.ordered_events_v3 import OrderedEventsV3Action, Primitive

from pipeline.lib.common import ActionBin
from pipeline.lib.events import (
    DeadZone,
    LabeledEvent,
    PolicyCounters,
    RawEvent,
    Window,
    apply_label_policy,
)

DEFAULT_CONTINUOUS_ACTION_HZ = 10.0


@dataclass
class FormatResult:
    labels: list[str]  # one per window, same order
    counters: PolicyCounters
    # Provenance for primitive-based formats (None for the aggregate format).
    primitive_counts: dict[str, int] | None = None


class ActionFormatter(Protocol):
    name: str
    #: The grammar whose codec renders this formatter's labels and whose
    #: ``describe()`` is the system prompt trained against them.
    grammar: str

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


def _render_bin(action_bin: ActionBin) -> str:
    """One aggregate bin -> its ``deltatype_v2`` label.

    The subset of that grammar this formatter can reach: no ``type()``, no
    ``MOVE()`` drag, and NO_OP as the only control token — a keylog records no
    intent to terminate.
    """
    dx = round(action_bin.move_dx)
    dy = round(action_bin.move_dy)
    scroll = round(action_bin.scroll)
    if dx == 0 and dy == 0 and scroll == 0 and not action_bin.events:
        return DELTATYPE_V2.format(DeltatypeV2Action(no_op=True))
    return DELTATYPE_V2.format(
        DeltatypeV2Action(
            dx,
            dy,
            scroll,
            elements=tuple(
                Element("event", name=name, pressed=sign == "+")
                for sign, name in action_bin.events
            ),
        )
    )


class CanonicalFormatter:
    """Per-window ``round(dx) round(dy) round(scroll) [; +KEY -KEY]`` / ``NO_OP``."""

    name = "canonical"
    grammar = "deltatype_v2"

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
            labels=[_render_bin(b) for b in bins],
            counters=counters,
        )


def _render_primitives(primitives: Sequence[Primitive]) -> str:
    """One window's primitives -> its ``ordered_events_v3`` label.

    An empty window is NO_OP; the grammar's TERMINATE is unreachable from a
    keylog, which records no intent to terminate.
    """
    return ORDERED_EVENTS_V3.format(
        OrderedEventsV3Action(primitives=tuple(primitives), no_op=not primitives)
    )


class OrderedFormatter:
    """Per-window ordered mini-program: ``primitive(; primitive)*`` / ``NO_OP``
    with primitives ``move(dx,dy)``, ``scroll(dx,dy)``, ``down(NAME)``,
    ``up(NAME)``. A click is ``down(LMB); up(LMB)`` — no click/type/wait/timing
    primitives, no invented transitions.

    Continuous events accumulate per (window, motor tick, kind) in floating
    point and round once at flush; the accumulator flushes when the tick or
    kind changes, a discrete transition occurs, or the window ends. ``move(0,0)``
    and ``scroll(0,0)`` are omitted; an empty window renders ``NO_OP``."""

    name = "ordered_events_v2"
    grammar = "ordered_events_v3"

    def __init__(self, continuous_action_hz: float = DEFAULT_CONTINUOUS_ACTION_HZ):
        if not math.isfinite(continuous_action_hz) or continuous_action_hz <= 0:
            raise ValueError("continuous_action_hz must be finite and positive")
        self.continuous_action_hz = continuous_action_hz

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
        hz = self.continuous_action_hz
        primitives: list[list[Primitive]] = [[] for _ in windows]
        # (window, motor tick, kind, dx sum, dy sum)
        pending: tuple[int, int, str, float, float] | None = None

        def flush() -> None:
            nonlocal pending
            if pending is None:
                return
            win, _tick, kind, dx, dy = pending
            rdx, rdy = round(dx), round(dy)
            if rdx != 0 or rdy != 0:
                primitives[win].append(Primitive(kind, dx=rdx, dy=rdy))
            pending = None

        for le in _ordered_owned(labeled):
            e = le.event
            win = le.window
            if e.kind in ("move", "scroll"):
                # Deltas are never clamped, so label_t is the native event time
                # and lies within the owning window's span.
                offset_s = le.label_t - windows[win].start / master_fps
                tick = math.floor(offset_s * hz + 1e-9)
                if pending is not None and pending[:3] == (win, tick, e.kind):
                    pending = (win, tick, e.kind, pending[3] + e.dx, pending[4] + e.dy)
                else:
                    flush()
                    pending = (win, tick, e.kind, e.dx, e.dy)
            else:
                flush()
                primitives[win].append(
                    Primitive("down" if e.kind == "press" else "up", name=e.name)
                )
        flush()

        counts = Counter(p.kind for window in primitives for p in window)
        return FormatResult(
            labels=[_render_primitives(window) for window in primitives],
            counters=counters,
            primitive_counts={k: counts.get(k, 0) for k in ("move", "scroll", "down", "up")},
        )


FORMATTERS: dict[str, Callable[[float], ActionFormatter]] = {
    CanonicalFormatter.name: lambda hz: CanonicalFormatter(),
    OrderedFormatter.name: OrderedFormatter,
}


def get_formatter(
    name: str, *, continuous_action_hz: float = DEFAULT_CONTINUOUS_ACTION_HZ
) -> ActionFormatter:
    try:
        factory = FORMATTERS[name]
    except KeyError:
        raise KeyError(
            f"unknown action format {name!r} (available: {sorted(FORMATTERS)})"
        ) from None
    return factory(continuous_action_hz)
