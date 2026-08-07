"""Action formatters: (events, windows, dead_zones) -> one label per window.

This is the single place the on-disk keylog becomes assistant-turn text. A
formatter receives the full raw event stream (state layer) plus the dead-zone
label policy's dispositions (label layer) via ``events.apply_label_policy`` — a
stateful format (e.g. cumulative cursor position) folds over every event while
emitting labels only for owned ones.

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
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from pipeline.lib.common import ActionBin, format_action
from pipeline.lib.events import (
    DeadZone,
    LabeledEvent,
    PolicyCounters,
    RawEvent,
    Window,
    apply_label_policy,
)

DEFAULT_CONTINUOUS_ACTION_HZ = 10.0

# Rendered names must be unambiguous inside the mini-program syntax.
_INPUT_NAME_RE = re.compile(r"^[^\s(),;]+$")


@dataclass
class FormatResult:
    labels: list[str]  # one per window, same order
    counters: PolicyCounters
    # Provenance for primitive-based formats (None for the aggregate format).
    primitive_counts: dict[str, int] | None = None


class ActionFormatter(Protocol):
    name: str
    # ``{what}`` is "the next action" / "the next action toward that goal";
    # stage 04 composes the default system prompts from this.
    reply_contract: str

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
    reply_contract = (
        "Reply with {what} as `<dx> <dy> <scroll>` optionally followed by "
        "` ; +KEY -KEY` events, or `NO_OP` if no action."
    )

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


@dataclass(frozen=True)
class ActionPrimitive:
    kind: str  # "move" | "scroll" | "down" | "up"
    dx: int | None = None
    dy: int | None = None
    input_name: str | None = None

    def render(self) -> str:
        if self.kind in ("move", "scroll"):
            return f"{self.kind}({self.dx},{self.dy})"
        return f"{self.kind}({self.input_name})"


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
    reply_contract = (
        "Reply with {what} as `; `-separated primitives in the order performed "
        "— `move(<dx>,<dy>)`, `scroll(<dx>,<dy>)`, `down(<KEY>)`, `up(<KEY>)` "
        "— or `NO_OP` if no action."
    )

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
        primitives: list[list[ActionPrimitive]] = [[] for _ in windows]
        # (window, motor tick, kind, dx sum, dy sum)
        pending: tuple[int, int, str, float, float] | None = None

        def flush() -> None:
            nonlocal pending
            if pending is None:
                return
            win, _tick, kind, dx, dy = pending
            rdx, rdy = round(dx), round(dy)
            if rdx != 0 or rdy != 0:
                primitives[win].append(ActionPrimitive(kind=kind, dx=rdx, dy=rdy))
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
                if not isinstance(e.name, str) or not _INPUT_NAME_RE.fullmatch(e.name):
                    raise ValueError(f"invalid input name for ordered action: {e.name!r}")
                primitives[win].append(
                    ActionPrimitive(kind="down" if e.kind == "press" else "up",
                                    input_name=e.name)
                )
        flush()

        counts = Counter(p.kind for window in primitives for p in window)
        return FormatResult(
            labels=[
                "; ".join(p.render() for p in window) if window else "NO_OP"
                for window in primitives
            ],
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
