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

Beside the labels every formatter publishes each window's ``WindowKeyboard``,
the grammar action projected to what the demonstrator typed. A consumer that
needs the activity rather than the text reads that instead of parsing a label
back — the annotation window planner did parse them back, and its typing-burst
invariant went silently inoperative the moment a grammar spelled typing
differently.

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

``OrderedTypingFormatter`` (``ordered_events_v3``) is that same extraction with
balanced typing runs collapsed into one ``type("...")`` primitive. Both emit the
``ordered_events_v3`` grammar, whose ``type()`` production only this one reaches.
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


@dataclass(frozen=True)
class WindowKeyboard:
    """One window's keyboard content, as the grammar action spells it.

    The projection of an action every consumer that asks *what the demonstrator
    typed* needs, and the only one that survives a change of grammar: names are
    the key/button transitions (press and release alike, unordered as far as any
    consumer cares), texts the coalesced typing bursts. A consumer reading this
    instead of the rendered label cannot be broken by a grammar that spells the
    same activity differently — ``ordered_events_v3`` collapses a typing run into
    ``type("...")``, where ``deltatype_v2`` spells every keystroke.
    """

    names: tuple[str, ...]
    texts: tuple[str, ...]


@dataclass
class FormatResult:
    labels: list[str]  # one per window, same order
    keyboard: list[WindowKeyboard]  # one per window, same order
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


def _bin_action(action_bin: ActionBin) -> DeltatypeV2Action:
    """One aggregate bin -> its ``deltatype_v2`` action.

    The subset of that grammar this formatter can reach: no ``type()``, no
    ``MOVE()`` drag, and NO_OP as the only control token — a keylog records no
    intent to terminate.
    """
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


def _element_keyboard(action: DeltatypeV2Action) -> WindowKeyboard:
    return WindowKeyboard(
        names=tuple(e.name for e in action.elements if e.kind == "event"),
        texts=tuple(e.text for e in action.elements if e.kind == "type"),
    )


def _primitive_keyboard(action: OrderedEventsV3Action) -> WindowKeyboard:
    return WindowKeyboard(
        names=tuple(p.name for p in action.primitives if p.kind in ("down", "up")),
        texts=tuple(p.text for p in action.primitives if p.kind == "type"),
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
        actions = [_bin_action(b) for b in bins]
        return FormatResult(
            labels=[DELTATYPE_V2.format(a) for a in actions],
            keyboard=[_element_keyboard(a) for a in actions],
            counters=counters,
        )


def _ordered_result(
    primitives: Sequence[Sequence[Primitive]],
    counters: PolicyCounters,
    kinds: tuple[str, ...],
) -> FormatResult:
    """Per-window primitives -> their ``ordered_events_v3`` labels + counts.

    An empty window is NO_OP; the grammar's TERMINATE is unreachable from a
    keylog, which records no intent to terminate.
    """
    counts = Counter(p.kind for window in primitives for p in window)
    actions = [
        OrderedEventsV3Action(primitives=tuple(window), no_op=not window)
        for window in primitives
    ]
    return FormatResult(
        labels=[ORDERED_EVENTS_V3.format(a) for a in actions],
        keyboard=[_primitive_keyboard(a) for a in actions],
        counters=counters,
        primitive_counts={k: counts.get(k, 0) for k in kinds},
    )


# US-layout printable map: key name -> (base char, shifted char). The names are
# the rdev namespace ``common.resolve_key_name`` passes through from the keylog
# (top-row digits are Num0..Num9, and the brackets are Left/RightBracket, not
# Bracket Left/Right — spelled the other way round this map never fired and
# trained a key press where a character belongs); the char pairs mirror the
# viewer's keyToChar map (tooling/visualize_frame_records.py), the project's
# existing convention. Keypad digits, Return, Tab, Backspace, Escape, arrows,
# F-keys and modifiers are deliberately absent: they render as down()/up() and
# break typing runs.
_US_PRINTABLE: dict[str, tuple[str, str]] = {
    "Space": (" ", " "),
    **{f"Key{c}": (c.lower(), c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
    **{f"Num{d}": (d, s) for d, s in zip("1234567890", "!@#$%^&*()", strict=True)},
    "BackQuote": ("`", "~"),
    "Minus": ("-", "_"),
    "Equal": ("=", "+"),
    "LeftBracket": ("[", "{"),
    "RightBracket": ("]", "}"),
    "BackSlash": ("\\", "|"),
    "SemiColon": (";", ":"),
    "Quote": ("'", '"'),
    "Comma": (",", "<"),
    "Dot": (".", ">"),
    "Slash": ("/", "?"),
}

_SHIFT_KEYS = frozenset({"ShiftLeft", "ShiftRight"})
# A held non-Shift modifier vetoes typing pairs (Ctrl+C is a chord, not typing).
_NON_SHIFT_MODIFIERS = frozenset({
    "ControlLeft", "ControlRight", "Control",
    "Alt", "AltLeft", "AltRight", "AltGr",
    "MetaLeft", "MetaRight", "Meta", "MetaGr",
    "Function",
})
_MODIFIER_KEYS = _SHIFT_KEYS | _NON_SHIFT_MODIFIERS


def _collapse_typing(
    prims: Sequence[Primitive], held_mods: set[str]
) -> list[Primitive]:
    """One window's primitives with typing runs collapsed to ``type("...")``.

    A *typing run* is a maximal contiguous span of ``down``/``up`` primitives of
    printable and Shift keys ONLY, that is BALANCED (every key pressed in the
    span is released in it and vice versa), entered while no non-Shift modifier
    is held. Characters are emitted in ``down`` order — so key *rollover*
    (``down h; down e; up h; up e`` from natural fast typing, where releases
    interleave or reorder) still collapses to ``type("he")`` instead of a string
    of per-key events.

    A span that never balances before a breaker renders explicitly instead: a
    printable key held across a window boundary (its ``up`` is in another window)
    -> ``down``/``up``; a Shift enclosing a non-typing primitive (move, Tab, …)
    -> the Shift renders while inner printable keys still ``type`` under it; a
    bare Shift tap (zero characters) -> ``down``/``up``. Any non-key primitive
    (move, scroll, mouse button, Return/Backspace/arrow, a non-Shift modifier)
    breaks the run. ``held_mods`` is updated in place for the next window.
    """
    n = len(prims)
    out: list[Primitive] = []
    mods = set(held_mods)

    def _is_typing_key(p: Primitive) -> bool:
        return p.kind in ("down", "up") and (
            p.name in _US_PRINTABLE or p.name in _SHIFT_KEYS
        )

    i = 0
    while i < n:
        p = prims[i]
        # A typing run can only OPEN on a printable/Shift key down, and only
        # when no non-Shift modifier is physically held.
        if p.kind == "down" and _is_typing_key(p) and not (mods & _NON_SHIFT_MODIFIERS):
            # Scan the longest BALANCED prefix of the contiguous key-only span
            # starting at i (balanced == every down matched by a later up in
            # the span). ``run_end`` is exclusive; None if it never balances.
            open_counts: dict[str, int] = {}
            run_end: int | None = None
            j = i
            while j < n and _is_typing_key(prims[j]):
                q = prims[j]
                if q.kind == "down":
                    open_counts[q.name] = open_counts.get(q.name, 0) + 1
                else:
                    c = open_counts.get(q.name, 0)
                    if c == 0:
                        break  # up with no matching down in-span -> held from before
                    open_counts[q.name] = c - 1
                    if open_counts[q.name] == 0:
                        del open_counts[q.name]
                j += 1
                if not open_counts:
                    run_end = j  # balanced here — candidate maximal run end
            if run_end is not None:
                # Emit the run [i, run_end): chars in down order, Shift folded
                # and absorbed. Non-Shift modifiers cannot occur inside (the
                # span is printable/Shift only and we entered with none held).
                shift_held = bool(mods & _SHIFT_KEYS)
                chars: list[str] = []
                for k in range(i, run_end):
                    q = prims[k]
                    if q.name in _SHIFT_KEYS:
                        shift_held = q.kind == "down"
                        continue
                    if q.kind == "down":
                        base, shifted = _US_PRINTABLE[q.name]
                        chars.append(shifted if shift_held else base)
                if chars:
                    out.append(Primitive("type", text="".join(chars)))
                    i = run_end
                    continue
                # Zero characters (e.g. a bare Shift tap) — fall through and
                # render the run's primitives explicitly rather than type("").
        if p.kind == "down" and p.name in _MODIFIER_KEYS:
            mods.add(p.name)
        elif p.kind == "up" and p.name in _MODIFIER_KEYS:
            mods.discard(p.name)
        out.append(p)
        i += 1

    held_mods.clear()
    held_mods.update(mods)
    return out


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
        primitives, counters = self._window_primitives(
            events, windows, dead_zones, master_fps=master_fps
        )
        return _ordered_result(primitives, counters, ("move", "scroll", "down", "up"))

    def _window_primitives(
        self,
        events: Sequence[RawEvent],
        windows: Sequence[Window],
        dead_zones: Sequence[DeadZone],
        *,
        master_fps: float,
    ) -> tuple[list[list[Primitive]], PolicyCounters]:
        """One ordered primitive list per window (the shared v2/v3 core)."""
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
        return primitives, counters


class OrderedTypingFormatter(OrderedFormatter):
    """``ordered_events_v2`` plus a ``type("...")`` primitive: maximal runs of
    plain typing collapse into one quoted string (the typing action the base
    model natively knows; per-key down/up typing costs ~8 tokens per character
    and truncated downstream). ShiftLeft/ShiftRight are absorbed into the run
    when they enclose only typing pairs — exactly one Shift held, no other
    modifier, giving the shifted US-layout character; a Shift whose scope
    includes anything else renders as normal down/up and breaks the run, as does
    every other rendered primitive. Everything else — motor grid, ordering,
    NO_OP, held-state accounting — is byte-identical to v2."""

    name = "ordered_events_v3"

    def format_segment(
        self,
        events: Sequence[RawEvent],
        windows: Sequence[Window],
        dead_zones: Sequence[DeadZone],
        *,
        master_fps: float,
    ) -> FormatResult:
        primitives, counters = self._window_primitives(
            events, windows, dead_zones, master_fps=master_fps
        )
        # The physical modifier held-set crosses windows: a balanced typing run
        # nets it to zero, so only explicitly-rendered modifier events move it.
        held_mods: set[str] = set()
        collapsed = [_collapse_typing(window, held_mods) for window in primitives]
        return _ordered_result(
            collapsed, counters, ("move", "scroll", "down", "up", "type")
        )


FORMATTERS: dict[str, Callable[[float], ActionFormatter]] = {
    CanonicalFormatter.name: lambda hz: CanonicalFormatter(),
    OrderedFormatter.name: OrderedFormatter,
    OrderedTypingFormatter.name: OrderedTypingFormatter,
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
