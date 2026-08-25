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

``jitter_deadband_px`` (default 0 == off) additionally sums every maximal
move/scroll-only run into at most one move + one scroll, then drops whichever of
those is pure axis noise: a SINGLE-axis delta within the deadband, which is hand
tension while operating a button, wheel or keyboard rather than pointer control.
A diagonal is never jitter however small, because moving two axes at once takes
intent that noise does not produce. It runs after the typing collapse in v3, so
a move sandwiched between two typed runs is eligible and dropping it reunites
them; and it never empties a window that had content (see ``_suppress_jitter``),
because turning a turn's only primitive into a bare NO_OP would teach the model
to idle where the demonstrator acted.
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

from pipeline.crowdcast.lib.common import ActionBin
from pipeline.crowdcast.lib.events import (
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
    #: Per-window primitives, for a consumer that has to RE-RENDER over merged
    #: windows rather than read the labels back (stage 04's typing coalesce).
    #: Parsing a label back is what the annotation window planner did, and its
    #: typing-burst invariant went silently inoperative the moment a grammar
    #: spelled typing differently. None for the aggregate format.
    primitives: list[list[Primitive]] | None = None


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
        primitives=[list(window) for window in primitives],
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

#: The pipeline's one definition of a text key: every name that puts a character
#: on the screen. Read here to fold a typing run into ``type()``, and by the
#: annotation window planner to decide what counts as typing. One set, because two
#: had drifted — the planner's own frozen marker list called ``.``/``;``/``'``/
#: ``=``/``[``/``]`` non-typing while this map folded them into a burst, so the
#: same keystroke was a typing burst or a bare key depending on the grammar.
TEXT_KEYS: frozenset[str] = frozenset(_US_PRINTABLE)

_SHIFT_KEYS = frozenset({"ShiftLeft", "ShiftRight"})
# A held non-Shift modifier vetoes typing pairs (Ctrl+C is a chord, not typing).
_NON_SHIFT_MODIFIERS = frozenset({
    "ControlLeft", "ControlRight", "Control",
    "Alt", "AltLeft", "AltRight", "AltGr",
    "MetaLeft", "MetaRight", "Meta", "MetaGr",
    "Function",
})
_MODIFIER_KEYS = _SHIFT_KEYS | _NON_SHIFT_MODIFIERS


def render_primitives(primitives: Sequence[Primitive]) -> str:
    """One window's primitives -> its label, through the grammar's own codec.

    The single place stage 04's typing coalesce turns merged windows back into
    text. It goes through `CODEC.format` rather than joining `render()` calls so
    a re-derived label cannot differ from a first-pass one.
    """
    return ORDERED_EVENTS_V3.format(
        OrderedEventsV3Action(primitives=tuple(primitives), no_op=not primitives)
    )


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


def _is_axis_jitter(p: Primitive, threshold_px: int) -> bool:
    """True iff ``p`` is a move/scroll that is purely horizontal or purely
    vertical (exactly one axis nonzero) AND that axis's magnitude is within
    ``threshold_px``. A DIAGONAL move/scroll (both axes nonzero) is never
    jitter, however small each axis is — moving on two axes at once takes
    deliberate intent that hand-tension noise does not produce."""
    if p.kind not in ("move", "scroll"):
        return False
    if p.dx != 0 and p.dy != 0:
        return False
    mag = abs(p.dx) if p.dx != 0 else abs(p.dy)
    return mag <= threshold_px


def _merge_move_scroll_run(run: Sequence[Primitive]) -> list[Primitive]:
    """Sum a maximal move/scroll-only run into at most one move + one scroll
    primitive, ordered by whichever kind appeared FIRST in the run. A kind
    absent from the run, or whose sum is (0,0), is omitted — ``move(0,0)`` and
    ``scroll(0,0)`` are never emitted, same as everywhere else in this module."""
    sums = {"move": [0, 0], "scroll": [0, 0]}
    order: list[str] = []
    for p in run:
        if p.kind not in order:
            order.append(p.kind)
        sums[p.kind][0] += p.dx
        sums[p.kind][1] += p.dy
    out: list[Primitive] = []
    for kind in order:
        dx, dy = sums[kind]
        if dx != 0 or dy != 0:
            out.append(Primitive(kind, dx=dx, dy=dy))
    return out


def _merge_adjacent_typing(prims: Sequence[Primitive]) -> list[Primitive]:
    """Concatenate adjacent ``type()`` primitives. ``_collapse_typing`` never
    produces these itself (a typing run always abuts either a boundary or the
    non-typing primitive that broke it) — they only arise here, when dropping an
    axis-jitter move/scroll between two typing runs reunites them, e.g.
    ``type("h"); move(1,0); type("i")`` -> ``type("h"); type("i")`` -> this pass
    -> ``type("hi")``."""
    out: list[Primitive] = []
    for p in prims:
        if p.kind == "type" and out and out[-1].kind == "type":
            out[-1] = Primitive("type", text=out[-1].text + p.text)
        else:
            out.append(p)
    return out


def _suppress_jitter(prims: Sequence[Primitive], threshold_px: int) -> list[Primitive]:
    """Sum every maximal move/scroll-only run (a click or typed text always
    breaks one) into at most one move + one scroll primitive — ordered by
    whichever kind appeared first in the run — then drop any resulting primitive
    that is pure axis noise (``_is_axis_jitter``): incidental cursor/wheel jitter
    from hand tension while operating a button, wheel or keyboard, not
    intentional pointer control. Dropping one can reunite two typing runs it had
    separated (``_merge_adjacent_typing``).

    If dropping every eligible primitive would leave the window with NOTHING at
    all, none of them are dropped — jitter suppression must never turn a turn's
    only content into a bare NO_OP. A window with real content elsewhere (even
    one surviving primitive) has no such floor."""
    merged: list[Primitive] = []
    i, n = 0, len(prims)
    while i < n:
        if prims[i].kind in ("move", "scroll"):
            j = i
            while j < n and prims[j].kind in ("move", "scroll"):
                j += 1
            merged.extend(_merge_move_scroll_run(prims[i:j]))
            i = j
        else:
            merged.append(prims[i])
            i += 1

    filtered = [p for p in merged if not _is_axis_jitter(p, threshold_px)]
    result = filtered if filtered or not merged else merged
    return _merge_adjacent_typing(result)


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

    def __init__(
        self,
        continuous_action_hz: float = DEFAULT_CONTINUOUS_ACTION_HZ,
        *,
        jitter_deadband_px: int = 0,
    ):
        if not math.isfinite(continuous_action_hz) or continuous_action_hz <= 0:
            raise ValueError("continuous_action_hz must be finite and positive")
        if jitter_deadband_px < 0:
            raise ValueError("jitter_deadband_px must be >= 0")
        self.continuous_action_hz = continuous_action_hz
        self.jitter_deadband_px = jitter_deadband_px

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
        if self.jitter_deadband_px > 0:
            primitives = [
                _suppress_jitter(window, self.jitter_deadband_px)
                for window in primitives
            ]
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
        if self.jitter_deadband_px > 0:
            # After the typing collapse, so a move sandwiched next to a
            # `type("...")` — not just a bare down/up — is eligible too, and so
            # dropping it can reunite the two runs it separated.
            collapsed = [
                _suppress_jitter(window, self.jitter_deadband_px)
                for window in collapsed
            ]
        return _ordered_result(
            collapsed, counters, ("move", "scroll", "down", "up", "type")
        )


#: ``(continuous_action_hz, jitter_deadband_px) -> ActionFormatter``. Both knobs
#: reach every factory even where one is inert (`canonical` renders aggregate
#: deltas, which have no primitive run to sum), so adding a formatter never means
#: touching the call site.
FORMATTERS: dict[str, Callable[[float, int], ActionFormatter]] = {
    CanonicalFormatter.name: lambda hz, jitter: CanonicalFormatter(),
    OrderedFormatter.name: lambda hz, jitter: OrderedFormatter(
        hz, jitter_deadband_px=jitter
    ),
    OrderedTypingFormatter.name: lambda hz, jitter: OrderedTypingFormatter(
        hz, jitter_deadband_px=jitter
    ),
}


def get_formatter(
    name: str,
    *,
    continuous_action_hz: float = DEFAULT_CONTINUOUS_ACTION_HZ,
    jitter_deadband_px: int = 0,
) -> ActionFormatter:
    try:
        factory = FORMATTERS[name]
    except KeyError:
        raise KeyError(
            f"unknown action format {name!r} (available: {sorted(FORMATTERS)})"
        ) from None
    return factory(continuous_action_hz, jitter_deadband_px)


# Cross-window typing coalescing (stage 04 --coalesce-typing)
# ---------------------------------------------------------------------------

# The idle label of the ordered text formats (a window with no primitives).
ORDERED_IDLE_LABEL = "NO_OP"

# Formats whose windows carry a ``type()`` primitive, i.e. the only ones where
# "this window is typing and nothing else" is a well-defined question. The
# aggregate formats spell typing as bare key transitions, indistinguishable
# from a chord.
TYPING_COALESCE_FORMATS = frozenset({OrderedTypingFormatter.name})


@dataclass
class TypingCoalescePlan:
    """Which windows survive a cross-window typing coalesce, and what each
    surviving window swallows.

    ``keep`` are the retained window indices (ascending); everything else is
    dropped — its keystrokes land in the label of the run's FIRST window once
    the caller re-derives labels over merged windows.

    ``spans`` maps a run's first window index to the LAST window index it
    absorbs (inclusive); windows absent from ``spans`` are untouched singletons.

    ``forced_idle`` are kept windows whose label must be overwritten with
    ``ORDERED_IDLE_LABEL`` because a run absorbed their span: a run is never
    allowed to end a conversation, so when one reaches a terminal window that
    window is retained as a trailing do-nothing turn (its frame is a fresh
    screenshot, and ``--terminate-token`` can overwrite it without destroying
    typing supervision). Such a window must be EXCLUDED from the re-derived
    window list, so the run's window extends over it."""

    keep: list[int]
    spans: dict[int, int]
    forced_idle: list[int]

    @property
    def n_dropped(self) -> int:
        return sum(end - start for start, end in self.spans.items()) - len(self.forced_idle)


def is_typing_only_window(
    prims: Sequence[Primitive], held_non_shift_mods: Sequence[str] | set[str] = ()
) -> bool:
    """True iff every primitive in the window is typing: a ``type()`` run, a
    Shift transition, or a bare printable-key transition — and no non-Shift
    modifier is physically held.

    The bare printable transitions are the ones ``_collapse_typing`` could not
    fold because the run did not BALANCE inside this window: key rollover at a
    frame boundary (``…; down(KeyC)`` here, ``up(KeyC); …`` next window) and a
    Shift held across frames. Admitting them is what lets the caller re-derive
    a single ``type()`` over the merged window.

    Breakers (anything else): move, scroll, mouse buttons, Return/Tab/Backspace/
    arrows/F-keys (deliberately kept as their own turns — backtracking and
    submitting are actions worth supervising per frame), and every non-Shift
    modifier. The held-modifier veto is what keeps a Ctrl+C whose halves fall in
    different windows (``down(ControlLeft)`` | ``down(KeyC); up(KeyC)`` |
    ``up(ControlLeft)``) from reading as typing."""
    if not prims or (set(held_non_shift_mods) & _NON_SHIFT_MODIFIERS):
        return False
    return all(
        p.kind == "type"
        or (
            p.kind in ("down", "up")
            and (p.name in _US_PRINTABLE or p.name in _SHIFT_KEYS)
        )
        for p in prims
    )


def plan_typing_coalesce(
    primitives: Sequence[Sequence[Primitive]],
    *,
    barrier_start: Sequence[int] | set[int] = (),
    terminal: Sequence[int] | set[int] = (),
    break_before: Sequence[int] | set[int] = (),
    max_frames: int = 0,
) -> TypingCoalescePlan:
    """Group maximal runs of typing-only windows (see ``is_typing_only_window``)
    into single turns.

    A run opens on a typing-only window and extends while the next window is
    typing-only OR idle — an idle window INSIDE a typing stretch is a pause the
    demonstrator took mid-sentence, not an action, so it is absorbed. The run
    ends at its last TYPING window: trailing idle windows are never absorbed
    (they stay their own turns, so the conversation keeps a screenshot of the
    settled screen).

    Hard stops, all of them meaning "the next window cannot join this run":
      * ``max_frames`` — the number of ORIGINAL windows one turn may span (0 ==
        unlimited). This is the staleness bound: the turn shows the run's FIRST
        screenshot while its label carries everything typed until the run's end,
        so a long run must split into consecutive capped chunks, each keeping its
        own screenshot.
      * ``barrier_start`` — windows that must begin a fresh run (a goal window's
        first frame: a run crossing into it would move that goal's opening
        keystrokes into a turn the goal does not contain).
      * ``break_before`` — windows whose gap from the previous window contains a
        dead zone: the policy discards/clamps keystrokes there, so a merged label
        would be text with a silent hole in it.
      * ``terminal`` — a window that must never be absorbed as a run's interior
        (a goal window's last frame, and the last window of the segment). A run
        that REACHES one absorbs its span but keeps it as a trailing
        ``forced_idle`` turn; a run cannot start at one and extend past it (that
        would leak post-goal keystrokes into the goal's final turn)."""
    n = len(primitives)
    barriers, terminals, breaks = set(barrier_start), set(terminal), set(break_before)
    cap = max_frames if max_frames and max_frames > 0 else n

    # Per-window typing eligibility. The non-Shift modifier held-set is
    # cross-window state (exactly as ``_collapse_typing`` tracks it: a modifier
    # only ever enters it via an explicitly rendered down()).
    typing: list[bool] = []
    mods: set[str] = set()
    for prims in primitives:
        typing.append(is_typing_only_window(prims, mods))
        for p in prims:
            if p.kind == "down" and p.name in _MODIFIER_KEYS:
                mods.add(p.name)
            elif p.kind == "up" and p.name in _MODIFIER_KEYS:
                mods.discard(p.name)
    idle = [not bool(prims) for prims in primitives]

    keep: list[int] = []
    spans: dict[int, int] = {}
    forced_idle: list[int] = []
    i = 0
    while i < n:
        keep.append(i)
        if not typing[i] or i in terminals:
            i += 1
            continue
        last_typing = i
        stop_terminal: int | None = None
        j = i + 1
        while (
            j < n
            and (j - i) < cap  # a run of [i..j] spans j-i+1 windows
            and j not in barriers
            and j not in breaks
            and (typing[j] or idle[j])
        ):
            if j in terminals:
                # Absorb it only as the run's trailing idle turn (typing
                # windows), else leave it alone (an idle window is not worth
                # absorbing at the cost of ending the run here).
                if typing[j]:
                    stop_terminal = last_typing = j
                break
            if typing[j]:
                last_typing = j
            j += 1
        end = stop_terminal if stop_terminal is not None else last_typing
        if end > i:
            spans[i] = end
            if stop_terminal is not None:
                keep.append(stop_terminal)
                forced_idle.append(stop_terminal)
        i = end + 1
    return TypingCoalescePlan(keep=keep, spans=spans, forced_idle=forced_idle)
