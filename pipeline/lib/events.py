"""Keylog event stream + the dead-zone LABEL policy (shared by all formatters).

Layering (see stage_03_filter / lib/views):
  * The master axis is integer ticks (1 tick = 1/master_fps s; tick i == master
    record i). Keylog events keep continuous realigned time and are bucketed to
    a tick ONLY at consumption: ``tick = floor(t_s * master_fps)``.
  * A *window* is the label-ownership span of one selected frame:
    ``[tick_i, tick_of_next_selected_frame)`` — what the demonstrator did after
    seeing frame i. Windows never overlap and tile the axis from the first
    selected frame to the end of master coverage.
  * A *dead zone* is a span whose pixels the trainee never sees: black frames,
    missing master coverage, and the span before the first selected frame.
    Idle-dropped spans are not dead zones (they are empty of events by
    definition; windows pass over them).

State layer vs label layer: ``apply_label_policy`` never deletes an event — it
returns every parsed event annotated with a disposition (owning window, clamped
label time, or a discard reason), so stateful formatters (e.g. cumulative
position) can fold over the full stream while emitting labels only for owned
events.

Dead-zone label policy:
  * Mouse move / scroll deltas inside a dead zone are discarded from labels
    (counted per zone reason).
  * When the keylog runs past the video's last frame, those events fall in the
    trailing ``no_coverage`` zone and are discarded — never folded into the last
    selected frame's action, since nothing was visible there.
  * A key/button press+release pair straddling a dead zone is completed by
    clamping the unseen endpoint to the zone boundary: release in a zone is
    emitted at the zone start (tail of the last visible window — after its
    native events); press in a zone is emitted at the zone end (head of the
    first visible window — before its native events). Clamping to boundaries
    preserves event order automatically and resolves staggered combos
    (+ALT +TAB) per key, so alt-tab-style transition supervision survives the
    black flash the transition itself causes.
  * A pair fully inside dead zones (no visible window between its endpoints)
    is discarded. A press in a dead zone that is never released is discarded
    too: clamping it forward would emit a press with no matching release.
  * Every removal/clamp is counted (``PolicyCounters``); zero dangling keys from
    dead zones by construction. Dropping a release instead of clamping it would
    leave the key held for the rest of the conversation, making every later
    label wrong undetectably. The counters double as a per-segment realignment
    health metric.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pipeline.lib.common import (
    load_keylog_entries,
    resolve_button_name,
    resolve_key_name,
)

# Dead-zone reasons. ``pre_first_frame`` and trailing ``no_coverage`` are also
# derived implicitly when an event falls outside every window and every
# explicit zone.
ZONE_BLACK = "black"
ZONE_NO_COVERAGE = "no_coverage"
ZONE_PRE_FIRST_FRAME = "pre_first_frame"


@dataclass(frozen=True)
class RawEvent:
    """One parsed keylog event on the realigned clock.

    ``kind`` is one of ``move`` / ``scroll`` / ``press`` / ``release``;
    press/release carry the resolved key/button ``name`` (Key* and Mouse*
    events share one namespace).
    Scroll events carry both raw axes in ``dx``/``dy`` plus the legacy
    collapsed scalar in ``scroll`` (y, falling back to x — what the canonical
    format consumes)."""

    seq: int
    t_s: float
    kind: str
    dx: float = 0.0
    dy: float = 0.0
    scroll: float = 0.0
    name: str | None = None


@dataclass
class EventStats:
    """Parse-layer counts over the whole keylog (before any window logic).

    The ``n_dropped_*`` counters close the parse layer: an entry counted in
    ``n_events`` either becomes a RawEvent, is a deliberate ignore
    (``ContextChanged`` / an event type this pipeline does not model), or is
    dropped as unparseable — and the last kind loses real demonstrator input,
    so it is counted rather than skipped invisibly. ``Unknown(-1)`` key names
    from the macOS recorder are the known live instance."""

    n_events: int = 0
    n_mousemove: int = 0
    n_scroll: int = 0
    n_keypress: int = 0
    n_keyrelease: int = 0
    n_mousepress: int = 0
    n_mouserelease: int = 0
    n_dropped_unresolved_name: int = 0
    n_dropped_bad_payload: int = 0
    n_dropped_bad_timestamp: int = 0
    n_ignored_other_type: int = 0


@dataclass(frozen=True)
class Window:
    """Label-ownership span of one selected frame: ``[start, end)`` master ticks."""

    master_idx: int
    start: int
    end: int


@dataclass(frozen=True)
class DeadZone:
    """Half-open ``[start, end)`` master-tick span with no usable pixels."""

    start: int
    end: int
    reason: str


@dataclass
class LabeledEvent:
    """One RawEvent + its label disposition. ``window is None`` == not emitted."""

    event: RawEvent
    label_t: float
    window: int | None
    discard_reason: str | None = None
    clamped: str | None = None  # "press_to_zone_end" | "release_to_zone_start"


@dataclass
class PolicyCounters:
    """Per-segment dead-zone accounting (a realignment health metric)."""

    n_discarded_black: int = 0
    n_discarded_no_coverage: int = 0
    n_discarded_pre_first_frame: int = 0
    n_pairs_dropped_dead_zone: int = 0
    n_unreleased_press_dropped: int = 0
    n_releases_clamped: int = 0
    n_presses_clamped: int = 0
    n_dangling_release: int = 0
    n_redundant_press: int = 0
    n_held_at_end: int = 0
    max_simultaneous_keys: int = 0

    def count_discarded_delta(self, reason: str) -> None:
        if reason == ZONE_BLACK:
            self.n_discarded_black += 1
        elif reason == ZONE_PRE_FIRST_FRAME:
            self.n_discarded_pre_first_frame += 1
        else:
            self.n_discarded_no_coverage += 1


_PER_TYPE_COUNTER = {
    "MouseMove": "n_mousemove",
    "MouseScroll": "n_scroll",
    "KeyPress": "n_keypress",
    "KeyRelease": "n_keyrelease",
    "MousePress": "n_mousepress",
    "MouseRelease": "n_mouserelease",
}


def load_events(keylog_path: Path) -> tuple[list[RawEvent], EventStats]:
    """Parse a realigned msgpack keylog into an ordered RawEvent stream + stats.

    One pass, so the counts describe exactly the stream returned. Every entry
    counted in ``n_events`` is accounted for: emitted, deliberately ignored, or
    counted into an ``n_dropped_*`` bucket. The realigned pipeline consumes
    corrected keylogs, so the timestamps are already master-clock."""
    stats = EventStats()
    events: list[RawEvent] = []
    for entry in load_keylog_entries(keylog_path):
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        timestamp, event = entry[0], entry[1]
        if not isinstance(event, list) or not event:
            continue
        stats.n_events += 1
        event_type = str(event[0])
        counter = _PER_TYPE_COUNTER.get(event_type)
        if counter is not None:
            setattr(stats, counter, getattr(stats, counter) + 1)
        try:
            timestamp_us = int(timestamp)
        except (TypeError, ValueError):
            stats.n_dropped_bad_timestamp += 1
            continue
        if event_type == "ContextChanged":
            continue
        payload = event[1] if len(event) > 1 else None
        t_s = timestamp_us / 1_000_000
        seq = len(events)

        if event_type == "MouseMove":
            if not (isinstance(payload, list) and len(payload) >= 2):
                stats.n_dropped_bad_payload += 1
                continue
            events.append(
                RawEvent(seq, t_s, "move", dx=float(payload[0]), dy=float(payload[1]))
            )
        elif event_type == "MouseScroll":
            if not (isinstance(payload, list) and len(payload) >= 2):
                stats.n_dropped_bad_payload += 1
                continue
            value = payload[1] if payload[1] != 0 else payload[0]
            events.append(
                RawEvent(
                    seq,
                    t_s,
                    "scroll",
                    dx=float(payload[0]),
                    dy=float(payload[1]),
                    scroll=float(value),
                )
            )
        elif event_type in ("KeyPress", "MousePress", "KeyRelease", "MouseRelease"):
            name = (
                resolve_key_name(payload)
                if event_type.startswith("Key")
                else resolve_button_name(payload)
            )
            if name is None:
                stats.n_dropped_unresolved_name += 1
                continue
            kind = "press" if event_type.endswith("Press") else "release"
            events.append(RawEvent(seq, t_s, kind, name=name))
        else:
            stats.n_ignored_other_type += 1
    return events, stats


class _Locator:
    """Classify a master tick as (window index, None) or (None, DeadZone).

    Dead zones win over window spans (a zone interior to a window's span takes
    its events away from the window). Ticks outside every window and every
    explicit zone resolve to implicit ``pre_first_frame`` / ``no_coverage``
    zones, so the partition is total: every event is owned by exactly one
    window, clamped, or discarded-with-counter."""

    def __init__(self, windows: Sequence[Window], dead_zones: Sequence[DeadZone]):
        self.windows = list(windows)
        self.zones = sorted(dead_zones, key=lambda z: z.start)
        self._win_starts = [w.start for w in self.windows]
        self._zone_starts = [z.start for z in self.zones]
        for a, b in zip(self.windows, self.windows[1:], strict=False):
            if b.start < a.end:
                raise ValueError(f"overlapping windows: {a} / {b}")
        self._first_start = self.windows[0].start if self.windows else 0
        self._axis_end = self.windows[-1].end if self.windows else 0

    def zone_at(self, tick: int) -> DeadZone | None:
        i = bisect_right(self._zone_starts, tick) - 1
        if i >= 0 and self.zones[i].start <= tick < self.zones[i].end:
            return self.zones[i]
        return None

    def locate(self, tick: int) -> tuple[int | None, DeadZone | None]:
        zone = self.zone_at(tick)
        if zone is not None:
            return None, zone
        if tick < self._first_start:
            return None, DeadZone(min(0, tick), self._first_start, ZONE_PRE_FIRST_FRAME)
        if tick >= self._axis_end:
            return None, DeadZone(self._axis_end, tick + 1, ZONE_NO_COVERAGE)
        i = bisect_right(self._win_starts, tick) - 1
        if i >= 0 and self.windows[i].start <= tick < self.windows[i].end:
            return i, None
        # Windows tile contiguously, so a gap here means malformed input.
        raise ValueError(f"tick {tick} in no window and no dead zone")

    def window_of(self, visible_tick: int) -> int:
        """Window index owning a tick known to be visible."""
        i = bisect_right(self._win_starts, visible_tick) - 1
        if i < 0 or not (self.windows[i].start <= visible_tick < self.windows[i].end):
            raise ValueError(f"tick {visible_tick} is not inside any window")
        return i

    def last_visible_before(self, tick: int) -> int | None:
        """Greatest visible tick < ``tick`` (walking left over adjacent zones)."""
        t = min(tick, self._axis_end) - 1
        while t >= self._first_start:
            zone = self.zone_at(t)
            if zone is None:
                return t
            t = zone.start - 1
        return None

    def first_visible_at_or_after(self, tick: int) -> int | None:
        """Smallest visible tick >= ``tick`` (walking right over adjacent zones)."""
        t = max(tick, self._first_start)
        while t < self._axis_end:
            zone = self.zone_at(t)
            if zone is None:
                return t
            t = zone.end
        return None


def _tick(t_s: float, master_fps: float) -> int:
    return int(t_s * master_fps) if t_s >= 0 else -1


def apply_label_policy(
    events: Sequence[RawEvent],
    windows: Sequence[Window],
    dead_zones: Sequence[DeadZone],
    *,
    master_fps: float,
) -> tuple[list[LabeledEvent], PolicyCounters]:
    """Assign every event a label disposition per the dead-zone policy.

    Returns one LabeledEvent per input event (same order) + counters. Pair
    matching runs a segment-global held-set over the full stream first (a press
    of an already-held name is redundant, a release of an un-held name is
    dangling), then each
    canonical pair is completed/clamped/dropped by where its endpoints fall."""
    counters = PolicyCounters()
    if not windows:
        raise ValueError("apply_label_policy needs at least one window")
    loc = _Locator(windows, dead_zones)

    labeled = [LabeledEvent(event=e, label_t=e.t_s, window=None) for e in events]

    # Pass 1: held-set pair matching over the full stream.
    held: dict[str, int] = {}  # name -> labeled index of the opening press
    pairs: list[tuple[int, int | None]] = []  # (press idx, release idx | None)
    for i, le in enumerate(labeled):
        e = le.event
        if e.kind == "press":
            if e.name in held:
                le.discard_reason = "redundant_press"
                counters.n_redundant_press += 1
            else:
                held[e.name] = i
                counters.max_simultaneous_keys = max(
                    counters.max_simultaneous_keys, len(held)
                )
        elif e.kind == "release":
            if e.name in held:
                pairs.append((held.pop(e.name), i))
            else:
                le.discard_reason = "dangling_release"
                counters.n_dangling_release += 1
    for press_idx in held.values():
        pairs.append((press_idx, None))
    counters.n_held_at_end = len(held)

    # Pass 2: move/scroll deltas — owned or discarded.
    for le in labeled:
        if le.event.kind not in ("move", "scroll"):
            continue
        win, zone = loc.locate(_tick(le.event.t_s, master_fps))
        if win is not None:
            le.window = win
        else:
            le.discard_reason = zone.reason
            counters.count_discarded_delta(zone.reason)

    # Pass 3: pairs — emit, clamp, or drop.
    def _drop_pair(press: LabeledEvent, release: LabeledEvent | None) -> None:
        press.window = None
        press.discard_reason = "pair_in_dead_zone"
        if release is not None:
            release.window = None
            release.discard_reason = "pair_in_dead_zone"
        counters.n_pairs_dropped_dead_zone += 1

    for press_idx, release_idx in pairs:
        press = labeled[press_idx]
        release = labeled[release_idx] if release_idx is not None else None

        p_win, p_zone = loc.locate(_tick(press.event.t_s, master_fps))
        if p_win is not None:
            press.window = p_win
        else:
            # Press unseen: clamp forward to the first visible tick after the
            # dead region (label_t at that tick's start; seq breaks the tie so
            # the clamped press sorts before the tick's native events).
            vt = loc.first_visible_at_or_after(p_zone.end)
            r_tick = (
                _tick(release.event.t_s, master_fps) if release is not None else None
            )
            if release is None:
                # Never released: a clamped press would emit a lone +KEY with
                # no matching release, so discard it.
                press.discard_reason = "unreleased_press_in_dead_zone"
                counters.n_unreleased_press_dropped += 1
                continue
            if vt is None or r_tick < vt:
                # No visible frame while the key was down: nothing was seen
                # being done with it — discard the whole pair.
                _drop_pair(press, release)
                continue
            press.window = loc.window_of(vt)
            press.label_t = vt / master_fps
            press.clamped = "press_to_zone_end"
            counters.n_presses_clamped += 1

        if release is None:
            press.window = None
            press.discard_reason = "unreleased_press"
            counters.n_unreleased_press_dropped += 1
            continue
        r_win, r_zone = loc.locate(_tick(release.event.t_s, master_fps))
        if r_win is not None:
            release.window = r_win
        else:
            # Release unseen: clamp back to the start of the dead region — one
            # past the last visible tick, so it sorts after that window's
            # native events. (A visible tick before the zone exists whenever
            # the press was emitted, so the drop below is defensive only.)
            vt = loc.last_visible_before(r_zone.start)
            if vt is None:
                _drop_pair(press, release)
                continue
            release.window = loc.window_of(vt)
            release.label_t = (vt + 1) / master_fps
            release.clamped = "release_to_zone_start"
            counters.n_releases_clamped += 1

    return labeled, counters
