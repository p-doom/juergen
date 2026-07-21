"""Dead-zone label policy: straddle clamping (both directions), staggered
combos, fully-inside pair discard, delta discards, counters, conservation, and
the state-vs-label layering (the full stream is always returned, annotated).

master_fps=1.0 throughout so seconds == ticks and the geometry is readable.
"""

from __future__ import annotations

import unittest

from realigned_pipeline.lib.action_format import get_formatter
from realigned_pipeline.lib.events import (
    DeadZone,
    RawEvent,
    Window,
    apply_label_policy,
)

M = 1.0

# Layout A: 20-tick axis, frames selected at ticks 0/5/15, black flash [10,15).
# Window 1 spans [5,15) — the zone is interior to its span and zone-wins.
WINDOWS_A = [Window(0, 0, 5), Window(5, 5, 15), Window(15, 15, 20)]
ZONES_A = [DeadZone(10, 15, "black")]


def _events(*specs: tuple) -> list[RawEvent]:
    """Build RawEvents from (t, kind, ...) specs; seq follows list order."""
    out: list[RawEvent] = []
    for seq, spec in enumerate(specs):
        t, kind = spec[0], spec[1]
        if kind == "move":
            out.append(RawEvent(seq, t, "move", dx=spec[2], dy=spec[3]))
        elif kind == "scroll":
            out.append(RawEvent(seq, t, "scroll", scroll=spec[2]))
        else:
            out.append(RawEvent(seq, t, kind, name=spec[2]))
    return out


def _format(events, windows=WINDOWS_A, zones=ZONES_A):
    result = get_formatter("canonical").format_segment(
        events, windows, zones, master_fps=M
    )
    return result.labels, result.counters


def _emitted_key_events(labels: list[str]) -> list[str]:
    """All ±NAME tokens across window labels, in emission order."""
    tokens: list[str] = []
    for label in labels:
        if ";" in label:
            tokens.extend(label.split(";", 1)[1].split())
    return tokens


def _assert_balanced(test: unittest.TestCase, labels: list[str]) -> None:
    """Every emitted +NAME has a matching later -NAME: no dangling keys."""
    held: set[str] = set()
    for tok in _emitted_key_events(labels):
        sign, name = tok[0], tok[1:]
        if sign == "+":
            test.assertNotIn(name, held, f"double press of {name} in {labels}")
            held.add(name)
        else:
            test.assertIn(name, held, f"release of un-held {name} in {labels}")
            held.remove(name)
    test.assertFalse(held, f"dangling keys {held} in {labels}")


class StraddleClampTest(unittest.TestCase):
    def test_release_in_zone_clamps_to_zone_start(self) -> None:
        labels, counters = _format(
            _events((6.0, "press", "KeyA"), (12.0, "release", "KeyA"))
        )
        self.assertEqual(labels, ["NO_OP", "0 0 0 ; +KeyA -KeyA", "NO_OP"])
        self.assertEqual(counters.n_releases_clamped, 1)
        self.assertEqual(counters.n_presses_clamped, 0)
        _assert_balanced(self, labels)

    def test_press_in_zone_clamps_before_native_events(self) -> None:
        # B pressed during the flash; the next window's native events (C's
        # press) must come AFTER the clamped +B.
        labels, counters = _format(
            _events(
                (11.0, "press", "KeyB"),
                (15.5, "press", "KeyC"),
                (16.0, "release", "KeyB"),
                (17.0, "release", "KeyC"),
            )
        )
        self.assertEqual(labels, ["NO_OP", "NO_OP", "0 0 0 ; +KeyB +KeyC -KeyB -KeyC"])
        self.assertEqual(counters.n_presses_clamped, 1)
        _assert_balanced(self, labels)

    def test_staggered_combo_keeps_per_key_order(self) -> None:
        # Alt+Tab: both releases fall in the transition flash; clamping both
        # to the zone start must preserve their relative order (seq tiebreak).
        labels, counters = _format(
            _events(
                (8.0, "press", "AltLeft"),
                (9.0, "press", "Tab"),
                (11.0, "release", "Tab"),
                (12.0, "release", "AltLeft"),
            )
        )
        self.assertEqual(
            labels, ["NO_OP", "0 0 0 ; +AltLeft +Tab -Tab -AltLeft", "NO_OP"]
        )
        self.assertEqual(counters.n_releases_clamped, 2)
        _assert_balanced(self, labels)

    def test_pair_fully_inside_zone_is_dropped(self) -> None:
        labels, counters = _format(
            _events((10.5, "press", "KeyD"), (11.5, "release", "KeyD"))
        )
        self.assertEqual(labels, ["NO_OP", "NO_OP", "NO_OP"])
        self.assertEqual(counters.n_pairs_dropped_dead_zone, 1)
        self.assertEqual(counters.n_presses_clamped, 0)
        self.assertEqual(counters.n_releases_clamped, 0)

    def test_unreleased_press_in_zone_is_dropped(self) -> None:
        # Pressed during the flash, never released anywhere: clamping forward
        # would emit a lone +KEY (unbalanced either way), so it is discarded.
        labels, counters = _format(_events((11.0, "press", "KeyI")))
        self.assertEqual(labels, ["NO_OP", "NO_OP", "NO_OP"])
        self.assertEqual(counters.n_unreleased_press_dropped, 1)
        self.assertEqual(counters.n_presses_clamped, 0)
        self.assertEqual(counters.n_held_at_end, 1)
        _assert_balanced(self, labels)

    def test_unreleased_visible_press_still_emitted(self) -> None:
        # A press on a VISIBLE frame with no release is real supervision at a
        # real time: it stays emitted (raw-keylog stickiness, counted only).
        labels, counters = _format(_events((6.0, "press", "KeyJ")))
        self.assertEqual(labels, ["NO_OP", "0 0 0 ; +KeyJ", "NO_OP"])
        self.assertEqual(counters.n_unreleased_press_dropped, 0)
        self.assertEqual(counters.n_held_at_end, 1)

    def test_pair_spanning_adjacent_zones_is_dropped(self) -> None:
        # Two abutting zones with no visible tick between press and release.
        zones = [DeadZone(10, 12, "black"), DeadZone(12, 15, "no_coverage")]
        labels, counters = _format(
            _events((10.5, "press", "KeyH"), (13.0, "release", "KeyH")),
            zones=zones,
        )
        self.assertEqual(labels, ["NO_OP", "NO_OP", "NO_OP"])
        self.assertEqual(counters.n_pairs_dropped_dead_zone, 1)

    def test_zone_interior_to_window_clamps_within_it(self) -> None:
        # Flash [8,10) sits inside window [5,15): the clamped press stays in
        # that same window, ordered at the flash end, before the release.
        windows = [Window(0, 0, 5), Window(5, 5, 15)]
        zones = [DeadZone(8, 10, "black")]
        labels, counters = _format(
            _events((8.5, "press", "KeyG"), (12.0, "release", "KeyG")),
            windows=windows,
            zones=zones,
        )
        self.assertEqual(labels, ["NO_OP", "0 0 0 ; +KeyG -KeyG"])
        self.assertEqual(counters.n_presses_clamped, 1)
        _assert_balanced(self, labels)


class DeltaDiscardTest(unittest.TestCase):
    def test_deltas_in_zone_discarded_visible_kept(self) -> None:
        labels, counters = _format(
            _events(
                (6.0, "move", 3.0, 0.0),
                (12.0, "move", 5.0, 5.0),  # in the flash: discarded
                (13.0, "scroll", 2.0),  # in the flash: discarded
            )
        )
        self.assertEqual(labels, ["NO_OP", "3 0 0", "NO_OP"])
        self.assertEqual(counters.n_discarded_black, 2)

    def test_pre_first_frame_zone(self) -> None:
        windows = [Window(3, 3, 10)]
        labels, counters = _format(
            _events(
                (1.0, "move", 9.0, 0.0),  # before the first selected frame
                (1.5, "press", "KeyE"),
                (5.0, "release", "KeyE"),
            ),
            windows=windows,
            zones=[],
        )
        self.assertEqual(labels, ["0 0 0 ; +KeyE -KeyE"])
        self.assertEqual(counters.n_discarded_pre_first_frame, 1)
        self.assertEqual(counters.n_presses_clamped, 1)
        _assert_balanced(self, labels)

    def test_trailing_no_coverage(self) -> None:
        labels, counters = _format(
            _events(
                (18.0, "press", "KeyF"),
                (22.0, "move", 4.0, 0.0),  # past the axis end
                (25.0, "release", "KeyF"),  # released after coverage ends
            )
        )
        self.assertEqual(labels, ["NO_OP", "NO_OP", "0 0 0 ; +KeyF -KeyF"])
        self.assertEqual(counters.n_discarded_no_coverage, 1)
        self.assertEqual(counters.n_releases_clamped, 1)
        _assert_balanced(self, labels)


class StateLayerTest(unittest.TestCase):
    def test_full_stream_annotated_and_conserved(self) -> None:
        events = _events(
            (1.0, "move", 2.0, 2.0),
            (6.0, "press", "KeyA"),
            (7.0, "press", "KeyA"),  # autorepeat: redundant
            (8.0, "release", "KeyZ"),  # dangling
            (12.0, "move", 1.0, 1.0),  # in flash
            (12.5, "release", "KeyA"),  # straddle: clamped
            (16.0, "scroll", -3.0),
        )
        labeled, counters = apply_label_policy(
            events, WINDOWS_A, ZONES_A, master_fps=M
        )
        # State layer: every input event comes back, in order, annotated.
        self.assertEqual(len(labeled), len(events))
        for le, e in zip(labeled, events, strict=True):
            self.assertIs(le.event, e)
        # Conservation: owned XOR discarded-with-reason, nothing silent.
        for le in labeled:
            self.assertTrue(
                (le.window is not None) != (le.discard_reason is not None),
                f"event {le.event} neither owned nor discarded (or both)",
            )
        self.assertEqual(counters.n_redundant_press, 1)
        self.assertEqual(counters.n_dangling_release, 1)
        self.assertEqual(counters.n_discarded_black, 1)
        self.assertEqual(counters.n_releases_clamped, 1)
        self.assertEqual(counters.max_simultaneous_keys, 1)

    def test_idle_spans_are_not_dead_zones(self) -> None:
        # An idle-thinned gap between selected frames stays inside the previous
        # frame's window: events there are owned, never discarded.
        windows = [Window(0, 0, 40), Window(40, 40, 50)]  # frames 1..39 thinned
        labels, counters = _format(
            _events((25.0, "move", 7.0, 0.0)), windows=windows, zones=[]
        )
        self.assertEqual(labels, ["7 0 0", "NO_OP"])
        self.assertEqual(counters.n_discarded_black, 0)
        self.assertEqual(counters.n_discarded_no_coverage, 0)


if __name__ == "__main__":
    unittest.main()
