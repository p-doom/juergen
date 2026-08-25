"""Regression gate: CanonicalFormatter must be byte-identical to the legacy
``format_action(aggregate_actions(...))`` path on dead-zone-free stretches.

The legacy path bins events at target fps f with ``bucket = int(t_s * f)``;
the new path buckets to master ticks (``int(t_s * M)``) and owns them via
contiguous windows ``[j*stride, (j+1)*stride)``. For integer strides these are
the same partition (``floor(floor(x*M)/stride) == floor(x*M/stride)``), so the
labels must match exactly — including held-set dedup, dangling-release drops,
and delta rounding. Since the formatter renders through the ``deltatype_v2``
codec and the legacy path does not, this gate now compares the two renderers too.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import grammars
import msgpack

from pipeline.crowdcast.lib.action_format import FORMATTERS, get_formatter
from pipeline.crowdcast.lib.common import aggregate_actions, format_action, resolve_key_name
from pipeline.crowdcast.lib.events import RawEvent, Window, load_events

MASTER_FPS = 15.0
TARGET_FPS = 1.0
STRIDE = 15


def _us(t_s: float) -> int:
    return round(t_s * 1_000_000)


def _write_keylog(path: Path, entries: list[tuple[float, list]]) -> None:
    packed = msgpack.packb([[_us(t), ev] for t, ev in entries])
    path.write_bytes(packed)


def _tiled_windows(n_slots: int, axis_end: int) -> list[Window]:
    """All slots selected: window j = [j*STRIDE, (j+1)*STRIDE), last to axis end."""
    return [
        Window(
            master_idx=j * STRIDE,
            start=j * STRIDE,
            end=(j + 1) * STRIDE if j < n_slots - 1 else axis_end,
        )
        for j in range(n_slots)
    ]


class ByteIdentityTest(unittest.TestCase):
    def assert_identical(self, entries: list[tuple[float, list]], duration_s: float) -> None:
        n_bins = int(duration_s * TARGET_FPS)
        axis_end = int(duration_s * MASTER_FPS)
        with tempfile.TemporaryDirectory() as tmp:
            keylog = Path(tmp) / "keylog.msgpack"
            _write_keylog(keylog, entries)

            bins, _ = aggregate_actions(keylog, n_bins, TARGET_FPS)
            legacy = [format_action(b) for b in bins]

            events, _ = load_events(keylog)
            result = get_formatter("canonical").format_segment(
                events, _tiled_windows(n_bins, axis_end), [], master_fps=MASTER_FPS
            )
        self.assertEqual(result.labels, legacy)

    def test_moves_scrolls_and_keys(self) -> None:
        self.assert_identical(
            [
                (0.1, ["MouseMove", [3.4, -1.2]]),
                (0.4, ["MouseMove", [0.2, 0.9]]),
                (0.9, ["MouseScroll", [0, -2]]),
                (1.2, ["KeyPress", [0, "KeyA"]]),
                (1.5, ["KeyRelease", [0, "KeyA"]]),
                (2.0, ["MousePress", ["Left"]]),  # exactly on a bin boundary
                (2.3, ["MouseRelease", ["Left"]]),
                (3.7, ["MouseScroll", [1, 0]]),  # y==0 -> falls back to x
                (4.2, ["MouseMove", [0.4, 0.4]]),  # rounds to NO_OP
            ],
            duration_s=6.0,
        )

    def test_held_set_dedup_and_dangling(self) -> None:
        self.assert_identical(
            [
                (0.2, ["KeyRelease", [0, "KeyZ"]]),  # dangling: dropped
                (0.5, ["KeyPress", [0, "ShiftLeft"]]),
                (0.8, ["KeyPress", [0, "ShiftLeft"]]),  # autorepeat: deduped
                (1.1, ["KeyPress", [0, "KeyA"]]),
                (1.3, ["KeyRelease", [0, "KeyA"]]),
                (1.4, ["KeyPress", [0, "KeyA"]]),  # re-press after release kept
                (1.6, ["KeyRelease", [0, "KeyA"]]),
                (2.5, ["KeyRelease", [0, "ShiftLeft"]]),
                (3.0, ["KeyPress", [0, "KeyB"]]),  # held at end (no release)
            ],
            duration_s=4.0,
        )

    def test_staggered_combo_order(self) -> None:
        self.assert_identical(
            [
                (0.10, ["KeyPress", [0, "AltLeft"]]),
                (0.20, ["KeyPress", [0, "Tab"]]),
                (0.30, ["KeyRelease", [0, "Tab"]]),
                (0.35, ["KeyPress", [0, "Tab"]]),
                (0.45, ["KeyRelease", [0, "Tab"]]),
                (0.90, ["KeyRelease", [0, "AltLeft"]]),
            ],
            duration_s=2.0,
        )

    def test_unknown_keycode_resolution(self) -> None:
        self.assert_identical(
            [
                (0.1, ["KeyPress", [0, "Unknown(115)"]]),  # -> Home (macOS map)
                (0.3, ["KeyRelease", [0, "Unknown(115)"]]),
                (0.5, ["KeyPress", [0, "Unknown(999)"]]),  # -> KC_999
                (0.7, ["KeyRelease", [0, "Unknown(999)"]]),
                (1.1, ["ContextChanged", []]),  # skipped by both paths
            ],
            duration_s=2.0,
        )

    def test_negative_keycode_is_not_a_keycode(self) -> None:
        """``Unknown(-1)`` used to resolve to the name ``KC_-1``, which the
        bare-token parser rejects — a label no eval could read."""
        self.assertEqual(resolve_key_name([0, "Unknown(-1)"]), None)
        self.assertEqual(resolve_key_name([0, "Unknown(1)"]), "KC_1")

    def test_idle_bins_are_noop(self) -> None:
        self.assert_identical(
            [(0.1, ["MouseMove", [10.0, 0.0]])],
            duration_s=5.0,
        )


def _move(seq: int, t_s: float, dx: float, dy: float) -> RawEvent:
    return RawEvent(seq, t_s, "move", dx=dx, dy=dy)


def _scroll(seq: int, t_s: float, dx: float, dy: float) -> RawEvent:
    # Same collapsed scalar the parser derives (y, falling back to x).
    return RawEvent(seq, t_s, "scroll", dx=dx, dy=dy, scroll=dy if dy != 0 else dx)


def _key(seq: int, t_s: float, kind: str, name: str) -> RawEvent:
    return RawEvent(seq, t_s, kind, name=name)


class OrderedFormatterTest(unittest.TestCase):
    """Ported from the yll/action-format branch's project_ordered_action tests,
    re-expressed over the realigned formatter interface (windows in master
    ticks at 15 fps; the default 10 Hz motor grid)."""

    def labels(self, events: list[RawEvent], windows: list[Window], hz: float = 10.0) -> list[str]:
        result = get_formatter("ordered_events_v2", continuous_action_hz=hz).format_segment(
            events, windows, [], master_fps=MASTER_FPS
        )
        return result.labels

    def one_window_label(self, events: list[RawEvent], hz: float = 10.0) -> str:
        return self.labels(events, [Window(master_idx=0, start=0, end=30)], hz=hz)[0]

    def test_discrete_event_splits_movement_inside_one_motor_tick(self) -> None:
        self.assertEqual(
            self.one_window_label([
                _move(0, 0.01, 1.0, 0.0),
                _move(1, 0.02, 3.0, -1.0),
                _key(2, 0.03, "press", "LMB"),
                _move(3, 0.04, 2.0, 0.0),
                _key(4, 0.05, "release", "LMB"),
            ]),
            "move(4,-1); down(LMB); move(2,0); up(LMB)",
        )

    def test_motor_tick_boundary_splits_continuous_actions(self) -> None:
        self.assertEqual(
            self.one_window_label([_move(0, 0.01, 1.0, 0.0), _move(1, 0.10, 2.0, 0.0)]),
            "move(1,0); move(2,0)",
        )

    def test_continuous_action_hz_widens_the_motor_tick(self) -> None:
        self.assertEqual(
            self.one_window_label(
                [_move(0, 0.01, 1.0, 0.0), _move(1, 0.10, 2.0, 0.0)], hz=1.0
            ),
            "move(3,0)",
        )

    def test_scroll_is_ordered_and_two_dimensional(self) -> None:
        self.assertEqual(
            self.one_window_label([
                _scroll(0, 0.01, 2.0, -3.0),
                _scroll(1, 0.02, 1.0, -2.0),
                _key(2, 0.03, "press", "KeyA"),
                _scroll(3, 0.04, -1.0, 4.0),
            ]),
            "scroll(3,-5); down(KeyA); scroll(-1,4)",
        )

    def test_rounding_happens_after_accumulation(self) -> None:
        self.assertEqual(
            self.one_window_label([_move(0, 0.01, 0.3, 0.3), _move(1, 0.02, 0.4, 0.4)]),
            "move(1,1)",
        )

    def test_zero_continuous_actions_are_omitted(self) -> None:
        self.assertEqual(
            self.one_window_label([
                _move(0, 0.01, 0.2, 0.2),
                _scroll(1, 0.02, 0.0, 0.0),
                _key(2, 0.03, "press", "LMB"),
                _key(3, 0.04, "release", "LMB"),
            ]),
            "down(LMB); up(LMB)",
        )

    def test_empty_window_is_no_op(self) -> None:
        windows = [
            Window(master_idx=0, start=0, end=15),
            Window(master_idx=15, start=15, end=30),
        ]
        self.assertEqual(
            self.labels([_move(0, 0.1, 5.0, 0.0)], windows),
            ["move(5,0)", "NO_OP"],
        )

    def test_press_and_release_land_in_their_own_windows(self) -> None:
        windows = [
            Window(master_idx=0, start=0, end=15),
            Window(master_idx=15, start=15, end=30),
        ]
        self.assertEqual(
            self.labels(
                [_key(0, 0.5, "press", "LMB"), _key(1, 1.5, "release", "LMB")], windows
            ),
            ["down(LMB)", "up(LMB)"],
        )

    def test_primitive_counts_reported(self) -> None:
        result = get_formatter("ordered_events_v2").format_segment(
            [
                _move(0, 0.01, 3.0, 0.0),
                _key(1, 0.03, "press", "LMB"),
                _key(2, 0.04, "release", "LMB"),
            ],
            [Window(master_idx=0, start=0, end=30)],
            [],
            master_fps=MASTER_FPS,
        )
        self.assertEqual(
            result.primitive_counts, {"move": 1, "scroll": 0, "down": 1, "up": 1}
        )

    def test_scroll_axes_survive_the_keylog_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            keylog = Path(tmp) / "keylog.msgpack"
            _write_keylog(keylog, [(0.1, ["MouseScroll", [2, -3]])])
            events, _ = load_events(keylog)
        windows = [Window(master_idx=0, start=0, end=15)]
        ordered = get_formatter("ordered_events_v2").format_segment(
            events, windows, [], master_fps=MASTER_FPS
        )
        canonical = get_formatter("canonical").format_segment(
            events, windows, [], master_fps=MASTER_FPS
        )
        self.assertEqual(ordered.labels, ["scroll(2,-3)"])
        self.assertEqual(canonical.labels, ["0 0 -3"])

    def test_invalid_rate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "continuous_action_hz"):
            get_formatter("ordered_events_v2", continuous_action_hz=0.0)

    def test_invalid_input_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "is not a name down\\(\\) can spell"):
            self.one_window_label([_key(0, 0.1, "press", "bad name")])


def _type(seq: int, t_s: float, name: str) -> list[RawEvent]:
    """One printable key's press+release, 10 ms apart."""
    return [_key(seq, t_s, "press", name), _key(seq + 1, t_s + 0.01, "release", name)]


class OrderedTypingFormatterTest(unittest.TestCase):
    """The ``ordered_events_v3`` emitter: v2's extraction with balanced typing
    runs collapsed into one ``type("...")``.

    The emitter was cut when the repo was rearchitected while the parser was
    kept, so v3 could be read but not written — and v3 is the format the
    crowd-cast ``eov3`` datasets are in.
    """

    def labels(self, events: list[RawEvent], windows: list[Window] | None = None) -> list[str]:
        return get_formatter("ordered_events_v3").format_segment(
            events,
            windows or [Window(master_idx=0, start=0, end=30)],
            [],
            master_fps=MASTER_FPS,
        ).labels

    def test_a_balanced_run_becomes_one_type_call(self) -> None:
        events = [*_type(0, 0.01, "KeyH"), *_type(2, 0.03, "KeyI")]
        self.assertEqual(self.labels(events), ['type("hi")'])

    def test_shift_is_absorbed_into_the_shifted_character(self) -> None:
        events = [
            _key(0, 0.01, "press", "ShiftLeft"),
            *_type(1, 0.02, "KeyH"),
            _key(3, 0.05, "release", "ShiftLeft"),
            *_type(4, 0.07, "KeyI"),
        ]
        self.assertEqual(self.labels(events), ['type("Hi")'])

    def test_rollover_still_collapses(self) -> None:
        """Natural fast typing releases out of order; characters follow the
        ``down`` order, not the release order."""
        events = [
            _key(0, 0.01, "press", "KeyH"),
            _key(1, 0.02, "press", "KeyE"),
            _key(2, 0.03, "release", "KeyH"),
            _key(3, 0.04, "release", "KeyE"),
        ]
        self.assertEqual(self.labels(events), ['type("he")'])

    def test_a_non_typing_primitive_breaks_the_run(self) -> None:
        events = [
            *_type(0, 0.01, "KeyH"),
            _key(2, 0.03, "press", "Return"),
            _key(3, 0.04, "release", "Return"),
            *_type(4, 0.06, "KeyI"),
        ]
        self.assertEqual(
            self.labels(events),
            ['type("h"); down(Return); up(Return); type("i")'],
        )

    def test_a_held_non_shift_modifier_vetoes_typing(self) -> None:
        """Ctrl+C is a chord, not typed text."""
        events = [
            _key(0, 0.01, "press", "ControlLeft"),
            *_type(1, 0.02, "KeyC"),
            _key(3, 0.05, "release", "ControlLeft"),
        ]
        self.assertEqual(
            self.labels(events),
            ["down(ControlLeft); down(KeyC); up(KeyC); up(ControlLeft)"],
        )

    def test_a_bare_shift_tap_renders_explicitly(self) -> None:
        events = [
            _key(0, 0.01, "press", "ShiftLeft"),
            _key(1, 0.02, "release", "ShiftLeft"),
        ]
        self.assertEqual(self.labels(events), ["down(ShiftLeft); up(ShiftLeft)"])

    def test_a_key_held_across_a_window_boundary_never_collapses(self) -> None:
        windows = [
            Window(master_idx=0, start=0, end=15),
            Window(master_idx=15, start=15, end=30),
        ]
        events = [_key(0, 0.5, "press", "KeyH"), _key(1, 1.5, "release", "KeyH")]
        self.assertEqual(self.labels(events, windows), ["down(KeyH)", "up(KeyH)"])

    def test_shift_held_across_a_boundary_still_shifts_the_next_window(self) -> None:
        """``held_mods`` crosses windows, so the second window's character is
        capital even though its Shift down is in the first."""
        windows = [
            Window(master_idx=0, start=0, end=15),
            Window(master_idx=15, start=15, end=30),
        ]
        events = [
            _key(0, 0.5, "press", "ShiftLeft"),
            *_type(1, 1.2, "KeyH"),
            _key(3, 1.5, "release", "ShiftLeft"),
        ]
        self.assertEqual(
            self.labels(events, windows),
            ["down(ShiftLeft)", 'type("H"); up(ShiftLeft)'],
        )

    def test_the_quote_payload_is_escaped_by_the_codec(self) -> None:
        events = [*_type(0, 0.01, "Quote"), *_type(2, 0.03, "BackSlash")]
        self.assertEqual(self.labels(events), ['type("\'\\\\")'])

    def test_v2_never_types_and_v3_shares_its_extraction(self) -> None:
        events = [_move(0, 0.01, 3.0, 0.0), *_type(1, 0.03, "KeyH")]
        self.assertEqual(
            get_formatter("ordered_events_v2").format_segment(
                events, [Window(master_idx=0, start=0, end=30)], [],
                master_fps=MASTER_FPS,
            ).labels,
            ["move(3,0); down(KeyH); up(KeyH)"],
        )
        self.assertEqual(self.labels(events), ['move(3,0); type("h")'])

    def test_type_is_counted(self) -> None:
        result = get_formatter("ordered_events_v3").format_segment(
            [_move(0, 0.01, 3.0, 0.0), *_type(1, 0.03, "KeyH")],
            [Window(master_idx=0, start=0, end=30)],
            [],
            master_fps=MASTER_FPS,
        )
        self.assertEqual(
            result.primitive_counts,
            {"move": 1, "scroll": 0, "down": 0, "up": 0, "type": 1},
        )


class EmitterSpeaksItsGrammarTest(unittest.TestCase):
    """Every label a formatter writes must parse — and re-render identically —
    under the codec of the grammar it names.

    The emitter and the eval parser used to be independent implementations:
    ``format_action`` interpolated ``f"{sign}{name}"`` with no name validation
    while the bare-token parser rejected anything outside
    ``[A-Za-z_][A-Za-z0-9_]*``, so nothing detected a label no eval could read.
    """

    def round_trip(self, format_name: str, events: list[RawEvent]) -> list[str]:
        formatter = get_formatter(format_name)
        codec = grammars.load(formatter.grammar)
        windows = [
            Window(master_idx=0, start=0, end=15),
            Window(master_idx=15, start=15, end=30),
        ]
        labels = formatter.format_segment(
            events, windows, [], master_fps=MASTER_FPS
        ).labels
        for label in labels:
            self.assertEqual(codec.format(codec.parse(label)), label, label)
        return labels

    def stream(self) -> list[RawEvent]:
        """Motion, a click, a chord, an unmapped keycode, a press held across the
        window boundary, and an idle stretch."""
        return [
            _move(0, 0.10, 4.0, -1.2),
            _key(1, 0.20, "press", "LMB"),
            _key(2, 0.25, "release", "LMB"),
            _scroll(3, 0.30, 0.0, -3.0),
            _key(4, 0.40, "press", "ControlLeft"),
            _key(5, 0.45, "press", "KeyC"),
            _key(6, 0.50, "release", "KeyC"),
            _key(7, 0.55, "release", "ControlLeft"),
            _key(8, 0.60, "press", "KC_999"),
            _key(9, 0.65, "release", "KC_999"),
            _key(10, 0.90, "press", "ShiftLeft"),
            _key(11, 1.40, "release", "ShiftLeft"),
        ]

    def test_canonical_labels_are_deltatype_v2(self) -> None:
        self.assertEqual(get_formatter("canonical").grammar, "deltatype_v2")
        labels = self.round_trip("canonical", self.stream())
        self.assertEqual(
            labels,
            [
                "4 -1 -3 ; +LMB -LMB +ControlLeft +KeyC -KeyC -ControlLeft "
                "+KC_999 -KC_999 +ShiftLeft",
                "0 0 0 ; -ShiftLeft",
            ],
        )

    def test_ordered_labels_are_ordered_events_v3(self) -> None:
        self.assertEqual(get_formatter("ordered_events_v2").grammar, "ordered_events_v3")
        labels = self.round_trip("ordered_events_v2", self.stream())
        self.assertEqual(
            labels,
            [
                "move(4,-1); down(LMB); up(LMB); scroll(0,-3); down(ControlLeft); "
                "down(KeyC); up(KeyC); up(ControlLeft); down(KC_999); up(KC_999); "
                "down(ShiftLeft)",
                "up(ShiftLeft)",
            ],
        )

    def test_idle_window_round_trips_as_the_grammar_spells_it(self) -> None:
        for format_name in sorted(FORMATTERS):
            self.assertEqual(self.round_trip(format_name, []), ["NO_OP", "NO_OP"])

    def test_a_typed_payload_round_trips_through_the_codec_escapes(self) -> None:
        labels = self.round_trip(
            "ordered_events_v3",
            [
                _key(0, 0.01, "press", "Quote"),
                _key(1, 0.02, "release", "Quote"),
                _key(2, 0.03, "press", "BackSlash"),
                _key(3, 0.04, "release", "BackSlash"),
            ],
        )
        self.assertEqual(labels, ['type("\'\\\\")', "NO_OP"])

    def emit(self, format_name: str, name: str) -> list[str]:
        return get_formatter(format_name).format_segment(
            [_key(0, 0.1, "press", name)],
            [Window(master_idx=0, start=0, end=15)],
            [],
            master_fps=MASTER_FPS,
        ).labels

    def test_a_name_the_grammar_cannot_spell_is_never_written(self) -> None:
        """Each emitter is held to its OWN grammar's name class, and the two
        differ: a bare-token tail spells a name after a ``+``/``-`` sign, so it
        must be an identifier, while a mini-program spells it inside parentheses
        and only needs to avoid the punctuation. ``KC_-1`` is the name
        ``resolve_key_name`` used to build for ``Unknown(-1)``.
        """
        with self.assertRaises(ValueError):
            self.emit("canonical", "KC_-1")
        self.assertEqual(self.emit("ordered_events_v2", "KC_-1"), ["down(KC_-1)"])
        # `resolve_button_name` spells an unrecognised button `M_<name>`, so a
        # parenthesised one reaches every emitter and none can express it.
        for format_name in sorted(FORMATTERS):
            with self.assertRaises(ValueError):
                self.emit(format_name, "M_Unknown(8)")


class SystemPromptTest(unittest.TestCase):
    """Stage 04 has no prompt of its own: it is the grammar's ``describe()``."""

    def test_prompt_is_the_grammars_own(self) -> None:
        for format_name in sorted(FORMATTERS):
            formatter = get_formatter(format_name)
            self.assertEqual(
                grammars.describe(formatter.grammar),
                grammars.load(formatter.grammar).describe(),
            )

    def test_every_formatter_names_a_registered_grammar(self) -> None:
        for format_name in sorted(FORMATTERS):
            self.assertIn(get_formatter(format_name).grammar, grammars.available())


class FormatterRegistryTest(unittest.TestCase):
    def test_lookup(self) -> None:
        self.assertEqual(get_formatter("canonical").name, "canonical")
        self.assertEqual(get_formatter("ordered_events_v2").name, "ordered_events_v2")
        self.assertEqual(
            get_formatter("ordered_events_v2", continuous_action_hz=5.0).continuous_action_hz,
            5.0,
        )
        with self.assertRaises(KeyError):
            get_formatter("nope")


if __name__ == "__main__":
    unittest.main()


class JitterDeadbandTest(unittest.TestCase):
    """``jitter_deadband_px``: drop incidental single-axis noise, keep intent.

    The knob exists because a demonstrator's hand rests on the mouse while
    clicking, scrolling and typing, and the recorder faithfully captures the
    resulting 1-2 px tremor. Trained on, it teaches the model that every click
    is preceded by a nudge. Suppression is deliberately narrow: one axis only,
    within the deadband, and never at the cost of a window's last primitive.
    """

    def labels(
        self,
        events: list[RawEvent],
        windows: list[Window] | None = None,
        *,
        grammar: str = "ordered_events_v2",
        hz: float = 10.0,
        jitter_px: int = 0,
    ) -> list[str]:
        return get_formatter(
            grammar, continuous_action_hz=hz, jitter_deadband_px=jitter_px
        ).format_segment(
            events,
            windows or [Window(master_idx=0, start=0, end=30)],
            [],
            master_fps=MASTER_FPS,
        ).labels

    def one(self, events: list[RawEvent], **kw: object) -> str:
        return self.labels(events, **kw)[0]  # type: ignore[arg-type]

    # ---------------------------------------------------------------- off

    def test_off_by_default(self) -> None:
        """Threshold 0 matches only an exact (0,0), which never renders — so
        the default cannot change any existing dataset's labels."""
        events = [
            _scroll(0, 0.01, 0.0, -1.0),
            _move(1, 0.02, -1.0, 0.0),
            _scroll(2, 0.03, 0.0, -4.0),
        ]
        self.assertEqual(self.one(events), "scroll(0,-1); move(-1,0); scroll(0,-4)")

    # ------------------------------------------------------------ dropping

    def test_drops_sandwiched_move_and_merges_the_flanks(self) -> None:
        """The two scrolls the noise move separated merge exactly as if it had
        never occurred — not `scroll(0,-1); scroll(0,-4)`."""
        events = [
            _scroll(0, 0.01, 0.0, -1.0),
            _move(1, 0.02, -1.0, 0.0),
            _scroll(2, 0.03, 0.0, -4.0),
        ]
        self.assertEqual(self.one(events, jitter_px=1), "scroll(0,-5)")

    def test_run_dropped_when_the_window_has_other_content(self) -> None:
        events = [
            _key(0, 0.01, "press", "LMB"),
            _key(1, 0.02, "release", "LMB"),
            _scroll(2, 0.03, 0.0, -1.0),
            _move(3, 0.04, 0.0, 1.0),
        ]
        self.assertEqual(self.one(events, jitter_px=1), "down(LMB); up(LMB)")

    def test_between_a_clicks_down_and_up(self) -> None:
        """down and up are different kinds; the move between them goes, and
        they never merge into each other."""
        events = [
            _key(0, 0.01, "press", "LMB"),
            _move(1, 0.02, -1.0, 0.0),
            _key(2, 0.03, "release", "LMB"),
        ]
        self.assertEqual(self.one(events, jitter_px=1), "down(LMB); up(LMB)")

    # -------------------------------------------------------------- keeping

    def test_never_empties_a_window(self) -> None:
        """Both primitives are individually jitter, but they are the window's
        entire content: emptying it would train an idle turn where the
        demonstrator acted. Order is preserved either way."""
        self.assertEqual(
            self.one(
                [_move(0, 0.01, 1.0, 0.0), _scroll(1, 0.02, 0.0, -1.0)], jitter_px=1
            ),
            "move(1,0); scroll(0,-1)",
        )
        self.assertEqual(
            self.one(
                [_scroll(0, 0.01, 0.0, -1.0), _move(1, 0.02, 1.0, 0.0)], jitter_px=1
            ),
            "scroll(0,-1); move(1,0)",
        )

    def test_a_diagonal_is_never_jitter(self) -> None:
        """Two axes at once takes intent hand tremor does not produce, however
        small each axis is."""
        self.assertEqual(
            self.one(
                [_move(0, 0.01, 1.0, 0.0), _move(1, 0.02, 0.0, -1.0)], jitter_px=1
            ),
            "move(1,-1)",
        )
        self.assertEqual(
            self.one(
                [
                    _key(0, 0.01, "press", "LMB"),
                    _move(1, 0.02, 1.0, 0.0),
                    _move(2, 0.03, 0.0, -1.0),
                    _key(3, 0.04, "release", "LMB"),
                ],
                jitter_px=1,
            ),
            "down(LMB); move(1,-1); up(LMB)",
        )

    def test_the_threshold_is_on_the_merged_result_not_its_parts(self) -> None:
        events = [
            _scroll(0, 0.01, 0.0, -1.0),
            _move(1, 0.02, 2.0, 0.0),
            _scroll(2, 0.03, 0.0, -1.0),
        ]
        self.assertEqual(self.one(events, jitter_px=1), "scroll(0,-2); move(2,0)")

    # -------------------------------------------------------------- merging

    def test_merges_across_a_motor_tick_boundary(self) -> None:
        """Still one contiguous move-only run, so it merges to ONE primitive
        whose sum clears the threshold and is kept."""
        self.assertEqual(
            self.one(
                [_move(0, 0.01, 1.0, 0.0), _move(1, 0.10, 1.0, 0.0)], jitter_px=1
            ),
            "move(2,0)",
        )

    def test_interleaved_scroll_and_move_sum_independently(self) -> None:
        """Each kind sums across the whole run; the output leads with whichever
        kind opened it."""
        self.assertEqual(
            self.one(
                [
                    _scroll(0, 0.01, 0.0, -1.0),
                    _move(1, 0.02, 0.0, 1.0),
                    _scroll(2, 0.03, 0.0, -2.0),
                    _move(3, 0.04, 1.0, 0.0),
                    _scroll(4, 0.05, 0.0, -11.0),
                ],
                jitter_px=1,
            ),
            "scroll(0,-14); move(1,1)",
        )
        self.assertEqual(
            self.one(
                [
                    _move(0, 0.01, 1.0, 0.0),
                    _scroll(1, 0.02, 0.0, -1.0),
                    _move(2, 0.03, 2.0, 0.0),
                    _scroll(3, 0.04, 0.0, -3.0),
                ],
                jitter_px=1,
            ),
            "move(3,0); scroll(0,-4)",
        )

    def test_a_run_summing_to_zero_is_omitted(self) -> None:
        """Not suppression emptying real content — there was genuinely no net
        motion, same as `move(0,0)` never rendering anywhere else."""
        self.assertEqual(
            self.one(
                [_move(0, 0.01, 1.0, 0.0), _move(1, 0.02, -1.0, 0.0)], jitter_px=1
            ),
            "NO_OP",
        )

    # ------------------------------------------------- v3: after the collapse

    def test_v3_sandwiched_between_typing_runs_reunites_them(self) -> None:
        """Applied AFTER the typing collapse, so it sees `type()` rather than
        the down/up pairs underneath — and dropping the move yields ONE
        `type("hi")`, not two adjacent `type("h"); type("i")`."""
        events = [
            *_type(0, 0.01, "KeyH"),
            _move(2, 0.03, 1.0, 0.0),
            *_type(3, 0.05, "KeyI"),
        ]
        self.assertEqual(
            self.one(events, grammar="ordered_events_v3", jitter_px=1), 'type("hi")'
        )

    def test_v3_reunion_needs_the_whole_separator_dropped(self) -> None:
        events = [
            *_type(0, 0.01, "KeyH"),
            _move(2, 0.03, 5.0, 0.0),
            *_type(3, 0.05, "KeyI"),
        ]
        self.assertEqual(
            self.one(events, grammar="ordered_events_v3", jitter_px=1),
            'type("h"); move(5,0); type("i")',
        )

    # ------------------------------------------------------------ validation

    def test_a_negative_deadband_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            get_formatter("ordered_events_v3", jitter_deadband_px=-1)
