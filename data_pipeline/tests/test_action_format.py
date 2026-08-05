"""Regression gate: CanonicalFormatter must be byte-identical to the legacy
``format_action(aggregate_actions(...))`` path on dead-zone-free stretches.

The legacy path bins events at target fps f with ``bucket = int(t_s * f)``;
the new path buckets to master ticks (``int(t_s * M)``) and owns them via
contiguous windows ``[j*stride, (j+1)*stride)``. For integer strides these are
the same partition (``floor(floor(x*M)/stride) == floor(x*M/stride)``), so the
labels must match exactly — including held-set dedup, dangling-release drops,
and delta rounding.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import msgpack

from pipeline.lib.action_format import get_formatter
from pipeline.lib.common import aggregate_actions, format_action
from pipeline.lib.config import SYSTEM_PROMPT
from pipeline.lib.events import RawEvent, Window, load_events
from pipeline.stage_04_build_conversations import default_system_prompt

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
        with self.assertRaisesRegex(ValueError, "invalid input name"):
            self.one_window_label([_key(0, 0.1, "press", "bad name")])


class DefaultSystemPromptTest(unittest.TestCase):
    """Canonical prompts must stay byte-identical to the historical constants."""

    def test_canonical_goal_prompt_matches_legacy(self) -> None:
        self.assertEqual(
            default_system_prompt(get_formatter("canonical"), goal_conditioned=True),
            SYSTEM_PROMPT,
        )

    def test_canonical_goal_free_prompt_matches_legacy(self) -> None:
        self.assertEqual(
            default_system_prompt(get_formatter("canonical"), goal_conditioned=False),
            "You operate a desktop computer. Each user turn shows the current screen. "
            "Reply with the next action as `<dx> <dy> <scroll>` optionally followed by "
            "` ; +KEY -KEY` events, or `NO_OP` if no action.",
        )

    def test_ordered_prompt_describes_the_grammar(self) -> None:
        prompt = default_system_prompt(
            get_formatter("ordered_events_v2"), goal_conditioned=True
        )
        self.assertIn("the next action toward that goal", prompt)
        self.assertIn("move(<dx>,<dy>)", prompt)
        self.assertIn("NO_OP", prompt)


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
