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

from realigned_pipeline.lib.action_format import get_formatter
from realigned_pipeline.lib.common import aggregate_actions, format_action
from realigned_pipeline.lib.events import Window, load_events

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


class FormatterRegistryTest(unittest.TestCase):
    def test_lookup(self) -> None:
        self.assertEqual(get_formatter("canonical").name, "canonical")
        with self.assertRaises(KeyError):
            get_formatter("nope")


if __name__ == "__main__":
    unittest.main()
