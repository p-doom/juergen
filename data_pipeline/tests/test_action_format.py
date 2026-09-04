from __future__ import annotations

from pathlib import Path

import msgpack
from grammars.deltatype_v2 import CODEC

from pipeline.lib.action_format import format_segment
from pipeline.lib.common import resolve_key_name
from pipeline.lib.events import Window, load_events

MASTER_FPS = 15.0
TARGET_FPS = 1.0
STRIDE = 15


def _write_keylog(path: Path, entries: list[tuple[float, list]]) -> None:
    path.write_bytes(
        msgpack.packb([[round(seconds * 1_000_000), event] for seconds, event in entries])
    )


def _windows(count: int, axis_end: int) -> list[Window]:
    return [
        Window(
            master_idx=index * STRIDE,
            start=index * STRIDE,
            end=(index + 1) * STRIDE if index < count - 1 else axis_end,
        )
        for index in range(count)
    ]


def _labels(entries: list[tuple[float, list]], duration_s: float, tmp_path: Path) -> list[str]:
    bins_count = int(duration_s * TARGET_FPS)
    keylog = tmp_path / "keylog.msgpack"
    _write_keylog(keylog, entries)
    events, _ = load_events(keylog)
    return format_segment(
        events,
        _windows(bins_count, int(duration_s * MASTER_FPS)),
        [],
        master_fps=MASTER_FPS,
    ).labels


def test_canonical_format_is_stable_and_parseable(tmp_path: Path):
    entries = [
        (0.1, ["MouseMove", [3.4, -1.2]]),
        (0.4, ["MouseMove", [0.2, 0.9]]),
        (0.9, ["MouseScroll", [0, -2]]),
        (1.2, ["KeyPress", [0, "KeyA"]]),
        (1.5, ["KeyRelease", [0, "KeyA"]]),
        (2.0, ["MousePress", ["Left"]]),
        (2.3, ["MouseRelease", ["Left"]]),
        (3.7, ["MouseScroll", [1, 0]]),
        (4.2, ["MouseMove", [0.4, 0.4]]),
    ]
    canonical = _labels(entries, 6, tmp_path)
    assert canonical == [
        "4 0 -2",
        "0 0 0 ; +KeyA -KeyA",
        "0 0 0 ; +LMB -LMB",
        "0 0 1",
        "NO_OP",
        "NO_OP",
    ]
    assert all(CODEC.format(CODEC.parse(label)) == label for label in canonical)


def test_canonical_format_preserves_held_state_and_transition_order(tmp_path: Path):
    entries = [
        (0.2, ["KeyRelease", [0, "KeyZ"]]),
        (0.5, ["KeyPress", [0, "ShiftLeft"]]),
        (0.8, ["KeyPress", [0, "ShiftLeft"]]),
        (1.1, ["KeyPress", [0, "KeyA"]]),
        (1.3, ["KeyRelease", [0, "KeyA"]]),
        (1.4, ["KeyPress", [0, "KeyA"]]),
        (1.6, ["KeyRelease", [0, "KeyA"]]),
        (2.5, ["KeyRelease", [0, "ShiftLeft"]]),
        (3.0, ["KeyPress", [0, "KeyB"]]),
    ]
    assert _labels(entries, 4, tmp_path) == [
        "0 0 0 ; +ShiftLeft",
        "0 0 0 ; +KeyA -KeyA +KeyA -KeyA",
        "0 0 0 ; -ShiftLeft",
        "0 0 0 ; +KeyB",
    ]


def test_unknown_keycodes_use_the_canonical_rdev_name(tmp_path: Path):
    assert resolve_key_name([0, "Unknown(-1)"]) is None
    assert resolve_key_name([0, "Unknown(1)"]) == "KC_1"
    canonical = _labels(
        [
            (0.1, ["KeyPress", [0, "Unknown(115)"]]),
            (0.3, ["KeyRelease", [0, "Unknown(115)"]]),
            (0.5, ["KeyPress", [0, "Unknown(999)"]]),
            (0.7, ["KeyRelease", [0, "Unknown(999)"]]),
        ],
        2,
        tmp_path,
    )
    assert canonical == ["0 0 0 ; +Home -Home +KC_999 -KC_999", "NO_OP"]
