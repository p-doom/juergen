from __future__ import annotations

from pathlib import Path

import msgpack
import pytest
from grammars.deltatype_v2 import CODEC

from pipeline.lib.action_format import format_segment
from pipeline.lib.common import resolve_key_name
from pipeline.lib.events import RawEvent, Window, load_events

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
    events = load_events(keylog)
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
        (0.9, ["MouseScroll", [0, -2, 0.0, 0.0]]),
        (1.2, ["KeyPress", [0, "KeyA"]]),
        (1.5, ["KeyRelease", [0, "KeyA"]]),
        (2.0, ["MousePress", ["Left", 0.0, 0.0]]),
        (2.3, ["MouseRelease", ["Left", 0.0, 0.0]]),
        (3.7, ["MouseScroll", [1, 0, 0.0, 0.0]]),
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


def test_canonical_format_preserves_transition_order_and_drops_terminal_press(
    tmp_path: Path,
):
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
        "NO_OP",
    ]


def test_unknown_keycodes_use_the_canonical_rdev_name(tmp_path: Path):
    assert resolve_key_name([0, "Unknown(-1)"]) is None
    assert resolve_key_name([0, "Unknown(1)"]) is None
    assert resolve_key_name([0, "RightArrow"]) == "ArrowRight"
    assert resolve_key_name([0, "Dot"]) == "Period"
    assert resolve_key_name([0, "LeftBracket"]) == "BracketLeft"
    canonical = _labels(
        [
            (0.1, ["KeyPress", [0, "Unknown(115)"]]),
            (0.3, ["KeyRelease", [0, "Unknown(115)"]]),
            (0.5, ["KeyPress", [0, "RightArrow"]]),
            (0.7, ["KeyRelease", [0, "RightArrow"]]),
        ],
        2,
        tmp_path,
    )
    assert canonical == ["0 0 0 ; +Home -Home +ArrowRight -ArrowRight", "NO_OP"]


@pytest.mark.parametrize(
    "event",
    [
        ["UnknownType", []],
        ["MouseMove", [1.0]],
        ["MouseMove", [True, 0.0]],
        ["MouseScroll", [0, 1]],
        ["MouseScroll", [0, float("nan"), 0.0, 0.0]],
        ["KeyPress", [0, "Unknown(999)"]],
        ["KeyPress", [0, "PlayPause"]],
        ["MousePress", ["Other", 0.0, 0.0]],
    ],
)
def test_actionable_events_fail_instead_of_disappearing(event: list, tmp_path: Path):
    keylog = tmp_path / "bad.msgpack"
    keylog.write_bytes(msgpack.packb([[0, event]]))
    with pytest.raises(ValueError, match=r"at .*bad\.msgpack:0"):
        load_events(keylog)


@pytest.mark.parametrize("timestamp", [True, 0.5, -1])
def test_malformed_timestamps_fail(timestamp: object, tmp_path: Path):
    keylog = tmp_path / "bad.msgpack"
    keylog.write_bytes(msgpack.packb([[timestamp, ["MouseMove", [1.0, 0.0]]]]))
    with pytest.raises(ValueError, match="invalid keylog event"):
        load_events(keylog)


def test_nonmonotonic_timestamps_fail(tmp_path: Path):
    keylog = tmp_path / "bad.msgpack"
    keylog.write_bytes(
        msgpack.packb(
            [
                [2, ["MouseMove", [1.0, 0.0]]],
                [1, ["MouseMove", [1.0, 0.0]]],
            ]
        )
    )
    with pytest.raises(ValueError, match="invalid keylog event"):
        load_events(keylog)


def test_missing_keylog_fails(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="keylog is missing"):
        load_events(tmp_path / "missing.msgpack")


def test_observed_non_action_events_are_validated_and_ignored(tmp_path: Path):
    keylog = tmp_path / "keylog.msgpack"
    keylog.write_bytes(
        msgpack.packb(
            [
                [0, ["ContextChanged", ["app"]]],
                [1, ["Metadata", [1728, 1080, 1728, 1080, "2026-06-18T12:34:56Z"]]],
                [
                    2,
                    [
                        "Metadata",
                        [
                            1728,
                            1080,
                            1.0,
                            1728,
                            1080,
                            1.0,
                            1728,
                            1080,
                            1.0,
                            "2026-06-18T12:34:56Z",
                        ],
                    ],
                ],
                [3, ["MouseMove", [1.0, 0.0]]],
            ]
        )
    )
    assert load_events(keylog) == [RawEvent(0, 0.000003, "move", dx=1.0, dy=0.0)]


@pytest.mark.parametrize(
    "event",
    [
        ["ContextChanged", []],
        ["ContextChanged", [""]],
        ["Metadata", [1728, 1080, 1728, 1080, "not-a-timestamp"]],
        [
            "Metadata",
            [
                1728,
                1080,
                1,
                1728,
                1080,
                1.0,
                1728,
                1080,
                1.0,
                "2026-06-18T12:34:56Z",
            ],
        ],
    ],
)
def test_malformed_non_action_payload_fails(event: list[object], tmp_path: Path):
    keylog = tmp_path / "bad.msgpack"
    keylog.write_bytes(msgpack.packb([[0, event]]))
    with pytest.raises(ValueError, match=r"invalid .* payload"):
        load_events(keylog)


def test_formatter_rejects_an_unexpected_event_kind():
    with pytest.raises(ValueError, match="unexpected event kind"):
        format_segment(
            [RawEvent(0, 0.0, "other")],
            [Window(0, 0, 1)],
            [],
            master_fps=1.0,
        )
