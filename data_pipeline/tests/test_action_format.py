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

import json
import re
import tempfile
import unittest
from pathlib import Path

import msgpack

from realigned_pipeline.lib.action_format import (
    RDEV_TO_COMPUTER_USE_KEY,
    get_formatter,
    render_tool_call,
)
from realigned_pipeline.lib.common import aggregate_actions, format_action
from realigned_pipeline.lib.config import SYSTEM_PROMPT
from realigned_pipeline.lib.events import RawEvent, Window, load_events
from realigned_pipeline.stage_04_build_conversations import default_system_prompt

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


class OrderedTypingFormatterTest(unittest.TestCase):
    """``ordered_events_v3``: v2 plus the ``type("...")`` collapse. Anything
    that is not a plain typing run must stay byte-identical to v2."""

    def labels(self, events: list[RawEvent], windows: list[Window], hz: float = 10.0) -> list[str]:
        result = get_formatter("ordered_events_v3", continuous_action_hz=hz).format_segment(
            events, windows, [], master_fps=MASTER_FPS
        )
        return result.labels

    def one_window_label(self, events: list[RawEvent], hz: float = 10.0) -> str:
        return self.labels(events, [Window(master_idx=0, start=0, end=30)], hz=hz)[0]

    @staticmethod
    def _typed(names: list[str], t0: float = 0.01, seq0: int = 0) -> list[RawEvent]:
        """One press+release pair per name, 10 ms apart."""
        events, t, seq = [], t0, seq0
        for name in names:
            events.append(_key(seq, t, "press", name))
            events.append(_key(seq + 1, t + 0.01, "release", name))
            t, seq = t + 0.02, seq + 2
        return events

    def test_no_typing_matches_v2_byte_for_byte(self) -> None:
        events = [
            _move(0, 0.01, 1.0, 0.0),
            _move(1, 0.02, 3.0, -1.0),
            _key(2, 0.03, "press", "LMB"),
            _move(3, 0.04, 2.0, 0.0),
            _key(4, 0.05, "release", "LMB"),
            _scroll(5, 0.30, 0.0, -2.0),
            _key(6, 1.10, "press", "Tab"),
            _key(7, 1.20, "release", "Tab"),
        ]
        windows = [
            Window(master_idx=0, start=0, end=15),
            Window(master_idx=15, start=15, end=30),
        ]
        v2 = get_formatter("ordered_events_v2").format_segment(
            events, windows, [], master_fps=MASTER_FPS
        )
        self.assertEqual(self.labels(events, windows), v2.labels)

    def test_lowercase_run_collapses(self) -> None:
        self.assertEqual(
            self.one_window_label(self._typed(["KeyH", "KeyE", "KeyL", "KeyL", "KeyO"])),
            'type("hello")',
        )

    def test_shift_absorbed_for_capital(self) -> None:
        events = [
            _key(0, 0.01, "press", "ShiftLeft"),
            _key(1, 0.02, "press", "KeyH"),
            _key(2, 0.03, "release", "KeyH"),
            _key(3, 0.04, "release", "ShiftLeft"),
            *self._typed(["KeyE", "KeyL", "KeyL", "KeyO"], t0=0.05, seq0=4),
        ]
        self.assertEqual(self.one_window_label(events), 'type("Hello")')

    def test_shifted_digit_is_us_layout(self) -> None:
        events = [
            _key(0, 0.01, "press", "ShiftRight"),
            _key(1, 0.02, "press", "Num1"),
            _key(2, 0.03, "release", "Num1"),
            _key(3, 0.04, "release", "ShiftRight"),
        ]
        self.assertEqual(self.one_window_label(events), 'type("!")')

    def test_quote_and_backslash_are_escaped(self) -> None:
        events = [
            _key(0, 0.01, "press", "ShiftLeft"),
            _key(1, 0.02, "press", "Quote"),  # shifted Quote -> "
            _key(2, 0.03, "release", "Quote"),
            _key(3, 0.04, "release", "ShiftLeft"),
            *self._typed(["BackSlash"], t0=0.05, seq0=4),  # -> \
        ]
        self.assertEqual(self.one_window_label(events), r'type("\"\\")')

    def test_mouse_move_breaks_the_run(self) -> None:
        events = [
            *self._typed(["KeyH"], t0=0.01, seq0=0),
            _move(2, 0.03, 5.0, 0.0),
            *self._typed(["KeyI"], t0=0.05, seq0=3),
        ]
        self.assertEqual(
            self.one_window_label(events), 'type("h"); move(5,0); type("i")'
        )

    def test_press_held_across_boundary_stays_down(self) -> None:
        windows = [
            Window(master_idx=0, start=0, end=15),
            Window(master_idx=15, start=15, end=30),
        ]
        self.assertEqual(
            self.labels(
                [_key(0, 0.5, "press", "KeyH"), _key(1, 1.5, "release", "KeyH")], windows
            ),
            ["down(KeyH)", "up(KeyH)"],
        )

    def test_shift_over_non_typing_key_is_rendered(self) -> None:
        events = [
            _key(0, 0.01, "press", "ShiftLeft"),
            _key(1, 0.02, "press", "Tab"),
            _key(2, 0.03, "release", "Tab"),
            _key(3, 0.04, "release", "ShiftLeft"),
        ]
        self.assertEqual(
            self.one_window_label(events),
            "down(ShiftLeft); down(Tab); up(Tab); up(ShiftLeft)",
        )

    def test_return_and_backspace_stay_discrete(self) -> None:
        events = [
            *self._typed(["KeyH"], t0=0.01, seq0=0),
            *self._typed(["Return"], t0=0.03, seq0=2),
            *self._typed(["KeyI"], t0=0.05, seq0=4),
            *self._typed(["Backspace"], t0=0.07, seq0=6),
        ]
        self.assertEqual(
            self.one_window_label(events),
            'type("h"); down(Return); up(Return); type("i"); '
            "down(Backspace); up(Backspace)",
        )

    def test_ctrl_chord_is_not_typing(self) -> None:
        events = [
            _key(0, 0.01, "press", "ControlLeft"),
            _key(1, 0.02, "press", "KeyC"),
            _key(2, 0.03, "release", "KeyC"),
            _key(3, 0.04, "release", "ControlLeft"),
        ]
        self.assertEqual(
            self.one_window_label(events),
            "down(ControlLeft); down(KeyC); up(KeyC); up(ControlLeft)",
        )

    def test_bare_shift_tap_is_rendered(self) -> None:
        events = [
            _key(0, 0.01, "press", "ShiftLeft"),
            _key(1, 0.02, "release", "ShiftLeft"),
        ]
        self.assertEqual(self.one_window_label(events), "down(ShiftLeft); up(ShiftLeft)")

    def test_shift_enclosing_a_move_renders_but_pairs_still_type(self) -> None:
        events = [
            _key(0, 0.01, "press", "ShiftLeft"),
            *self._typed(["KeyH"], t0=0.02, seq0=1),
            _move(3, 0.04, 5.0, 0.0),
            *self._typed(["KeyE"], t0=0.06, seq0=4),
            _key(6, 0.08, "release", "ShiftLeft"),
        ]
        self.assertEqual(
            self.one_window_label(events),
            'down(ShiftLeft); type("H"); move(5,0); type("E"); up(ShiftLeft)',
        )

    def test_shift_held_across_windows_shifts_but_is_not_absorbed(self) -> None:
        windows = [
            Window(master_idx=0, start=0, end=15),
            Window(master_idx=15, start=15, end=30),
        ]
        events = [
            _key(0, 0.5, "press", "ShiftLeft"),
            *self._typed(["KeyH"], t0=1.1, seq0=1),
            _key(3, 1.5, "release", "ShiftLeft"),
        ]
        self.assertEqual(
            self.labels(events, windows),
            ["down(ShiftLeft)", 'type("H"); up(ShiftLeft)'],
        )

    def test_primitive_counts_include_type(self) -> None:
        result = get_formatter("ordered_events_v3").format_segment(
            [
                *self._typed(["KeyH", "KeyI"], t0=0.01, seq0=0),
                _key(4, 0.06, "press", "LMB"),
                _key(5, 0.07, "release", "LMB"),
            ],
            [Window(master_idx=0, start=0, end=30)],
            [],
            master_fps=MASTER_FPS,
        )
        self.assertEqual(
            result.primitive_counts,
            {"move": 0, "scroll": 0, "down": 1, "up": 1, "type": 1},
        )

    def test_registry_and_default_prompt(self) -> None:
        self.assertEqual(get_formatter("ordered_events_v3").name, "ordered_events_v3")
        prompt = default_system_prompt(
            get_formatter("ordered_events_v3"), goal_conditioned=True
        )
        self.assertIn('type("<text>")', prompt)
        self.assertIn("NO_OP", prompt)


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


CUA_V4_PROMPT = (Path(__file__).resolve().parents[1] / "realigned_pipeline"
                 / "system_prompts" / "cua_v4_thinking.txt")

# One single-line JSON body per block; the body never contains a raw newline
# (json escapes them), so the strict per-line pattern is exact.
_TOOL_CALL_RE = re.compile(r"<tool_call>\n([^\n]*)\n</tool_call>")

# Arguments each action requires beyond "action" — the "Required only by"
# notes of the cua_v4 tool spec. The formatter must emit exactly these.
CUA_REQUIRED_ARGS = {
    "key": {"keys"}, "type": {"text"}, "mouse_move_rel": {"delta"},
    "left_click": set(), "right_click": set(), "middle_click": set(),
    "double_click": set(), "triple_click": set(),
    "button_down": {"button"}, "button_up": {"button"},
    "key_down": {"key"}, "key_up": {"key"},
    "scroll": {"pixels"}, "hscroll": {"pixels"},
    "wait": {"time"}, "terminate": {"status"},
}


def load_cua_v4_tool_spec() -> dict:
    """The computer_use function spec exactly as the system prompt binds it."""
    text = CUA_V4_PROMPT.read_text()
    # the prompt mentions "<tools></tools> XML tags" in prose first; the real
    # block is the LAST <tools> occurrence
    tools = text.split("<tools>")[-1].split("</tools>", 1)[0].strip()
    return json.loads(tools)


class ComputerUseFormatterTest(unittest.TestCase):
    """``computer_use_rel_v1``: Qwen3-VL-native <tool_call> JSON blocks. Every
    emitted label is validated against the cua_v4_thinking.txt tool spec
    (round-trips through json; action in the enum; exactly the arguments the
    spec requires)."""

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        spec = load_cua_v4_tool_spec()["function"]["parameters"]["properties"]
        cls.action_enum = set(spec["action"]["enum"])
        cls.arg_names = set(spec)
        assert set(CUA_REQUIRED_ARGS) == cls.action_enum

    def validate(self, label: str) -> list[dict]:
        """Parse + spec-validate one label; returns the arguments dicts."""
        bodies = _TOOL_CALL_RE.findall(label)
        rebuilt = "\n".join(f"<tool_call>\n{b}\n</tool_call>" for b in bodies)
        self.assertEqual(label, rebuilt, "label is not pure tool_call blocks")
        self.assertGreaterEqual(len(bodies), 1)
        calls = []
        for body in bodies:
            obj = json.loads(body)
            self.assertEqual(set(obj), {"name", "arguments"})
            self.assertEqual(obj["name"], "computer_use")
            args = obj["arguments"]
            action = args["action"]
            self.assertIn(action, self.action_enum)
            self.assertEqual(set(args), {"action"} | CUA_REQUIRED_ARGS[action])
            self.assertLessEqual(set(args), self.arg_names)
            calls.append(args)
        return calls

    def calls(self, events: list[RawEvent],
              windows: list[Window] | None = None) -> list[list[dict]]:
        windows = windows or [Window(master_idx=0, start=0, end=30)]
        self.result = get_formatter("computer_use_rel_v1").format_segment(
            events, windows, [], master_fps=MASTER_FPS
        )
        return [self.validate(label) for label in self.result.labels]

    def one_window(self, events: list[RawEvent]) -> list[dict]:
        return self.calls(events)[0]

    @staticmethod
    def _typed(names: list[str], t0: float = 0.01, seq0: int = 0) -> list[RawEvent]:
        events, t, seq = [], t0, seq0
        for name in names:
            events.append(_key(seq, t, "press", name))
            events.append(_key(seq + 1, t + 0.005, "release", name))
            t, seq = t + 0.01, seq + 2
        return events

    @staticmethod
    def _pairs(*names: str, t0: float = 0.01, seq0: int = 0) -> list[RawEvent]:
        """Nested press-in-order/release-in-reverse scope over the names."""
        events, seq = [], seq0
        for k, name in enumerate(names):
            events.append(_key(seq, t0 + 0.01 * k, "press", name))
            seq += 1
        for k, name in enumerate(reversed(names)):
            events.append(_key(seq, t0 + 0.01 * (len(names) + k), "release", name))
            seq += 1
        return events

    # --- motion / scroll ----------------------------------------------------

    def test_moves_accumulate_across_motor_ticks(self) -> None:
        # 0.01 / 0.30 / 0.55 s are three different 10 Hz motor ticks: v2/v3
        # would split; the barrier-level formatter must not.
        self.assertEqual(
            self.one_window([
                _move(0, 0.01, 1.0, 0.0),
                _move(1, 0.30, 3.0, -1.0),
                _move(2, 0.55, 2.4, 0.4),
            ]),
            [{"action": "mouse_move_rel", "delta": [6, -1]}],
        )

    def test_press_is_a_barrier_and_drag_decomposes(self) -> None:
        self.assertEqual(
            self.one_window([
                _move(0, 0.01, 3.0, 0.0),
                _key(1, 0.50, "press", "LMB"),
                _move(2, 0.60, 2.0, 0.0),
                _key(3, 0.90, "release", "LMB"),
            ]),
            [
                {"action": "mouse_move_rel", "delta": [3, 0]},
                {"action": "button_down", "button": "left"},
                {"action": "mouse_move_rel", "delta": [2, 0]},
                {"action": "button_up", "button": "left"},
            ],
        )

    def test_scroll_accumulates_y_before_x(self) -> None:
        self.assertEqual(
            self.one_window([_scroll(0, 0.01, 2.0, -3.0), _scroll(1, 0.50, 1.0, -2.0)]),
            [{"action": "scroll", "pixels": -5}, {"action": "hscroll", "pixels": 3}],
        )

    def test_horizontal_only_scroll(self) -> None:
        self.assertEqual(
            self.one_window([_scroll(0, 0.01, 3.0, 0.0)]),
            [{"action": "hscroll", "pixels": 3}],
        )

    def test_move_and_scroll_accumulate_independently(self) -> None:
        self.assertEqual(
            self.one_window([
                _move(0, 0.01, 1.0, 0.0),
                _scroll(1, 0.02, 0.0, -2.0),
                _move(2, 0.03, 2.0, 0.0),
                _scroll(3, 0.04, 0.0, -3.0),
            ]),
            [
                {"action": "mouse_move_rel", "delta": [3, 0]},
                {"action": "scroll", "pixels": -5},
            ],
        )

    def test_zero_rounded_motion_is_omitted(self) -> None:
        self.assertEqual(
            self.one_window([
                _move(0, 0.01, 0.2, 0.2),
                _key(1, 0.03, "press", "Return"),
                _key(2, 0.04, "release", "Return"),
            ]),
            [{"action": "key", "keys": ["enter"]}],
        )

    # --- clicks ---------------------------------------------------------------

    def test_click_survives_subpixel_motion_between(self) -> None:
        self.assertEqual(
            self.one_window([
                _key(0, 0.10, "press", "LMB"),
                _move(1, 0.15, 0.2, 0.2),
                _key(2, 0.20, "release", "LMB"),
            ]),
            [{"action": "left_click"}],
        )

    def test_double_and_triple_click(self) -> None:
        pair = [_key(0, 0.10, "press", "LMB"), _key(1, 0.12, "release", "LMB")]
        two = [*pair, _key(2, 0.20, "press", "LMB"), _key(3, 0.22, "release", "LMB")]
        three = [*two, _key(4, 0.30, "press", "LMB"), _key(5, 0.32, "release", "LMB")]
        four = [*three, _key(6, 0.40, "press", "LMB"), _key(7, 0.42, "release", "LMB")]
        self.assertEqual(self.one_window(two), [{"action": "double_click"}])
        self.assertEqual(self.one_window(three), [{"action": "triple_click"}])
        self.assertEqual(self.one_window(four),
                         [{"action": "triple_click"}, {"action": "left_click"}])

    def test_right_and_middle_clicks_never_multi_collapse(self) -> None:
        self.assertEqual(
            self.one_window([
                _key(0, 0.10, "press", "RMB"), _key(1, 0.12, "release", "RMB"),
                _key(2, 0.20, "press", "RMB"), _key(3, 0.22, "release", "RMB"),
            ]),
            [{"action": "right_click"}, {"action": "right_click"}],
        )
        self.assertEqual(
            self.one_window([_key(0, 0.1, "press", "MMB"), _key(1, 0.2, "release", "MMB")]),
            [{"action": "middle_click"}],
        )

    # --- typing -----------------------------------------------------------------

    def test_typing_collapses_with_json_escaping(self) -> None:
        events = [
            _key(0, 0.01, "press", "ShiftLeft"),
            _key(1, 0.02, "press", "Quote"),  # shifted Quote -> "
            _key(2, 0.03, "release", "Quote"),
            _key(3, 0.04, "release", "ShiftLeft"),
            *self._typed(["BackSlash"], t0=0.05, seq0=4),  # -> \
        ]
        calls = self.one_window(events)
        self.assertEqual(calls, [{"action": "type", "text": '"\\'}])
        # json-level escaping is in the raw label (no manual escaping)
        self.assertIn(r'"text": "\"\\"', self.result.labels[0])

    def test_render_tool_call_keeps_unicode_raw(self) -> None:
        block = render_tool_call({"action": "type", "text": 'café — "τ\\"'})
        self.assertIn("café — ", block)  # ensure_ascii=False
        body = _TOOL_CALL_RE.fullmatch(block).group(1)
        self.assertEqual(json.loads(body)["arguments"]["text"], 'café — "τ\\"')

    def test_enter_between_typing_runs(self) -> None:
        events = [
            *self._typed(["KeyH"], t0=0.01, seq0=0),
            *self._typed(["Return"], t0=0.03, seq0=2),
            *self._typed(["KeyI"], t0=0.05, seq0=4),
            *self._typed(["Backspace"], t0=0.07, seq0=6),
        ]
        self.assertEqual(
            self.one_window(events),
            [
                {"action": "type", "text": "h"},
                {"action": "key", "keys": ["enter"]},
                {"action": "type", "text": "i"},
                {"action": "key", "keys": ["backspace"]},
            ],
        )

    # --- keys / chords ------------------------------------------------------

    def test_simple_chord(self) -> None:
        self.assertEqual(
            self.one_window(self._pairs("ControlLeft", "KeyC")),
            [{"action": "key", "keys": ["ctrl", "c"]}],
        )

    def test_nested_modifier_chord(self) -> None:
        self.assertEqual(
            self.one_window(self._pairs("ControlLeft", "ShiftLeft", "KeyS")),
            [{"action": "key", "keys": ["ctrl", "shift", "s"]}],
        )

    def test_meta_maps_to_command(self) -> None:
        self.assertEqual(
            self.one_window(self._pairs("MetaLeft", "KeyD")),
            [{"action": "key", "keys": ["command", "d"]}],
        )

    def test_bare_modifier_tap(self) -> None:
        self.assertEqual(
            self.one_window(self._pairs("ShiftLeft")),
            [{"action": "key", "keys": ["shift"]}],
        )

    def test_multi_pair_chord_decomposes_exactly(self) -> None:
        events = [
            _key(0, 0.01, "press", "ControlLeft"),
            *self._typed(["KeyC"], t0=0.02, seq0=1),
            *self._typed(["KeyV"], t0=0.05, seq0=3),
            _key(5, 0.08, "release", "ControlLeft"),
        ]
        self.assertEqual(
            self.one_window(events),
            [
                {"action": "key_down", "key": "ctrl"},
                {"action": "key", "keys": ["c"]},
                {"action": "key", "keys": ["v"]},
                {"action": "key_up", "key": "ctrl"},
            ],
        )

    def test_modifier_scope_over_a_click_decomposes(self) -> None:
        events = [
            _key(0, 0.01, "press", "ControlLeft"),
            _key(1, 0.02, "press", "LMB"),
            _key(2, 0.03, "release", "LMB"),
            _key(3, 0.04, "release", "ControlLeft"),
        ]
        self.assertEqual(
            self.one_window(events),
            [
                {"action": "key_down", "key": "ctrl"},
                {"action": "left_click"},
                {"action": "key_up", "key": "ctrl"},
            ],
        )

    def test_held_across_window_button_and_key(self) -> None:
        windows = [
            Window(master_idx=0, start=0, end=15),
            Window(master_idx=15, start=15, end=30),
        ]
        self.assertEqual(
            self.calls(
                [_key(0, 0.5, "press", "LMB"), _key(1, 1.5, "release", "LMB")],
                windows,
            ),
            [[{"action": "button_down", "button": "left"}],
             [{"action": "button_up", "button": "left"}]],
        )
        self.assertEqual(
            self.calls(
                [_key(0, 0.5, "press", "ControlLeft"),
                 _key(1, 1.5, "release", "ControlLeft")],
                windows,
            ),
            [[{"action": "key_down", "key": "ctrl"}],
             [{"action": "key_up", "key": "ctrl"}]],
        )
        # a printable key held across the boundary is not a typing pair
        self.assertEqual(
            self.calls(
                [_key(0, 0.5, "press", "KeyH"), _key(1, 1.5, "release", "KeyH")],
                windows,
            ),
            [[{"action": "key_down", "key": "h"}],
             [{"action": "key_up", "key": "h"}]],
        )

    def test_unmapped_key_lowercases_and_counts(self) -> None:
        self.assertEqual(
            self.one_window(self._pairs("KC_999")),
            [{"action": "key", "keys": ["kc_999"]}],
        )
        self.assertEqual(self.result.primitive_counts["unmapped_key_names"], 1)

    # --- empty windows / provenance / registry ------------------------------

    def test_empty_window_is_wait_for_the_window_span(self) -> None:
        windows = [
            Window(master_idx=0, start=0, end=15),
            Window(master_idx=15, start=15, end=45),
        ]
        calls = self.calls([_move(0, 0.1, 5.0, 0.0)], windows)
        self.assertEqual(calls[0], [{"action": "mouse_move_rel", "delta": [5, 0]}])
        self.assertEqual(calls[1], [{"action": "wait", "time": 2.0}])  # 30 ticks @ 15fps
        self.assertEqual(
            self.calls([], [Window(master_idx=0, start=0, end=15)]),
            [[{"action": "wait", "time": 1.0}]],
        )

    def test_primitive_counts_cover_every_emitted_action(self) -> None:
        self.calls([
            _move(0, 0.01, 5.0, 0.0),
            _key(1, 0.10, "press", "LMB"),
            _key(2, 0.12, "release", "LMB"),
            *self._typed(["KeyH"], t0=0.20, seq0=3),
            *self._pairs("ControlLeft", "KeyC", t0=0.30, seq0=5),
            _scroll(9, 0.50, 0.0, -3.0),
        ])
        counts = self.result.primitive_counts
        self.assertEqual(counts["mouse_move_rel"], 1)
        self.assertEqual(counts["left_click"], 1)
        self.assertEqual(counts["type"], 1)
        self.assertEqual(counts["key"], 1)
        self.assertEqual(counts["scroll"], 1)
        self.assertEqual(counts["wait"], 0)
        self.assertEqual(counts["unmapped_key_names"], 0)

    def test_key_table_matches_eval_conventions(self) -> None:
        t = RDEV_TO_COMPUTER_USE_KEY
        self.assertEqual(t["KeyA"], "a")
        self.assertEqual((t["Num1"], t["Digit1"]), ("1", "1"))
        self.assertEqual(t["Return"], "enter")
        self.assertEqual(t["Escape"], "esc")
        self.assertEqual((t["ControlLeft"], t["ControlRight"]), ("ctrl", "ctrl"))
        self.assertEqual((t["ShiftLeft"], t["ShiftRight"]), ("shift", "shift"))
        self.assertEqual((t["MetaLeft"], t["MetaRight"]), ("command", "command"))
        self.assertEqual((t["UpArrow"], t["ArrowUp"]), ("up", "up"))
        # rdev spellings and eval's alternate casings stay in sync
        self.assertEqual((t["SemiColon"], t["Semicolon"]), (";", ";"))
        self.assertEqual((t["BackSlash"], t["Backslash"]), ("\\", "\\"))
        self.assertEqual((t["Dot"], t["Period"]), (".", "."))

    def test_terminate_line_all_formatters(self) -> None:
        for name in ("canonical", "ordered_events_v2", "ordered_events_v3"):
            self.assertEqual(get_formatter(name).terminate_line(), "TERMINATE")
        line = get_formatter("computer_use_rel_v1").terminate_line()
        (args,) = self.validate(line)
        self.assertEqual(args, {"action": "terminate", "status": "success"})

    def test_is_idle_label_all_formatters(self) -> None:
        for name in ("canonical", "ordered_events_v2", "ordered_events_v3"):
            fmt = get_formatter(name)
            self.assertTrue(fmt.is_idle_label("NO_OP"))
            self.assertFalse(fmt.is_idle_label("move(1,2)"))
            self.assertFalse(fmt.is_idle_label("0 0 0"))
        cu = get_formatter("computer_use_rel_v1")
        wait = ('<tool_call>\n{"name": "computer_use", "arguments": '
                '{"action": "wait", "time": 2.0}}\n</tool_call>')
        click = ('<tool_call>\n{"name": "computer_use", "arguments": '
                 '{"action": "left_click"}}\n</tool_call>')
        self.assertTrue(cu.is_idle_label(wait))
        self.assertFalse(cu.is_idle_label(click))
        self.assertFalse(cu.is_idle_label(wait + "\n" + click))
        self.assertFalse(cu.is_idle_label(cu.terminate_line()))
        self.assertFalse(cu.is_idle_label("NO_OP"))
        self.assertFalse(cu.is_idle_label("<tool_call>\nnot json\n</tool_call>"))

    def test_registry_and_default_prompt(self) -> None:
        self.assertEqual(get_formatter("computer_use_rel_v1").name,
                         "computer_use_rel_v1")
        prompt = default_system_prompt(
            get_formatter("computer_use_rel_v1"), goal_conditioned=True
        )
        self.assertIn("<tool_call>", prompt)
        self.assertIn('"computer_use"', prompt)


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
