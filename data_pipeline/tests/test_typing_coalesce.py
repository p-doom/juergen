"""Cross-frame typing coalescing (stage 04 ``--coalesce-typing``).

Two layers are covered:
  * ``plan_typing_coalesce`` -- which windows fuse, which break a run, and where
    a forced-idle tail lands (pure index logic over rendered primitives).
  * ``reformat_segment_actions`` end-to-end on a synthetic keylog -- the two-pass
    re-derivation really produces ONE ``type()`` per merged turn, really drops the
    frames, and repairs typing that no single window could balance.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import msgpack

from pipeline.crowdcast.lib.action_format import (
    get_formatter,
    is_typing_only_window,
    plan_typing_coalesce,
)
from grammars.ordered_events_v3.codec import Primitive
from pipeline.crowdcast.stage_04_build_conversations import (
    coalesce_typing_windows,
    dead_zone_breaks,
    goal_coalesce_bounds,
)
from pipeline.crowdcast.lib.events import DeadZone, Window

MASTER_FPS = 10.0


# --- primitive builders ----------------------------------------------------

def typ(text: str) -> Primitive:
    return Primitive("type", text=text)


def down(name: str) -> Primitive:
    return Primitive("down", name=name)


def up(name: str) -> Primitive:
    return Primitive("up", name=name)


def move(dx: int, dy: int) -> Primitive:
    return Primitive("move", dx=dx, dy=dy)


IDLE: list[Primitive] = []


class ClassificationTest(unittest.TestCase):
    def test_type_and_shift_and_bare_printable_are_typing(self):
        # type() runs, absorbed-Shift artifacts (#2/#4) and rollover leftovers
        # (#3) all count as typing.
        self.assertTrue(is_typing_only_window([typ("abc")]))
        self.assertTrue(is_typing_only_window([down("ShiftLeft")]))
        self.assertTrue(is_typing_only_window([typ("a"), down("ShiftLeft"), up("ShiftLeft")]))
        self.assertTrue(is_typing_only_window([typ("ab"), down("KeyC")]))
        self.assertTrue(is_typing_only_window([up("KeyC"), typ("de")]))

    def test_breakers(self):
        for prims in (
            [],                                  # idle is not typing
            [move(3, 1)],
            [Primitive(kind="scroll", dx=0, dy=-2)],
            [typ("hi"), down("Return"), up("Return")],       # submit (#1)
            [typ("hel"), down("Backspace"), up("Backspace")],  # backtrack (#1)
            [down("Tab"), up("Tab")],
            [down("LMB"), up("LMB")],
            [down("ControlLeft"), down("KeyC"), up("KeyC"), up("ControlLeft")],
            [typ("a"), down("F5"), up("F5")],
        ):
            self.assertFalse(is_typing_only_window(prims), prims)

    def test_held_non_shift_modifier_vetoes(self):
        # A Ctrl+C whose halves fall in different windows must not read as typing.
        self.assertFalse(is_typing_only_window([down("KeyC"), up("KeyC")], {"ControlLeft"}))
        self.assertTrue(is_typing_only_window([down("KeyC"), up("KeyC")], {"ShiftLeft"}))

    def test_held_modifier_state_carries_across_windows(self):
        prims = [
            [down("ControlLeft")],                 # breaker, Ctrl now held
            [down("KeyC"), up("KeyC")],            # chord half -- vetoed
            [down("KeyV"), up("KeyV")],            # still vetoed
            [up("ControlLeft")],                   # breaker, Ctrl released
            [typ("ab")], [typ("cd")],              # typing again -> mergeable
        ]
        plan = plan_typing_coalesce(prims, terminal={5})
        self.assertEqual(plan.spans, {4: 5})
        self.assertEqual(plan.forced_idle, [5])


class PlanTest(unittest.TestCase):
    def test_user_example_trace(self):
        """The trace from the spec:

            1 type | 2 type | 3 type+mouse | 4 type | 5 type | 6 idle | 7 type | 8 type
        ->  1 type (1-2) | 3 unchanged | 4 type (4-8, idle absorbed, 8 -> NO_OP tail)
        """
        prims = [
            [typ("a")],
            [typ("b")],
            [typ("c"), move(5, 0), typ("d"), down("LMB"), up("LMB"), down("Return"), up("Return")],
            [typ("e")],
            [typ("f")],
            IDLE,
            [typ("g")],
            [typ("h")],
        ]
        plan = plan_typing_coalesce(prims, terminal={7})
        self.assertEqual(plan.keep, [0, 2, 3, 7])
        self.assertEqual(plan.spans, {0: 1, 3: 7})
        self.assertEqual(plan.forced_idle, [7])   # frame 8 kept as the NO_OP tail
        self.assertEqual(plan.n_dropped, 4)       # frames 2, 5, 6, 7 (1-based)

    def test_trailing_idle_not_absorbed(self):
        prims = [[typ("a")], [typ("b")], IDLE, IDLE, [move(1, 1)]]
        plan = plan_typing_coalesce(prims, terminal={4})
        self.assertEqual(plan.spans, {0: 1})
        self.assertEqual(plan.keep, [0, 2, 3, 4])
        self.assertEqual(plan.forced_idle, [])

    def test_run_of_one_is_untouched(self):
        prims = [[move(1, 0)], [typ("a")], [move(2, 0)]]
        plan = plan_typing_coalesce(prims, terminal={2})
        self.assertEqual(plan.spans, {})
        self.assertEqual(plan.keep, [0, 1, 2])

    def test_max_coalesce_frames_splits_into_chunks(self):
        prims = [[typ(c)] for c in "abcdefg"]
        plan = plan_typing_coalesce(prims, terminal={6}, max_frames=3)
        # [0..2], [3..5], then 6 is terminal -> its own turn.
        self.assertEqual(plan.spans, {0: 2, 3: 5})
        self.assertEqual(plan.keep, [0, 3, 6])
        self.assertEqual(plan.forced_idle, [])

    def test_cap_counts_absorbed_idle_frames(self):
        prims = [[typ("a")], IDLE, IDLE, [typ("b")], [typ("c")]]
        plan = plan_typing_coalesce(prims, terminal={4}, max_frames=3)
        # The cap is reached at index 2, and the run ends at its last TYPING
        # frame -- which is index 0, so nothing merges there.
        self.assertEqual(plan.spans, {3: 4})
        self.assertEqual(plan.keep, [0, 1, 2, 3, 4])
        self.assertEqual(plan.forced_idle, [4])

    def testdead_zone_breaks_a_run(self):
        prims = [[typ("a")], [typ("b")], [typ("c")]]
        plan = plan_typing_coalesce(prims, terminal={2}, break_before={2})
        self.assertEqual(plan.spans, {0: 1})
        self.assertEqual(plan.keep, [0, 2])

    def test_goal_barrier_and_terminal(self):
        prims = [[typ(c)] for c in "abcdef"]
        # goal A = windows 0..2, goal B = windows 3..5
        plan = plan_typing_coalesce(prims, barrier_start={0, 3}, terminal={2, 5})
        self.assertEqual(plan.spans, {0: 2, 3: 5})
        self.assertEqual(plan.forced_idle, [2, 5])
        self.assertEqual(plan.keep, [0, 2, 3, 5])

    def test_run_cannot_start_at_a_terminal_and_extend_past_it(self):
        # Overlapping goals: window 1 ends goal A but is interior to goal B.
        prims = [[move(1, 0)], [typ("a")], [typ("b")], [typ("c")]]
        plan = plan_typing_coalesce(prims, terminal={1, 3})
        self.assertEqual(plan.spans, {2: 3})     # never {1: 3}
        self.assertEqual(plan.keep, [0, 1, 2, 3])
        self.assertEqual(plan.forced_idle, [3])

    def test_no_typing_at_all(self):
        prims = [[move(1, 0)], IDLE, [down("LMB"), up("LMB")]]
        plan = plan_typing_coalesce(prims, terminal={2})
        self.assertEqual(plan.spans, {})
        self.assertEqual(plan.keep, [0, 1, 2])
        self.assertEqual(plan.n_dropped, 0)


class BoundsHelperTest(unittest.TestCase):
    def test_goal_bounds_from_master_intervals(self):
        """A goal is a half-open master interval and a view frame knows its own
        tick, so the two join with no second coordinate system in between."""
        ticks = [0, 5, 10, 15, 20, 25]
        barriers, terminals = goal_coalesce_bounds(ticks, [(0, 13), (13, 30)])
        self.assertEqual(barriers, {0, 3})
        self.assertEqual(terminals, {2, 5})

    def test_the_interval_is_half_open(self):
        """A frame exactly at end_master_idx belongs to the NEXT goal, which is
        what keeps two adjacent goals from both claiming it as a boundary."""
        ticks = [0, 10, 20]
        barriers, terminals = goal_coalesce_bounds(ticks, [(0, 10)])
        self.assertEqual((barriers, terminals), ({0}, {0}))

    def test_goal_span_with_no_frames_is_ignored(self):
        barriers, terminals = goal_coalesce_bounds([0, 5], [(100, 200)])
        self.assertEqual((barriers, terminals), (set(), set()))

    def test_overlapping_spans_only_ever_reduce_merging(self):
        ticks = [0, 5, 10, 15]
        barriers, terminals = goal_coalesce_bounds(ticks, [(0, 12), (5, 20)])
        self.assertEqual(barriers, {0, 1})
        self.assertEqual(terminals, {2, 3})

    def testdead_zone_breaks_between_frames(self):
        ticks = [0, 10, 20, 30, 40]
        zones = [DeadZone(12, 18, "black"), DeadZone(35, 36, "black")]
        # a zone inside [10,20) breaks index 2; one inside [30,40) breaks index 4
        self.assertEqual(dead_zone_breaks(ticks, zones), {2, 4})

    def test_no_zones_no_breaks(self):
        self.assertEqual(dead_zone_breaks([0, 5, 10], []), set())


# --- end-to-end over a synthetic keylog ------------------------------------

def _key(t_s: float, name: str, press: bool) -> tuple[float, list]:
    # Keylog shape: [ts_us, ["KeyPress", [raw_code, name]]] (common.resolve_key_name
    # reads payload[1]).
    return (t_s, ["KeyPress" if press else "KeyRelease", [0, name]])


def _tap(t_s: float, name: str, dur: float = 0.02) -> list[tuple[float, list]]:
    return [_key(t_s, name, True), _key(t_s + dur, name, False)]


class _StubFrame:
    """The four attributes `coalesce_typing_windows` reads off a ViewFrame."""

    def __init__(self, view_idx: int, tick: int, win_end: int) -> None:
        self.view_idx = view_idx
        self.master_idx = tick
        self.win_start = tick
        self.win_end = win_end
        self.image = f"ar://images.array_record#{view_idx}"


class _StubView:
    def __init__(self, ticks, dead_zones=(), master_fps=MASTER_FPS) -> None:
        self.frames = [
            _StubFrame(i, t, ticks[i + 1] if i + 1 < len(ticks) else t + 10)
            for i, t in enumerate(ticks)
        ]
        self.dead_zones = list(dead_zones)
        self.master_fps = master_fps


class EndToEndTest(unittest.TestCase):
    """Drive `coalesce_typing_windows` over a real keylog and a stage-03 view.

    The formatter runs for real — these assert on what the codec actually
    renders over merged windows, not on a re-joined string.
    """

    def _run(self, entries, ticks, *, coalesce=False, max_frames=0, goal_spans=()):
        from pipeline.crowdcast.lib.events import load_events

        with tempfile.TemporaryDirectory() as td:
            keylog = Path(td) / "keylog.msgpack"
            keylog.write_bytes(msgpack.packb(
                [[round(t * 1_000_000), ev] for t, ev in sorted(entries, key=lambda e: e[0])]
            ))
            events, _ = load_events(keylog)
        view = _StubView(ticks)
        result = get_formatter("ordered_events_v3").format_segment(
            events, view.windows() if hasattr(view, "windows") else [
                Window(f.master_idx, f.win_start, f.win_end) for f in view.frames
            ], [], master_fps=MASTER_FPS,
        )
        if not coalesce:
            return list(range(len(ticks))), list(result.labels), 0
        return coalesce_typing_windows(
            view, result, get_formatter("ordered_events_v3"), events,
            goal_spans=list(goal_spans), max_frames=max_frames,
        )

    def test_merged_run_yields_one_type_and_drops_frames(self):
        """One character typed per frame is what the recorder gives us; trained
        as-is it teaches the model to emit a character and wait for a
        screenshot."""
        entries = [e for i, c in enumerate("HELLO") for e in _tap(i * 1.0 + 0.1, f"Key{c}")]
        keep, labels, dropped = self._run(
            entries, [0, 10, 20, 30, 40], coalesce=True)
        # 0..3 fuse; frame 4 is the segment's terminal, kept as the NO_OP tail so
        # the conversation still ends on a fresh screenshot.
        self.assertEqual(labels, ['type("hello")', "NO_OP"])
        self.assertEqual(keep, [0, 4])
        self.assertEqual(dropped, 3)

    def test_without_the_flag_nothing_changes(self):
        entries = [e for i, c in enumerate("HELLO") for e in _tap(i * 1.0 + 0.1, f"Key{c}")]
        _, labels, dropped = self._run(entries, [0, 10, 20, 30, 40])
        self.assertEqual(
            labels, ['type("h")', 'type("e")', 'type("l")', 'type("l")', 'type("o")'])
        self.assertEqual(dropped, 0)

    def test_rollover_across_a_frame_boundary_is_repaired(self):
        # "ab" typed with rollover: KeyB goes down before KeyA comes up, and the
        # frame boundary at tick 10 falls between them. Per-frame v3 cannot
        # balance either window; the merged window can.
        entries = [
            _key(0.10, "KeyA", True),
            _key(0.95, "KeyB", True),
            _key(1.05, "KeyA", False),
            _key(1.20, "KeyB", False),
        ]
        ticks = [0, 10, 20]
        _, labels, _ = self._run(entries, ticks)
        # Neither window balances, so v3 renders every transition explicitly --
        # ~8 tokens per character instead of one type() run.
        self.assertEqual(labels[0], "down(KeyA); down(KeyB)")
        self.assertEqual(labels[1], "up(KeyA); up(KeyB)")

        _, labels, dropped = self._run(entries, ticks, coalesce=True)
        self.assertEqual(labels, ['type("ab")', "NO_OP"])
        self.assertEqual(dropped, 1)

    def test_return_breaks_the_run(self):
        entries = (
            _tap(0.1, "KeyA") + _tap(1.1, "KeyB")
            + _tap(2.1, "Return")           # submit -- its own turn
            + _tap(3.1, "KeyC") + _tap(4.1, "KeyD")
        )
        _, labels, _ = self._run(entries, [0, 10, 20, 30, 40], coalesce=True)
        self.assertEqual(
            labels,
            ['type("ab")', "down(Return); up(Return)", 'type("cd")', "NO_OP"],
        )

    def test_mouse_move_breaks_the_run(self):
        entries = (
            _tap(0.1, "KeyA")
            + [(1.1, ["MouseMove", [4, -2]])]
            + _tap(2.1, "KeyB") + _tap(3.1, "KeyC")
        )
        _, labels, _ = self._run(entries, [0, 10, 20, 30], coalesce=True)
        self.assertEqual(
            labels, ['type("a")', "move(4,-2)", 'type("bc")', "NO_OP"])

    def test_max_coalesce_frames_bounds_the_span(self):
        entries = [e for i, c in enumerate("ABCDEF") for e in _tap(i * 1.0 + 0.1, f"Key{c}")]
        keep, labels, _ = self._run(
            entries, [0, 10, 20, 30, 40, 50], coalesce=True, max_frames=2)
        # Chunks [0,1] and [2,3]; the last reaches the segment's terminal frame,
        # so 5's span folds into 4 and 5 is retained as the NO_OP tail. Each
        # chunk keeps its OWN screenshot -- that is what the bound buys.
        self.assertEqual(labels, ['type("ab")', 'type("cd")', 'type("ef")', "NO_OP"])
        self.assertEqual(keep, [0, 2, 4, 5])


if __name__ == "__main__":
    unittest.main()
