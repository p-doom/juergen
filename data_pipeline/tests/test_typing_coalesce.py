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

from realigned_pipeline.lib.action_format import (
    ActionPrimitive,
    get_formatter,
    is_typing_only_window,
    plan_typing_coalesce,
)
from realigned_pipeline.stage_04_build_conversations import (
    _dead_zone_breaks,
    _goal_coalesce_bounds,
    reformat_segment_actions,
)
from realigned_pipeline.lib.events import DeadZone

MASTER_FPS = 10.0


# --- primitive builders ----------------------------------------------------

def typ(text: str) -> ActionPrimitive:
    return ActionPrimitive(kind="type", text=text)


def down(name: str) -> ActionPrimitive:
    return ActionPrimitive(kind="down", input_name=name)


def up(name: str) -> ActionPrimitive:
    return ActionPrimitive(kind="up", input_name=name)


def move(dx: int, dy: int) -> ActionPrimitive:
    return ActionPrimitive(kind="move", dx=dx, dy=dy)


IDLE: list[ActionPrimitive] = []


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
            [ActionPrimitive(kind="scroll", dx=0, dy=-2)],
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

    def test_dead_zone_breaks_a_run(self):
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
    def test_goal_bounds_from_source_spans(self):
        frames = [{"source_frame_idx": s} for s in (0, 5, 10, 15, 20, 25)]
        barriers, terminals = _goal_coalesce_bounds(frames, [(0, 12), (13, 30)])
        self.assertEqual(barriers, {0, 3})
        self.assertEqual(terminals, {2, 5})

    def test_goal_span_with_no_frames_is_ignored(self):
        frames = [{"source_frame_idx": s} for s in (0, 5)]
        barriers, terminals = _goal_coalesce_bounds(frames, [(100, 200)])
        self.assertEqual((barriers, terminals), (set(), set()))

    def test_dead_zone_breaks_between_frames(self):
        ticks = [0, 10, 20, 30, 40]
        zones = [DeadZone(12, 18, "black"), DeadZone(35, 36, "black")]
        # a zone inside [10,20) breaks index 2; one inside [30,40) breaks index 4
        self.assertEqual(_dead_zone_breaks(ticks, zones), {2, 4})

    def test_no_zones_no_breaks(self):
        self.assertEqual(_dead_zone_breaks([0, 5, 10], []), set())


# --- end-to-end over a synthetic keylog ------------------------------------

def _key(t_s: float, name: str, press: bool) -> tuple[float, list]:
    # Keylog shape: [ts_us, ["KeyPress", [raw_code, name]]] (common.resolve_key_name
    # reads payload[1]).
    return (t_s, ["KeyPress" if press else "KeyRelease", [0, name]])


def _tap(t_s: float, name: str, dur: float = 0.02) -> list[tuple[float, list]]:
    return [_key(t_s, name, True), _key(t_s + dur, name, False)]


class EndToEndTest(unittest.TestCase):
    """Drive ``reformat_segment_actions`` over a real keylog + master manifest."""

    def _run(self, entries, ticks, n_records, **kw):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            keylog = root / "keylog.msgpack"
            keylog.write_bytes(msgpack.packb(
                [[round(t * 1_000_000), ev] for t, ev in sorted(entries, key=lambda e: e[0])]
            ))
            manifest = root / "frame_manifest.jsonl"
            manifest.write_text("".join(
                '{"mean_luma": 128.0, "frac_dark": 0.0}\n' for _ in range(n_records)
            ))
            frames = [
                {"image_path": f"ar://{root}/images.array_record#{i}",
                 "master_record_index": t, "source_frame_idx": t, "global_frame_idx": i}
                for i, t in enumerate(ticks)
            ]
            info = reformat_segment_actions(
                frames,
                {"segment_id": "seg", "keylog_path": str(keylog), "master_fps": MASTER_FPS},
                formatter=get_formatter("ordered_events_v3"),
                sample_cfg={"master_fps": MASTER_FPS, "frames_master_dir": None},
                dead_zone_flag_frac=1.0,
                **kw,
            )
            return frames, info

    def test_merged_run_yields_one_type_and_drops_frames(self):
        # One character typed per 10-tick frame: "h" "e" "l" "l" "o".
        entries = [e for i, c in enumerate("HELLO") for e in _tap(i * 1.0 + 0.1, f"Key{c}")]
        ticks = [0, 10, 20, 30, 40]
        frames, info = self._run(entries, ticks, n_records=50,
                                 coalesce_typing=True, max_coalesce_frames=0)
        # Frames 0..3 fuse; frame 4 is the segment's terminal -> NO_OP tail.
        self.assertEqual([f["action"] for f in frames], ['type("hello")', "NO_OP"])
        self.assertEqual([f["global_frame_idx"] for f in frames], [0, 4])
        self.assertEqual(frames[0]["coalesced_n_frames"], 5)
        self.assertTrue(frames[1]["coalesce_forced_idle"])
        self.assertEqual(info["coalesced_turns"], 1)
        self.assertEqual(info["coalesced_frames_dropped"], 3)

    def test_without_the_flag_nothing_changes(self):
        entries = [e for i, c in enumerate("HELLO") for e in _tap(i * 1.0 + 0.1, f"Key{c}")]
        ticks = [0, 10, 20, 30, 40]
        frames, info = self._run(entries, ticks, n_records=50)
        self.assertEqual([f["action"] for f in frames],
                         ['type("h")', 'type("e")', 'type("l")', 'type("l")', 'type("o")'])
        self.assertNotIn("coalesced_turns", info)

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
        frames, _ = self._run(entries, ticks, n_records=30)
        # Neither window balances, so v3 renders every transition explicitly --
        # 8 tokens/char instead of a type() run.
        self.assertEqual(frames[0]["action"], "down(KeyA); down(KeyB)")
        self.assertEqual(frames[1]["action"], "up(KeyA); up(KeyB)")

        frames, info = self._run(entries, ticks, n_records=30,
                                 coalesce_typing=True, max_coalesce_frames=0)
        self.assertEqual([f["action"] for f in frames], ['type("ab")', "NO_OP"])
        self.assertEqual(info["coalesced_frames_dropped"], 1)

    def test_return_breaks_the_run(self):
        entries = (
            _tap(0.1, "KeyA") + _tap(1.1, "KeyB")
            + _tap(2.1, "Return")           # submit -- its own turn
            + _tap(3.1, "KeyC") + _tap(4.1, "KeyD")
        )
        ticks = [0, 10, 20, 30, 40]
        frames, info = self._run(entries, ticks, n_records=50,
                                 coalesce_typing=True, max_coalesce_frames=0)
        self.assertEqual(
            [f["action"] for f in frames],
            ['type("ab")', "down(Return); up(Return)", 'type("cd")', "NO_OP"],
        )
        self.assertEqual(info["coalesced_turns"], 2)

    def test_mouse_move_breaks_the_run(self):
        entries = (
            _tap(0.1, "KeyA")
            + [(1.1, ["MouseMove", [4, -2]])]
            + _tap(2.1, "KeyB") + _tap(3.1, "KeyC")
        )
        ticks = [0, 10, 20, 30]
        frames, _ = self._run(entries, ticks, n_records=40,
                              coalesce_typing=True, max_coalesce_frames=0)
        self.assertEqual(
            [f["action"] for f in frames],
            ['type("a")', "move(4,-2)", 'type("bc")', "NO_OP"],
        )

    def test_max_coalesce_frames_bounds_the_span(self):
        entries = [e for i, c in enumerate("ABCDEF") for e in _tap(i * 1.0 + 0.1, f"Key{c}")]
        ticks = [0, 10, 20, 30, 40, 50]
        frames, _ = self._run(entries, ticks, n_records=60,
                              coalesce_typing=True, max_coalesce_frames=2)
        # Chunks [0,1] and [2,3]; the last chunk [4,5] reaches the segment's
        # terminal frame, so frame 5's span folds into frame 4 and frame 5 is
        # retained as the NO_OP tail.
        self.assertEqual([f["action"] for f in frames],
                         ['type("ab")', 'type("cd")', 'type("ef")', "NO_OP"])
        self.assertEqual([f["global_frame_idx"] for f in frames], [0, 2, 4, 5])


if __name__ == "__main__":
    unittest.main()
