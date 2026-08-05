"""Per-turn action counts, verbatim rendering and action search in the
frame-records viewer.

The viewer's ``hud`` translation sums a turn's movement into one
``<dx> <dy> <scroll>`` vector — that is the radar's own semantics (one pointer
state per bin) and it stops there. Everything the eye reads is the label as
written: ``disp`` is the action line verbatim, primitive for primitive, and the
per-turn counts say how long that line really is. These tests pin all three —
the counts behind the ``n`` column and the ``actions/turn`` filters, the
no-collapsing rendering, and the substring search that finds the turns — across
every action format the viewer browses.
"""

from __future__ import annotations

import json
import unittest

from realigned_pipeline.visualize_frame_records import (
    compute_metrics,
    find_actions,
    parse_ordered_action,
    turn_action_counts,
)


def _tool_call(action: str, **args: object) -> str:
    payload = {"name": "computer_use", "arguments": {"action": action, **args}}
    return f"<tool_call>{json.dumps(payload)}</tool_call>"


def _frames(actions: list[str]) -> list[dict[str, object]]:
    return [{"action": a, "is_noop": a.strip() in ("", "NO_OP")} for a in actions]


class _FakeSegment:
    def __init__(self, actions: list[str]) -> None:
        self.frames = [
            {"action": a, "disp": (parse_ordered_action(a) or (None, None))[1]}
            for a in actions
        ]


class _FakeDataset:
    def __init__(self, segments: dict[str, list[str]]) -> None:
        self.segments = {sid: _FakeSegment(a) for sid, a in segments.items()}


class OrderedTurnCountsTest(unittest.TestCase):
    """ordered_events_v2/v3: one action per primitive."""

    def test_repeated_moves_count_individually(self):
        # the case the hud/disp collapse hides: 11 motor ticks, not one move
        turn = "; ".join(["move(-100,10)"] * 11)
        self.assertEqual(turn_action_counts(turn), (11, 11))

    def test_mouse_is_the_pointer_subset(self):
        turn = ('move(4,-1); move(6,0); down(LMB); up(LMB); '
                'down(ControlLeft); type("hi"); up(ControlLeft)')
        self.assertEqual(turn_action_counts(turn), (7, 4))

    def test_scroll_counts_as_mouse(self):
        self.assertEqual(turn_action_counts("scroll(0,-3); scroll(0,-3)"), (2, 2))

    def test_noop_and_terminate_are_not_actions(self):
        self.assertEqual(turn_action_counts("NO_OP"), (0, 0))
        self.assertEqual(turn_action_counts("TERMINATE"), (0, 0))
        # a TERMINATE suffix does not inflate the turn it is appended to
        self.assertEqual(turn_action_counts("move(1,2)\nTERMINATE"), (1, 1))

    def test_thinking_block_is_stripped(self):
        self.assertEqual(
            turn_action_counts('<think>I should click it.</think>\ndown(LMB); up(LMB)'),
            (2, 2))

    def test_v2_turn_without_type(self):
        self.assertEqual(turn_action_counts("down(KeyA); up(KeyA)"), (2, 0))


class NativeTurnCountsTest(unittest.TestCase):
    """computer_use tool calls: one action per call."""

    def test_gestures_and_the_mouse_subset(self):
        text = (_tool_call("mouse_move_rel", delta=[10, 4])
                + _tool_call("left_click")
                + _tool_call("type", text="hello")
                + _tool_call("scroll", pixels=-3))
        self.assertEqual(turn_action_counts(text), (4, 3))

    def test_key_and_wait_are_not_mouse(self):
        text = _tool_call("key", keys=["ctrl", "c"]) + _tool_call("wait", time=1)
        self.assertEqual(turn_action_counts(text), (2, 0))

    def test_malformed_call_still_counts_as_one_action(self):
        text = "<tool_call>{not json}</tool_call>" + _tool_call("left_click")
        self.assertEqual(turn_action_counts(text), (2, 1))


class PlainTurnCountsTest(unittest.TestCase):
    """format_action: a turn is one binned action, counted by what it does."""

    def test_movement_scroll_and_events(self):
        self.assertEqual(turn_action_counts("100 -20 0 ; +LMB -LMB"), (3, 3))
        self.assertEqual(turn_action_counts("0 0 -3"), (1, 1))
        self.assertEqual(turn_action_counts("0 0 0 ; +KeyA -KeyA"), (2, 0))

    def test_noop_and_empty(self):
        self.assertEqual(turn_action_counts("NO_OP"), (0, 0))
        self.assertEqual(turn_action_counts(""), (0, 0))

    def test_hud_is_used_when_the_raw_text_is_not_parseable(self):
        # a prose-prefixed conversation turn: the hud translation is authoritative
        self.assertEqual(turn_action_counts("click the icon\n5 5 0 ; +LMB -LMB",
                                            "5 5 0 ; +LMB -LMB"), (3, 3))


class MetricsReductionTest(unittest.TestCase):
    def test_frames_are_annotated_and_reduced(self):
        frames = _frames([
            "; ".join(["move(-100,10)"] * 11),      # 11 actions, 11 mouse
            "NO_OP",                                # not an acting turn
            'type("hi"); down(Return); up(Return)',  # 3 actions, 0 mouse
        ])
        m = compute_metrics(frames)
        self.assertEqual([f["n_act"] for f in frames], [11, 0, 3])
        self.assertEqual([f["n_mouse"] for f in frames], [11, 0, 0])
        self.assertEqual(m["acts_max"], 11)
        self.assertEqual(m["mouse_max"], 11)
        # means are per ACTING turn (2 of the 3), not per frame
        self.assertEqual(m["acts_mean"], 7.0)
        self.assertEqual(m["mouse_mean"], 5.5)

    def test_segment_without_actions(self):
        m = compute_metrics(_frames(["NO_OP", "NO_OP"]))
        self.assertEqual((m["acts_max"], m["acts_mean"]), (0, 0.0))
        self.assertEqual((m["mouse_max"], m["mouse_mean"]), (0, 0.0))

    def test_existing_metrics_are_unaffected(self):
        m = compute_metrics(_frames(["10 0 0 ; +LMB -LMB", "0 0 0 ; +KeyA -KeyA"]))
        self.assertEqual(m["clicks"], 1)
        self.assertEqual(m["keys"], 1)
        self.assertEqual(m["chars"], 1)


class VerbatimDispTest(unittest.TestCase):
    """Nothing is coalesced, folded or abbreviated on the way to the screen."""

    def test_every_move_survives(self):
        line = "; ".join([f"move({i},0)" for i in range(1, 12)])
        hud, disp = parse_ordered_action(line)
        self.assertEqual(disp, line)              # all eleven, in order
        self.assertEqual(hud.split(" ; ")[0], "66 0 0")   # the HUD still sums

    def test_click_pairs_stay_down_and_up(self):
        _hud, disp = parse_ordered_action("move(4,-1); down(LMB); up(LMB)")
        self.assertEqual(disp, "move(4,-1); down(LMB); up(LMB)")

    def test_typing_is_not_re_gathered_or_truncated(self):
        long_text = "x" * 120
        _hud, disp = parse_ordered_action(f'type("{long_text}"); down(Return); up(Return)')
        self.assertEqual(disp, f'type("{long_text}"); down(Return); up(Return)')
        self.assertNotIn("…", disp)

    def test_modifiers_are_not_folded_into_a_chord(self):
        line = "down(ControlLeft); down(KeyC); up(KeyC); up(ControlLeft)"
        self.assertEqual(parse_ordered_action(line)[1], line)

    def test_thinking_is_marked_but_its_block_is_lifted_out(self):
        _hud, disp = parse_ordered_action("<think>plan</think>\ndown(LMB); up(LMB)")
        self.assertEqual(disp, "💭 down(LMB); up(LMB)")

    def test_terminate_suffix_joins_the_line(self):
        self.assertEqual(parse_ordered_action("move(1,2)\nTERMINATE")[1],
                         "move(1,2); TERMINATE")

    def test_noop(self):
        self.assertEqual(parse_ordered_action("NO_OP")[1], "NO_OP")


class FindActionsTest(unittest.TestCase):
    def _ds(self):
        return _FakeDataset({
            "a": ["move(1,0); down(LMB); up(LMB)", "NO_OP", "scroll(0,-3)"],
            "b": ["move(2,0); move(3,0)", 'type("hello")'],
            "c": ["NO_OP"],
        })

    def test_empty_query_matches_nothing(self):
        out = find_actions(self._ds(), "  ")
        self.assertEqual((out["n_turns"], out["n_segments"]), (0, 0))
        self.assertEqual(out["segments"], {})

    def test_hits_are_per_segment_turn_counts(self):
        out = find_actions(self._ds(), "move(")
        self.assertEqual(out["segments"], {"a": 1, "b": 1})
        self.assertEqual((out["n_turns"], out["n_segments"]), (2, 2))

    def test_case_insensitive_and_substring(self):
        self.assertEqual(find_actions(self._ds(), "DOWN(lmb)")["segments"], {"a": 1})
        self.assertEqual(find_actions(self._ds(), 'type("hel')["segments"], {"b": 1})

    def test_a_turn_counts_once_however_often_it_matches(self):
        out = find_actions(_FakeDataset({"a": ["move(1,0); move(2,0); move(3,0)"]}), "move(")
        self.assertEqual(out["segments"], {"a": 1})

    def test_segments_without_frames_are_skipped(self):
        class Bare:
            segments = {"x": object()}
        self.assertEqual(find_actions(Bare(), "move(")["segments"], {})


if __name__ == "__main__":
    unittest.main()
