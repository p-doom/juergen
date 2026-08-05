"""The ordered_events_v3 distribution viewer's aggregation logic.

The viewer itself is an inspection tool, but the numbers it prints are read as
facts about a dataset ("89% of typing collapsed", "338 turns move→click→move"),
so the *derivation* is worth pinning. Rendering is not covered here; these are
the pure functions between an assistant turn's text and a counter:

  * ``parse_turn`` — thinking blocks, the ``TERMINATE`` suffix, ``NO_OP`` and
    genuine grammar violations each land in their own bucket.
  * ``_shape`` — the program-shape folding that groups turns.
  * ``Aggregator`` — primitive mix, ordering facts, click/drag pairing, typing
    collapse and the cross-turn held-state accounting, over hand-written turns
    whose expected numbers are obvious by inspection.

The turns below are written in the wire format on purpose: they are exactly what
a stage-04 ``conversations.jsonl`` carries, so a formatter change that alters the
rendering shows up here as a changed expectation.
"""

from __future__ import annotations

import unittest

from realigned_pipeline.visualize_ordered_action_distribution import (
    Aggregator,
    TurnFilter,
    _shape,
    parse_turn,
    search_turns,
    turn_counts,
)


class _FakeSegment:
    def __init__(self, actions: list[str]) -> None:
        self.frames = [{"action": a} for a in actions]


class _FakeDataset:
    """The shape ``_collect_segments`` reads: ``.segments`` of frames whose
    ``action`` is the verbatim assistant turn."""

    def __init__(self, segments: dict[str, list[str]]) -> None:
        self.segments = {sid: _FakeSegment(a) for sid, a in segments.items()}


def agg_of(turns: list[str], *, hz: float = 10.0, fps: float = 1.0,
           segments: int = 1) -> Aggregator:
    """Aggregate ``turns`` as ``segments`` identical segments."""
    a = Aggregator(hz=hz, fps=fps)
    for i in range(segments):
        a.add_segment(f"seg{i}", list(turns))
    return a


class ParseTurnTest(unittest.TestCase):
    def test_plain_line(self):
        t = parse_turn("move(4,-1); down(LMB); up(LMB)")
        self.assertIsNone(t.error)
        self.assertEqual([p.kind for p in t.prims], ["move", "down", "up"])
        self.assertFalse(t.noop or t.terminate)
        self.assertEqual(t.think_chars, 0)

    def test_noop(self):
        t = parse_turn("NO_OP")
        self.assertTrue(t.noop)
        self.assertEqual(t.prims, ())
        self.assertIsNone(t.error)

    def test_thinking_block_is_stripped_and_measured(self):
        t = parse_turn("<think>I should click the icon.</think>\ndown(LMB); up(LMB)")
        self.assertIsNone(t.error)
        self.assertEqual(len(t.prims), 2)
        self.assertGreater(t.think_chars, 0)
        self.assertEqual(t.body, "down(LMB); up(LMB)")

    def test_terminate_alone_is_not_an_action(self):
        t = parse_turn("TERMINATE")
        self.assertTrue(t.terminate)
        self.assertEqual(t.prims, ())
        self.assertIsNone(t.error)      # stage 04 doing its job, not a violation
        self.assertFalse(t.noop)

    def test_terminate_suffix_keeps_the_primitives(self):
        t = parse_turn("move(1,2)\nTERMINATE")
        self.assertTrue(t.terminate)
        self.assertEqual([p.kind for p in t.prims], ["move"])

    def test_typed_text_with_separator_inside(self):
        # "; " inside a type() payload must not split the line.
        t = parse_turn('type("a; b"); down(Return)')
        self.assertIsNone(t.error)
        self.assertEqual(t.prims[0].text, "a; b")
        self.assertEqual(len(t.prims), 2)

    def test_grammar_violations_are_reported(self):
        for bad in ("100 -20 0 ; +LMB -LMB",              # canonical format
                    'move(4,-1);; up(LMB)',                # empty primitive
                    "click(LMB)",                          # not a primitive
                    '<tool_call>{"name": "computer_use"}</tool_call>'):
            with self.subTest(bad=bad):
                t = parse_turn(bad)
                self.assertIsNotNone(t.error, f"{bad!r} should not parse")
                self.assertEqual(t.prims, ())

    def test_empty_turn_is_an_error(self):
        self.assertIsNotNone(parse_turn("").error)
        self.assertIsNotNone(parse_turn("<think>hmm</think>").error)


class ShapeTest(unittest.TestCase):
    def test_runs_fold(self):
        self.assertEqual(_shape(["move", "move", "move", "down", "up"]),
                         "move+; down; up")

    def test_alternation_is_preserved(self):
        self.assertEqual(_shape(["move", "scroll", "move"]), "move; scroll; move")

    def test_single_primitive(self):
        self.assertEqual(_shape(["type"]), "type")


class TurnBucketsTest(unittest.TestCase):
    """Every turn lands in exactly one of active / NO_OP / terminate-only /
    parse-error — the tiles' denominators depend on it."""

    def test_buckets_partition_the_turns(self):
        a = agg_of([
            "move(1,0)",                 # active
            "NO_OP",                     # idle
            "TERMINATE",                 # terminate-only
            "100 0 0 ; +LMB",            # violation
        ])
        self.assertEqual((a.n_turns, a.n_active, a.n_noop, a.n_error), (4, 1, 1, 1))
        self.assertEqual(a.n_terminate, 1)
        self.assertEqual(len(a.errors), 1)
        self.assertEqual(a.errors[0]["turn"], 3)
        # the terminate-only turn contributes no program shape
        self.assertEqual(list(a.shapes), ["move"])


class PrimitiveMixTest(unittest.TestCase):
    def test_counts_and_turns_containing(self):
        a = agg_of(["move(1,0); move(2,0)", 'type("hi")', "NO_OP"])
        self.assertEqual(a.kind_counts["move"], 2)
        self.assertEqual(a.kind_turns["move"], 1)      # ONE turn contained moves
        self.assertEqual(a.kind_counts["type"], 1)
        self.assertEqual(a.prims_total, 3)
        self.assertEqual(a.prims_per_turn[2], 1)

    def test_motion_is_measured_per_primitive(self):
        # 3 primitives of 3-4-5 triangles: travel is summed per primitive, not
        # per turn (a turn's net displacement would be 0 here).
        a = agg_of(["move(3,4); move(-3,-4); move(0,5)"])
        self.assertEqual(round(a.travel_px), 15)
        self.assertEqual(a.move_top["move(3,4)"], 1)

    def test_scroll_axes(self):
        a = agg_of(["scroll(0,-3); scroll(2,0); scroll(1,1)"])
        self.assertEqual(a.scroll_axes["vertical only"], 1)
        self.assertEqual(a.scroll_axes["horizontal only"], 1)
        self.assertEqual(a.scroll_axes["both axes"], 1)
        self.assertEqual((a.scroll_v_mag, a.scroll_h_mag), (4, 3))


class StructureTest(unittest.TestCase):
    """The ordering facts — what the aggregate format could not express."""

    def test_move_click_move(self):
        a = agg_of(["move(10,0); down(LMB); up(LMB); move(5,0)"])
        self.assertEqual(a.structure["move_click_move"], 1)
        self.assertEqual(a.structure["motion_then_click"], 1)
        self.assertEqual(a.structure["click_then_motion"], 1)
        self.assertEqual(a.structure["segmented_motion"], 1)
        self.assertEqual(a.structure["drag"], 0)

    def test_click_without_motion_sets_nothing(self):
        a = agg_of(["down(LMB); up(LMB)"])
        for flag in ("move_click_move", "motion_then_click", "click_then_motion",
                     "drag", "segmented_motion"):
            self.assertEqual(a.structure[flag], 0, flag)
        self.assertEqual(a.structure["keys_only"], 1)

    def test_drag_is_a_button_held_across_motion(self):
        a = agg_of(["down(LMB); move(20,0); move(5,0); up(LMB)"])
        self.assertEqual(a.structure["drag"], 1)
        self.assertEqual(a.drags["LMB"], 1)
        self.assertEqual(a.clicks["LMB"], 0)          # not an adjacent pair
        self.assertEqual(a.unpaired_down["LMB"], 1)

    def test_clicks_doubles_and_triples(self):
        a = agg_of([
            "down(LMB); up(LMB)",
            "down(LMB); up(LMB); down(LMB); up(LMB)",
            "down(LMB); up(LMB); down(LMB); up(LMB); down(LMB); up(LMB)",
        ])
        self.assertEqual(a.clicks["LMB"], 6)          # every adjacent pair
        self.assertEqual((a.dbl_clicks, a.tri_clicks), (1, 1))
        self.assertEqual(a.structure["multi_click"], 2)

    def test_motion_only_vs_keys_only(self):
        a = agg_of(["move(1,0); scroll(0,2)", 'type("x")'])
        self.assertEqual(a.structure["motion_only"], 1)
        self.assertEqual(a.structure["move_and_scroll"], 1)
        self.assertEqual(a.structure["keys_only"], 1)


class TypingTest(unittest.TestCase):
    def test_collapse_ratio_counts_both_spellings(self):
        # Four characters: two inside a type(), two spelled out as key pairs the
        # formatter could not fold (rollover across a window boundary).
        a = agg_of(['type("ab")', "down(KeyC); down(KeyD); up(KeyC); up(KeyD)"])
        self.assertEqual(a.type_chars, 2)
        self.assertEqual(a.explicit_chars, 2)
        self.assertEqual(a.typed["seg0"], "abcd")

    def test_shift_shifts_explicit_characters(self):
        a = agg_of(["down(ShiftLeft); down(KeyA); up(KeyA); up(ShiftLeft)"])
        self.assertEqual(a.typed["seg0"], "A")
        self.assertEqual(a.shift_transitions, 1)

    def test_shift_held_across_turns(self):
        a = agg_of(["down(ShiftLeft)", "down(KeyA); up(KeyA)", "up(ShiftLeft)"])
        self.assertEqual(a.typed["seg0"], "A")        # the hold carries over
        self.assertEqual(a.held_across_turns, 1)

    def test_backspace_edits_the_reconstruction(self):
        a = agg_of(['type("abc"); down(Backspace); up(Backspace)'])
        self.assertEqual(a.typed["seg0"], "ab")
        self.assertEqual(a.key_down["Backspace"], 1)
        self.assertEqual(a.structure["type_and_edit"], 1)

    def test_chord_keys_do_not_type(self):
        a = agg_of(["down(ControlLeft); down(KeyC); up(KeyC); up(ControlLeft)"])
        self.assertNotIn("seg0", a.typed)             # Ctrl+C is not text
        self.assertEqual(a.explicit_chars, 0)
        self.assertEqual(a.chords["Ctrl+KeyC"], 1)

    def test_type_under_modifier_is_flagged(self):
        a = agg_of(["down(ControlLeft)", 'type("x")', "up(ControlLeft)"])
        self.assertEqual(a.typing_under_mod, 1)

    def test_char_classes_and_lengths(self):
        a = agg_of(['type("Ab 1!")'])
        self.assertEqual(a.char_classes["letters"], 2)
        self.assertEqual(a.char_classes["uppercase"], 1)
        self.assertEqual(a.char_classes["digits"], 1)
        self.assertEqual(a.char_classes["space"], 1)
        self.assertEqual(a.char_classes["punctuation"], 1)
        self.assertEqual(a.max_type_len, 5)
        self.assertEqual(a.type_len[5], 1)


class HeldStateTest(unittest.TestCase):
    """Cross-turn press/release bookkeeping — invisible to a per-window pass."""

    def test_dangling_release(self):
        a = agg_of(["up(KeyA)"])
        self.assertEqual(a.dangling_up, 1)
        self.assertEqual(a.held_across_turns, 0)

    def test_redundant_press(self):
        a = agg_of(["down(KeyA)", "down(KeyA)"])
        self.assertEqual(a.redundant_down, 1)

    def test_still_held_at_segment_end(self):
        a = agg_of(["down(ShiftLeft)"], segments=2)
        self.assertEqual(a.held_at_end["ShiftLeft"], 2)   # once per segment
        # ...and the hold does NOT leak into the next segment
        self.assertEqual(a.dangling_up, 0)

    def test_chords_survive_a_turn_boundary(self):
        a = agg_of(["down(MetaLeft)", "down(KeyC); up(KeyC)", "up(MetaLeft)"])
        self.assertEqual(a.chords["Meta+KeyC"], 1)

    def test_per_segment_key_coverage(self):
        a = agg_of(["down(KeyA); up(KeyA)", "down(KeyA); up(KeyA)"], segments=3)
        self.assertEqual(a.key_down["KeyA"], 6)
        self.assertEqual(a.key_segments["KeyA"], 3)      # segments, not presses


class GridBoundTest(unittest.TestCase):
    def test_within_budget(self):
        # 10 Hz / 1 fps = 10 ticks per window: ten moves are legal.
        a = agg_of(["; ".join(["move(1,0)"] * 10)])
        self.assertEqual(a.over_grid, 0)

    def test_over_budget(self):
        a = agg_of(["; ".join(["move(1,0)"] * 25)])
        self.assertEqual(a.over_grid, 1)

    def test_barriers_raise_the_bound(self):
        # Each discrete primitive also flushes the accumulator, so a turn with
        # many key transitions may legitimately hold more move primitives.
        turn = "; ".join(["move(1,0); down(KeyA); up(KeyA)"] * 12)
        self.assertEqual(agg_of([turn]).over_grid, 0)

    def test_no_fps_means_no_check(self):
        a = agg_of(["; ".join(["move(1,0)"] * 50)], fps=0.0)
        self.assertEqual(a.over_grid, 0)


class ResultPayloadTest(unittest.TestCase):
    def test_shapes_examples_and_manifest_check(self):
        a = agg_of(["move(1,0); down(LMB); up(LMB)", 'type("hi")'], segments=2)
        out = a.result(
            mode="conversations", action_format="ordered_events_v3",
            manifest={"primitive_counts": {"move": 2, "scroll": 0, "down": 2,
                                           "up": 2, "type": 2}},
        )
        self.assertEqual(out["n_turns"], 4)
        self.assertEqual(out["n_active"], 4)
        shapes = {s["shape"]: s["count"] for s in out["shapes"]}
        self.assertEqual(shapes, {"move; down; up": 2, "type": 2})
        rows = {r["kind"]: (r["recomputed"], r["manifest"])
                for r in out["manifest_check"]["rows"]}
        self.assertEqual(rows["move"], (2, 2))
        self.assertFalse(out["manifest_check"]["partial"])
        self.assertEqual(out["typing"]["collapse_pct"], 100.0)
        # server-side state is separated from the UI payload
        self.assertIn("_typed", out)
        self.assertIn("_shape_examples", out)
        self.assertEqual(out["_shape_examples"]["type"][0]["turn"], 1)

    def test_histograms_cover_every_observation(self):
        a = agg_of(["move(1,0)", "NO_OP", 'type("abc")'])
        out = a.result(mode="conversations", action_format="ordered_events_v3",
                       manifest=None)
        for key in ("prims_per_turn_hist", "mouse_per_turn_hist",
                    "moves_per_turn_hist", "scrolls_per_turn_hist",
                    "clicks_per_turn_hist", "types_per_turn_hist"):
            with self.subTest(hist=key):
                self.assertEqual(sum(out[key]["counts"]), out["n_active"])
        self.assertEqual(sum(out["type_len_hist"]["counts"]), out["typing"]["n_type"])
        self.assertEqual(sum(out["moves"]["dx_hist"]["counts"]), out["moves"]["count"])

    def test_per_turn_hists_carry_their_filter_field_and_bounds(self):
        """A per-turn histogram is a filter control in the UI: each bucket's
        bounds ARE the min/max a click sets, so they must be on the payload and
        must line up with the labels."""
        a = agg_of(["move(1,0); down(LMB); up(LMB)"])
        out = a.result(mode="conversations", action_format="ordered_events_v3",
                       manifest=None)
        fields = {"prims_per_turn_hist": "actions", "mouse_per_turn_hist": "mouse",
                  "moves_per_turn_hist": "move", "scrolls_per_turn_hist": "scroll",
                  "clicks_per_turn_hist": "click", "types_per_turn_hist": "type"}
        for key, fld in fields.items():
            with self.subTest(hist=key):
                h = out[key]
                self.assertEqual(h["field"], fld)
                self.assertEqual(len(h["bounds"]), len(h["counts"]))
                self.assertEqual(h["bounds"][0], [0, 0])       # the idle bucket
                self.assertIsNone(h["bounds"][-1][1])          # open-ended tail
        # a histogram that is not a per-turn count is not clickable
        self.assertIsNone(out["type_len_hist"]["field"])


class TurnCountsTest(unittest.TestCase):
    """What the numeric filter selects on. ``mouse`` is the union the UI
    advertises (move + scroll + button transitions); ``click`` is presses only,
    so a click contributes 1 while its down/up pair contributes 2 to ``mouse``."""

    def test_mixed_turn(self):
        c = turn_counts(parse_turn(
            'move(1,0); move(2,0); scroll(0,-3); down(LMB); up(LMB); '
            'down(ControlLeft); type("hi"); up(ControlLeft)'))
        self.assertEqual(c["actions"], 8)
        self.assertEqual(c["move"], 2)
        self.assertEqual(c["scroll"], 1)
        self.assertEqual(c["mouse"], 5)      # 2 move + 1 scroll + down/up LMB
        self.assertEqual(c["click"], 1)      # presses only
        self.assertEqual(c["key"], 2)        # the modifier's down/up
        self.assertEqual(c["type"], 1)

    def test_idle_turns_count_zero(self):
        for text in ("NO_OP", "TERMINATE"):
            with self.subTest(text=text):
                self.assertEqual(turn_counts(parse_turn(text)),
                                 dict.fromkeys(
                                     ("actions", "mouse", "move", "scroll",
                                      "click", "key", "type"), 0))

    def test_counts_match_the_histogram_buckets(self):
        """The filter and the per-turn histograms must agree, or clicking a
        bucket would select a different set than the bucket counted."""
        turns = ["move(1,0); scroll(0,-1); down(LMB); up(LMB)", 'type("abc")']
        a = agg_of(turns)
        self.assertEqual(a.mouse_per_turn[4], 1)   # move + scroll + down + up
        self.assertEqual(a.clicks_per_turn[1], 1)
        self.assertEqual(a.clicks_per_turn[0], 1)  # the type() turn
        self.assertEqual(a.prims_per_turn[4], 1)


class TurnFilterTest(unittest.TestCase):
    def test_empty_filter_is_falsey_and_matches_everything(self):
        tf = TurnFilter.from_query({})
        self.assertFalse(tf)
        self.assertTrue(tf.accepts(turn_counts(parse_turn("NO_OP"))))
        self.assertEqual(tf.label(), "")

    def test_bounds_are_inclusive(self):
        tf = TurnFilter.from_query({"min_actions": ["2"], "max_actions": ["3"]})
        counts = [turn_counts(parse_turn(t)) for t in
                  ("move(1,0)", "move(1,0); move(2,0)",
                   "move(1,0); down(LMB); up(LMB)",
                   "move(1,0); move(2,0); down(LMB); up(LMB)")]
        self.assertEqual([tf.accepts(c) for c in counts],
                         [False, True, True, False])

    def test_bounds_and_together(self):
        # "scrolls but never moves"
        tf = TurnFilter.from_query({"min_scroll": ["1"], "max_move": ["0"]})
        self.assertTrue(tf.accepts(turn_counts(parse_turn("scroll(0,-3)"))))
        self.assertFalse(tf.accepts(turn_counts(parse_turn("move(1,0); scroll(0,-3)"))))

    def test_blank_and_junk_values_are_no_bound(self):
        for q in ({"min_move": [""]}, {"min_move": ["  "]}, {"max_move": ["abc"]}):
            with self.subTest(q=q):
                self.assertFalse(TurnFilter.from_query(q))

    def test_label_reads_as_the_bound(self):
        self.assertEqual(
            TurnFilter.from_query({"min_actions": ["10"]}).label(), "actions/turn ≥10")
        self.assertEqual(
            TurnFilter.from_query({"max_move": ["0"]}).label(), "move/turn ≤0")
        self.assertEqual(
            TurnFilter.from_query({"min_click": ["1"], "max_click": ["1"]}).label(),
            "click/turn 1")
        self.assertEqual(
            TurnFilter.from_query({"min_mouse": ["2"], "max_mouse": ["5"]}).label(),
            "mouse/turn 2–5")


class SearchTurnsTest(unittest.TestCase):
    """The filter endpoint: substring AND count bounds, over whole turns."""

    def _ds(self):
        return _FakeDataset({
            "a": ["move(1,0)",                                    # 1 action
                  "move(1,0); move(2,0); down(LMB); up(LMB)",     # 4, 1 click
                  "NO_OP"],
            "b": ["scroll(0,-3); scroll(0,-3)",                   # 2, no move
                  '<think>plan</think>type("hi"); down(Return); up(Return)',
                  "100 0 0 ; +LMB"],                              # violation
        })

    def test_no_criteria_matches_nothing(self):
        out = search_turns(self._ds(), {}, "")
        self.assertEqual((out["n_turns"], out["n_segments"]), (0, 0))
        self.assertEqual(out["turns_total"], 6)

    def test_count_bound_alone(self):
        out = search_turns(self._ds(), {}, "",
                           TurnFilter.from_query({"min_actions": ["2"]}))
        self.assertEqual(out["n_turns"], 3)          # the 4-, 2- and 3-prim turns
        self.assertEqual(out["n_segments"], 2)
        self.assertEqual(out["filter"], "actions/turn ≥2")

    def test_mouse_bound_excludes_typing_turns(self):
        out = search_turns(self._ds(), {}, "",
                           TurnFilter.from_query({"min_mouse": ["2"]}))
        self.assertEqual(out["n_turns"], 2)          # the 4-prim turn + the scrolls
        mix = {m["key"]: m["count"] for m in out["mix"]}
        self.assertEqual(mix["move"], 2)
        self.assertEqual(mix["scroll"], 2)
        self.assertEqual(mix["click"], 1)
        self.assertEqual(mix["actions"], 6)

    def test_scroll_without_move(self):
        out = search_turns(self._ds(), {}, "",
                           TurnFilter.from_query({"min_scroll": ["1"],
                                                  "max_move": ["0"]}))
        self.assertEqual(out["n_turns"], 1)
        self.assertEqual(out["examples"][0]["segment_id"], "b")

    def test_idle_turns_are_selectable(self):
        out = search_turns(self._ds(), {}, "",
                           TurnFilter.from_query({"max_actions": ["0"]}))
        self.assertEqual(out["n_turns"], 1)          # the NO_OP; the violation is not
        self.assertEqual(out["unparsed"], 1)

    def test_substring_and_bound_are_anded(self):
        ds = self._ds()
        loose = search_turns(ds, {}, "move(")
        self.assertEqual(loose["n_turns"], 2)
        tight = search_turns(ds, {}, "move(",
                             TurnFilter.from_query({"min_move": ["2"]}))
        self.assertEqual(tight["n_turns"], 1)
        self.assertEqual(tight["examples"][0]["turn"], 1)

    def test_examples_strip_thinking_blocks(self):
        out = search_turns(self._ds(), {}, 'type("')
        self.assertEqual(out["n_turns"], 1)
        self.assertNotIn("<think>", out["examples"][0]["text"])

    def test_typed_text_coverage_is_substring_only(self):
        typed = {"b": "hi"}
        out = search_turns(self._ds(), typed, "hi",
                           TurnFilter.from_query({"min_actions": ["99"]}))
        self.assertEqual(out["n_turns"], 0)          # no turn has 99 primitives
        self.assertEqual(out["typed_hits"], 1)       # segment text is unfiltered
        out = search_turns(self._ds(), typed, "",
                           TurnFilter.from_query({"min_actions": ["1"]}))
        self.assertEqual(out["typed_hits"], 0)       # nothing to look for


if __name__ == "__main__":
    unittest.main()
