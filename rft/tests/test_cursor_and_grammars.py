"""Regression gates for defects #4, #5, #14, #15 — measurement and grammar defects.

* #4  ``info.cursor_after`` is stale; motion must be measured BETWEEN steps.
* #5  a computer_use-only mouse-op detector scores the bare-line grammar a fake 0%.
* #14 11-13% of deltatype completions omit the scroll token; strict parsing counts
      them as no-moves, penalising only that grammar.
* #15 the 4B-Thinking chat template injects the OPENING ``<think>`` into the prompt.
"""

from __future__ import annotations

import unittest

from rft import anchors
from rft.cursor import (
    STALE_CURSOR_AFTER_THRESHOLD,
    cursor_motion_between_steps,
    detect_stale_cursor_after,
)
from rft.errors import MissingFieldError, SchemaError
from rft.evalparser import MISSING_SYMBOLS, describe
from rft.grammars import (
    COMPUTER_USE_MOUSE_OPS,
    available_grammars,
    get_grammar,
    has_mouse_op,
    parse_completion,
    strip_thinking,
)


def _step(before: tuple[int, int], after: tuple[int, int] | None = None) -> dict:
    info: dict = {"cursor_before": list(before)}
    if after is not None:
        info["cursor_after"] = list(after)
    return {"info": info}


class Defect04StaleCursorAfterTest(unittest.TestCase):
    """Defect #4: ``cursor_after == cursor_before`` on 97.4% of 17,090 steps.

    A within-step ``cursor_after - cursor_before`` therefore reads ~0 motion for a
    policy that may be moving the cursor a lot. The corroborating grounding-harness
    reading is ``telemetry_staleness.stale_frac = 0.983``.
    """

    def test_stale_field_is_detected_and_named(self) -> None:
        # 39 stale steps out of 40 => 97.5%, the observed condition.
        steps = [_step((100, 100), (100, 100)) for _ in range(39)]
        steps.append(_step((100, 100), (110, 100)))
        report = detect_stale_cursor_after(steps)
        self.assertEqual(report.n_steps, 40)
        self.assertEqual(report.n_equal, 39)
        self.assertTrue(report.is_stale)
        self.assertIn("STALE", report.describe())
        self.assertIn("BETWEEN steps", report.describe())

    def test_threshold_matches_the_observed_defect_rate(self) -> None:
        anchor = anchors.get_anchor("cursor_after_stale_rate")
        self.assertGreater(anchor.value, STALE_CURSOR_AFTER_THRESHOLD)

    def test_healthy_harness_is_not_flagged(self) -> None:
        steps = [_step((0, 0), (10, 0)), _step((10, 0), (20, 0)), _step((20, 0), (20, 0))]
        self.assertFalse(detect_stale_cursor_after(steps).is_stale)

    def test_between_step_motion_sees_what_within_step_motion_misses(self) -> None:
        """The whole point: with a stale cursor_after, only the between-step read works."""
        steps = [
            _step((100, 100), (100, 100)),  # stale: says "no motion"
            _step((150, 120), (150, 120)),  # but the cursor clearly moved +50,+20
            _step((150, 90), (150, 90)),
        ]
        # within-step (the wrong way) sees nothing:
        self.assertTrue(detect_stale_cursor_after(steps).is_stale)
        # between-step (the only supported way) sees the real motion:
        report = cursor_motion_between_steps(steps)
        self.assertEqual(report.displacements, ((50.0, 20.0), (0.0, -30.0)))
        self.assertEqual(report.n_moved, 2)
        self.assertEqual(report.n_transitions, 2)
        self.assertAlmostEqual(report.moved_fraction, 1.0)

    def test_no_within_step_helper_exists(self) -> None:
        """There must be no API that subtracts cursor_after from cursor_before."""
        import rft.cursor as cursor_mod

        for name in dir(cursor_mod):
            self.assertNotIn(
                "within_step", name, "a within-step motion helper must not exist"
            )

    def test_single_step_trajectory_has_undefined_motion(self) -> None:
        report = cursor_motion_between_steps([_step((0, 0))])
        with self.assertRaises(SchemaError):
            _ = report.moved_fraction

    def test_missing_cursor_observation_raises(self) -> None:
        with self.assertRaises(MissingFieldError):
            cursor_motion_between_steps([{"info": {}}, {"info": {"cursor_before": [1, 1]}}])


class Defect05MouseOpDetectorTest(unittest.TestCase):
    """Defect #5: a detector matching only computer_use op names scored the bare-line
    grammar a fake 0% mouse-op rate. The real rate was 80.1%.
    """

    BARE_WITH_MOUSE = [
        "120 -40 0 ; +LMB -LMB",
        "0 0 0 ; +LMB",
        "-15 3 0",
        "0 0 -2",
    ]
    BARE_WITHOUT_MOUSE = [
        "NO_OP",
        "0 0 0 ; +KeyA -KeyA",
    ]

    def test_naive_computer_use_only_detector_reports_zero(self) -> None:
        """Reproduce the defect to prove the test is testing something."""

        def naive_detector(text: str) -> bool:
            return any(op in text for op in COMPUTER_USE_MOUSE_OPS)

        rate = sum(naive_detector(t) for t in self.BARE_WITH_MOUSE) / len(
            self.BARE_WITH_MOUSE
        )
        self.assertEqual(rate, 0.0, "the naive detector must score bare-line at 0%")

    def test_grammar_dispatched_detector_is_correct(self) -> None:
        for text in self.BARE_WITH_MOUSE:
            self.assertTrue(has_mouse_op(text, grammar="bare_line"), text)
        for text in self.BARE_WITHOUT_MOUSE:
            self.assertFalse(has_mouse_op(text, grammar="bare_line"), text)

    def test_detector_must_be_told_the_grammar(self) -> None:
        """There is no auto-detection; an unnamed grammar is an error."""
        with self.assertRaises(TypeError):
            has_mouse_op("120 -40 0")  # type: ignore[call-arg]
        with self.assertRaises(SchemaError):
            has_mouse_op("120 -40 0", grammar="guess")

    def test_bare_line_mouse_op_anchor_is_reproducible(self) -> None:
        """Build a fixture at the anchored 80.1% rate and read it back."""
        anchor = anchors.get_anchor("bare_line_mouse_op_rate")
        n = 1000
        n_mouse = round(anchor.value * n)
        fixture = ["10 10 0 ; +LMB -LMB"] * n_mouse + ["NO_OP"] * (n - n_mouse)
        observed = sum(has_mouse_op(t, grammar="bare_line") for t in fixture) / n
        anchor.check(observed)
        # And the naive detector must FAIL the same anchor - that is the defect.
        with self.assertRaises(Exception):
            anchor.check(0.0)

    def test_scroll_only_counts_as_a_mouse_op(self) -> None:
        parsed = parse_completion("0 0 -3", grammar="bare_line")
        self.assertTrue(parsed.has_mouse_op)
        self.assertEqual([o.kind for o in parsed.mouse_ops], ["scroll"])


@unittest.skipIf(
    not get_grammar("deltatype", require_available=False).available,
    f"deltatype needs eval/action_parser.py symbols; {describe()}",
)
class Defect14MissingScrollTokenTest(unittest.TestCase):
    """Defect #14: 11-13% of deltatype completions omit the scroll token.

    ``dx dy ; +LMB -LMB`` instead of ``dx dy scroll ; +LMB -LMB``. Strict three-token
    parsing counts those as no-moves — a penalty landing only on this grammar. The
    move must be RECOVERED and the omission COUNTED, never silently repaired.
    """

    def test_two_token_mouse_segment_recovers_the_move(self) -> None:
        parsed = parse_completion("120 -40 ; +LMB -LMB", grammar="deltatype")
        self.assertEqual(parsed.net_delta, (120, -40))
        self.assertTrue(parsed.has_mouse_op)

    def test_the_omission_is_counted_as_an_anomaly(self) -> None:
        parsed = parse_completion("120 -40 ; +LMB -LMB", grammar="deltatype")
        self.assertIn("missing_scroll_token", parsed.anomalies)

    def test_well_formed_completions_carry_no_anomaly(self) -> None:
        parsed = parse_completion("120 -40 0 ; +LMB -LMB", grammar="deltatype")
        self.assertEqual(parsed.anomalies, ())
        self.assertEqual(parsed.net_delta, (120, -40))

    def test_strict_parsing_would_have_scored_these_as_no_moves(self) -> None:
        """Reproduce the penalty to prove the recovery matters."""
        from rft.evalparser import parse_deltatype

        with self.assertRaises(ValueError):
            parse_deltatype("120 -40 ; +LMB -LMB")

    def test_control_tokens_are_not_mistaken_for_a_two_token_mouse_segment(self) -> None:
        for token in ("NO_OP", "TERMINATE", "FAIL"):
            parsed = parse_completion(token, grammar="deltatype")
            self.assertEqual(parsed.anomalies, ())
            self.assertFalse(parsed.has_mouse_op)
        self.assertTrue(parse_completion("TERMINATE", grammar="deltatype").terminate)

    def test_missing_scroll_rate_matches_the_anchor(self) -> None:
        anchor = anchors.get_anchor("deltatype_missing_scroll_rate")
        n = 1000
        n_bad = round(anchor.value * n)
        fixture = ["10 -5 ; +LMB -LMB"] * n_bad + ["10 -5 0 ; +LMB -LMB"] * (n - n_bad)
        observed = sum(
            1
            for t in fixture
            if "missing_scroll_token" in parse_completion(t, grammar="deltatype").anomalies
        ) / n
        anchor.check(observed)


class Defect15ThinkingTemplateTest(unittest.TestCase):
    """Defect #15: the 4B-Thinking chat template injects the OPENING ``<think>`` into
    the prompt, so the completion carries only the closing tag. A parser requiring a
    balanced pair rejects everything and reports a false all-zeros.
    """

    def test_closing_tag_only_is_the_template_injected_case(self) -> None:
        visible, had = strip_thinking("I should move right.</think>120 0 0 ; +LMB -LMB")
        self.assertTrue(had)
        self.assertEqual(visible, "120 0 0 ; +LMB -LMB")

    def test_balanced_pair_also_works(self) -> None:
        visible, had = strip_thinking("<think>reason</think>\n120 0 0")
        self.assertTrue(had)
        self.assertEqual(visible, "120 0 0")

    def test_no_tags_is_untouched(self) -> None:
        visible, had = strip_thinking("120 0 0")
        self.assertFalse(had)
        self.assertEqual(visible, "120 0 0")

    def test_a_naive_balanced_pair_parser_rejects_everything(self) -> None:
        """Reproduce the false all-zeros."""
        import re

        naive = re.compile(r"<think>.*?</think>\s*(.+)", re.DOTALL)
        completions = ["reasoning</think>120 0 0"] * 50
        matched = sum(1 for c in completions if naive.match(c))
        self.assertEqual(matched, 0, "the naive parser must reject 100% of these")
        # the correct handling parses all of them:
        parsed = [strip_thinking(c)[0] for c in completions]
        self.assertTrue(all(p == "120 0 0" for p in parsed))

    def test_thinking_completion_parses_through_the_grammar(self) -> None:
        parsed = parse_completion(
            "The target is right.</think>120 -40 0 ; +LMB -LMB", grammar="bare_line"
        )
        self.assertEqual(parsed.net_delta, (120, -40))

    def test_unterminated_thinking_is_an_error_not_a_convention(self) -> None:
        with self.assertRaises(SchemaError):
            strip_thinking("<think>I am still reasoning and got truncated")


class GrammarRegistryTest(unittest.TestCase):
    """The registry must be honest about what the loaded eval parser supports."""

    def test_availability_is_reported_not_faked(self) -> None:
        avail = available_grammars()
        self.assertIn("bare_line", avail)
        self.assertTrue(avail["bare_line"], "bare_line needs only parse_action")
        for name, ok in avail.items():
            if not ok:
                self.assertTrue(
                    get_grammar(name, require_available=False).missing,
                    f"{name} is unavailable but names no missing symbol",
                )

    def test_unavailable_grammar_raises_with_the_missing_symbol(self) -> None:
        for name, ok in available_grammars().items():
            if ok:
                continue
            with self.assertRaises(SchemaError) as ctx:
                get_grammar(name)
            self.assertTrue(
                any(sym in str(ctx.exception) for sym in MISSING_SYMBOLS),
                "the error must name the missing eval-parser symbol",
            )
            break

    def test_relative_and_absolute_conventions_are_declared(self) -> None:
        """The native_rel lesson: the convention must travel with the grammar."""
        self.assertTrue(get_grammar("bare_line").relative)
        abs_g = get_grammar("computer_use_absolute", require_available=False)
        rel_g = get_grammar("computer_use_move_rel", require_available=False)
        self.assertFalse(abs_g.relative)
        self.assertTrue(rel_g.relative)
        self.assertIn("ABSOLUTE", abs_g.convention)
        self.assertIn("RELATIVE", rel_g.convention)

    def test_parser_is_the_harness_parser_not_a_copy(self) -> None:
        from rft.evalparser import ACTION_PARSER_PATH

        self.assertTrue(ACTION_PARSER_PATH.endswith("eval/action_parser.py"))


if __name__ == "__main__":
    unittest.main()
