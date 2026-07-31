"""The action-span conversion CONTRACT (C1-C4), ported from the reference fix.

Absorbed from ``audit_operand/action_span_conversion.py`` (the ladder owner's tested
reference, 7/7 groups passing). Its test groups T1-T7 are reproduced here as the
production suite, keeping the two aspects that matter most:

* **T6** simulates the original buggy ``if fmt == "absolute": return resp_text`` and
  asserts the C2 check catches it. The standard: *a regression test that would not
  have caught the real bug is worthless.* Every defect test in this package follows
  it — each one first reproduces the defect, then shows the guard firing.
* **T7** treats a multi-block move+click as a single action span, the case a naive
  single-span implementation gets wrong.
* **T4** asserts the identity round trip.

The contract:

  C1  Conversion rewrites ONLY the action span; every other byte passes through.
  C2  Two format arms from the same source are byte-identical OUTSIDE their spans.
  C3  Prose is format-independent — never a function of the action grammar.
  C4  Dropping prose is an explicit, SYMMETRIC option applied to all arms.

The reference deliberately did NOT patch ``build_osworld_format_records.py`` in place,
because other owners' jobs read it. This module is the production path; retiring the
buggy branch behind it is a coordinated cutover, not a unilateral edit.
"""

from __future__ import annotations

import unittest

from rft.conversion import (
    assert_only_action_span_changed,
    assert_prose_policy_symmetric,
    convert_action_span,
    split_response,
)
from rft.errors import SchemaError

#: The reference fixtures, verbatim from action_span_conversion.py.
TOOL = (
    'Action: Click the "X" button on the top-right corner of the '
    '"Can\'t update Chrome" pop-up to close it.\n'
    '<tool_call>\n{"name": "computer_use", "arguments": '
    '{"action": "left_click", "coordinate": [982, 127]}}\n</tool_call>'
)
BARE = (
    "<think>\nI want to close the update popup.\n</think>\n"
    "925 -403 0 ; +LMB -LMB"
)
MULTI = (
    'Move then click.\n<tool_call>\n{"a": 1}\n</tool_call>\n'
    '<tool_call>\n{"a": 2}\n</tool_call>'
)


class T1ProseSurvivesConversionTest(unittest.TestCase):
    """T1 — the actual bug: prose must survive an action-format conversion."""

    def test_prose_preserved_action_replaced(self) -> None:
        out = convert_action_span(TOOL, lambda _a: "925 -403 0 ; +LMB -LMB")
        self.assertIn("top-right corner", out)
        self.assertIn("925 -403 0 ; +LMB -LMB", out)
        self.assertNotIn("982", out, "the old action must be gone")


class T2ArmsIdenticalOutsideSpanTest(unittest.TestCase):
    """T2 — C2: two arms from one source share prefix and suffix byte-for-byte."""

    def test_prefix_and_suffix_identical_across_arms(self) -> None:
        a = convert_action_span(TOOL, lambda _s: "AAA")
        b = convert_action_span(TOOL, lambda _s: "BBBBBB")
        pa = split_response(a)
        pb = split_response(b)
        self.assertEqual(pa.prefix, pb.prefix)
        self.assertEqual(pa.suffix, pb.suffix)
        self.assertNotEqual(pa.action_span, pb.action_span)


class T3ThinkingBlocksArePoseTest(unittest.TestCase):
    """T3 — a ``<think>`` block is prose (C3) and must survive."""

    def test_think_block_preserved(self) -> None:
        out = convert_action_span(BARE, lambda _a: "1 2 0")
        self.assertIn("<think>", out)
        self.assertIn("close the update popup", out)
        self.assertTrue(out.rstrip().endswith("1 2 0"))


class T4IdentityRoundTripTest(unittest.TestCase):
    """T4 — the identity converter must be the identity on the whole response."""

    def test_identity_round_trip(self) -> None:
        for src in (TOOL, BARE, MULTI, "NO_OP", "", "   ", "a\n\nb\n925 -403 0\n"):
            with self.subTest(src=src[:40]):
                out = convert_action_span(src, lambda a: a, require_action=False)
                self.assertEqual(out, src)

    def test_split_is_a_partition_for_every_fixture(self) -> None:
        for src in (TOOL, BARE, MULTI, "NO_OP", "", "   ", "a\n\nb\n925 -403 0\n"):
            with self.subTest(src=src[:40]):
                p = split_response(src)
                self.assertEqual(p.prefix + p.action_span + p.suffix, src)


class T5KeepProseFalseIsSymmetricTest(unittest.TestCase):
    """T5 / C4 — ``keep_prose=False`` yields action-only for ANY grammar."""

    def test_action_only_for_both_grammars(self) -> None:
        for src in (TOOL, BARE):
            with self.subTest(src=src[:40]):
                out = convert_action_span(src, lambda a: a, keep_prose=False)
                self.assertNotIn("<think>", out)
                self.assertNotIn("top-right", out)

    def test_symmetric_policy_passes(self) -> None:
        assert_prose_policy_symmetric({"absolute": False, "moverel": False, "diffabs": False})
        assert_prose_policy_symmetric({"absolute": True, "moverel": True})

    def test_asymmetric_policy_is_the_defect_and_is_refused(self) -> None:
        """This IS the original bug, stated as a policy: absolute keeps, others drop."""
        with self.assertRaises(SchemaError) as ctx:
            assert_prose_policy_symmetric(
                {"absolute": True, "moverel": False, "diffabs": False}
            )
        msg = str(ctx.exception)
        self.assertIn("ASYMMETRIC", msg)
        self.assertIn("2383/2383 vs 0/2441", msg)

    def test_empty_policy_map_is_an_error(self) -> None:
        with self.assertRaises(SchemaError):
            assert_prose_policy_symmetric({})


class T6OriginalBugIsDetectedTest(unittest.TestCase):
    """T6 — simulate the ORIGINAL buggy branch and assert the C2 check catches it.

    "A regression test that wouldn't have caught the real bug is worthless."
    """

    @staticmethod
    def _buggy(text: str, fmt: str) -> str:
        """Verbatim reproduction of ``convert_response``'s branch structure."""
        if fmt == "absolute":
            return text  # prose kept
        return split_response(text).action_span  # prose dropped

    def test_the_asymmetry_is_detected_by_the_prefix_comparison(self) -> None:
        absolute = self._buggy(TOOL, "absolute")
        moverel = self._buggy(TOOL, "moverel")
        self.assertNotEqual(
            split_response(absolute).prefix,
            split_response(moverel).prefix,
            "the C2 check would not have caught the real bug",
        )

    def test_the_gate_raises_on_the_buggy_output(self) -> None:
        from rft.conversion import ContextAlteredError

        with self.assertRaises(ContextAlteredError):
            assert_only_action_span_changed(TOOL, self._buggy(TOOL, "moverel"))

    def test_the_gate_passes_on_the_fixed_output(self) -> None:
        fixed = convert_action_span(TOOL, lambda _a: "925 -403 0 ; +LMB -LMB")
        assert_only_action_span_changed(TOOL, fixed)

    def test_the_arm_parity_check_also_catches_it(self) -> None:
        from rft.arms import UncontrolledComparisonError, assert_arms_differ_only_in

        arms = {
            "absolute": [self._buggy(TOOL, "absolute")] * 20,
            "moverel": [self._buggy(TOOL, "moverel")] * 20,
        }
        with self.assertRaises(UncontrolledComparisonError):
            assert_arms_differ_only_in(arms, dimension="action_format")


class T7MultiBlockSpanTest(unittest.TestCase):
    """T7 — consecutive tool calls are ONE action span (move + click, drags)."""

    def test_two_blocks_form_one_span(self) -> None:
        parts = split_response(MULTI)
        self.assertEqual(parts.action_span.count("<tool_call>"), 2)
        self.assertEqual(parts.prefix.strip(), "Move then click.")

    def test_converting_a_multi_block_span_replaces_all_of_it(self) -> None:
        out = convert_action_span(MULTI, lambda _a: "SINGLE")
        self.assertEqual(out, "Move then click.\nSINGLE")
        self.assertNotIn("<tool_call>", out)

    def test_the_whitespace_between_blocks_belongs_to_the_span(self) -> None:
        parts = split_response(MULTI)
        self.assertIn("</tool_call>\n<tool_call>", parts.action_span)


if __name__ == "__main__":
    unittest.main()
