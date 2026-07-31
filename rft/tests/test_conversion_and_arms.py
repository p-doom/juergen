"""HIGH-PRIORITY gates: conversion must touch only the action span, and format arms
must differ in exactly one thing.

**The defect.** ``build_osworld_format_records.py::convert_response`` began with
``if fmt == "absolute": return resp_text`` — the teacher's response verbatim, prose
preamble and canonical ``<tools>`` schema included — while every relative branch
re-rendered action-only and silently discarded the prose. Measured over the shipped
datasets: reasoning preamble present in **2383/2383 absolute vs 0/2441 relative**
records; ``<tools>`` schema likewise.

**Why it is severe.** The reasoning preamble is format-INDEPENDENT natural language.
Only the ACTION span ever needed converting. The pipeline therefore deleted the
visual-reasoning scratchpad from exactly the arm that had to learn a new convention
and kept it for the arm that had nothing to learn — invalidating every trained
absolute-vs-relative comparison the research programme reasoned from. Every output
looked individually plausible, so nobody noticed for weeks.

``tests/fixtures/teacher_arm_targets.json`` holds 40 REAL assistant targets from each
shipped arm, so these tests run against the actual data, not synthetic action-only
strings.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rft.arms import (
    UncontrolledComparisonError,
    arm_parity_report,
    assert_arms_differ_only_in,
    profile_arm,
)
from rft.conversion import (
    ContextAlteredError,
    ContextFeatures,
    assert_only_action_span_changed,
    convert_action_span,
    split_response,
)
from rft.errors import SchemaError
from rft.records import build_records

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TEACHER_ARMS = json.loads((FIXTURES / "teacher_arm_targets.json").read_text())

#: A real absolute-arm target: reasoning preamble + tool call.
REAL_ABSOLUTE = (
    "Action: Press Ctrl+Shift+T to reopen the most recently closed tab in the browser.\n"
    '<tool_call>\n{"name": "computer_use", "arguments": {"action": "key", '
    '"keys": ["ctrl", "shift", "t"]}}\n</tool_call>'
)
#: The same record as the relative arm actually shipped it: preamble DELETED.
REAL_MOVEREL = (
    '<tool_call>\n{"name": "computer_use", "arguments": {"action": "key", '
    '"keys": ["ctrl", "shift", "t"]}}\n</tool_call>'
)


class SplitResponseTest(unittest.TestCase):
    """The split must be an exact partition, or nothing built on it is trustworthy."""

    def test_partition_is_exact_on_real_targets(self) -> None:
        for arm, records in TEACHER_ARMS.items():
            for rec in records:
                text = rec["target"]
                parts = split_response(text)
                self.assertEqual(
                    parts.prefix + parts.action_span + parts.suffix,
                    text,
                    f"{arm}/{rec['sample_id']}: split is not a partition",
                )

    def test_tool_call_span_is_isolated(self) -> None:
        parts = split_response(REAL_ABSOLUTE)
        self.assertEqual(parts.kind, "tool_call")
        self.assertTrue(parts.prefix.startswith("Action: Press Ctrl+Shift+T"))
        self.assertTrue(parts.action_span.startswith("<tool_call>"))
        self.assertTrue(parts.action_span.endswith("</tool_call>"))
        self.assertTrue(parts.has_prose_context)

    def test_consecutive_tool_calls_form_one_span(self) -> None:
        """A drag is mouse_down + mouse_move + mouse_up; splitting them is meaningless."""
        text = (
            "Reasoning here.\n"
            '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_mouse_down"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "computer_use", "arguments": {"action": "move_rel", "coordinate": [10, 5]}}\n</tool_call>'
        )
        parts = split_response(text)
        self.assertEqual(parts.action_span.count("<tool_call>"), 2)
        self.assertEqual(parts.prefix, "Reasoning here.\n")
        self.assertEqual(parts.suffix, "")

    def test_bare_line_span_is_isolated(self) -> None:
        parts = split_response("The target is right and up.\n120 -40 0 ; +LMB -LMB\n")
        self.assertEqual(parts.kind, "bare_line")
        self.assertEqual(parts.prefix, "The target is right and up.\n")
        self.assertEqual(parts.action_span, "120 -40 0 ; +LMB -LMB")
        self.assertTrue(parts.has_prose_context)

    def test_action_only_response_has_no_prose(self) -> None:
        self.assertFalse(split_response(REAL_MOVEREL).has_prose_context)

    def test_unrecognised_action_is_flagged_not_guessed(self) -> None:
        """Prose-only text falls to the permissive `last_line` split, which is fine to
        copy through but is NOT a recognised action and must not be converted."""
        parts = split_response("I am not sure what to do here.")
        self.assertEqual(parts.kind, "last_line")
        self.assertFalse(parts.recognised)

    def test_blank_text_has_no_action_at_all(self) -> None:
        parts = split_response("   ")
        self.assertEqual(parts.kind, "none")
        self.assertFalse(parts.recognised)


class Test1ConversionAltersOnlyTheActionSpanTest(unittest.TestCase):
    """TEST 1 — a conversion must alter ONLY the action span.

    The general form of the bug: anything that rewrites the whole response can delete
    format-independent content. Run against real teacher responses with prose.
    """

    def test_the_historical_converter_is_caught(self) -> None:
        """Reproduce ``convert_response``'s relative branch and prove it fails."""

        def historical_relative_branch(resp_text: str) -> str:
            # This is what the shipped code did: re-render action-only, dropping
            # everything around it.
            return REAL_MOVEREL

        converted = historical_relative_branch(REAL_ABSOLUTE)
        with self.assertRaises(ContextAlteredError) as ctx:
            assert_only_action_span_changed(REAL_ABSOLUTE, converted, context="moverel")
        msg = str(ctx.exception)
        self.assertIn("OUTSIDE the action span", msg)
        self.assertIn("PREFIX changed", msg)
        self.assertIn("format-INDEPENDENT", msg)

    def test_convert_action_span_preserves_prose_by_construction(self) -> None:
        """The fix: a converter expressed as action_span -> action_span CANNOT drop it."""
        converted = convert_action_span(REAL_ABSOLUTE, lambda _span: REAL_MOVEREL)
        assert_only_action_span_changed(REAL_ABSOLUTE, converted)
        self.assertTrue(converted.startswith("Action: Press Ctrl+Shift+T"))
        self.assertIn(REAL_MOVEREL, converted)

    def test_real_teacher_responses_survive_a_grammar_change(self) -> None:
        """Over every real absolute target: convert the action to a bare-line delta and
        require every non-action byte to be identical."""
        n = 0
        for rec in TEACHER_ARMS["absolute"]:
            source = rec["target"]
            if split_response(source).kind == "none":
                continue
            converted = convert_action_span(source, lambda _span: "120 -40 0 ; +LMB -LMB")
            assert_only_action_span_changed(
                source, converted, context=str(rec["sample_id"])
            )
            n += 1
        self.assertGreater(n, 30, "fixture should contain many convertible targets")

    def test_prose_bearing_sources_are_actually_in_the_fixture(self) -> None:
        """Guard the guard: a fixture of action-only strings would prove nothing."""
        with_prose = sum(
            1 for rec in TEACHER_ARMS["absolute"] if split_response(rec["target"]).has_prose_context
        )
        self.assertGreater(
            with_prose, 30, "the absolute fixture must contain reasoning prose"
        )

    def test_a_suffix_change_is_also_caught(self) -> None:
        source = "Reasoning.\n<tool_call>\n{}\n</tool_call>\nTrailing note."
        converted = "Reasoning.\n<tool_call>\n{}\n</tool_call>"
        with self.assertRaises(ContextAlteredError) as ctx:
            assert_only_action_span_changed(source, converted)
        self.assertIn("SUFFIX changed", str(ctx.exception))

    def test_an_identical_response_passes(self) -> None:
        assert_only_action_span_changed(REAL_ABSOLUTE, REAL_ABSOLUTE)

    def test_converting_a_prose_free_response_still_works(self) -> None:
        converted = convert_action_span(REAL_MOVEREL, lambda _s: "NO_OP")
        self.assertEqual(converted, "NO_OP")

    def test_a_response_with_no_recognised_action_refuses_wholesale_rewrite(self) -> None:
        with self.assertRaises(SchemaError) as ctx:
            convert_action_span("no action at all here", lambda _s: "0 0 0")
        self.assertIn("RECOGNISED", str(ctx.exception))

    def test_unrecognised_text_can_be_copied_through_explicitly(self) -> None:
        text = "no action at all here"
        self.assertEqual(
            convert_action_span(text, lambda _s: "0 0 0", require_action=False), text
        )


class Test2FormatArmParityTest(unittest.TestCase):
    """TEST 2 — arms built from the same source must be byte-identical outside the
    action span.
    """

    def test_the_shipped_arms_fail_parity(self) -> None:
        """The headline: the real shipped data is NOT a controlled comparison."""
        arms = {
            name: [rec["target"] for rec in records] for name, records in TEACHER_ARMS.items()
        }
        with self.assertRaises(UncontrolledComparisonError) as ctx:
            assert_arms_differ_only_in(arms, dimension="action_format")
        msg = str(ctx.exception)
        self.assertIn("NOT a controlled comparison", msg)
        self.assertIn("has_reasoning_preamble differs across arms", msg)

    def test_the_preamble_counts_reproduce_the_measured_asymmetry(self) -> None:
        """absolute = all records carry a preamble; relative arms = none do."""
        report = arm_parity_report(
            {n: [r["target"] for r in recs] for n, recs in TEACHER_ARMS.items()},
            dimension="action_format",
        )
        by_name = {p.name: p for p in report.profiles}
        absolute = by_name["absolute"]
        self.assertEqual(absolute.n_with_preamble, absolute.n_records)
        for name in ("moverel", "diffabs", "deltatype_raw"):
            if name not in by_name:
                continue
            self.assertEqual(
                by_name[name].n_with_preamble, 0, f"{name} unexpectedly has preambles"
            )

    def test_tools_schema_presence_is_also_compared(self) -> None:
        arms = {
            "absolute": ["<tools>{}</tools>\nReasoning.\n" + REAL_MOVEREL] * 10,
            "moverel": [REAL_MOVEREL] * 10,
        }
        report = arm_parity_report(arms, dimension="action_format")
        self.assertFalse(report.controlled)
        self.assertTrue(
            any("has_tools_schema differs" in v for v in report.violations),
            report.violations,
        )

    def test_correctly_built_arms_pass(self) -> None:
        """Same sources, action span converted, prose preserved => controlled."""
        sources = [rec["target"] for rec in TEACHER_ARMS["absolute"]]
        sources = [s for s in sources if split_response(s).kind != "none"]
        arms = {
            "absolute": sources,
            "moverel": [
                convert_action_span(s, lambda _sp: REAL_MOVEREL) for s in sources
            ],
            "diffabs": [
                convert_action_span(s, lambda _sp: "120 -40 0 ; +LMB -LMB") for s in sources
            ],
        }
        report = assert_arms_differ_only_in(arms, dimension="action_format")
        self.assertTrue(report.controlled, report.describe())

    def test_differing_record_counts_is_a_violation(self) -> None:
        arms = {"a": [REAL_MOVEREL] * 10, "b": [REAL_MOVEREL] * 9}
        report = arm_parity_report(arms, dimension="action_format")
        self.assertTrue(
            any("different record counts" in v for v in report.violations), report.violations
        )

    def test_opt_out_is_permitted_but_recorded(self) -> None:
        arms = {
            name: [rec["target"] for rec in records] for name, records in TEACHER_ARMS.items()
        }
        report = assert_arms_differ_only_in(
            arms,
            dimension="action_format",
            opt_out_reason="historical datasets, kept for reference only",
        )
        self.assertFalse(report.controlled)
        self.assertIn("OPT-OUT RECORDED", report.describe())
        self.assertEqual(
            report.as_dict()["opt_out_reason"],
            "historical datasets, kept for reference only",
        )

    def test_dimension_must_be_named_from_the_known_set(self) -> None:
        arms = {"a": [REAL_MOVEREL], "b": [REAL_MOVEREL]}
        with self.assertRaises(SchemaError) as ctx:
            assert_arms_differ_only_in(arms, dimension="vibes")
        self.assertIn("known:", str(ctx.exception))

    def test_parity_needs_at_least_two_arms(self) -> None:
        with self.assertRaises(SchemaError):
            arm_parity_report({"only": [REAL_MOVEREL]}, dimension="action_format")

    def test_empty_arm_is_an_error(self) -> None:
        with self.assertRaises(SchemaError):
            profile_arm("empty", [])


class BuildStageEnforcesActionSpanIsolationTest(unittest.TestCase):
    """The build stage must refuse to emit records whose conversion dropped context —
    so this is the builder's responsibility, not a reviewer's.
    """

    def _heldout(self, tmp: Path) -> Path:
        p = tmp / "heldout.tasks"
        p.write_text("\n".join(f"held{i}" for i in range(20)))
        return p

    def _rollouts(self, n_tasks: int = 20) -> list[dict]:
        return [
            {
                "task_id": f"t{t}",
                "sample_id": f"s{t}",
                "rollout_index": 0,
                "app": "chrome",
                "scores": {"reward": 1.0},
                "response": REAL_ABSOLUTE,
            }
            for t in range(n_tasks)
        ]

    @staticmethod
    def _dropping_convert(rollout):
        """The historical converter: rewrites the whole response, dropping prose."""
        return [{
            "step": 0,
            "source_response": rollout["response"],
            "messages": [
                {"role": "user", "content": "do it"},
                {"role": "assistant", "content": "120 -40 0 ; +LMB -LMB"},
            ],
        }]

    @staticmethod
    def _preserving_convert(rollout):
        """The correct converter: only the action span changes."""
        target = convert_action_span(
            rollout["response"], lambda _span: "120 -40 0 ; +LMB -LMB"
        )
        return [{
            "step": 0,
            "source_response": rollout["response"],
            "messages": [
                {"role": "user", "content": "do it"},
                {"role": "assistant", "content": target},
            ],
        }]

    def test_context_dropping_converter_is_refused_and_nothing_is_written(self) -> None:
        with TemporaryDirectory(dir="/tmp") as tmp:
            tmpp = Path(tmp)
            out = tmpp / "out"
            with self.assertRaises(ContextAlteredError) as ctx:
                build_records(
                    self._rollouts(),
                    grammar="bare_line",
                    convert=self._dropping_convert,
                    out_dir=out,
                    heldout_tasks_path=self._heldout(tmpp),
                )
            self.assertIn("OUTSIDE the action span", str(ctx.exception))
            self.assertFalse((out / "_normalized").exists())

    def test_preserving_converter_passes_and_records_the_profile(self) -> None:
        with TemporaryDirectory(dir="/tmp") as tmp:
            tmpp = Path(tmp)
            out = tmpp / "out"
            report, _ = build_records(
                self._rollouts(),
                grammar="bare_line",
                convert=self._preserving_convert,
                out_dir=out,
                heldout_tasks_path=self._heldout(tmpp),
            )
            self.assertEqual(report.context_audit["n_violations"], 0)
            self.assertEqual(report.context_audit["n_records_checked"], report.n_records)
            # every record kept its preamble, and the manifest says so
            self.assertEqual(
                report.context_profile["n_with_preamble"], report.n_records
            )
            manifest = json.loads((out / "build_manifest.json").read_text())
            self.assertEqual(manifest["context_audit"]["n_violations"], 0)
            self.assertIn("context_profile", manifest)

    def test_opt_out_writes_the_reason_into_the_manifest(self) -> None:
        with TemporaryDirectory(dir="/tmp") as tmp:
            tmpp = Path(tmp)
            out = tmpp / "out"
            report, _ = build_records(
                self._rollouts(),
                grammar="bare_line",
                convert=self._dropping_convert,
                out_dir=out,
                heldout_tasks_path=self._heldout(tmpp),
                context_opt_out_reason="action-only arm requested deliberately",
            )
            manifest = json.loads((out / "build_manifest.json").read_text())
            self.assertEqual(
                manifest["context_audit"]["opt_out_reason"],
                "action-only arm requested deliberately",
            )
            self.assertGreater(manifest["context_audit"]["n_violations"], 0)


class ContextFeaturesTest(unittest.TestCase):
    def test_action_marker_alone_is_not_a_reasoning_preamble(self) -> None:
        """`Action:` is a format-independent marker, not prose; don't confuse them."""
        f = ContextFeatures.of("Action: \n120 -40 0")
        self.assertTrue(f.has_action_marker)
        self.assertFalse(f.has_reasoning_preamble)

    def test_prose_after_the_marker_is_a_preamble(self) -> None:
        f = ContextFeatures.of(REAL_ABSOLUTE)
        self.assertTrue(f.has_reasoning_preamble)
        self.assertTrue(f.has_action_marker)

    def test_action_only_target_has_neither(self) -> None:
        f = ContextFeatures.of(REAL_MOVEREL)
        self.assertFalse(f.has_reasoning_preamble)
        self.assertFalse(f.has_tools_schema)

    def test_tools_schema_is_not_counted_as_prose(self) -> None:
        f = ContextFeatures.of("<tools>{...}</tools>\n" + REAL_MOVEREL)
        self.assertTrue(f.has_tools_schema)
        self.assertFalse(f.has_reasoning_preamble)


if __name__ == "__main__":
    unittest.main()
