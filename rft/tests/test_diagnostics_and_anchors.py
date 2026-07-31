"""Metric-validation gates: the three mandatory diagnostics, and the anchors.

This is the file that enforces the standing rule — **every metric is validated
against a reference whose answer is already known, and that validation is an
automated test.** Defect #2 happened because that check was skipped: scoring the
off-the-shelf model through the same broken reader would have shown 0 too.

The magnitude-ratio tests run against the REAL gold/pred pairs from the
teacher-forcing isolation decomposition (``tests/fixtures/pairs_iso_*.jsonl``,
200 pairs per format) and reproduce the recorded ``decomp_iso_*.json`` readings
exactly. Those pairs are committed here because the lattice analysis over them was
done ad-hoc and **lost**; keeping the raw material in-repo is why it cannot be lost
again.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from rft import anchors
from rft.diagnostics import LATTICE_VALUES, delta_diagnostics, deltas_from_completions
from rft.errors import AnchorMismatch, SchemaError

FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: fixture format -> (pairs file, recorded reading, rft grammar name)
ISO_CASES = {
    "diffabs": ("pairs_iso_diffabs.jsonl", "decomp_iso_diffabs.json", "bare_line"),
    "diffabsnorm": ("pairs_iso_diffabsnorm.jsonl", "decomp_iso_diffabsnorm.json", "bare_line"),
    "moverel": ("pairs_iso_moverel.jsonl", "decomp_iso_moverel.json",
                "computer_use_move_rel"),
}


def _load_pairs(name: str) -> list[tuple[str, str]]:
    path = FIXTURES / name
    out: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        out.append((obj["gold"], obj["pred"]))
    return out


class MandatoryDiagnosticsTest(unittest.TestCase):
    """The three numbers are part of the output contract, not a bespoke script."""

    def test_collapsed_policy_is_visible_in_distinct_delta_count(self) -> None:
        collapsed = [(10.0, 0.0)] * 200
        d = delta_diagnostics(collapsed)
        self.assertEqual(d.n_distinct_deltas, 1)
        self.assertEqual(d.most_common[0], ((10.0, 0.0), 200))

    def test_healthy_policy_has_many_distinct_deltas(self) -> None:
        varied = [(float(i), float(-i)) for i in range(1, 201)]
        self.assertEqual(delta_diagnostics(varied).n_distinct_deltas, 200)

    def test_lattice_fraction_catches_magnitude_quantisation(self) -> None:
        quantised = [(x, y) for x in (0, 1, -1, 10, -10, 100, -100) for y in (0, 10, -100)]
        self.assertAlmostEqual(delta_diagnostics(quantised).lattice_fraction, 1.0)
        continuous = [(float(i), float(i * 3 + 7)) for i in range(1, 60)]
        self.assertLess(delta_diagnostics(continuous).lattice_fraction, 0.1)

    def test_lattice_values_are_the_documented_set(self) -> None:
        self.assertEqual(LATTICE_VALUES, frozenset({0, 1, -1, 10, -10, 100, -100}))

    def test_magnitude_ratio_detects_compression(self) -> None:
        golds = [(100.0, 0.0)] * 50
        preds = [(10.0, 0.0)] * 50
        d = delta_diagnostics(preds, golds)
        self.assertAlmostEqual(d.median_magnitude_ratio, 0.1)

    def test_zero_gold_is_skipped_and_reported_not_divided_by(self) -> None:
        d = delta_diagnostics([(5.0, 0.0), (5.0, 0.0)], [(0.0, 0.0), (10.0, 0.0)])
        self.assertEqual(d.n_zero_gold_skipped, 1)
        self.assertEqual(d.n_gold_pairs, 1)
        self.assertAlmostEqual(d.median_magnitude_ratio, 0.5)

    def test_no_gold_yields_none_not_zero(self) -> None:
        d = delta_diagnostics([(1.0, 1.0)])
        self.assertIsNone(d.median_magnitude_ratio)
        self.assertIn("n/a", d.describe())

    def test_empty_predictions_have_no_diagnostics(self) -> None:
        with self.assertRaises(SchemaError):
            delta_diagnostics([])

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(SchemaError):
            delta_diagnostics([(1.0, 1.0)], [(1.0, 1.0), (2.0, 2.0)])

    def test_describe_carries_all_three_numbers(self) -> None:
        text = delta_diagnostics([(1.0, 2.0), (10.0, 0.0)], [(2.0, 4.0), (10.0, 0.0)]).describe()
        for token in ("distinct_deltas", "lattice_fraction", "median|pred|/|gold|"):
            self.assertIn(token, text)


class ExcludingZeroPredHidesCollapseTest(unittest.TestCase):
    """The convention that hid the magnitude-encoding collapse.

    ``tf_decomp_iso.py`` computed the ratio only over pairs where BOTH gold and
    pred were non-zero (``if gn>0 and pn>0``). A policy that has collapsed to
    emitting ``(0,0)`` is therefore *absent from its own magnitude ratio*, which
    then reads a healthy ~1.0. Our default includes zero predictions, and reports
    how many there were.
    """

    def test_the_exclusion_makes_a_collapsed_policy_look_healthy(self) -> None:
        golds = [(100.0, 0.0)] * 100
        # 80% of predictions collapsed to a zero delta; 20% are perfect.
        preds = [(0.0, 0.0)] * 80 + [(100.0, 0.0)] * 20

        hidden = delta_diagnostics(preds, golds, exclude_zero_pred=True)
        honest = delta_diagnostics(preds, golds, exclude_zero_pred=False)

        self.assertAlmostEqual(hidden.median_magnitude_ratio, 1.0)
        self.assertAlmostEqual(honest.median_magnitude_ratio, 0.0)
        self.assertEqual(hidden.n_zero_pred, 80)
        self.assertEqual(honest.n_zero_pred, 80)
        self.assertIn("EXCLUDED (hides collapse)", hidden.describe())
        self.assertIn("80 zero-pred included", honest.describe())

    def test_default_is_the_honest_convention(self) -> None:
        d = delta_diagnostics([(0.0, 0.0)], [(10.0, 0.0)])
        self.assertFalse(d.zero_pred_excluded)
        self.assertAlmostEqual(d.median_magnitude_ratio, 0.0)

    def test_median_convention_is_explicit(self) -> None:
        values_preds = [(1.0, 0.0), (3.0, 0.0)]
        golds = [(1.0, 0.0), (1.0, 0.0)]
        avg = delta_diagnostics(values_preds, golds, median_kind="average")
        upper = delta_diagnostics(values_preds, golds, median_kind="upper")
        self.assertAlmostEqual(avg.median_magnitude_ratio, 2.0)
        self.assertAlmostEqual(upper.median_magnitude_ratio, 3.0)

    def test_unknown_median_kind_raises(self) -> None:
        with self.assertRaises(SchemaError):
            delta_diagnostics([(1.0, 0.0)], [(1.0, 0.0)], median_kind="mode")


class RealPairsReproduceRecordedReadingsTest(unittest.TestCase):
    """Validate the magnitude-ratio reader against readings whose answer is known.

    For each of the three formats, recompute ``magratio_median`` from the raw pairs
    and require it to match ``decomp_iso_<fmt>.json`` — using the reference's own
    conventions (upper median, zero-pred excluded). This is the "score the known
    reference through THIS reader first" rule as a test.
    """

    def _deltas(self, fmt: str) -> tuple[list, list, int]:
        pairs_file, _, grammar = ISO_CASES[fmt]
        from rft.grammars import get_grammar

        if not get_grammar(grammar, require_available=False).available:
            self.skipTest(f"grammar {grammar} unavailable in this eval/action_parser.py")
        pairs = _load_pairs(pairs_file)
        golds, preds = [], []
        n_invalid = 0
        for gold, pred in pairs:
            g, _ = deltas_from_completions([gold], grammar=grammar, skip_unparseable=True)
            p, _ = deltas_from_completions([pred], grammar=grammar, skip_unparseable=True)
            if not g:
                continue
            if not p:
                n_invalid += 1
                continue
            golds.append(g[0])
            preds.append(p[0])
        return preds, golds, n_invalid

    def test_reproduces_recorded_magratio_median(self) -> None:
        for fmt, (_, recorded_file, _) in ISO_CASES.items():
            with self.subTest(fmt=fmt):
                recorded = json.loads((FIXTURES / recorded_file).read_text())
                preds, golds, _ = self._deltas(fmt)
                self.assertGreater(len(preds), 50, "too few usable pairs to validate")
                d = delta_diagnostics(
                    preds, golds, exclude_zero_pred=True, median_kind="upper"
                )
                self.assertIsNotNone(d.median_magnitude_ratio)
                self.assertAlmostEqual(
                    d.median_magnitude_ratio,
                    recorded["magratio_median"],
                    places=3,
                    msg=(
                        f"{fmt}: recomputed {d.median_magnitude_ratio} != recorded "
                        f"{recorded['magratio_median']}. Either the reader changed or the "
                        "fixture did - find out which before trusting any new reading."
                    ),
                )

    def test_recorded_readings_are_the_expected_shape(self) -> None:
        """Pin the reference values so a fixture swap is visible."""
        expected = {"diffabs": 1.0, "diffabsnorm": 0.8417346698668298,
                    "moverel": 0.9993239532760837}
        for fmt, value in expected.items():
            recorded = json.loads((FIXTURES / ISO_CASES[fmt][1]).read_text())
            self.assertAlmostEqual(recorded["magratio_median"], value, places=9, msg=fmt)

    def test_lattice_fraction_over_the_real_pairs_is_now_computable(self) -> None:
        """The analysis that was LOST. It must be a one-call diagnostic forever."""
        results: dict[str, float] = {}
        for fmt in ISO_CASES:
            preds, golds, _ = self._deltas(fmt)
            d = delta_diagnostics(preds, golds)
            results[fmt] = d.lattice_fraction
            # A real 1920x1080 pointer delta distribution is nowhere near the
            # {0,+-1,+-10,+-100} lattice; a high value here means quantisation.
            self.assertGreaterEqual(d.lattice_fraction, 0.0)
            self.assertLessEqual(d.lattice_fraction, 1.0)
            self.assertGreater(d.n_distinct_deltas, 1, f"{fmt} looks collapsed")
        self.assertEqual(set(results), set(ISO_CASES))


class AnchorTest(unittest.TestCase):
    """The anchors themselves: bands, exclusions, and the "was it checked" guard."""

    def test_every_anchor_has_a_usable_band(self) -> None:
        for name, anchor in anchors.ANCHORS.items():
            band = anchor.band()
            self.assertLessEqual(band.low, band.high, name)
            self.assertTrue(band.contains(anchor.value), f"{name} value outside its own band")

    def test_offshelf_8b_anchor_pins_the_reward_sum_not_a_percentage(self) -> None:
        """The 30.85 / 31.14 ambiguity is one run over two denominators."""
        s = anchors.OFFSHELF_8B_HELDOUT_REWARD_SUM
        self.assertAlmostEqual(s / 107, 0.30845, places=4)
        self.assertAlmostEqual(s / 106, 0.31136, places=4)
        self.assertEqual(anchors.OFFSHELF_8B_HELDOUT_N_SOLVED, 31)
        self.assertEqual(anchors.OFFSHELF_8B_HELDOUT_N_NAN, 1)

    def test_partial_credit_exists_so_binary_scoring_is_wrong(self) -> None:
        self.assertTrue(
            all(0.0 < p < 1.0 for p in anchors.OFFSHELF_8B_HELDOUT_PARTIALS),
            "the anchor run has partial credit; reward == 1.0 is the wrong success test",
        )

    def test_4b_anchor_records_both_denominators(self) -> None:
        over_done = anchors.OFFSHELF_4B_RATE_OVER_DONE
        over_total = over_done * anchors.OFFSHELF_4B_N_DONE / anchors.OFFSHELF_4B_N_TOTAL
        self.assertAlmostEqual(over_done, 0.2684773, places=6)
        self.assertAlmostEqual(over_total, 0.2627, places=3)
        # and it must sit within touching distance of the published number
        self.assertLess(abs(over_total - anchors.OFFSHELF_4B_PUBLISHED), 0.01)

    def test_absolute_and_relative_grounding_gap_is_in_the_same_run(self) -> None:
        """An absolute arm at 0.97 and a relative arm at 0.06 in ONE run cannot both
        be a harness artifact - that is what makes the gap a finding."""
        abs_a = anchors.get_anchor("absolute_single_step_grounding")
        rel_a = anchors.get_anchor("relative_moverel_single_step_grounding")
        self.assertEqual(abs_a.n_scored, rel_a.n_scored)
        self.assertGreater(abs_a.value - rel_a.value, 0.8)

    def test_anchor_check_accepts_a_matching_reading(self) -> None:
        anchors.check_anchor("offshelf_8b_closed_loop", 0.3084)

    def test_anchor_check_rejects_a_wrong_reading(self) -> None:
        with self.assertRaises(AnchorMismatch):
            anchors.check_anchor("offshelf_8b_closed_loop", 0.01)

    def test_unknown_anchor_lists_the_known_ones(self) -> None:
        with self.assertRaises(SchemaError) as ctx:
            anchors.get_anchor("made_up")
        self.assertIn("offshelf_8b_closed_loop", str(ctx.exception))

    def test_reporting_without_a_reference_check_is_refused(self) -> None:
        """Defect #2 as a guard: a metric reported unvalidated must raise."""
        with self.assertRaises(AnchorMismatch) as ctx:
            anchors.assert_reference_check_ran(
                metric_name="full_completion_rate",
                reference_reading=None,
                anchor_name="offshelf_8b_closed_loop",
            )
        self.assertIn("without ever having been run", str(ctx.exception))

    def test_reference_check_passes_when_the_reader_is_sound(self) -> None:
        anchors.assert_reference_check_ran(
            metric_name="full_completion_rate",
            reference_reading=0.3084,
            anchor_name="offshelf_8b_closed_loop",
        )

    def test_a_broken_reader_fails_the_reference_check(self) -> None:
        """The exact defect-#2 scenario: the reader returns 0 for everything, so it
        returns 0 for the KNOWN-GOOD model too, and the check catches it."""
        with self.assertRaises(AnchorMismatch):
            anchors.assert_reference_check_ran(
                metric_name="counts_a_nonexistent_success_field",
                reference_reading=0.0,
                anchor_name="offshelf_8b_closed_loop",
            )

    def test_describe_all_mentions_provenance(self) -> None:
        text = anchors.describe_all()
        self.assertIn("offshelf_8b_closed_loop", text)
        self.assertIn("unscored excluded", text)

    def test_anchor_rejects_a_non_proportion(self) -> None:
        with self.assertRaises(SchemaError):
            anchors.Anchor(name="x", value=1.5, n_scored=10, provenance="p")

    def test_anchor_rejects_an_empty_denominator(self) -> None:
        with self.assertRaises(SchemaError):
            anchors.Anchor(name="x", value=0.5, n_scored=0, provenance="p")


class DeltasFromCompletionsTest(unittest.TestCase):
    def test_unparseable_raises_by_default(self) -> None:
        with self.assertRaises(Exception):
            deltas_from_completions(["not an action"], grammar="bare_line")

    def test_errors_are_returned_not_dropped_when_skipping(self) -> None:
        deltas, errors = deltas_from_completions(
            ["10 20 0", "garbage"], grammar="bare_line", skip_unparseable=True
        )
        self.assertEqual(deltas, [(10.0, 20.0)])
        self.assertEqual(len(errors), 1)
        self.assertIn("completion[1]", errors[0])

    def test_no_op_has_no_relative_delta(self) -> None:
        deltas, errors = deltas_from_completions(
            ["NO_OP"], grammar="bare_line", skip_unparseable=True
        )
        self.assertEqual(deltas, [])
        self.assertEqual(len(errors), 1)

    def test_nan_never_appears_in_a_delta(self) -> None:
        deltas, _ = deltas_from_completions(["10 -20 0"], grammar="bare_line")
        for dx, dy in deltas:
            self.assertFalse(math.isnan(dx) or math.isnan(dy))


if __name__ == "__main__":
    unittest.main()
