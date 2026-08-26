import sys
import unittest
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from bc_offline_score import MAG_FLOOR, analyze_pair, score_pairs

CLICK = "; down(LMB); up(LMB)"


def move(dx: int, dy: int, click: bool = False) -> str:
    return f"move({dx},{dy})" + (CLICK if click else "")


def s(pairs, **kw):
    return score_pairs(pairs, action_format="oev3", **kw)["scores"]


class AnalyzePairTest(unittest.TestCase):
    def test_matched_move_scores_cosine_and_relerr(self):
        r = analyze_pair(move(100, 0), move(50, 0), action_format="oev3")
        self.assertTrue(r["gold_moved"])
        self.assertTrue(r["pred_moved"])
        self.assertAlmostEqual(r["cosine"], 1.0)
        self.assertAlmostEqual(r["relerr"], 0.5)

    def test_gold_move_pred_no_move_is_cosine_zero_not_dropped(self):
        r = analyze_pair(move(100, 0), "NO_OP", action_format="oev3")
        self.assertTrue(r["gold_moved"])
        self.assertFalse(r["pred_moved"])
        self.assertEqual(r["cosine"], 0.0)
        self.assertIsNone(r["relerr"])

    def test_gold_move_pred_click_only_is_cosine_zero(self):
        r = analyze_pair(move(100, 0), "down(LMB); up(LMB)", action_format="oev3")
        self.assertEqual(r["cosine"], 0.0)

    def test_gold_move_pred_unparseable_is_cosine_zero(self):
        r = analyze_pair(move(100, 0), "wiggle the mouse", action_format="oev3")
        self.assertFalse(r["pred_valid"])
        self.assertEqual(r["cosine"], 0.0)

    def test_no_gold_move_has_no_cosine(self):
        r = analyze_pair("NO_OP", move(100, 0), action_format="oev3")
        self.assertFalse(r["gold_moved"])
        self.assertIsNone(r["cosine"])

    def test_terminate_gold_is_not_a_move(self):
        r = analyze_pair("TERMINATE", move(10, 10), action_format="oev3")
        self.assertTrue(r["gold_term"])
        self.assertFalse(r["gold_moved"])


class DenominatorTest(unittest.TestCase):
    """A checkpoint that answers half the gold moves must not look perfect."""

    def test_pred_no_move_drags_the_mean_down(self):
        pairs = [(move(100, 0), move(100, 0)), (move(100, 0), "NO_OP")]
        sc = s(pairs)
        self.assertAlmostEqual(sc["move_dir_cosine_mean"], 0.5)
        self.assertAlmostEqual(sc["move_dir_cosine_mean_matched"], 1.0)
        self.assertAlmostEqual(sc["move_coverage"], 0.5)
        self.assertEqual(sc["move_n"], 2)

    def test_denominator_is_checkpoint_independent(self):
        gold = [move(100, 0)] * 4
        eager = list(zip(gold, [move(100, 0)] * 4))
        lazy = list(zip(gold, [move(100, 0), "NO_OP", "NO_OP", "NO_OP"]))
        self.assertEqual(s(eager)["move_n"], s(lazy)["move_n"])
        self.assertAlmostEqual(s(eager)["move_coverage"], 1.0)
        self.assertAlmostEqual(s(lazy)["move_coverage"], 0.25)
        self.assertAlmostEqual(s(lazy)["move_dir_cosine_mean"], 0.25)

    def test_n_move_steps_counts_gold_not_matched(self):
        res = score_pairs(
            [(move(100, 0), move(100, 0)), (move(100, 0), "NO_OP")], action_format="oev3"
        )
        self.assertEqual(res["n_move_steps"], 2)
        self.assertEqual(res["n_move_steps_matched"], 1)


class DistributionMetricsTest(unittest.TestCase):
    def test_median_and_tail_fractions(self):
        pairs = [
            (move(100, 0), move(100, 0)),
            (move(100, 0), move(100, 5)),
            (move(100, 0), move(0, 100)),
            (move(100, 0), move(-100, 0)),
        ]
        sc = s(pairs)
        self.assertAlmostEqual(sc["move_dir_cosine_frac_above_0p9"], 0.5)
        self.assertAlmostEqual(sc["move_dir_cosine_frac_negative"], 0.25)
        self.assertAlmostEqual(sc["move_dir_cosine_median"], 0.4994, places=4)
        self.assertAlmostEqual(sc["move_dir_cosine_mean"], (1.0 + 0.99875 + 0.0 - 1.0) / 4, places=4)

    def test_mean_can_hide_a_bimodal_split(self):
        pairs = [(move(100, 0), move(100, 0))] * 5 + [(move(100, 0), move(-100, 0))] * 5
        sc = s(pairs)
        self.assertAlmostEqual(sc["move_dir_cosine_mean"], 0.0)
        self.assertAlmostEqual(sc["move_dir_cosine_frac_negative"], 0.5)
        self.assertAlmostEqual(sc["move_dir_cosine_frac_above_0p9"], 0.5)


class MagnitudeFloorTest(unittest.TestCase):
    def test_floor_drops_subtoken_gold_moves(self):
        tiny, big = MAG_FLOOR - 10, MAG_FLOOR + 10
        pairs = [
            (move(tiny, 0), move(-tiny, 0)),
            (move(big, 0), move(big, 0)),
        ]
        sc = s(pairs)
        self.assertEqual(sc["move_n"], 2)
        self.assertEqual(sc["move_big_n"], 1)
        self.assertAlmostEqual(sc["move_dir_cosine_mean"], 0.0)
        self.assertAlmostEqual(sc["move_big_dir_cosine_mean"], 1.0)

    def test_floor_is_on_euclidean_norm(self):
        pairs = [(move(40, 40), move(40, 40))]
        self.assertEqual(s(pairs)["move_big_n"], 1)
        pairs = [(move(30, 30), move(30, 30))]
        self.assertEqual(s(pairs)["move_big_n"], 0)

    def test_empty_floor_bucket_is_zero_not_crash(self):
        sc = s([(move(1, 0), move(1, 0))])
        self.assertEqual(sc["move_big_n"], 0)
        self.assertEqual(sc["move_big_dir_cosine_mean"], 0.0)
        self.assertEqual(sc["move_big_coverage"], 0.0)


class PoolSplitTest(unittest.TestCase):
    def test_pool_split_reports_each_pool(self):
        pairs = [
            (move(100, 0), move(100, 0)),
            (move(100, 0), move(100, 0)),
            (move(100, 0), move(-100, 0)),
            (move(100, 0), move(-100, 0)),
        ]
        res = score_pairs(
            pairs,
            action_format="oev3",
            pools=["success", "success", "failure", "failure"],
        )
        self.assertEqual(set(res["by_pool"]), {"success", "failure"})
        self.assertAlmostEqual(res["by_pool"]["success"]["move_dir_cosine_mean"], 1.0)
        self.assertAlmostEqual(res["by_pool"]["failure"]["move_dir_cosine_mean"], -1.0)
        self.assertAlmostEqual(res["scores"]["move_dir_cosine_mean"], 0.0)

    def test_missing_pools_are_skipped(self):
        pairs = [(move(100, 0), move(100, 0)), (move(100, 0), move(100, 0))]
        res = score_pairs(pairs, action_format="oev3", pools=[None, None])
        self.assertNotIn("by_pool", res)

    def test_partial_pools_only_split_what_is_labelled(self):
        pairs = [(move(100, 0), move(100, 0)), (move(100, 0), move(-100, 0))]
        res = score_pairs(pairs, action_format="oev3", pools=["success", None])
        self.assertEqual(set(res["by_pool"]), {"success"})
        self.assertEqual(res["by_pool"]["success"]["move_n"], 1)


class UntouchedMetricsTest(unittest.TestCase):
    def test_type_and_click_metrics_survive_the_rewrite(self):
        pairs = [
            (move(100, 0, click=True), move(90, 0, click=True)),
            ("NO_OP", "NO_OP"),
            ("TERMINATE", "TERMINATE"),
            (move(100, 0), "not an action"),
        ]
        sc = s(pairs)
        self.assertAlmostEqual(sc["format_validity_rate"], 0.75)
        self.assertAlmostEqual(sc["type_accuracy_overall"], 0.75)
        self.assertAlmostEqual(sc["click_f1"], 1.0)
        self.assertAlmostEqual(sc["terminate_f1"], 1.0)
        self.assertEqual(sc["n_pairs"], 4)

    def test_relerr_median_uses_matched_steps_only(self):
        pairs = [(move(100, 0), move(50, 0)), (move(100, 0), "NO_OP")]
        self.assertAlmostEqual(s(pairs)["move_mag_relerr_median"], 0.5)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            score_pairs([], action_format="oev3")


if __name__ == "__main__":
    unittest.main()
