import sys
import unittest
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(EVAL_DIR.parent) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR.parent))

import cursor_probe as cp

SW, SH = 1920, 1080


def fake_sprite():
    alpha = np.zeros((cp.SPR_H, cp.SPR_W), np.float32)
    alpha[cp.HOT_Y : cp.HOT_Y + 12, cp.HOT_X : cp.HOT_X + 8] = 1.0
    rgb = np.zeros((cp.SPR_H, cp.SPR_W, 3), np.float32)
    rgb[..., :] = 250.0
    rgb[cp.HOT_Y : cp.HOT_Y + 12, cp.HOT_X] = 0.0
    return rgb, alpha


def scene(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 200, (SH, SW, 3)).astype(np.float32)


def row(shift_px, pred_a, pred_b, screen=(SW, SH)):
    return {
        "screen": list(screen),
        "shift_px": list(shift_px),
        "pred_a": pred_a,
        "pred_b": pred_b,
    }


class CompositeTest(unittest.TestCase):
    def test_composite_writes_only_under_the_alpha(self):
        rgb, alpha = fake_sprite()
        img = scene()
        before = img.copy()
        cp.composite(img, rgb, alpha, 500, 400)
        changed = np.abs(img - before).sum(axis=2) > 0
        self.assertEqual(int(changed.sum()), int((alpha > 0).sum()))
        ys, xs = np.nonzero(changed)
        self.assertEqual((ys.min(), xs.min()), (400, 500))

    def test_opaque_pixels_take_the_sprite_colour(self):
        rgb, alpha = fake_sprite()
        img = scene()
        cp.composite(img, rgb, alpha, 300, 300)
        self.assertTrue(np.allclose(img[300, 300], rgb[cp.HOT_Y, cp.HOT_X]))


class BoundsTest(unittest.TestCase):
    def test_hotspot_needs_room_for_sprite_and_halo(self):
        self.assertTrue(cp._in_bounds(500, 500, SW, SH))
        self.assertFalse(cp._in_bounds(5, 500, SW, SH))
        self.assertFalse(cp._in_bounds(500, SH - 5, SW, SH))

    def test_sampled_shift_stays_in_bounds_and_long_enough(self):
        import random

        rng = random.Random(7)
        for cx, cy in ((100, 100), (960, 540), (SW - 100, SH - 100)):
            for _ in range(30):
                s = cp.sample_shift(rng, cx, cy, SW, SH, 200, 450)
                self.assertIsNotNone(s)
                self.assertTrue(cp._in_bounds(cx + s[0], cy + s[1], SW, SH))
                self.assertGreaterEqual(np.hypot(*s), 200)


class BuildPairTest(unittest.TestCase):
    """The A/B pair has to differ in exactly two places: C and C'."""

    def setUp(self):
        self.rgb, self.alpha = fake_sprite()
        self.masks = cp.sprite_masks(self.alpha)
        self.thr = {
            "bg_tol": 1.5, "bg_hit_tol": 0.002, "core_min": 8.0,
            "prev_core_min": 30.0, "arrow_tol": 15.0,
        }
        self.prev = scene(1)
        self.cx, self.cy = 600, 500

    def _cur(self, cursor_at):
        cur = self.prev.copy()
        cp.composite(cur, self.rgb, self.alpha, *cursor_at)
        return cur

    def test_clean_case_moves_the_cursor_and_nothing_else(self):
        cur = self._cur((self.cx, self.cy))
        nx, ny = self.cx + 300, self.cy + 120
        a, b, diag = cp.build_pair(
            cur, self.prev, self.rgb, self.alpha, self.masks,
            self.cx, self.cy, nx, ny, self.thr,
        )
        self.assertIsNotNone(a, diag)
        self.assertTrue(np.allclose(a, cur))
        diff = np.abs(a - b).sum(axis=2) > 0
        ys, xs = np.nonzero(diff)
        self.assertEqual((xs.min(), ys.min()), (self.cx, self.cy))
        self.assertEqual((xs.max(), ys.max()), (nx + 7, ny + 11))
        self.assertTrue(np.allclose(b[ny, nx], self.rgb[cp.HOT_Y, cp.HOT_X]))
        self.assertTrue(np.allclose(b[self.cy, self.cx], self.prev[self.cy, self.cx]))

    def test_rejects_when_the_background_moved(self):
        cur = self._cur((self.cx, self.cy))
        cur[self.cy - 20 : self.cy + 20, self.cx + 30 : self.cx + 70] = 0.0
        a, _, diag = cp.build_pair(
            cur, self.prev, self.rgb, self.alpha, self.masks,
            self.cx, self.cy, self.cx + 300, self.cy, self.thr,
        )
        self.assertIsNone(a)
        self.assertGreater(diag["bg_err"], self.thr["bg_tol"])

    def test_rejects_a_stray_pointer_the_mean_would_dilute(self):
        cur = self._cur((self.cx, self.cy))
        prev = self.prev.copy()
        cp.composite(prev, self.rgb, self.alpha, self.cx + 14, self.cy + 6)
        a, _, diag = cp.build_pair(
            cur, prev, self.rgb, self.alpha, self.masks,
            self.cx, self.cy, self.cx + 300, self.cy, self.thr,
        )
        self.assertIsNone(a)

    def test_rejects_when_the_cursor_barely_moved(self):
        cur = self._cur((self.cx, self.cy))
        prev = self.prev.copy()
        cp.composite(prev, self.rgb, self.alpha, self.cx + 1, self.cy + 1)
        a, _, diag = cp.build_pair(
            cur, prev, self.rgb, self.alpha, self.masks,
            self.cx, self.cy, self.cx + 300, self.cy, self.thr,
        )
        self.assertIsNone(a)
        self.assertLess(diag["prev_core_err"], self.thr["prev_core_min"])

    def test_rejects_a_cursor_that_is_not_the_arrow(self):
        other_rgb, other_alpha = fake_sprite()
        other_alpha[:] = 0.0
        other_alpha[cp.HOT_Y : cp.HOT_Y + 12, cp.HOT_X + 4 : cp.HOT_X + 12] = 1.0
        cur = self.prev.copy()
        cp.composite(cur, other_rgb, other_alpha, self.cx, self.cy)
        a, _, diag = cp.build_pair(
            cur, self.prev, self.rgb, self.alpha, self.masks,
            self.cx, self.cy, self.cx + 300, self.cy, self.thr,
        )
        self.assertIsNone(a)
        self.assertGreater(diag["arrow_err"], self.thr["arrow_tol"])


class NetMoveTest(unittest.TestCase):
    def test_sums_the_move_primitives(self):
        self.assertEqual(cp.net_move("move(10,20); move(5,-5); down(LMB); up(LMB)"), (15, 15))

    def test_non_moves_are_none(self):
        for line in ("NO_OP", "TERMINATE", "", "down(LMB); up(LMB)", "gibberish"):
            self.assertIsNone(cp.net_move(line), line)


class ScoreRowsTest(unittest.TestCase):
    def test_a_perfect_cursor_reader_scores_slope_one(self):
        rows = []
        for sx, sy in ((384, 0), (-384, 108), (192, -216)):
            ex, ey = round(-sx / SW * cp.GRID), round(-sy / SH * cp.GRID)
            rows.append(row((sx, sy), "move(100,100)", f"move({100 + ex},{100 + ey})"))
        res = cp.score_rows(rows)
        self.assertAlmostEqual(res["pooled"]["slope"], 1.0, places=2)
        self.assertAlmostEqual(res["pooled"]["r2"], 1.0, places=2)
        self.assertEqual(res["n_both_move"], 3)

    def test_a_model_ignoring_the_cursor_scores_slope_zero(self):
        rows = [row((384, 0), "move(100,100)", "move(100,100)") for _ in range(5)]
        res = cp.score_rows(rows)
        self.assertAlmostEqual(res["pooled"]["slope"], 0.0)
        self.assertAlmostEqual(res["prediction_changed_rate"], 0.0)

    def test_half_compensation_scores_half(self):
        rows = []
        for sx in (384, -384, 192):
            ex = -sx / SW * cp.GRID
            rows.append(row((sx, 0), "move(100,0)", f"move({round(100 + ex / 2)},0)"))
        self.assertAlmostEqual(cp.score_rows(rows)["axis_x"]["slope"], 0.5, places=2)

    def test_axes_are_scaled_by_their_own_screen_dimension(self):
        rows = [row((0, 108), "move(3,0)", "move(3,-100)")]
        self.assertAlmostEqual(cp.score_rows(rows)["axis_y"]["slope"], 1.0, places=6)

    def test_rows_without_a_move_in_both_conditions_are_excluded(self):
        rows = [
            row((384, 0), "move(100,0)", "NO_OP"),
            row((384, 0), "move(100,0)", "move(-100,0)"),
        ]
        res = cp.score_rows(rows)
        self.assertEqual(res["n_both_move"], 1)
        self.assertAlmostEqual(res["move_pair_rate"], 0.5)

    def test_robust_slope_survives_one_wild_prediction(self):
        rows = []
        for sx in (384, -384, 192, -192, 288, -288):
            ex = round(-sx / SW * cp.GRID)
            rows.append(row((sx, 0), "move(100,0)", f"move({100 + ex},0)"))
        rows.append(row((192, 0), "move(100,0)", "move(9000,0)"))
        res = cp.score_rows(rows)
        self.assertAlmostEqual(res["axis_x"]["slope_robust"], 1.0, places=2)
        self.assertLess(res["axis_x"]["slope"], 0.0)

    def test_subgroups_split_by_pool_and_by_aim_quality(self):
        SX = 384
        ex = round(-SX / SW * cp.GRID)
        good = row((SX, 0), "move(100,0)", f"move({100 + ex},0)")
        good.update(pool="success", gold="move(80,0)")
        bad = row((SX, 0), "move(-100,0)", "move(-100,0)")
        bad.update(pool="failure", gold="move(80,0)")
        res = cp.score_rows([good, bad])
        self.assertAlmostEqual(res["by_pool"]["success"]["slope"], 1.0, places=2)
        self.assertAlmostEqual(res["by_pool"]["failure"]["slope"], 0.0)
        self.assertAlmostEqual(res["gold_aligned"]["slope"], 1.0, places=2)
        self.assertAlmostEqual(res["gold_misaligned"]["slope"], 0.0)

    def test_empty_rows_do_not_crash(self):
        res = cp.score_rows([])
        self.assertEqual(res["n_both_move"], 0)
        self.assertEqual(res["pooled"]["slope"], 0.0)


if __name__ == "__main__":
    unittest.main()
