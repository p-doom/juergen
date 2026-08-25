"""Goal projection: membership against actual selected frames (mid-goal
holes), snap_start policies, rejection counting, view-span -> master
conversion, and schema validation.
"""

from __future__ import annotations

import unittest

from pipeline.crowdcast.lib.goals import (
    assert_same_artifact,
    goals_by_segment,
    project_goals,
    validate_goal_row,
    view_span_to_master,
)
from pipeline.crowdcast.lib.views import build_segment_view


def _goal(start: int, end: int, *, goal_id: str = "g0", segment_id: str = "s0", **extra) -> dict:
    return {
        "goal_id": goal_id,
        "segment_id": segment_id,
        "recording_id": "r0",
        "start_master_idx": start,
        "end_master_idx": end,
        "instruction": "do the thing",
        "method": "describe_extract",
        "model": "test-model",
        "prompt_pack_sha": "deadbeef",
        **extra,
    }


def _view(kept_ranges, dropped=None, n_records=150, fps=1.0):
    return build_segment_view(
        {
            "segment_id": "s0",
            "recording_id": "r0",
            "segment_idx": 0,
            "master_fps": 15.0,
            "n_master_records": n_records,
            "shard_path": "/nowhere/frames/s0/images.array_record",
            "keylog_path": None,
            "alignment_status": "aligned",
            "kept_ranges": kept_ranges,
            "dropped": dropped or [],
        },
        fps=fps,
    )


class ProjectGoalsTest(unittest.TestCase):
    def test_membership_by_actual_frames_not_fps_arithmetic(self) -> None:
        # Slot 4 (tick 60) is masked: a goal spanning [60, 105) starts inside
        # the hole — fps arithmetic would claim a frame at 60; the projection
        # must use the frames that actually exist (75, 90) plus the snapped
        # observation frame at 45.
        view = _view([[0, 60], [75, 150]], [{"start": 60, "end": 75, "reason": "black"}])
        projections, stats = project_goals([_goal(60, 105)], view)
        self.assertEqual(stats.n_projected, 1)
        [p] = projections
        self.assertEqual([f.master_idx for f in p.frames], [45, 75, 90])
        self.assertTrue(p.snapped_start)
        self.assertEqual(stats.n_snapped, 1)

    def test_mid_goal_holes_are_skipped_not_substituted(self) -> None:
        view = _view([[0, 60], [75, 150]], [{"start": 60, "end": 75, "reason": "black"}])
        projections, _stats = project_goals([_goal(30, 120)], view)
        [p] = projections
        # tick 60 masked: no frame between 45 and 75.
        self.assertEqual([f.master_idx for f in p.frames], [30, 45, 75, 90, 105])
        self.assertFalse(p.snapped_start)

    def test_snap_start_inside_excludes_prior_frame(self) -> None:
        view = _view([[0, 60], [75, 150]], [{"start": 60, "end": 75, "reason": "black"}])
        projections, _ = project_goals([_goal(60, 105)], view, snap_start="inside")
        [p] = projections
        self.assertEqual([f.master_idx for f in p.frames], [75, 90])
        self.assertFalse(p.snapped_start)

    def test_snap_start_noop_when_frame_at_goal_start(self) -> None:
        view = _view([[0, 150]])
        projections, stats = project_goals([_goal(45, 90)], view)
        [p] = projections
        self.assertEqual([f.master_idx for f in p.frames], [45, 60, 75])
        self.assertFalse(p.snapped_start)
        self.assertEqual(stats.n_snapped, 0)

    def test_empty_projection_counted(self) -> None:
        # Goal entirely inside the masked hole: no frames, rejected, counted.
        view = _view([[0, 60], [75, 150]], [{"start": 60, "end": 75, "reason": "black"}])
        projections, stats = project_goals([_goal(61, 74)], view)
        self.assertEqual(projections, [])
        self.assertEqual(stats.n_empty_projection, 1)
        self.assertEqual(stats.rejected, [{"goal_id": "g0", "reason": "empty_projection"}])

    def test_too_few_frames_counted(self) -> None:
        view = _view([[0, 150]])
        projections, stats = project_goals([_goal(45, 60)], view, min_frames=3, snap_start="inside")
        self.assertEqual(projections, [])
        self.assertEqual(stats.n_too_few_frames, 1)

    def test_wrong_segment_refused(self) -> None:
        view = _view([[0, 150]])
        with self.assertRaises(ValueError):
            project_goals([_goal(0, 30, segment_id="OTHER")], view)

    def test_bad_snap_mode_refused(self) -> None:
        with self.assertRaises(ValueError):
            project_goals([], _view([[0, 150]]), snap_start="nearest")


class ViewSpanToMasterTest(unittest.TestCase):
    def test_span_uses_window_boundaries(self) -> None:
        # Frames at 0,15,30,45,75,... (60 masked). Span [3,5) = frames 45,75:
        # starts at tick 45, ends at frame-75's window end (90).
        view = _view([[0, 60], [75, 150]], [{"start": 60, "end": 75, "reason": "black"}])
        self.assertEqual(view_span_to_master(view, 3, 5), (45, 90))
        # Full span reaches the axis end via the last window.
        n = len(view.frames)
        self.assertEqual(view_span_to_master(view, 0, n), (0, 150))

    def test_out_of_range_refused(self) -> None:
        view = _view([[0, 150]])
        n = len(view.frames)
        for bad in ((0, 0), (-1, 2), (2, 1), (0, n + 1)):
            with self.assertRaises(ValueError):
                view_span_to_master(view, *bad)

    def test_roundtrip_projection(self) -> None:
        # A span converted to master and projected back onto the SAME view
        # yields exactly the original frames (no snap needed: starts on one).
        view = _view([[0, 60], [75, 150]], [{"start": 60, "end": 75, "reason": "black"}])
        start, end = view_span_to_master(view, 2, 6)
        projections, _ = project_goals([_goal(start, end)], view)
        [p] = projections
        self.assertEqual(
            [f.view_idx for f in p.frames], [2, 3, 4, 5], f"master span [{start},{end})"
        )


class SchemaTest(unittest.TestCase):
    def test_valid_row_passes(self) -> None:
        validate_goal_row(_goal(0, 30, instruction_variants=["a", "b"]))

    def test_violations_raise(self) -> None:
        bad_rows = [
            {k: v for k, v in _goal(0, 30).items() if k != "prompt_pack_sha"},  # missing key
            _goal(30, 30),  # empty interval
            _goal(-1, 30),  # negative start
            _goal(0, 30) | {"start_master_idx": 0.5},  # float coordinate
            _goal(0, 30) | {"instruction": "  "},  # blank instruction
            _goal(0, 30) | {"instruction_variants": "not-a-list"},
        ]
        for row in bad_rows:
            with self.assertRaises(ValueError, msg=row):
                validate_goal_row(row)

    def test_goals_by_segment_sorts(self) -> None:
        rows = [
            _goal(60, 90, goal_id="g2"),
            _goal(0, 30, goal_id="g1"),
            _goal(0, 15, goal_id="g0", segment_id="s1"),
        ]
        grouped = goals_by_segment(rows)
        self.assertEqual([g["goal_id"] for g in grouped["s0"]], ["g1", "g2"])
        self.assertEqual([g["goal_id"] for g in grouped["s1"]], ["g0"])

    def test_assert_same_artifact(self) -> None:
        assert_same_artifact("x::1", "x::1", what="filter")
        with self.assertRaises(ValueError):
            assert_same_artifact("x::1", "x::2", what="filter")


if __name__ == "__main__":
    unittest.main()
