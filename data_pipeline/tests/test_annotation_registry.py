"""Annotation package: method discovery, prompt packs (+sha, snapshot), unit
chunking (submission-snapped cuts, tail buffer), view-local -> master
conversion of a method result, and the plans quality flags. No labeler calls.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from realigned_pipeline.annotation.lib.prompts import PromptPack
from realigned_pipeline.annotation.lib.registry import discover_methods, load_method
from realigned_pipeline.annotation.lib.units import AnnotationUnit, build_units, plan_windows
from realigned_pipeline.annotation.methods.describe_extract.annotator import (
    clean_goals,
    snap_goal_starts,
)
from realigned_pipeline.annotation.methods.plans.annotator import goal_start_frame, plan_flags
from realigned_pipeline.lib.goals import validate_goal_row, view_span_to_master
from realigned_pipeline.lib.views import build_segment_view


def _view(n_records: int = 150, fps: float = 1.0):
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
            "kept_ranges": [[0, n_records]],
            "dropped": [],
        },
        fps=fps,
    )


class RegistryTest(unittest.TestCase):
    def test_discovery_and_kinds(self) -> None:
        methods = discover_methods()
        self.assertIn("describe_extract", methods)
        self.assertIn("plans", methods)
        self.assertEqual(load_method("describe_extract").input_kind, "frames")
        self.assertEqual(load_method("plans").input_kind, "goals")

    def test_unknown_method_refused(self) -> None:
        with self.assertRaises(KeyError):
            load_method("nope")

    def test_prompt_pack_sha_and_snapshot(self) -> None:
        m = load_method("describe_extract")
        self.assertEqual(len(m.prompts.sha), 16)
        # Placeholders resolve; JSON braces pass through untouched.
        rendered = m.prompts.render("describe_prose", n_frames=42, frame_period_s="2")
        self.assertIn("42 frames attached", rendered)
        self.assertNotIn("${n_frames}", rendered)
        extract = m.prompts.render("extract", description="D", n_frames=3, frame_period_s="2")
        self.assertIn('"goals": [', extract)
        with tempfile.TemporaryDirectory() as tmp:
            snap = m.prompts.snapshot_to(Path(tmp))
            self.assertEqual(PromptPack(snap).sha, m.prompts.sha)


class UnitTest(unittest.TestCase):
    def test_single_window_keeps_segment_id(self) -> None:
        view = _view()
        # max_frames_per_window set explicitly: no image decode needed.
        units = build_units(view, ["NO_OP"] * len(view.frames),
                            context_limit=10_000_000, completion_reserve=32000,
                            safety_margin=28000, max_frames_per_window=1000)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].unit_id, "s0")
        self.assertEqual(units[0].tail_buffer, 0)

    def test_split_units_snap_to_submission_and_get_tail_buffer(self) -> None:
        view = _view()
        n = len(view.frames)  # 10 frames
        actions = ["1 0 0 ; +KeyA -KeyA"] * n
        actions[6] = "0 0 0 ; +Return -Return"  # submission: ideal cut after it
        units = build_units(view, actions,
                            context_limit=10_000_000, completion_reserve=32000,
                            safety_margin=28000, max_frames_per_window=8,
                            snap_slack=3, tail_buffer=2)
        self.assertEqual([u.unit_id for u in units], ["s0__w0", "s0__w1"])
        self.assertEqual(units[0].hi, 7)  # cut BEFORE frame 7 (after the Return)
        self.assertEqual(units[0].tail_buffer, 2)
        self.assertEqual(units[1].tail_buffer, 0)
        # Sent frames include the buffer; owned range excludes it.
        self.assertEqual(units[0].sent_view_indices, list(range(0, 9)))
        self.assertEqual(units[0].owned_hi_view_idx, 6)

    def test_plan_windows_mid_burst_avoided(self) -> None:
        # All frames mid typing-burst except an idle pair at 8/9: the cut moves
        # off the ideal boundary (10, cost 3: both sides typing) to 9 (cost 1:
        # neither side typing), within the ±3 slack.
        actions = ["0 0 0 ; +KeyA"] * 20
        actions[8] = actions[9] = "NO_OP"
        wins = plan_windows(20, 12, actions=actions, times=[float(i) for i in range(20)],
                            slack=3)
        self.assertEqual(wins, [(0, 9), (9, 20)])


class ViewLocalConversionTest(unittest.TestCase):
    def test_clean_goals_clamps_and_drops_buffer_starts(self) -> None:
        parsed = {"goals": [
            {"instruction": "do a", "start_frame": 2, "end_frame": 5},
            {"instruction": "in buffer", "start_frame": 8, "end_frame": 9},  # past own_hi
            {"instruction": "swapped", "start_frame": 6, "end_frame": 4},
            {"instruction": ""},  # empty: dropped
            {"instruction": "unbounded"},
        ]}
        goals = clean_goals(parsed, frame_lo=0, frame_hi=9, own_hi=7)
        self.assertEqual([g["instruction"] for g in goals], ["do a", "swapped", "unbounded"])
        self.assertEqual((goals[1]["start_frame"], goals[1]["end_frame"]), (4, 6))
        self.assertIsNone(goals[2]["start_frame"])

    def test_snap_goal_starts_walks_back_typing_burst(self) -> None:
        view = _view()
        actions = ["NO_OP"] * len(view.frames)
        actions[3] = "0 0 0 ; +KeyH -KeyH"
        actions[4] = "0 0 0 ; +KeyI -KeyI"
        actions[5] = "0 0 0 ; +Return -Return"
        unit = AnnotationUnit(unit_id="s0", view=view, window_index=0, n_windows=1,
                              lo=0, hi=len(view.frames), tail_buffer=0, actions=actions)
        goals = [{"instruction": "x", "start_frame": 4, "end_frame": 6}]
        snap_goal_starts(goals, unit)
        self.assertEqual(goals[0]["start_frame"], 3)  # pulled to burst start
        # A mouse goal is untouched.
        goals2 = [{"instruction": "y", "start_frame": 8, "end_frame": 9}]
        snap_goal_starts(goals2, unit)
        self.assertEqual(goals2[0]["start_frame"], 8)

    def test_view_span_to_master_roundtrip_row_validates(self) -> None:
        view = _view()
        start_m, end_m = view_span_to_master(view, 2, 6)
        row = {
            "goal_id": "s0_g00", "segment_id": "s0", "recording_id": "r0",
            "start_master_idx": start_m, "end_master_idx": end_m,
            "instruction": "do the thing", "method": "describe_extract",
            "model": "m", "prompt_pack_sha": "abc", "unit_id": "s0",
        }
        validate_goal_row(row)
        self.assertEqual((start_m, end_m), (30, 90))


class PlansHelpersTest(unittest.TestCase):
    def test_plan_flags(self) -> None:
        self.assertEqual(plan_flags("", "any"), ["empty"])
        self.assertIn("restates_instruction",
                      plan_flags("I need to open the settings page. I'll open the settings page.",
                                 "open the settings page"))
        good = ("The build failed on the missing import earlier, so I'll check the "
                "dependency list first.")
        self.assertEqual(plan_flags(good, "fix the build"), [])
        self.assertIn("not_first_person", plan_flags("Open the file and edit it.", "edit the file"))

    def test_goal_start_frame_matching(self) -> None:
        view = _view()  # frames at ticks 0,15,...,135
        self.assertEqual(goal_start_frame(view, 45).master_idx, 45)   # exact
        self.assertEqual(goal_start_frame(view, 50).master_idx, 60)   # nearest after
        self.assertEqual(goal_start_frame(view, 149).master_idx, 135)  # before (tail)


if __name__ == "__main__":
    unittest.main()
