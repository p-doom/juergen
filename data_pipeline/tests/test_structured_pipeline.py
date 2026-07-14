import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from annotation_pipeline import build_sft
from annotation_pipeline.stage_04_refine_boundaries import refine_boundaries
from annotation_pipeline.stage_05_assemble_trajectories import assemble_trajectories
from annotation_pipeline.stage_06_project_sft import project_sft


def _observation(
    frame_idx: int,
    *,
    time_s: float | None = None,
    interval_end_s: float | None = None,
    activity: str = "other",
) -> dict:
    time_s = float(frame_idx) if time_s is None else time_s
    interval_end_s = time_s + 1.0 if interval_end_s is None else interval_end_s
    action_bin = {
        "move_dx": 15.0 if frame_idx == 0 else 0.0,
        "move_dy": -2.0 if frame_idx == 0 else 0.0,
        "scroll": 0.0,
        "events": [["+", "LMB"], ["-", "LMB"]] if frame_idx == 0 else [],
    }
    return {
        "recording_id": "rec",
        "segment_id": "seg",
        "segment_idx": 0,
        "global_frame_idx": frame_idx,
        "local_frame_idx": frame_idx,
        "source_frame_idx": frame_idx * 30,
        "local_time_s": time_s,
        "global_time_s": time_s,
        "interval_start_s": time_s,
        "interval_end_s": interval_end_s,
        "interval_start_global_s": time_s,
        "interval_end_global_s": interval_end_s,
        "image_path": f"ar:///tmp/images.array_record#{frame_idx}",
        "events": [],
        "action_bin": action_bin,
        "activity": activity,
        "has_submission": False,
        "is_noop": frame_idx != 0,
    }


def _proposal(start: int, end: int) -> dict:
    return {
        "instruction": "select the visible item",
        "instruction_variants": ["choose the item shown"],
        "anchor": "visible click",
        "grounding": "The item is selected.",
        "start_frame": start,
        "end_frame": end,
    }


class StructuredPipelineTest(unittest.TestCase):
    def test_boundary_policy_is_an_explicit_stage_choice(self) -> None:
        observations = [
            _observation(0),
            _observation(1, activity="type"),
            _observation(2, activity="type"),
            _observation(3),
        ]

        visual = refine_boundaries([_proposal(2, 3)], observations, policy="vision_only")
        refined = refine_boundaries([_proposal(2, 3)], observations, policy="keylog_refined")

        self.assertEqual(visual[0]["start_frame_idx"], 2)
        self.assertEqual(refined[0]["start_frame_idx"], 1)

    def test_nonoverlap_clamps_to_an_existing_annotation_observation(self) -> None:
        observations = [_observation(0), _observation(2), _observation(4)]

        goals = refine_boundaries(
            [_proposal(0, 4), _proposal(4, 4)],
            observations,
            policy="vision_only",
        )

        self.assertEqual(goals[0]["end_frame_idx"], 2)
        self.assertEqual(goals[0]["end_time_s"], 3.0)

    def test_boundaries_carry_half_open_times_between_sampling_views(self) -> None:
        annotation_view = [_observation(0), _observation(1)]

        goals = refine_boundaries([_proposal(0, 1)], annotation_view, policy="vision_only")

        self.assertEqual(goals[0]["start_time_s"], 0.0)
        self.assertEqual(goals[0]["end_time_s"], 2.0)

    def test_assembly_uses_training_view_timestamps_not_annotation_frame_indices(self) -> None:
        annotation_view = [_observation(0), _observation(1)]
        goals = refine_boundaries([_proposal(0, 1)], annotation_view, policy="vision_only")
        training_view = [
            _observation(10, time_s=0.0, interval_end_s=0.5),
            _observation(11, time_s=0.5, interval_end_s=1.0),
            _observation(12, time_s=1.0, interval_end_s=1.5),
            _observation(13, time_s=1.5, interval_end_s=2.0),
        ]

        trajectories, rejected = assemble_trajectories(training_view, goals)

        self.assertEqual(rejected, [])
        self.assertEqual(trajectories[0]["n_observations"], 4)
        self.assertEqual(trajectories[0]["start_frame_idx"], 10)
        self.assertEqual(trajectories[0]["end_frame_idx"], 13)

    def test_coarser_training_interval_is_kept_when_it_overlaps_the_goal(self) -> None:
        annotation_view = [_observation(0), _observation(1)]
        goals = refine_boundaries([_proposal(1, 1)], annotation_view, policy="vision_only")
        coarse_training_view = [_observation(20, time_s=0.0, interval_end_s=2.0)]

        trajectories, rejected = assemble_trajectories(coarse_training_view, goals)

        self.assertEqual(rejected, [])
        self.assertEqual(trajectories[0]["n_observations"], 1)

    def test_structured_trajectory_contains_events_but_no_messages_or_action_text(self) -> None:
        goals = refine_boundaries(
            [_proposal(0, 1)], [_observation(0), _observation(1)], policy="vision_only"
        )

        trajectories, rejected = assemble_trajectories([_observation(0), _observation(1)], goals)

        self.assertEqual(rejected, [])
        self.assertNotIn("messages", trajectories[0])
        self.assertNotIn("action", trajectories[0]["steps"][0])
        self.assertEqual(trajectories[0]["steps"][0]["action_bin"]["move_dx"], 15.0)

    def test_sft_projection_uses_the_current_action_format_without_plans(self) -> None:
        goals = refine_boundaries(
            [_proposal(0, 1)], [_observation(0), _observation(1)], policy="vision_only"
        )
        trajectories, _ = assemble_trajectories([_observation(0), _observation(1)], goals)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectories_path = root / "trajectories.jsonl"
            trajectories_path.write_text(json.dumps(trajectories[0]) + "\n")
            manifest = project_sft(
                trajectories_path=trajectories_path,
                output_dir=root / "sft",
                val_frac=0.0,
            )
            sample = json.loads((root / "sft" / "chat.jsonl").read_text())

        assistant_texts = [
            message["content"][0]["text"]
            for message in sample["messages"]
            if message["role"] == "assistant"
        ]
        self.assertEqual(assistant_texts, ["15 -2 0 ; +LMB -LMB", "NO_OP"])
        self.assertFalse(any("plan" in text.lower() for text in assistant_texts))
        self.assertEqual(manifest["action_schema"], "aggregate_delta_keys_v1")

    def test_window_goals_receive_unique_global_indices(self) -> None:
        first = {**_proposal(0, 1), "goal_idx": 0, "boundary_policy": "vision_only"}
        second = {**_proposal(2, 3), "goal_idx": 0, "boundary_policy": "vision_only"}

        self.assertTrue(hasattr(build_sft, "merge_window_goals"))
        merged = build_sft.merge_window_goals([(1, 2, [second]), (0, 2, [first])])

        self.assertEqual([goal["goal_idx"] for goal in merged], [0, 1])
        self.assertEqual([goal["source_window_idx"] for goal in merged], [0, 1])

    def test_window_goal_merge_rejects_incomplete_parent_annotations(self) -> None:
        goal = {**_proposal(0, 1), "goal_idx": 0, "boundary_policy": "vision_only"}

        with self.assertRaisesRegex(ValueError, "Missing annotation windows"):
            build_sft.merge_window_goals([(0, 2, [goal])])

    def test_build_sft_materializes_and_uses_independent_training_fps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            base_dir = run_dir / "_modalities" / "clips" / "seg" / "stage_01_base"
            unit_dir = run_dir / "model" / "clips" / "seg" / "stage_04_boundaries"
            base_dir.mkdir(parents=True)
            unit_dir.mkdir(parents=True)

            frames = [
                {
                    key: value
                    for key, value in _observation(
                        frame_idx,
                        time_s=frame_idx / 2,
                        interval_end_s=(frame_idx + 1) / 2,
                    ).items()
                    if key
                    in {
                        "recording_id",
                        "segment_id",
                        "segment_idx",
                        "global_frame_idx",
                        "local_frame_idx",
                        "source_frame_idx",
                        "local_time_s",
                        "global_time_s",
                        "image_path",
                    }
                }
                for frame_idx in range(4)
            ]
            (base_dir / "manifest.json").write_text(
                json.dumps({"stage": "base_modalities", "base_fps": 2.0})
            )
            (base_dir / "frames.jsonl").write_text(
                "".join(json.dumps(frame) + "\n" for frame in frames)
            )
            (base_dir / "events.jsonl").write_text("")
            (base_dir / "segment_summaries.json").write_text(
                json.dumps([{"segment_id": "seg", "duration_s": 2.0}])
            )
            stage00 = base_dir.parent / "stage_00"
            stage00.mkdir()
            (stage00 / "manifest.jsonl").write_text(
                json.dumps({"recording_id": "rec", "segment_id": "seg"}) + "\n"
            )

            goal = {
                **_proposal(0, 3),
                "goal_idx": 0,
                "boundary_policy": "vision_only",
                "start_frame_idx": 0,
                "end_frame_idx": 3,
                "start_time_s": 0.0,
                "end_time_s": 2.0,
            }
            (unit_dir / "goals.jsonl").write_text(json.dumps(goal) + "\n")
            (unit_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "parent_segment_id": "seg",
                        "window_index": 0,
                        "n_windows": 1,
                    }
                )
            )

            output_dir = root / "output"
            argv = [
                "build_sft",
                "--run-dir",
                str(run_dir),
                "--out",
                str(output_dir),
                "--training-fps",
                "2.0",
                "--training-idle-keep-head",
                "10",
                "--training-idle-keep-tail",
                "10",
                "--val-frac",
                "0",
            ]
            with patch("sys.argv", argv):
                build_sft.main()

            trajectory = json.loads(
                (output_dir / "stage_05_trajectories" / "trajectories.jsonl").read_text().strip()
            )
            training_manifest = json.loads(
                (
                    output_dir / "stage_02_training_views" / "clips" / "seg" / "manifest.json"
                ).read_text()
            )

            output_1fps = root / "output_1fps"
            argv_1fps = [
                "build_sft",
                "--run-dir",
                str(run_dir),
                "--out",
                str(output_1fps),
                "--training-fps",
                "1.0",
                "--training-idle-keep-head",
                "10",
                "--training-idle-keep-tail",
                "10",
                "--val-frac",
                "0",
            ]
            with patch("sys.argv", argv_1fps):
                build_sft.main()
            trajectory_1fps = json.loads(
                (output_1fps / "stage_05_trajectories" / "trajectories.jsonl").read_text().strip()
            )

        self.assertEqual(trajectory["n_observations"], 4)
        self.assertEqual(trajectory_1fps["n_observations"], 2)
        self.assertEqual(training_manifest["view_name"], "training")
        self.assertEqual(training_manifest["observation_fps"], 2.0)


if __name__ == "__main__":
    unittest.main()
