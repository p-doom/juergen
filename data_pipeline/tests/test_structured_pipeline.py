import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from annotation_pipeline import build_sft
from annotation_pipeline.stage_04_refine_boundaries import refine_boundaries
from annotation_pipeline.stage_05_assemble_trajectories import assemble_trajectories
from annotation_pipeline.stage_06_project_sft import project_sft


def _observation(frame_idx: int, *, time_s: float | None = None, activity: str = "other") -> dict:
    time_s = float(frame_idx) if time_s is None else time_s
    events = (
        [
            {
                "source_event_idx": 0,
                "local_time_s": time_s + 0.1,
                "kind": "move",
                "dx": 15.0,
                "dy": -2.0,
            },
            {
                "source_event_idx": 1,
                "local_time_s": time_s + 0.2,
                "kind": "press",
                "key": "LMB",
            },
            {
                "source_event_idx": 2,
                "local_time_s": time_s + 0.3,
                "kind": "release",
                "key": "LMB",
            },
        ]
        if frame_idx == 0
        else []
    )
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
        "interval_end_s": time_s + 1.0,
        "image_path": f"ar:///tmp/images.array_record#{frame_idx}",
        "events": events,
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

    def test_nonoverlap_clamps_to_an_existing_observation(self) -> None:
        observations = [_observation(0), _observation(2), _observation(4)]

        goals = refine_boundaries(
            [_proposal(0, 4), _proposal(4, 4)],
            observations,
            policy="vision_only",
        )

        self.assertEqual(goals[0]["end_frame_idx"], 2)

    def test_structured_trajectory_contains_events_but_no_messages_or_action_text(self) -> None:
        observations = [_observation(0), _observation(1)]
        goals = refine_boundaries([_proposal(0, 1)], observations, policy="vision_only")

        trajectories, rejected = assemble_trajectories(observations, goals)

        self.assertEqual(rejected, [])
        self.assertNotIn("messages", trajectories[0])
        self.assertNotIn("action", trajectories[0]["steps"][0])
        self.assertEqual(trajectories[0]["steps"][0]["interval_start_s"], 0.0)
        self.assertEqual(trajectories[0]["steps"][0]["interval_end_s"], 1.0)
        self.assertEqual(trajectories[0]["steps"][0]["action_bin"]["move_dx"], 15.0)

    def test_sft_projection_uses_ordered_v2_by_default(self) -> None:
        observations = [_observation(0), _observation(1)]
        goals = refine_boundaries([_proposal(0, 1)], observations, policy="vision_only")
        trajectories, _ = assemble_trajectories(observations, goals)

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
        self.assertEqual(
            assistant_texts,
            ["move(15,-2); down(LMB); up(LMB)", "NO_OP"],
        )
        self.assertFalse(any("plan" in text.lower() for text in assistant_texts))
        self.assertEqual(manifest["action_schema"], "ordered_events_v2")
        self.assertEqual(manifest["continuous_action_hz"], 10.0)
        self.assertEqual(
            manifest["primitive_counts"],
            {"down": 1, "move": 1, "scroll": 0, "up": 1},
        )
        self.assertEqual(manifest["n_no_op_turns"], 1)

    def test_sft_projection_keeps_aggregate_v1_as_explicit_ablation(self) -> None:
        observations = [_observation(0), _observation(1)]
        goals = refine_boundaries([_proposal(0, 1)], observations, policy="vision_only")
        trajectories, _ = assemble_trajectories(observations, goals)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectories_path = root / "trajectories.jsonl"
            trajectories_path.write_text(json.dumps(trajectories[0]) + "\n")
            manifest = project_sft(
                trajectories_path=trajectories_path,
                output_dir=root / "sft",
                action_schema="aggregate_delta_keys_v1",
                val_frac=0.0,
            )
            sample = json.loads((root / "sft" / "chat.jsonl").read_text())

        assistant_texts = [
            message["content"][0]["text"]
            for message in sample["messages"]
            if message["role"] == "assistant"
        ]
        self.assertEqual(assistant_texts, ["15 -2 0 ; +LMB -LMB", "NO_OP"])
        self.assertEqual(manifest["action_schema"], "aggregate_delta_keys_v1")
        self.assertIsNone(manifest["continuous_action_hz"])

    def test_sft_projection_reports_held_state_anomalies_without_mutating_actions(self) -> None:
        observations = [_observation(0), _observation(1)]
        observations[0]["events"] = [
            {
                "source_event_idx": 0,
                "local_time_s": 0.1,
                "kind": "release",
                "key": "LMB",
            },
            {
                "source_event_idx": 1,
                "local_time_s": 0.2,
                "kind": "press",
                "key": "KeyA",
            },
            {
                "source_event_idx": 2,
                "local_time_s": 0.3,
                "kind": "press",
                "key": "KeyA",
            },
        ]
        goals = refine_boundaries([_proposal(0, 1)], observations, policy="vision_only")
        trajectories, _ = assemble_trajectories(observations, goals)

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

        assistant_text = next(
            message["content"][0]["text"]
            for message in sample["messages"]
            if message["role"] == "assistant"
        )
        self.assertEqual(assistant_text, "up(LMB); down(KeyA); down(KeyA)")
        self.assertEqual(
            manifest["state_diagnostics"],
            {
                "dangling_up": 1,
                "duplicate_down": 1,
                "held_at_trajectory_end": 1,
                "non_neutral_trajectory": 1,
            },
        )

    def test_window_goals_receive_unique_global_indices(self) -> None:
        first = {**_proposal(0, 1), "goal_idx": 0, "boundary_policy": "vision_only"}
        second = {**_proposal(2, 3), "goal_idx": 0, "boundary_policy": "vision_only"}

        merged = build_sft.merge_window_goals([(1, 2, [second]), (0, 2, [first])])

        self.assertEqual([goal["goal_idx"] for goal in merged], [0, 1])
        self.assertEqual([goal["source_window_idx"] for goal in merged], [0, 1])

    def test_window_goal_merge_rejects_incomplete_parent_annotations(self) -> None:
        goal = {**_proposal(0, 1), "goal_idx": 0, "boundary_policy": "vision_only"}

        with self.assertRaisesRegex(ValueError, "Missing annotation windows"):
            build_sft.merge_window_goals([(0, 2, [goal])])

    def test_build_sft_uses_the_prepared_observation_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            modalities = run_dir / "_modalities" / "clips" / "seg"
            view_dir = modalities / "stage_02_view"
            stage00 = modalities / "stage_00"
            boundaries = run_dir / "model" / "clips" / "seg" / "stage_04_boundaries"
            view_dir.mkdir(parents=True)
            stage00.mkdir()
            boundaries.mkdir(parents=True)

            observations = [_observation(0), _observation(1)]
            (view_dir / "observations.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in observations)
            )
            (stage00 / "manifest.jsonl").write_text(
                json.dumps({"recording_id": "rec", "segment_id": "seg"}) + "\n"
            )
            goal = {
                **_proposal(0, 1),
                "goal_idx": 0,
                "boundary_policy": "vision_only",
                "start_frame_idx": 0,
                "end_frame_idx": 1,
            }
            (boundaries / "goals.jsonl").write_text(json.dumps(goal) + "\n")
            (boundaries / "manifest.json").write_text(
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
                "--val-frac",
                "0",
            ]
            with patch("sys.argv", argv):
                build_sft.main()

            trajectory = json.loads(
                (output_dir / "stage_05_trajectories" / "trajectories.jsonl").read_text().strip()
            )
            manifest = json.loads(
                (output_dir / "stage_05_trajectories" / "manifest.json").read_text()
            )

        self.assertEqual(trajectory["n_observations"], 2)
        self.assertEqual(
            manifest["source_observation_views"]["seg"],
            str(view_dir / "observations.jsonl"),
        )


if __name__ == "__main__":
    unittest.main()
