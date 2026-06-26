import argparse
import json
import tempfile
import unittest
from pathlib import Path

from annotation_pipeline.run_pipeline import (
    aggregate_frame_cache_outputs,
    commit_stage02_clip,
    commit_stage03_clip,
    frames_cache_dir,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class RunPipelineAggregationTest(unittest.TestCase):
    def test_aggregates_frame_cache_to_run_level_stage00_and_stage01(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run"
            cache_root = root / "cache"
            clip = {
                "recording_id": "rec_a_123456",
                "segment_start": 0,
                "segment_end": 0,
            }
            args = argparse.Namespace(
                target_fps=2,
                target_height=720,
                stage01_max_noop_run=2,
                cache_root=cache_root,
            )
            cache = frames_cache_dir(clip, 2, 720, 2, root=cache_root)
            _write_jsonl(cache / "stage_00_manifest" / "manifest.jsonl", [{"segment_id": "s0"}])
            _write_json(cache / "stage_00_manifest" / "manifest_summary.json", {"n_segments": 1})
            _write_jsonl(
                cache / "stage_01_frames_actions" / "frame_records.jsonl",
                [{"image_path": "/tmp/frame.jpg", "action": "NO_OP"}],
            )
            _write_json(
                cache / "stage_01_frames_actions" / "segment_summaries.json",
                [{"segment_id": "s0", "n_frames": 1}],
            )
            _write_json(
                cache / "stage_01_frames_actions" / "frames_actions_summary.json",
                {"n_frame_records": 1},
            )

            stage00_summary, stage01_summary = aggregate_frame_cache_outputs(
                run_dir,
                {"clip_a": clip},
                args,
            )

            self.assertEqual(stage00_summary["n_manifest_rows"], 1)
            self.assertEqual(stage01_summary["n_frame_records"], 1)
            manifest = _read_jsonl(run_dir / "stage_00_manifest" / "manifest.jsonl")
            frames = _read_jsonl(run_dir / "stage_01_frames_actions" / "frame_records.jsonl")
            self.assertEqual(manifest[0]["clip_id"], "clip_a")
            self.assertEqual(frames[0]["clip_id"], "clip_a")

    def test_commits_stage02_temp_output_to_run_level_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            stage02 = Path(tmpdir) / "stage02_temp"
            _write_json(
                stage02 / "trajectories_raw.json",
                {
                    "recording_id": "rec_a",
                    "annotation_source": "vlm_two_pass",
                    "trajectories": [{"start_time_s": 1.0, "end_time_s": 9.0, "verified": True}],
                },
            )
            _write_jsonl(stage02 / "pass_a_candidates.jsonl", [{"start_time_s": 1.0}])
            _write_jsonl(stage02 / "pass_a_merged_segments.jsonl", [{"start_time_s": 1.0}])
            _write_json(stage02 / "naming_rejected.json", [{"reason": "empty_instruction"}])
            _write_json(stage02 / "stage02_summary.json", {"n_trajectories": 1, "n_verified": 1})

            commit_stage02_clip(run_dir, "clip_a", stage02)

            summary = json.loads((run_dir / "stage_02_segment" / "stage02_summary.json").read_text())
            self.assertEqual(summary["n_clips"], 1)
            self.assertEqual(summary["n_trajectories"], 1)
            rows = _read_jsonl(run_dir / "stage_02_segment" / "trajectories_raw.jsonl")
            self.assertEqual(rows[0]["clip_id"], "clip_a")
            self.assertEqual(rows[0]["recording_id"], "rec_a")
            rejected = _read_jsonl(run_dir / "stage_02_segment" / "naming_rejected.jsonl")
            self.assertEqual(rejected[0]["clip_id"], "clip_a")

    def test_commits_stage03_temp_output_to_run_level_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            stage03 = Path(tmpdir) / "stage03_temp"
            _write_jsonl(
                stage03 / "trajectories.jsonl",
                [
                    {
                        "sample_id": "rec_a_traj0000",
                        "recording_id": "rec_a",
                        "start_time_s": 1.0,
                        "end_time_s": 9.0,
                    }
                ],
            )
            _write_jsonl(
                stage03 / "rejected_trajectories.jsonl",
                [{"trajectory_idx": 1, "reason": "too_short"}],
            )
            _write_json(stage03 / "assemble_summary.json", {"n_samples": 1, "n_rejected": 1})

            commit_stage03_clip(run_dir, "clip_a", stage03)

            summary = json.loads((run_dir / "stage_03_assemble" / "assemble_summary.json").read_text())
            self.assertEqual(summary["n_samples"], 1)
            rows = _read_jsonl(run_dir / "stage_03_assemble" / "trajectories.jsonl")
            self.assertEqual(rows[0]["clip_id"], "clip_a")
            self.assertEqual(rows[0]["sample_id"], "rec_a_traj0000")
            rejected = _read_jsonl(run_dir / "stage_03_assemble" / "rejected_trajectories.jsonl")
            self.assertEqual(rejected[0]["clip_id"], "clip_a")
            self.assertEqual(summary["reject_reasons"], {"too_short": 1})


if __name__ == "__main__":
    unittest.main()
