import tempfile
import unittest
from pathlib import Path

from annotation_pipeline import run_dataset


class RunDatasetTest(unittest.TestCase):
    def test_completed_unit_is_found_under_any_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            goals = (
                run_root
                / "previous_model"
                / "clips"
                / "unit"
                / "stage_04_boundaries"
                / "goals.jsonl"
            )
            goals.parent.mkdir(parents=True)
            goals.write_text("")

            self.assertTrue(run_dataset._boundaries_exist(run_root, "unit"))

    def test_progress_isolated_by_phase(self) -> None:
        self.assertTrue(hasattr(run_dataset, "progress_path_for_phase"))
        root = Path("/tmp/run")

        self.assertEqual(
            run_dataset.progress_path_for_phase(root, "prepare", ".shard0_of_2"),
            root / "progress.prepare.shard0_of_2.jsonl",
        )
        self.assertEqual(
            run_dataset.progress_path_for_phase(root, "annotate", ""),
            root / "progress.annotate.jsonl",
        )

    def test_failed_progress_rows_are_retried(self) -> None:
        self.assertTrue(hasattr(run_dataset, "completed_segment_ids"))

        completed = run_dataset.completed_segment_ids(
            [
                {"segment_id": "complete"},
                {"segment_id": "retry", "error": "temporary API failure"},
            ]
        )

        self.assertEqual(completed, {"complete"})


if __name__ == "__main__":
    unittest.main()
