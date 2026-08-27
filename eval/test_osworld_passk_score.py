import json
import sys
import tempfile
import unittest
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from osworld_passk_score import _sample_path, collect, summarize  # noqa: E402


def build(base: Path, plan: dict[tuple[str, str], list[float | None]]) -> None:
    for (app, tid), rewards in plan.items():
        for i, r in enumerate(rewards):
            if r is None:
                continue
            p = _sample_path(base, app, tid, i)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"scores": {"reward": r, "n_steps_taken": 7}}))


def run(plan: dict[tuple[str, str], list[float | None]], k: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        build(base, plan)
        return summarize(collect(base, sorted(plan), k), k)


class SamplePathTest(unittest.TestCase):
    def test_sample_zero_keeps_the_historical_layout(self):
        p = _sample_path(Path("/out"), "os", "abc", 0)
        self.assertEqual(p, Path("/out/os/abc/result.json"))

    def test_later_samples_nest_without_colliding(self):
        paths = {_sample_path(Path("/out"), "os", "abc", i) for i in range(4)}
        self.assertEqual(len(paths), 4)
        self.assertIn(Path("/out/os/abc/sample_3/result.json"), paths)


class MetricsTest(unittest.TestCase):
    def test_one_lucky_sample_makes_a_task_pass_at_k(self):
        d = run({("os", "t1"): [0.0, 0.0, 1.0, 0.0]}, k=4)
        self.assertAlmostEqual(d["pass_at_1"], 0.25)
        self.assertAlmostEqual(d["pass_at_k"], 1.0)
        self.assertAlmostEqual(d["mean_best_of_k"], 1.0)

    def test_partial_credit_survives_into_best_of_k(self):
        d = run({("os", "t1"): [0.25, 0.5, 0.0, 0.0]}, k=4)
        self.assertAlmostEqual(d["pass_at_1"], 0.1875)
        self.assertAlmostEqual(d["pass_at_k"], 1.0)
        self.assertAlmostEqual(d["mean_best_of_k"], 0.5)

    def test_never_solved_task_scores_zero_everywhere(self):
        d = run({("os", "t1"): [0.0, 0.0, 0.0, 0.0]}, k=4)
        self.assertAlmostEqual(d["pass_at_1"], 0.0)
        self.assertAlmostEqual(d["pass_at_k"], 0.0)
        self.assertAlmostEqual(d["mean_best_of_k"], 0.0)

    def test_missing_samples_are_counted_not_scored_as_failures(self):
        d = run({("os", "t1"): [1.0, None, None, None]}, k=4)
        self.assertEqual(d["n_samples"], 1)
        self.assertEqual(d["n_samples_missing"], 3)
        self.assertAlmostEqual(d["pass_at_1"], 1.0)
        self.assertAlmostEqual(d["mean_best_of_k"], 1.0)

    def test_nan_reward_counts_as_a_failed_sample(self):
        d = run({("os", "t1"): [float("nan"), float("nan"), 1.0, float("nan")]}, k=4)
        self.assertEqual(d["n_samples"], 4)
        self.assertEqual(d["n_samples_missing"], 0)
        self.assertAlmostEqual(d["pass_at_1"], 0.25)
        self.assertAlmostEqual(d["pass_at_k"], 1.0)

    def test_pass_at_k_is_over_tasks_while_pass_at_1_is_over_samples(self):
        d = run(
            {
                ("os", "t1"): [0.0, 0.0, 1.0, 0.0],
                ("os", "t2"): [0.0, 0.0, 0.0, 0.0],
                ("vs_code", "t3"): [1.0, 1.0, 1.0, 1.0],
            },
            k=4,
        )
        self.assertAlmostEqual(d["pass_at_1"], 5 / 12)
        self.assertAlmostEqual(d["pass_at_k"], 2 / 3)
        self.assertAlmostEqual(d["mean_best_of_k"], 2 / 3)
        self.assertEqual(d["n_solved_any"], 2)

    def test_per_app_splits_follow_the_same_definitions(self):
        d = run(
            {
                ("os", "t1"): [0.0, 0.0, 1.0, 0.0],
                ("vs_code", "t3"): [0.0, 0.0, 0.0, 0.0],
            },
            k=4,
        )
        self.assertAlmostEqual(d["per_app"]["os"]["pass_at_k"], 1.0)
        self.assertAlmostEqual(d["per_app"]["vs_code"]["pass_at_k"], 0.0)
        self.assertAlmostEqual(d["per_app"]["os"]["pass_at_1"], 0.25)


class LegacySchemaTest(unittest.TestCase):
    def test_k_one_reduces_to_the_single_sample_numbers(self):
        d = run({("os", "t1"): [1.0], ("os", "t2"): [0.0], ("os", "t3"): [0.5]}, k=1)
        self.assertEqual(d["n"], 3)
        self.assertAlmostEqual(d["mean_reward"], 0.5)
        self.assertAlmostEqual(d["pass_at_1"], 0.5)
        self.assertAlmostEqual(d["pass_at_k"], 2 / 3)

    def test_per_task_stays_the_sample_zero_pairs_the_old_aggregator_emitted(self):
        d = run({("os", "t1"): [1.0, 0.0], ("os", "t2"): [0.0, 1.0]}, k=2)
        self.assertEqual([task for task, _ in d["per_task"]], ["os/t1", "os/t2"])
        self.assertEqual([scores["reward"] for _, scores in d["per_task"]], [1.0, 0.0])
        self.assertAlmostEqual(d["mean_reward"], 0.5)


if __name__ == "__main__":
    unittest.main()
