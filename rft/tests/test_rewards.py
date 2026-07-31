"""Regression gates for the reward-reading defects: #1, #2, #3.

Each test corresponds to a confidently-wrong conclusion that a careless reward read
produced. If any of these starts passing for the wrong reason, the corresponding
research verdict becomes unreliable again.
"""

from __future__ import annotations

import math
import unittest

from rft.errors import MissingFieldError, SchemaError, UnscoredRewardError
from rft.rewards import (
    CANONICAL_REWARD_PATH,
    UNSCORED,
    UnscoredPolicy,
    aggregate_rewards,
    has_field,
    read_reward,
    require_field,
)

# The real result.json envelope (juergen/eval/result.py writes this shape).
REAL_RESULT = {
    "schema_version": 1,
    "task": "osworld_fullbench",
    "scores": {"reward": 1.0, "n_steps_taken": 7, "stop_reason_code": 1},
    "params": {"task_id": "030eeff7", "app": "chrome", "stop_reason": "agent_terminate"},
    "inputs": {"model_path": "/x", "served_model_name": "y"},
    "n_samples": 1,
    "elapsed_s": 160,
}


class Defect01RewardIsNestedTest(unittest.TestCase):
    """Defect #1: the reward lives at ``scores.reward``, NOT top level.

    ``payload.get("reward", 0.0)`` cannot distinguish "absent" from "zero", so an
    aggregate keyed on a top-level ``reward`` silently returned all zeros. The read
    must RAISE when the path is wrong.
    """

    def test_canonical_path_is_nested(self) -> None:
        self.assertEqual(CANONICAL_REWARD_PATH, "scores.reward")

    def test_reads_nested_reward(self) -> None:
        self.assertEqual(read_reward(REAL_RESULT), 1.0)

    def test_top_level_read_raises_instead_of_returning_zero(self) -> None:
        with self.assertRaises(MissingFieldError) as ctx:
            read_reward(REAL_RESULT, path="reward")
        # The message must list what IS there, so the fix is obvious.
        self.assertIn("scores", str(ctx.exception))

    def test_no_fallback_search(self) -> None:
        """A reader must not go hunting for a reward-shaped field elsewhere."""
        payload = {"reward": 0.0, "scores": {"reward": 1.0}}
        self.assertEqual(read_reward(payload, path="scores.reward"), 1.0)
        self.assertEqual(read_reward(payload, path="reward"), 0.0)

    def test_descending_into_non_mapping_raises(self) -> None:
        with self.assertRaises(SchemaError):
            read_reward({"scores": 5}, path="scores.reward")


class Defect02AbsentSuccessFieldTest(unittest.TestCase):
    """Defect #2: a ``success`` field is absent from 100% of result files.

    A "zero full completions" verdict was counting a nonexistent key. Reading a
    verdict-bearing field must raise; and coverage must be *reportable* so the
    absence is visible before it becomes a finding.
    """

    def test_success_is_not_in_the_real_schema(self) -> None:
        self.assertFalse(has_field(REAL_RESULT, "success"))

    def test_require_field_raises_on_absent_success(self) -> None:
        with self.assertRaises(MissingFieldError):
            require_field(REAL_RESULT, "success")

    def test_has_field_is_only_for_reporting(self) -> None:
        """has_field reports; it must never be a silent branch to a default."""
        self.assertTrue(has_field(REAL_RESULT, "scores.reward"))
        self.assertFalse(has_field(REAL_RESULT, "scores.success"))

    def test_bool_in_a_reward_slot_is_a_schema_error(self) -> None:
        """Storing `success` where a reward belongs must not read as 1.0/0.0."""
        with self.assertRaises(SchemaError):
            read_reward({"scores": {"reward": True}})


class Defect03NanMeansUnscoredTest(unittest.TestCase):
    """Defect #3: ``final_reward`` inits to NaN; ``evaluate()`` raising leaves it NaN.

    NaN means the task was NEVER SCORED. Coercing it to 0 counts an instrument
    failure as a capability failure. Confirmed init sites: ``baseline_eval_shard.py``
    :106, ``format_eval_shard.py``:99, ``osworld_fullbench_runner.py``:175.
    """

    def test_nan_reads_as_unscored_not_zero(self) -> None:
        r = read_reward({"scores": {"reward": float("nan")}})
        self.assertIs(r, UNSCORED)
        self.assertNotEqual(r, 0.0)

    def test_null_reads_as_unscored(self) -> None:
        self.assertIs(read_reward({"scores": {"reward": None}}), UNSCORED)

    def test_unscored_is_not_truthy_or_falsy(self) -> None:
        """Truth-testing an unscored reward must raise, not silently be False."""
        with self.assertRaises(UnscoredRewardError):
            bool(UNSCORED)

    def test_unscored_cannot_be_summed_accidentally(self) -> None:
        with self.assertRaises(TypeError):
            _ = UNSCORED + 1.0  # type: ignore[operator]

    def test_aggregate_requires_an_explicit_policy(self) -> None:
        with self.assertRaises(TypeError):
            aggregate_rewards([1.0, 0.0], unscored="exclude")  # type: ignore[arg-type]

    def test_raise_policy_refuses_to_hide_unscored(self) -> None:
        with self.assertRaises(UnscoredRewardError):
            aggregate_rewards([1.0, UNSCORED], unscored=UnscoredPolicy.RAISE)

    def test_exclude_and_count_as_zero_differ_and_both_report(self) -> None:
        rewards = [1.0, 1.0, 0.0, UNSCORED]
        exc = aggregate_rewards(rewards, unscored=UnscoredPolicy.EXCLUDE)
        caz = aggregate_rewards(rewards, unscored=UnscoredPolicy.COUNT_AS_ZERO)
        self.assertAlmostEqual(exc.mean, 2 / 3)
        self.assertAlmostEqual(caz.mean, 2 / 4)
        # Both must expose the unscored count; that is what makes the choice visible.
        for agg in (exc, caz):
            self.assertEqual(agg.n_unscored, 1)
            self.assertEqual(agg.n_total, 4)
        self.assertEqual(exc.denominator, 3)
        self.assertEqual(caz.denominator, 4)

    def test_empty_scored_set_has_no_mean(self) -> None:
        """Defect #9's cousin: 0/0 is undefined, never 0.0."""
        with self.assertRaises(UnscoredRewardError):
            aggregate_rewards([UNSCORED, UNSCORED], unscored=UnscoredPolicy.EXCLUDE)

    def test_infinite_reward_is_rejected(self) -> None:
        with self.assertRaises(SchemaError):
            read_reward({"scores": {"reward": math.inf}})

    def test_n_at_one_is_reported_for_binary_rewards(self) -> None:
        agg = aggregate_rewards([1.0, 1.0, 0.0], unscored=UnscoredPolicy.RAISE)
        self.assertEqual(agg.n_at_one, 2)

    def test_n_at_one_is_none_for_partial_credit(self) -> None:
        """The off-shelf anchor run has partial credit, so a 0/1 assumption is wrong."""
        agg = aggregate_rewards([1.0, 0.8949, 0.0], unscored=UnscoredPolicy.RAISE)
        self.assertIsNone(agg.n_at_one)


class NanPassesThresholdFilterTest(unittest.TestCase):
    """LIVE BUG this design prevents, found in ``tier2/build_k8_records.py``:232-236.

    That code does ``r = reward_of(td); if r < args.min_reward: continue``. Since
    ``float('nan') < 1e-6`` is **False**, a NaN-reward rollout is ACCEPTED as a
    training success and then poisons ``mean_reward_kept`` to NaN. This test pins the
    arithmetic fact and proves the typed API cannot express the bug.
    """

    def test_nan_comparison_is_false_in_both_directions(self) -> None:
        nan = float("nan")
        self.assertFalse(nan < 1e-6)
        self.assertFalse(nan >= 1e-6)

    def test_unscored_never_reaches_a_threshold_predicate(self) -> None:
        from rft.scoring import Verdict, score_rollout, threshold_predicate

        seen: list[float] = []

        def spy(reward: float) -> bool:
            seen.append(reward)
            return threshold_predicate(1e-6)(reward)

        scored = score_rollout({"scores": {"reward": float("nan")}}, accept=spy)
        self.assertIs(scored.verdict, Verdict.UNSCORED)
        self.assertEqual(seen, [], "the predicate must never see an unscored reward")


if __name__ == "__main__":
    unittest.main()
