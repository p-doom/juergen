"""Regression gates for defects #11, #12, #13, #16, #17 and the round-trip audit.

* #11 ``val_steps=15`` over a 65-record val split with a non-restarting grain
      iterator scores a DIFFERENT 15 records each eval.
* #12 ``keep_latest=1`` + ``keep_period=305`` deleted every intermediate checkpoint.
* #13 omegalax logs ``val/loss`` only to wandb, never to stdout.
* #16 a record-level split scattered one task's k sibling trajectories across BOTH
      splits.
* #17 ``slug = app__task_id`` collides across sample roots.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rft.errors import (
    LeakError,
    MissingFieldError,
    RetentionError,
    RoundTripError,
    SchemaError,
    ValCoverageError,
)
from rft.records import assistant_target, build_records, verify_written_split
from rft.roundtrip import assert_convention_declared, assert_roundtrip_clean, audit_roundtrip
from rft.splits import (
    assert_no_leak,
    assert_unique_sample_ids,
    load_task_ids,
    make_sample_id,
    partition_records,
    task_level_split,
)
from rft.training import (
    REQUIRED_OMEGALAX_FLAGS,
    ValLossTee,
    assert_paths_on_project,
    assert_required_flags,
    resolve_val_steps,
    retention_plan,
    validate_retention,
)


class Defect17SlugCollisionTest(unittest.TestCase):
    """Defect #17: ``slug = f"{app}__{task_id}"`` collides across sample roots.

    The k=8 collection stores the SAME task under 8 different ``sample_i/`` roots, and
    two sharded collectors both name their subdirs ``sample_0..7`` — so merging
    untagged roots produced duplicate ``sample_id``s.
    """

    def test_the_old_slug_collides(self) -> None:
        """Reproduce it: the app__task_id slug is identical across roots."""
        def old_slug(app: str, task: str) -> str:
            return f"{app}__{task}"

        a = old_slug("chrome", "030eeff7")
        b = old_slug("chrome", "030eeff7")
        self.assertEqual(a, b, "the old slug cannot distinguish two sample roots")

    def test_sample_id_differs_across_roots(self) -> None:
        a = make_sample_id(sample_root="/fast/project/runA/samples", task_id="030eeff7",
                           rollout_index=0, app="chrome")
        b = make_sample_id(sample_root="/fast/project/runB/samples", task_id="030eeff7",
                           rollout_index=0, app="chrome")
        self.assertNotEqual(a, b)

    def test_sample_id_differs_across_rollouts(self) -> None:
        ids = {
            make_sample_id(sample_root="/tmp/r", task_id="t", rollout_index=i)
            for i in range(8)
        }
        self.assertEqual(len(ids), 8)

    def test_same_basename_different_parent_still_differs(self) -> None:
        """The exact k=8 case: both roots are named `samples`."""
        a = make_sample_id(sample_root="/fast/project/x/legacy/samples", task_id="t",
                           rollout_index=0)
        b = make_sample_id(sample_root="/fast/project/x/fast/samples", task_id="t",
                           rollout_index=0)
        self.assertNotEqual(a, b)

    def test_duplicate_detection_names_the_defect(self) -> None:
        recs = [{"sample_id": "dup"}, {"sample_id": "dup"}, {"sample_id": "ok"}]
        with self.assertRaises(SchemaError) as ctx:
            assert_unique_sample_ids(recs)
        self.assertIn("defect #17", str(ctx.exception))

    def test_missing_sample_id_raises(self) -> None:
        with self.assertRaises(MissingFieldError):
            assert_unique_sample_ids([{"task_id": "t"}])

    def test_empty_task_id_is_rejected(self) -> None:
        with self.assertRaises(SchemaError):
            make_sample_id(sample_root="/tmp/r", task_id="", rollout_index=0)


class Defect16TaskLevelSplitTest(unittest.TestCase):
    """Defect #16: a record-level split scattered one task's k siblings across both
    sides. Rejection sampling produces k correlated rollouts per task, so a
    record-level split is a leak by construction.
    """

    def test_record_level_split_leaks_siblings(self) -> None:
        """Reproduce the leak with the naive approach."""
        import random

        records = [{"task_id": f"t{t}", "step": s} for t in range(10) for s in range(8)]
        rng = random.Random(0)
        rng.shuffle(records)
        cut = int(0.9 * len(records))
        train_tasks = {r["task_id"] for r in records[:cut]}
        val_tasks = {r["task_id"] for r in records[cut:]}
        self.assertTrue(train_tasks & val_tasks, "record-level split must leak siblings")

    def test_task_level_split_is_disjoint(self) -> None:
        split = task_level_split([f"t{i}" for i in range(200)], val_fraction=0.1)
        self.assertEqual(split.train & split.val, frozenset())
        self.assertEqual(split.n_tasks, 200)

    def test_all_k_rollouts_of_a_task_land_on_one_side(self) -> None:
        tasks = [f"t{i}" for i in range(60)]
        split = task_level_split(tasks, val_fraction=0.2)
        records = [{"task_id": t, "step": s} for t in tasks for s in range(8)]
        train, val = partition_records(records, split)
        train_tasks = {r["task_id"] for r in train}
        val_tasks = {r["task_id"] for r in val}
        self.assertEqual(train_tasks & val_tasks, set())
        self.assertEqual(len(train) + len(val), len(records))
        # every task contributed all 8 of its records to exactly one side
        for t in tasks:
            n_train = sum(1 for r in train if r["task_id"] == t)
            n_val = sum(1 for r in val if r["task_id"] == t)
            self.assertIn((n_train, n_val), [(8, 0), (0, 8)], t)

    def test_split_is_stable_when_tasks_are_added(self) -> None:
        """A hash split does not reshuffle existing tasks when the set grows.

        ``random.shuffle(sorted(ids))`` does, which makes val numbers incomparable
        across data revisions.
        """
        first = task_level_split([f"t{i}" for i in range(100)], val_fraction=0.1)
        grown = task_level_split([f"t{i}" for i in range(150)], val_fraction=0.1)
        for i in range(100):
            t = f"t{i}"
            self.assertEqual(t in first.val, t in grown.val, t)

    def test_split_is_reproducible_across_calls(self) -> None:
        a = task_level_split([f"t{i}" for i in range(80)], val_fraction=0.15)
        b = task_level_split([f"t{i}" for i in range(80)], val_fraction=0.15)
        self.assertEqual(a.val, b.val)

    def test_unassigned_task_is_an_error_not_a_silent_drop(self) -> None:
        split = task_level_split([f"t{i}" for i in range(20)], val_fraction=0.2)
        with self.assertRaises(LeakError):
            partition_records([{"task_id": "stranger"}], split)

    def test_record_without_task_id_raises(self) -> None:
        split = task_level_split([f"t{i}" for i in range(20)], val_fraction=0.2)
        with self.assertRaises(MissingFieldError):
            partition_records([{"step": 0}], split)


class EvalLeakGuardTest(unittest.TestCase):
    """The 259/110 rule: held-out tasks are eval-only, never training-adjacent."""

    def test_leak_is_detected(self) -> None:
        with self.assertRaises(LeakError) as ctx:
            assert_no_leak(["a", "b", "c"], ["c", "d"], context="unit test")
        self.assertIn("EVAL LEAK", str(ctx.exception))

    def test_clean_sets_pass(self) -> None:
        assert_no_leak(["a", "b"], ["c", "d"])

    def test_empty_heldout_set_is_refused(self) -> None:
        """A leak check against nothing proves nothing."""
        with self.assertRaises(SchemaError):
            assert_no_leak(["a"], [])

    def test_real_split_files_are_disjoint_and_sized(self) -> None:
        """Gate the vendored 259/110 split itself."""
        root = Path(__file__).resolve().parents[2] / "osworld_parity" / "split"
        train_p = root / "osworld_train.tasks"
        held_p = root / "osworld_eval_heldout.tasks"
        if not train_p.is_file() or not held_p.is_file():
            self.skipTest(f"vendored split not present under {root}")
        train = load_task_ids(train_p)
        held = load_task_ids(held_p)
        self.assertEqual(len(train), 259)
        self.assertEqual(len(held), 110)
        # the .tasks files store "app/task_id"; ids must still be disjoint
        train_ids = {t.split("/")[-1] for t in train}
        held_ids = {t.split("/")[-1] for t in held}
        self.assertEqual(train_ids & held_ids, set())
        self.assertEqual(len(train_ids) + len(held_ids), 369)


class Defect12CheckpointRetentionTest(unittest.TestCase):
    """Defect #12: ``keep_latest=1`` + ``keep_period=305`` deleted every intermediate
    checkpoint, so val-based selection was impossible after the fact.
    """

    def test_the_historical_config_keeps_exactly_one_checkpoint(self) -> None:
        plan = retention_plan(total_steps=300, save_interval=150, keep_period=305,
                              keep_latest=1)
        self.assertEqual(plan.surviving_steps, (300,))
        self.assertEqual(plan.n_surviving, 1)

    def test_the_historical_config_is_rejected(self) -> None:
        with self.assertRaises(RetentionError) as ctx:
            validate_retention(total_steps=300, save_interval=150, keep_period=305,
                               keep_latest=1)
        self.assertIn("defect #12", str(ctx.exception))

    def test_keep_period_must_be_a_multiple_of_save_interval(self) -> None:
        """keep_period=305 with save_every=150 matches NO saved step."""
        plan = retention_plan(total_steps=600, save_interval=150, keep_period=305,
                              keep_latest=0)
        self.assertEqual(plan.surviving_steps, ())

    def test_the_tier2_fix_retains_everything(self) -> None:
        """save_every == keep_period, keep_latest high => every save survives."""
        plan = validate_retention(total_steps=600, save_interval=100, keep_period=100,
                                  keep_latest=8)
        self.assertEqual(plan.surviving_steps, (100, 200, 300, 400, 500, 600))

    def test_the_shipped_labctl_recipe_config_is_thin(self) -> None:
        """num_steps=300 / save_every=150 / keep_period=150 / keep_latest=1 leaves 2."""
        plan = retention_plan(total_steps=300, save_interval=150, keep_period=150,
                              keep_latest=1)
        self.assertEqual(plan.surviving_steps, (150, 300))
        with self.assertRaises(RetentionError):
            validate_retention(total_steps=300, save_interval=150, keep_period=150,
                               keep_latest=1, min_checkpoints_for_selection=3)


class Defect11ValCoverageTest(unittest.TestCase):
    """Defect #11: ``val_steps=15`` over a 65-record val split with a NON-restarting
    grain iterator scores a DIFFERENT 15 records at each eval, so val numbers are
    comparable only within matched windows.
    """

    def test_the_historical_config_is_partial(self) -> None:
        with self.assertRaises(ValCoverageError) as ctx:
            resolve_val_steps(n_val_records=65, global_batch_size=1, requested_val_steps=15)
        msg = str(ctx.exception)
        self.assertIn("15 of 65", msg)
        self.assertIn("DIFFERENT subset", msg)
        self.assertIn("val_steps=65", msg)

    def test_default_covers_the_full_split(self) -> None:
        plan = resolve_val_steps(n_val_records=65, global_batch_size=1)
        self.assertEqual(plan.val_steps, 65)
        self.assertTrue(plan.covers_full_split)
        self.assertEqual(plan.n_records_scored, 65)

    def test_batching_is_accounted_for(self) -> None:
        plan = resolve_val_steps(n_val_records=65, global_batch_size=8)
        self.assertEqual(plan.val_steps, 9)  # ceil(65/8)
        self.assertTrue(plan.covers_full_split)

    def test_partial_must_be_opted_into_by_name(self) -> None:
        plan = resolve_val_steps(n_val_records=65, global_batch_size=1,
                                 requested_val_steps=15, allow_partial=True)
        self.assertFalse(plan.covers_full_split)
        self.assertIn("full_split=False", plan.describe())

    def test_two_interleaved_windows_are_the_observed_symptom(self) -> None:
        """Tier-1's val/sup_tokens alternated between two 15-record windows."""
        n_val, val_steps = 65, 15
        offsets = [(i * val_steps) % n_val for i in range(4)]
        self.assertEqual(len(set(offsets)), 4, "each eval starts at a different offset")


class Defect13ValLossToStdoutTest(unittest.TestCase):
    """Defect #13: omegalax logs ``val/loss`` only to wandb, never to stdout, so a
    run's val curve is invisible in the slurm log and selection depends on a wandb
    read (which has its own defect, #10).
    """

    def test_val_loss_is_echoed_with_the_step(self) -> None:
        tee = ValLossTee()
        echoes = tee.scan([
            "step 100 | loss 1.2\n",
            "step 100 | val/loss = 0.8123\n",
            "step 200 | loss 1.0\n",
            "step 200 | val/loss = 0.7000\n",
        ])
        self.assertEqual(len(echoes), 2)
        self.assertIn("[rft.val] step=100 val/loss=0.812300", echoes[0])
        self.assertEqual(tee.points, [(100, 0.8123), (200, 0.7)])

    def test_best_checkpoint_is_the_argmin(self) -> None:
        tee = ValLossTee()
        tee.scan(["step 1 val/loss: 0.9", "step 2 val/loss: 0.4", "step 3 val/loss: 0.6"])
        self.assertEqual(tee.best(), (2, 0.4))

    def test_dict_style_logging_is_also_recognised(self) -> None:
        tee = ValLossTee()
        tee.scan(["step=50 {'val/loss': 0.55, 'train/loss': 1.1}"])
        self.assertEqual(tee.points, [(50, 0.55)])

    def test_nan_val_loss_is_surfaced_not_filtered(self) -> None:
        tee = ValLossTee()
        echoes = tee.scan(["step 10 val/loss = nan"])
        self.assertIn("DIVERGED", echoes[0])

    def test_no_val_points_is_an_error_not_an_empty_selection(self) -> None:
        tee = ValLossTee()
        tee.scan(["step 1 | loss 1.0", "step 2 | loss 0.9"])
        with self.assertRaises(ValCoverageError) as ctx:
            tee.best()
        self.assertIn("Checkpoint selection cannot proceed", str(ctx.exception))

    def test_all_nan_curve_is_also_an_error(self) -> None:
        tee = ValLossTee()
        tee.scan(["step 1 val/loss = nan", "step 2 val/loss = nan"])
        with self.assertRaises(ValCoverageError):
            tee.best()


class OmegalaxRequiredFlagsTest(unittest.TestCase):
    """``train_vlm_sft.py`` on origin/main hard-requires these; labctl validate does
    not catch their absence and the job crashes at start."""

    def test_all_required_flags_are_enforced(self) -> None:
        for flag in REQUIRED_OMEGALAX_FLAGS:
            flags = {f: "x" for f in REQUIRED_OMEGALAX_FLAGS if f != flag}
            with self.assertRaises(SchemaError, msg=flag) as ctx:
                assert_required_flags(flags)
            self.assertIn(flag, str(ctx.exception))

    def test_false_values_count_as_present(self) -> None:
        assert_required_flags({f: False for f in REQUIRED_OMEGALAX_FLAGS})

    def test_outputs_must_be_on_project(self) -> None:
        with self.assertRaises(SchemaError) as ctx:
            assert_paths_on_project(["/fast/home/franz.srambical/ckpt"])
        self.assertIn("95G", str(ctx.exception))
        assert_paths_on_project(["/fast/project/HFMI_SynergyUnit/x"])


class RoundTripAuditTest(unittest.TestCase):
    """The mandatory audit: every converted target must survive the EXACT eval parser."""

    def test_clean_conversion_passes(self) -> None:
        items = [(f"s{i}", "120 -40 0 ; +LMB -LMB") for i in range(50)]
        report = audit_roundtrip(items, grammar="bare_line")
        self.assertEqual(report.n_checked, 50)
        self.assertTrue(report.clean)
        assert_roundtrip_clean(report)

    def test_unparseable_target_is_counted_and_fails_the_gate(self) -> None:
        items = [("good", "0 0 0"), ("bad", "move the mouse a bit right")]
        report = audit_roundtrip(items, grammar="bare_line")
        self.assertEqual(report.n_unparseable, 1)
        self.assertFalse(report.clean)
        with self.assertRaises(RoundTripError) as ctx:
            assert_roundtrip_clean(report)
        self.assertIn("bad", str(ctx.exception))

    def test_report_names_the_parser_that_produced_it(self) -> None:
        report = audit_roundtrip([("s", "0 0 0")], grammar="bare_line")
        self.assertIn("action_parser.py", report.describe())

    def test_anomalies_are_reported_in_the_audit(self) -> None:
        from rft.grammars import get_grammar

        if not get_grammar("deltatype", require_available=False).available:
            self.skipTest("deltatype unavailable in this eval/action_parser.py")
        report = audit_roundtrip([("s", "10 -5 ; +LMB")], grammar="deltatype")
        self.assertEqual(report.anomaly_counts.get("missing_scroll_token"), 1)
        self.assertIn("missing_scroll_token", report.describe())

    def test_records_must_declare_their_grammar(self) -> None:
        with self.assertRaises(MissingFieldError) as ctx:
            assert_convention_declared([{"sample_id": "a"}])
        self.assertIn("relative delta", str(ctx.exception))

    def test_declared_grammar_must_be_registered(self) -> None:
        with self.assertRaises(SchemaError):
            assert_convention_declared([{"grammar": "made_up"}])


class BuildRecordsEndToEndTest(unittest.TestCase):
    """Stage 3 must run audit + leak check BEFORE writing anything."""

    def _heldout(self, tmp: Path) -> Path:
        p = tmp / "heldout.tasks"
        p.write_text("\n".join(f"held{i}" for i in range(20)))
        return p

    def _rollouts(self, n_tasks: int = 40, k: int = 4) -> list[dict]:
        return [
            {"task_id": f"t{t}", "sample_id": f"s{t}_{j}", "rollout_index": j,
             "app": "chrome", "scores": {"reward": 1.0}}
            for t in range(n_tasks)
            for j in range(k)
        ]

    @staticmethod
    def _convert(rollout):
        return [{
            "step": 0,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "do it"},
                {"role": "assistant", "content": "120 -40 0 ; +LMB -LMB"},
            ],
        }]

    def test_happy_path_writes_task_level_split(self) -> None:
        with TemporaryDirectory(dir="/tmp") as tmp:
            tmpp = Path(tmp)
            report, split = build_records(
                self._rollouts(),
                grammar="bare_line",
                convert=self._convert,
                out_dir=tmpp / "out",
                heldout_tasks_path=self._heldout(tmpp),
            )
            self.assertEqual(report.n_records, 160)
            self.assertTrue(report.roundtrip["clean"])
            self.assertEqual(report.leak_check["overlap"], 0)
            verified = verify_written_split(tmpp / "out", self._heldout(tmpp))
            self.assertEqual(verified["task_overlap"], 0)
            self.assertEqual(
                verified["n_train_records"] + verified["n_val_records"], 160
            )
            self.assertGreater(len(split.val), 0)

    def test_a_failed_audit_writes_nothing(self) -> None:
        def bad_convert(_rollout):
            return [{"step": 0, "messages": [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "please move right a little"},
            ]}]

        with TemporaryDirectory(dir="/tmp") as tmp:
            tmpp = Path(tmp)
            out = tmpp / "out"
            with self.assertRaises(RoundTripError):
                build_records(
                    self._rollouts(),
                    grammar="bare_line",
                    convert=bad_convert,
                    out_dir=out,
                    heldout_tasks_path=self._heldout(tmpp),
                )
            self.assertFalse((out / "_normalized" / "train" / "chat.jsonl").exists())

    def test_a_leaking_dataset_writes_nothing(self) -> None:
        with TemporaryDirectory(dir="/tmp") as tmp:
            tmpp = Path(tmp)
            held = tmpp / "heldout.tasks"
            held.write_text("t3\nt4\n")  # t3/t4 are in the training rollouts
            out = tmpp / "out"
            with self.assertRaises(LeakError):
                build_records(
                    self._rollouts(n_tasks=10, k=2),
                    grammar="bare_line",
                    convert=self._convert,
                    out_dir=out,
                    heldout_tasks_path=held,
                )
            self.assertFalse((out / "_normalized").exists())

    def test_zero_records_is_refused(self) -> None:
        with TemporaryDirectory(dir="/tmp") as tmp:
            tmpp = Path(tmp)
            with self.assertRaises(SchemaError) as ctx:
                build_records(
                    self._rollouts(n_tasks=5, k=1),
                    grammar="bare_line",
                    convert=lambda _r: [],
                    out_dir=tmpp / "out",
                    heldout_tasks_path=self._heldout(tmpp),
                )
            self.assertIn("refusing to write an empty dataset", str(ctx.exception))

    def test_assistant_target_must_be_the_last_turn(self) -> None:
        with self.assertRaises(SchemaError):
            assistant_target({"messages": [
                {"role": "assistant", "content": "0 0 0"},
                {"role": "user", "content": "hm"},
            ]})

    def test_written_split_violation_is_detected_after_the_fact(self) -> None:
        """verify_written_split runs at the start of stage 4 too."""
        with TemporaryDirectory(dir="/tmp") as tmp:
            tmpp = Path(tmp)
            for name in ("train", "val"):
                d = tmpp / "out" / "_normalized" / name
                d.mkdir(parents=True)
                (d / "chat.jsonl").write_text(json.dumps({"task_id": "shared"}) + "\n")
            with self.assertRaises(SchemaError) as ctx:
                verify_written_split(tmpp / "out", self._heldout(tmpp))
            self.assertIn("defect #16", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
