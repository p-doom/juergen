"""Within-record label leakage, and provenance without Postgres.

Two invariant classes that caught things no downstream check would have:

* **preamble digit-leak = 0.** If a reasoning preamble carries the target
  coordinates, the experiment silently degenerates into text arithmetic and produces
  a spectacular false positive that looks exactly like success. This is *not* dataset
  overlap — it is information leakage inside one record.
* **geometry, not identifiers.** Deduplicate eval instances on exact
  ``(cursor, bbox)`` geometry: scene identifiers can differ while the geometry is
  duplicated, so an id-based check passes duplicates straight through.

Plus the 4-hop, Postgres-free provenance walk, and the hard-fail blacklist for the
two checkpoints whose provenance is unrecoverable.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rft.arms import UncontrolledComparisonError
from rft.errors import LeakError, SchemaError
from rft.label_leak import (
    LabelLeakError,
    assert_no_geometry_overlap,
    assert_no_label_leak,
    check_label_leak,
    duplicate_geometries,
    geometry_key,
    numbers_in,
    prose_digit_leak,
)
from rft.provenance import (
    UNRESOLVABLE_CHECKPOINTS,
    BlacklistedCheckpointError,
    CheckpointProvenance,
    ProvenanceError,
    assert_comparable,
    assert_not_blacklisted,
    resolve_checkpoint,
)


class NumberExtractionTest(unittest.TestCase):
    def test_standalone_numbers_only(self) -> None:
        self.assertEqual(numbers_in("move to x=127, y=-403"), [127.0, -403.0])

    def test_a_longer_number_is_not_a_match_for_its_prefix(self) -> None:
        """127 must not be 'found' inside 1279 - that is why we compare numbers."""
        self.assertNotIn(127.0, numbers_in("the value is 1279"))

    def test_decimals_are_handled(self) -> None:
        self.assertEqual(numbers_in("ratio 0.5 of 12.25"), [0.5, 12.25])


class PreambleDigitLeakTest(unittest.TestCase):
    """preamble digit-leak must be 0."""

    def test_a_preamble_carrying_the_coordinates_is_caught(self) -> None:
        records = [
            ("r1", "Action: click the button at (982, 127) to close it.",
             '{"coordinate": [982, 127]}'),
        ]
        with self.assertRaises(LabelLeakError) as ctx:
            assert_no_label_leak(records)
        msg = str(ctx.exception)
        self.assertIn("LABEL LEAK", msg)
        self.assertIn("text arithmetic", msg)

    def test_a_clean_preamble_passes(self) -> None:
        records = [
            ("r1", 'Action: Click the "X" button on the top-right corner of the pop-up.',
             '{"coordinate": [982, 127]}'),
        ]
        report = assert_no_label_leak(records)
        self.assertTrue(report.clean)
        self.assertEqual(report.n_records_leaking, 0)

    def test_common_small_values_are_ignored_by_default(self) -> None:
        """'step 2' in prose must not be a leak just because a delta contains 2."""
        records = [("r1", "This is step 2 of the plan.", "2 0 0")]
        self.assertTrue(check_label_leak(records).clean)

    def test_ignoring_can_be_narrowed(self) -> None:
        records = [("r1", "This is step 2 of the plan.", "2 0 0")]
        report = check_label_leak(records, ignored_values=())
        self.assertFalse(report.clean)

    def test_leak_rate_over_zero_records_is_undefined(self) -> None:
        report = check_label_leak([])
        with self.assertRaises(SchemaError):
            _ = report.leak_rate

    def test_chat_records_prose_and_user_turn_are_both_checked(self) -> None:
        leaky_user = [{
            "sample_id": "s1",
            "messages": [
                {"role": "user", "content": "The target is at 982, 127."},
                {"role": "assistant", "content": "0 0 0\n982 127 0 ; +LMB -LMB"},
            ],
        }]
        report = prose_digit_leak(leaky_user)
        self.assertFalse(report.clean, report.describe())

    def test_clean_chat_records_pass(self) -> None:
        clean = [{
            "sample_id": "s1",
            "messages": [
                {"role": "user", "content": "Close the update popup."},
                {"role": "assistant",
                 "content": "Action: click the X in the corner.\n982 127 0 ; +LMB -LMB"},
            ],
        }]
        self.assertTrue(prose_digit_leak(clean).clean)

    def test_build_stage_refuses_leaking_records(self) -> None:
        from rft.records import build_records

        def convert(rollout):
            return [{
                "step": 0,
                "messages": [
                    {"role": "user", "content": "go"},
                    {"role": "assistant",
                     "content": "Action: move by 120 and -40.\n120 -40 0 ; +LMB -LMB"},
                ],
            }]

        rollouts = [
            {"task_id": f"t{i}", "sample_id": f"s{i}", "rollout_index": 0,
             "scores": {"reward": 1.0}}
            for i in range(20)
        ]
        with TemporaryDirectory(dir="/tmp") as tmp:
            tmpp = Path(tmp)
            held = tmpp / "heldout.tasks"
            held.write_text("\n".join(f"held{i}" for i in range(10)))
            with self.assertRaises(LabelLeakError):
                build_records(
                    rollouts,
                    grammar="bare_line",
                    convert=convert,
                    out_dir=tmpp / "out",
                    heldout_tasks_path=held,
                    source_text_key=None,
                )


class GeometryLeakTest(unittest.TestCase):
    """Dedupe on geometry, not on identifiers."""

    def test_identical_geometry_under_different_ids_is_caught(self) -> None:
        train = [geometry_key((10, 10), (100, 100, 200, 200))]
        held = [geometry_key((10, 10), (100, 100, 200, 200))]
        with self.assertRaises(LeakError) as ctx:
            assert_no_geometry_overlap(train, held, context="bbox29")
        self.assertIn("GEOMETRY LEAK", str(ctx.exception))
        self.assertIn("identifiers", str(ctx.exception).lower())

    def test_an_id_based_check_would_have_passed_the_same_pair(self) -> None:
        """Reproduce the weakness the geometry check replaces."""
        train_ids = {"scene_a"}
        held_ids = {"scene_b"}
        self.assertEqual(train_ids & held_ids, set(), "id check sees no overlap...")
        # ...yet the geometry is identical:
        with self.assertRaises(LeakError):
            assert_no_geometry_overlap(
                [geometry_key((10, 10), (1, 2, 3, 4))],
                [geometry_key((10, 10), (1, 2, 3, 4))],
            )

    def test_distinct_geometry_passes(self) -> None:
        assert_no_geometry_overlap(
            [geometry_key((10, 10), (1, 2, 3, 4))],
            [geometry_key((11, 10), (1, 2, 3, 4))],
        )

    def test_empty_heldout_is_refused(self) -> None:
        with self.assertRaises(SchemaError):
            assert_no_geometry_overlap([geometry_key((0, 0), (1, 1, 2, 2))], [])

    def test_duplicates_within_one_split_are_reported(self) -> None:
        g = geometry_key((5, 5), (0, 0, 10, 10))
        self.assertEqual(duplicate_geometries([g, g, g]), {g: 3})

    def test_malformed_geometry_raises(self) -> None:
        with self.assertRaises(SchemaError):
            geometry_key((1, 2, 3), (1, 2, 3, 4))
        with self.assertRaises(SchemaError):
            geometry_key((1, 2), (1, 2, 3))


class ProvenanceBlacklistTest(unittest.TestCase):
    """Two checkpoints have unrecoverable provenance and must hard-fail."""

    def test_the_blacklist_names_both(self) -> None:
        self.assertEqual(len(UNRESOLVABLE_CHECKPOINTS), 2)
        joined = " ".join(sorted(UNRESOLVABLE_CHECKPOINTS))
        self.assertIn("abl_hf_oe2/003000", joined)
        self.assertIn("abl_hf_v7u/003000", joined)

    def test_blacklisted_paths_raise(self) -> None:
        for suffix in UNRESOLVABLE_CHECKPOINTS:
            path = Path("/fast/project/HFMI_SynergyUnit/p-doom_shared") / suffix
            with self.assertRaises(BlacklistedCheckpointError) as ctx:
                assert_not_blacklisted(path)
            self.assertIn('"inputs": []', str(ctx.exception))

    def test_resolve_refuses_a_blacklisted_checkpoint_before_touching_disk(self) -> None:
        path = (
            "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/checkpoints/"
            "mihir.mahajan/abl_hf_oe2/003000"
        )
        with self.assertRaises(BlacklistedCheckpointError):
            resolve_checkpoint(path)

    def test_an_ordinary_path_is_not_blacklisted(self) -> None:
        assert_not_blacklisted("/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/"
                              "checkpoints/franz.srambical/some_run/000300")


class ProvenanceWalkTest(unittest.TestCase):
    """The 4-hop walk, driven by a synthetic labctl tree (no Postgres, no cluster)."""

    def _tree(self, tmp: Path, *, inputs: list[dict], manifest: dict) -> Path:
        root = tmp / "labctl"
        ckpt = root / "checkpoints" / "owner" / "stream" / "000300"
        ckpt.mkdir(parents=True)
        (ckpt / ".meta.json").write_text(json.dumps({
            "id": "artifact_abc",
            "kind": "checkpoint",
            "user": "owner",
            "producer_run_id": "run_1",
            "metadata": {"producer_recipe": "train_recipe", "step": 300,
                         "marker": "_CHECKPOINT_METADATA"},
        }))
        lab = root / "labctl_runs" / "runs" / "owner" / "run_1" / ".lab"
        lab.mkdir(parents=True)
        (lab / "context.json").write_text(json.dumps({
            "inputs": inputs,
            "provenance": {"git_head": "deadbeef", "repo_path": "/repo/omegalax"},
        }))
        ds = tmp / "dataset"
        ds.mkdir()
        (ds / "manifest.json").write_text(json.dumps(manifest))
        return ckpt

    def test_walk_reaches_the_dataset_manifest(self) -> None:
        with TemporaryDirectory(dir="/tmp") as tmp:
            tmpp = Path(tmp)
            ckpt = self._tree(
                tmpp,
                inputs=[{"role": "dataset", "resolved_path": str(tmpp / "dataset"),
                         "artifact_id": "artifact_ds"}],
                manifest={"params": {"action_format": "moverel", "goal_conditioned": False,
                                     "max_length": 16384, "window": 12}},
            )
            prov = resolve_checkpoint(ckpt, labctl_root=tmpp / "labctl")
            self.assertEqual(prov.producer_recipe, "train_recipe")
            self.assertEqual(prov.step, 300)
            self.assertEqual(prov.training_run_id, "run_1")
            self.assertEqual(prov.repo_head, "deadbeef")
            decisive = prov.decisive()
            self.assertEqual(decisive["action_format"], "moverel")
            self.assertEqual(decisive["window"], 12)
            self.assertIn("dataset_manifest", [h.kind for h in prov.hops])
            self.assertIn("action_format", prov.describe())

    def test_missing_meta_json_is_an_error_not_an_unknown(self) -> None:
        with TemporaryDirectory(dir="/tmp") as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            with self.assertRaises(ProvenanceError) as ctx:
                resolve_checkpoint(ckpt, labctl_root=Path(tmp))
            self.assertIn("no .meta.json", str(ctx.exception))

    def test_empty_inputs_list_stops_the_walk_loudly(self) -> None:
        """The blacklisted checkpoints' failure mode, reproduced structurally."""
        with TemporaryDirectory(dir="/tmp") as tmp:
            tmpp = Path(tmp)
            ckpt = self._tree(tmpp, inputs=[], manifest={})
            with self.assertRaises(ProvenanceError) as ctx:
                resolve_checkpoint(ckpt, labctl_root=tmpp / "labctl")
            self.assertIn('"inputs": []', str(ctx.exception))

    def test_the_walk_needs_no_database(self) -> None:
        """Nothing in rft.provenance imports a DB client."""
        import inspect

        import rft.provenance as mod

        source = inspect.getsource(mod)
        for forbidden in ("psycopg", "sqlalchemy", "asyncpg", "labctl status"):
            self.assertNotIn(forbidden, source)


class CheckpointComparabilityTest(unittest.TestCase):
    """The guard that would have stopped the "goals vs no-goals" comparison whose
    members also differed in action format, window and sequence length."""

    @staticmethod
    def _prov(name: str, decisive: dict) -> CheckpointProvenance:
        return CheckpointProvenance(
            checkpoint_dir=Path(f"/ckpt/{name}"),
            artifact_id=name,
            owner="owner",
            producer_recipe="r",
            producer_run_id="run",
            step=300,
            training_run_id="run",
            dataset_path=Path("/ds"),
            dataset_manifest={"params": decisive},
        )

    def test_a_clean_single_dimension_pair_passes(self) -> None:
        a = self._prov("a", {"goal_conditioned": True, "action_format": "moverel",
                             "window": 12, "max_length": 16384})
        b = self._prov("b", {"goal_conditioned": False, "action_format": "moverel",
                             "window": 12, "max_length": 16384})
        fields = assert_comparable([a, b], dimension="goal_conditioned")
        self.assertIn("goal_conditioned", fields)

    def test_the_real_confounded_pair_is_refused(self) -> None:
        a = self._prov("a", {"goal_conditioned": True, "action_format": "moverel",
                             "window": 12, "max_length": 16384})
        b = self._prov("b", {"goal_conditioned": False, "action_format": "diffabs",
                             "window": 48, "max_length": 65536})
        with self.assertRaises(UncontrolledComparisonError) as ctx:
            assert_comparable([a, b], dimension="goal_conditioned")
        msg = str(ctx.exception)
        for confound in ("action_format", "window", "max_length"):
            self.assertIn(confound, msg)

    def test_an_undeclared_field_is_an_unknown_not_a_match(self) -> None:
        a = self._prov("a", {"goal_conditioned": True, "action_format": "moverel"})
        b = self._prov("b", {"goal_conditioned": False})
        with self.assertRaises(UncontrolledComparisonError) as ctx:
            assert_comparable([a, b], dimension="goal_conditioned")
        self.assertIn("undeclared", str(ctx.exception))

    def test_at_least_two_checkpoints_are_required(self) -> None:
        with self.assertRaises(SchemaError):
            assert_comparable([self._prov("a", {"x": 1})], dimension="x")


if __name__ == "__main__":
    unittest.main()
