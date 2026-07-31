from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "generate_index.py"
SPEC = importlib.util.spec_from_file_location("trace_index_generator", MODULE_PATH)
assert SPEC and SPEC.loader
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_run_context(runs_root: Path, run_id: str, recipe: str, artifact: Path, job_id: str) -> None:
    lab = runs_root / run_id / ".lab"
    lab.mkdir(parents=True)
    write_json(
        lab / "context.json",
        {
            "run_id": run_id,
            "recipe_name": recipe,
            "outputs": {"result": {"path": str(artifact)}},
        },
    )
    (lab / f"{recipe}_{job_id}.log").write_text("fixture job log\n")


def add_meta(artifact: Path, artifact_id: str, run_id: str, recipe: str, marker: str, manifest: dict) -> None:
    write_json(
        artifact / ".meta.json",
        {
            "id": artifact_id,
            "kind": "eval_result",
            "user": "franz.srambical",
            "alias": artifact.name,
            "producer_run_id": run_id,
            "created_at": 1,
            "metadata": {
                "marker": marker,
                "producer_recipe": recipe,
                "result": manifest,
            },
        },
    )


class TraceIndexGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.eval_root = self.root / "eval"
        self.runs_root = self.root / "runs"
        self.output = self.root / "viewer"
        self.eval_root.mkdir()
        self.runs_root.mkdir()
        self._add_relative_fixture()
        self._add_typing_fixture()
        self.rules = self.root / "rules.json"
        write_json(
            self.rules,
            {
                "schema_version": 1,
                "expected_user": "franz.srambical",
                "rules": [
                    {"name": "relative", "glob": "fixture_relative_*", "adapter": "relative_mouse", "min_matches": 1},
                    {"name": "typing", "glob": "fixture_typing_*", "adapter": "typing", "min_matches": 1},
                ],
            },
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _add_relative_fixture(self):
        artifact = self.eval_root / "fixture_relative_eval"
        artifact.mkdir()
        rows = artifact / "rows.jsonl"
        rows.write_text((HERE / "fixtures/relative_row.json").read_text())
        report = artifact / "report.json"
        write_json(report, {"status": "complete"})
        image = artifact / "scene.png"
        image.write_text("fixture image placeholder")
        scene = {
            "scene_id": "fixture_scene",
            "kind": "short",
            "cursor": [10, 10],
            "bbox": [25, 15, 35, 25],
            "target_center": [30, 20],
            "image_path": str(image),
            "distance_px": 22.36,
        }
        (artifact / "scenes.jsonl").write_text(json.dumps(scene) + "\n")
        manifest = {
            "artifact_type": "synthetic_factorial_eval",
            "status": "complete",
            "rows": "rows.jsonl",
            "rows_sha256": digest(rows),
            "report": "report.json",
            "report_sha256": digest(report),
            "row_contract": {"count": 1},
            "grammar_name": "deltatype_raw",
            "preamble": False,
            "model_provenance": {"arm": "fixture", "step": 10, "model_dir": "fixture_r8_s10"},
        }
        write_json(artifact / "eval_manifest.json", manifest)
        add_meta(artifact, "artifact_fixture_relative", "run_fixturerelative", "fixture_relative_recipe", "eval_manifest.json", manifest)
        add_run_context(self.runs_root, "run_fixturerelative", "fixture_relative_recipe", artifact, "101")
        self.relative_rows = rows

    def _add_typing_fixture(self):
        artifact = self.eval_root / "fixture_typing_eval"
        artifact.mkdir()
        rows = artifact / "typing_generation_rows.jsonl"
        rows.write_text((HERE / "fixtures/typing_row.json").read_text())
        report = artifact / "typing_generation_report.json"
        write_json(report, {"status": "complete"})
        teacher_rows = artifact / "typing_teacher_forced_rows.jsonl"
        teacher_rows.write_text('{"sample_id":"typing_fixture_000"}\n')
        teacher_report = artifact / "typing_teacher_forced_report.json"
        write_json(teacher_report, {"status": "complete"})
        generation_manifest = {
            "status": "complete",
            "lineage": "A",
            "target_format": "perkey",
            "model_manifest": {"lora_rank": 8},
            "rows_sha256": digest(rows),
            "report_sha256": digest(report),
        }
        generation_path = artifact / "typing_generation_manifest.json"
        write_json(generation_path, generation_manifest)
        manifest = {
            "artifact_type": "synthetic_typing_factorial_cell_eval",
            "status": "complete",
            "lineage": "A",
            "target_format": "perkey",
            "n_examples": 1,
            "generation_rows_sha256": digest(rows),
            "generation_report_sha256": digest(report),
            "generation_manifest_sha256": digest(generation_path),
            "teacher_forced_rows_sha256": digest(teacher_rows),
            "teacher_forced_report_sha256": digest(teacher_report),
        }
        write_json(artifact / "typing_eval_manifest.json", manifest)
        add_meta(artifact, "artifact_fixture_typing", "run_fixturetyping", "fixture_typing_recipe", "typing_eval_manifest.json", manifest)
        add_run_context(self.runs_root, "run_fixturetyping", "fixture_typing_recipe", artifact, "102")

    def args(self):
        return SimpleNamespace(
            eval_root=self.eval_root,
            runs_root=self.runs_root,
            rules=self.rules,
            output_dir=self.output,
        )

    def test_builds_relative_assets_and_typing_events(self):
        index, sealed = generator.build(self.args())
        self.assertEqual(index["status"], "complete", index["errors"])
        self.assertEqual(len(index["runs"]), 2)
        self.assertEqual(len(index["traces"]), 2)
        mouse = next(trace for trace in index["traces"] if trace["modality"] == "mouse")
        typing = next(trace for trace in index["traces"] if trace["modality"] == "typing")
        self.assertTrue(mouse["steps"][0]["screenshot"].startswith("data/assets/"))
        self.assertEqual(typing["steps"][0]["typing"]["events"], ["+KeyO", "-KeyO", "+KeyK", "-KeyK"])
        generator.write_bundle(self.output, index, sealed)
        self.assertTrue((self.output / "data/index.json").is_file())
        self.assertTrue((self.output / "data/assets/artifact_fixture_relative").is_symlink())
        serialized = (self.output / "data/index.json").read_text()
        self.assertNotIn(str(self.eval_root), serialized)

    def test_hash_mismatch_is_visible_and_never_partial(self):
        self.relative_rows.write_text(self.relative_rows.read_text() + "{}\n")
        index, sealed = generator.build(self.args())
        self.assertEqual(index["status"], "error")
        self.assertEqual(index["runs"], [])
        self.assertEqual(index["traces"], [])
        self.assertEqual(sealed, [])
        self.assertTrue(
            any("SHA-256 mismatch" in error for error in index["errors"]),
            index["errors"],
        )

    def test_static_client_has_no_remote_dependency(self):
        for filename in generator.STATIC_FILES:
            text = (HERE.parent / filename).read_text()
            self.assertIsNone(re.search(r"(?:src|href)=[\"']https?://", text))
            self.assertNotIn("fetch(\"http", text)


if __name__ == "__main__":
    unittest.main()
