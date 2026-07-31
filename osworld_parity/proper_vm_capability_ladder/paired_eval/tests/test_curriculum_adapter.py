from __future__ import annotations

from dataclasses import make_dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from ..curriculum_adapter import (
    CURRICULUM_SUITE,
    adapt_curriculum_manifest,
    load_curriculum_evaluation_manifest,
)
from ..manifest import ManifestError
from .helpers import evaluation_manifest, ready_marker, sealed_file, task_manifest, task_row


def _validated_curriculum_task():
    row = task_row()
    row.pop("suite")
    row["natural_multistep"] = True
    row["max_action_turns"] = 4
    for step in row["semantic_steps"]:
        step.pop("value_ref", None)
        step["arguments"] = {}
    previous = row["initial_cursor"]
    for cursor, step in zip(
        row["gold_cursor_history"], row["semantic_steps"], strict=True
    ):
        cursor["target_ref"] = step["target_ref"]
        cursor["cursor_before"] = previous
        previous = cursor["cursor_after"]
    # Re-seal after adding exact curriculum-v1 fields.
    import hashlib

    from ..contracts import canonical_json

    unsigned = dict(row)
    unsigned.pop("fixture_sha256")
    row["fixture_sha256"] = hashlib.sha256(canonical_json(unsigned)).hexdigest()
    task_type = make_dataclass(
        "ValidatedCurriculumTask",
        [(key, Any) for key in row],
        frozen=True,
        namespace={"verify": lambda self: None},
    )
    return task_type(**row)


def test_exact_curriculum_adapter_joins_loader_validated_tasks(tmp_path) -> None:
    _, ready_sha = ready_marker(tmp_path / "EXECUTOR_READY.json")
    _, task_seal = task_manifest()
    evaluation_path, _ = sealed_file(
        tmp_path / "evaluation.json",
        evaluation_manifest(task_seal, ready_sha, attempts=1),
    )
    curriculum = SimpleNamespace(
        split="development",
        tasks=(_validated_curriculum_task(),),
        manifest_payload_sha256=task_seal,
    )
    manifest = adapt_curriculum_manifest(
        evaluation_path=evaluation_path,
        curriculum=curriculum,
    )
    assert manifest.task_suite == CURRICULUM_SUITE
    assert manifest.tasks[0].semantic_step_count == 2


def test_curriculum_loader_rejects_non_development_path_before_import(tmp_path) -> None:
    with pytest.raises(ManifestError, match="only explicit development.json"):
        load_curriculum_evaluation_manifest(
            tmp_path / "evaluation.json",
            tmp_path / "sealed_eval.json",
        )


def test_adapter_against_installed_semantic_curriculum_v1(tmp_path) -> None:
    curriculum_module = pytest.importorskip(
        "osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.manifests"
    )
    curriculum = curriculum_module.load_manifest("development")
    _, ready_sha = ready_marker(tmp_path / "EXECUTOR_READY.json")
    evaluation_path, _ = sealed_file(
        tmp_path / "evaluation.json",
        evaluation_manifest(
            curriculum.manifest_payload_sha256,
            ready_sha,
            attempts=1,
        ),
    )
    manifest = adapt_curriculum_manifest(
        evaluation_path=evaluation_path,
        curriculum=curriculum,
    )
    assert len(manifest.tasks) == 5
    assert {task.app for task in manifest.tasks} == {
        "writer",
        "calc",
        "files",
        "chrome",
        "vscode",
    }
