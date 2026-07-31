from __future__ import annotations

import json
from pathlib import Path

import pytest

from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.manifests import (
    load_manifest,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.schema import (
    CurriculumSchemaError,
)
from osworld_parity.proper_vm_capability_ladder.rung2_sameapp.curriculum.setup_validation import (
    ARTIFACT_BASENAME,
    SCHEMA_ID,
    load_task_setup_validation,
)


def _payload():
    manifest = load_manifest("development")
    task_ids = sorted(task.task_id for task in manifest.tasks)
    return manifest, {
        "schema_version": 1,
        "schema_id": SCHEMA_ID,
        "artifact_role": "task_setup_validation",
        "artifact_id": "artifact-task-setup-1",
        "status": "passed",
        "development_only": True,
        "heldout_inputs_present": False,
        "sealed_eval_executed": False,
        "task_manifest_payload_sha256": manifest.manifest_payload_sha256,
        "vm_snapshot_id": "osworld_ready",
        "setup_commit": "a" * 40,
        "fixtures": [
            {
                "task_id": task.task_id,
                "fixture_sha256": task.fixture_sha256,
                "assets": sorted(
                    [
                        {
                            "asset_id": asset.asset_id,
                            "content_sha256": asset.content_sha256,
                        }
                        for asset in task.assets
                    ],
                    key=lambda item: item["asset_id"],
                ),
            }
            for task in sorted(manifest.tasks, key=lambda task: task.task_id)
        ],
        "coverage": {
            "expected_task_ids": task_ids,
            "validated_task_ids": task_ids,
            "full_fixture_coverage": True,
        },
    }


def test_setup_validation_is_exactly_bound_to_development_manifest(tmp_path: Path) -> None:
    manifest, payload = _payload()
    path = tmp_path / ARTIFACT_BASENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_task_setup_validation(path, manifest) == payload


def test_setup_validation_rejects_partial_fixture_coverage(tmp_path: Path) -> None:
    manifest, payload = _payload()
    payload["fixtures"].pop()
    path = tmp_path / ARTIFACT_BASENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CurriculumSchemaError, match="incomplete/unsorted"):
        load_task_setup_validation(path, manifest)


def test_setup_validation_rejects_mutable_or_unpinned_setup(tmp_path: Path) -> None:
    manifest, payload = _payload()
    payload["setup_commit"] = "HEAD"
    path = tmp_path / ARTIFACT_BASENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CurriculumSchemaError, match="lowercase 40-hex"):
        load_task_setup_validation(path, manifest)
