"""Fail-closed reader for immutable development setup-validation artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .manifests import TaskManifest
from .schema import CurriculumSchemaError


ARTIFACT_BASENAME = "task_setup_validation.json"
SCHEMA_ID = "multistep_sameapp_task_setup_validation_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "schema_id",
    "artifact_role",
    "artifact_id",
    "status",
    "development_only",
    "heldout_inputs_present",
    "sealed_eval_executed",
    "task_manifest_payload_sha256",
    "vm_snapshot_id",
    "setup_commit",
    "fixtures",
    "coverage",
}


def load_task_setup_validation(
    path: Path, manifest: TaskManifest
) -> dict[str, Any]:
    """Read one already-produced artifact; never synthesize or rerun setup."""

    if path.name != ARTIFACT_BASENAME:
        raise CurriculumSchemaError(
            f"setup validation basename must be {ARTIFACT_BASENAME!r}"
        )
    if manifest.split != "development":
        raise CurriculumSchemaError("setup validation is development-only")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CurriculumSchemaError(f"cannot read setup validation: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != _TOP_LEVEL_KEYS:
        raise CurriculumSchemaError("setup validation top-level schema drift")
    required = {
        "schema_version": 1,
        "schema_id": SCHEMA_ID,
        "artifact_role": "task_setup_validation",
        "development_only": True,
        "heldout_inputs_present": False,
        "sealed_eval_executed": False,
        "task_manifest_payload_sha256": manifest.manifest_payload_sha256,
        "vm_snapshot_id": "osworld_ready",
    }
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise CurriculumSchemaError(f"setup validation binding drift: {key}")
    if not isinstance(raw.get("artifact_id"), str) or not raw["artifact_id"]:
        raise CurriculumSchemaError("setup validation artifact ID is missing")
    if raw.get("status") not in {"ready", "passed"}:
        raise CurriculumSchemaError("setup validation status is not ready/passed")
    if not isinstance(raw.get("setup_commit"), str) or not _COMMIT.fullmatch(
        raw["setup_commit"]
    ):
        raise CurriculumSchemaError("setup validation commit is not lowercase 40-hex")

    expected_ids = sorted(task.task_id for task in manifest.tasks)
    coverage = raw.get("coverage")
    if not isinstance(coverage, dict) or set(coverage) != {
        "expected_task_ids",
        "validated_task_ids",
        "full_fixture_coverage",
    }:
        raise CurriculumSchemaError("setup validation coverage schema drift")
    if coverage.get("full_fixture_coverage") is not True:
        raise CurriculumSchemaError("setup validation is not full-fixture coverage")
    for field in ("expected_task_ids", "validated_task_ids"):
        values = coverage.get(field)
        if values != expected_ids or values != sorted(set(values or [])):
            raise CurriculumSchemaError(f"setup validation {field} is incomplete/unsorted")

    fixtures = raw.get("fixtures")
    if not isinstance(fixtures, list):
        raise CurriculumSchemaError("setup validation fixtures are missing")
    fixture_ids = [row.get("task_id") for row in fixtures if isinstance(row, dict)]
    if fixture_ids != expected_ids or fixture_ids != sorted(set(fixture_ids)):
        raise CurriculumSchemaError("setup validation fixture rows are incomplete/unsorted")
    tasks = {task.task_id: task for task in manifest.tasks}
    for row in fixtures:
        if not isinstance(row, dict) or set(row) != {
            "task_id",
            "fixture_sha256",
            "assets",
        }:
            raise CurriculumSchemaError("setup validation fixture schema drift")
        task = tasks[row["task_id"]]
        if row["fixture_sha256"] != task.fixture_sha256 or not _SHA256.fullmatch(
            str(row["fixture_sha256"])
        ):
            raise CurriculumSchemaError(f"{task.task_id}: fixture hash drift")
        assets = row["assets"]
        expected_assets = sorted(
            (
                {"asset_id": asset.asset_id, "content_sha256": asset.content_sha256}
                for asset in task.assets
            ),
            key=lambda item: item["asset_id"],
        )
        if not isinstance(assets, list) or assets != expected_assets:
            raise CurriculumSchemaError(f"{task.task_id}: asset coverage drift")
        asset_ids = [asset.get("asset_id") for asset in assets]
        asset_hashes = [asset.get("content_sha256") for asset in assets]
        if asset_ids != sorted(set(asset_ids)) or len(asset_hashes) != len(
            set(asset_hashes)
        ) or not all(
            isinstance(value, str) and _SHA256.fullmatch(value)
            for value in asset_hashes
        ):
            raise CurriculumSchemaError(f"{task.task_id}: asset IDs/hashes are not unique")
    return raw
