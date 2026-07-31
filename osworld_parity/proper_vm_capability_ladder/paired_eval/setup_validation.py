"""Consume the immutable task setup-validation dependency."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import EvaluationManifest
from .readiness import DEPENDENCY_ROLES


SCHEMA_ID = "multistep_sameapp_task_setup_validation_v1"
ARTIFACT_BASENAME = "task_setup_validation.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TOKEN = object()
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


class SetupValidationError(RuntimeError):
    pass


@dataclass(frozen=True, init=False)
class ConsumedTaskSetupValidation:
    path: str
    artifact_id: str
    labctl_context_path: str
    artifact_sha256: str
    schema_id: str
    task_manifest_payload_sha256: str
    vm_snapshot_id: str
    setup_commit: str
    consumed_at: str
    artifact: dict[str, Any]
    _token: object = field(repr=False, compare=False)

    def __init__(self, *_: Any, **__: Any) -> None:
        raise TypeError(
            "ConsumedTaskSetupValidation can only be created by consuming an artifact"
        )

    @property
    def consumed(self) -> bool:
        return self._token is _TOKEN


def consume_task_setup_validation(
    path: Path,
    *,
    manifest: EvaluationManifest,
    labctl_context_path: Path,
    curriculum_manifest: Any | None = None,
) -> ConsumedTaskSetupValidation:
    """Consume one pinned artifact; never synthesize or rerun setup validation."""

    if path.name != ARTIFACT_BASENAME:
        raise SetupValidationError(
            f"setup validation path must name {ARTIFACT_BASENAME}"
        )
    artifact_id = _validate_labctl_binding(
        labctl_context_path,
        artifact_path=path,
        expected_artifact_id=manifest.expected_task_setup_validation_artifact_id,
    )
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise SetupValidationError(f"cannot consume task setup validation: {exc}") from exc
    observed_sha = hashlib.sha256(raw_bytes).hexdigest()
    if observed_sha != manifest.expected_task_setup_validation_sha256:
        raise SetupValidationError(
            f"task setup-validation hash mismatch: {observed_sha} != "
            f"{manifest.expected_task_setup_validation_sha256}"
        )
    if not isinstance(raw, dict) or set(raw) != _TOP_LEVEL_KEYS:
        raise SetupValidationError("task setup-validation top-level schema drift")

    expected = {
        "schema_version": 1,
        "schema_id": manifest.expected_task_setup_validation_schema,
        "artifact_role": "task_setup_validation",
        "artifact_id": artifact_id,
        "development_only": True,
        "heldout_inputs_present": False,
        "sealed_eval_executed": False,
        "task_manifest_payload_sha256": manifest.task_manifest_payload_sha256,
        "vm_snapshot_id": "osworld_ready",
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise SetupValidationError(f"task setup-validation binding drift: {key}")
    if raw.get("status") not in {"ready", "passed"}:
        raise SetupValidationError("task setup-validation status is not ready/passed")
    setup_commit = raw.get("setup_commit")
    if not isinstance(setup_commit, str) or not _COMMIT.fullmatch(setup_commit):
        raise SetupValidationError("task setup-validation commit is not lowercase 40-hex")

    _validate_coverage(raw, manifest)
    if curriculum_manifest is not None:
        if (
            getattr(curriculum_manifest, "manifest_payload_sha256", None)
            != manifest.task_manifest_payload_sha256
        ):
            raise SetupValidationError("curriculum/setup manifest binding mismatch")
        try:
            from ..rung2_sameapp.curriculum.setup_validation import (
                load_task_setup_validation,
            )

            loaded = load_task_setup_validation(path, curriculum_manifest)
        except Exception as exc:
            raise SetupValidationError(
                f"curriculum setup-validation loader rejected artifact: {exc}"
            ) from exc
        if loaded != raw:
            raise SetupValidationError("curriculum setup-validation loader result drift")

    values = {
        "path": str(path.resolve()),
        "artifact_id": artifact_id,
        "labctl_context_path": str(labctl_context_path.resolve()),
        "artifact_sha256": observed_sha,
        "schema_id": SCHEMA_ID,
        "task_manifest_payload_sha256": manifest.task_manifest_payload_sha256,
        "vm_snapshot_id": "osworld_ready",
        "setup_commit": setup_commit,
        "consumed_at": datetime.now(timezone.utc).isoformat(),
        "artifact": dict(raw),
        "_token": _TOKEN,
    }
    consumed = object.__new__(ConsumedTaskSetupValidation)
    for key, value in values.items():
        object.__setattr__(consumed, key, value)
    return consumed


def _validate_coverage(raw: dict[str, Any], manifest: EvaluationManifest) -> None:
    task_ids = sorted(task.task_id for task in manifest.tasks)
    coverage = raw.get("coverage")
    if not isinstance(coverage, dict) or set(coverage) != {
        "expected_task_ids",
        "validated_task_ids",
        "full_fixture_coverage",
    }:
        raise SetupValidationError("task setup-validation coverage schema drift")
    if coverage != {
        "expected_task_ids": task_ids,
        "validated_task_ids": task_ids,
        "full_fixture_coverage": True,
    }:
        raise SetupValidationError("task setup-validation coverage is incomplete")
    fixtures = raw.get("fixtures")
    if not isinstance(fixtures, list) or [
        item.get("task_id") for item in fixtures if isinstance(item, dict)
    ] != task_ids:
        raise SetupValidationError("task setup-validation fixtures are incomplete/unsorted")
    by_id = {task.task_id: task for task in manifest.tasks}
    for fixture in fixtures:
        if not isinstance(fixture, dict) or set(fixture) != {
            "task_id",
            "fixture_sha256",
            "assets",
        }:
            raise SetupValidationError("task setup-validation fixture schema drift")
        task = by_id[fixture["task_id"]]
        if fixture.get("fixture_sha256") != task.fixture_sha256:
            raise SetupValidationError(f"{task.task_id}: setup fixture hash drift")
        assets = fixture.get("assets")
        expected_assets = sorted(
            [
                {
                    "asset_id": asset["asset_id"],
                    "content_sha256": asset["content_sha256"],
                }
                for asset in task.raw.get("assets", [])
            ],
            key=lambda item: item["asset_id"],
        )
        if not expected_assets or assets != expected_assets:
            raise SetupValidationError(f"{task.task_id}: setup asset coverage drift")
        ids = [asset.get("asset_id") for asset in assets]
        hashes = [asset.get("content_sha256") for asset in assets]
        if (
            ids != sorted(set(ids))
            or len(hashes) != len(set(hashes))
            or not all(isinstance(value, str) and _SHA256.fullmatch(value) for value in hashes)
        ):
            raise SetupValidationError(f"{task.task_id}: setup assets are not unique")


def _validate_labctl_binding(
    context_path: Path,
    *,
    artifact_path: Path,
    expected_artifact_id: str,
) -> str:
    try:
        context = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SetupValidationError(f"cannot read LABCTL_CONTEXT: {exc}") from exc
    inputs = context.get("inputs") if isinstance(context, dict) else None
    if (
        not isinstance(inputs, list)
        or len(inputs) != 2
        or any(
            not isinstance(value, dict)
            or set(value) != {"role", "artifact_id", "resolved_path"}
            for value in inputs
        )
        or {value["role"] for value in inputs} != DEPENDENCY_ROLES
    ):
        raise SetupValidationError(
            "LABCTL_CONTEXT inputs must be exactly executor_readiness and task_setup_validation"
        )
    binding = next(value for value in inputs if value["role"] == "task_setup_validation")
    if binding.get("artifact_id") != expected_artifact_id:
        raise SetupValidationError("task setup-validation artifact ID mismatch")
    root = binding.get("resolved_path")
    if not isinstance(root, str) or not root:
        raise SetupValidationError("task setup-validation resolved_path is missing")
    if artifact_path.resolve() != Path(root).resolve() / ARTIFACT_BASENAME:
        raise SetupValidationError(
            "explicit task_setup_validation.json is not the registered artifact"
        )
    return expected_artifact_id
