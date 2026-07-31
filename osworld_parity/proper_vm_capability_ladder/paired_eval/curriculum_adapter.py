"""Strict adapter for semantic curriculum schema v1 (commit 1ff594c)."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .contracts import canonical_json
from .manifest import (
    EvaluationManifest,
    ManifestError,
    _read_json_object,
    _remove_and_validate_seal,
    validate_evaluation_manifest,
)


CURRICULUM_SUITE = "proper_vm_sameapp_semantic_curriculum_v1"
CURRICULUM_SCHEMA_VERSION = 1
CURRICULUM_SPLIT = "development"
CURRICULUM_TASK_FIELDS = {
    "schema_version",
    "task_id",
    "family_id",
    "app",
    "split",
    "parameter_seed",
    "instruction",
    "natural_multistep",
    "semantic_steps",
    "semantic_step_count",
    "max_action_turns",
    "snapshot",
    "assets",
    "params",
    "expected",
    "near_miss",
    "verifier",
    "initial_cursor",
    "gold_cursor_history",
    "coverage",
    "exclusions",
    "transport_requirements",
    "reset_contract",
    "fixture_sha256",
}


def load_curriculum_evaluation_manifest(
    evaluation_path: Path,
    task_manifest_path: Path,
) -> EvaluationManifest:
    """Load the exact v1 curriculum without discovering any other split.

    ``task_manifest_path`` must explicitly identify the materialized
    ``development.json``.  Its parent is passed to the curriculum's own sealed
    loader; the adapter never lists a directory and never asks for a sealed or
    held-out split.
    """

    if task_manifest_path.name != "development.json":
        raise ManifestError(
            "semantic curriculum adapter accepts only explicit development.json"
        )
    try:
        from ..rung2_sameapp.curriculum.manifests import load_manifest
    except ImportError as exc:
        raise ManifestError("semantic curriculum v1 is not installed") from exc

    curriculum = load_manifest(CURRICULUM_SPLIT, root=task_manifest_path.parent)
    return adapt_curriculum_manifest(
        evaluation_path=evaluation_path,
        curriculum=curriculum,
    )


def adapt_curriculum_manifest(
    *,
    evaluation_path: Path,
    curriculum: Any,
) -> EvaluationManifest:
    """Join a loader-validated ``TaskManifest`` to the paired config."""

    if getattr(curriculum, "split", None) != CURRICULUM_SPLIT:
        raise ManifestError("curriculum adapter received a non-development split")
    seal = getattr(curriculum, "manifest_payload_sha256", None)
    tasks = getattr(curriculum, "tasks", None)
    if not isinstance(tasks, tuple) or not tasks:
        raise ManifestError("curriculum adapter requires a non-empty task tuple")
    rows: list[dict[str, Any]] = []
    for task in tasks:
        verify = getattr(task, "verify", None)
        if not callable(verify):
            raise ManifestError("curriculum task does not expose schema verification")
        verify()
        row = json.loads(canonical_json(asdict(task)))
        if set(row) != CURRICULUM_TASK_FIELDS:
            raise ManifestError("semantic curriculum task field set drift")
        if row.get("schema_version") != CURRICULUM_SCHEMA_VERSION:
            raise ManifestError("semantic curriculum task schema drift")
        if row.get("split") != CURRICULUM_SPLIT:
            raise ManifestError("semantic curriculum contains a non-development task")
        if row.get("natural_multistep") is not True:
            raise ManifestError("semantic curriculum task is not natural multistep")
        if not 2 <= int(row.get("semantic_step_count", 0)) <= 4:
            raise ManifestError("semantic curriculum task is not 2-4 semantic steps")
        if int(row.get("max_action_turns", 0)) < int(row["semantic_step_count"]):
            raise ManifestError("semantic curriculum max_action_turns is invalid")
        for step in row.get("semantic_steps", []):
            if not isinstance(step, dict) or set(step) != {
                "step_id",
                "intent",
                "target_ref",
                "capabilities",
                "arguments",
            }:
                raise ManifestError("semantic curriculum step field set drift")
        for cursor in row.get("gold_cursor_history", []):
            if not isinstance(cursor, dict) or set(cursor) != {
                "prefix_length",
                "step_id",
                "target_ref",
                "cursor_before",
                "cursor_after",
            }:
                raise ManifestError("semantic curriculum cursor field set drift")
        rows.append(row)

    evaluation_raw, evaluation_seal = _remove_and_validate_seal(
        _read_json_object(evaluation_path, "evaluation manifest"),
        label="evaluation",
    )
    task_raw = {
        "schema_version": CURRICULUM_SCHEMA_VERSION,
        "suite": CURRICULUM_SUITE,
        "split": CURRICULUM_SPLIT,
        "sealed": False,
        "tasks": rows,
    }
    result = validate_evaluation_manifest(
        evaluation_raw,
        task_raw,
        evaluation_manifest_payload_sha256=evaluation_seal,
        task_manifest_payload_sha256=seal,
    )
    if result.task_suite != CURRICULUM_SUITE:
        raise ManifestError("semantic curriculum suite identity drift")
    return result
