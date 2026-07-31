"""Fail-closed loaders for the two materialized curriculum splits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..fixtures import canonical_json
from .families import FAMILY_IDS, build_task
from .schema import (
    APPS,
    MATERIALIZED_SPLITS,
    PRIMARY_CAPABILITIES,
    CurriculumSchemaError,
    SemanticTask,
)


ROOT = Path(__file__).with_name("manifests")
FAMILY_REGISTRY = Path(__file__).with_name("family_commitments.json")


@dataclass(frozen=True)
class TaskManifest:
    split: str
    tasks: tuple[SemanticTask, ...]
    manifest_payload_sha256: str

    def by_id(self, task_id: str) -> SemanticTask:
        matches = [task for task in self.tasks if task.task_id == task_id]
        if len(matches) != 1:
            raise CurriculumSchemaError(f"task id is not unique: {task_id!r}")
        return matches[0]


def _read_sealed_object(path: Path, seal_key: str) -> tuple[dict[str, Any], str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CurriculumSchemaError(f"cannot read curriculum metadata {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CurriculumSchemaError(f"curriculum metadata is not an object: {path}")
    seal = raw.pop(seal_key, None)
    observed = hashlib.sha256(canonical_json(raw)).hexdigest()
    if not isinstance(seal, str) or seal != observed:
        raise CurriculumSchemaError(f"curriculum metadata seal mismatch: {path}")
    return raw, seal


def load_family_commitments(path: Path = FAMILY_REGISTRY) -> dict[str, Any]:
    raw, seal = _read_sealed_object(path, "registry_payload_sha256")
    if raw.get("schema_version") != 1:
        raise CurriculumSchemaError("family commitment schema drift")
    if raw.get("suite") != "proper_vm_sameapp_semantic_curriculum_v1":
        raise CurriculumSchemaError("family commitment suite drift")
    if raw.get("materialization_scope") != "train_and_development_only":
        raise CurriculumSchemaError("curriculum scope is not train/development-only")
    if raw.get("official_osworld_reuse") is not False:
        raise CurriculumSchemaError("official OSWorld material is forbidden")
    if raw.get("held_inputs_present") is not False:
        raise CurriculumSchemaError("curriculum must contain zero held inputs")
    if raw.get("primary_gate_capabilities") != list(PRIMARY_CAPABILITIES):
        raise CurriculumSchemaError("primary gate no longer matches Phase-B coverage")
    if raw.get("explicitly_unsupported") != [
        "horizontal_scroll",
        "timing_sensitive_double_click",
    ]:
        raise CurriculumSchemaError("unsupported-action commitment drift")
    families = raw.get("families")
    if not isinstance(families, list) or {
        item.get("family_id") for item in families if isinstance(item, dict)
    } != set(FAMILY_IDS):
        raise CurriculumSchemaError("family registry is incomplete")
    for family in families:
        if not isinstance(family, dict) or family.get("app") not in APPS:
            raise CurriculumSchemaError("invalid family commitment")
        commitments = family.get("split_commitments")
        if not isinstance(commitments, dict) or set(commitments) != {
            "train",
            "development",
            "sealed_eval",
        }:
            raise CurriculumSchemaError("family split commitments are incomplete")
        for split in MATERIALIZED_SPLITS:
            if commitments[split] != {"materialized": True, "task_count": 1}:
                raise CurriculumSchemaError(f"{family['family_id']}: {split} drift")
        sealed = commitments["sealed_eval"]
        if sealed != {
            "materialized": False,
            "task_count_commitment": 8,
            "seed_assignment": "future_external_sealer_only",
            "inputs_present": False,
        }:
            raise CurriculumSchemaError(
                f"{family['family_id']}: sealed commitment contains inputs or drifted"
            )
    raw["registry_payload_sha256"] = seal
    return raw


def load_manifest(split: str, root: Path = ROOT) -> TaskManifest:
    # Reject before path construction or I/O: a sealed manifest is intentionally absent.
    if split not in MATERIALIZED_SPLITS:
        raise CurriculumSchemaError(
            f"split {split!r} is not materialized; only train/development are accessible"
        )
    raw, seal = _read_sealed_object(root / f"{split}.json", "manifest_payload_sha256")
    required = {
        "schema_version": 1,
        "suite": "proper_vm_sameapp_semantic_curriculum_v1",
        "split": split,
        "official_osworld_reuse": False,
        "held_inputs_present": False,
        "observation_contract": "instruction_and_screenshot_only",
    }
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise CurriculumSchemaError(f"{split} manifest contract drift: {key}")
    seeds = raw.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != len(FAMILY_IDS):
        raise CurriculumSchemaError(f"{split} must seed exactly one task per family")
    tasks: list[SemanticTask] = []
    for row in seeds:
        if not isinstance(row, dict) or set(row) != {"family_id", "parameter_seed"}:
            raise CurriculumSchemaError(f"invalid {split} seed row")
        tasks.append(build_task(str(row["family_id"]), split, int(row["parameter_seed"])))
    if {task.family_id for task in tasks} != set(FAMILY_IDS):
        raise CurriculumSchemaError(f"{split} family coverage drift")
    if {task.app for task in tasks} != set(APPS):
        raise CurriculumSchemaError(f"{split} application coverage drift")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise CurriculumSchemaError(f"{split} duplicate task ids")
    if len({task.parameter_seed for task in tasks}) != len(tasks):
        raise CurriculumSchemaError(f"{split} duplicate parameter seeds")
    return TaskManifest(split, tuple(tasks), seal)


def load_materialized_curriculum() -> dict[str, TaskManifest]:
    load_family_commitments()
    manifests = {split: load_manifest(split) for split in MATERIALIZED_SPLITS}
    tasks = [task for manifest in manifests.values() for task in manifest.tasks]
    ids = [task.task_id for task in tasks]
    seeds = [task.parameter_seed for task in tasks]
    if len(ids) != len(set(ids)) or len(seeds) != len(set(seeds)):
        raise CurriculumSchemaError("train/development identity overlap")
    capabilities = {
        capability
        for task in tasks
        for capability in task.coverage["primary_capabilities"]
    }
    if capabilities != set(PRIMARY_CAPABILITIES):
        raise CurriculumSchemaError("materialized tasks do not cover the Phase-B gate")
    signs = {
        sign for task in tasks for sign in task.coverage["signed_vertical_scroll"]
    }
    if signs != {"up", "down"}:
        raise CurriculumSchemaError("materialized tasks do not cover both scroll signs")
    edge_cases = {case for task in tasks for case in task.coverage["edge_cases"]}
    if edge_cases != {"unicode", "file_drag", "ctrl_s"}:
        raise CurriculumSchemaError("edge-case labels are incomplete")
    if not all(
        task.coverage["thin_cases"]
        for task in tasks
        if task.app in {"files", "chrome", "vscode"}
    ):
        raise CurriculumSchemaError("thin single-family cases are not explicit")
    return manifests
