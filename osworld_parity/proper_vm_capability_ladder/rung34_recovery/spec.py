from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..rung1b_realapps.fixtures import Fixture, sha256_value
from ..rung1b_realapps.training.splits import materialize_tasks


MANIFEST_DIR = Path(__file__).with_name("manifests")
PERTURBATION_BY_TEMPLATE = {
    "vscode_focus_type": "wrong_focus",
    "local_document_scroll": "opposite_scroll",
    "files_drag": "wrong_file_drag",
}
RECOVERY_SLACK = 2


class RecoveryManifestError(RuntimeError):
    pass


class SealedEvaluationError(RecoveryManifestError):
    pass


@dataclass(frozen=True)
class RecoveryTask:
    id: str
    split: str
    perturbation: str
    fixture: Fixture
    base_horizon: int
    recovery_horizon: int
    task_sha256: str

    @property
    def instruction(self) -> str:
        return self.fixture.instruction


def _load_public_manifest(split: str) -> dict[str, Any]:
    path = MANIFEST_DIR / f"{split}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryManifestError(f"cannot read recovery manifest {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RecoveryManifestError("recovery manifest must be an object")
    seal = raw.pop("manifest_payload_sha256", None)
    if not isinstance(seal, str) or sha256_value(raw) != seal:
        raise RecoveryManifestError(f"recovery manifest seal mismatch: {split}")
    required = {
        "schema_version": 1,
        "suite": "roadmap_3_4_controlled_recovery",
        "split": split,
        "namespace": f"r34-{split}-",
        "base_commit": "48a54e8585eb9d6abff31e2ba6ea857c946a7d3d",
        "recovery_slack": RECOVERY_SLACK,
        "official_osworld_reuse": False,
        "policy_observation_contract": "instruction_and_screenshot_only",
        "oracle_visibility": "trainer_only",
    }
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise RecoveryManifestError(f"recovery manifest contract drift: {key}")
    if not isinstance(raw.get("tasks"), list):
        raise RecoveryManifestError("recovery manifest tasks missing")
    return raw


def load_recovery_tasks(split: str) -> tuple[RecoveryTask, ...]:
    # Fail before resolving or reading any sealed-evaluation path.
    if split == "evaluation_sealed":
        raise SealedEvaluationError(
            "sealed recovery evaluation is an opaque commitment and cannot be opened"
        )
    if split not in {"train", "development"}:
        raise RecoveryManifestError(f"unknown recovery split: {split!r}")
    raw = _load_public_manifest(split)
    fixtures = {
        (fixture.template, fixture.parameter_seed): fixture
        for fixture in materialize_tasks(split)
    }
    prefix = str(raw["namespace"])
    tasks: list[RecoveryTask] = []
    seen: set[str] = set()
    seen_cells: set[tuple[str, int]] = set()
    for row in raw["tasks"]:
        if not isinstance(row, dict):
            raise RecoveryManifestError("recovery task row must be an object")
        try:
            task_id = str(row["id"])
            template = str(row["template"])
            seed = int(row["parameter_seed"])
            perturbation = str(row["perturbation"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RecoveryManifestError(f"invalid recovery task row: {exc}") from exc
        if not task_id.startswith(prefix) or task_id in seen:
            raise RecoveryManifestError(f"non-unique/wrong-namespace task id: {task_id}")
        if perturbation != PERTURBATION_BY_TEMPLATE.get(template):
            raise RecoveryManifestError(f"perturbation/template mismatch: {task_id}")
        cell = (template, seed)
        if cell in seen_cells or cell not in fixtures:
            raise RecoveryManifestError(f"duplicate/unknown base fixture cell: {cell}")
        fixture = fixtures[cell]
        unsigned = {
            "id": task_id,
            "split": split,
            "perturbation": perturbation,
            "base_fixture_sha256": fixture.fixture_sha256,
            "base_horizon": fixture.horizon,
            "recovery_horizon": fixture.horizon + RECOVERY_SLACK,
        }
        tasks.append(
            RecoveryTask(
                task_id,
                split,
                perturbation,
                fixture,
                fixture.horizon,
                fixture.horizon + RECOVERY_SLACK,
                sha256_value(unsigned),
            )
        )
        seen.add(task_id)
        seen_cells.add(cell)
    if not tasks:
        raise RecoveryManifestError(f"empty recovery manifest: {split}")
    return tuple(tasks)


def load_sealed_commitment() -> dict[str, Any]:
    """Return public metadata only; this file intentionally contains no task rows."""
    path = MANIFEST_DIR / "evaluation_sealed.commitment.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    seal = raw.pop("manifest_payload_sha256", None)
    if not isinstance(seal, str) or sha256_value(raw) != seal:
        raise RecoveryManifestError("sealed-evaluation commitment seal mismatch")
    required = {
        "schema_version": 1,
        "suite": "roadmap_3_4_controlled_recovery",
        "split": "evaluation_sealed",
        "namespace": "r34-sealed-",
        "content_policy": "opaque_commitment_only_no_task_rows",
        "task_count": 12,
    }
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise RecoveryManifestError(f"sealed commitment contract drift: {key}")
    if "tasks" in raw or "parameter_seeds" in raw:
        raise RecoveryManifestError("sealed commitment leaked evaluation rows")
    commitment = raw.get("content_sha256")
    if not isinstance(commitment, str) or len(commitment) != 64:
        raise RecoveryManifestError("sealed content commitment missing")
    return raw
