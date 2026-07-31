from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


PACKAGE_ROOT = Path(__file__).resolve().parent
TRAIN_MANIFEST = PACKAGE_ROOT / "manifests" / "train.json"
DEVELOPMENT_MANIFEST = PACKAGE_ROOT / "manifests" / "development.json"
SEALED_EVALUATION_MANIFEST = PACKAGE_ROOT / "manifests" / "sealed_evaluation.json"

Split = Literal["train", "development", "sealed_evaluation"]
StepKind = Literal["focus", "coalesced_type", "scroll", "click", "drag"]
CANVAS = (1000, 700)

SEQUENCES: dict[str, tuple[StepKind, ...]] = {
    "focus_type": ("focus", "coalesced_type"),
    "scroll_click": ("scroll", "click"),
    "drag_click": ("drag", "click"),
    "focus_type_click": ("focus", "coalesced_type", "click"),
    "focus_type_scroll": ("focus", "coalesced_type", "scroll"),
    "focus_type_drag": ("focus", "coalesced_type", "drag"),
    "scroll_drag_click": ("scroll", "drag", "click"),
    "focus_type_scroll_click": (
        "focus",
        "coalesced_type",
        "scroll",
        "click",
    ),
}


class ManifestError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ManifestCell:
    task_id: str
    sequence_id: str
    semantic_step_count: int
    common_action_horizon: int
    seed: int | None
    slot_sha256: str


@dataclass(frozen=True)
class TaskManifest:
    split: Split
    access: str
    generator_version: str
    manifest_payload_sha256: str
    cells: tuple[ManifestCell, ...]
    materialized: bool

    def task_ids(self) -> tuple[str, ...]:
        return tuple(cell.task_id for cell in self.cells)


@dataclass(frozen=True)
class SceneGeometry:
    initial_cursor: tuple[int, int]
    field_center: tuple[int, int]
    click_center: tuple[int, int]
    decoy_center: tuple[int, int]
    drag_start: tuple[int, int]
    drag_end: tuple[int, int]

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "initial_cursor": list(self.initial_cursor),
            "field_center": list(self.field_center),
            "click_center": list(self.click_center),
            "decoy_center": list(self.decoy_center),
            "drag_start": list(self.drag_start),
            "drag_end": list(self.drag_end),
        }


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    split: Literal["train", "development", "sealed_evaluation"]
    sequence_id: str
    steps: tuple[StepKind, ...]
    semantic_step_count: int
    horizon: int
    parameter_seed: int
    instruction: str
    geometry: SceneGeometry
    target_text: str
    initial_text: str
    scroll_clicks: int
    task_sha256: str

    def public_reset_payload(self) -> dict[str, Any]:
        """Policy-visible reset payload; hidden targets are deliberately absent."""
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "semantic_step_count": self.semantic_step_count,
            "horizon": self.horizon,
            "canvas": list(CANVAS),
        }

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "sequence_id": self.sequence_id,
            "steps": list(self.steps),
            "semantic_step_count": self.semantic_step_count,
            "horizon": self.horizon,
            "parameter_seed": self.parameter_seed,
            "instruction": self.instruction,
            "geometry": self.geometry.as_dict(),
            "target_text": self.target_text,
            "initial_text": self.initial_text,
            "scroll_clicks": self.scroll_clicks,
        }


def _cell_payload(cell: dict[str, Any], *, sealed: bool) -> dict[str, Any]:
    keys = (
        "task_id",
        "sequence_id",
        "semantic_step_count",
        "common_action_horizon",
    )
    value = {key: cell[key] for key in keys}
    if not sealed:
        value["seed"] = cell["seed"]
    return value


def load_manifest(path: Path) -> TaskManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("manifest must be an object")
    seal = raw.pop("manifest_payload_sha256", None)
    if not isinstance(seal, str) or payload_sha256(raw) != seal:
        raise ManifestError(f"manifest seal mismatch: {path}")
    if raw.get("schema_version") != 1:
        raise ManifestError("unsupported mixed-action manifest schema")
    if raw.get("suite") != "roadmap_3_3_mixed_action_short_vm":
        raise ManifestError("manifest suite mismatch")
    if raw.get("official_osworld_reuse") is not False:
        raise ManifestError("official OSWorld reuse must remain false")
    split = raw.get("split")
    if split not in {"train", "development", "sealed_evaluation"}:
        raise ManifestError(f"invalid split: {split!r}")
    sealed = split == "sealed_evaluation"
    access = raw.get("access")
    if sealed:
        if access != "sealed_commitments_only":
            raise ManifestError("sealed evaluation access drift")
        if raw.get("materialized") is not False:
            raise ManifestError("sealed evaluation payload must not be materialized")
        if "seed" in canonical_bytes(raw).decode("utf-8").lower():
            raise ManifestError("sealed evaluation metadata exposes seed material")
    elif access != "trainer_authorized":
        raise ManifestError("train/development access drift")
    values = raw.get("cells")
    if not isinstance(values, list) or not values:
        raise ManifestError("manifest cells missing")
    cells: list[ManifestCell] = []
    for value in values:
        if not isinstance(value, dict):
            raise ManifestError("manifest cell must be an object")
        sequence_id = str(value.get("sequence_id", ""))
        if sequence_id not in SEQUENCES:
            raise ManifestError(f"unknown sequence {sequence_id!r}")
        sequence = SEQUENCES[sequence_id]
        if not 2 <= len(sequence) <= 4:
            raise ManifestError(f"sequence length is outside 2--4: {sequence_id}")
        if int(value.get("semantic_step_count", -1)) != len(sequence):
            raise ManifestError(f"semantic step count drift: {sequence_id}")
        horizon = common_action_horizon(sequence)
        if int(value.get("common_action_horizon", -1)) != horizon:
            raise ManifestError(f"action horizon drift: {sequence_id}")
        cell_payload = _cell_payload(value, sealed=sealed)
        slot_sha = str(value.get("slot_sha256", ""))
        if payload_sha256(cell_payload) != slot_sha:
            raise ManifestError(f"cell seal mismatch: {value.get('task_id')}")
        seed = None if sealed else int(value["seed"])
        cells.append(
            ManifestCell(
                task_id=str(value["task_id"]),
                sequence_id=sequence_id,
                semantic_step_count=len(sequence),
                common_action_horizon=horizon,
                seed=seed,
                slot_sha256=slot_sha,
            )
        )
    ids = [cell.task_id for cell in cells]
    if len(ids) != len(set(ids)):
        raise ManifestError("duplicate task ids")
    seeds = [cell.seed for cell in cells if cell.seed is not None]
    if len(seeds) != len(set(seeds)):
        raise ManifestError("duplicate parameter seeds")
    return TaskManifest(
        split=split,
        access=str(access),
        generator_version=str(raw.get("generator_version")),
        manifest_payload_sha256=seal,
        cells=tuple(cells),
        materialized=bool(raw.get("materialized")),
    )


def common_action_horizon(sequence: tuple[StepKind, ...]) -> int:
    # Native drag is move-to-start + drag-to-end. Compact drag is explicit
    # press, held movement, release. All other semantic steps are one turn.
    base = len(sequence)
    compact_extra = 2 * sequence.count("drag")
    return base + compact_extra


def _instruction(steps: tuple[StepKind, ...], seed: int) -> str:
    target_text = f"München μ-{seed}"
    clauses = {
        "focus": "focus the note field",
        "coalesced_type": f"type {target_text!r} as one text action",
        "scroll": "scroll the activity panel in the requested direction",
        "click": f"click Confirm {seed % 997:03d}",
        "drag": "drag the level handle to its marked destination",
    }
    return "Complete in order: " + "; then ".join(clauses[step] for step in steps) + "."


def _make_task(
    cell: ManifestCell, split: Literal["train", "development"]
) -> TaskDefinition:
    if cell.seed is None:
        raise ManifestError("sealed cell cannot be materialized")
    seed = cell.seed
    rng = random.Random(seed ^ 0x3300A17)
    field = (310 + rng.randint(-35, 35), 180 + rng.randint(-20, 20))
    click = (760 + rng.randint(-35, 35), 570 + rng.randint(-20, 20))
    decoy = (760 + rng.randint(-35, 35), 470 + rng.randint(-20, 20))
    drag_y = 360 + rng.randint(-25, 25)
    drag_left = 430 + rng.randint(-30, 20)
    drag_right = 760 + rng.randint(-20, 30)
    if rng.choice((False, True)):
        drag_start, drag_end = (drag_left, drag_y), (drag_right, drag_y)
    else:
        drag_start, drag_end = (drag_right, drag_y), (drag_left, drag_y)
    geometry = SceneGeometry(
        initial_cursor=(rng.randint(60, 180), rng.randint(80, 620)),
        field_center=field,
        click_center=click,
        decoy_center=decoy,
        drag_start=drag_start,
        drag_end=drag_end,
    )
    steps = SEQUENCES[cell.sequence_id]
    draft = {
        "task_id": cell.task_id,
        "split": split,
        "sequence_id": cell.sequence_id,
        "steps": list(steps),
        "semantic_step_count": len(steps),
        "horizon": cell.common_action_horizon,
        "parameter_seed": seed,
        "instruction": _instruction(steps, seed),
        "geometry": geometry.as_dict(),
        "target_text": f"München μ-{seed}",
        "initial_text": f"draft-{seed % 101}",
        "scroll_clicks": rng.choice((-7, -5, 5, 7)),
    }
    return TaskDefinition(
        task_id=cell.task_id,
        split=split,
        sequence_id=cell.sequence_id,
        steps=steps,
        semantic_step_count=len(steps),
        horizon=cell.common_action_horizon,
        parameter_seed=seed,
        instruction=str(draft["instruction"]),
        geometry=geometry,
        target_text=str(draft["target_text"]),
        initial_text=str(draft["initial_text"]),
        scroll_clicks=int(draft["scroll_clicks"]),
        task_sha256=payload_sha256(draft),
    )


def materialize_tasks(manifest: TaskManifest) -> tuple[TaskDefinition, ...]:
    if manifest.split == "sealed_evaluation" or not manifest.materialized:
        raise ManifestError(
            "sealed evaluation payload is owner-held and cannot be materialized "
            "by pre-gate ROADMAP 3.3 code"
        )
    split: Literal["train", "development"] = manifest.split
    tasks = tuple(_make_task(cell, split) for cell in manifest.cells)
    for task in tasks:
        if payload_sha256(task.unsigned_payload()) != task.task_sha256:
            raise ManifestError(
                f"task generation was not deterministic: {task.task_id}"
            )
    return tasks


def load_authorized_tasks(
    split: Literal["train", "development"]
) -> tuple[TaskDefinition, ...]:
    path = TRAIN_MANIFEST if split == "train" else DEVELOPMENT_MANIFEST
    return materialize_tasks(load_manifest(path))
