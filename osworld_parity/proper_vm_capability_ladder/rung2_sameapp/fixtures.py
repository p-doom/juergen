from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).with_name("manifests")
SPLITS = ("train", "development", "sealed_eval")
APPS = ("writer", "calc", "files", "chrome")
HORIZONS = {"writer": 4, "calc": 6, "files": 8, "chrome": 6}
SEMANTIC_STEP_BOUNDS = {"writer": (3, 3), "calc": (4, 4), "files": (3, 3), "chrome": (3, 3)}


class ManifestError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


@dataclass(frozen=True)
class Fixture:
    id: str
    app: str
    split: str
    parameter_seed: int
    semantic_steps: int
    horizon: int
    instruction: str
    params: dict[str, Any]
    expected: dict[str, Any]
    near_miss: dict[str, Any]
    fixture_sha256: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Fixture":
        return cls(**value)

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "app": self.app,
            "split": self.split,
            "parameter_seed": self.parameter_seed,
            "semantic_steps": self.semantic_steps,
            "horizon": self.horizon,
            "instruction": self.instruction,
            "params": self.params,
            "expected": self.expected,
            "near_miss": self.near_miss,
        }

    def verify_hash(self) -> None:
        observed = hashlib.sha256(canonical_json(self.unsigned_payload())).hexdigest()
        if observed != self.fixture_sha256:
            raise ManifestError(
                f"fixture hash mismatch for {self.id}: {observed} != {self.fixture_sha256}"
            )


@dataclass(frozen=True)
class FixtureManifest:
    split: str
    sealed: bool
    fixtures: tuple[Fixture, ...]
    manifest_payload_sha256: str

    def by_id(self, fixture_id: str) -> Fixture:
        matches = [item for item in self.fixtures if item.id == fixture_id]
        if len(matches) != 1:
            raise ManifestError(f"fixture id is not unique: {fixture_id!r}")
        return matches[0]


def load_manifest(split: str, root: Path = ROOT) -> FixtureManifest:
    if split not in SPLITS:
        raise ManifestError(f"unknown split: {split!r}")
    path = root / f"{split}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read fixture manifest {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("fixture manifest must be an object")
    seal = raw.pop("manifest_payload_sha256", None)
    if not isinstance(seal, str):
        raise ManifestError("manifest payload seal is missing")
    observed = hashlib.sha256(canonical_json(raw)).hexdigest()
    if observed != seal:
        raise ManifestError(f"manifest payload hash mismatch: {observed} != {seal}")
    if raw.get("schema_version") != 1:
        raise ManifestError("unsupported same-app manifest schema")
    if raw.get("suite") != "proper_vm_rung2_same_application_v1":
        raise ManifestError("same-app suite identity drift")
    if raw.get("split") != split:
        raise ManifestError("manifest filename/split mismatch")
    sealed = raw.get("sealed")
    if sealed is not (split == "sealed_eval"):
        raise ManifestError("only sealed_eval may be sealed")
    if raw.get("official_osworld_reuse") is not False:
        raise ManifestError("official OSWorld material is forbidden")
    if raw.get("observation_contract") != "instruction_and_screenshot_only":
        raise ManifestError("observation contract drift")
    values = raw.get("fixtures")
    if not isinstance(values, list):
        raise ManifestError("fixtures must be a list")
    try:
        fixtures = tuple(Fixture.from_dict(value) for value in values)
    except (KeyError, TypeError) as exc:
        raise ManifestError(f"invalid fixture row: {exc}") from exc
    _validate(split, fixtures)
    return FixtureManifest(split, bool(sealed), fixtures, seal)


def load_all_manifests(root: Path = ROOT) -> dict[str, FixtureManifest]:
    manifests = {split: load_manifest(split, root) for split in SPLITS}
    ids: set[str] = set()
    seeds: set[int] = set()
    for split, manifest in manifests.items():
        current_ids = {fixture.id for fixture in manifest.fixtures}
        current_seeds = {fixture.parameter_seed for fixture in manifest.fixtures}
        if ids & current_ids:
            raise ManifestError(f"fixture IDs overlap at split {split}")
        if seeds & current_seeds:
            raise ManifestError(f"parameter seeds overlap at split {split}")
        ids.update(current_ids)
        seeds.update(current_seeds)
    return manifests


def assert_collectable_split(split: str) -> None:
    if split == "sealed_eval":
        raise ManifestError("sealed_eval is eval-only and cannot be replayed or collected")
    if split not in {"train", "development"}:
        raise ManifestError(f"unknown collectable split: {split!r}")


def _validate(split: str, fixtures: tuple[Fixture, ...]) -> None:
    ids = [fixture.id for fixture in fixtures]
    if len(ids) != len(set(ids)):
        raise ManifestError("duplicate fixture IDs")
    seeds = [fixture.parameter_seed for fixture in fixtures]
    if len(seeds) != len(set(seeds)):
        raise ManifestError("duplicate parameter seeds")
    if {fixture.app for fixture in fixtures} != set(APPS):
        raise ManifestError("each split must contain Writer, Calc, Files, and Chrome")
    uuid = re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        re.IGNORECASE,
    )
    for fixture in fixtures:
        fixture.verify_hash()
        if fixture.split != split:
            raise ManifestError(f"row split mismatch for {fixture.id}")
        if fixture.app not in APPS:
            raise ManifestError(f"unknown app for {fixture.id}")
        if fixture.horizon != HORIZONS[fixture.app]:
            raise ManifestError(f"horizon drift for {fixture.id}")
        low, high = SEMANTIC_STEP_BOUNDS[fixture.app]
        if not low <= fixture.semantic_steps <= high or not 2 <= fixture.semantic_steps <= 4:
            raise ManifestError(f"semantic step count drift for {fixture.id}")
        serialized = json.dumps(fixture.unsigned_payload(), ensure_ascii=False).lower()
        if uuid.search(serialized):
            raise ManifestError(f"official-style UUID forbidden in {fixture.id}")
        for forbidden in ("evaluation_examples", "task_config", "osworld_train", "osworld_eval"):
            if forbidden in serialized:
                raise ManifestError(f"forbidden official-task reference in {fixture.id}")
