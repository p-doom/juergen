from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST_PATH = Path(__file__).with_name("fixtures.json")
TEMPLATES = ("click", "focus_type", "scroll", "drag")
HORIZONS = {"click": 2, "focus_type": 3, "scroll": 2, "drag": 4}


class FixtureManifestError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


@dataclass(frozen=True)
class Fixture:
    id: str
    template: str
    split: str
    parameter_seed: int
    horizon: int
    instruction: str
    params: dict[str, Any]
    expected: dict[str, Any]
    fixture_sha256: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Fixture":
        return cls(**value)

    def unsigned_payload(self) -> dict[str, Any]:
        value = {
            "id": self.id,
            "template": self.template,
            "split": self.split,
            "parameter_seed": self.parameter_seed,
            "horizon": self.horizon,
            "instruction": self.instruction,
            "params": self.params,
            "expected": self.expected,
        }
        return value

    def verify_hash(self) -> None:
        observed = hashlib.sha256(_canonical(self.unsigned_payload())).hexdigest()
        if observed != self.fixture_sha256:
            raise FixtureManifestError(
                f"fixture hash mismatch for {self.id}: {observed} != {self.fixture_sha256}"
            )


@dataclass(frozen=True)
class FixtureManifest:
    fixtures: tuple[Fixture, ...]
    manifest_payload_sha256: str

    def by_id(self, fixture_id: str) -> Fixture:
        matches = [item for item in self.fixtures if item.id == fixture_id]
        if len(matches) != 1:
            raise FixtureManifestError(f"fixture id is not unique: {fixture_id!r}")
        return matches[0]

    def select(self, *, split: str | None = None) -> tuple[Fixture, ...]:
        if split is None:
            return self.fixtures
        return tuple(item for item in self.fixtures if item.split == split)


def load_manifest(path: Path = MANIFEST_PATH) -> FixtureManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureManifestError(f"cannot read fixture manifest {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise FixtureManifestError("fixture manifest must be a JSON object")
    seal = raw.pop("manifest_payload_sha256", None)
    if not isinstance(seal, str):
        raise FixtureManifestError("fixture manifest seal is missing")
    observed_seal = hashlib.sha256(_canonical(raw)).hexdigest()
    if observed_seal != seal:
        raise FixtureManifestError(
            f"manifest payload hash mismatch: {observed_seal} != {seal}"
        )
    if raw.get("schema_version") != 1:
        raise FixtureManifestError("unsupported fixture schema")
    if raw.get("suite") != "rung1a_instrumented_browser_microbench":
        raise FixtureManifestError("suite must remain explicitly classified as rung1a")
    if raw.get("official_osworld_reuse") is not False:
        raise FixtureManifestError("official OSWorld reuse must be false")
    if raw.get("policy_observation_contract") != "instruction_and_screenshot_only":
        raise FixtureManifestError("policy observation contract drift")
    values = raw.get("fixtures")
    if not isinstance(values, list):
        raise FixtureManifestError("fixtures must be a list")
    try:
        fixtures = tuple(Fixture.from_dict(value) for value in values)
    except (TypeError, KeyError) as exc:
        raise FixtureManifestError(f"invalid fixture record: {exc}") from exc
    _validate_fixtures(fixtures)
    return FixtureManifest(fixtures=fixtures, manifest_payload_sha256=seal)


def _validate_fixtures(fixtures: tuple[Fixture, ...]) -> None:
    ids = [item.id for item in fixtures]
    if len(ids) != len(set(ids)):
        raise FixtureManifestError("duplicate fixture ids")
    seeds = [(item.template, item.split, item.parameter_seed) for item in fixtures]
    if len(seeds) != len(set(seeds)):
        raise FixtureManifestError("duplicate template/split/seed cells")
    uuid_pattern = re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        re.IGNORECASE,
    )
    for fixture in fixtures:
        fixture.verify_hash()
        if fixture.template not in TEMPLATES:
            raise FixtureManifestError(f"unknown template: {fixture.template}")
        if fixture.split not in {"development", "evaluation"}:
            raise FixtureManifestError(f"unknown split: {fixture.split}")
        if fixture.horizon != HORIZONS[fixture.template]:
            raise FixtureManifestError(f"horizon drift for {fixture.id}")
        serialized = json.dumps(fixture.unsigned_payload(), ensure_ascii=False).lower()
        if uuid_pattern.search(serialized):
            raise FixtureManifestError(f"official-style UUID forbidden: {fixture.id}")
        for forbidden in (
            "evaluation_examples",
            "task_config",
            "heldout",
            "osworld_train",
            "osworld_eval",
        ):
            if forbidden in serialized:
                raise FixtureManifestError(
                    f"forbidden official-task reference {forbidden!r} in {fixture.id}"
                )
    for template in TEMPLATES:
        development = [
            item for item in fixtures if item.template == template and item.split == "development"
        ]
        evaluation = [
            item for item in fixtures if item.template == template and item.split == "evaluation"
        ]
        if len(development) != 2 or len(evaluation) != 8:
            raise FixtureManifestError(
                f"{template}: expected 2 development and 8 evaluation fixtures, "
                f"got {len(development)} and {len(evaluation)}"
            )
