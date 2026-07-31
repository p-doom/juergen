from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST_PATH = Path(__file__).with_name("fixtures.json")
TEMPLATES = ("vscode_focus_type", "local_document_scroll", "files_drag")
HORIZONS = {"vscode_focus_type": 4, "local_document_scroll": 2, "files_drag": 3}


class FixtureManifestError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


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
    near_miss: dict[str, Any]
    gate_role: str
    coverage_label: str
    fixture_sha256: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Fixture":
        return cls(**value)

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "template": self.template,
            "split": self.split,
            "parameter_seed": self.parameter_seed,
            "horizon": self.horizon,
            "instruction": self.instruction,
            "params": self.params,
            "expected": self.expected,
            "near_miss": self.near_miss,
            "gate_role": self.gate_role,
            "coverage_label": self.coverage_label,
        }

    def verify_hash(self) -> None:
        observed = sha256_value(self.unsigned_payload())
        if observed != self.fixture_sha256:
            raise FixtureManifestError(
                f"fixture hash mismatch for {self.id}: {observed} != {self.fixture_sha256}"
            )


@dataclass(frozen=True)
class FixtureManifest:
    fixtures: tuple[Fixture, ...]
    manifest_payload_sha256: str

    def by_id(self, fixture_id: str) -> Fixture:
        matches = [fixture for fixture in self.fixtures if fixture.id == fixture_id]
        if len(matches) != 1:
            raise FixtureManifestError(f"fixture id is not unique: {fixture_id!r}")
        return matches[0]

    def select(self, *, split: str = "development") -> tuple[Fixture, ...]:
        return tuple(fixture for fixture in self.fixtures if fixture.split == split)


def load_manifest(path: Path = MANIFEST_PATH) -> FixtureManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureManifestError(f"cannot read fixture manifest {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise FixtureManifestError("fixture manifest must be an object")
    seal = raw.pop("manifest_payload_sha256", None)
    if not isinstance(seal, str) or sha256_value(raw) != seal:
        raise FixtureManifestError("manifest payload hash mismatch")
    required = {
        "schema_version": 1,
        "suite": "rung1b_real_application_development",
        "split_policy": "development_only_no_evaluation_rows",
        "official_osworld_reuse": False,
        "policy_observation_contract": "instruction_and_screenshot_only",
        "oracle_visibility": "fresh_host_process_only",
    }
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise FixtureManifestError(f"manifest contract drift: {key}")
    values = raw.get("fixtures")
    if not isinstance(values, list):
        raise FixtureManifestError("fixtures must be a list")
    try:
        fixtures = tuple(Fixture.from_dict(value) for value in values)
    except (KeyError, TypeError) as exc:
        raise FixtureManifestError(f"invalid fixture row: {exc}") from exc
    _validate(fixtures)
    return FixtureManifest(fixtures, seal)


def _validate(fixtures: tuple[Fixture, ...]) -> None:
    if len(fixtures) != 6 or len({item.id for item in fixtures}) != len(fixtures):
        raise FixtureManifestError("expected six unique development fixtures")
    forbidden = re.compile(
        r"(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|"
        r"official[_ -]?osworld|task_config|heldout|evaluation_examples)",
        re.IGNORECASE,
    )
    for fixture in fixtures:
        fixture.verify_hash()
        if fixture.split != "development" or fixture.template not in TEMPLATES:
            raise FixtureManifestError(f"non-development/unknown fixture: {fixture.id}")
        if fixture.horizon != HORIZONS[fixture.template]:
            raise FixtureManifestError(f"horizon drift: {fixture.id}")
        if fixture.gate_role not in {"primary_gate", "capability_probe"}:
            raise FixtureManifestError(f"unlabelled gate role: {fixture.id}")
        if fixture.gate_role == "capability_probe" and not fixture.coverage_label.endswith(
            "_probe"
        ):
            raise FixtureManifestError(f"capability probe is not explicit: {fixture.id}")
        if fixture.template == "local_document_scroll":
            if fixture.coverage_label != "thin_coverage_scroll_probe":
                raise FixtureManifestError(f"scroll coverage label drift: {fixture.id}")
            if fixture.near_miss != {
                "kind": "correct_direction_undershoot",
                "direction": fixture.params["direction"],
            }:
                raise FixtureManifestError(f"scroll near miss semantics drift: {fixture.id}")
        if (
            fixture.template == "files_drag"
            and fixture.coverage_label != "thin_coverage_drag_probe"
        ):
            raise FixtureManifestError(f"drag coverage label drift: {fixture.id}")
        if forbidden.search(json.dumps(fixture.unsigned_payload(), ensure_ascii=False)):
            raise FixtureManifestError(f"forbidden benchmark reference: {fixture.id}")
    counts = {template: sum(item.template == template for item in fixtures) for template in TEMPLATES}
    if set(counts.values()) != {2}:
        raise FixtureManifestError(f"expected two fixtures per template: {counts}")
    scroll_directions = {
        str(item.params["direction"])
        for item in fixtures
        if item.template == "local_document_scroll"
    }
    if scroll_directions != {"up", "down"}:
        raise FixtureManifestError("scroll fixtures must cover both signs")
    if any("slider" in json.dumps(item.unsigned_payload()).lower() for item in fixtures):
        raise FixtureManifestError("browser-slider drag fallback is forbidden")
    primary = [item for item in fixtures if item.gate_role == "primary_gate"]
    if len(primary) != 1 or primary[0].coverage_label != "phaseb_ascii_coalesced_typing":
        raise FixtureManifestError("primary gate must be the Phase-B ASCII coalesced-type fixture")
    primary_text = str(primary[0].expected.get("text", ""))
    if not primary_text.isascii():
        raise FixtureManifestError("primary Phase-B typing fixture must be ASCII")
    unicode_probes = [
        item
        for item in fixtures
        if item.coverage_label == "unicode_coalesced_typing_probe"
    ]
    if len(unicode_probes) != 1 or unicode_probes[0].template != "vscode_focus_type":
        raise FixtureManifestError("Unicode coalesced-type probe coverage drift")
    if str(unicode_probes[0].expected.get("text", "")).isascii():
        raise FixtureManifestError("Unicode capability probe silently became ASCII")
    expected_probes = {
        "unicode_coalesced_typing_probe",
        "thin_coverage_scroll_probe",
        "thin_coverage_drag_probe",
    }
    observed_probes = {
        item.coverage_label for item in fixtures if item.gate_role == "capability_probe"
    }
    if observed_probes != expected_probes:
        raise FixtureManifestError(f"capability probe coverage drift: {observed_probes}")
