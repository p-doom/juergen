from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..rung2_sameapp.fixtures import Fixture


CORPUS_PATH = Path(__file__).with_name("corpus.json")
APPS = ("writer", "calc", "files", "chrome")
DIFFICULTIES = ("easy", "medium", "hard")
HORIZONS = {"writer": 4, "calc": 6, "files": 8, "chrome": 6}
SEMANTIC_STEPS = {"writer": 3, "calc": 4, "files": 3, "chrome": 3}
REQUIRED_CAPABILITIES = {
    "click",
    "coalesced_type",
    "signed_vertical_scroll",
    "drag",
    "hotkey",
    "multi_step_state_change",
}


class CorpusError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class Task:
    id: str
    app: str
    split: str
    parameter_seed: int
    difficulty: str
    semantic_steps: int
    horizon: int
    instruction: str
    capabilities: tuple[str, ...]
    params: dict[str, Any]
    expected: dict[str, Any]
    near_miss: dict[str, Any]
    recovery: dict[str, Any]
    reset: dict[str, Any]
    verifier: dict[str, Any]
    fixture_sha256: str
    task_sha256: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Task":
        row = dict(value)
        row["capabilities"] = tuple(row["capabilities"])
        return cls(**row)

    def fixture_payload(self) -> dict[str, Any]:
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

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            **self.fixture_payload(),
            "difficulty": self.difficulty,
            "capabilities": list(self.capabilities),
            "recovery": self.recovery,
            "reset": self.reset,
            "verifier": self.verifier,
            "fixture_sha256": self.fixture_sha256,
        }

    def as_fixture(self) -> Fixture:
        return Fixture(**self.fixture_payload(), fixture_sha256=self.fixture_sha256)

    def verify(self) -> None:
        if self.split != "development":
            raise CorpusError(f"{self.id}: only the development split is permitted")
        if self.app not in APPS or self.difficulty not in DIFFICULTIES:
            raise CorpusError(f"{self.id}: invalid app or difficulty")
        if self.horizon != HORIZONS[self.app]:
            raise CorpusError(f"{self.id}: horizon drift")
        if self.semantic_steps != SEMANTIC_STEPS[self.app]:
            raise CorpusError(f"{self.id}: semantic-step drift")
        if "multi_step_state_change" not in self.capabilities:
            raise CorpusError(f"{self.id}: missing multi-step capability")
        if self.reset != {
            "snapshot": "osworld_ready",
            "strategy": "restore_then_seed_private_fixture",
            "reproducible_signature_required": True,
            "state_isolation": "unique_guest_root_per_task",
        }:
            raise CorpusError(f"{self.id}: reset contract drift")
        if self.verifier != {
            "module": "osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.oracle",
            "fresh_process": True,
            "machine_readable": True,
            "reset_reject": True,
            "near_miss_reject": True,
            "gold_pass": True,
        }:
            raise CorpusError(f"{self.id}: verifier contract drift")
        if set(self.recovery) != {"near_miss_class", "corrective_action"}:
            raise CorpusError(f"{self.id}: recovery opportunity is incomplete")
        if sha256_value(self.fixture_payload()) != self.fixture_sha256:
            raise CorpusError(f"{self.id}: fixture seal mismatch")
        if sha256_value(self.unsigned_payload()) != self.task_sha256:
            raise CorpusError(f"{self.id}: task seal mismatch")
        serialized = json.dumps(self.unsigned_payload(), ensure_ascii=False).lower()
        forbidden = (
            "evaluation_examples",
            "official_osworld",
            "sealed_eval",
            "test_all",
            "heldout",
            "mixed_manifest",
        )
        if any(token in serialized for token in forbidden):
            raise CorpusError(f"{self.id}: prohibited source reference")
        if re.search(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            serialized,
        ):
            raise CorpusError(f"{self.id}: benchmark-style UUID is forbidden")


@dataclass(frozen=True)
class Corpus:
    tasks: tuple[Task, ...]
    manifest_payload_sha256: str
    provenance: dict[str, Any]
    eligibility: dict[str, Any]

    def by_id(self, task_id: str) -> Task:
        matches = [task for task in self.tasks if task.id == task_id]
        if len(matches) != 1:
            raise CorpusError(f"task id is not unique: {task_id!r}")
        return matches[0]


def load_corpus(path: Path = CORPUS_PATH) -> Corpus:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(f"cannot read clean-room corpus: {exc}") from exc
    seal = raw.pop("manifest_payload_sha256", None)
    if not isinstance(seal, str) or sha256_value(raw) != seal:
        raise CorpusError("manifest payload seal mismatch")
    required = {
        "schema_version": 1,
        "suite": "cleanroom_natural_multistep_vm_development_v1",
        "split": "development",
        "task_count": 40,
        "observation_contract": "instruction_and_screenshot_only",
        "oracle_visibility": "fresh_host_process_only",
        "model_runs": False,
    }
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise CorpusError(f"manifest contract drift: {key}")
    provenance = raw.get("provenance")
    if provenance != {
        "construction": "first_principles_parameterized_local_app_primitives",
        "source_scope": "explicit_safe_development_fixture_apis_only",
        "external_benchmark_material_consumed": False,
        "test_derived": False,
    }:
        raise CorpusError("clean-room provenance contract drift")
    eligibility = raw.get("eligibility")
    if eligibility != {
        "purpose": "auxiliary_development_only",
        "stage0": False,
        "final": False,
    }:
        raise CorpusError("auxiliary-corpus eligibility contract drift")
    values = raw.get("tasks")
    if not isinstance(values, list):
        raise CorpusError("tasks must be a list")
    try:
        tasks = tuple(Task.from_dict(value) for value in values)
    except (KeyError, TypeError) as exc:
        raise CorpusError(f"invalid task row: {exc}") from exc
    if len(tasks) != 40:
        raise CorpusError("expected exactly 40 development tasks")
    if len({task.id for task in tasks}) != 40:
        raise CorpusError("task IDs are not unique")
    if len({task.parameter_seed for task in tasks}) != 40:
        raise CorpusError("parameter seeds are not unique")
    for task in tasks:
        task.verify()
    app_counts = {app: sum(task.app == app for task in tasks) for app in APPS}
    if set(app_counts.values()) != {10}:
        raise CorpusError(f"application balance drift: {app_counts}")
    capabilities = {capability for task in tasks for capability in task.capabilities}
    if not REQUIRED_CAPABILITIES <= capabilities:
        raise CorpusError(f"capability coverage drift: {sorted(capabilities)}")
    difficulty_counts = {
        difficulty: sum(task.difficulty == difficulty for task in tasks)
        for difficulty in DIFFICULTIES
    }
    if min(difficulty_counts.values()) < 10:
        raise CorpusError(f"difficulty balance drift: {difficulty_counts}")
    return Corpus(tasks, seal, dict(provenance), dict(eligibility))
