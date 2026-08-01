from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..rung1b_realapps.fixtures import Fixture as VscodeFixture
from ..rung2_sameapp.fixtures import Fixture as SameAppFixture
from .schema import CorpusError, Task, load_corpus, sha256_value


SMOKE_PATH = Path(__file__).with_name("plumbing_smoke.json")
SMOKE_APPS = ("writer", "calc", "files", "chrome", "vscode")


@dataclass(frozen=True)
class SmokeTask:
    id: str
    mode: str
    bridge: str
    anchor_app: str
    ordered_components: tuple[dict[str, Any], ...]
    eligibility: dict[str, Any]
    source_task: dict[str, Any]
    source_task_payload_sha256: str
    record_sha256: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SmokeTask":
        row = dict(value)
        row["ordered_components"] = tuple(dict(item) for item in row["ordered_components"])
        return cls(**row)

    @property
    def app(self) -> str:
        return str(self.source_task["app"])

    @property
    def split(self) -> str:
        return str(self.source_task["split"])

    @property
    def parameter_seed(self) -> int:
        return int(self.source_task["parameter_seed"])

    @property
    def difficulty(self) -> str:
        return str(self.source_task["difficulty"])

    @property
    def semantic_steps(self) -> int:
        return int(self.source_task["semantic_steps"])

    @property
    def horizon(self) -> int:
        return int(self.source_task["horizon"])

    @property
    def instruction(self) -> str:
        return str(self.source_task["instruction"])

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(self.source_task["capabilities"])

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.source_task["params"])

    @property
    def expected(self) -> dict[str, Any]:
        return dict(self.source_task["expected"])

    @property
    def near_miss(self) -> dict[str, Any]:
        return dict(self.source_task["near_miss"])

    @property
    def fixture_sha256(self) -> str:
        return str(self.source_task["fixture_sha256"])

    @property
    def task_sha256(self) -> str:
        return str(self.source_task["task_sha256"])

    def unsigned_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "bridge": self.bridge,
            "anchor_app": self.anchor_app,
            "ordered_components": list(self.ordered_components),
            "eligibility": self.eligibility,
            "source_task": self.source_task,
            "source_task_payload_sha256": self.source_task_payload_sha256,
        }

    def verify(self) -> None:
        if self.mode != "single" or self.bridge != "single_app":
            raise CorpusError(f"{self.id}: plumbing smoke must bind the single-app stratum")
        if self.anchor_app != self.app or self.app not in SMOKE_APPS:
            raise CorpusError(f"{self.id}: anchor-app drift")
        if self.eligibility != {
            "purpose": "plumbing_smoke_only",
            "stage0": False,
            "final": False,
        }:
            raise CorpusError(f"{self.id}: smoke-only eligibility drift")
        expected_components = (
            {
                "order": 1,
                "app": self.app,
                "task_id": self.id,
                "semantic_steps": self.semantic_steps,
            },
        )
        if self.ordered_components != expected_components:
            raise CorpusError(f"{self.id}: ordered component drift")
        if self.source_task.get("id") != self.id:
            raise CorpusError(f"{self.id}: source-task identity drift")
        if sha256_value(self.source_task) != self.source_task_payload_sha256:
            raise CorpusError(f"{self.id}: source-task payload seal mismatch")
        if sha256_value(self.unsigned_record()) != self.record_sha256:
            raise CorpusError(f"{self.id}: smoke record seal mismatch")
        if self.app == "vscode":
            if self.split != "development" or self.semantic_steps != 3 or self.horizon != 4:
                raise CorpusError(f"{self.id}: VS Code smoke task contract drift")
            if self.source_task.get("reset") != {
                "snapshot": "osworld_ready",
                "strategy": "restore_then_seed_private_fixture",
                "reproducible_signature_required": True,
                "state_isolation": "unique_guest_root_per_task",
            }:
                raise CorpusError(f"{self.id}: VS Code reset contract drift")
            if sha256_value(
                {
                    key: self.source_task[key]
                    for key in (
                        "id", "app", "split", "parameter_seed", "semantic_steps",
                        "horizon", "instruction", "params", "expected", "near_miss"
                    )
                }
            ) != self.fixture_sha256:
                raise CorpusError(f"{self.id}: VS Code fixture seal mismatch")
        else:
            Task.from_dict(self.source_task).verify()

    def as_fixture(self) -> SameAppFixture:
        if self.app == "vscode":
            raise ValueError("VS Code uses the rung1b fixture adapter")
        return SameAppFixture(
            id=self.id,
            app=self.app,
            split=self.split,
            parameter_seed=self.parameter_seed,
            semantic_steps=self.semantic_steps,
            horizon=self.horizon,
            instruction=self.instruction,
            params=self.params,
            expected=self.expected,
            near_miss=self.near_miss,
            fixture_sha256=self.fixture_sha256,
        )

    def as_vscode_fixture(self) -> VscodeFixture:
        if self.app != "vscode":
            raise ValueError("only the VS Code smoke task uses this adapter")
        return VscodeFixture(
            id=self.id,
            template="vscode_focus_type",
            split="development",
            parameter_seed=self.parameter_seed,
            horizon=self.horizon,
            instruction=self.instruction,
            params=self.params,
            expected=self.expected,
            near_miss=self.near_miss,
            gate_role="capability_probe",
            coverage_label="cleanroom_plumbing_smoke_probe",
            fixture_sha256=self.fixture_sha256,
        )


@dataclass(frozen=True)
class SmokeInventory:
    tasks: tuple[SmokeTask, ...]
    manifest_payload_sha256: str
    provenance: dict[str, Any]
    eligibility: dict[str, Any]

    def by_id(self, task_id: str) -> SmokeTask:
        matches = [task for task in self.tasks if task.id == task_id]
        if len(matches) != 1:
            raise CorpusError(f"smoke task id is not unique: {task_id!r}")
        return matches[0]


def load_smoke(path: Path = SMOKE_PATH) -> SmokeInventory:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(f"cannot read clean-room plumbing smoke: {exc}") from exc
    seal = raw.pop("inventory_payload_sha256", None)
    if not isinstance(seal, str) or sha256_value(raw) != seal:
        raise CorpusError("plumbing-smoke inventory payload seal mismatch")
    if set(raw) != {
        "schema_version",
        "suite",
        "status",
        "split",
        "development_only",
        "role",
        "source_policy",
        "tasks",
    }:
        raise CorpusError("plumbing-smoke top-level schema drift")
    required = {
        "schema_version": 1,
        "suite": "natural_dev_disjoint_smoke_inventory_v1",
        "status": "authored",
        "split": "development",
        "development_only": True,
        "role": "plumbing_smoke_only",
    }
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise CorpusError(f"plumbing-smoke contract drift: {key}")
    source_policy = raw.get("source_policy")
    if source_policy != {
        "construction": "first_principles_parameterized_local_app_primitives",
        "deny_before_open": True,
        "external_benchmark_material_consumed": False,
        "external_rollout_systems_used": False,
        "model_runs": False,
        "source_scope": "explicit_safe_development_fixture_apis_only",
        "test_derived": False,
    }:
        raise CorpusError("plumbing-smoke source policy drift")
    values = raw.get("tasks")
    if not isinstance(values, list) or len(values) != 5:
        raise CorpusError("plumbing-smoke inventory must contain exactly five tasks")
    tasks = tuple(SmokeTask.from_dict(value) for value in values)
    if {task.anchor_app for task in tasks} != set(SMOKE_APPS):
        raise CorpusError("plumbing-smoke must contain one task per anchor app")
    if len({task.id for task in tasks}) != 5 or len({task.parameter_seed for task in tasks}) != 5:
        raise CorpusError("plumbing-smoke IDs/seeds are not unique")
    for task in tasks:
        task.verify()
    stage0 = load_corpus()
    if {task.id for task in tasks} & {task.id for task in stage0.tasks}:
        raise CorpusError("plumbing-smoke task IDs overlap the 40-task corpus")
    if {task.parameter_seed for task in tasks} & {task.parameter_seed for task in stage0.tasks}:
        raise CorpusError("plumbing-smoke seeds overlap the 40-task corpus")
    eligibility = {"purpose": "plumbing_smoke_only", "stage0": False, "final": False}
    return SmokeInventory(tasks, seal, dict(source_policy), eligibility)
