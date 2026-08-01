from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STAGE0_INVENTORY_PATH = Path(__file__).with_name("stage0_inventory.json")
ANCHOR_APPS = ("writer", "calc", "files", "chrome", "vscode")
MODES = ("single", "multi")
CELLS_PER_ANCHOR_MODE = 4
DIFFICULTY_BY_CELL = {1: "easy", 2: "medium", 3: "hard", 4: "medium"}

RESET_CONTRACT = {
    "snapshot": "osworld_ready",
    "strategy": "restore_then_seed_private_fixture",
    "reproducible_signature_required": True,
    "state_isolation": "unique_guest_root_per_task",
}
SOURCE_VERIFIER = {
    "module": "osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.oracle",
    "fresh_process": True,
    "machine_readable": True,
    "reset_reject": True,
    "near_miss_reject": True,
    "gold_pass": True,
}
RECORD_ELIGIBILITY = {
    "purpose": "natural_dev_stage0",
    "stage0": True,
    "final": False,
}
DEVELOPMENT = {
    "inventory_role": "natural_dev_stage0",
    "development_only": True,
    "stage0_eligible": True,
    "final_eligible": False,
    "composition_balance": "5_apps_x_2_modes_x_4_cells",
}
SOURCE_POLICY = {
    "deny_before_open": True,
    "construction": "first_principles_local_app_primitives_only",
    "external_benchmark_material_consumed": False,
    "model_output_material_consumed": False,
    "prior_task_material_consumed": False,
    "network_material_consumed": False,
}

_FORBIDDEN_REFERENCES = (
    "evaluation_examples",
    "official_osworld",
    "sealed_eval",
    "test_all",
    "mixed_manifest",
    "natural_panel",
)
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


class Stage0InventoryError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def _expect_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise Stage0InventoryError(
            f"{context}: key drift missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )


@dataclass(frozen=True)
class Stage0SourceTask:
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
    def from_dict(cls, value: dict[str, Any]) -> "Stage0SourceTask":
        _expect_keys(
            value,
            {
                "id", "app", "split", "parameter_seed", "difficulty",
                "semantic_steps", "horizon", "instruction", "capabilities",
                "params", "expected", "near_miss", "recovery", "reset",
                "verifier", "fixture_sha256", "task_sha256",
            },
            "source task",
        )
        row = dict(value)
        row["capabilities"] = tuple(row["capabilities"])
        task = cls(**row)
        task.verify()
        return task

    def to_dict(self) -> dict[str, Any]:
        row = dict(self.__dict__)
        row["capabilities"] = list(self.capabilities)
        return row

    def verify(self) -> None:
        if not self.id.startswith("cln-s0-src-"):
            raise Stage0InventoryError(f"{self.id}: invalid clean-room source ID")
        if self.app not in ANCHOR_APPS or self.split != "development":
            raise Stage0InventoryError(f"{self.id}: app/split drift")
        if not 920000 <= self.parameter_seed < 925000:
            raise Stage0InventoryError(f"{self.id}: source parameter seed outside reserved range")
        if self.semantic_steps not in {3, 4} or self.horizon < self.semantic_steps:
            raise Stage0InventoryError(f"{self.id}: semantic-step/horizon drift")
        if self.difficulty not in set(DIFFICULTY_BY_CELL.values()):
            raise Stage0InventoryError(f"{self.id}: invalid difficulty")
        if not self.instruction.strip() or len(self.capabilities) != len(set(self.capabilities)):
            raise Stage0InventoryError(f"{self.id}: instruction/capability drift")
        if "multi_step_state_change" not in self.capabilities:
            raise Stage0InventoryError(f"{self.id}: missing multi-step capability")
        if self.reset != RESET_CONTRACT or self.verifier != SOURCE_VERIFIER:
            raise Stage0InventoryError(f"{self.id}: reset/verifier drift")
        if set(self.recovery) != {"near_miss_class", "corrective_action"}:
            raise Stage0InventoryError(f"{self.id}: recovery drift")
        fixture_keys = (
            "id", "app", "split", "parameter_seed", "semantic_steps", "horizon",
            "instruction", "params", "expected", "near_miss",
        )
        fixture = {key: self.to_dict()[key] for key in fixture_keys}
        if sha256_value(fixture) != self.fixture_sha256:
            raise Stage0InventoryError(f"{self.id}: fixture seal mismatch")
        if sha256_value(_without(self.to_dict(), "task_sha256")) != self.task_sha256:
            raise Stage0InventoryError(f"{self.id}: task seal mismatch")
        required_caps = {
            "writer": {"click", "coalesced_type", "hotkey"},
            "calc": {"click", "coalesced_type", "hotkey"},
            "files": {"click", "drag", "coalesced_type", "hotkey"},
            "chrome": {"click", "signed_vertical_scroll"},
            "vscode": {"click", "coalesced_type", "hotkey"},
        }[self.app]
        if not required_caps <= set(self.capabilities):
            raise Stage0InventoryError(f"{self.id}: app capability drift")
        required_params = {
            "writer": {"file_name", "initial_text"},
            "calc": {"file_name", "cell", "initial_value"},
            "files": {"source_name", "destination_name", "decoy_name", "content"},
            "chrome": {"port", "section", "setting", "initial_scroll_y", "scroll_direction", "minimum_scroll_delta"},
            "vscode": {"file_name", "initial_text"},
        }[self.app]
        required_expected = {
            "writer": {"text", "bold"},
            "calc": {"formula", "display_value"},
            "files": {"destination", "final_name"},
            "chrome": {"section", "setting_enabled"},
            "vscode": {"text"},
        }[self.app]
        if set(self.params) != required_params or set(self.expected) != required_expected:
            raise Stage0InventoryError(f"{self.id}: app parameter/expectation drift")
        if set(self.near_miss) != required_expected or self.expected == self.near_miss:
            raise Stage0InventoryError(f"{self.id}: near-miss drift")

    def as_fixture(self) -> Any:
        if self.app == "vscode":
            raise ValueError("VS Code uses the rung1b fixture adapter")
        from ..rung2_sameapp.fixtures import Fixture

        return Fixture(
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

    def as_vscode_fixture(self) -> Any:
        if self.app != "vscode":
            raise ValueError("only VS Code uses the rung1b fixture adapter")
        from ..rung1b_realapps.fixtures import Fixture

        return Fixture(
            id=self.id,
            template="vscode_focus_type",
            split=self.split,
            parameter_seed=self.parameter_seed,
            horizon=self.horizon,
            instruction=self.instruction,
            params=self.params,
            expected=self.expected,
            near_miss=self.near_miss,
            gate_role="capability_probe",
            coverage_label="cleanroom_stage0_development",
            fixture_sha256=self.fixture_sha256,
        )


@dataclass(frozen=True)
class Stage0Record:
    id: str
    mode: str
    bridge: str
    anchor_app: str
    ordered_components: tuple[dict[str, Any], ...]
    eligibility: dict[str, Any]
    instruction: str
    parameter_seed: int
    difficulty: str
    semantic_steps: int
    capabilities: tuple[str, ...]
    reset: dict[str, Any]
    verifier: dict[str, Any]
    record_sha256: str
    source_task: Stage0SourceTask | None = None
    source_task_payload_sha256: str | None = None
    source_tasks: tuple[Stage0SourceTask, ...] = ()
    source_task_payload_sha256s: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Stage0Record":
        mode = value.get("mode")
        common = {
            "id", "mode", "bridge", "anchor_app", "ordered_components",
            "eligibility", "instruction", "parameter_seed", "difficulty",
            "semantic_steps", "capabilities", "reset", "verifier", "record_sha256",
        }
        variant = (
            {"source_task", "source_task_payload_sha256"}
            if mode == "single"
            else {"source_tasks", "source_task_payload_sha256s"}
        )
        _expect_keys(value, common | variant, "record")
        row = dict(value)
        row["ordered_components"] = tuple(dict(item) for item in row["ordered_components"])
        row["capabilities"] = tuple(row["capabilities"])
        if mode == "single":
            row["source_task"] = Stage0SourceTask.from_dict(row["source_task"])
        elif mode == "multi":
            row["source_tasks"] = tuple(Stage0SourceTask.from_dict(item) for item in row["source_tasks"])
            row["source_task_payload_sha256s"] = tuple(row["source_task_payload_sha256s"])
        else:
            raise Stage0InventoryError("record: invalid mode")
        task = cls(**row)
        task.verify()
        return task

    @property
    def component_tasks(self) -> tuple[Stage0SourceTask, ...]:
        if self.mode == "single":
            if self.source_task is None:
                raise Stage0InventoryError(f"{self.id}: missing single source task")
            return (self.source_task,)
        return self.source_tasks

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": self.id,
            "mode": self.mode,
            "bridge": self.bridge,
            "anchor_app": self.anchor_app,
            "ordered_components": list(self.ordered_components),
            "eligibility": self.eligibility,
            "instruction": self.instruction,
            "parameter_seed": self.parameter_seed,
            "difficulty": self.difficulty,
            "semantic_steps": self.semantic_steps,
            "capabilities": list(self.capabilities),
            "reset": self.reset,
            "verifier": self.verifier,
        }
        if self.mode == "single":
            row["source_task"] = self.source_task.to_dict() if self.source_task else None
            row["source_task_payload_sha256"] = self.source_task_payload_sha256
        else:
            row["source_tasks"] = [task.to_dict() for task in self.source_tasks]
            row["source_task_payload_sha256s"] = list(self.source_task_payload_sha256s)
        return {**row, "record_sha256": self.record_sha256}

    def verify(self) -> None:
        if self.anchor_app not in ANCHOR_APPS or self.mode not in MODES:
            raise Stage0InventoryError(f"{self.id}: anchor/mode drift")
        if self.bridge != f"{self.mode}_app":
            raise Stage0InventoryError(f"{self.id}: bridge drift")
        match = re.fullmatch(r"cln-s0-dev-([a-z]+)-(single|multi)-c(\d\d)", self.id)
        if not match or match.group(1) != self.anchor_app:
            raise Stage0InventoryError(f"{self.id}: record ID drift")
        cell = int(match.group(3))
        if cell not in DIFFICULTY_BY_CELL or self.difficulty != DIFFICULTY_BY_CELL[cell]:
            raise Stage0InventoryError(f"{self.id}: balance-cell drift")
        if not 910000 <= self.parameter_seed < 911000:
            raise Stage0InventoryError(f"{self.id}: record seed outside reserved range")
        if self.eligibility != RECORD_ELIGIBILITY or self.reset != RESET_CONTRACT:
            raise Stage0InventoryError(f"{self.id}: eligibility/reset drift")
        tasks = self.component_tasks
        expected_count = 1 if self.mode == "single" else 2
        if len(tasks) != expected_count:
            raise Stage0InventoryError(f"{self.id}: source-task count drift")
        if tasks[0].app != self.anchor_app:
            raise Stage0InventoryError(f"{self.id}: anchor source drift")
        if self.mode == "multi" and tasks[1].app == self.anchor_app:
            raise Stage0InventoryError(f"{self.id}: multi-app partner drift")
        for order, task in enumerate(tasks, start=1):
            expected_id = f"cln-s0-src-{self.anchor_app}-{self.mode}-c{cell:02d}-{order:02d}"
            if task.id != expected_id:
                raise Stage0InventoryError(f"{self.id}: ordered source ID drift")
        expected_components = tuple(
            {
                "order": order,
                "app": task.app,
                "task_id": task.id,
                "semantic_steps": task.semantic_steps if self.mode == "single" else 1,
            }
            for order, task in enumerate(tasks, start=1)
        )
        if self.ordered_components != expected_components:
            raise Stage0InventoryError(f"{self.id}: ordered-components drift")
        expected_steps = tasks[0].semantic_steps if self.mode == "single" else 2
        if self.semantic_steps != expected_steps or not 2 <= self.semantic_steps <= 4:
            raise Stage0InventoryError(f"{self.id}: record semantic-step drift")
        required_caps = {cap for task in tasks for cap in task.capabilities} | {"multi_step_state_change"}
        if self.mode == "multi":
            required_caps.add("app_switch")
            if "Alt+Tab" not in self.instruction:
                raise Stage0InventoryError(f"{self.id}: visible switch instruction missing")
        if set(self.capabilities) != required_caps:
            raise Stage0InventoryError(f"{self.id}: capability union drift")
        source_ids = [task.id for task in tasks]
        expected_verifier = {
            "module": "osworld_parity.proper_vm_capability_ladder.natural_dev_cleanroom.stage0_oracle",
            "kind": "fresh_composed_private_state",
            "fresh_process": True,
            "machine_readable": True,
            "composition": "ordered_all_components",
            "component_source_task_ids": source_ids,
            "reset_reject": True,
            "near_miss_reject": True,
            "gold_pass": True,
        }
        if self.verifier != expected_verifier:
            raise Stage0InventoryError(f"{self.id}: composed verifier drift")
        payloads = [task.to_dict() for task in tasks]
        seals = [sha256_value(payload) for payload in payloads]
        if self.mode == "single":
            if self.source_task_payload_sha256 != seals[0]:
                raise Stage0InventoryError(f"{self.id}: source payload seal mismatch")
        elif tuple(seals) != self.source_task_payload_sha256s:
            raise Stage0InventoryError(f"{self.id}: source payload seals mismatch")
        if sha256_value(_without(self.to_dict(), "record_sha256")) != self.record_sha256:
            raise Stage0InventoryError(f"{self.id}: record seal mismatch")


@dataclass(frozen=True)
class Stage0Inventory:
    tasks: tuple[Stage0Record, ...]
    manifest_payload_sha256: str
    development: dict[str, Any]
    source_policy: dict[str, Any]

    @property
    def eligibility(self) -> dict[str, Any]:
        return dict(RECORD_ELIGIBILITY)

    def by_id(self, record_id: str) -> Stage0Record:
        matches = [task for task in self.tasks if task.id == record_id]
        if len(matches) != 1:
            raise Stage0InventoryError(f"record ID is not unique: {record_id}")
        return matches[0]

    def source_by_id(self, source_id: str) -> Stage0SourceTask:
        matches = [source for record in self.tasks for source in record.component_tasks if source.id == source_id]
        if len(matches) != 1:
            raise Stage0InventoryError(f"source-task ID is not unique: {source_id}")
        return matches[0]


def load_stage0_inventory(path: Path = STAGE0_INVENTORY_PATH) -> Stage0Inventory:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage0InventoryError(f"cannot read Stage0 inventory: {exc}") from exc
    _expect_keys(
        document,
        {
            "schema_version", "suite", "split", "stage", "development_only",
            "task_count", "anchor_apps", "modes", "cells_per_anchor_mode",
            "development", "source_policy", "tasks", "manifest_payload_sha256",
        },
        "manifest",
    )
    seal = document["manifest_payload_sha256"]
    payload = _without(document, "manifest_payload_sha256")
    if sha256_value(payload) != seal:
        raise Stage0InventoryError("manifest payload seal mismatch")
    required = {
        "schema_version": 1,
        "suite": "cleanroom_natural_dev_stage0_v1",
        "split": "development",
        "stage": "stage0",
        "development_only": True,
        "task_count": 40,
        "anchor_apps": list(ANCHOR_APPS),
        "modes": list(MODES),
        "cells_per_anchor_mode": CELLS_PER_ANCHOR_MODE,
        "development": DEVELOPMENT,
        "source_policy": SOURCE_POLICY,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise Stage0InventoryError(f"manifest contract drift: {key}")
    tasks = tuple(Stage0Record.from_dict(value) for value in payload["tasks"])
    if len(tasks) != 40 or len({task.id for task in tasks}) != 40:
        raise Stage0InventoryError("expected exactly 40 unique Stage0 records")
    sources = [source for task in tasks for source in task.component_tasks]
    all_ids = [task.id for task in tasks] + [source.id for source in sources]
    all_seeds = [task.parameter_seed for task in tasks] + [source.parameter_seed for source in sources]
    if len(all_ids) != len(set(all_ids)) or len(all_seeds) != len(set(all_seeds)):
        raise Stage0InventoryError("record/source IDs or parameter seeds are not globally unique")
    balance = {
        (app, mode, cell): sum(
            task.anchor_app == app
            and task.mode == mode
            and task.id.endswith(f"-c{cell:02d}")
            for task in tasks
        )
        for app in ANCHOR_APPS
        for mode in MODES
        for cell in range(1, CELLS_PER_ANCHOR_MODE + 1)
    }
    if set(balance.values()) != {1}:
        raise Stage0InventoryError(f"5 x 2 x 4 balance drift: {balance}")
    for anchor in ANCHOR_APPS:
        partners = {
            task.component_tasks[1].app
            for task in tasks
            if task.anchor_app == anchor and task.mode == "multi"
        }
        if partners != set(ANCHOR_APPS) - {anchor}:
            raise Stage0InventoryError(f"{anchor}: partner coverage drift")
    # The reserved ranges are already disjoint by construction; compare with
    # both allowed local development inventories so later edits fail closed.
    from .schema import load_corpus
    from .smoke_schema import load_smoke

    prior_ids = {task.id for task in load_corpus().tasks} | {task.id for task in load_smoke().tasks}
    prior_seeds = {task.parameter_seed for task in load_corpus().tasks} | {task.parameter_seed for task in load_smoke().tasks}
    if set(all_ids) & prior_ids or set(all_seeds) & prior_seeds:
        raise Stage0InventoryError("Stage0 inventory overlaps an allowed prior development inventory")
    serialized = canonical_json(document).decode("utf-8").lower()
    if any(token in serialized for token in _FORBIDDEN_REFERENCES) or _UUID_RE.search(serialized):
        raise Stage0InventoryError("manifest contains a prohibited imported reference or UUID")
    return Stage0Inventory(tasks, seal, dict(DEVELOPMENT), dict(SOURCE_POLICY))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the sealed clean-room Stage0 inventory")
    parser.add_argument("path", nargs="?", type=Path, default=STAGE0_INVENTORY_PATH)
    args = parser.parse_args(argv)
    inventory = load_stage0_inventory(args.path)
    print(json.dumps({"path": str(args.path), "task_count": len(inventory.tasks), "manifest_payload_sha256": inventory.manifest_payload_sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
