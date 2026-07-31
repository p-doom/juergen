"""Action-format-independent task records for the same-app curriculum.

The records in this module describe goals, assets, semantic steps, reset state,
and verifier contracts.  Native-absolute and compact-relative encodings are a
runtime concern and deliberately do not appear in :class:`SemanticTask`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from ..fixtures import canonical_json


SCHEMA_VERSION = 1
MATERIALIZED_SPLITS = ("train", "development")
APPS = ("writer", "calc", "files", "chrome", "vscode")
PRIMARY_CAPABILITIES = (
    "click",
    "signed_vertical_scroll",
    "drag",
    "coalesced_type",
    "hotkey",
)
EXCLUSIONS = ("horizontal_scroll", "timing_sensitive_double_click")


class CurriculumSchemaError(RuntimeError):
    """A curriculum task cannot be proven to satisfy the frozen contract."""


@dataclass(frozen=True)
class SemanticStep:
    step_id: int
    intent: str
    target_ref: str
    capabilities: tuple[str, ...]
    arguments: dict[str, Any]


@dataclass(frozen=True)
class CursorMilestone:
    prefix_length: int
    step_id: int
    target_ref: str
    cursor_before: tuple[int, int]
    cursor_after: tuple[int, int]


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    kind: str
    generator: str
    seed: int
    content_sha256: str


@dataclass(frozen=True)
class SemanticTask:
    schema_version: int
    task_id: str
    family_id: str
    app: str
    split: str
    parameter_seed: int
    instruction: str
    natural_multistep: bool
    semantic_steps: tuple[SemanticStep, ...]
    semantic_step_count: int
    max_action_turns: int
    snapshot: dict[str, Any]
    assets: tuple[AssetSpec, ...]
    params: dict[str, Any]
    expected: dict[str, Any]
    near_miss: dict[str, Any]
    verifier: dict[str, Any]
    initial_cursor: tuple[int, int]
    gold_cursor_history: tuple[CursorMilestone, ...]
    coverage: dict[str, Any]
    exclusions: tuple[str, ...]
    transport_requirements: dict[str, Any]
    reset_contract: dict[str, Any]
    fixture_sha256: str

    def unsigned_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("fixture_sha256")
        return payload

    def verify(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise CurriculumSchemaError(f"{self.task_id}: schema version drift")
        if self.split not in MATERIALIZED_SPLITS:
            raise CurriculumSchemaError(f"{self.task_id}: non-materialized split")
        if self.app not in APPS:
            raise CurriculumSchemaError(f"{self.task_id}: unsupported app")
        if self.natural_multistep is not True:
            raise CurriculumSchemaError(f"{self.task_id}: task is not natural multistep")
        if self.semantic_step_count != len(self.semantic_steps):
            raise CurriculumSchemaError(f"{self.task_id}: semantic-step count mismatch")
        if not 2 <= self.semantic_step_count <= 4:
            raise CurriculumSchemaError(f"{self.task_id}: task must have 2-4 semantic steps")
        if self.max_action_turns < self.semantic_step_count:
            raise CurriculumSchemaError(f"{self.task_id}: action horizon is too small")
        if tuple(step.step_id for step in self.semantic_steps) != tuple(
            range(1, self.semantic_step_count + 1)
        ):
            raise CurriculumSchemaError(f"{self.task_id}: semantic steps are not contiguous")
        if tuple(item.prefix_length for item in self.gold_cursor_history) != tuple(
            range(1, self.semantic_step_count + 1)
        ):
            raise CurriculumSchemaError(f"{self.task_id}: cursor history is not prefix-aligned")
        for step, cursor in zip(
            self.semantic_steps, self.gold_cursor_history, strict=True
        ):
            if (step.step_id, step.target_ref) != (cursor.step_id, cursor.target_ref):
                raise CurriculumSchemaError(
                    f"{self.task_id}: cursor milestone does not identify its semantic step"
                )
        if self.snapshot.get("id") != "osworld_ready":
            raise CurriculumSchemaError(f"{self.task_id}: snapshot id drift")
        if self.snapshot.get("reset_strategy") != "restore_snapshot_then_seeded_setup":
            raise CurriculumSchemaError(f"{self.task_id}: reset strategy drift")
        if self.verifier.get("fresh_process") is not True:
            raise CurriculumSchemaError(f"{self.task_id}: final verifier is not isolated")
        if tuple(self.exclusions) != EXCLUSIONS:
            raise CurriculumSchemaError(f"{self.task_id}: unsupported-action exclusions drift")
        if self.reset_contract != {
            "reset_reject": True,
            "near_miss_reject": True,
            "gold_pass": True,
            "reproducible_reset": True,
            "fresh_process_final_oracle": True,
            "zero_held_inputs": True,
        }:
            raise CurriculumSchemaError(f"{self.task_id}: reset/oracle contract drift")
        capabilities = set(self.coverage.get("primary_capabilities", ()))
        if not capabilities <= set(PRIMARY_CAPABILITIES):
            raise CurriculumSchemaError(f"{self.task_id}: non-Phase-B primary capability")
        signs = self.coverage.get("signed_vertical_scroll", [])
        if not isinstance(signs, list) or not set(signs) <= {"up", "down"}:
            raise CurriculumSchemaError(f"{self.task_id}: invalid vertical scroll sign")
        serialized = json.dumps(self.unsigned_payload(), ensure_ascii=False).lower()
        for forbidden in (
            "official_osworld_reuse\":true",
            "evaluation_examples",
            "task_config",
            "test_all.json",
        ):
            if forbidden in serialized.replace(" ", ""):
                raise CurriculumSchemaError(
                    f"{self.task_id}: forbidden official/held input reference"
                )
        observed = hashlib.sha256(canonical_json(self.unsigned_payload())).hexdigest()
        if observed != self.fixture_sha256:
            raise CurriculumSchemaError(
                f"{self.task_id}: fixture seal mismatch {observed} != {self.fixture_sha256}"
            )


def seal_task(values: dict[str, Any]) -> SemanticTask:
    """Construct and seal one deterministic task from family-generated values."""

    unsigned = dict(values)
    unsigned["schema_version"] = SCHEMA_VERSION
    digest = hashlib.sha256(canonical_json(unsigned)).hexdigest()
    task = SemanticTask(
        schema_version=SCHEMA_VERSION,
        task_id=str(unsigned["task_id"]),
        family_id=str(unsigned["family_id"]),
        app=str(unsigned["app"]),
        split=str(unsigned["split"]),
        parameter_seed=int(unsigned["parameter_seed"]),
        instruction=str(unsigned["instruction"]),
        natural_multistep=bool(unsigned["natural_multistep"]),
        semantic_steps=tuple(
            SemanticStep(
                step_id=int(item["step_id"]),
                intent=str(item["intent"]),
                target_ref=str(item["target_ref"]),
                capabilities=tuple(item["capabilities"]),
                arguments=dict(item.get("arguments", {})),
            )
            for item in unsigned["semantic_steps"]
        ),
        semantic_step_count=int(unsigned["semantic_step_count"]),
        max_action_turns=int(unsigned["max_action_turns"]),
        snapshot=dict(unsigned["snapshot"]),
        assets=tuple(
            AssetSpec(
                asset_id=str(item["asset_id"]),
                kind=str(item["kind"]),
                generator=str(item["generator"]),
                seed=int(item["seed"]),
                content_sha256=str(item["content_sha256"]),
            )
            for item in unsigned["assets"]
        ),
        params=dict(unsigned["params"]),
        expected=dict(unsigned["expected"]),
        near_miss=dict(unsigned["near_miss"]),
        verifier=dict(unsigned["verifier"]),
        initial_cursor=tuple(unsigned["initial_cursor"]),
        gold_cursor_history=tuple(
            CursorMilestone(
                prefix_length=int(item["prefix_length"]),
                step_id=int(item["step_id"]),
                target_ref=str(item["target_ref"]),
                cursor_before=tuple(item["cursor_before"]),
                cursor_after=tuple(item["cursor_after"]),
            )
            for item in unsigned["gold_cursor_history"]
        ),
        coverage=dict(unsigned["coverage"]),
        exclusions=tuple(unsigned["exclusions"]),
        transport_requirements=dict(unsigned["transport_requirements"]),
        reset_contract=dict(unsigned["reset_contract"]),
        fixture_sha256=digest,
    )
    task.verify()
    return task
