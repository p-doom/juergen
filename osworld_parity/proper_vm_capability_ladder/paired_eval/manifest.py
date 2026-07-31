from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    ACTION_INTERFACES,
    APPROVED_CURRICULUM_COMMIT,
    APPROVED_CURRICULUM_RUNTIME_BINDING_SCHEMA,
    ARMS,
    GOLD_PREFIX_HORIZONS,
    MODES,
    canonical_json,
)


class ManifestError(RuntimeError):
    pass


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object")
    return value


def _remove_and_validate_seal(raw: dict[str, Any], *, label: str) -> tuple[dict[str, Any], str]:
    value = dict(raw)
    seal = value.pop("manifest_payload_sha256", None)
    if not _is_sha256(seal):
        raise ManifestError(f"{label} manifest_payload_sha256 is missing or invalid")
    observed = hashlib.sha256(canonical_json(value)).hexdigest()
    if observed != seal:
        raise ManifestError(f"{label} payload hash mismatch: {observed} != {seal}")
    return value, seal


def _cursor(value: Any, *, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise ManifestError(f"{label} must be [int, int]")
    return int(value[0]), int(value[1])


@dataclass(frozen=True)
class SemanticStep:
    step_id: str
    intent: str
    target_ref: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class Task:
    task_id: str
    family_id: str
    app: str
    parameter_seed: int
    instruction: str
    snapshot_id: str
    reset_strategy: str
    verifier_kind: str
    verifier_module: str
    budget_contract: dict[str, Any]
    geometry_contract: dict[str, Any]
    initial_cursor_contract: dict[str, Any]
    semantic_steps: tuple[SemanticStep, ...]
    gold_cursor_history: tuple[dict[str, Any], ...]
    fixture_sha256: str
    raw: dict[str, Any]

    @property
    def semantic_step_count(self) -> int:
        return len(self.semantic_steps)

    def expected_target(self, prefix_length: int) -> str:
        if not 0 <= prefix_length < self.semantic_step_count:
            raise IndexError(f"no next semantic target after prefix {prefix_length}")
        return self.semantic_steps[prefix_length].target_ref

    def cursor_ref_for_prefix(self, prefix_length: int) -> str:
        if prefix_length == 0:
            return "runtime.initial_cursor"
        if not 0 < prefix_length <= self.semantic_step_count:
            raise IndexError(f"invalid gold prefix length {prefix_length}")
        previous = self.semantic_steps[prefix_length - 1]
        for index, entry in enumerate(self.gold_cursor_history, start=1):
            marker = entry.get(
                "prefix_length",
                entry.get("after_semantic_step", entry.get("step_index", index)),
            )
            marker_matches = marker == prefix_length or marker == previous.step_id
            if not marker_matches and entry.get("step_id") != previous.step_id:
                continue
            cursor_ref = entry.get("cursor_after_ref")
            if isinstance(cursor_ref, str) and cursor_ref:
                return cursor_ref
        raise ManifestError(
            f"{self.task_id} has no unambiguous gold cursor ref for prefix {prefix_length}"
        )

    @property
    def pair_primitive_action_cap(self) -> int:
        return max(
            int(value)
            for value in self.budget_contract["primitive_action_caps"].values()
        )

    @property
    def pair_primitive_event_cap(self) -> int:
        return max(
            int(value)
            for value in self.budget_contract["primitive_event_caps"].values()
        )


@dataclass(frozen=True)
class Arm:
    name: str
    action_interface: str
    checkpoint: str
    checkpoint_sha256: str
    prompt_id: str
    prompt_sha256: str
    generation: dict[str, Any]

    @property
    def system_label(self) -> str:
        return f"{self.checkpoint}@{self.prompt_id}/{self.action_interface}"


@dataclass(frozen=True)
class RuntimeBinding:
    runtime_id: str
    module: str
    factory: str
    source_sha256: str
    contract_schema: str


@dataclass(frozen=True)
class EvaluationManifest:
    suite: str
    split: str
    task_suite: str
    task_manifest_payload_sha256: str
    evaluation_manifest_payload_sha256: str
    expected_executor_ready_sha256: str
    expected_executor_ready_artifact_id: str
    expected_executor_certification_schema: str
    expected_task_setup_validation_sha256: str
    expected_task_setup_validation_artifact_id: str
    expected_task_setup_validation_schema: str
    curriculum_commit: str
    curriculum_runtime_binding_schema: str
    evaluator_commit: str
    order_seed: int
    shard_seed: int
    sampling_seed: int
    bootstrap_seed: int
    bootstrap_resamples: int
    attempts_per_cell: int
    budget: dict[str, int | float]
    modes: tuple[str, ...]
    arms: tuple[Arm, ...]
    runtime: RuntimeBinding
    tasks: tuple[Task, ...]
    excluded_task_ids: frozenset[str]

    def arm(self, name: str) -> Arm:
        matches = [arm for arm in self.arms if arm.name == name]
        if len(matches) != 1:
            raise ManifestError(f"arm is not unique: {name}")
        return matches[0]

    def task(self, task_id: str) -> Task:
        matches = [task for task in self.tasks if task.task_id == task_id]
        if len(matches) != 1:
            raise ManifestError(f"task is not unique: {task_id}")
        return matches[0]

    @property
    def comparison_label(self) -> str:
        labels = " vs ".join(self.arm(name).system_label for name in ARMS)
        return f"complete-system comparison: {labels}"


def load_evaluation_manifest(
    evaluation_path: Path,
    task_manifest_path: Path,
) -> EvaluationManifest:
    """Load, seal-check, and join an evaluation config with curriculum tasks."""

    evaluation_raw, evaluation_seal = _remove_and_validate_seal(
        _read_json_object(evaluation_path, "evaluation manifest"),
        label="evaluation",
    )
    task_raw, task_seal = _remove_and_validate_seal(
        _read_json_object(task_manifest_path, "task manifest"),
        label="task",
    )
    return validate_evaluation_manifest(
        evaluation_raw,
        task_raw,
        evaluation_manifest_payload_sha256=evaluation_seal,
        task_manifest_payload_sha256=task_seal,
    )


def validate_evaluation_manifest(
    evaluation: dict[str, Any],
    task_manifest: dict[str, Any],
    *,
    evaluation_manifest_payload_sha256: str | None = None,
    task_manifest_payload_sha256: str | None = None,
) -> EvaluationManifest:
    if evaluation.get("schema_version") != 1:
        raise ManifestError("unsupported paired-evaluation manifest schema")
    if task_manifest.get("schema_version") != 1:
        raise ManifestError("unsupported task manifest schema")
    if evaluation.get("split") != "development":
        raise ManifestError("paired evaluation is development-only")
    if evaluation.get("development_only") is not True:
        raise ManifestError("development_only must be true")
    if evaluation.get("heldout_access") is not False:
        raise ManifestError("heldout_access must be false")
    if task_manifest.get("split") != "development":
        raise ManifestError("only materialized development tasks are accepted")
    if task_manifest.get("sealed") is True:
        raise ManifestError("sealed or held-out task manifests are forbidden")
    serialized_task_manifest = json.dumps(task_manifest, ensure_ascii=False).lower()
    for forbidden in ("heldout", "held-out", "sealed_eval", "official_osworld"):
        if forbidden in serialized_task_manifest:
            raise ManifestError(f"forbidden task material marker: {forbidden}")

    expected_task_seal = evaluation.get("task_manifest_payload_sha256")
    actual_task_seal = task_manifest_payload_sha256 or hashlib.sha256(
        canonical_json(task_manifest)
    ).hexdigest()
    if expected_task_seal != actual_task_seal:
        raise ManifestError(
            "evaluation/task manifest seal mismatch: "
            f"{expected_task_seal} != {actual_task_seal}"
        )

    readiness_sha = evaluation.get("expected_executor_ready_sha256")
    if not _is_sha256(readiness_sha):
        raise ManifestError("expected_executor_ready_sha256 must pin a readiness marker")
    readiness_artifact_id = evaluation.get("expected_executor_ready_artifact_id")
    if not isinstance(readiness_artifact_id, str) or not readiness_artifact_id:
        raise ManifestError("expected_executor_ready_artifact_id is required")
    readiness_schema = evaluation.get("expected_executor_certification_schema")
    if readiness_schema != "proper_vm_executor_cert_v1":
        raise ManifestError("expected executor certification schema drift")
    setup_sha = evaluation.get("expected_task_setup_validation_sha256")
    if not _is_sha256(setup_sha):
        raise ManifestError("expected_task_setup_validation_sha256 must pin an artifact")
    setup_artifact_id = evaluation.get("expected_task_setup_validation_artifact_id")
    if not isinstance(setup_artifact_id, str) or not setup_artifact_id:
        raise ManifestError("expected_task_setup_validation_artifact_id is required")
    setup_schema = evaluation.get("expected_task_setup_validation_schema")
    if setup_schema != "multistep_sameapp_task_setup_validation_v1":
        raise ManifestError("expected task setup-validation schema drift")
    curriculum_commit = evaluation.get("curriculum_commit")
    if curriculum_commit != APPROVED_CURRICULUM_COMMIT:
        raise ManifestError("evaluation does not pin the approved curriculum commit")
    binding_schema = evaluation.get("curriculum_runtime_binding_schema")
    if binding_schema != APPROVED_CURRICULUM_RUNTIME_BINDING_SCHEMA:
        raise ManifestError("evaluation curriculum runtime-binding schema drift")
    evaluator_commit = evaluation.get("evaluator_commit")
    if not _is_git_commit(evaluator_commit):
        raise ManifestError("evaluator_commit must be a lowercase 40-hex commit")

    raw_budget = evaluation.get("budget")
    if not isinstance(raw_budget, dict):
        raise ManifestError("one common pair budget is required")
    budget = _validate_budget(raw_budget)

    raw_arms = evaluation.get("arms")
    if not isinstance(raw_arms, list) or len(raw_arms) != 2:
        raise ManifestError("exactly two arms are required")
    arms = tuple(_parse_arm(value) for value in raw_arms)
    if {arm.name for arm in arms} != set(ARMS):
        raise ManifestError(f"arms must be exactly {ARMS}")
    for value in raw_arms:
        if any(key in value for key in ("budget", "seed", "snapshot", "cursor")):
            raise ManifestError("budget, seed, snapshot, and cursor must not vary by arm")
    runtime = _parse_runtime(evaluation.get("runtime"))

    raw_modes = evaluation.get("modes")
    if not isinstance(raw_modes, list) or set(raw_modes) != set(MODES):
        raise ManifestError(f"development modes must be exactly {MODES}")
    if len(raw_modes) != len(set(raw_modes)):
        raise ManifestError("duplicate evaluation mode")
    horizons = evaluation.get("gold_prefix_horizons")
    if horizons != list(GOLD_PREFIX_HORIZONS):
        raise ManifestError("gold-prefix horizons are frozen at [2, 4, 8]")

    attempts = evaluation.get("attempts_per_cell", 1)
    if not isinstance(attempts, int) or isinstance(attempts, bool) or not 8 <= attempts <= 128:
        raise ManifestError("true pass@8 requires attempts_per_cell in [8, 128]")
    if evaluation.get("sampling_seed_policy") != "paired_fixed_per_attempt_v1":
        raise ManifestError("sampling_seed_policy must be paired_fixed_per_attempt_v1")
    seeds = {}
    for key in ("order_seed", "shard_seed", "sampling_seed", "bootstrap_seed"):
        value = evaluation.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ManifestError(f"{key} must be a non-negative integer")
        seeds[key] = value
    resamples = evaluation.get("bootstrap_resamples", 10_000)
    if not isinstance(resamples, int) or not 100 <= resamples <= 1_000_000:
        raise ManifestError("bootstrap_resamples must be in [100, 1000000]")

    raw_tasks = task_manifest.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ManifestError("task manifest needs a non-empty tasks list")
    tasks = tuple(_parse_task(value) for value in raw_tasks)
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ManifestError("duplicate task IDs")
    task_seeds = [task.parameter_seed for task in tasks]
    if len(task_seeds) != len(set(task_seeds)):
        raise ManifestError("task parameter seeds must be unique")

    exclusions = _validate_exclusions(evaluation.get("exclusions", []), set(ids))
    return EvaluationManifest(
        suite=str(evaluation.get("suite", "")),
        split="development",
        task_suite=str(task_manifest.get("suite", "")),
        task_manifest_payload_sha256=actual_task_seal,
        evaluation_manifest_payload_sha256=(
            evaluation_manifest_payload_sha256
            or hashlib.sha256(canonical_json(evaluation)).hexdigest()
        ),
        expected_executor_ready_sha256=readiness_sha,
        expected_executor_ready_artifact_id=readiness_artifact_id,
        expected_executor_certification_schema=readiness_schema,
        expected_task_setup_validation_sha256=setup_sha,
        expected_task_setup_validation_artifact_id=setup_artifact_id,
        expected_task_setup_validation_schema=setup_schema,
        curriculum_commit=curriculum_commit,
        curriculum_runtime_binding_schema=binding_schema,
        evaluator_commit=evaluator_commit,
        order_seed=seeds["order_seed"],
        shard_seed=seeds["shard_seed"],
        sampling_seed=seeds["sampling_seed"],
        bootstrap_seed=seeds["bootstrap_seed"],
        bootstrap_resamples=resamples,
        attempts_per_cell=attempts,
        budget=budget,
        modes=tuple(raw_modes),
        arms=arms,
        runtime=runtime,
        tasks=tasks,
        excluded_task_ids=frozenset(exclusions),
    )


def _validate_budget(value: dict[str, Any]) -> dict[str, int | float]:
    required_keys = {
        "max_model_turns_per_trial",
        "max_model_turns_per_semantic_step",
        "max_logical_semantic_steps",
        "max_primitive_actions_per_trial",
        "max_emitted_primitive_events_per_trial",
        "max_output_tokens_per_turn",
        "max_total_output_tokens",
        "wall_time_seconds",
    }
    if set(value) != required_keys:
        raise ManifestError(f"budget keys must be exactly {sorted(required_keys)}")
    for required in required_keys - {"wall_time_seconds"}:
        item = value.get(required)
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise ManifestError(f"budget.{required} must be a positive integer")
    wall = value["wall_time_seconds"]
    if not isinstance(wall, (int, float)) or isinstance(wall, bool) or wall <= 0:
        raise ManifestError("budget.wall_time_seconds must be positive")
    if value["max_model_turns_per_trial"] < 8:
        raise ManifestError("budget must admit the frozen horizon 8")
    if value["max_model_turns_per_semantic_step"] < 8:
        raise ManifestError("semantic-step budget must admit bounded multi-action plans")
    if value["max_logical_semantic_steps"] < 4:
        raise ManifestError("budget must admit 2-4-step natural tasks")
    if value["max_total_output_tokens"] < (
        value["max_output_tokens_per_turn"] * value["max_model_turns_per_trial"]
    ):
        raise ManifestError("total output-token budget cannot cover all admitted turns")
    return dict(value)


def _parse_arm(value: Any) -> Arm:
    if not isinstance(value, dict):
        raise ManifestError("arm rows must be objects")
    name = value.get("name")
    if name not in ARMS:
        raise ManifestError(f"unknown arm: {name!r}")
    if value.get("action_interface") != ACTION_INTERFACES[name]:
        raise ManifestError(f"action interface drift for {name}")
    for key in ("checkpoint", "prompt_id"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ManifestError(f"{name}.{key} is required")
    for key in ("checkpoint_sha256", "prompt_sha256"):
        if not _is_sha256(value.get(key)):
            raise ManifestError(f"{name}.{key} must be a SHA-256")
    generation = value.get("generation", {})
    if not isinstance(generation, dict):
        raise ManifestError(f"{name}.generation must be an object")
    if generation.get("do_sample") is not True:
        raise ManifestError(f"{name}.generation.do_sample must be true for pass@k")
    temperature = generation.get("temperature")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or temperature <= 0:
        raise ManifestError(f"{name}.generation.temperature must be positive")
    return Arm(
        name=name,
        action_interface=ACTION_INTERFACES[name],
        checkpoint=value["checkpoint"],
        checkpoint_sha256=value["checkpoint_sha256"],
        prompt_id=value["prompt_id"],
        prompt_sha256=value["prompt_sha256"],
        generation=dict(generation),
    )


def _parse_runtime(value: Any) -> RuntimeBinding:
    if not isinstance(value, dict) or set(value) != {
        "runtime_id",
        "module",
        "factory",
        "source_sha256",
        "contract_schema",
    }:
        raise ManifestError("runtime binding field set drifted")
    for key in ("runtime_id", "module", "factory"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ManifestError(f"runtime.{key} is required")
    if value.get("contract_schema") != "proper_vm_paired_runtime_v1":
        raise ManifestError("runtime contract schema drift")
    if not _is_sha256(value.get("source_sha256")):
        raise ManifestError("runtime.source_sha256 is invalid")
    return RuntimeBinding(
        runtime_id=value["runtime_id"],
        module=value["module"],
        factory=value["factory"],
        source_sha256=value["source_sha256"],
        contract_schema=value["contract_schema"],
    )


def _parse_task(value: Any) -> Task:
    if not isinstance(value, dict):
        raise ManifestError("task rows must be objects")
    required_strings = ("task_id", "family_id", "app", "instruction")
    for key in required_strings:
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ManifestError(f"task.{key} is required")
    if value.get("split") != "development":
        raise ManifestError(f"{value['task_id']}: only development tasks are allowed")
    seed = value.get("parameter_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ManifestError(f"{value['task_id']}: invalid parameter_seed")
    raw_steps = value.get("semantic_steps")
    count = value.get("semantic_step_count")
    if not isinstance(raw_steps, list) or not 2 <= len(raw_steps) <= 4:
        raise ManifestError(f"{value['task_id']}: semantic_steps must contain 2-4 steps")
    if count != len(raw_steps):
        raise ManifestError(f"{value['task_id']}: semantic_step_count mismatch")
    steps: list[SemanticStep] = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            raise ManifestError(f"{value['task_id']}: semantic step must be an object")
        step_id = raw_step.get("step_id", str(index))
        intent = raw_step.get("intent")
        target = raw_step.get("target_ref")
        if not isinstance(step_id, (str, int)) or not str(step_id):
            raise ManifestError(f"{value['task_id']}: invalid step_id")
        if not isinstance(intent, str) or not intent:
            raise ManifestError(f"{value['task_id']}: step intent is required")
        if not isinstance(target, str) or not target:
            raise ManifestError(f"{value['task_id']}: target_ref is required")
        steps.append(SemanticStep(str(step_id), intent, target, dict(raw_step)))
    if len({step.step_id for step in steps}) != len(steps):
        raise ManifestError(f"{value['task_id']}: duplicate semantic step IDs")

    snapshot = value.get("snapshot")
    if not isinstance(snapshot, dict):
        raise ManifestError(f"{value['task_id']}: snapshot is required")
    for key in ("id", "reset_strategy"):
        if not isinstance(snapshot.get(key), str) or not snapshot[key]:
            raise ManifestError(f"{value['task_id']}: snapshot.{key} is required")
    verifier = value.get("verifier")
    if not isinstance(verifier, dict) or verifier.get("fresh_process") is not True:
        raise ManifestError(f"{value['task_id']}: verifier must run in a fresh process")
    for key in ("kind", "module"):
        if not isinstance(verifier.get(key), str) or not verifier[key]:
            raise ManifestError(f"{value['task_id']}: verifier.{key} is required")
    if (
        verifier.get("entrypoint") != "main"
        or verifier.get("result_schema") != "semantic_oracle_result_v2"
        or verifier.get("state_extractor_entrypoint") != "extract_state"
    ):
        raise ManifestError(f"{value['task_id']}: verifier API drift")
    initial = value.get("initial_cursor")
    if initial != {
        "source": "live_probe",
        "probe_version": "rung1_cursor_position_v1",
    }:
        raise ManifestError(f"{value['task_id']}: initial cursor contract drift")
    geometry = value.get("geometry_contract")
    if (
        not isinstance(geometry, dict)
        or geometry.get("source") != "live_probe"
        or not isinstance(geometry.get("probe_version"), str)
        or not isinstance(geometry.get("state_probe_version"), str)
        or not isinstance(geometry.get("required_targets"), list)
    ):
        raise ManifestError(f"{value['task_id']}: geometry contract drift")
    budget_contract = value.get("budget_contract")
    if not isinstance(budget_contract, dict) or set(budget_contract) != {
        "kind",
        "semantic_steps",
        "primitive_action_caps",
        "primitive_event_caps",
        "resolution",
        "resolved_budget_hash_required",
    }:
        raise ManifestError(f"{value['task_id']}: task budget-contract field set drift")
    if (
        budget_contract["kind"] != "conservative_caps"
        or budget_contract["resolution"] != "after_live_binding"
        or budget_contract["resolved_budget_hash_required"] is not True
    ):
        raise ManifestError(f"{value['task_id']}: task budget contract is not live-resolved")
    if budget_contract["semantic_steps"] != count:
        raise ManifestError(f"{value['task_id']}: semantic budget mismatch")
    for budget_name in ("primitive_action_caps", "primitive_event_caps"):
        arm_values = budget_contract[budget_name]
        if not isinstance(arm_values, dict) or set(arm_values) != set(ACTION_INTERFACES.values()):
            raise ManifestError(f"{value['task_id']}: {budget_name} interface drift")
        if any(not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in arm_values.values()):
            raise ManifestError(f"{value['task_id']}: invalid {budget_name}")
    history = value.get("gold_cursor_history")
    if not isinstance(history, list) or not all(isinstance(item, dict) for item in history):
        raise ManifestError(f"{value['task_id']}: gold_cursor_history must be object rows")

    fixture_sha = value.get("fixture_sha256")
    if not _is_sha256(fixture_sha):
        raise ManifestError(f"{value['task_id']}: fixture_sha256 is invalid")
    unsigned = dict(value)
    unsigned.pop("fixture_sha256", None)
    observed = hashlib.sha256(canonical_json(unsigned)).hexdigest()
    if observed != fixture_sha:
        raise ManifestError(
            f"{value['task_id']}: fixture hash mismatch: {observed} != {fixture_sha}"
        )
    task = Task(
        task_id=value["task_id"],
        family_id=value["family_id"],
        app=value["app"],
        parameter_seed=seed,
        instruction=value["instruction"],
        snapshot_id=snapshot["id"],
        reset_strategy=snapshot["reset_strategy"],
        verifier_kind=verifier["kind"],
        verifier_module=verifier["module"],
        budget_contract=dict(budget_contract),
        geometry_contract=dict(geometry),
        initial_cursor_contract=dict(initial),
        semantic_steps=tuple(steps),
        gold_cursor_history=tuple(dict(item) for item in history),
        fixture_sha256=fixture_sha,
        raw=dict(value),
    )
    for prefix in range(1, task.semantic_step_count + 1):
        task.cursor_ref_for_prefix(prefix)
    return task


def _validate_exclusions(value: Any, task_ids: set[str]) -> set[str]:
    if not isinstance(value, list):
        raise ManifestError("exclusions must be a list")
    excluded: set[str] = set()
    forbidden_keys = {"arm", "checkpoint", "prompt", "success", "outcome", "score"}
    for row in value:
        if not isinstance(row, dict):
            raise ManifestError("exclusion rows must be objects")
        if forbidden_keys & set(row):
            raise ManifestError("exclusions must be registered without arm or outcome fields")
        if set(row) != {"task_id", "reason", "evidence_sha256"}:
            raise ManifestError("exclusion requires task_id, reason, and evidence_sha256")
        task_id = row["task_id"]
        if task_id not in task_ids:
            raise ManifestError(f"exclusion references unknown task: {task_id}")
        if task_id in excluded:
            raise ManifestError(f"duplicate task exclusion: {task_id}")
        if not isinstance(row["reason"], str) or not row["reason"]:
            raise ManifestError("exclusion reason is required")
        if not _is_sha256(row["evidence_sha256"]):
            raise ManifestError("exclusion evidence must be hashed")
        excluded.add(task_id)
    return excluded


def _is_git_commit(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 40 or value.lower() != value:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
