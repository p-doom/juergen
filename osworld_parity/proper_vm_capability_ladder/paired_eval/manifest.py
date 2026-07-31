from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ACTION_INTERFACES, ARMS, GOLD_PREFIX_HORIZONS, MODES, canonical_json


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
    initial_cursor: tuple[int, int]
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

    def cursor_for_prefix(self, prefix_length: int) -> tuple[int, int]:
        if prefix_length == 0:
            return self.initial_cursor
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
            for key in ("cursor", "cursor_after", "expected_cursor"):
                if key in entry:
                    return _cursor(entry[key], label=f"{self.task_id} gold cursor {prefix_length}")
        raise ManifestError(
            f"{self.task_id} has no unambiguous gold cursor for prefix {prefix_length}"
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
class EvaluationManifest:
    suite: str
    split: str
    task_suite: str
    task_manifest_payload_sha256: str
    evaluation_manifest_payload_sha256: str
    expected_executor_ready_sha256: str
    order_seed: int
    shard_seed: int
    bootstrap_seed: int
    bootstrap_resamples: int
    attempts_per_cell: int
    budget: dict[str, int | float]
    modes: tuple[str, ...]
    arms: tuple[Arm, ...]
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

    raw_modes = evaluation.get("modes")
    if not isinstance(raw_modes, list) or set(raw_modes) != set(MODES):
        raise ManifestError(f"development modes must be exactly {MODES}")
    if len(raw_modes) != len(set(raw_modes)):
        raise ManifestError("duplicate evaluation mode")
    horizons = evaluation.get("gold_prefix_horizons")
    if horizons != list(GOLD_PREFIX_HORIZONS):
        raise ManifestError("gold-prefix horizons are frozen at [2, 4, 8]")

    attempts = evaluation.get("attempts_per_cell", 1)
    if not isinstance(attempts, int) or isinstance(attempts, bool) or not 1 <= attempts <= 128:
        raise ManifestError("attempts_per_cell must be an integer in [1, 128]")
    seeds = {}
    for key in ("order_seed", "shard_seed", "bootstrap_seed"):
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
        order_seed=seeds["order_seed"],
        shard_seed=seeds["shard_seed"],
        bootstrap_seed=seeds["bootstrap_seed"],
        bootstrap_resamples=resamples,
        attempts_per_cell=attempts,
        budget=budget,
        modes=tuple(raw_modes),
        arms=arms,
        tasks=tasks,
        excluded_task_ids=frozenset(exclusions),
    )


def _validate_budget(value: dict[str, Any]) -> dict[str, int | float]:
    allowed = {"max_actions", "max_model_calls", "max_output_tokens", "wall_time_seconds"}
    if set(value) - allowed:
        raise ManifestError(f"unknown budget keys: {sorted(set(value) - allowed)}")
    for required in ("max_actions", "max_model_calls", "max_output_tokens"):
        item = value.get(required)
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise ManifestError(f"budget.{required} must be a positive integer")
    wall = value.get("wall_time_seconds")
    if wall is not None and (not isinstance(wall, (int, float)) or wall <= 0):
        raise ManifestError("budget.wall_time_seconds must be positive")
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
    return Arm(
        name=name,
        action_interface=ACTION_INTERFACES[name],
        checkpoint=value["checkpoint"],
        checkpoint_sha256=value["checkpoint_sha256"],
        prompt_id=value["prompt_id"],
        prompt_sha256=value["prompt_sha256"],
        generation=dict(generation),
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
    initial = _cursor(value.get("initial_cursor"), label=f"{value['task_id']} initial_cursor")
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
        initial_cursor=initial,
        semantic_steps=tuple(steps),
        gold_cursor_history=tuple(dict(item) for item in history),
        fixture_sha256=fixture_sha,
        raw=dict(value),
    )
    for prefix in range(1, task.semantic_step_count + 1):
        task.cursor_for_prefix(prefix)
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
