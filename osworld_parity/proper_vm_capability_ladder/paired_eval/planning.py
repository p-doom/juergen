from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .contracts import ARMS, GOLD_PREFIX_HORIZONS
from .manifest import EvaluationManifest, Task


def _digest_int(*values: object) -> int:
    payload = "\x00".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class TrialSpec:
    pair_id: str
    cell_id: str
    task_id: str
    fixture_sha256: str
    snapshot_id: str
    parameter_seed: int
    initial_cursor_ref: str
    budget: dict[str, int | float]
    mode: str
    gold_prefix_length: int
    horizon: int
    attempt_id: int
    generation_seed: int
    arm_order: tuple[str, str]
    shard_index: int
    shard_count: int


def task_shard(
    manifest: EvaluationManifest,
    task: Task,
    shard_count: int,
) -> int:
    """Assign the whole task (all modes, attempts, and both arms) to one shard."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    return _digest_int(
        manifest.shard_seed,
        manifest.suite,
        task.task_id,
        task.parameter_seed,
    ) % shard_count


def _trial(
    manifest: EvaluationManifest,
    task: Task,
    *,
    mode: str,
    prefix: int,
    horizon: int,
    attempt: int,
    shard_index: int,
    shard_count: int,
) -> TrialSpec:
    cell_payload = (
        manifest.suite,
        task.task_id,
        task.parameter_seed,
        mode,
        prefix,
        horizon,
    )
    cell_id = hashlib.sha256(
        "\x00".join(str(item) for item in cell_payload).encode("utf-8")
    ).hexdigest()
    pair_id = hashlib.sha256(f"{cell_id}\x00{attempt}".encode("utf-8")).hexdigest()
    generation_seed = _digest_int(manifest.sampling_seed, pair_id, "generation") & 0x7FFFFFFF
    order_bit = _digest_int(manifest.order_seed, pair_id, "arm-order") & 1
    arm_order = ARMS if order_bit == 0 else tuple(reversed(ARMS))
    logical_steps = (
        1
        if mode == "gold_history_one_step"
        else task.semantic_step_count - prefix
    )
    if task.pair_primitive_action_cap > int(
        manifest.budget["max_primitive_actions_per_trial"]
    ):
        raise ValueError(f"{task.task_id}: primitive action budget exceeds evaluation ceiling")
    if task.pair_primitive_event_cap > int(
        manifest.budget["max_emitted_primitive_events_per_trial"]
    ):
        raise ValueError(f"{task.task_id}: primitive event budget exceeds evaluation ceiling")
    if mode == "gold_history_one_step" and horizon > int(
        manifest.budget["max_model_turns_per_semantic_step"]
    ):
        raise ValueError(f"{task.task_id}: semantic model-turn budget exceeds ceiling")
    if horizon > int(manifest.budget["max_model_turns_per_trial"]):
        raise ValueError(f"{task.task_id}: trial model-turn budget exceeds ceiling")
    budget = {
        "model_turns": horizon,
        "logical_semantic_steps": logical_steps,
        "primitive_actions": task.pair_primitive_action_cap,
        "emitted_primitive_events": task.pair_primitive_event_cap,
        "output_tokens_per_turn": int(manifest.budget["max_output_tokens_per_turn"]),
        "total_output_tokens": min(
            int(manifest.budget["max_total_output_tokens"]),
            horizon * int(manifest.budget["max_output_tokens_per_turn"]),
        ),
        "wall_time_seconds": manifest.budget["wall_time_seconds"],
    }
    return TrialSpec(
        pair_id=pair_id,
        cell_id=cell_id,
        task_id=task.task_id,
        fixture_sha256=task.fixture_sha256,
        snapshot_id=task.snapshot_id,
        parameter_seed=task.parameter_seed,
        initial_cursor_ref=task.cursor_ref_for_prefix(prefix),
        budget=budget,
        mode=mode,
        gold_prefix_length=prefix,
        horizon=horizon,
        attempt_id=attempt,
        generation_seed=generation_seed,
        arm_order=arm_order,  # type: ignore[arg-type]
        shard_index=shard_index,
        shard_count=shard_count,
    )


def build_plan(
    manifest: EvaluationManifest,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
) -> tuple[TrialSpec, ...]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    plan: list[TrialSpec] = []
    for task in sorted(manifest.tasks, key=lambda item: item.task_id):
        if task.task_id in manifest.excluded_task_ids:
            continue
        assigned = task_shard(manifest, task, shard_count)
        if assigned != shard_index:
            continue
        cells: list[tuple[str, int, int]] = []
        for prefix in range(task.semantic_step_count):
            cells.append(
                (
                    "gold_history_one_step",
                    prefix,
                    task.pair_primitive_action_cap,
                )
            )
            for horizon in GOLD_PREFIX_HORIZONS:
                cells.append(("gold_prefix_horizon", prefix, horizon))
        cells.append(
            (
                "natural_closed_loop",
                0,
                task.pair_primitive_action_cap,
            )
        )
        for mode, prefix, horizon in cells:
            if mode not in manifest.modes:
                continue
            for attempt in range(manifest.attempts_per_cell):
                plan.append(
                    _trial(
                        manifest,
                        task,
                        mode=mode,
                        prefix=prefix,
                        horizon=horizon,
                        attempt=attempt,
                        shard_index=assigned,
                        shard_count=shard_count,
                    )
                )
    pair_ids = [trial.pair_id for trial in plan]
    if len(pair_ids) != len(set(pair_ids)):
        raise AssertionError("deterministic plan produced duplicate pair IDs")
    generation_seeds: dict[str, set[int]] = {}
    for trial in plan:
        seen = generation_seeds.setdefault(trial.cell_id, set())
        if trial.generation_seed in seen:
            raise AssertionError("generation seed collision within a pass@k cell")
        seen.add(trial.generation_seed)
    return tuple(plan)
