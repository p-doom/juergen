"""Sanitized row-level trace contract for the coarse official pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contract import (
    ARMS,
    PAIRED_SEEDS,
    PILOT_TASK_COUNT,
    RESET_PROTOCOL,
    SCHEMA_VERSION,
    SOURCE_PROTOCOL,
)


class RecordError(ValueError):
    """A pilot row violates the frozen trace or pairing contract."""


ACTION_CLASSES = {
    "click",
    "drag",
    "scroll",
    "key_chord",
    "type",
    "terminate",
    "unparsed",
}
ERROR_CODES = {None, "parse_error", "dispatch_error", "ineffective_action"}
TERMINATIONS = {"model_terminate", "horizon", "parse_error", "executor_error"}

STEP_KEYS = {
    "step_index",
    "parse_success",
    "executor_success",
    "action_class",
    "ineffective",
    "error_code",
}
ROW_KEYS = {
    "schema_version",
    "pilot_id",
    "source_protocol",
    "cluster_index",
    "pair_seed",
    "pair_key",
    "arm",
    "arm_order",
    "reset_protocol",
    "reset_ordinal",
    "reset_success",
    "setup_success",
    "oracle_evaluated",
    "parse_success",
    "executor_success",
    "task_success",
    "termination",
    "steps",
}


@dataclass(frozen=True)
class StepTrace:
    step_index: int
    parse_success: bool
    executor_success: bool
    action_class: str
    ineffective: bool
    error_code: str | None


@dataclass(frozen=True)
class EpisodeRow:
    pilot_id: str
    cluster_index: int
    pair_seed: int
    pair_key: str
    arm: str
    arm_order: tuple[str, str]
    reset_ordinal: int
    reset_success: bool
    setup_success: bool
    oracle_evaluated: bool
    parse_success: bool
    executor_success: bool
    task_success: bool
    termination: str
    steps: tuple[StepTrace, ...]


def expected_arm_order(cluster_index: int, pair_seed: int) -> tuple[str, str]:
    try:
        seed_index = PAIRED_SEEDS.index(pair_seed)
    except ValueError as exc:
        raise RecordError(f"unregistered pair seed {pair_seed}") from exc
    if (cluster_index + seed_index) % 2:
        return ARMS[1], ARMS[0]
    return ARMS


def expected_pair_key(cluster_index: int, pair_seed: int) -> str:
    return f"cluster-{cluster_index:03d}-seed-{pair_seed}"


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise RecordError(f"{label} must be a boolean")
    return value


def _parse_step(payload: Any, expected_index: int) -> StepTrace:
    if not isinstance(payload, dict) or set(payload) != STEP_KEYS:
        raise RecordError("step trace keys do not match the sanitized schema")
    if payload["step_index"] != expected_index:
        raise RecordError("step indices must be contiguous and zero-based")
    parse_success = _require_bool(payload["parse_success"], "step parse_success")
    executor_success = _require_bool(payload["executor_success"], "step executor_success")
    ineffective = _require_bool(payload["ineffective"], "step ineffective")
    action_class = payload["action_class"]
    error_code = payload["error_code"]
    if action_class not in ACTION_CLASSES or error_code not in ERROR_CODES:
        raise RecordError("step action class or error code is not registered")
    if not parse_success:
        if executor_success or action_class != "unparsed" or error_code != "parse_error":
            raise RecordError("parse-failure trace is internally inconsistent")
    elif not executor_success:
        if error_code != "dispatch_error" or action_class == "unparsed":
            raise RecordError("executor-failure trace is internally inconsistent")
    elif ineffective:
        if error_code != "ineffective_action":
            raise RecordError("ineffective action must use its registered error code")
    elif error_code is not None:
        raise RecordError("successful effective action cannot carry an error code")
    if ineffective and not executor_success:
        raise RecordError("an undispatched action cannot be classified ineffective")
    return StepTrace(
        step_index=expected_index,
        parse_success=parse_success,
        executor_success=executor_success,
        action_class=action_class,
        ineffective=ineffective,
        error_code=error_code,
    )


def parse_episode_row(payload: Any, *, expected_pilot_id: str) -> EpisodeRow:
    """Parse a row while forbidding raw outputs, instructions, and task IDs."""

    if not isinstance(payload, dict) or set(payload) != ROW_KEYS:
        raise RecordError("episode row keys do not match the sanitized schema")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise RecordError("episode schema version mismatch")
    if payload["pilot_id"] != expected_pilot_id:
        raise RecordError("pilot_id mismatch")
    if payload["source_protocol"] != SOURCE_PROTOCOL:
        raise RecordError("source protocol mismatch")
    cluster_index = payload["cluster_index"]
    pair_seed = payload["pair_seed"]
    reset_ordinal = payload["reset_ordinal"]
    if (
        not isinstance(cluster_index, int)
        or isinstance(cluster_index, bool)
        or not 0 <= cluster_index < PILOT_TASK_COUNT
    ):
        raise RecordError("cluster_index is outside the frozen pilot")
    if (
        not isinstance(pair_seed, int)
        or isinstance(pair_seed, bool)
        or pair_seed not in PAIRED_SEEDS
    ):
        raise RecordError("pair_seed is not registered")
    arm = payload["arm"]
    if arm not in ARMS:
        raise RecordError("unknown arm")
    arm_order = expected_arm_order(cluster_index, pair_seed)
    if payload["arm_order"] != list(arm_order):
        raise RecordError("arm order violates deterministic counterbalancing")
    expected_ordinal = arm_order.index(arm) + 1
    if reset_ordinal != expected_ordinal:
        raise RecordError("reset ordinal does not match arm order")
    if payload["reset_protocol"] != RESET_PROTOCOL:
        raise RecordError("reset protocol mismatch")
    if payload["pair_key"] != expected_pair_key(cluster_index, pair_seed):
        raise RecordError("pair_key mismatch")
    steps_payload = payload["steps"]
    if not isinstance(steps_payload, list) or not steps_payload:
        raise RecordError("episode must contain at least one step trace")
    steps = tuple(_parse_step(step, index) for index, step in enumerate(steps_payload))
    parse_success = _require_bool(payload["parse_success"], "parse_success")
    executor_success = _require_bool(payload["executor_success"], "executor_success")
    if parse_success != all(step.parse_success for step in steps):
        raise RecordError("episode parse_success does not match step traces")
    if executor_success != all(step.executor_success for step in steps):
        raise RecordError("episode executor_success does not match step traces")
    termination = payload["termination"]
    if termination not in TERMINATIONS:
        raise RecordError("unregistered termination")
    if termination == "parse_error" and parse_success:
        raise RecordError("parse_error termination requires a parse failure")
    if termination == "executor_error" and executor_success:
        raise RecordError("executor_error termination requires an executor failure")
    return EpisodeRow(
        pilot_id=expected_pilot_id,
        cluster_index=cluster_index,
        pair_seed=pair_seed,
        pair_key=payload["pair_key"],
        arm=arm,
        arm_order=arm_order,
        reset_ordinal=reset_ordinal,
        reset_success=_require_bool(payload["reset_success"], "reset_success"),
        setup_success=_require_bool(payload["setup_success"], "setup_success"),
        oracle_evaluated=_require_bool(payload["oracle_evaluated"], "oracle_evaluated"),
        parse_success=parse_success,
        executor_success=executor_success,
        task_success=_require_bool(payload["task_success"], "task_success"),
        termination=termination,
        steps=steps,
    )
