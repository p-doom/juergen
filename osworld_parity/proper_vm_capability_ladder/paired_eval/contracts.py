from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol


ARMS = ("native_absolute_control", "compact_raw_phaseb")
ACTION_INTERFACES = {
    "native_absolute_control": "native_absolute_sequence_v1",
    "compact_raw_phaseb": "compact_raw_phaseb_v1",
}
RUNTIME_CONTRACT_SCHEMA = "proper_vm_paired_runtime_v1"
APPROVED_CURRICULUM_COMMIT = "c6039b7658e89ef6d1aae607bd0c19281b0354ef"
APPROVED_CURRICULUM_RUNTIME_BINDING_SCHEMA = (
    "proper_vm_sameapp_runtime_binding_receipt_v1"
)
GENERATION_SEED_SOURCE = "trial_generation_seeds_by_arm_v1"
SAMPLING_SEED_POLICY = "deterministic_unique_per_attempt_arm_v1"
GENERATION_SEED_DERIVATION = "sha256_sampling_seed_pair_id_arm_generation_63bit_v1"
MODES = (
    "gold_history_one_step",
    "gold_prefix_horizon",
    "natural_closed_loop",
)
GOLD_PREFIX_HORIZONS = (2, 4, 8)

# These are harness/infrastructure failures.  They may exclude an entire pair.
# Model parse and action dispatch errors are deliberately absent: those are
# scored complete-system outcomes.
INFRA_FAILURE_CLASSES = (
    "vm_reset",
    "vm_setup",
    "observation_capture",
    "model_service",
    "verifier",
    "harness_io",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def resolved_segment_budget_payload(
    *,
    task_id: str,
    fixture_sha256: str,
    action_schema: str,
    semantic_step_index: int,
    actions: tuple[Any, ...],
    resolved_primitive_actions: int,
    resolved_primitive_events: int,
    binding_revision: int,
    binding_sha256: str,
    expected_cursor_before: tuple[int, int],
    expected_cursor_after: tuple[int, int],
) -> dict[str, Any]:
    """Exact approved ``CompiledSegment`` resolved-budget payload."""

    return {
        "schema_version": 1,
        "task_id": task_id,
        "fixture_sha256": fixture_sha256,
        "action_schema": action_schema,
        "semantic_step_index": semantic_step_index,
        "resolved_primitive_actions": resolved_primitive_actions,
        "resolved_primitive_events": resolved_primitive_events,
        "binding_revision": binding_revision,
        "binding_sha256": binding_sha256,
        "expected_cursor_before": list(expected_cursor_before),
        "expected_cursor_after": list(expected_cursor_after),
        "actions": actions,
    }


def hash_observation(payload: bytes | str | dict[str, Any] | list[Any]) -> str:
    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = canonical_json(payload)
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class Observation:
    """A policy-visible observation.

    Only its hash is persisted by the evaluator.  A runtime may retain the
    payload transiently while asking the model for an action.
    """

    payload: bytes | str | dict[str, Any] | list[Any]
    media_type: str

    @property
    def sha256(self) -> str:
        return hash_observation(self.payload)


@dataclass(frozen=True)
class RequestedAction:
    value: Any
    model_call_id: str
    usage: dict[str, int] = field(default_factory=dict)
    generation_seed: int | None = None


@dataclass(frozen=True)
class ExecutionReceipt:
    """What the executor actually dispatched for one requested action."""

    executed_action: Any
    cursor_before: tuple[int, int]
    cursor_after: tuple[int, int]
    parse_status: str = "ok"
    dispatch_status: str = "ok"
    action_classes: tuple[str, ...] = ()
    semantic_operations: tuple[Any, ...] = ()
    lowered_operations: tuple[Any, ...] = ()
    operations: tuple[Any, ...] = ()
    backend_primitives: tuple[str, ...] = ()
    executor_evidence: dict[str, Any] = field(default_factory=dict)
    primitive_action_count: int = 1
    resolved_actions: tuple[Any, ...] = ()
    semantic_step_index: int = 0
    resolved_primitive_actions: int = 0
    resolved_primitive_events: int = 0
    resolved_budget_sha256: str = ""
    binding_sha256: str = ""
    binding_revision: int = 0
    binding_receipt: dict[str, Any] | None = None
    compiled_segment: dict[str, Any] | None = None
    dispatches: tuple[tuple[dict[str, Any], ...], ...] = ()
    executed_segment_receipt: dict[str, Any] | None = None


@dataclass(frozen=True)
class VerifierState:
    status: str
    task_solved: bool
    semantic_step_index: int
    semantic_state: dict[str, Any]
    matched_target_ref: str | None = None
    reason: str = ""
    oracle_pid: int = 0
    verifier_module: str = ""

    @property
    def semantic_state_sha256(self) -> str:
        return sha256_json(self.semantic_state)


@dataclass(frozen=True)
class StateProbe:
    """Read-only state extraction performed after an executor turn."""

    state: dict[str, Any]
    evidence: dict[str, Any]


@dataclass(frozen=True)
class SessionStart:
    task_id: str
    snapshot_id: str
    parameter_seed: int
    cursor_ref: str
    cursor: tuple[int, int]
    reset_signature: str
    cursor_source: str
    cursor_precentered: bool
    binding_receipt: dict[str, Any]
    prefix_replay: tuple[dict[str, Any], ...]


class ArmSession(Protocol):
    """Stateful real-VM session owned by exactly one evaluation arm."""

    start: SessionStart

    def observe(self) -> Observation: ...

    def request_action(
        self,
        *,
        observation: Observation,
        history: tuple[dict[str, Any], ...],
        generation_seed: int,
        budget: dict[str, Any],
    ) -> RequestedAction: ...

    def execute(self, requested: RequestedAction) -> ExecutionReceipt: ...

    def probe_state(self) -> StateProbe: ...

    def close(self) -> None: ...


class PairedRuntime(Protocol):
    """Bridge implemented by the real model/VM executor integration."""

    contract: dict[str, Any]

    def open_session(
        self,
        *,
        task: Any,
        arm: Any,
        mode: str,
        gold_prefix_length: int,
        horizon: int,
        generation_seed: int,
    ) -> ArmSession:
        """Reset the task and replay the action-neutral semantic gold prefix."""
        ...


class InfrastructureFailure(RuntimeError):
    """Known unscored failure emitted by a runtime integration."""

    def __init__(self, failure_class: str, message: str) -> None:
        if failure_class not in INFRA_FAILURE_CLASSES:
            raise ValueError(f"unknown infrastructure failure class: {failure_class}")
        super().__init__(message)
        self.failure_class = failure_class
