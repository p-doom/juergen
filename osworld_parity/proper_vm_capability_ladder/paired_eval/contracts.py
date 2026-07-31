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

# These are harness/infrastructure failure candidates.  They interrupt a pair
# and require operator review; the offline aggregator cannot authenticate a
# self-contained exclusion receipt and therefore emits no estimate.  Model
# parse and action dispatch errors are deliberately absent: those remain
# scored complete-system outcomes.
INFRA_FAILURE_CLASSES = (
    "vm_reset",
    "vm_setup",
    "observation_capture",
    "model_service",
    "verifier",
    "harness_io",
)

# Infrastructure failure candidates are well-formed only when the operation
# that failed is compatible with the runner-captured phase.  Aggregation
# derives class and phase from the structured event, checks this matrix, then
# still fails closed because row-local hashes are not external attestation.
INFRA_FAILURE_CLASS_PHASES = {
    "vm_reset": frozenset({"open_session"}),
    "vm_setup": frozenset({"open_session"}),
    "observation_capture": frozenset(
        {"initial_state_probe", "observe", "state_probe"}
    ),
    "model_service": frozenset({"request_action"}),
    "verifier": frozenset({"initial_verifier", "verifier"}),
    "harness_io": frozenset(
        {
            "open_session",
            "initial_state_probe",
            "observe",
            "request_action",
            "execute",
            "state_probe",
            "initial_verifier",
            "verifier",
            "close",
        }
    ),
}
INFRA_FAILURE_SOURCES = {
    "vm_reset": "vm_reset_provider",
    "vm_setup": "vm_setup_provider",
    "observation_capture": "observation_transport",
    "model_service": "model_service",
    "verifier": "fresh_process_verifier",
    "harness_io": "harness_io",
}
INFRA_FAILURE_OPERATIONS = {
    "open_session": "open_session",
    "initial_state_probe": "state_probe",
    "observe": "capture_observation",
    "request_action": "generate_action",
    "execute": "execute_action",
    "state_probe": "state_probe",
    "initial_verifier": "verify_state",
    "verifier": "verify_state",
    "close": "close_session",
}
INFRA_FAILURE_SOURCE_RECEIPT_TYPE = "infrastructure_failure_source_receipt_v1"
INFRA_FAILURE_EVENT_RECEIPT_TYPE = "paired_infrastructure_failure_event_v1"


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

    def __init__(
        self,
        failure_class: str,
        message: str,
        *,
        source_receipt: dict[str, Any],
    ) -> None:
        if failure_class not in INFRA_FAILURE_CLASSES:
            raise ValueError(f"unknown infrastructure failure class: {failure_class}")
        validate_infrastructure_failure_source_receipt(
            source_receipt,
            expected_class=failure_class,
        )
        super().__init__(message)
        self.failure_class = failure_class
        self.source_receipt = dict(source_receipt)


def infrastructure_failure_source_receipt(
    failure_class: str,
    *,
    operation: str,
    raw_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Seal the source-owned evidence captured when an operation fails."""

    if failure_class not in INFRA_FAILURE_CLASSES:
        raise ValueError(f"unknown infrastructure failure class: {failure_class}")
    if not isinstance(operation, str) or not operation:
        raise ValueError("infrastructure failure operation is required")
    if not isinstance(raw_evidence, dict) or not raw_evidence:
        raise ValueError("infrastructure failure raw evidence is required")
    receipt = {
        "schema_version": 1,
        "receipt_type": INFRA_FAILURE_SOURCE_RECEIPT_TYPE,
        "failure_class": failure_class,
        "source": INFRA_FAILURE_SOURCES[failure_class],
        "operation": operation,
        "status": "failed",
        "raw_evidence": dict(raw_evidence),
        "raw_evidence_sha256": sha256_json(raw_evidence),
    }
    receipt["source_receipt_sha256"] = sha256_json(receipt)
    return receipt


def validate_infrastructure_failure_source_receipt(
    receipt: Any,
    *,
    expected_class: str,
    expected_operation: str | None = None,
) -> dict[str, Any]:
    """Validate a source receipt without trusting an outer result-row seal."""

    expected_fields = {
        "schema_version",
        "receipt_type",
        "failure_class",
        "source",
        "operation",
        "status",
        "raw_evidence",
        "raw_evidence_sha256",
        "source_receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_fields:
        raise ValueError("infrastructure failure source receipt schema mismatch")
    if expected_class not in INFRA_FAILURE_CLASSES:
        raise ValueError(f"unknown infrastructure failure class: {expected_class}")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("receipt_type") != INFRA_FAILURE_SOURCE_RECEIPT_TYPE
        or receipt.get("failure_class") != expected_class
        or receipt.get("source") != INFRA_FAILURE_SOURCES[expected_class]
        or receipt.get("status") != "failed"
    ):
        raise ValueError("infrastructure failure source receipt identity mismatch")
    operation = receipt.get("operation")
    if not isinstance(operation, str) or not operation:
        raise ValueError("infrastructure failure source operation is missing")
    if expected_operation is not None and operation != expected_operation:
        raise ValueError("infrastructure failure source operation mismatch")
    raw_evidence = receipt.get("raw_evidence")
    if not isinstance(raw_evidence, dict) or not raw_evidence:
        raise ValueError("infrastructure failure raw evidence is missing")
    if receipt.get("raw_evidence_sha256") != sha256_json(raw_evidence):
        raise ValueError("infrastructure failure raw evidence hash mismatch")
    unsigned = dict(receipt)
    seal = unsigned.pop("source_receipt_sha256")
    if seal != sha256_json(unsigned):
        raise ValueError("infrastructure failure source receipt hash mismatch")
    return dict(receipt)
