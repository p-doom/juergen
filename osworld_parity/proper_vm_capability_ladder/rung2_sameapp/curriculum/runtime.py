"""Bind semantic tasks to live VM state, geometry, and cursor probes."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .oracle import initial_state
from .schema import SemanticTask
from ..fixtures import canonical_json


class RuntimeProbeError(RuntimeError):
    """A live VM probe does not satisfy the task's declared contract."""


@dataclass(frozen=True)
class ResetCycleEvidence:
    session_id: str
    reset_id: str
    generation_id: str
    sequence: int
    provider_reset_sequence: int
    provider_session_id: str
    prior_provider_generation_id: str
    provider_reset_receipt_sha256: str
    provider_state_before_sha256: str
    provider_state_after_sha256: str
    provider_path_sha256: str
    reset_started_monotonic_ns: int
    provider_reset_completed_monotonic_ns: int
    probe_completed_monotonic_ns: int
    captured_wall_time_ns: int
    vm_snapshot_id: str
    setup_commit: str
    reset_provider: str
    transport_endpoint_sha256: str
    task_id: str
    fixture_sha256: str
    probe_sha256: str
    issuer_mac: str
    evidence_sha256: str


@dataclass(frozen=True)
class StepRefreshEvidence:
    session_id: str
    refresh_id: str
    sequence: int
    task_id: str
    fixture_sha256: str
    reset_generation_id: str
    completed_step: int
    prior_binding_sha256: str
    executed_segment_sha256: str
    action_started_monotonic_ns: int
    action_completed_monotonic_ns: int
    probe_started_monotonic_ns: int
    probe_completed_monotonic_ns: int
    captured_wall_time_ns: int
    before_scroll_y: int
    after_scroll_y: int
    observed_scroll_delta: int
    required_minimum_delta: int
    expected_scroll_direction: str
    probe_sha256: str
    issuer_mac: str
    evidence_sha256: str


@dataclass(frozen=True)
class RuntimeProbe:
    state: dict[str, Any]
    geometry: dict[str, tuple[int, int]]
    initial_cursor: tuple[int, int]
    screen_size: tuple[int, int]
    geometry_probe_version: str
    state_probe_version: str
    cursor_probe_version: str = "rung1_cursor_position_v1"
    reset_cycle_evidence: ResetCycleEvidence | None = None
    refresh_evidence: StepRefreshEvidence | None = None
    observation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    observed_monotonic_ns: int = field(default_factory=time.monotonic_ns)


def _probe_payload(probe: RuntimeProbe) -> dict[str, Any]:
    return {
        "state": probe.state,
        "geometry": {name: list(point) for name, point in sorted(probe.geometry.items())},
        "initial_cursor": list(probe.initial_cursor),
        "screen_size": list(probe.screen_size),
        "geometry_probe_version": probe.geometry_probe_version,
        "state_probe_version": probe.state_probe_version,
        "cursor_probe_version": probe.cursor_probe_version,
        "observation_id": probe.observation_id,
        "observed_monotonic_ns": probe.observed_monotonic_ns,
    }


def _probe_sha256(probe: RuntimeProbe) -> str:
    return hashlib.sha256(canonical_json(_probe_payload(probe))).hexdigest()


class RuntimeEvidenceLedger:
    """Issues and consumes reset/refresh evidence for one live VM session."""

    def __init__(
        self,
        *,
        setup_commit: str,
        vm_snapshot_id: str = "osworld_ready",
        reset_provider: str,
        reset_attestor: Any,
        max_age_seconds: float = 120.0,
    ) -> None:
        if len(setup_commit) != 40 or any(c not in "0123456789abcdef" for c in setup_commit):
            raise RuntimeProbeError("setup commit must be lowercase 40-hex")
        if vm_snapshot_id != "osworld_ready":
            raise RuntimeProbeError("runtime evidence requires osworld_ready")
        self.setup_commit = setup_commit
        self.vm_snapshot_id = vm_snapshot_id
        self.reset_provider = reset_provider
        if not callable(getattr(reset_attestor, "consume_provider_reset_receipt", None)):
            raise RuntimeProbeError("external provider reset attestor is required")
        self.reset_attestor = reset_attestor
        self.max_age_ns = int(max_age_seconds * 1_000_000_000)
        self.session_id = uuid.uuid4().hex
        self._secret = secrets.token_bytes(32)
        self._reset_sequence = 0
        self._refresh_sequence = 0
        self._consumed_reset_hashes: set[str] = set()
        self._consumed_refresh_hashes: set[str] = set()
        self._issued_observation_ids: set[str] = set()
        self._issued_probe_objects: set[int] = set()
        self._recorded_executed_receipts: dict[str, tuple[Any, Any]] = {}

    def _mac(self, payload: dict[str, Any]) -> str:
        return hmac.new(self._secret, canonical_json(payload), hashlib.sha256).hexdigest()

    def issue_reset_probe(
        self,
        task: SemanticTask,
        probe: RuntimeProbe,
        *,
        provider_reset_receipt: Any,
        transport_endpoint: str,
    ) -> RuntimeProbe:
        from ...rung1.vm import ProviderResetReceipt

        if probe.reset_cycle_evidence is not None or probe.refresh_evidence is not None:
            raise RuntimeProbeError("cannot re-issue evidence for an attributed probe")
        if id(probe) in self._issued_probe_objects or (
            probe.observation_id in self._issued_observation_ids
        ):
            raise RuntimeProbeError("raw runtime observation re-sign detected")
        if not isinstance(provider_reset_receipt, ProviderResetReceipt):
            raise RuntimeProbeError("provider reset receipt type mismatch")
        try:
            self.reset_attestor.consume_provider_reset_receipt(provider_reset_receipt)
        except Exception as exc:
            raise RuntimeProbeError(f"provider reset attestation rejected: {exc}") from exc
        if provider_reset_receipt.snapshot_id != self.vm_snapshot_id:
            raise RuntimeProbeError("provider reset snapshot mismatch")
        if provider_reset_receipt.reset_completed_monotonic_ns >= probe.observed_monotonic_ns:
            raise RuntimeProbeError("runtime observation predates provider reset completion")
        now = time.monotonic_ns()
        if probe.observed_monotonic_ns > now or now - probe.observed_monotonic_ns > self.max_age_ns:
            raise RuntimeProbeError("reset probe is stale or future-dated")
        self._reset_sequence += 1
        payload = {
            "session_id": self.session_id,
            "reset_id": provider_reset_receipt.reset_id,
            "generation_id": provider_reset_receipt.new_generation_id,
            "sequence": self._reset_sequence,
            "provider_reset_sequence": provider_reset_receipt.reset_sequence,
            "provider_session_id": provider_reset_receipt.provider_session_id,
            "prior_provider_generation_id": provider_reset_receipt.prior_generation_id,
            "provider_reset_receipt_sha256": provider_reset_receipt.receipt_sha256,
            "provider_state_before_sha256": provider_reset_receipt.provider_state_before_sha256,
            "provider_state_after_sha256": provider_reset_receipt.provider_state_after_sha256,
            "provider_path_sha256": provider_reset_receipt.provider_path_sha256,
            "reset_started_monotonic_ns": provider_reset_receipt.reset_started_monotonic_ns,
            "provider_reset_completed_monotonic_ns": provider_reset_receipt.reset_completed_monotonic_ns,
            "probe_completed_monotonic_ns": probe.observed_monotonic_ns,
            "captured_wall_time_ns": time.time_ns(),
            "vm_snapshot_id": self.vm_snapshot_id,
            "setup_commit": self.setup_commit,
            "reset_provider": self.reset_provider,
            "transport_endpoint_sha256": hashlib.sha256(
                transport_endpoint.encode("utf-8")
            ).hexdigest(),
            "task_id": task.task_id,
            "fixture_sha256": task.fixture_sha256,
            "probe_sha256": _probe_sha256(probe),
        }
        issuer_mac = self._mac(payload)
        evidence_sha256 = hashlib.sha256(
            canonical_json({**payload, "issuer_mac": issuer_mac})
        ).hexdigest()
        self._issued_probe_objects.add(id(probe))
        self._issued_observation_ids.add(probe.observation_id)
        return replace(
            probe,
            reset_cycle_evidence=ResetCycleEvidence(
                **payload, issuer_mac=issuer_mac, evidence_sha256=evidence_sha256
            ),
        )

    def consume_reset_probes(
        self, task: SemanticTask, probes: tuple[RuntimeProbe, ...]
    ) -> None:
        if len({id(probe) for probe in probes}) != len(probes):
            raise RuntimeProbeError("duplicate reset probe object evidence")
        now = time.monotonic_ns()
        evidence_rows: list[ResetCycleEvidence] = []
        for probe in probes:
            evidence = probe.reset_cycle_evidence
            if evidence is None:
                raise RuntimeProbeError("reset probe has no cycle evidence")
            payload = asdict(evidence)
            evidence_sha256 = payload.pop("evidence_sha256")
            issuer_mac = payload.pop("issuer_mac")
            if issuer_mac != self._mac(payload):
                raise RuntimeProbeError("reset evidence issuer/provenance mismatch")
            observed_sha = hashlib.sha256(
                canonical_json({**payload, "issuer_mac": issuer_mac})
            ).hexdigest()
            if evidence_sha256 != observed_sha:
                raise RuntimeProbeError("reset evidence was mutated")
            if evidence.session_id != self.session_id:
                raise RuntimeProbeError("reset evidence belongs to another VM session")
            if (evidence.task_id, evidence.fixture_sha256) != (
                task.task_id,
                task.fixture_sha256,
            ):
                raise RuntimeProbeError("reset evidence task identity mismatch")
            if evidence.probe_sha256 != _probe_sha256(probe):
                raise RuntimeProbeError("reset probe content was mutated")
            if evidence.probe_completed_monotonic_ns > now or (
                now - evidence.probe_completed_monotonic_ns > self.max_age_ns
            ):
                raise RuntimeProbeError("reset evidence is stale or future-dated")
            if evidence_sha256 in self._consumed_reset_hashes:
                raise RuntimeProbeError("reset evidence replay detected")
            evidence_rows.append(evidence)
        if len({row.evidence_sha256 for row in evidence_rows}) != len(evidence_rows):
            raise RuntimeProbeError("duplicate reset evidence content")
        if len({row.reset_id for row in evidence_rows}) != len(evidence_rows) or len(
            {row.generation_id for row in evidence_rows}
        ) != len(evidence_rows):
            raise RuntimeProbeError("reset/generation IDs are not distinct")
        for first, second in zip(evidence_rows, evidence_rows[1:]):
            if second.sequence != first.sequence + 1:
                raise RuntimeProbeError("reset evidence sequence is not monotonic")
            if second.reset_started_monotonic_ns <= first.probe_completed_monotonic_ns:
                raise RuntimeProbeError("reset generations overlap or are out of order")
            if second.provider_reset_sequence != first.provider_reset_sequence + 1 or (
                second.prior_provider_generation_id != first.generation_id
            ) or second.provider_session_id != first.provider_session_id or (
                second.provider_path_sha256 != first.provider_path_sha256
            ):
                raise RuntimeProbeError("provider reset generation chain is discontinuous")
        self._consumed_reset_hashes.update(row.evidence_sha256 for row in evidence_rows)

    def record_executed_segment(
        self,
        task: SemanticTask,
        binding: "ValidatedRuntimeBinding",
        compiled_segment: Any,
        dispatches: Any,
        executed_receipt: Any,
        *,
        near_miss: bool,
    ) -> None:
        from .program import (
            CompiledSegment,
            ExecutedSegmentReceipt,
            compile_semantic_step,
            record_executed_segment,
            validate_executed_segment_receipt,
        )

        if not isinstance(compiled_segment, CompiledSegment) or not isinstance(
            executed_receipt, ExecutedSegmentReceipt
        ):
            raise RuntimeProbeError("executed segment object/schema mismatch")
        binding.validate_for_task(task)
        expected_segment = compile_semantic_step(
            task,
            compiled_segment.action_schema,
            binding=binding,
            semantic_step_index=compiled_segment.semantic_step_index,
            near_miss=near_miss,
        )
        if compiled_segment != expected_segment:
            raise RuntimeProbeError(
                "compiled segment does not match the declared semantic trajectory"
            )
        try:
            validate_executed_segment_receipt(executed_receipt)
            reconstructed = record_executed_segment(
                compiled_segment,
                dispatches,
                execution_started_monotonic_ns=(
                    executed_receipt.execution_started_monotonic_ns
                ),
                execution_completed_monotonic_ns=(
                    executed_receipt.execution_completed_monotonic_ns
                ),
            )
        except ValueError as exc:
            raise RuntimeProbeError(str(exc)) from exc
        if reconstructed != executed_receipt:
            raise RuntimeProbeError(
                "executed receipt does not match verified dispatch journal"
            )
        expected = (
            compiled_segment.task_id,
            compiled_segment.fixture_sha256,
            compiled_segment.action_schema,
            compiled_segment.semantic_step_index,
            compiled_segment.resolved_primitive_actions,
            compiled_segment.resolved_primitive_events,
            compiled_segment.resolved_budget_sha256,
            compiled_segment.binding_revision,
            compiled_segment.binding_sha256,
        )
        observed = (
            executed_receipt.task_id,
            executed_receipt.fixture_sha256,
            executed_receipt.action_schema,
            executed_receipt.semantic_step_index,
            executed_receipt.resolved_primitive_actions,
            executed_receipt.resolved_primitive_events,
            executed_receipt.resolved_budget_sha256,
            executed_receipt.binding_revision,
            executed_receipt.binding_sha256,
        )
        if observed != expected or (
            executed_receipt.binding_sha256 != binding.binding_sha256
            or executed_receipt.binding_revision != binding.binding_revision
        ):
            raise RuntimeProbeError("executed receipt does not match compiled segment/binding")
        if executed_receipt.executed_receipt_sha256 in self._recorded_executed_receipts:
            raise RuntimeProbeError("executed segment receipt replay detected")
        self._recorded_executed_receipts[executed_receipt.executed_receipt_sha256] = (
            compiled_segment,
            reconstructed,
        )

    def issue_refresh_probe(
        self,
        task: SemanticTask,
        binding: "ValidatedRuntimeBinding",
        probe: RuntimeProbe,
        *,
        completed_step: int,
        executed_segment: Any,
        action_started_monotonic_ns: int,
        action_completed_monotonic_ns: int,
        probe_started_monotonic_ns: int,
        probe_completed_monotonic_ns: int,
    ) -> RuntimeProbe:
        if probe.reset_cycle_evidence is not None or probe.refresh_evidence is not None:
            raise RuntimeProbeError("cannot refresh with attributed/stale probe evidence")
        from .program import ExecutedSegmentReceipt, validate_executed_segment_receipt

        if not isinstance(executed_segment, ExecutedSegmentReceipt):
            raise RuntimeProbeError("Chrome refresh requires ExecutedSegmentReceipt")
        try:
            validate_executed_segment_receipt(executed_segment)
        except ValueError as exc:
            raise RuntimeProbeError(str(exc)) from exc
        recorded = self._recorded_executed_receipts.get(
            executed_segment.executed_receipt_sha256
        )
        if recorded is None or recorded[1] != executed_segment:
            raise RuntimeProbeError("Chrome refresh receipt was not ledger-recorded")
        if (
            completed_step != 2
            or executed_segment.semantic_step_index != 2
            or executed_segment.task_id != task.task_id
            or executed_segment.fixture_sha256 != task.fixture_sha256
            or executed_segment.binding_sha256 != binding.binding_sha256
            or executed_segment.binding_revision != binding.binding_revision
        ):
            raise RuntimeProbeError("Chrome refresh receipt step/task/binding mismatch")
        active = binding.initial_probe.reset_cycle_evidence
        if active is None:
            raise RuntimeProbeError("binding has no active reset generation")
        if not (
            active.probe_completed_monotonic_ns
            < action_started_monotonic_ns
            <= action_completed_monotonic_ns
            < probe_started_monotonic_ns
            <= probe_completed_monotonic_ns
            <= time.monotonic_ns()
        ):
            raise RuntimeProbeError("refresh is not causally after the executed action")
        before = int(binding.initial_probe.state.get("scroll_y", 0))
        after = int(probe.state.get("scroll_y", 0))
        delta = after - before
        minimum = int(task.params["minimum_scroll_delta"])
        direction = str(task.params["scroll_direction"])
        if abs(delta) < minimum or (direction == "down" and delta <= 0) or (
            direction == "up" and delta >= 0
        ):
            raise RuntimeProbeError("refresh does not prove the required signed scroll delta")
        self._refresh_sequence += 1
        payload = {
            "session_id": self.session_id,
            "refresh_id": uuid.uuid4().hex,
            "sequence": self._refresh_sequence,
            "task_id": task.task_id,
            "fixture_sha256": task.fixture_sha256,
            "reset_generation_id": active.generation_id,
            "completed_step": completed_step,
            "prior_binding_sha256": binding.binding_sha256,
            "executed_segment_sha256": executed_segment.executed_receipt_sha256,
            "action_started_monotonic_ns": action_started_monotonic_ns,
            "action_completed_monotonic_ns": action_completed_monotonic_ns,
            "probe_started_monotonic_ns": probe_started_monotonic_ns,
            "probe_completed_monotonic_ns": probe_completed_monotonic_ns,
            "captured_wall_time_ns": time.time_ns(),
            "before_scroll_y": before,
            "after_scroll_y": after,
            "observed_scroll_delta": delta,
            "required_minimum_delta": minimum,
            "expected_scroll_direction": direction,
            "probe_sha256": _probe_sha256(probe),
        }
        issuer_mac = self._mac(payload)
        evidence_sha256 = hashlib.sha256(
            canonical_json({**payload, "issuer_mac": issuer_mac})
        ).hexdigest()
        return replace(
            probe,
            refresh_evidence=StepRefreshEvidence(
                **payload, issuer_mac=issuer_mac, evidence_sha256=evidence_sha256
            ),
        )

    def consume_refresh_probe(
        self,
        task: SemanticTask,
        binding: "ValidatedRuntimeBinding",
        probe: RuntimeProbe,
        *,
        completed_step: int,
        executed_segment: Any,
    ) -> None:
        from .program import ExecutedSegmentReceipt, validate_executed_segment_receipt

        if not isinstance(executed_segment, ExecutedSegmentReceipt):
            raise RuntimeProbeError("Chrome refresh requires ExecutedSegmentReceipt")
        try:
            validate_executed_segment_receipt(executed_segment)
        except ValueError as exc:
            raise RuntimeProbeError(str(exc)) from exc
        if self._recorded_executed_receipts.get(
            executed_segment.executed_receipt_sha256
        ) is None:
            raise RuntimeProbeError("Chrome refresh receipt was not ledger-recorded")
        evidence = probe.refresh_evidence
        if evidence is None:
            raise RuntimeProbeError("post-step probe has no refresh evidence")
        payload = asdict(evidence)
        evidence_sha256 = payload.pop("evidence_sha256")
        issuer_mac = payload.pop("issuer_mac")
        if issuer_mac != self._mac(payload) or evidence_sha256 != hashlib.sha256(
            canonical_json({**payload, "issuer_mac": issuer_mac})
        ).hexdigest():
            raise RuntimeProbeError("refresh evidence was forged or mutated")
        active = binding.initial_probe.reset_cycle_evidence
        if active is None or evidence.reset_generation_id != active.generation_id:
            raise RuntimeProbeError("refresh evidence uses a stale reset generation")
        if evidence.prior_binding_sha256 != binding.binding_sha256:
            raise RuntimeProbeError("refresh evidence uses a stale binding")
        if evidence.completed_step != completed_step or completed_step != 2:
            raise RuntimeProbeError("Chrome refresh must follow semantic step 2")
        if evidence.executed_segment_sha256 != executed_segment.executed_receipt_sha256:
            raise RuntimeProbeError("refresh is not tied to the executed scroll receipt")
        if evidence.probe_sha256 != _probe_sha256(probe):
            raise RuntimeProbeError("refreshed probe content was mutated")
        now = time.monotonic_ns()
        if evidence.probe_completed_monotonic_ns > now or (
            now - evidence.probe_completed_monotonic_ns > self.max_age_ns
        ):
            raise RuntimeProbeError("refresh evidence is stale or future-dated")
        if evidence_sha256 in self._consumed_refresh_hashes:
            raise RuntimeProbeError("refresh evidence replay detected")
        self._consumed_refresh_hashes.add(evidence_sha256)


@dataclass(frozen=True)
class ValidatedRuntimeBinding:
    task_id: str
    fixture_sha256: str
    reset_probe_count: int
    reset_probes: tuple[RuntimeProbe, ...]
    initial_probe: RuntimeProbe
    initial_cursor_ref: str
    resolved_initial_cursor: tuple[int, int]
    refreshed_after_steps: dict[int, RuntimeProbe]
    binding_revision: int
    parent_binding_sha256: str | None
    refresh_evidence_sha256: str | None
    evidence_fresh_until_monotonic_ns: int
    binding_sha256: str

    def receipt(self) -> dict[str, Any]:
        geometry = {
            name: list(point) for name, point in sorted(self.initial_probe.geometry.items())
        }
        refresh_transitions = []
        for _, probe in sorted(self.refreshed_after_steps.items()):
            if probe.refresh_evidence is None:
                continue
            transition = {
                "pre_binding_revision": self.binding_revision - 1,
                "post_binding_revision": self.binding_revision,
                "pre_binding_sha256": self.parent_binding_sha256,
                "post_binding_sha256": self.binding_sha256,
                "refresh_evidence": asdict(probe.refresh_evidence),
            }
            transition["transition_receipt_sha256"] = hashlib.sha256(
                canonical_json(transition)
            ).hexdigest()
            refresh_transitions.append(transition)
        payload = {
            "schema_version": 1,
            "task_id": self.task_id,
            "fixture_sha256": self.fixture_sha256,
            "binding_revision": self.binding_revision,
            "binding_sha256": self.binding_sha256,
            "parent_binding_sha256": self.parent_binding_sha256,
            "refresh_evidence_sha256": self.refresh_evidence_sha256,
            "evidence_fresh_until_monotonic_ns": self.evidence_fresh_until_monotonic_ns,
            "reset_cycles": [
                {
                    "session_id": probe.reset_cycle_evidence.session_id,
                    "reset_id": probe.reset_cycle_evidence.reset_id,
                    "generation_id": probe.reset_cycle_evidence.generation_id,
                    "sequence": probe.reset_cycle_evidence.sequence,
                    "provider_reset_sequence": probe.reset_cycle_evidence.provider_reset_sequence,
                    "provider_session_id": probe.reset_cycle_evidence.provider_session_id,
                    "prior_provider_generation_id": probe.reset_cycle_evidence.prior_provider_generation_id,
                    "provider_reset_receipt_sha256": probe.reset_cycle_evidence.provider_reset_receipt_sha256,
                    "provider_state_before_sha256": probe.reset_cycle_evidence.provider_state_before_sha256,
                    "provider_state_after_sha256": probe.reset_cycle_evidence.provider_state_after_sha256,
                    "provider_path_sha256": probe.reset_cycle_evidence.provider_path_sha256,
                    "reset_started_monotonic_ns": probe.reset_cycle_evidence.reset_started_monotonic_ns,
                    "provider_reset_completed_monotonic_ns": probe.reset_cycle_evidence.provider_reset_completed_monotonic_ns,
                    "probe_completed_monotonic_ns": probe.reset_cycle_evidence.probe_completed_monotonic_ns,
                    "captured_wall_time_ns": probe.reset_cycle_evidence.captured_wall_time_ns,
                    "vm_snapshot_id": probe.reset_cycle_evidence.vm_snapshot_id,
                    "setup_commit": probe.reset_cycle_evidence.setup_commit,
                    "reset_provider": probe.reset_cycle_evidence.reset_provider,
                    "transport_endpoint_sha256": probe.reset_cycle_evidence.transport_endpoint_sha256,
                    "probe_sha256": probe.reset_cycle_evidence.probe_sha256,
                    "evidence_sha256": probe.reset_cycle_evidence.evidence_sha256,
                }
                for probe in self.reset_probes
                if probe.reset_cycle_evidence is not None
            ],
            "resolved_initial_cursor": list(self.resolved_initial_cursor),
            "initial_geometry": geometry,
            "initial_geometry_sha256": hashlib.sha256(
                canonical_json(geometry)
            ).hexdigest(),
            "refresh_transitions": refresh_transitions,
        }
        payload["binding_receipt_sha256"] = hashlib.sha256(
            canonical_json(payload)
        ).hexdigest()
        return payload

    def validate_for_task(self, task: SemanticTask) -> None:
        if self.task_id != task.task_id or self.fixture_sha256 != task.fixture_sha256:
            raise RuntimeProbeError(f"{task.task_id}: runtime binding identity mismatch")
        if time.monotonic_ns() > self.evidence_fresh_until_monotonic_ns:
            raise RuntimeProbeError(f"{task.task_id}: runtime binding evidence is stale")
        if self.reset_probe_count < 2 or self.reset_probe_count != len(self.reset_probes):
            raise RuntimeProbeError(f"{task.task_id}: fewer than two reset probes")
        if self.initial_probe != self.reset_probes[-1]:
            raise RuntimeProbeError(f"{task.task_id}: initial/reset probe mismatch")
        reset_rows: list[ResetCycleEvidence] = []
        for probe in self.reset_probes:
            evidence = probe.reset_cycle_evidence
            if evidence is None:
                raise RuntimeProbeError(f"{task.task_id}: reset evidence is missing")
            payload = asdict(evidence)
            evidence_sha = payload.pop("evidence_sha256")
            if evidence_sha != hashlib.sha256(canonical_json(payload)).hexdigest():
                raise RuntimeProbeError(f"{task.task_id}: reset evidence was mutated")
            if evidence.probe_sha256 != _probe_sha256(probe):
                raise RuntimeProbeError(f"{task.task_id}: reset probe content was mutated")
            if (evidence.task_id, evidence.fixture_sha256) != (
                task.task_id,
                task.fixture_sha256,
            ):
                raise RuntimeProbeError(f"{task.task_id}: reset evidence identity mismatch")
            reset_rows.append(evidence)
        if len({row.reset_id for row in reset_rows}) != len(reset_rows) or len(
            {row.generation_id for row in reset_rows}
        ) != len(reset_rows):
            raise RuntimeProbeError(f"{task.task_id}: duplicate reset generation evidence")
        for first, second in zip(reset_rows, reset_rows[1:]):
            if second.sequence != first.sequence + 1 or (
                second.reset_started_monotonic_ns <= first.probe_completed_monotonic_ns
            ) or second.provider_reset_sequence != first.provider_reset_sequence + 1 or (
                second.prior_provider_generation_id != first.generation_id
            ) or second.provider_session_id != first.provider_session_id or (
                second.provider_path_sha256 != first.provider_path_sha256
            ):
                raise RuntimeProbeError(f"{task.task_id}: reset evidence ordering drift")
        for following in self.reset_probes[:-1]:
            validate_repeated_runtime_probes(task, self.initial_probe, following)
        if self.initial_cursor_ref != "runtime.initial_cursor" or (
            self.resolved_initial_cursor != self.initial_probe.initial_cursor
        ):
            raise RuntimeProbeError(f"{task.task_id}: resolved initial cursor mismatch")
        if not self.refreshed_after_steps and (
            self.binding_revision != 1
            or self.parent_binding_sha256 is not None
            or self.refresh_evidence_sha256 is not None
        ):
            raise RuntimeProbeError(f"{task.task_id}: initial binding revision drift")
        if self.refreshed_after_steps and (
            self.binding_revision != 2
            or self.parent_binding_sha256 is None
            or self.refresh_evidence_sha256 is None
        ):
            raise RuntimeProbeError(f"{task.task_id}: refreshed binding revision drift")
        for step, probe in self.refreshed_after_steps.items():
            _validate_probe_envelope(task, probe, expect_initial_state=False)
            evidence = probe.refresh_evidence
            if evidence is None:
                raise RuntimeProbeError(f"{task.task_id}: refresh evidence is missing")
            payload = asdict(evidence)
            evidence_sha = payload.pop("evidence_sha256")
            if evidence_sha != hashlib.sha256(canonical_json(payload)).hexdigest():
                raise RuntimeProbeError(f"{task.task_id}: refresh evidence was mutated")
            if evidence.probe_sha256 != _probe_sha256(probe):
                raise RuntimeProbeError(f"{task.task_id}: refreshed probe content was mutated")
            if evidence.completed_step != step or evidence.completed_step != 2:
                raise RuntimeProbeError(f"{task.task_id}: refresh step identity drift")
            if evidence.prior_binding_sha256 != self.parent_binding_sha256:
                raise RuntimeProbeError(f"{task.task_id}: refresh parent binding drift")
            if evidence.evidence_sha256 != self.refresh_evidence_sha256:
                raise RuntimeProbeError(f"{task.task_id}: refresh evidence seal drift")
            if evidence.reset_generation_id != reset_rows[-1].generation_id:
                raise RuntimeProbeError(f"{task.task_id}: stale refresh generation")
            before = int(self.initial_probe.state.get("scroll_y", 0))
            after = int(probe.state.get("scroll_y", 0))
            delta = after - before
            if (
                evidence.before_scroll_y != before
                or evidence.after_scroll_y != after
                or evidence.observed_scroll_delta != delta
                or evidence.required_minimum_delta
                != int(task.params["minimum_scroll_delta"])
                or evidence.expected_scroll_direction
                != str(task.params["scroll_direction"])
                or abs(delta) < evidence.required_minimum_delta
                or (evidence.expected_scroll_direction == "down" and delta <= 0)
                or (evidence.expected_scroll_direction == "up" and delta >= 0)
            ):
                raise RuntimeProbeError(f"{task.task_id}: refresh scroll proof drift")
            if not (
                reset_rows[-1].probe_completed_monotonic_ns
                < evidence.action_started_monotonic_ns
                <= evidence.action_completed_monotonic_ns
                < evidence.probe_started_monotonic_ns
                <= evidence.probe_completed_monotonic_ns
                <= self.evidence_fresh_until_monotonic_ns
            ):
                raise RuntimeProbeError(f"{task.task_id}: refresh causality drift")
        observed = hashlib.sha256(
            canonical_json(
                _binding_payload(
                    task,
                    self.reset_probes,
                    self.refreshed_after_steps,
                    binding_revision=self.binding_revision,
                    parent_binding_sha256=self.parent_binding_sha256,
                    refresh_evidence_sha256=self.refresh_evidence_sha256,
                    evidence_fresh_until_monotonic_ns=self.evidence_fresh_until_monotonic_ns,
                )
            )
        ).hexdigest()
        if observed != self.binding_sha256:
            raise RuntimeProbeError(f"{task.task_id}: runtime binding seal mismatch")

    def probe_for_step(self, task: SemanticTask, step_index: int) -> RuntimeProbe:
        self.validate_for_task(task)
        if task.app == "chrome" and step_index > 2:
            try:
                return self.refreshed_after_steps[2]
            except KeyError as exc:
                raise RuntimeProbeError(
                    f"{task.task_id}: Chrome step {step_index} requires post-step-2 refresh"
                ) from exc
        return self.initial_probe


def probe_runtime(
    transport: Any, task: SemanticTask, *, expect_initial_state: bool = True
) -> RuntimeProbe:
    """Run the declared existing VM probes; never source task coordinates."""

    if task.app == "vscode":
        from ...rung1b_realapps.vm import probe_fixture, probe_geometry
        from .oracle import as_vscode_fixture

        fixture = as_vscode_fixture(task)
        state = probe_fixture(transport, fixture)
        live = probe_geometry(transport, fixture)
        geometry = {"editor": tuple(live.editor)}
    else:
        from ..vm import probe_geometry, probe_state
        from .oracle import as_sameapp_fixture

        fixture = as_sameapp_fixture(task)
        state = probe_state(transport, fixture)
        geometry = probe_geometry(transport, fixture, state)
    probe = RuntimeProbe(
        state=state,
        geometry=geometry,
        initial_cursor=tuple(transport.cursor_position()),
        screen_size=tuple(transport.screen_size()),
        geometry_probe_version=task.geometry_contract["probe_version"],
        state_probe_version=task.geometry_contract["state_probe_version"],
    )
    _validate_probe_envelope(task, probe, expect_initial_state=expect_initial_state)
    return probe


def _validate_probe_envelope(
    task: SemanticTask, probe: RuntimeProbe, *, expect_initial_state: bool
) -> None:
    contract = task.geometry_contract
    if probe.geometry_probe_version != contract["probe_version"]:
        raise RuntimeProbeError(f"{task.task_id}: geometry probe version mismatch")
    if probe.state_probe_version != contract["state_probe_version"]:
        raise RuntimeProbeError(f"{task.task_id}: state probe version mismatch")
    if probe.cursor_probe_version != task.initial_cursor["probe_version"]:
        raise RuntimeProbeError(f"{task.task_id}: cursor probe version mismatch")
    if expect_initial_state:
        expected_state = initial_state(task)
        expected_state.pop("held_inputs")
        if probe.state != expected_state:
            raise RuntimeProbeError(
                f"{task.task_id}: setup-state mismatch: {probe.state!r} != {expected_state!r}"
            )
    required = set(contract["required_targets"])
    if set(probe.geometry) != required:
        raise RuntimeProbeError(
            f"{task.task_id}: geometry targets drifted: "
            f"{sorted(probe.geometry)} != {sorted(required)}"
        )
    width, height = probe.screen_size
    if width < 1 or height < 1:
        raise RuntimeProbeError(f"{task.task_id}: invalid live viewport")
    for name, point in probe.geometry.items():
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or not all(isinstance(value, int) for value in point)
            or not 0 <= point[0] < width
            or not 0 <= point[1] < height
        ):
            raise RuntimeProbeError(f"{task.task_id}: invalid live target {name}={point!r}")
    x, y = probe.initial_cursor
    if not 0 <= x < width or not 0 <= y < height:
        raise RuntimeProbeError(f"{task.task_id}: initial cursor is outside viewport")


def validate_runtime_probe(task: SemanticTask, probe: RuntimeProbe) -> None:
    _validate_probe_envelope(task, probe, expect_initial_state=True)


def validate_repeated_runtime_probes(
    task: SemanticTask, first: RuntimeProbe, second: RuntimeProbe
) -> None:
    validate_runtime_probe(task, first)
    validate_runtime_probe(task, second)
    if first.geometry != second.geometry:
        raise RuntimeProbeError(f"{task.task_id}: geometry drift across exact resets")
    if first.initial_cursor != second.initial_cursor:
        raise RuntimeProbeError(f"{task.task_id}: cursor drift across exact resets")
    if first.screen_size != second.screen_size:
        raise RuntimeProbeError(f"{task.task_id}: viewport drift across exact resets")


def _binding_payload(
    task: SemanticTask,
    probes: tuple[RuntimeProbe, ...],
    refreshed_after_steps: dict[int, RuntimeProbe],
    *,
    binding_revision: int = 1,
    parent_binding_sha256: str | None = None,
    refresh_evidence_sha256: str | None = None,
    evidence_fresh_until_monotonic_ns: int = 0,
) -> dict[str, Any]:
    def row(probe: RuntimeProbe) -> dict[str, Any]:
        return {
            "state_sha256": hashlib.sha256(canonical_json(probe.state)).hexdigest(),
            "geometry": {name: list(point) for name, point in sorted(probe.geometry.items())},
            "initial_cursor": list(probe.initial_cursor),
            "screen_size": list(probe.screen_size),
            "geometry_probe_version": probe.geometry_probe_version,
            "state_probe_version": probe.state_probe_version,
            "cursor_probe_version": probe.cursor_probe_version,
            "reset_evidence_sha256": (
                probe.reset_cycle_evidence.evidence_sha256
                if probe.reset_cycle_evidence is not None
                else None
            ),
            "refresh_evidence_sha256": (
                probe.refresh_evidence.evidence_sha256
                if probe.refresh_evidence is not None
                else None
            ),
        }

    return {
        "schema_version": 1,
        "task_id": task.task_id,
        "fixture_sha256": task.fixture_sha256,
        "binding_revision": binding_revision,
        "parent_binding_sha256": parent_binding_sha256,
        "refresh_evidence_sha256": refresh_evidence_sha256,
        "evidence_fresh_until_monotonic_ns": evidence_fresh_until_monotonic_ns,
        "reset_probes": [row(probe) for probe in probes],
        "refreshed_after_steps": {
            str(step): row(probe) for step, probe in sorted(refreshed_after_steps.items())
        },
    }


def bind_repeated_runtime_probes(
    task: SemanticTask,
    probes: tuple[RuntimeProbe, ...] | list[RuntimeProbe],
    *,
    ledger: RuntimeEvidenceLedger,
) -> ValidatedRuntimeBinding:
    values = tuple(probes)
    if len(values) < 2:
        raise RuntimeProbeError(f"{task.task_id}: at least two reset probes are required")
    ledger.consume_reset_probes(task, values)
    first = values[0]
    for following in values[1:]:
        validate_repeated_runtime_probes(task, first, following)
    fresh_until = min(
        probe.reset_cycle_evidence.probe_completed_monotonic_ns + ledger.max_age_ns
        for probe in values
        if probe.reset_cycle_evidence is not None
    )
    payload = _binding_payload(
        task,
        values,
        {},
        evidence_fresh_until_monotonic_ns=fresh_until,
    )
    return ValidatedRuntimeBinding(
        task_id=task.task_id,
        fixture_sha256=task.fixture_sha256,
        reset_probe_count=len(values),
        reset_probes=values,
        initial_probe=values[-1],
        initial_cursor_ref="runtime.initial_cursor",
        resolved_initial_cursor=values[-1].initial_cursor,
        refreshed_after_steps={},
        binding_revision=1,
        parent_binding_sha256=None,
        refresh_evidence_sha256=None,
        evidence_fresh_until_monotonic_ns=fresh_until,
        binding_sha256=hashlib.sha256(canonical_json(payload)).hexdigest(),
    )


def refresh_binding_after_step(
    task: SemanticTask,
    binding: ValidatedRuntimeBinding,
    *,
    completed_step: int,
    probe: RuntimeProbe,
    executed_segment: Any,
    ledger: RuntimeEvidenceLedger,
) -> ValidatedRuntimeBinding:
    binding.probe_for_step(task, min(completed_step, 2))
    _validate_probe_envelope(task, probe, expect_initial_state=False)
    ledger.consume_refresh_probe(
        task,
        binding,
        probe,
        completed_step=completed_step,
        executed_segment=executed_segment,
    )
    refreshed = {**binding.refreshed_after_steps, completed_step: probe}
    transition_sha = probe.refresh_evidence.evidence_sha256
    payload = _binding_payload(
        task,
        binding.reset_probes,
        refreshed,
        binding_revision=binding.binding_revision + 1,
        parent_binding_sha256=binding.binding_sha256,
        refresh_evidence_sha256=transition_sha,
        evidence_fresh_until_monotonic_ns=binding.evidence_fresh_until_monotonic_ns,
    )
    return replace(
        binding,
        refreshed_after_steps=refreshed,
        binding_revision=binding.binding_revision + 1,
        parent_binding_sha256=binding.binding_sha256,
        refresh_evidence_sha256=transition_sha,
        binding_sha256=hashlib.sha256(canonical_json(payload)).hexdigest(),
    )


def resolved_cursor_history(
    task: SemanticTask, probe: RuntimeProbe
) -> tuple[dict[str, Any], ...]:
    """Resolve semantic cursor refs only after the live probe passes."""

    validate_runtime_probe(task, probe)
    values: dict[str, tuple[int, int]] = {
        "runtime.initial_cursor": probe.initial_cursor,
        **{f"geometry.{name}": point for name, point in probe.geometry.items()},
    }
    rows: list[dict[str, Any]] = []
    for milestone in task.gold_cursor_history:
        try:
            before = values[milestone.cursor_before_ref]
            after = values[milestone.cursor_after_ref]
        except KeyError as exc:
            raise RuntimeProbeError(
                f"{task.task_id}: unresolved cursor reference {exc.args[0]!r}"
            ) from exc
        rows.append(
            {
                "prefix_length": milestone.prefix_length,
                "step_id": milestone.step_id,
                "target_ref": milestone.target_ref,
                "cursor_before_ref": milestone.cursor_before_ref,
                "cursor_before": before,
                "cursor_after_ref": milestone.cursor_after_ref,
                "cursor_after": after,
            }
        )
    return tuple(rows)


def compile_from_runtime_binding(
    task: SemanticTask,
    action_schema: str,
    binding: ValidatedRuntimeBinding,
    *,
    semantic_step_index: int,
    near_miss: bool = False,
) -> Any:
    from .program import compile_semantic_step

    return compile_semantic_step(
        task,
        action_schema,
        binding=binding,
        semantic_step_index=semantic_step_index,
        near_miss=near_miss,
    )
