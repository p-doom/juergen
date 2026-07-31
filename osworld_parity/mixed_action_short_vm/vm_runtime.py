from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from osworld_parity.proper_vm_capability_ladder.rung1.executor import (
    CompactRawExecutor,
    NativeAbsoluteExecutor,
)

from .manifest import TaskDefinition, canonical_bytes
from .runtime import Arm, Observation, ResetReceipt, StepResult


RunMode = Literal["teacher_collection", "development_replay", "scientific_evaluation"]


class VmRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class GateAuthorization:
    suite: str
    sealed_manifest_payload_sha256: str
    owner_approval: bool
    status: str
    authorization_sha256: str


@dataclass(frozen=True)
class VmResetState:
    reset_fingerprint: str
    transport: Any


@dataclass(frozen=True)
class HostScore:
    oracle_status: str
    solved: bool
    hidden_state_sha256: str


class VmHostHooks(Protocol):
    """Trainer-owned VM boundary; none of these methods are policy-visible."""

    def reset_to_ready(self, task: TaskDefinition) -> VmResetState: ...

    def capture_observation(
        self, task: TaskDefinition, *, step_index: int, include_instruction: bool
    ) -> Observation: ...

    def score_hidden_state(self, task: TaskDefinition) -> HostScore: ...


def load_gate_authorization(
    path: Path, *, expected_sealed_manifest_payload_sha256: str
) -> GateAuthorization:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VmRuntimeError(f"cannot read scientific gate receipt: {exc}") from exc
    if not isinstance(raw, dict):
        raise VmRuntimeError("scientific gate receipt must be an object")
    observed = raw.pop("authorization_sha256", None)
    digest = hashlib.sha256(canonical_bytes(raw)).hexdigest()
    if not isinstance(observed, str) or observed != digest:
        raise VmRuntimeError("scientific gate receipt seal mismatch")
    if raw.get("suite") != "roadmap_3_3_mixed_action_short_vm":
        raise VmRuntimeError("scientific gate suite mismatch")
    if raw.get("status") != "approved" or raw.get("owner_approval") is not True:
        raise VmRuntimeError("scientific execution is not owner-approved")
    if (
        raw.get("sealed_manifest_payload_sha256")
        != expected_sealed_manifest_payload_sha256
    ):
        raise VmRuntimeError("scientific gate references a different sealed manifest")
    return GateAuthorization(
        suite=str(raw["suite"]),
        sealed_manifest_payload_sha256=str(raw["sealed_manifest_payload_sha256"]),
        owner_approval=True,
        status="approved",
        authorization_sha256=observed,
    )


class VmEpisode:
    """VM-backed policy API using the audited common native/compact executors."""

    def __init__(
        self,
        task: TaskDefinition,
        arm: Arm,
        hooks: VmHostHooks,
        *,
        mode: RunMode,
        gate_authorization: GateAuthorization | None = None,
        task_manifest_payload_sha256: str | None = None,
    ) -> None:
        if arm not in {"native_absolute_control", "compact_raw_phaseb"}:
            raise VmRuntimeError(f"unknown VM action arm: {arm}")
        if mode not in {
            "teacher_collection",
            "development_replay",
            "scientific_evaluation",
        }:
            raise VmRuntimeError(f"unknown VM run mode: {mode}")
        if mode == "scientific_evaluation":
            if gate_authorization is None:
                raise VmRuntimeError(
                    "scientific evaluation is fail-closed until an owner gate receipt "
                    "is supplied"
                )
            if (
                gate_authorization.suite
                != "roadmap_3_3_mixed_action_short_vm"
                or gate_authorization.status != "approved"
                or not gate_authorization.owner_approval
            ):
                raise VmRuntimeError("scientific gate receipt is not owner-approved")
            if task.split != "sealed_evaluation":
                raise VmRuntimeError(
                    "scientific evaluation requires an owner-materialized sealed task"
                )
            if (
                task_manifest_payload_sha256
                != gate_authorization.sealed_manifest_payload_sha256
            ):
                raise VmRuntimeError(
                    "scientific task is not bound to the authorized sealed manifest"
                )
        elif gate_authorization is not None or task_manifest_payload_sha256 is not None:
            raise VmRuntimeError(
                "sealed manifest/gate receipt is only valid for scientific evaluation"
            )
        elif mode == "development_replay" and task.split != "development":
            raise VmRuntimeError("development replay accepts only development tasks")
        elif mode == "teacher_collection" and task.split not in {
            "train",
            "development",
        }:
            raise VmRuntimeError("teacher collection cannot access sealed evaluation")
        self.task = task
        self.arm = arm
        self.mode = mode
        self._hooks = hooks
        self._generation = 0
        self._step_index = 0
        self._done = True
        self._executor: NativeAbsoluteExecutor | CompactRawExecutor | None = None
        self._canonical_reset_fingerprint: str | None = None

    def reset(self) -> ResetReceipt:
        state = self._hooks.reset_to_ready(self.task)
        if self._canonical_reset_fingerprint is None:
            self._canonical_reset_fingerprint = state.reset_fingerprint
        elif state.reset_fingerprint != self._canonical_reset_fingerprint:
            raise VmRuntimeError("VM reset fingerprint drifted across episodes")
        reset_score = self._hooks.score_hidden_state(self.task)
        if reset_score.oracle_status != "ok":
            raise VmRuntimeError("reset-state host oracle returned non-ok status")
        if reset_score.solved:
            raise VmRuntimeError("reset-state host oracle unexpectedly accepted")
        self._generation += 1
        self._step_index = 0
        self._done = False
        self._executor = (
            NativeAbsoluteExecutor(state.transport)
            if self.arm == "native_absolute_control"
            else CompactRawExecutor(state.transport)
        )
        observation = self._hooks.capture_observation(
            self.task, step_index=0, include_instruction=True
        )
        return ResetReceipt(
            task_id=self.task.task_id,
            task_sha256=self.task.task_sha256,
            reset_fingerprint=state.reset_fingerprint,
            generation=self._generation,
            observation=observation,
        )

    def step(self, action: dict[str, Any] | str) -> StepResult:
        if self._done or self._executor is None:
            raise VmRuntimeError("VM step called before reset or after done")
        if self._step_index >= self.task.horizon:
            raise VmRuntimeError("VM step called beyond frozen horizon")
        try:
            dispatch = self._executor.execute(action)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise VmRuntimeError(f"VM action parse/dispatch failed: {exc}") from exc
        if dispatch.executor_dispatch_status != "ok":
            raise VmRuntimeError("VM executor dispatch returned non-ok status")
        self._step_index += 1
        score = self._hooks.score_hidden_state(self.task)
        if score.oracle_status != "ok":
            raise VmRuntimeError("host-only state oracle returned non-ok status")
        truncated = not score.solved and self._step_index == self.task.horizon
        self._done = score.solved or truncated
        observation = self._hooks.capture_observation(
            self.task, step_index=self._step_index, include_instruction=False
        )
        return StepResult(
            observation=observation,
            reward=1.0 if score.solved else 0.0,
            done=self._done,
            truncated=truncated,
            step_index=self._step_index,
            horizon=self.task.horizon,
            parse_status=dispatch.parse_status,
            executor_dispatch_status=dispatch.executor_dispatch_status,
            action_class=dispatch.action_class,
        )
