"""Versioned, content-addressed task-level rollout schema.

Every step carries both screenshots, both structured states, the raw and parsed
action, reward, and terminal flags.  Actor checkpoint identity is part of every
episode rather than an ambient process setting.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from stage5_rft.util import (
    ContractError,
    require_nonempty,
    require_sha256,
    sha256_bytes,
    sha256_json,
)


SCHEMA_VERSION = "stage5.vm_episode.v1"
ALLOWED_SOURCE_SPLITS = frozenset({"train", "train_adjacent", "validation"})


def is_compact_raw_action_schema(value: str) -> bool:
    normalized = value.lower().replace("-", "_")
    return "compact_raw" in normalized or "deltatype_raw" in normalized


class FailureKind(StrEnum):
    NONE = "none"
    RESET_FAILED = "reset_failed"
    RESET_NONDETERMINISTIC = "reset_nondeterministic"
    POLICY_TIMEOUT = "policy_timeout"
    POLICY_ERROR = "policy_error"
    POLICY_PROVENANCE_MISMATCH = "policy_provenance_mismatch"
    PARSE_ERROR = "parse_error"
    SCHEMA_ERROR = "schema_error"
    INVALID_ACTION = "invalid_action"
    DISPATCH_ERROR = "dispatch_error"
    OBSERVATION_ERROR = "observation_error"
    REWARD_ERROR = "reward_error"
    VM_ERROR = "vm_error"
    TASK_FAILURE = "task_failure"
    MAX_STEPS = "max_steps"
    ACTOR_INTERRUPTED = "actor_interrupted"
    REPLAY_DIVERGENCE = "replay_divergence"
    CONTAMINATION = "contamination"


@dataclass(frozen=True)
class PolicyProvenance:
    policy_id: str
    version: str
    checkpoint_uri: str
    checkpoint_sha256: str
    source_repo: str
    source_commit: str
    action_schema: str
    sampling: dict[str, Any]
    role: str = "candidate"

    def validate(self) -> None:
        for name in (
            "policy_id",
            "version",
            "checkpoint_uri",
            "source_repo",
            "source_commit",
            "action_schema",
        ):
            require_nonempty(str(getattr(self, name)), f"policy.{name}")
        require_sha256(self.checkpoint_sha256, "policy.checkpoint_sha256")
        if self.role not in {"candidate", "native_absolute_baseline"}:
            raise ContractError(f"unsupported policy role: {self.role!r}")
        if self.role == "candidate" and not is_compact_raw_action_schema(self.action_schema):
            raise ContractError(
                "Stage-5 candidate policy must declare compact_raw/deltatype_raw action schema"
            )
        if self.role == "native_absolute_baseline" and "absolute" not in self.action_schema.lower():
            raise ContractError(
                "native_absolute_baseline policy must declare an absolute action schema"
            )
        if not isinstance(self.sampling, dict):
            raise ContractError("policy.sampling must be an explicit JSON object")
        required = {"temperature", "top_p", "max_tokens"}
        missing = sorted(required - self.sampling.keys())
        if missing:
            raise ContractError(f"policy.sampling lacks required fields: {missing}")

    @property
    def fingerprint(self) -> str:
        self.validate()
        return sha256_json(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicyProvenance":
        obj = cls(**dict(value))
        obj.validate()
        return obj


@dataclass(frozen=True)
class ResetSpec:
    task_id: str
    task_content_sha256: str
    source_split: str
    vm_snapshot_id: str
    vm_snapshot_sha256: str
    setup_sha256: str
    seed: int
    reset_protocol: str
    state_schema: str
    expected_initial_screenshot_sha256: str
    expected_initial_state_sha256: str

    def validate(self) -> None:
        require_nonempty(self.task_id, "reset.task_id")
        require_nonempty(self.vm_snapshot_id, "reset.vm_snapshot_id")
        require_nonempty(self.reset_protocol, "reset.reset_protocol")
        require_nonempty(self.state_schema, "reset.state_schema")
        for name in (
            "task_content_sha256",
            "vm_snapshot_sha256",
            "setup_sha256",
            "expected_initial_screenshot_sha256",
            "expected_initial_state_sha256",
        ):
            require_sha256(str(getattr(self, name)), f"reset.{name}")
        if self.source_split not in ALLOWED_SOURCE_SPLITS:
            raise ContractError(
                f"reset.source_split={self.source_split!r} is not collection-authorized; "
                f"allowed={sorted(ALLOWED_SOURCE_SPLITS)}"
            )
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ContractError("reset.seed must be a non-negative integer")

    @property
    def fingerprint(self) -> str:
        self.validate()
        return sha256_json(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResetSpec":
        obj = cls(**dict(value))
        obj.validate()
        return obj


@dataclass(frozen=True)
class TaskSpec:
    episode_id: str
    instruction: str
    instruction_sha256: str
    condition: str
    max_steps: int
    reward_schema: str
    reward_config_sha256: str
    reset: ResetSpec

    def validate(self) -> None:
        require_nonempty(self.episode_id, "task.episode_id")
        require_nonempty(self.instruction, "task.instruction")
        require_sha256(self.instruction_sha256, "task.instruction_sha256")
        if sha256_bytes(self.instruction.encode("utf-8")) != self.instruction_sha256:
            raise ContractError("task.instruction_sha256 does not match instruction")
        if self.condition not in {"single_step", "multi_step"}:
            raise ContractError("task.condition must be single_step or multi_step")
        if self.condition == "single_step" and self.max_steps != 1:
            raise ContractError("single_step tasks must set max_steps=1")
        if self.condition == "multi_step" and self.max_steps < 2:
            raise ContractError("multi_step tasks must set max_steps>=2")
        require_nonempty(self.reward_schema, "task.reward_schema")
        require_sha256(self.reward_config_sha256, "task.reward_config_sha256")
        self.reset.validate()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSpec":
        data = dict(value)
        data["reset"] = ResetSpec.from_dict(data["reset"])
        obj = cls(**data)
        obj.validate()
        return obj


@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    sha256: str
    media_type: str = "image/png"
    size_bytes: int = 0

    def validate(self) -> None:
        require_nonempty(self.uri, "artifact.uri")
        require_sha256(self.sha256, "artifact.sha256")
        require_nonempty(self.media_type, "artifact.media_type")
        if self.size_bytes < 0:
            raise ContractError("artifact.size_bytes cannot be negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        obj = cls(**dict(value))
        obj.validate()
        return obj


@dataclass(frozen=True)
class StateRef:
    payload: dict[str, Any]
    sha256: str

    def validate(self) -> None:
        require_sha256(self.sha256, "state.sha256")
        if sha256_json(self.payload) != self.sha256:
            raise ContractError("state.sha256 does not match canonical state payload")

    @classmethod
    def capture(cls, payload: Mapping[str, Any]) -> "StateRef":
        data = dict(payload)
        return cls(payload=data, sha256=sha256_json(data))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateRef":
        obj = cls(payload=dict(value["payload"]), sha256=str(value["sha256"]))
        obj.validate()
        return obj


@dataclass(frozen=True)
class ActionTrace:
    raw_output: str
    parsed_action: dict[str, Any] | None
    parser: str
    schema: str
    served_policy_fingerprint: str
    valid: bool
    dispatched: bool
    logprob: float | None = None

    def validate(self) -> None:
        require_nonempty(self.parser, "action.parser")
        require_nonempty(self.schema, "action.schema")
        require_sha256(
            self.served_policy_fingerprint, "action.served_policy_fingerprint"
        )
        if self.valid != (self.parsed_action is not None):
            raise ContractError("action.valid must exactly match parsed_action presence")
        if self.dispatched and not self.valid:
            raise ContractError("invalid action must never be dispatched")
        if self.logprob is not None and not math.isfinite(self.logprob):
            raise ContractError("action.logprob must be finite when present")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionTrace":
        data = dict(value)
        if data.get("parsed_action") is not None:
            data["parsed_action"] = dict(data["parsed_action"])
        obj = cls(**data)
        obj.validate()
        return obj


@dataclass(frozen=True)
class StepTrace:
    step_index: int
    request_id: str
    sampling_seed: int
    screenshot_before: ArtifactRef
    state_before: StateRef
    action: ActionTrace
    screenshot_after: ArtifactRef
    state_after: StateRef
    reward: float
    done: bool
    task_success: bool
    failure_kind: FailureKind = FailureKind.NONE
    elapsed_ms: int = 0
    info: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.step_index < 0:
            raise ContractError("step_index must be non-negative")
        require_nonempty(self.request_id, "step.request_id")
        if self.sampling_seed < 0:
            raise ContractError("step.sampling_seed must be non-negative")
        self.screenshot_before.validate()
        self.screenshot_after.validate()
        self.state_before.validate()
        self.state_after.validate()
        self.action.validate()
        if not math.isfinite(self.reward):
            raise ContractError("step.reward must be finite")
        if self.task_success and not self.done:
            raise ContractError("task_success requires done=true")
        if self.task_success and self.failure_kind != FailureKind.NONE:
            raise ContractError("successful terminal step cannot carry failure_kind")
        if self.elapsed_ms < 0:
            raise ContractError("step.elapsed_ms cannot be negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StepTrace":
        data = dict(value)
        data["screenshot_before"] = ArtifactRef.from_dict(data["screenshot_before"])
        data["screenshot_after"] = ArtifactRef.from_dict(data["screenshot_after"])
        data["state_before"] = StateRef.from_dict(data["state_before"])
        data["state_after"] = StateRef.from_dict(data["state_after"])
        data["action"] = ActionTrace.from_dict(data["action"])
        data["failure_kind"] = FailureKind(data.get("failure_kind", "none"))
        data["info"] = dict(data.get("info", {}))
        obj = cls(**data)
        obj.validate()
        return obj


@dataclass(frozen=True)
class EpisodeTrace:
    schema_version: str
    episode_id: str
    instruction: str
    instruction_sha256: str
    condition: str
    max_steps: int
    reward_schema: str
    reward_config_sha256: str
    policy: PolicyProvenance
    actor_id: str
    reset: ResetSpec
    collection_attempt: int
    steps: tuple[StepTrace, ...]
    total_reward: float
    success: bool
    terminal_failure: FailureKind
    complete: bool = True

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"unsupported episode schema: {self.schema_version!r}")
        require_nonempty(self.episode_id, "episode.episode_id")
        require_nonempty(self.actor_id, "episode.actor_id")
        require_sha256(self.instruction_sha256, "episode.instruction_sha256")
        if sha256_bytes(self.instruction.encode("utf-8")) != self.instruction_sha256:
            raise ContractError("episode instruction digest mismatch")
        if self.condition not in {"single_step", "multi_step"}:
            raise ContractError("episode condition must be single_step or multi_step")
        if self.max_steps < 1 or len(self.steps) > self.max_steps:
            raise ContractError("episode step count violates max_steps")
        if self.condition == "single_step" and self.max_steps != 1:
            raise ContractError("single_step episode must set max_steps=1")
        if self.condition == "multi_step" and self.max_steps < 2:
            raise ContractError("multi_step episode must set max_steps>=2")
        require_nonempty(self.reward_schema, "episode.reward_schema")
        require_sha256(self.reward_config_sha256, "episode.reward_config_sha256")
        if not self.steps:
            raise ContractError("episode must contain at least one trace step")
        self.policy.validate()
        self.reset.validate()
        if self.collection_attempt < 1:
            raise ContractError("collection_attempt must be >=1")
        for expected, step in enumerate(self.steps):
            step.validate()
            if step.action.served_policy_fingerprint != self.policy.fingerprint:
                raise ContractError(
                    "step served-policy fingerprint differs from episode policy"
                )
            if step.step_index != expected:
                raise ContractError("steps must have contiguous zero-based indices")
            if expected and self.steps[expected - 1].done:
                raise ContractError("trace contains a step after done=true")
            if expected and step.state_before.sha256 != self.steps[expected - 1].state_after.sha256:
                raise ContractError("adjacent state hashes do not join")
            if expected and step.screenshot_before.sha256 != self.steps[expected - 1].screenshot_after.sha256:
                raise ContractError("adjacent screenshot hashes do not join")
        if self.complete and not self.steps[-1].done:
            raise ContractError("complete episode must terminate with done=true")
        if self.success != self.steps[-1].task_success:
            raise ContractError("episode success disagrees with terminal step")
        expected_failure = self.steps[-1].failure_kind if not self.success else FailureKind.NONE
        if self.terminal_failure != expected_failure:
            raise ContractError("episode terminal_failure disagrees with terminal step")
        if not math.isclose(self.total_reward, sum(s.reward for s in self.steps), abs_tol=1e-9):
            raise ContractError("episode total_reward disagrees with step rewards")

    @property
    def trace_sha256(self) -> str:
        self.validate()
        return sha256_json(self.as_dict())

    @property
    def match_key(self) -> str:
        return sha256_json(
            {
                "instruction_sha256": self.instruction_sha256,
                "condition": self.condition,
                "max_steps": self.max_steps,
                "reward_schema": self.reward_schema,
                "reward_config_sha256": self.reward_config_sha256,
                "reset": asdict(self.reset),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeTrace":
        data = dict(value)
        data["policy"] = PolicyProvenance.from_dict(data["policy"])
        data["reset"] = ResetSpec.from_dict(data["reset"])
        data["steps"] = tuple(StepTrace.from_dict(v) for v in data["steps"])
        data["terminal_failure"] = FailureKind(data.get("terminal_failure", "none"))
        obj = cls(**data)
        obj.validate()
        return obj
