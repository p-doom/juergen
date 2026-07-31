from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from stage5_rft.collector import (  # noqa: E402
    ActorRequest,
    EnvObservation,
    EnvTransition,
    PolicyOutput,
)
from stage5_rft.schema import (  # noqa: E402
    FailureKind,
    PolicyProvenance,
    ResetSpec,
    TaskSpec,
)
from stage5_rft.util import sha256_bytes, sha256_json  # noqa: E402


INITIAL_PNG = b"mock-png-position-0"
INITIAL_STATE = {"position": 0, "screen": "mock"}


def digest(char: str) -> str:
    return char * 64


def make_policy(*, role: str = "candidate", action_schema: str = "compact_raw.v1") -> PolicyProvenance:
    return PolicyProvenance(
        policy_id=f"policy-{role}",
        version="v1",
        checkpoint_uri=f"/fast/project/mock/{role}/checkpoint",
        checkpoint_sha256=digest("a" if role == "candidate" else "b"),
        source_repo="github.com/p-doom/juergen",
        source_commit=digest("c"),
        action_schema=action_schema,
        sampling={"temperature": 0.0, "top_p": 1.0, "max_tokens": 64},
        role=role,
    )


def make_task(
    episode_id: str,
    *,
    task_id: str | None = None,
    max_steps: int = 2,
    seed: int = 7,
) -> TaskSpec:
    instruction = f"Advance task {task_id or episode_id}"
    return TaskSpec(
        episode_id=episode_id,
        instruction=instruction,
        instruction_sha256=sha256_bytes(instruction.encode("utf-8")),
        condition="single_step" if max_steps == 1 else "multi_step",
        max_steps=max_steps,
        reward_schema="mock.task_reward.v1",
        reward_config_sha256=digest("8"),
        reset=ResetSpec(
            task_id=task_id or episode_id,
            task_content_sha256=sha256_json({"task": task_id or episode_id}),
            source_split="train_adjacent",
            vm_snapshot_id="mock-snapshot-v1",
            vm_snapshot_sha256=digest("d"),
            setup_sha256=digest("e"),
            seed=seed,
            reset_protocol="mock.reset.v1",
            state_schema="mock.vm_state.v1",
            expected_initial_screenshot_sha256=sha256_bytes(INITIAL_PNG),
            expected_initial_state_sha256=sha256_json(INITIAL_STATE),
        ),
    )


class MockEnv:
    def __init__(self, goals: Mapping[str, int] | None = None, *, drift: bool = False) -> None:
        self.goals = dict(goals or {})
        self.position = 0
        self.task_id = ""
        self.closed = False
        self.drift = drift

    def _observation(self) -> EnvObservation:
        position = self.position + (1 if self.drift else 0)
        return EnvObservation(
            screenshot_png=f"mock-png-position-{position}".encode(),
            state={"position": position, "screen": "mock"},
        )

    def reset(self, spec: ResetSpec) -> EnvObservation:
        self.task_id = spec.task_id
        self.position = 0
        return self._observation()

    def step(self, action: Mapping[str, Any]) -> EnvTransition:
        self.position += int(action["delta"])
        goal = self.goals.get(self.task_id, 2)
        success = self.position >= goal
        return EnvTransition(
            observation=self._observation(),
            reward=1.0 if success else 0.0,
            done=success,
            task_success=success,
            failure_kind=FailureKind.NONE,
            info={"goal": goal},
        )

    def close(self) -> None:
        self.closed = True


class MockActor:
    def __init__(
        self,
        policy: PolicyProvenance,
        *,
        interrupt_on_call: int | None = None,
        served_fingerprint: str | None = None,
        invalid: bool = False,
    ) -> None:
        self._policy = policy
        self.calls = 0
        self.interrupt_on_call = interrupt_on_call
        self.served_fingerprint = served_fingerprint
        self.invalid = invalid

    @property
    def provenance(self) -> PolicyProvenance:
        return self._policy

    def sample(self, request: ActorRequest) -> PolicyOutput:
        self.calls += 1
        if self.interrupt_on_call == self.calls:
            raise KeyboardInterrupt("simulated preemption")
        return PolicyOutput(
            raw_output="{\"delta\":1}" if not self.invalid else "not-json",
            parsed_action={"delta": 1} if not self.invalid else None,
            parser="mock.parser.v1",
            served_policy_fingerprint=(
                self.served_fingerprint or self._policy.fingerprint
            ),
            logprob=-0.1,
        )


@pytest.fixture
def blocklist():
    from stage5_rft.contamination import ContaminationBlocklist

    return ContaminationBlocklist(
        frozenset(), frozenset(), "mock-heldout-hashes", testing_only=True
    )
