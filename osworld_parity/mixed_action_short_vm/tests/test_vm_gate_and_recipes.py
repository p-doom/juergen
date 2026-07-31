from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from osworld_parity.mixed_action_short_vm.manifest import (
    load_authorized_tasks,
    payload_sha256,
)
from osworld_parity.mixed_action_short_vm.runtime import (
    Observation,
    StateTrackingTransport,
)
from osworld_parity.mixed_action_short_vm.teacher import native_gold_actions
from osworld_parity.mixed_action_short_vm.vm_runtime import (
    GateAuthorization,
    HostScore,
    VmEpisode,
    VmResetState,
    VmRuntimeError,
)


ROOT = Path(__file__).resolve().parents[3]


class _NeverCalledHooks:
    def reset_to_ready(self, task):  # pragma: no cover - constructor must stop first
        raise AssertionError("gate failed open")

    def capture_observation(self, task, *, step_index, include_instruction):
        raise AssertionError("gate failed open")

    def score_hidden_state(self, task):
        raise AssertionError("gate failed open")


class _DevelopmentHooks:
    def __init__(self) -> None:
        self.transport = None

    def reset_to_ready(self, task):
        self.transport = StateTrackingTransport(task)
        return VmResetState(
            reset_fingerprint=payload_sha256(self.transport.hidden_snapshot()),
            transport=self.transport,
        )

    def capture_observation(self, task, *, step_index, include_instruction):
        return Observation(
            task_id=task.task_id,
            instruction=task.instruction if include_instruction else None,
            frame_uri=f"vm-frame://{task.task_id}/{step_index}",
            frame_sha256=payload_sha256(
                {"task_id": task.task_id, "step_index": step_index}
            ),
            step_index=step_index,
            horizon=task.horizon,
        )

    def score_hidden_state(self, task):
        assert self.transport is not None
        return HostScore(
            oracle_status="ok",
            solved=self.transport.solved(),
            hidden_state_sha256=payload_sha256(self.transport.hidden_snapshot()),
        )


def test_scientific_vm_episode_is_fail_closed_without_owner_gate() -> None:
    task = load_authorized_tasks("development")[0]
    with pytest.raises(VmRuntimeError, match="fail-closed"):
        VmEpisode(
            task,
            "native_absolute_control",
            _NeverCalledHooks(),
            mode="scientific_evaluation",
        )


def test_scientific_gate_cannot_authorize_a_development_task() -> None:
    task = load_authorized_tasks("development")[0]
    authorization = GateAuthorization(
        suite="roadmap_3_3_mixed_action_short_vm",
        sealed_manifest_payload_sha256="0" * 64,
        owner_approval=True,
        status="approved",
        authorization_sha256="1" * 64,
    )
    with pytest.raises(VmRuntimeError, match="owner-materialized sealed task"):
        VmEpisode(
            task,
            "native_absolute_control",
            _NeverCalledHooks(),
            mode="scientific_evaluation",
            gate_authorization=authorization,
            task_manifest_payload_sha256="0" * 64,
        )


def test_vm_development_api_uses_shared_executor_and_hidden_host_oracle() -> None:
    task = load_authorized_tasks("development")[0]
    episode = VmEpisode(
        task,
        "native_absolute_control",
        _DevelopmentHooks(),
        mode="development_replay",
    )
    reset_a = episode.reset()
    final = None
    for action in native_gold_actions(task):
        final = episode.step(action)
    assert final is not None and final.done and final.reward == 1
    reset_b = episode.reset()
    assert reset_a.reset_fingerprint == reset_b.reset_fingerprint


def test_labctl_recipes_are_cpu_only_and_do_not_launch_models() -> None:
    recipes = (
        ROOT
        / "osworld_parity"
        / "labctl"
        / "recipes"
        / "roadmap33_mixed_action_build_cpu.toml",
        ROOT
        / "osworld_parity"
        / "labctl"
        / "recipes"
        / "roadmap33_mixed_action_teacher_cpu.toml",
    )
    for path in recipes:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
        assert value["resources"]["gpus"] == 0
        command = " ".join(value["command"])
        assert "CUDA_VISIBLE_DEVICES" in command
        assert "sglang" not in command.lower()
        assert "train.py" not in command.lower()
        assert "sealed_evaluation" not in command
