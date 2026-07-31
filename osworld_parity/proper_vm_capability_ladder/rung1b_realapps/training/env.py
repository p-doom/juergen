from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from ...rung1.executor import CompactRawExecutor, NativeAbsoluteExecutor
from ...rung1.vm import KvmFixtureSession
from ..fixtures import Fixture
from ..oracle import evaluate_state
from ..trajectory import UiGeometry
from ..vm import GuestFixture, probe_fixture, setup_fixture
from .splits import materialize_tasks


@dataclass(frozen=True)
class PolicyObservation:
    instruction: str
    screenshot_png: bytes

    def as_model_input(self) -> dict[str, Any]:
        return {"instruction": self.instruction, "screenshot_png": self.screenshot_png}


@dataclass(frozen=True)
class BackendSnapshot:
    screenshot_png: bytes
    hidden_state: dict[str, Any]
    geometry: UiGeometry
    cursor: tuple[int, int]


class EnvironmentBackend(Protocol):
    def reset(self, fixture: Fixture) -> BackendSnapshot: ...
    def dispatch(self, fixture: Fixture, arm: str, action: dict[str, Any] | str) -> BackendSnapshot: ...


class VmEnvironmentBackend:
    """KVM backend. Hidden state stays here and is never returned in observation."""

    def __init__(self, session: KvmFixtureSession) -> None:
        self.session = session
        self.transport = None
        self.geometry = UiGeometry()

    def _screenshot(self) -> bytes:
        if self.transport is None:
            raise RuntimeError("VM backend was not reset")
        with urllib.request.urlopen(
            self.transport.base_url + "/screenshot", timeout=self.transport.timeout_s
        ) as response:
            value = response.read()
        if not value.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("VM screenshot endpoint did not return PNG")
        return value

    def reset(self, fixture: Fixture) -> BackendSnapshot:
        self.transport = self.session.reset_to_ready()
        guest = setup_fixture(self.transport, fixture)
        self.geometry = guest.geometry
        return BackendSnapshot(
            self._screenshot(),
            guest.state,
            guest.geometry,
            self.transport.cursor_position(),
        )

    def dispatch(
        self, fixture: Fixture, arm: str, action: dict[str, Any] | str
    ) -> BackendSnapshot:
        if self.transport is None:
            raise RuntimeError("VM backend was not reset")
        if arm == "native_absolute_control":
            if not isinstance(action, dict):
                raise TypeError("native action must be an object")
            result = NativeAbsoluteExecutor(self.transport).execute(action)
        elif arm == "compact_raw_phaseb":
            if not isinstance(action, str):
                raise TypeError("compact action must be text")
            result = CompactRawExecutor(self.transport).execute(action)
        else:
            raise ValueError(f"unsupported arm: {arm}")
        if result.executor_dispatch_status != "ok":
            raise RuntimeError(f"executor dispatch failed: {result}")
        return BackendSnapshot(
            self._screenshot(),
            probe_fixture(self.transport, fixture),
            self.geometry,
            self.transport.cursor_position(),
        )


class Rung1bTrainingEnv:
    """Small Gym-like contract with trainer-only hidden reward evaluation."""

    def __init__(
        self,
        backend: EnvironmentBackend,
        *,
        split: str = "train",
        arm: str = "compact_raw_phaseb",
    ) -> None:
        if split == "evaluation_sealed":
            raise ValueError("sealed evaluation split cannot be opened by a training environment")
        self.backend = backend
        self.tasks = materialize_tasks(split)
        self.split = split
        self.arm = arm
        self._fixture: Fixture | None = None
        self._hidden_state: dict[str, Any] | None = None
        self._steps = 0
        self._done = True

    def reset(self, *, task_index: int = 0) -> tuple[PolicyObservation, dict[str, Any]]:
        fixture = self.tasks[task_index % len(self.tasks)]
        snapshot = self.backend.reset(fixture)
        self._fixture = fixture
        self._hidden_state = snapshot.hidden_state
        self._steps = 0
        self._done = False
        observation = PolicyObservation(fixture.instruction, snapshot.screenshot_png)
        return observation, {
            "task_id": fixture.id,
            "split": fixture.split,
            "horizon": fixture.horizon,
            "arm": self.arm,
        }

    def step(
        self, action: dict[str, Any] | str
    ) -> tuple[PolicyObservation, float, bool, bool, dict[str, Any]]:
        if self._done or self._fixture is None:
            raise RuntimeError("environment requires reset")
        snapshot = self.backend.dispatch(self._fixture, self.arm, action)
        self._hidden_state = snapshot.hidden_state
        self._steps += 1
        result = evaluate_state(self._fixture, snapshot.hidden_state)
        if result.oracle_status != "ok":
            raise RuntimeError(f"trainer-only hidden oracle failed: {result.reason}")
        reward = 1.0 if result.MOUSE_SOLVED else 0.0
        terminated = bool(result.MOUSE_SOLVED)
        truncated = self._steps >= self._fixture.horizon and not terminated
        self._done = terminated or truncated
        observation = PolicyObservation(self._fixture.instruction, snapshot.screenshot_png)
        info = {
            "task_id": self._fixture.id,
            "step": self._steps,
            "horizon": self._fixture.horizon,
            "arm": self.arm,
        }
        return observation, reward, terminated, truncated, info


class DeterministicTestBackend:
    """Unit-test backend; reward state changes only on an explicit success action."""

    def __init__(self) -> None:
        self.fixture: Fixture | None = None
        self.state: dict[str, Any] = {}
        self.counter = 0

    def reset(self, fixture: Fixture) -> BackendSnapshot:
        from ..states import reset_state

        self.fixture = fixture
        self.state = reset_state(fixture)
        self.counter = 0
        return BackendSnapshot(b"\x89PNG\r\n\x1a\nreset", self.state, UiGeometry(), (50, 50))

    def dispatch(
        self, fixture: Fixture, arm: str, action: dict[str, Any] | str
    ) -> BackendSnapshot:
        from ..states import gold_state

        self.counter += 1
        success = action == {"test": "gold"} or action == "TEST_GOLD"
        if success:
            self.state = gold_state(fixture)
        return BackendSnapshot(
            b"\x89PNG\r\n\x1a\n" + str(self.counter).encode(),
            self.state,
            UiGeometry(),
            (50, 50),
        )
