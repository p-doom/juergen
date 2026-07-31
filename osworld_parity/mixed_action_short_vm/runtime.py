from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

from osworld_parity.proper_vm_capability_ladder.rung1.executor import (
    CompactRawExecutor,
    DispatchResult,
    NativeAbsoluteExecutor,
)
from osworld_parity.proper_vm_capability_ladder.rung1.transport import (
    RecordingTransport,
)

from .manifest import CANVAS, TaskDefinition, payload_sha256


Arm = Literal["native_absolute_control", "compact_raw_phaseb"]


class EpisodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Observation:
    task_id: str
    instruction: str | None
    frame_uri: str
    frame_sha256: str
    step_index: int
    horizon: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "frame_uri": self.frame_uri,
            "frame_sha256": self.frame_sha256,
            "step_index": self.step_index,
            "horizon": self.horizon,
        }


@dataclass(frozen=True)
class ResetReceipt:
    task_id: str
    task_sha256: str
    reset_fingerprint: str
    generation: int
    observation: Observation


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    done: bool
    truncated: bool
    step_index: int
    horizon: int
    parse_status: str
    executor_dispatch_status: str
    action_class: str

    def as_policy_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.as_dict(),
            "reward": self.reward,
            "done": self.done,
            "truncated": self.truncated,
            "step_index": self.step_index,
            "horizon": self.horizon,
            "parse_status": self.parse_status,
            "executor_dispatch_status": self.executor_dispatch_status,
            "action_class": self.action_class,
        }


@dataclass
class _HiddenState:
    completed_steps: list[str]
    focused: bool
    text: str
    scroll_total: int
    clicked: bool
    drag_complete: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "completed_steps": list(self.completed_steps),
            "focused": self.focused,
            "text": self.text,
            "scroll_total": self.scroll_total,
            "clicked": self.clicked,
            "drag_complete": self.drag_complete,
        }


class StateTrackingTransport(RecordingTransport):
    """CPU contract backend driven by the shared production action executors.

    It is intentionally trainer-side. A VM backend can supply the same transport
    interface while reading its hidden final state from a host-only oracle.
    """

    HIT_RADIUS = 22

    def __init__(self, task: TaskDefinition) -> None:
        super().__init__(cursor=task.geometry.initial_cursor, screen=CANVAS)
        self.task = task
        self.state = _HiddenState(
            completed_steps=[],
            focused=False,
            text=task.initial_text,
            scroll_total=0,
            clicked=False,
            drag_complete=False,
        )
        self._mouse_down_at: tuple[int, int] | None = None
        self._mouse_moved_while_down = False
        self._drag_started = False

    @staticmethod
    def _near(left: tuple[int, int], right: tuple[int, int], radius: int) -> bool:
        return abs(left[0] - right[0]) <= radius and abs(left[1] - right[1]) <= radius

    def _expected_step(self) -> str | None:
        index = len(self.state.completed_steps)
        return self.task.steps[index] if index < len(self.task.steps) else None

    def _complete(self, kind: str) -> None:
        if self._expected_step() == kind:
            self.state.completed_steps.append(kind)

    def move_to(self, x: int, y: int) -> None:
        before = self.cursor_position()
        super().move_to(x, y)
        if self.audit.held_buttons and self.cursor_position() != before:
            self._mouse_moved_while_down = True

    def mouse_down(self, button: str = "left") -> None:
        super().mouse_down(button)
        if button == "left":
            self._mouse_down_at = self.cursor_position()
            self._mouse_moved_while_down = False
            self._drag_started = self._near(
                self.cursor_position(), self.task.geometry.drag_start, self.HIT_RADIUS
            )

    def mouse_up(self, button: str = "left") -> None:
        release_at = self.cursor_position()
        down_at = self._mouse_down_at
        moved = self._mouse_moved_while_down
        drag_started = self._drag_started
        super().mouse_up(button)
        if button != "left" or down_at is None:
            return
        if (
            drag_started
            and moved
            and self._near(release_at, self.task.geometry.drag_end, self.HIT_RADIUS)
        ):
            self.state.drag_complete = True
            self._complete("drag")
        elif not moved:
            if self._near(release_at, self.task.geometry.field_center, self.HIT_RADIUS):
                self.state.focused = True
                self._complete("focus")
            elif self._near(
                release_at, self.task.geometry.click_center, self.HIT_RADIUS
            ):
                self.state.clicked = True
                self._complete("click")
            # The decoy and empty canvas are deliberately clean no-ops.
        self._mouse_down_at = None
        self._mouse_moved_while_down = False
        self._drag_started = False

    def scroll(self, clicks: int) -> None:
        super().scroll(clicks)
        self.state.scroll_total += int(clicks)
        expected = self.task.scroll_clicks
        if (
            self._expected_step() == "scroll"
            and self.state.scroll_total * expected > 0
            and abs(self.state.scroll_total) >= abs(expected)
        ):
            self._complete("scroll")

    def coalesced_type(self, text: str) -> None:
        super().coalesced_type(text)
        if self.state.focused:
            self.state.text = text
            if text == self.task.target_text:
                self._complete("coalesced_type")

    def hidden_snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "task_sha256": self.task.task_sha256,
            "cursor": list(self.cursor_position()),
            "held_buttons": sorted(self.audit.held_buttons),
            "held_keys": sorted(self.audit.held_keys),
            **copy.deepcopy(self.state.as_dict()),
        }

    def solved(self) -> bool:
        if tuple(self.state.completed_steps) != self.task.steps:
            return False
        if self.audit.held_buttons or self.audit.held_keys:
            return False
        if (
            "coalesced_type" in self.task.steps
            and self.state.text != self.task.target_text
        ):
            return False
        if "scroll" in self.task.steps:
            expected = self.task.scroll_clicks
            if self.state.scroll_total * expected <= 0:
                return False
        if "click" in self.task.steps and not self.state.clicked:
            return False
        if "drag" in self.task.steps and not self.state.drag_complete:
            return False
        return True


class Episode:
    """Policy-facing horizon/reward/done API.

    The policy receives only :class:`Observation` and :class:`StepResult`.
    Hidden task parameters and oracle snapshots remain held by the trainer-side
    backend and are never included in these values.
    """

    def __init__(self, task: TaskDefinition, arm: Arm) -> None:
        if arm not in {"native_absolute_control", "compact_raw_phaseb"}:
            raise EpisodeError(f"unknown action arm: {arm}")
        self.task = task
        self.arm = arm
        self._generation = 0
        self._step_index = 0
        self._done = True
        self._transport: StateTrackingTransport | None = None
        self._executor: NativeAbsoluteExecutor | CompactRawExecutor | None = None
        self._last_receipt: ResetReceipt | None = None

    def _observation(self, *, include_instruction: bool) -> Observation:
        if self._transport is None:
            raise EpisodeError("episode is not reset")
        visible_frame = {
            "task_id": self.task.task_id,
            "cursor": list(self._transport.cursor_position()),
            "step_index": self._step_index,
            # This opaque digest stands in for VM pixels in CPU contract tests.
            "visual_generation": self._generation,
        }
        digest = payload_sha256(visible_frame)
        return Observation(
            task_id=self.task.task_id,
            instruction=self.task.instruction if include_instruction else None,
            frame_uri=(
                f"frame://{self.task.task_id}/{self._generation}/{self._step_index}"
            ),
            frame_sha256=digest,
            step_index=self._step_index,
            horizon=self.task.horizon,
        )

    def reset(self) -> ResetReceipt:
        self._generation += 1
        self._step_index = 0
        self._done = False
        self._transport = StateTrackingTransport(self.task)
        self._executor = (
            NativeAbsoluteExecutor(self._transport)
            if self.arm == "native_absolute_control"
            else CompactRawExecutor(self._transport)
        )
        reset_payload = self._transport.hidden_snapshot()
        reset_fingerprint = payload_sha256(reset_payload)
        receipt = ResetReceipt(
            task_id=self.task.task_id,
            task_sha256=self.task.task_sha256,
            reset_fingerprint=reset_fingerprint,
            generation=self._generation,
            observation=self._observation(include_instruction=True),
        )
        self._last_receipt = receipt
        return receipt

    def step(self, action: dict[str, Any] | str) -> StepResult:
        if self._done or self._executor is None or self._transport is None:
            raise EpisodeError("step called before reset or after done")
        if self._step_index >= self.task.horizon:
            raise EpisodeError("step called beyond frozen horizon")
        try:
            result: DispatchResult = self._executor.execute(
                action
            )
        except (TypeError, ValueError) as exc:
            # Parse errors are fail-loud infrastructure results, never hidden or
            # rewritten as a task/oracle failure.
            raise EpisodeError(f"action parse/dispatch failed: {exc}") from exc
        if result.executor_dispatch_status != "ok":
            raise EpisodeError("executor dispatch returned non-ok status")
        self._step_index += 1
        solved = self._transport.solved()
        truncated = not solved and self._step_index == self.task.horizon
        self._done = solved or truncated
        return StepResult(
            observation=self._observation(include_instruction=False),
            reward=1.0 if solved else 0.0,
            done=self._done,
            truncated=truncated,
            step_index=self._step_index,
            horizon=self.task.horizon,
            parse_status=result.parse_status,
            executor_dispatch_status=result.executor_dispatch_status,
            action_class=result.action_class,
        )

    @property
    def last_reset_receipt(self) -> ResetReceipt:
        if self._last_receipt is None:
            raise EpisodeError("episode has not been reset")
        return self._last_receipt

    def _trainer_hidden_snapshot(self) -> dict[str, Any]:
        if self._transport is None:
            raise EpisodeError("episode is not reset")
        return self._transport.hidden_snapshot()

    def _trainer_solved(self) -> bool:
        return bool(self._transport is not None and self._transport.solved())
