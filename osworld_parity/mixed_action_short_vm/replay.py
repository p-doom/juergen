from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from .hidden_oracle import HiddenOracleResult, evaluate_in_fresh_process
from .manifest import TaskDefinition
from .runtime import Arm, Episode
from .teacher import convert_native_actions, native_gold_actions


@dataclass(frozen=True)
class ReplayReport:
    task_id: str
    arm: Arm
    kind: str
    reset_fingerprint: str
    action_count: int
    horizon: int
    reward: float
    done: bool
    truncated: bool
    final_pointer_mask: int
    oracle: HiddenOracleResult

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["oracle"] = asdict(self.oracle)
        return value


def replay(
    task: TaskDefinition, *, arm: Arm, near_miss: bool = False
) -> ReplayReport:
    episode = Episode(task, arm)
    receipt = episode.reset()
    native = native_gold_actions(task, near_miss=near_miss)
    actions: list[dict[str, Any] | str]
    if arm == "native_absolute_control":
        actions = list(native)
        padding: dict[str, Any] | str = {"action": "wait", "time": 0}
    else:
        actions = list(convert_native_actions(native, task.geometry.initial_cursor))
        padding = "0 0 0"
    final = None
    for action in actions:
        final = episode.step(action)
    while final is not None and not final.done:
        final = episode.step(padding)
        actions.append(padding)
    if final is None:
        raise RuntimeError("replay produced no action")
    state = episode._trainer_hidden_snapshot()
    oracle = evaluate_in_fresh_process(task, state)
    pointer_mask = 0
    held = state.get("held_buttons", [])
    if "left" in held:
        pointer_mask |= 1 << 8
    if "middle" in held:
        pointer_mask |= 1 << 9
    if "right" in held:
        pointer_mask |= 1 << 10
    report = ReplayReport(
        task_id=task.task_id,
        arm=arm,
        kind="near_miss" if near_miss else "gold",
        reset_fingerprint=receipt.reset_fingerprint,
        action_count=len(actions),
        horizon=task.horizon,
        reward=final.reward,
        done=final.done,
        truncated=final.truncated,
        final_pointer_mask=pointer_mask,
        oracle=oracle,
    )
    if oracle.oracle_pid == os.getpid():
        raise RuntimeError("replay oracle was not trainer-only fresh process")
    if near_miss:
        if oracle.oracle_status != "ok" or oracle.MOUSE_SOLVED or final.reward != 0:
            raise RuntimeError(
                f"near-miss replay was not a clean negative: {task.task_id}"
            )
    elif oracle.oracle_status != "ok" or not oracle.MOUSE_SOLVED or final.reward != 1:
        raise RuntimeError(f"gold replay failed: {task.task_id}")
    return report
