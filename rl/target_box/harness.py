"""target_box harness: `DesktopHarness` over the real VM pool.

The only VM-backed RL env. It differs from the eval families in exactly two ways,
both inside the preparer: the box is derived from the task key rather than labelled,
and every observation is annotated with that box.

Dropped on the way in: the "exactly one tool call per step" rule that ended a
rollout with `multiple_actions_parsed`, and the few-shot pair whose demonstrated
deltas were computed in **raw pixels** while the executor read them as normalized
0-999 — ~1.9x too large in x at 1920x1080, with a system prompt that never mentioned
normalization. Both were compensations for the convention living outside the
grammar. It now lives in the codec, so the prompt is `codec.describe()` and a turn
may carry as many operations as the grammar allows.
"""

from __future__ import annotations

from typing import Any

from evals.harness import (
    DesktopHarness,
    DesktopHarnessConfig,
    DesktopPoolConfig,
    HistoryConfig,
    ImageBudgetConfig,
    SettleConfig,
)
from evals.tasks import DesktopTaskData, osworld_task_config, register_preparer
from rl.geometry import distance_to_box, in_bbox
from rl.target_box.geometry import (
    TargetBoxConfig,
    annotate,
    sample_box,
    sample_cursor_start,
)

__all__ = ["TargetBoxHarness", "TargetBoxHarnessConfig", "TargetBoxPreparer"]


def _scene(task: DesktopTaskData) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int]]:
    setup = task.setup
    screen = tuple(int(v) for v in (setup.get("screen") or (1920, 1080)))
    config = TargetBoxConfig(**dict(setup.get("box") or {}))
    key = str(setup.get("instance_key") or task.name or "")
    box = sample_box(config, screen_width=screen[0], screen_height=screen[1], instance_key=key)
    cursor = sample_cursor_start(
        config, box, screen_width=screen[0], screen_height=screen[1], instance_key=key
    )
    return box, cursor, screen


class TargetBoxPreparer:
    """Reset the VM to a real desktop, derive the box, place the real cursor."""

    kind = "target_box"

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        if task.setup.get("config"):
            # One session contract, not two: `setup()` takes the whole OSWorld
            # task JSON everywhere (`evals/vm.py`), because that is what lets the
            # `evaluator` block reach the session and `evaluate()` stay
            # argument-free. target_box has no evaluator and does not want one —
            # its reward is the declared box — so this is a shape change only.
            session.setup(osworld_task_config(task))
        box, cursor, screen = _scene(task)
        observed = tuple(session.screen_size())
        if observed != screen:
            raise ValueError(
                f"target_box screen {observed!r} does not match the configured {screen!r}"
            )
        session.execute_pyautogui(f"pyautogui.moveTo({cursor[0]}, {cursor[1]})")
        return {"box": list(box), "cursor_start": list(cursor), "screen": list(screen)}

    def observe(self, frame: bytes, task: DesktopTaskData) -> bytes:
        """Draw the target box on every observation.

        The box must be on *every* frame, not just the first: the model has to see
        where it is going after each move, and an un-annotated later frame silently
        turns a grounded task into a memory task.
        """
        box, _, _ = _scene(task)
        return annotate(frame, box)

    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        box, _, _ = _scene(task)
        cursor = tuple(session.cursor_position())
        inside = in_bbox(cursor, box)
        return {
            "cursor": list(cursor),
            "box": list(box),
            "in_bbox": inside,
            "distance": distance_to_box(cursor, box),
            "postcondition_status": "ok",
            # Success needs the model to *declare* it, so entering the box does not
            # end the episode; the reward requires the terminate as well.
            "postcondition_success": False,
        }


register_preparer(TargetBoxPreparer())


class TargetBoxHarnessConfig(DesktopHarnessConfig):
    id: str = "rl_target_box"
    codec: str = "move_rel"
    history: HistoryConfig = HistoryConfig(name="latest_image_only", n_history_frames=10)
    """Accumulate turns (the model's own reasoning is load-bearing here) but send
    only the newest image — the only shape that kept a 10-step VM rollout inside the
    renderer's image cache."""
    images: ImageBudgetConfig = ImageBudgetConfig(max_images=1, media="png")
    settle: SettleConfig = SettleConfig(min_delay_s=1.0, per_kind={})
    pool: DesktopPoolConfig = DesktopPoolConfig(key="target_box_vm", max_node_slots=14)
    max_steps: int = 10
    require_unsolved_start: bool = True
    evaluate_on_finish: bool = False
    """The OSWorld task's own scorer is irrelevant: the goal is the synthetic box,
    not the task the background came from."""


class TargetBoxHarness(DesktopHarness):
    """Uses the inherited `pool_factory` — real VMs from `pixeldesk.vm.pool`."""
