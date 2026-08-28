"""movebox harness: `DesktopHarness` plus a canvas pool and a preparer.

The episode loop is shared; no-op/repeat accounting is `MouseIndicators` plus one
metric.
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
from evals.tasks import DesktopTaskData, register_preparer
from rl.desktop import VirtualDesktop, canvas_pool
from rl.geometry import distance_to_box, in_bbox
from rl.movebox.dataset import MoveBoxScene, load_canvas

__all__ = ["MoveBoxHarness", "MoveBoxHarnessConfig", "MoveBoxPreparer"]


def _declared_box(task: DesktopTaskData) -> tuple[int, int, int, int]:
    """The row's box, or refuse the episode.

    `bbox` is optional on `DesktopTaskData` because the OSWorld families have
    none, so this family asserts its own requirement. The placeholder this
    replaces was a 1x1 box at the origin, which no cursor enters: every rollout
    in the group scored 0.0 and nothing recorded why.
    """
    if task.bbox is None:
        raise ValueError(f"movebox task {task.name!r} declares no bbox")
    return task.bbox


class MoveBoxPreparer:
    """Installs the scene on the virtual desktop and reports box containment.

    `postcondition_success` in the probe stops the episode the moment the cursor
    lands in the box.
    """

    kind = "movebox"

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        if not isinstance(session, VirtualDesktop):
            raise TypeError("movebox requires a virtual desktop session")
        if task.cursor_start is None:
            raise ValueError(f"movebox task {task.name!r} declares no cursor_start")
        screen = task.setup["screen"]
        scene = MoveBoxScene(
            idx=task.idx,
            background_path=str(task.setup["background_path"]),
            box=_declared_box(task),
            cursor_start=tuple(task.cursor_start),
            screen_w=int(screen[0]),
            screen_h=int(screen[1]),
            band=str(task.setup["band"]),
            start_distance=float(task.setup["start_distance"]),
        )
        session.configure(
            canvas=load_canvas(scene),
            cursor=scene.cursor_start,
            screen=(scene.screen_w, scene.screen_h),
        )
        return {
            "band": scene.band,
            "start_distance": scene.start_distance,
            "cursor_start": list(scene.cursor_start),
            "box": list(scene.box),
        }

    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        cursor = tuple(session.cursor_position())
        box = _declared_box(task)
        inside = in_bbox(cursor, box)
        return {
            "cursor": list(cursor),
            "in_bbox": inside,
            "distance": distance_to_box(cursor, box),
            "postcondition_status": "ok",
            "postcondition_success": inside,
        }


register_preparer(MoveBoxPreparer())


class MoveBoxHarnessConfig(DesktopHarnessConfig):
    id: str = "rl_movebox"
    codec: str = "move_rel"
    """`move_rel` owns the normalized 0-999 relative convention. The env does not
    know the convention exists — `compile` hands it absolute pixels."""
    history: HistoryConfig = HistoryConfig(name="stateless_single_turn")
    """Stateless: a fresh single-turn prompt each step makes the per-step grounding
    decision identifiable."""
    images: ImageBudgetConfig = ImageBudgetConfig(max_images=1)
    settle: SettleConfig = SettleConfig(min_delay_s=0.0, per_kind={})
    pool: DesktopPoolConfig = DesktopPoolConfig(
        key="movebox_canvas", max_node_slots=64, hide_gpu_during_boot=False
    )
    max_steps: int = 8
    require_unsolved_start: bool = True


class MoveBoxHarness(DesktopHarness):
    def pool_factory(self) -> Any:
        return canvas_pool(VirtualDesktop)
