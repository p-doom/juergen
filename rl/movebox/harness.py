"""movebox harness: `DesktopHarness` plus a canvas pool and a preparer.

The whole env-specific surface is ~40 lines because the episode loop is shared.
Pre-refactor this was `harness.py` (78) delegating to `rollout.py` (171) which
re-implemented parse -> apply -> render -> terminate with its own no-op/repeat
accounting; that accounting is now `MouseIndicators` plus one metric.
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


class MoveBoxPreparer:
    """Installs the scene on the virtual desktop and reports box containment.

    `postcondition_success` in the probe is what stops the episode the moment the
    cursor lands in the box — the same early exit the standalone rollout had, now
    expressed through the shared driver's one stopping rule.
    """

    kind = "movebox"

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        if not isinstance(session, VirtualDesktop):
            raise TypeError("movebox requires a virtual desktop session")
        screen = tuple(task.setup.get("screen") or (1920, 1080))
        scene = MoveBoxScene(
            idx=task.idx,
            background_path=str(task.setup["background_path"]),
            box=tuple(task.bbox or (0, 0, 1, 1)),
            cursor_start=tuple(task.cursor_start or (0, 0)),
            screen_w=int(screen[0]),
            screen_h=int(screen[1]),
            band=str(task.setup.get("band", "uniform")),
            start_distance=float(task.setup.get("start_distance", -1.0)),
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
        box = tuple(task.bbox or (0, 0, 1, 1))
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
    """`move_rel` owns the normalized 0-999 relative convention. The env no longer
    knows the convention exists — `compile` hands it absolute pixels."""
    history: HistoryConfig = HistoryConfig(name="stateless_single_turn")
    """Stateless by design: a fresh single-turn prompt each step makes the per-step
    grounding decision identifiable, which is the point of this env."""
    images: ImageBudgetConfig = ImageBudgetConfig(max_images=1, media="png")
    settle: SettleConfig = SettleConfig(min_delay_s=0.0, per_kind={})
    pool: DesktopPoolConfig = DesktopPoolConfig(
        key="movebox_canvas", max_node_slots=64, hide_gpu_during_boot=False
    )
    max_steps: int = 8
    require_unsolved_start: bool = True


class MoveBoxHarness(DesktopHarness):
    def pool_factory(self) -> Any:
        return canvas_pool(VirtualDesktop)
