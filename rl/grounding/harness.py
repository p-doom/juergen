"""grounding harness: `DesktopHarness` over a cached-screenshot canvas."""

from __future__ import annotations

from pathlib import Path
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

__all__ = ["GroundingHarness", "GroundingHarnessConfig", "GroundingCanvasPreparer"]


class GroundingCanvasPreparer:
    """Loads the labelled screenshot and places the stratified cursor start.

    `postcondition_success` is deliberately left unset: this env does not stop on a
    hit. The screenshot is a final state and the eval is single-step, so there is
    nothing to stop early *from* — and in the multi-step VM-backed variant the fixed
    K frames are what makes trajectory length comparable across regimes.
    """

    kind = "grounding_canvas"

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        if not isinstance(session, VirtualDesktop):
            raise TypeError("grounding canvas requires a virtual desktop session")
        from PIL import Image

        screen = tuple(task.setup.get("screen") or (1920, 1080))
        with Image.open(Path(str(task.setup["image_path"]))) as handle:
            canvas = handle.convert("RGB")
        session.configure(
            canvas=canvas,
            cursor=tuple(task.cursor_start or (0, 0)),
            screen=(int(screen[0]), int(screen[1])),
        )
        return {
            "regime": task.regime,
            "cursor_start": list(task.cursor_start or (0, 0)),
            "bbox": list(task.bbox or ()),
        }

    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        cursor = tuple(session.cursor_position())
        box = tuple(task.bbox or (0, 0, 1, 1))
        return {
            "cursor": list(cursor),
            "in_bbox": in_bbox(cursor, box),
            "distance": distance_to_box(cursor, box),
            "postcondition_status": "ok",
        }


register_preparer(GroundingCanvasPreparer())


class GroundingHarnessConfig(DesktopHarnessConfig):
    id: str = "rl_grounding"
    codec: str = "move_rel"
    history: HistoryConfig = HistoryConfig(name="stateless_single_turn")
    images: ImageBudgetConfig = ImageBudgetConfig(max_images=1, media="png")
    settle: SettleConfig = SettleConfig(min_delay_s=0.0, per_kind={})
    pool: DesktopPoolConfig = DesktopPoolConfig(
        key="grounding_canvas", max_node_slots=64, hide_gpu_during_boot=False
    )
    max_steps: int = 1
    require_unsolved_start: bool = False
    """The cursor sampler already guarantees an outside-bbox start except against a
    screen edge; refusing those cells would silently drop the hardest targets."""


class GroundingHarness(DesktopHarness):
    def pool_factory(self) -> Any:
        return canvas_pool(VirtualDesktop)
