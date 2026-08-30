from __future__ import annotations

import io
import math
import random
from collections.abc import Iterable
from typing import Any

import verifiers.v1 as vf
from PIL import Image, ImageDraw
from pydantic import Field

from evals.harness import DesktopHarness, DesktopHarnessConfig
from evals.tasks import (
    RESULT_KEY,
    DesktopTask,
    DesktopTaskData,
    distance_to_box,
    in_bbox,
    register_preparer,
    valid_result,
)

TARGET_BOX_KIND = "target_box"
TARGET_BOX_INSTRUCTION = (
    "Move the cursor into the green box, then terminate with success."
)


def _box(
    *, index: int, seed: int, screen: tuple[int, int], size: tuple[int, int], margin: int
) -> tuple[int, int, int, int]:
    width, height = screen
    box_width, box_height = size
    if box_width <= 0 or box_height <= 0 or margin < 0:
        raise ValueError("target-box size must be positive and margin non-negative")
    if box_width + 2 * margin > width or box_height + 2 * margin > height:
        raise ValueError("target box plus margins exceeds the desktop")
    rng = random.Random(f"{seed}:{index}:box")
    left = rng.randint(margin, width - margin - box_width)
    top = rng.randint(margin, height - margin - box_height)
    return left, top, left + box_width, top + box_height


def _cursor(
    *, index: int, seed: int, screen: tuple[int, int], box: tuple[int, int, int, int]
) -> tuple[int, int]:
    rng = random.Random(f"{seed}:{index}:cursor")
    for _ in range(100):
        candidate = rng.randrange(screen[0]), rng.randrange(screen[1])
        if not in_bbox(candidate, box):
            return candidate
    raise ValueError("target box leaves no sampled cursor start outside it")


def _annotate(frame: bytes, box: tuple[int, int, int, int]) -> bytes:
    with Image.open(io.BytesIO(frame)) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=(0, 255, 0), width=4)
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=92, subsampling=2, optimize=False)
    return encoded.getvalue()


class TargetBoxTask(DesktopTask):
    @vf.reward
    async def success(self, trace: vf.Trace) -> float:
        result = valid_result(trace, TARGET_BOX_KIND)
        return float(result["task_reward"])

    @vf.reward
    async def closest_approach(self, trace: vf.Trace) -> float:
        result = valid_result(trace, TARGET_BOX_KIND)
        if result["task_reward"] >= 1.0:
            return 0.0
        best = float(result["best_distance"])
        return -0.15 if best < 0 else 0.3 * math.exp(-best / 400.0)

    @vf.metric
    async def target_box_stats(self, trace: vf.Trace) -> dict[str, float]:
        result = trace.info.get(RESULT_KEY) or {}
        return {
            "best_distance_px": float(result.get("best_distance", -1.0)),
            "entered_box": float(int(result.get("reach_frame", -1)) >= 0),
            "declared_success": float(result.get("control_terminate") == "terminate"),
        }


class TargetBoxTasksetConfig(vf.TasksetConfig):
    n_tasks: int = Field(default=1024, ge=1)
    max_steps: int = Field(default=10, ge=1)
    seed: int = 0
    screen_width: int = Field(default=1920, ge=1)
    screen_height: int = Field(default=1080, ge=1)
    box_width: int = Field(default=150, ge=1)
    box_height: int = Field(default=150, ge=1)
    margin: int = Field(default=40, ge=0)


class TargetBoxTaskset(vf.Taskset[TargetBoxTask, TargetBoxTasksetConfig]):
    def load(self) -> Iterable[TargetBoxTask]:
        screen = self.config.screen_width, self.config.screen_height
        size = self.config.box_width, self.config.box_height
        for index in range(self.config.n_tasks):
            box = _box(
                index=index,
                seed=self.config.seed,
                screen=screen,
                size=size,
                margin=self.config.margin,
            )
            cursor = _cursor(
                index=index,
                seed=self.config.seed,
                screen=screen,
                box=box,
            )
            yield TargetBoxTask(
                DesktopTaskData(
                    idx=index,
                    name=f"target-box-{index:05d}",
                    prompt=TARGET_BOX_INSTRUCTION,
                    instruction=TARGET_BOX_INSTRUCTION,
                    kind=TARGET_BOX_KIND,
                    max_steps=self.config.max_steps,
                    bbox=box,
                    cursor_start=cursor,
                    setup={"screen": list(screen)},
                ),
                self.config.task,
            )


class TargetBoxPreparer:
    kind = TARGET_BOX_KIND

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        if task.bbox is None or task.cursor_start is None:
            raise ValueError("target-box task has no scene")
        expected = tuple(task.setup["screen"])
        observed = tuple(session.screen_size())
        if observed != expected:
            raise ValueError(
                f"target-box screen {observed!r} does not match {expected!r}"
            )
        from desktop.ir import Operation

        session.execute_atomic((Operation("move_to", task.cursor_start),))
        return {"box": list(task.bbox), "cursor_start": list(task.cursor_start)}

    def observe(self, frame: bytes, task: DesktopTaskData) -> bytes:
        if task.bbox is None:
            raise ValueError("target-box task has no box")
        return _annotate(frame, task.bbox)

    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        if task.bbox is None:
            raise ValueError("target-box task has no box")
        cursor = tuple(session.cursor_position())
        return {
            "cursor": list(cursor),
            "box": list(task.bbox),
            "in_bbox": in_bbox(cursor, task.bbox),
            "distance": distance_to_box(cursor, task.bbox),
            "postcondition_status": "ok",
            "postcondition_success": False,
        }

    def evaluate(
        self, session: Any, task: DesktopTaskData, *, declared: str | None
    ) -> float:
        return float(
            declared == "terminate" and self.probe(session, task)["in_bbox"]
        )


register_preparer(TargetBoxPreparer())


class TargetBoxHarness(DesktopHarness):
    def __init__(self, config: DesktopHarnessConfig) -> None:
        if not config.evaluate_on_finish:
            raise ValueError("target-box requires evaluate_on_finish=true")
        super().__init__(config)
