"""grounding: single-step relative-move onto a labelled UI target.

Reward = sparse reach + a bounded shaping term, with an explicit no-move penalty.
The total ordering is deliberate and is the reason a *negative* term exists at all:

    no-move  -0.15   <   miss  (0, 0.3)   <   hit  1.0

A model that emits `wait`, `terminate`, or a coordinate-less click never moves and
so can never miss; without the penalty that is a strictly safer policy than trying,
and GRPO finds it. The `-0.15` makes not moving worse than any miss.

Shaping parameters are module constants, not config fields: inside a `vf.Task`,
`self.config` is the per-task `TaskConfig` with no custom fields, so reading one
raises `AttributeError`, and a single throwing reward inside `Task.score`'s
`asyncio.gather` drops the whole group's rewards.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import verifiers.v1 as vf

from evals.indicators import MouseIndicators, SamplingProvenance
from evals.tasks import RESULT_KEY, DesktopTask, DesktopTaskData
from rl.grounding.dataset import REGIMES, cursor_start, load_targets

__all__ = ["GroundingTask", "GroundingTaskset", "GroundingTasksetConfig"]

SHAPING_WEIGHT = 0.3
SHAPING_SCALE = 400.0
NO_MOVE_PENALTY = 0.15


class GroundingTask(MouseIndicators, SamplingProvenance, DesktopTask):
    @vf.reward
    async def reach(self, trace: vf.Trace) -> float:
        result = trace.info.get(RESULT_KEY)
        if not isinstance(result, dict):
            raise RuntimeError("grounding rollout published no result")
        return 1.0 if int(result.get("reach_frame", -1)) >= 0 else 0.0

    @vf.reward
    async def shaped_progress(self, trace: vf.Trace) -> float:
        result = trace.info.get(RESULT_KEY) or {}
        if int(result.get("reach_frame", -1)) >= 0:
            return 0.0
        if _never_moved(result):
            return -NO_MOVE_PENALTY
        distance = float(result.get("best_distance", -1.0))
        if distance < 0:
            return -NO_MOVE_PENALTY
        return SHAPING_WEIGHT * math.exp(-distance / max(SHAPING_SCALE, 1.0))

    @vf.metric
    async def grounding(self, trace: vf.Trace) -> dict[str, float]:
        result = trace.info.get(RESULT_KEY) or {}
        return {
            "reached": 1.0 if int(result.get("reach_frame", -1)) >= 0 else 0.0,
            "distance_px": float(result.get("best_distance", -1.0)),
            "never_moved": 1.0 if _never_moved(result) else 0.0,
        }


def _never_moved(result: dict) -> bool:
    """No dispatched movement at all — the no-op attractor.

    Distinct from an unparseable reply: a well-formed `wait`/`terminate`/
    coordinate-less click parses fine and still moves nothing, and conflating the
    two hides which failure the policy actually has.
    """
    steps = result.get("steps_detail") or []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("cursor_before") != step.get("cursor_after"):
            return False
    return True


class GroundingTasksetConfig(vf.TasksetConfig):
    bboxes_jsonl: str = ""
    regimes: list[str] = list(REGIMES)
    max_targets: int = 0
    target_idxs: list[int] = []
    max_steps: int = 1
    """Single-step by default: the screenshot is a FINAL state, so `wait` and
    `terminate` are meaningless and a second turn would score a stale scene."""


class GroundingTaskset(vf.Taskset[GroundingTask, GroundingTasksetConfig]):
    def load(self) -> Iterable[GroundingTask]:
        targets = load_targets(Path(self.config.bboxes_jsonl))
        keep = set(self.config.target_idxs)
        if keep:
            targets = [t for t in targets if t.idx in keep]
        if self.config.max_targets:
            targets = targets[: self.config.max_targets]
        idx = 0
        for target in targets:
            for regime in self.config.regimes:
                start = cursor_start(target, target.screen[0], target.screen[1], regime)
                yield GroundingTask(
                    DesktopTaskData(
                        idx=idx,
                        name=f"{target.app}/{target.task_id}/{regime}",
                        prompt=target.instruction,
                        instruction=target.instruction,
                        kind="grounding_canvas",
                        max_steps=self.config.max_steps,
                        app=target.app,
                        bbox=target.bbox,
                        regime=regime,
                        cursor_start=start,
                        expected={"bbox": list(target.bbox)},
                        setup={
                            "image_path": str(target.image_path),
                            "screen": list(target.screen),
                        },
                    ),
                    self.config.task,
                )
                idx += 1
